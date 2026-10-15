"""Music structure analysis: tempo, beats, downbeats, onsets, drops, energy, sections.

Pure numpy/scipy (no librosa). Only ``ctx.audio``, ``ctx.sr`` and the timeline
duration are used, so the same function analyses standalone music files (a
context with zero or one video frame is timed by the audio length instead).

Algorithm
---------
1. **Spectral front end.** Centred Hann STFT (n_fft 2048 / hop 512 at 22.05 kHz,
   scaled with ``ctx.sr``) -> 64-band mel power -> dB with an 80 dB floor. The
   onset strength envelope is the spectral flux (positive frame-to-frame dB
   differences) averaged over bands, with the bands below ``low_hz`` carrying
   ``low_share`` of the weight: kicks / 808s define the pulse in the target
   genres, and in a plain band average broadband hi-hats dominate and cause
   double-tempo errors. A 1 s local mean is subtracted, the result half-wave
   rectified and scaled to unit variance. A short-window flux (n_fft 512 /
   hop 128) gives ~6 ms timing for snapping beats and onsets.
2. **Tempo.** Autocorrelation of the envelope over the lags of
   ``bpm_min..bpm_max``, weighted by a log-normal prior (``bpm_center``,
   ``bpm_sigma`` octaves); the period is refined by parabolic interpolation at
   the peak and at its 2x-4x multiples. The normalised ACF value at the period
   ("pulse clarity") becomes ``Music.confidence``; below ``min_clarity`` no beat
   grid is reported.
3. **Beats.** Ellis (2007) dynamic programming,
   ``score(t) = onset(t) + max_tau [score(tau) - tightness * log((t - tau) / P)^2]``,
   solved in blocks of P/2 frames (a frame never depends on its own block),
   backtracked from the last strong maximum; weak leading/trailing beats are
   trimmed. **Metrical level:** the grid is compared with its double- and
   half-tempo alternatives by prior-weighted beat salience (mean onset strength
   at the grid points) and re-tracked at the winning period, which resolves the
   usual octave errors (hi-hat eighths vs. kicks, accented half notes). Beats are
   snapped to the strongest fine-onset peak within ``snap_window`` and the BPM is
   re-estimated by a robust line fit of beat time vs. beat index.
4. **Downbeats** (``meter`` 4): the beat phase whose beats carry the strongest
   low-frequency accent (bass level + bass onset, z-scored) wins; a detected drop
   adds a small vote for its own phase (drops land on bar starts).
5. **Onsets:** peaks of the envelope above an adaptive (local mean) and an
   absolute threshold, at least ``onset_min_gap`` apart, timed on the fine flux.
6. **Energy:** frame RMS smoothed over ``energy_smooth`` s, divided by its 95th
   percentile and clipped to 0..1; resampled to a ``energy_dt`` Curve.
7. **Drops:** mean energy over the next ``drop_window`` s vs. the previous one;
   the geometric mean of the full-band and bass-band ratios must exceed
   ``drop_ratio`` (both rising, loud afterwards, not a start from silence); the
   strongest maxima >= ``drop_min_gap`` apart are snapped to the nearest beat.
   ``Drop.strength`` is the full-band energy ratio.
8. **Sections:** silence runs, drops and strong level changes split the track;
   segments are labelled from their energy and their position relative to the
   drops (build/intro before the first drop, drop while energy stays high,
   break for later lulls, outro for a trailing decay, body/intro/outro/break
   when the track has no drop).

Every threshold is tunable through ``ctx.config.p("audio", <name>, default)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from scipy import fft as sfft
from scipy import ndimage, signal

from editast.analyzers.types import AudioAnalysis
from editast.ast.nodes import AudioLabel, AudioSection, Curve, Drop, Music
from editast.context import AnalysisContext

NAME = "audio"
_REF_SR = 22050.0
_EPS = 1e-10
_BASS_BAND = (30.0, 150.0)  # Hz, kick / 808 / sub-bass fundamentals

ParamFn = Callable[[str, Any], Any]


# --------------------------------------------------------------------------- front end


@dataclass
class _Features:
    """Frame-level features; coarse frame ``i`` is centred at ``i * hop_s`` seconds."""

    hop_s: float
    env: np.ndarray  # onset strength: detrended, rectified, unit variance
    rms: np.ndarray  # frame RMS (linear amplitude)
    bass: np.ndarray  # bass-band amplitude (linear, arbitrary scale)
    bass_db: np.ndarray  # bass-band power in dB (80 dB floor)
    mel_db: np.ndarray  # (n, n_mels) log-mel spectrogram (80 dB floor)
    fine_env: np.ndarray  # short-window flux, frame ``j`` at ``j * fine_hop_s``
    fine_hop_s: float


def _pow2(x: float) -> int:
    return int(2 ** max(6, round(math.log2(max(x, 1.0)))))


def _odd(n: float) -> int:
    k = max(1, int(round(n)))
    return k if k % 2 else k + 1


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(
    sr: float, n_fft: int, n_mels: int, fmin: float = 30.0, fmax: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Triangular (HTK mel) filters ``(n_mels, n_fft // 2 + 1)`` and their centre frequencies."""
    fmax = min(fmax or sr / 2.0, sr / 2.0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    fb = np.maximum(0.0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    # Low bands can be narrower than one FFT bin: give each its nearest bin.
    empty = fb.sum(axis=1) <= 0.0
    if empty.any():
        nearest = np.abs(freqs[None, :] - mid[empty]).argmin(axis=1)
        fb[np.flatnonzero(empty), nearest] = 1.0
    return fb.astype(np.float32), edges[1:-1]


def _spectral_frames(
    y: np.ndarray, sr: float, n_fft: int, hop: int, fb: np.ndarray, chunk: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Mel power ``(n, n_mels)`` and bass-band power ``(n,)`` of centred Hann frames."""
    n = 1 + y.size // hop
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, n_fft // 2), n_fft)[::hop][:n]
    window = signal.get_window("hann", n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bass_bins = (freqs >= _BASS_BAND[0]) & (freqs <= _BASS_BAND[1])
    if not bass_bins.any():
        bass_bins[1] = True
    fb_t = np.ascontiguousarray(fb.T)
    mel = np.empty((n, fb.shape[0]), np.float32)
    bass = np.empty(n, np.float32)
    for i in range(0, n, chunk):  # chunked to bound memory on long tracks
        spec = sfft.rfft(frames[i:i + chunk] * window, axis=1)
        power = spec.real ** 2 + spec.imag ** 2
        mel[i:i + chunk] = power @ fb_t
        bass[i:i + chunk] = power[:, bass_bins].sum(axis=1)
    return mel, bass


def _db(power: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    db = 10.0 * np.log10(np.maximum(power, _EPS))
    return np.maximum(db, db.max() - top_db)


def _band_weights(centers: np.ndarray, low_hz: float, low_share: float) -> np.ndarray:
    """Band weights summing to 1 with ``low_share`` of the total on bands below ``low_hz``."""
    low = centers <= low_hz
    w = np.full(centers.size, 1.0 / centers.size)
    if 0 < low.sum() < centers.size and 0.0 < low_share < 1.0:
        w[low] = low_share / low.sum()
        w[~low] = (1.0 - low_share) / (~low).sum()
    return w.astype(np.float32)


def _flux(db: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted positive dB difference; the frame before the first is the (silent) floor."""
    prev = np.vstack([np.full((1, db.shape[1]), db.min()), db[:-1]])
    return (np.maximum(db - prev, 0.0) @ weights).astype(np.float32)


def _frame_rms(y: np.ndarray, n_frames: int, frame: int, hop: int) -> np.ndarray:
    yp = np.pad(y.astype(np.float64), frame // 2)
    cs = np.concatenate([[0.0], np.cumsum(yp * yp)])
    starts = np.arange(n_frames) * hop
    return np.sqrt(np.maximum(cs[starts + frame] - cs[starts], 0.0) / frame).astype(np.float32)


def _features(y: np.ndarray, sr: float, p: ParamFn) -> _Features:
    scale = sr / _REF_SR
    n_fft, hop = _pow2(2048 * scale), max(1, int(round(512 * scale)))
    hop_s = hop / sr
    fmax = min(11025.0, sr / 2.0)
    low_hz, low_share = float(p("low_hz", 180.0)), float(p("low_share", 0.5))

    fb, centers = mel_filterbank(sr, n_fft, 64, 30.0, fmax)
    mel, bass_pow = _spectral_frames(y, sr, n_fft, hop, fb)
    mel_db = _db(mel)
    flux = _flux(mel_db, _band_weights(centers, low_hz, low_share))
    trend = ndimage.uniform_filter1d(flux, _odd(float(p("detrend_window", 1.0)) / hop_s), mode="nearest")
    env = np.maximum(flux - trend, 0.0)
    env = env / env.std() if env.std() > 0 else env

    fine_fft, fine_hop = _pow2(512 * scale), max(1, hop // 4)
    fb_f, centers_f = mel_filterbank(sr, fine_fft, 40, 30.0, fmax)
    mel_f, _ = _spectral_frames(y, sr, fine_fft, fine_hop, fb_f)
    fine = _flux(_db(mel_f), _band_weights(centers_f, low_hz, low_share))
    fine = np.convolve(fine, [0.25, 0.5, 0.25], mode="same").astype(np.float32)

    return _Features(
        hop_s=hop_s, env=env.astype(np.float32), rms=_frame_rms(y, mel.shape[0], n_fft, hop),
        bass=np.sqrt(bass_pow).astype(np.float32), bass_db=_db(bass_pow).astype(np.float32),
        mel_db=mel_db.astype(np.float32), fine_env=fine, fine_hop_s=fine_hop / sr,
    )


# --------------------------------------------------------------------------- tempo


def _prior(bpm: np.ndarray | float, p: ParamFn, sigma: Optional[float] = None) -> np.ndarray:
    """Log-normal tempo preference (1 at ``bpm_center``; ``sigma`` in octaves)."""
    center = float(p("bpm_center", 120.0))
    sigma = float(p("bpm_sigma", 0.8)) if sigma is None else sigma
    return np.exp(-0.5 * (np.log2(np.asarray(bpm, dtype=float) / center) / sigma) ** 2)


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased autocorrelation of ``x - mean`` for lags ``0..max_lag``, normalised to r[0] = 1."""
    x = x.astype(np.float64) - x.mean()
    n = x.size
    size = sfft.next_fast_len(2 * n)
    spec = sfft.rfft(x, size)
    r = sfft.irfft(spec.real ** 2 + spec.imag ** 2, size)[: max_lag + 1]
    r /= n - np.arange(max_lag + 1)
    return r / r[0] if r[0] > 0 else np.zeros_like(r)


def _parabolic(r: np.ndarray, i: int) -> float:
    """Sub-sample offset of the peak at ``r[i]`` from a parabola through its neighbours."""
    a, b, c = r[i - 1], r[i], r[i + 1]
    den = a - 2.0 * b + c
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0


def _comb(r: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Mean autocorrelation at 1x, 2x and 4x each lag (beat, half bar, bar).

    The k-th multiple takes the maximum within +-k/2 frames (integer lags
    quantise the true period); multiples beyond the ACF are skipped.
    """
    total = np.zeros(lags.size)
    count = np.zeros(lags.size)
    for k in (1, 2, 4):
        half = k // 2
        idx = k * lags[:, None] + np.arange(-half, half + 1)
        ok = idx[:, -1] < r.size
        total[ok] += r[idx[ok]].max(axis=1)
        count[ok] += 1
    return np.where(count >= 2, total / np.maximum(count, 1), -np.inf)


def _estimate_period(env: np.ndarray, hop_s: float, p: ParamFn) -> tuple[float, float]:
    """Beat period in coarse frames (``nan`` if none) and the pulse clarity 0..1.

    Each lag is scored by ``prior(bpm) * comb(lag)``: a period whose half-bar and
    bar multiples also repeat beats one that only matches a sub-pattern (a dotted
    figure at 2/3 or 3/2 of the tempo). The clarity is the comb value itself.
    """
    lag_lo = 60.0 / (float(p("bpm_max", 200.0)) * hop_s)
    lag_hi = 60.0 / (float(p("bpm_min", 60.0)) * hop_s)
    max_lag = min(int(0.75 * env.size), int(4 * lag_hi) + 4)
    lags = np.arange(max(2, math.ceil(lag_lo)), min(math.floor(lag_hi), (max_lag - 1) // 2) + 1)
    if lags.size == 0:
        return float("nan"), 0.0
    r = _autocorr(env, max_lag)
    comb = _comb(r, lags)
    # A broad prior: this stage only has to find the metrical family; the
    # octave within it is settled on the tracked beats (see _beat_grid).
    score = _prior(60.0 / (lags * hop_s), p, float(p("family_sigma", 1.5))) * np.maximum(comb, 0.0)
    if not np.any(score > 0):
        return float("nan"), 0.0
    i = int(np.argmax(score))
    best = int(lags[i])
    # Sub-frame period: parabolic peaks at the lag and at its 2x..4x multiples
    # (the k-th multiple pins the period down k times more precisely).
    estimates: list[float] = []
    weights: list[float] = []
    for k in (1, 2, 3, 4):
        c = k * best
        if c + k + 1 >= r.size:
            break
        j = c - k + int(np.argmax(r[c - k:c + k + 1]))
        if 0 < j < r.size - 1 and r[j] > 0 and r[j] >= max(r[j - 1], r[j + 1]):
            estimates.append((j + _parabolic(r, j)) / k)
            weights.append(r[j])
    period = float(np.average(estimates, weights=weights)) if estimates else float(best)
    return period, float(np.clip(comb[i], 0.0, 1.0))


# --------------------------------------------------------------------------- beats


def _track(local: np.ndarray, period: float, tightness: float) -> np.ndarray:
    """Ellis DP beat tracker; returns beat frame indices (ascending)."""
    n = local.size
    lo = max(1, int(round(period / 2.0)))
    hi = max(lo + 1, int(round(2.0 * period)))
    if n <= lo:
        return np.zeros(0, int)
    d = np.arange(lo, hi + 1)
    tx = -tightness * np.log(d / period) ** 2
    cum = local.astype(np.float64)
    back = np.full(n, -1, np.int64)
    # Every predecessor is >= lo frames back, so a block of lo frames is independent.
    for t0 in range(lo, n, lo):
        ts = np.arange(t0, min(t0 + lo, n))
        prev = ts[:, None] - d[None, :]
        cand = np.where(prev >= 0, cum[np.maximum(prev, 0)] + tx, -np.inf)
        k = cand.argmax(axis=1)
        rows = np.arange(ts.size)
        cum[ts] = local[ts] + cand[rows, k]
        back[ts] = prev[rows, k]
    # The best chain ends at the global maximum: a frame after the last real beat
    # can only extend that chain at a transition cost (ties in trailing silence
    # resolve to the earliest frame, i.e. the last real beat).
    beats = []
    t = int(np.argmax(cum))
    while t >= 0:
        beats.append(t)
        t = int(back[t])
    return _trim(local, np.asarray(beats[::-1], int))


def _trim(local: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Drop leading/trailing beats that fall in weak (e.g. silent) stretches."""
    if beats.size < 3:
        return beats
    win = signal.get_window("hann", 5)
    # Normalised convolution: edge beats are judged by their real neighbours only.
    smooth = np.convolve(local[beats], win, mode="same") / np.convolve(np.ones(beats.size), win, mode="same")
    strong = np.flatnonzero(smooth > 0.5 * np.sqrt(np.mean(smooth ** 2)))
    return beats[strong[0]:strong[-1] + 1] if strong.size else beats


def _salience(env: np.ndarray, frames: np.ndarray) -> float:
    return float(env[np.clip(frames, 0, env.size - 1)].mean()) if frames.size else 0.0


def _beat_grid(feats: _Features, p: ParamFn) -> tuple[np.ndarray, float, float]:
    """Beat frames, period (frames) and pulse clarity; empty when no steady pulse."""
    env, hop_s = feats.env, feats.hop_s
    period, clarity = _estimate_period(env, hop_s, p)
    if not np.isfinite(period) or clarity < float(p("min_clarity", 0.1)):
        return np.zeros(0, int), float("nan"), clarity
    tightness = float(p("tightness", 100.0))
    bpm_min, bpm_max = float(p("bpm_min", 60.0)), float(p("bpm_max", 200.0))

    def local_score(per: float) -> np.ndarray:
        return ndimage.gaussian_filter1d(env, max(0.5, per / 32.0))

    beats = _track(local_score(period), period, tightness)
    # Metrical level: the tracked grid vs. its double- and half-tempo versions,
    # each scored by prior(bpm) x mean onset strength at its grid points
    # (double = beats + midpoints, half = the stronger alternate beats).
    peak_env = ndimage.maximum_filter1d(env, 3)
    for _ in range(2):
        if beats.size < 4:
            break
        bpm = 60.0 / (period * hop_s)
        on = _salience(peak_env, beats)
        options = {period: float(_prior(bpm, p)) * on}
        if 2.0 * bpm <= bpm_max:
            off = _salience(peak_env, (beats[:-1] + beats[1:]) // 2)
            options[period / 2.0] = float(_prior(2.0 * bpm, p)) * 0.5 * (on + off)
        if bpm / 2.0 >= bpm_min:
            alt = max(_salience(peak_env, beats[0::2]), _salience(peak_env, beats[1::2]))
            options[period * 2.0] = float(_prior(bpm / 2.0, p)) * alt
        best = max(options, key=options.__getitem__)
        if best == period:
            break
        period = best
        beats = _track(local_score(period), period, tightness)
    return beats, period, clarity


def _snap(times: np.ndarray, fine: np.ndarray, fine_hop_s: float, window_s: float) -> np.ndarray:
    """Move each time to the strongest fine-onset peak within ``window_s`` (if it is a real hit)."""
    if times.size == 0:
        return times
    w = max(1, int(round(window_s / fine_hop_s)))
    pos = np.clip(np.round(times / fine_hop_s).astype(int)[:, None] + np.arange(-w, w + 1), 0, fine.size - 1)
    vals = fine[pos]
    k = vals.argmax(axis=1)
    peak = vals[np.arange(times.size), k]
    real = peak >= 0.3 * np.median(peak)
    return np.where(real, pos[np.arange(times.size), k] * fine_hop_s, times)


def _fit_period(times: np.ndarray, period_s: float) -> float:
    """Robust least-squares beat period from (beat index, time) pairs."""
    if times.size < 4:
        return period_s
    steps = np.maximum(1.0, np.round(np.diff(times) / period_s))
    k = np.concatenate([[0.0], np.cumsum(steps)])
    a = np.column_stack([k, np.ones_like(k)])
    keep = np.ones(times.size, bool)
    slope = period_s
    for _ in range(2):
        (slope, icpt), *_ = np.linalg.lstsq(a[keep], times[keep], rcond=None)
        resid = np.abs(times - (slope * k + icpt))
        keep = resid <= max(3.0 * 1.4826 * np.median(resid), 0.01)
        if keep.sum() < 4:
            break
    return float(slope) if abs(slope / period_s - 1.0) < 0.05 else period_s


def _downbeat_phase(beat_frames: np.ndarray, feats: _Features, meter: int,
                    drop_beats: list[int], p: ParamFn) -> int:
    """Index (0..meter-1) of the first downbeat within ``beat_frames``.

    Accent = z-scored peak bass level at the beat (kicks / 808 hits are loudest on
    bar starts) + ``downbeat_novelty`` x z-scored spectral change between the beat
    before and after it (chords and patterns change on bar lines). The phase with
    the highest median accent wins.
    """
    if beat_frames.size < meter:
        return 0
    n = feats.bass_db.size
    level = feats.bass_db[np.clip(beat_frames[:, None] + np.arange(-1, 3), 0, n - 1)].max(axis=1)
    bounds = np.concatenate([beat_frames, [min(n, 2 * beat_frames[-1] - beat_frames[-2])]])
    seg = np.stack([feats.mel_db[a:max(b, a + 1)].mean(axis=0) for a, b in zip(bounds[:-1], bounds[1:])])
    novelty = np.zeros(beat_frames.size)
    novelty[1:] = np.sqrt(np.mean((seg[1:] - seg[:-1]) ** 2, axis=1))

    def z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)

    accent = z(level) + float(p("downbeat_novelty", 0.5)) * z(novelty)
    score = np.array([np.median(accent[ph::meter]) for ph in range(meter)])
    for j in drop_beats:
        score[j % meter] += float(p("downbeat_drop_vote", 0.3))
    return int(np.argmax(score))


# --------------------------------------------------------------------------- onsets & energy


def _onsets(feats: _Features, p: ParamFn) -> np.ndarray:
    env, hop_s = feats.env, feats.hop_s
    min_gap = float(p("onset_min_gap", 0.06))
    peaks, _ = signal.find_peaks(env, distance=max(1, int(round(min_gap / hop_s))))
    if peaks.size == 0:
        return np.zeros(0)
    local = ndimage.uniform_filter1d(env, _odd(1.0 / hop_s), mode="nearest")
    ref = np.percentile(env[peaks], 95)
    keep = (env[peaks] >= float(p("onset_adapt", 1.5)) * local[peaks]) & (
        env[peaks] >= float(p("onset_strength", 0.3)) * ref)
    times = np.sort(_snap(peaks[keep] * hop_s, feats.fine_env, feats.fine_hop_s, 1.5 * hop_s))
    if times.size > 1:  # snapping can pull two peaks together
        times = times[np.concatenate([[True], np.diff(times) >= min_gap])]
    return times


def _smooth_norm(x: np.ndarray, seconds: float, hop_s: float) -> np.ndarray:
    """Moving average over ``seconds``, divided by its 95th percentile, clipped to 0..1."""
    s = ndimage.uniform_filter1d(x.astype(np.float64), _odd(seconds / hop_s), mode="nearest")
    ref = np.percentile(s, 95)
    return np.clip(s / ref, 0.0, 1.0) if ref > 0 else np.zeros_like(s)


def _window_means(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Means of ``x`` over ``[t - w, t)`` and ``[t, t + w)``; mask where both hold >= w/2 frames."""
    n = x.size
    cs = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(n)
    a, b = np.maximum(t - w, 0), np.minimum(t + w, n)
    before = (cs[t] - cs[a]) / np.maximum(t - a, 1)
    after = (cs[b] - cs[t]) / np.maximum(b - t, 1)
    valid = (t - a >= w / 2) & (b - t >= w / 2)
    return before, after, valid


def _nearest(values: np.ndarray, t: float) -> Optional[float]:
    return float(values[np.argmin(np.abs(values - t))]) if values.size else None


def _detect_drops(level: np.ndarray, bass: np.ndarray, hop_s: float, duration: float,
                  beats: np.ndarray, onsets: np.ndarray, p: ParamFn) -> list[Drop]:
    """Sustained jumps of the (beat-smoothed) full-band and bass levels."""
    window = min(float(p("drop_window", 2.0)), duration / 4.0)
    if window < 1.0:  # too short to tell a drop from the beat pulsing
        return []
    w = int(round(window / hop_s))
    eps = 0.02
    e_before, e_after, valid = _window_means(level, w)
    b_before, b_after, _ = _window_means(bass, w)
    r_e = (e_after + eps) / (e_before + eps)
    r_b = (b_after + eps) / (b_before + eps)
    score = np.sqrt(r_e * r_b)
    loudest = e_after[valid].max() if valid.any() else 0.0
    ok = (valid & (r_e > 1.1) & (r_b > 1.1)
          & (e_after >= float(p("drop_min_level", 0.6)) * loudest)
          & (e_before >= float(p("silence_level", 0.04))))
    score = np.where(ok, score, 0.0)
    peaks, _ = signal.find_peaks(score, height=float(p("drop_ratio", 1.5)),
                                 distance=max(1, int(round(float(p("drop_min_gap", 4.0)) / hop_s))))
    peaks = sorted(peaks, key=lambda i: -score[i])[: int(p("max_drops", 3))]
    drops: dict[float, Drop] = {}
    for i in peaks:
        t = i * hop_s
        anchor = _nearest(beats, t) if beats.size else _nearest(onsets, t)
        if anchor is not None and abs(anchor - t) <= 0.5:
            t = anchor
        if t not in drops:
            drops[t] = Drop(time=float(t), strength=float(r_e[i]))
    return [drops[t] for t in sorted(drops)]


# --------------------------------------------------------------------------- sections


@dataclass
class _Seg:
    start: float
    end: float
    level: float = 0.0
    silent: bool = False
    drop: bool = False  # starts at a drop
    label: AudioLabel = "body"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index ranges where ``mask`` is True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _segment(energy: np.ndarray, level: np.ndarray, silent: np.ndarray, hop_s: float, duration: float,
             drops: list[Drop], beats: np.ndarray, p: ParamFn) -> list[_Seg]:
    """Split ``[0, duration]`` at silence edges, drops and strong ``level`` changes."""
    min_len = float(p("min_section", 1.5))
    min_sil = int(round(float(p("min_silence", 0.5)) / hop_s))
    sil = [(a * hop_s, min(b * hop_s, duration)) for a, b in _runs(silent) if b - a >= min_sil]
    sounds, pos = [], 0.0
    for a, b in sil:
        if a > pos:
            sounds.append((pos, a))
        pos = b
    if pos < duration:
        sounds.append((pos, duration))

    w = int(round(min(float(p("seg_window", 2.0)), duration / 4.0) / hop_s))
    changes: list[tuple[float, float]] = []  # (time, |log step|)
    if w >= 4:
        before, after, valid = _window_means(level, w)
        step = np.where(valid, np.abs(np.log((after + 0.02) / (before + 0.02))), 0.0)
        peaks, _ = signal.find_peaks(step, height=math.log(float(p("seg_ratio", 1.5))),
                                     distance=max(1, int(round(min_len / hop_s))))
        for i in peaks:
            t = i * hop_s
            snap = _nearest(beats, t)
            changes.append((snap if snap is not None and abs(snap - t) <= 0.3 else t, step[i]))

    segs = [_Seg(a, b, silent=True) for a, b in sil]
    drop_times = [d.time for d in drops]
    for a, b in sounds:
        cuts = [t for t in drop_times if a + 0.05 < t < b - 0.05]
        for t, _ in sorted(changes, key=lambda c: -c[1]):  # strongest changes claim space first
            if a + min_len <= t <= b - min_len and all(abs(t - c) >= min_len for c in cuts):
                cuts.append(t)
        edges = [a, *sorted(cuts), b]
        for s, e in zip(edges[:-1], edges[1:]):
            segs.append(_Seg(s, e, drop=any(abs(s - t) < 0.05 for t in drop_times)))
    segs.sort(key=lambda s: s.start)
    for s in segs:
        f0, f1 = int(round(s.start / hop_s)), max(int(round(s.end / hop_s)), int(round(s.start / hop_s)) + 1)
        s.level = float(energy[f0:f1].mean()) if f0 < energy.size else 0.0
    return segs


def _label(segs: list[_Seg], drops: list[Drop], p: ParamFn) -> None:
    sound = [s for s in segs if not s.silent]
    if not sound:
        return
    ref = max(s.level for s in sound) or 1.0
    quiet = float(p("quiet_level", 0.5))
    first_drop = drops[0].time if drops else None
    drop_level: Optional[float] = None
    for k, s in enumerate(sound):
        later = k < len(sound) - 1
        if first_drop is not None and s.start < first_drop - 0.05:
            s.label = "intro" if s.level < float(p("intro_level", 0.25)) * ref else "build"
        elif s.drop:
            s.label, drop_level = "drop", s.level
        elif drop_level is not None:
            if s.level >= float(p("drop_keep", 0.7)) * drop_level:
                s.label = "drop"
            else:
                s.label = "break" if later else "outro"
        elif s.level < quiet * ref:
            earlier = k > 0
            s.label = "intro" if not earlier and later else ("outro" if earlier and not later else "break")
        else:
            s.label = "body"
    # A lull that leads straight into a later drop and gets louder is its build-up.
    for k in range(1, len(sound) - 1):
        s, nxt = sound[k], sound[k + 1]
        if s.label == "break" and nxt.drop and sound[k - 1].label == "break" and s.level > sound[k - 1].level:
            s.label = "build"
    for s in segs:
        if s.silent:
            s.label = "silence"


def _sections(energy: np.ndarray, level: np.ndarray, silent: np.ndarray, hop_s: float, duration: float,
              drops: list[Drop], beats: np.ndarray, p: ParamFn) -> list[AudioSection]:
    segs = _segment(energy, level, silent, hop_s, duration, drops, beats, p)
    _label(segs, drops, p)
    merged: list[_Seg] = []
    for s in segs:
        if merged and merged[-1].label == s.label:
            m = merged[-1]
            total = (m.end - m.start) + (s.end - s.start)
            m.level = (m.level * (m.end - m.start) + s.level * (s.end - s.start)) / max(total, _EPS)
            m.end = s.end
        else:
            merged.append(s)
    return [AudioSection(start=float(s.start), end=float(s.end), label=s.label, energy=float(s.level))
            for s in merged if s.end > s.start]


# --------------------------------------------------------------------------- entry point


def _timeline(ctx: AnalysisContext) -> tuple[np.ndarray, float]:
    """Mono audio trimmed / zero-padded to the timeline duration, and that duration."""
    audio = np.asarray(ctx.audio, np.float32).ravel()
    sr = float(ctx.sr)
    if ctx.n_frames > 1 and ctx.fps > 0:
        duration = ctx.duration
    else:  # standalone music: no meaningful video timeline
        duration = audio.size / sr if sr > 0 else 0.0
    n = int(round(duration * sr))
    if audio.size < n:
        audio = np.pad(audio, (0, n - audio.size))
    audio = np.nan_to_num(audio[:n])
    return audio - audio.mean() if audio.size else audio, duration


def analyze_audio(ctx: AnalysisContext) -> AudioAnalysis:
    """Tempo, beat grid, downbeats, onsets, drops, energy and sections of ``ctx.audio``."""

    def p(key: str, default: Any) -> Any:
        return ctx.config.p(NAME, key, default)

    sr = float(ctx.sr)
    hop_s = max(1, int(round(512 * sr / _REF_SR))) / sr if sr > 0 else 0.0
    y, duration = _timeline(ctx)
    floor = 10 ** (float(p("silence_db", -60.0)) / 20)
    if sr <= 0 or y.size < _pow2(2048 * sr / _REF_SR) or np.sqrt(np.mean(y.astype(np.float64) ** 2)) < floor:
        return AudioAnalysis(music=None, signals={}, hop_seconds=hop_s)

    feats = _features(y, sr, p)
    hop_s = feats.hop_s
    smooth_s = float(p("energy_smooth", 0.2))
    energy = _smooth_norm(feats.rms, smooth_s, hop_s)

    beat_frames, period, clarity = _beat_grid(feats, p)
    beats = np.zeros(0)
    bpm: Optional[float] = None
    if beat_frames.size >= 2:
        beats = _snap(beat_frames * hop_s, feats.fine_env, feats.fine_hop_s, float(p("snap_window", 0.04)))
        period_s = period * hop_s
        keep = np.concatenate([[True], np.diff(beats) >= 0.5 * period_s])
        beats, beat_frames = beats[keep], beat_frames[keep]
        bpm = 60.0 / _fit_period(beats, period_s)

    onsets = _onsets(feats, p)
    # Level changes are judged on energy smoothed over a whole beat, so the
    # kick pulsing of slow tracks does not look like rises and falls.
    level_s = max(smooth_s, 60.0 / bpm if bpm else 0.5)
    level = _smooth_norm(feats.rms, level_s, hop_s)
    drops = _detect_drops(level, _smooth_norm(feats.bass, level_s, hop_s), hop_s, duration, beats, onsets, p)

    meter = int(p("meter", 4))
    downbeats: list[float] = []
    if beats.size:
        drop_beats = [int(np.argmin(np.abs(beats - d.time))) for d in drops]
        phase = _downbeat_phase(beat_frames, feats, meter, drop_beats, p)
        downbeats = beats[phase::meter].tolist()

    smooth_rms = ndimage.uniform_filter1d(feats.rms, _odd(smooth_s / hop_s))
    silent = (energy < float(p("silence_level", 0.04))) | (
        smooth_rms < 10 ** (float(p("silence_frame_db", -50.0)) / 20))
    sections = _sections(energy, level, silent, hop_s, duration, drops, beats, p)

    dt = float(p("energy_dt", 0.05))
    grid = np.arange(int(math.ceil(duration / dt - 1e-9))) * dt
    curve = Curve(t0=0.0, dt=dt, values=np.interp(grid, np.arange(energy.size) * hop_s, energy).tolist())

    music = Music(
        bpm=bpm, beat_offset=float(beats[0]) if beats.size else None, meter=meter,
        beats=beats.tolist(), downbeats=downbeats, onsets=onsets.tolist(), drops=drops,
        sections=sections, energy=curve, has_speech=None, confidence=float(np.clip(clarity, 0.0, 1.0)),
    )
    signals = {
        "onset_env": feats.env,
        "rms": feats.rms,
        "energy": energy.astype(np.float32),
    }
    return AudioAnalysis(music=music, signals=signals, hop_seconds=hop_s)
