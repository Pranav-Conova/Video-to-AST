"""Time effects: freeze frames, duplicate-frame slow motion and speed ramps.

Viral "velocity" edits manipulate playback speed: a clip runs at normal (or
fast) speed, ramps sharply down into slow motion on a beat and ramps back up.
Playback rate is not observable directly, but it scales all *footage* motion
(camera travel, subjects, parallax) while leaving post effects that run on the
edit clock (digital zoom punches, shake, rotation) untouched. The analyzer
therefore builds its own footage-speed signal instead of using raw flow
magnitude:

1. **Flow decomposition.** DIS optical flow on downscaled gray frames; a
   similarity (zoom + rotation + translation of the frame centre) is fitted to
   the dense flow by trimmed least squares. Zoom and rotation are dropped, the
   centre translation and the non-rigid residual (mean deviation of the flow
   from the fit: subjects, parallax) are divided by the cumulative zoom of the
   shot so a digital zoom-in does not read as faster footage.
2. **Still frames.** A frame repeats its predecessor when its mean gray
   difference, its flow magnitude and the largest difference of any ~16 px
   block (a moving subject in an otherwise static shot keeps one block busy)
   are all at the level of compression noise; the block limit is relative to
   the shot's typical block change, within absolute bounds.
3. **Duplicate-frame slow motion.** Editors without frame interpolation repeat
   source frames: 0.5x shows every frame twice (novel, still, novel, still ...).
   Alternations of short still runs and short novel runs over at least three
   cycles, with a steady novel-to-novel gap and novel frames clearly above the
   repeats, form a segment whose rate is the inverse mean novel-to-novel
   gap, divided by the video's baseline cadence, so 30 fps footage in a 60 fps
   container (which repeats every frame everywhere) is not slow motion.
4. **Freezes.** Longer still runs (at least ``max(3 frames, 0.12 s)`` of
   identical displayed frames) inside a shot whose other frames clearly move:
   camera travel (so a flash or a digital zoom does not count as motion), or,
   in a static (tripod) shot, a subject whose motion stops dead into the hold
   and resumes right after it. A static shot with nothing moving, or whose
   subject comes to rest or leaves the frame, never yields a freeze. A hold
   running into a transition that covers the rest of the shot is assumed to
   continue under it.
5. **Speed curve.** Each novel frame's motion is spread evenly over itself and
   the repeats that precede it (the source advanced one frame over that whole
   run, so duplicate-frame slow motion reads as a smooth low rate, exactly like
   interpolated slow motion); freeze frames are dropped. The centre
   translation components and the residual are median filtered over at least
   one jitter period (wider inside Shake spans reported by the motion
   analysis): a median removes oscillating shake but keeps hard speed cuts
   sharp. Speed = |median translation| + median residual. The shot's "normal
   speed" reference is the typical level of its top decile and the relative
   rate is ``r(t) = speed(t) / reference``.
6. **Speed events.** Sustained ``r < slow_rate`` runs are slow-motion
   candidates. The transitions into and out of them are fitted with a
   plateau-ramp-plateau model (ramp start, end and easing by exhaustive least
   squares in log-rate); changes of at least 2x within ``ramp_max`` seconds
   (or deeper ones up to ``ramp_max_long`` that start from a steady plateau)
   become ``ramp_down`` / ``ramp_up`` while quicker ones are hard speed cuts
   that only delimit the slow span. A candidate entered and left only
   gradually (a car accelerating, a camera slowing down) is natural motion and
   is dropped; the others become ``slowmo``. A shot whose normal-speed motion
   is far above the video's typical motion gets a low-confidence ``fast`` node.
   Shots are judged only when the camera travel or the subject (residual)
   motion is clearly measurable on its own, and slow / fast spans running into
   a transition that covers the rest of the shot continue under it.

Rates are relative to the shot's own normal speed (motion only reveals relative
speed), so confidences stay in the 0.4-0.8 range except where frame repetition
gives direct evidence. ``Speed`` keyframes sample ``r(t)`` every ~0.1 s; a ramp's
``rate_from``/``rate_to``/``easing`` follow :func:`editast.fx.interp` semantics
(rate interpolated linearly in rate space along the easing curve). Spans use
the renderer's conventions (:func:`editast.render.compiler.source_positions`):
the rate at frame ``t`` drives the step from ``t`` to ``t + 1`` (flow measured
into frame ``t + 1``), and a ``Freeze`` over frames ``[a, b)`` means ``b - a``
identical frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from scipy import ndimage

from editast import fx
from editast.analyzers.types import EffectsResult, MotionAnalysis, ShotSegmentation, ShotSpan
from editast.ast.nodes import Freeze, Keyframe, Shake, Speed, SpeedKind
from editast.context import AnalysisContext

NAME = "timefx"
_RAMP_EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out")


class _Params:
    """``ctx.config.p("timefx", ...)`` with seconds-to-frames conversion."""

    def __init__(self, ctx: AnalysisContext) -> None:
        self._cfg = ctx.config
        self.fps = ctx.fps

    def __call__(self, key: str, default: Any) -> Any:
        return self._cfg.p(NAME, key, default)

    def frames(self, key: str, seconds: float, minimum: int = 1) -> int:
        return max(minimum, int(round(float(self(key, seconds)) * self.fps)))


@dataclass
class _Flow:
    """Per-frame global motion (index ``t`` = motion from frame ``t-1`` to ``t``; 0 at shot starts).

    Footage speed is ``|trans| + resid``; ``|trans|`` alone is camera travel.
    """

    trans: np.ndarray  # (n, 2) frame-centre translation, fraction of width, zoom-normalised
    resid: np.ndarray  # (n,) mean non-rigid flow, fraction of width, zoom-normalised
    total: np.ndarray  # (n,) mean flow magnitude, fraction of width
    log_zoom: np.ndarray  # (n,) inter-frame log zoom of the similarity fit


@dataclass
class _SpeedEvent:
    """A speed node in absolute frames ``[f0, f1)`` before conversion to AST."""

    kind: SpeedKind
    f0: int
    f1: int
    rate_from: float
    rate_to: float
    easing: str = "linear"
    confidence: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Ramp:
    start: int  # first ramp frame (index into the analysed curve)
    end: int  # first frame after the ramp (at the target level)
    rate_from: float
    rate_to: float
    easing: str
    long: bool = False  # longer than ``ramp_max`` (less certain to be an edit)


# --------------------------------------------------------------------------- public API


def analyze_time_effects(
    ctx: AnalysisContext,
    seg: ShotSegmentation,
    motion: Optional[MotionAnalysis] = None,
) -> EffectsResult:
    """Detect ``Speed`` (slowmo / fast / ramp_down / ramp_up) and ``Freeze`` effects per shot.

    ``motion`` is optional; its Shake effects widen the jitter filter of the
    speed curve inside their spans (the footage-speed signal itself is always
    computed here, see the module docstring). Signals: ``speed_rate``
    (relative playback rate of the step from frame ``t`` to ``t + 1``, the
    renderer's convention; 1 where not judged), ``motion_magnitude`` (smoothed
    footage speed from frame ``t - 1`` to ``t``, fraction of width per frame)
    and ``frame_repeat`` (1 where a frame repeats its predecessor).
    """
    prm = _Params(ctx)
    n = ctx.n_frames
    result = EffectsResult()
    shots = [ShotSpan(s.index, max(0, s.f0), min(s.f1, n)) for s in seg.shots if min(s.f1, n) > max(s.f0, 0)]
    if n < 2 or not shots:
        result.signals = {k: np.ones(n) if k == "speed_rate" else np.zeros(n)
                          for k in ("speed_rate", "motion_magnitude", "frame_repeat")}
        return result

    starts = np.zeros(n, bool)
    for s in shots:
        starts[s.f0] = True
    starts[0] = True
    usable = ~_fit_length(seg.transition_frames(), n) & ~starts
    flow = _flow(ctx.gray, starts, prm)
    camera = np.hypot(flow.trans[:, 0], flow.trans[:, 1])  # global (camera) travel per frame
    diff = np.asarray(ctx.features.diff, float)
    block = _block_change(ctx.gray, starts, prm)
    # Flashes / inversions / glitches: their flow says nothing about playback speed.
    outliers = _photometric_outliers(ctx, shots, prm)
    still = _still_frames(diff, block, flow.total, usable, shots, prm)
    flashed = _photometric_stills(ctx.gray, still, outliers, usable, camera, shots, prm)
    still |= flashed
    # A repeat under a flash differs from its predecessor only photometrically.
    diff_held, block_held = np.where(flashed, 0.0, diff), np.where(flashed, 0.0, block)

    freeze = np.zeros(n, bool)
    jump = np.zeros(n, bool)
    dup_segments: list[tuple[int, int, int, float]] = []  # (shot index, f0, f1, novel fraction)
    for shot in shots:
        dups, freezes = _repeat_structure(ctx.gray, shot, still, usable, diff_held, block_held, camera,
                                          outliers, prm)
        dup_segments += [(shot.index, a, b, q) for a, b, q in dups]
        for a, b, conf in freezes:  # [a, b): the held frame a and its repeats
            freeze[a + 1:b] = True
            result.add(shot.index, Freeze(start=ctx.t(a), end=ctx.t(b), confidence=conf,
                                          params={"frames": b - a}))
            if b < shot.f1:
                jump[b] = True  # motion into the first frame after a freeze may span the held time

    # Frames whose flow says nothing about playback speed: photometric outliers, and
    # fast digital zoom punches (dense flow is unreliable at several % zoom per frame).
    unjudged = outliers.copy()
    unjudged[1:] |= unjudged[:-1]  # the flow into frame t also depends on frame t-1
    unjudged |= np.abs(flow.log_zoom) > float(prm("zoom_max", 0.03))
    filled, valid = _fill_repeats(np.column_stack([flow.trans, flow.resid]), still,
                                  usable & ~freeze & ~jump & ~unjudged, shots, prm)
    window = _median_windows(ctx, motion, shots, n, prm)
    rate = np.ones(n)
    smooth = np.zeros(n)
    refs: dict[int, tuple[float, int]] = {}
    events: dict[int, list[_SpeedEvent]] = {}
    min_camera = float(prm("min_motion", 0.003))
    min_subject = float(prm("min_subject_motion", 0.004))
    min_seg = prm.frames("min_segment", 0.3, minimum=6)
    bridge_max = prm.frames("bridge_max", 0.3)
    bridge_piece = int(prm("bridge_min_run", 3))
    pct = float(prm("ref_percentile", 90.0))
    trans_frames = _fit_length(seg.transition_frames(), n)
    for shot in shots:
        fine, sm, ok, parts = _speed_curve(filled, valid, window, shot, prm)
        smooth[shot.f0:shot.f1] = np.where(ok, sm, 0.0)
        if ok.sum() < min_seg:
            continue
        # Judge speed only from clear camera travel or clear subject motion: a small
        # subject over a drifting camera is mostly flow noise and fit bias.
        if (_reference_level(parts[ok, 0], pct) < min_camera
                and _reference_level(parts[ok, 1], pct) < min_subject):
            continue
        ref = _reference_level(sm[ok], pct)
        refs[shot.index] = (ref, int(ok.sum()))
        r = np.where(ok, sm / ref, 1.0)
        r_fine = np.where(ok, fine / ref, 1.0)
        events[shot.index] = []
        for a, b in _bridge(ok, bridge_max, bridge_piece):
            known = ok[a:b]
            r[a:b] = _fill_gaps(r[a:b], known)
            r_fine[a:b] = _fill_gaps(r_fine[a:b], known)
            if known.sum() >= min_seg:
                events[shot.index] += _speed_events(r[a:b], r_fine[a:b], shot.f0 + a, prm)
        rate[shot.f0:shot.f1] = r
        for ev in events[shot.index]:
            _through_transitions(ev, shot, ok, trans_frames)
            _to_rate_frames(ev, shot)

    rate = _to_rate_signal(rate, shots)
    _add_fast_events(events, refs, rate, shots, prm)
    q_base = _baseline_cadence(still, usable & ~freeze, shots, refs, prm)
    _merge_duplicate_slowmo(events, dup_segments, usable & ~still, q_base, prm)

    step = prm.frames("keyframe_step", 0.1)
    for shot_index, evs in events.items():
        for ev in evs:
            result.add(shot_index, _to_speed(ctx, ev, rate, step))
    for effects in result.by_shot.values():
        effects.sort(key=lambda e: (e.start, e.end))
    result.signals = {
        "speed_rate": rate,
        "motion_magnitude": smooth,
        "frame_repeat": still.astype(float),
    }
    return result


# --------------------------------------------------------------------------- motion signal


def _fit_length(mask: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, bool)
    m = np.asarray(mask, bool)[:n]
    out[:m.size] = m
    return out


def _flow(gray: np.ndarray, starts: np.ndarray, prm: _Params) -> _Flow:
    """Dense DIS flow on downscaled frames, decomposed into a similarity and a residual.

    The similarity ``u = aX - bY + tx, v = bX + aY + ty`` (``X, Y`` relative to
    the frame centre) is fitted by least squares, refitted without points far
    from the first fit (independently moving subjects), and its zoom
    ``hypot(1 + a, b)`` accumulated per shot to normalise the translation and
    the residual to the zoom level of the shot start.
    """
    n, h, w = gray.shape
    s = min(1.0, float(prm("flow_long_side", 160)) / max(h, w))
    size = (max(16, int(round(w * s))), max(16, int(round(h * s))))
    sw, sh = size
    border = int(round(0.08 * min(sw, sh)))  # content entering the frame has no valid flow
    ys, xs = np.mgrid[border:sh - border:3, border:sw - border:3]
    ys, xs = ys.ravel(), xs.ravel()
    px, py = xs - (sw - 1) / 2.0, ys - (sh - 1) / 2.0
    one, zero = np.ones_like(px), np.zeros_like(px)
    design = np.vstack([np.column_stack([px, -py, one, zero]), np.column_stack([py, px, zero, one])])
    pinv = np.linalg.pinv(design)
    m = xs.size
    min_log = float(prm("zoom_noise", 0.003))  # per-frame |log zoom| below this is fit noise
    # Flow error grows with displacement (e.g. at the rim of a fast zoom); only deviations
    # beyond that count as independent motion.
    err_rel = float(prm("flow_rel_error", 0.2))
    err_floor = float(prm("flow_floor_px", 0.1))

    trans, resid, total = np.zeros((n, 2)), np.zeros(n), np.zeros(n)
    zoom = np.ones(n)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev = cv2.resize(gray[0], size, interpolation=cv2.INTER_AREA)
    for t in range(1, n):
        cur = cv2.resize(gray[t], size, interpolation=cv2.INTER_AREA)
        if not starts[t]:
            fl = dis.calc(prev, cur, None)
            obs = np.concatenate([fl[ys, xs, 0], fl[ys, xs, 1]]).astype(np.float64)
            p = pinv @ obs
            err = np.hypot(*(design @ p - obs).reshape(2, m))
            keep = err <= max(2.5 * float(np.median(err)), 0.25)
            if keep.mean() < 0.98:
                rows = np.concatenate([keep, keep])
                p = np.linalg.lstsq(design[rows], obs[rows], rcond=None)[0]
            model = design @ p
            err = np.hypot(*(model - obs).reshape(2, m))
            excess = err - err_rel * np.hypot(*model.reshape(2, m)) - err_floor
            trans[t] = p[2:] / sw
            resid[t] = float(np.maximum(excess, 0.0).mean()) / sw
            total[t] = float(np.hypot(obs[:m], obs[m:]).mean()) / sw
            zoom[t] = float(np.hypot(1.0 + p[0], p[1]))
        prev = cur

    # Cumulative zoom since the shot start (reset at every shot start).
    log_zoom = np.log(np.clip(zoom, 0.8, 1.25))
    log_zoom[starts] = 0.0
    steps = np.where(np.abs(log_zoom) >= min_log, log_zoom, 0.0)
    seg_id = np.cumsum(starts)
    cum = np.cumsum(steps)
    cum -= np.concatenate(([0.0], cum))[np.flatnonzero(starts)][seg_id - 1]
    scale = np.exp(np.clip(cum, -np.log(3.0), np.log(3.0)))
    return _Flow(trans=trans / scale[:, None], resid=resid / scale, total=total, log_zoom=log_zoom)


def _block_change(gray: np.ndarray, starts: np.ndarray, prm: _Params) -> np.ndarray:
    """Largest mean absolute gray difference (0..1) of any block, per frame.

    A moving subject in an otherwise static shot changes a few blocks a lot
    while barely moving the frame mean; a held frame changes no block beyond
    compression noise.
    """
    n, h, w = gray.shape
    block = max(4, int(round(float(prm("still_block", 0.09)) * min(h, w))))
    size = (max(1, w // block), max(1, h // block))
    out = np.zeros(n)
    for t in range(1, n):
        if not starts[t]:
            d = cv2.absdiff(gray[t], gray[t - 1]).astype(np.float32)
            out[t] = float(cv2.resize(d, size, interpolation=cv2.INTER_AREA).max()) / 255.0
    return out


def _still_frames(
    diff: np.ndarray,
    block: np.ndarray,
    flow_total: np.ndarray,
    usable: np.ndarray,
    shots: list[ShotSpan],
    prm: _Params,
) -> np.ndarray:
    """Frames that repeat their predecessor up to compression noise.

    Mean difference and flow must be at noise level everywhere; the block
    limit adapts to the shot: a fraction of its typical (75th percentile) block
    change, clipped to ``[still_block_floor, still_block_max]``, so a slowly
    moving subject in a static shot is not mistaken for a repeat while
    re-encoded holds in fast shots still are. Overlays (captions, stickers)
    animate on the edit clock over held footage: up to ``overlay_frames``
    frames that change only locally between two repeats are repeats too (a
    moving subject changes blocks continuously instead).
    """
    loose = (
        usable
        & (diff < float(prm("still_diff_max", 0.003)))
        & (flow_total < float(prm("still_flow_max", 0.002)))
    )
    still = loose.copy()
    rel = float(prm("still_rel", 0.35))
    lo, hi = float(prm("still_block_floor", 0.004)), float(prm("still_block_max", 0.025))
    for shot in shots:
        sl = slice(shot.f0 + 1, shot.f1)
        ok = usable[sl]
        level = float(np.percentile(block[sl][ok], 75)) if ok.any() else 0.0
        still[sl] &= block[sl] < float(np.clip(rel * level, lo, hi))
    max_overlay = int(prm("overlay_frames", 3))
    for a, b in _runs(loose & ~still):
        if b - a <= max_overlay and a > 0 and b < still.size and still[a - 1] and still[b]:
            still[a:b] = True
    return still


def _photometric_stills(
    gray: np.ndarray,
    still: np.ndarray,
    outlier: np.ndarray,
    usable: np.ndarray,
    camera: np.ndarray,
    shots: list[ShotSpan],
    prm: _Params,
) -> np.ndarray:
    """Repeats hidden under a flash or a grade change (a flash on a freeze frame).

    A frame next to a photometric jump repeats the last decidable frame when
    one is an affine function of the other, ``y = a x + b`` (a blend toward a
    colour, a brightness / contrast change, an inversion), up to compression
    noise (:func:`_affine_fit` residual at most ``photo_still_residual`` gray
    levels on half-resolution frames). The fitted contrast ratios are chained
    from the last normal frame; frames left with less than
    ``photo_min_contrast`` of it (the peak of a strong flash) are undecidable
    and take the verdict of the next decidable frame. Only shots whose camera
    moves are considered (a small moving subject is affine almost everywhere
    too), and only chains that touch a plain repeat.
    """
    n = gray.shape[0]
    out = np.zeros(n, bool)
    cand = outlier.copy()
    cand[1:] |= outlier[:-1]
    cand &= usable & ~still
    if not cand.any():
        return out
    tol = float(prm("photo_still_residual", 0.85))
    min_contrast = float(prm("photo_min_contrast", 0.35))
    min_motion = float(prm("freeze_min_motion", 0.002))
    h, w = gray.shape[1:]
    size = (max(8, w // 2), max(8, h // 2))
    for shot in shots:
        sl = slice(shot.f0 + 1, shot.f1)
        moving = usable[sl] & ~still[sl] & ~outlier[sl]
        if not cand[sl].any() or not moving.any() or float(np.median(camera[sl][moving])) < min_motion:
            continue
        last, level, pending = shot.f0, 1.0, []  # last decidable frame and its relative contrast
        for t in range(shot.f0 + 1, shot.f1):
            if not cand[t]:
                last, level, pending = t, 1.0, []
                continue
            x, y = (cv2.resize(gray[i], size, interpolation=cv2.INTER_AREA) for i in (last, t))
            residual, ratio = _affine_fit(x, y)
            if level * ratio < min_contrast:
                pending.append(t)
                continue
            if residual <= tol:
                out[t] = True
                out[pending] = True
            last, level, pending = t, level * ratio, []
    for a, b in _runs(out | still):
        if not still[a:b].any():
            out[a:b] = False
    return out


def _affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """``(residual, contrast ratio y / x)`` of a robust affine fit between two images.

    The lower-contrast image is fitted as ``a * other + b`` (so the noise of
    the high-contrast one is not amplified), twice, the second time without
    the 20 % worst pixels (an overlay that does not flash with the footage).
    The residual is the mean of the 90 % best pixels divided by ``|a|`` (<= 1),
    in gray levels: a contrast drop hides motion in proportion. ``|a|`` is the
    contrast ratio of the footage itself, unaffected by overlays.
    """
    swap = float(y.std()) > float(x.std())
    lo, hi = (x, y) if swap else (y, x)
    xs, ys = hi.astype(np.float32).ravel(), lo.astype(np.float32).ravel()
    keep = np.ones(xs.size, bool)
    for _ in range(2):
        xm, ym = float(xs[keep].mean()), float(ys[keep].mean())
        var = float(((xs[keep] - xm) ** 2).mean())
        if var < 1e-6:
            return float("inf"), 0.0
        a = float(((xs[keep] - xm) * (ys[keep] - ym)).mean()) / var
        res = np.abs(ys - a * xs - (ym - a * xm))
        keep = res <= np.quantile(res, 0.8)
    m = int(0.9 * res.size)
    scale = max(abs(a), 1e-3)
    return float(np.partition(res, m - 1)[:m].mean()) / scale, (1.0 / scale if swap else scale)


def _photometric_outliers(ctx: AnalysisContext, shots: list[ShotSpan], prm: _Params) -> np.ndarray:
    """Frames whose brightness, contrast or colour histogram jump away from the shot's
    rolling median (flashes, inversions, glitches): their optical flow is meaningless."""
    f = ctx.features
    out = np.zeros(ctx.n_frames, bool)
    size = prm.frames("photo_window", 0.5, minimum=5) | 1
    for shot in shots:
        sl = slice(shot.f0, shot.f1)
        k = min(size, (shot.f1 - shot.f0) | 1)
        if k < 3:
            continue
        lum, std = np.asarray(f.luma_mean[sl], float), np.asarray(f.luma_std[sl], float)
        hist = np.asarray(f.hist[sl], float)
        hmed = ndimage.median_filter(hist, size=(k, 1), mode="nearest")
        hmed /= np.maximum(hmed.sum(axis=1, keepdims=True), 1e-9)
        smed = ndimage.median_filter(std, size=k, mode="nearest")
        out[sl] = (
            (np.abs(lum - ndimage.median_filter(lum, size=k, mode="nearest")) > float(prm("photo_luma", 0.03)))
            | (np.abs(std / np.maximum(smed, 1e-3) - 1.0) > float(prm("photo_contrast", 0.15)))
            | (1.0 - np.minimum(hist, hmed).sum(axis=1) > float(prm("photo_hist", 0.12)))
        )
    return out


def _median_windows(
    ctx: AnalysisContext, motion: Optional[MotionAnalysis], shots: list[ShotSpan], n: int, prm: _Params
) -> np.ndarray:
    """Per-frame median window (odd, frames): one jitter period inside Shake spans."""
    base = prm.frames("smooth_median", 0.23, minimum=3) | 1
    window = np.full(n, base, int)
    if motion is None:
        return window
    cap = prm.frames("shake_window_max", 0.5, minimum=base) | 1
    for shot in shots:
        for eff in motion.effects.by_shot.get(shot.index, []):
            if not isinstance(eff, Shake) or eff.frequency <= 0:
                continue
            size = min(cap, int(np.ceil(ctx.fps / eff.frequency)) | 1)
            a, b = max(shot.f0, ctx.f(eff.start)), min(shot.f1, ctx.f(eff.end) + 1)
            window[a:b] = np.maximum(window[a:b], size)
    return window


# --------------------------------------------------------------------------- repeats & freezes


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index pairs of the True runs of a boolean array."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _bridge(ok: np.ndarray, max_gap: int, min_run: int) -> list[tuple[int, int]]:
    """Runs of ``ok`` (at least ``min_run`` long) joined across gaps of at most ``max_gap``.

    A digital zoom punch or a flash on the beat is unjudgeable for a few frames,
    and edits often change speed exactly there: the levels on both sides of
    such a gap must be compared as one continuous curve.
    """
    out: list[list[int]] = []
    for a, b in _runs(ok):
        if b - a < min_run:
            continue
        if out and a - out[-1][1] <= max_gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _fill_gaps(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """``values`` with the unknown entries interpolated linearly in log space."""
    if known.all():
        return values
    idx = np.flatnonzero(known)
    logs = np.log(np.maximum(values[idx], 1e-3))
    return np.exp(np.interp(np.arange(values.size), idx, logs))


def _repeat_structure(
    gray: np.ndarray,
    shot: ShotSpan,
    still: np.ndarray,
    usable: np.ndarray,
    diff: np.ndarray,
    block: np.ndarray,
    camera: np.ndarray,
    outlier: np.ndarray,
    prm: _Params,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Duplicate-frame segments ``(f0, f1, novel_fraction)`` and freezes ``(held, end, confidence)``.

    Works on the shot interior (frames after the first), absolute frame indices.
    """
    base = shot.f0 + 1
    st, ok = still[base:shot.f1], usable[base:shot.f1]
    if st.size < 3:
        return [], []
    state = np.where(~ok, 2, np.where(st, 1, 0))  # 0 novel, 1 still, 2 unusable
    change = np.flatnonzero(np.diff(state)) + 1
    bounds = np.concatenate(([0], change, [state.size]))
    runs = [(int(state[s]), int(s), int(e)) for s, e in zip(bounds[:-1], bounds[1:])]
    sl = slice(base, shot.f1)
    dups = [(base + s, base + e, q) for s, e, q in _duplicate_segments(runs, st, ok, diff[sl], block[sl], prm)]
    freezes = _freezes(gray, base, runs, st, ok, [(s - base, e - base) for s, e, _ in dups],
                       diff[sl], block[sl], camera[sl], outlier[sl], prm)
    return dups, [(base + a, base + b, conf) for a, b, conf in freezes]


def _duplicate_segments(
    runs: list[tuple[int, int, int]],
    st: np.ndarray,
    ok: np.ndarray,
    diff: np.ndarray,
    block: np.ndarray,
    prm: _Params,
) -> list[tuple[int, int, float]]:
    """Duplicate cadence: short still runs separated by short novel runs, over several cycles.

    ``runs`` are ``(state, start, end)`` with state 0 novel, 1 still, 2 unusable;
    returns ``(start, end, novel_fraction)`` in the same (shot-interior) indices.
    """
    max_dup = int(prm("max_dup_run", 4))
    max_novel = int(prm("max_novel_run", 2))
    segments: list[list[int]] = []  # [start, end, still_runs]
    cur: Optional[list[int]] = None
    for k, (kind, s, e) in enumerate(runs):
        if kind == 1 and e - s <= max_dup:
            if cur is None:
                start = s - 1 if k > 0 and runs[k - 1][0] == 0 else s
                cur = [start, e, 1]
            else:
                cur[1], cur[2] = e, cur[2] + 1
        elif kind == 0 and e - s <= max_novel and cur is not None:
            continue
        elif cur is not None:
            segments.append(cur)
            cur = None
    if cur is not None:
        segments.append(cur)
    merged: list[list[int]] = []
    for sgm in segments:  # tolerate an isolated misclassified frame
        if merged and sgm[0] - merged[-1][1] <= int(prm("dup_merge_gap", 3)):
            merged[-1][1], merged[-1][2] = sgm[1], merged[-1][2] + sgm[2]
        else:
            merged.append(sgm)
    out: list[tuple[int, int, float]] = []
    min_len = prm.frames("dup_min_duration", 0.2, minimum=4)
    for s, e, cycles in merged:
        if cycles < int(prm("dup_min_cycles", 3)) or e - s < min_len:
            continue
        seg_still = st[s:e] & ok[s:e]
        seg_novel = ~st[s:e] & ok[s:e]
        if not seg_novel.any():
            continue
        d, bl = diff[s:e], block[s:e]
        if (np.median(d[seg_novel]) < 2.5 * max(float(np.median(d[seg_still])), 1e-4)
                or np.median(bl[seg_novel]) < 2.5 * max(float(np.median(bl[seg_still])), 1e-4)):
            continue  # repeats are not clearly separated from novel frames
        # Rate from complete cycles (novel-to-novel gaps); the last cycle may be truncated.
        pos = np.flatnonzero(seg_novel)
        gaps = np.diff(pos)
        if gaps.size >= 2 and gaps.std() > float(prm("dup_regularity", 0.35)) * gaps.mean():
            continue  # no steady cadence: motion flickering around the noise floor
        q = 1.0 / float(gaps.mean()) if pos.size >= 3 else float(seg_novel.sum()) / float(ok[s:e].sum())
        out.append((s, e, q))
    return out


def _freezes(
    gray: np.ndarray,
    base: int,
    runs: list[tuple[int, int, int]],
    st: np.ndarray,
    ok: np.ndarray,
    dups: list[tuple[int, int]],
    diff: np.ndarray,
    block: np.ndarray,
    camera: np.ndarray,
    outlier: np.ndarray,
    prm: _Params,
) -> list[tuple[int, int, float]]:
    """Holds ``(held, end, confidence)`` in shot-interior indices (``base`` = absolute index 0).

    Long still runs, or short ones outside a duplicate cadence, that are clearly
    calmer than the rest of the shot, which must move: either the camera
    (global travel ``camera``), or, in a static shot, a subject. Motion must stop
    dead into the hold and resume at once after it, relative to the local
    motion level (``freeze_onset``): a camera or subject easing to a stop, or a
    ramp down to zero speed, is not a freeze, while a freeze after a ramp into
    slow motion is. In a static shot the subject's motion must also continue
    across the hold (:func:`_continuity`), since footage resumes where it was
    frozen while a subject that paused by itself usually moves on differently.
    """
    max_dup = int(prm("max_dup_run", 4))
    min_hold = max(int(prm("freeze_min_frames", 3)), prm.frames("freeze_min_duration", 0.12))
    min_motion = float(prm("freeze_min_motion", 0.002))
    min_block = float(prm("freeze_min_block", 0.006))
    diff_max = float(prm("freeze_diff_max", 0.003))
    rel = float(prm("freeze_rel", 0.35))
    onset = float(prm("freeze_onset", 0.5))
    window = prm.frames("freeze_onset_window", 0.2, minimum=3)
    min_continuity = float(prm("freeze_continuity", 0.5))
    novel = ok & ~st
    freezes: list[tuple[int, int, float]] = []
    for k, (kind, s, e) in enumerate(runs):
        if kind != 1 or e - s + 1 < min_hold:
            continue
        if e - s <= max_dup and any(a <= s and e <= b for a, b in dups):
            continue
        moving = novel.copy()
        moving[s:e] = False
        if moving.sum() < max(3, 0.2 * (st.size - (e - s))):
            continue
        d_move, b_move = float(np.median(diff[moving])), float(np.median(block[moving]))
        d_hold, b_hold = float(np.median(diff[s:e])), float(np.median(block[s:e]))
        if d_hold > diff_max or (d_hold > rel * d_move and b_hold > rel * b_move):
            continue  # the hold is not clearly calmer than the rest of the shot
        conf = 0.9 if d_hold < 0.5 * diff_max else 0.7
        static = float(np.median(camera[moving])) < min_motion
        # Subject change in a static shot, else the mean change (flow fails on the jump out of a
        # freeze whose footage kept running).
        level = block if static else diff
        if not _abrupt(level, novel & ~outlier, s - 1, e, window, onset):
            continue
        if static:
            # Only subjects move: the motion into the held frame (index s - 1) and out of
            # the hold (index e) must both be at full strength. A hold that runs to the
            # shot end could be a subject that left the frame.
            if (s < 1 or e >= st.size or not (ok[s - 1] and ok[e]) or b_move < min_block
                    or min(block[s - 1], block[e]) < onset * b_move
                    or _continuity(gray, base + s - 1, base + e) < min_continuity):
                continue
            conf = min(conf, 0.75)
        # A transition covering the rest of the shot hides the hold: assume it continues.
        held = 0 if k == 1 and runs[0][0] == 2 else s
        end = st.size if k == len(runs) - 2 and runs[-1][0] == 2 else e
        freezes.append((held - 1, end, conf))
    return freezes


def _abrupt(level: np.ndarray, valid: np.ndarray, into: int, out_of: int, window: int, onset: float) -> bool:
    """Does motion stop dead at ``into`` (the step into the held frame) and resume at ``out_of``?

    Each step must reach ``onset`` times the median of the ``window`` valid
    steps on its side; a side without such context (or an invalid step, e.g.
    under a flash) is not judged.
    """
    n = level.size
    if 0 <= into < n and valid[into]:
        before = level[max(0, into - window):into][valid[max(0, into - window):into]]
        if before.size >= 2 and level[into] < onset * float(np.median(before)):
            return False
    if 0 <= out_of < n and valid[out_of]:
        after = level[out_of + 1:out_of + 1 + window][valid[out_of + 1:out_of + 1 + window]]
        if after.size >= 2 and level[out_of] < onset * float(np.median(after)):
            return False
    return True


def _continuity(gray: np.ndarray, held: int, nxt: int) -> float:
    """Does the motion out of a hold continue the motion into it? Weighted cosine, -1..1.

    Both flows live on the held frame's grid: backward to its predecessor and
    forward to the first frame after the hold, so on the moving pixels the
    forward flow points opposite to the backward one when the footage moves on
    the same way. Only the direction is compared: footage that kept running
    under the freeze resumes with a jump.
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    back = dis.calc(gray[held], gray[held - 1], None)
    fwd = dis.calc(gray[held], gray[nxt], None)
    mb, mf = np.linalg.norm(back, axis=2), np.linalg.norm(fwd, axis=2)
    both = (mb > 0.5) & (mf > 0.5)
    if not both.any():
        return 0.0
    dot = -float((fwd[both] * back[both]).sum())
    return dot / max(float((mb * mf)[both].sum()), 1e-6)


def _fill_repeats(
    values: np.ndarray,
    still: np.ndarray,
    usable: np.ndarray,
    shots: list[ShotSpan],
    prm: _Params,
) -> tuple[np.ndarray, np.ndarray]:
    """Spread each novel frame's motion ``values`` (n, k) over itself and the repeats before it.

    Frames ``k+1 .. m-1`` repeating frame ``k`` and the next novel frame ``m``
    together show one source step, so each gets ``1 / (m - k)`` of the motion
    measured into ``m`` (and is valid only when ``m`` is ``usable``). Returns
    ``(filled, valid)``; long still runs keep their own (zero) motion, and a
    short run of repeats at the very end of a shot (no novel frame follows, so
    its cadence is unknown) is invalid.
    """
    max_dup = int(prm("max_dup_run", 4))
    filled = values.astype(float).copy()
    valid = usable.copy()
    for shot in shots:
        a, b = shot.f0 + 1, shot.f1
        if b - a < 1:
            continue
        # Work back to front: a novel frame then opens the group of the repeats preceding it.
        st = still[a:b][::-1]
        vals = values[a:b][::-1]
        group = np.cumsum(~st)  # group 0 = trailing repeats of the shot (no novel frame after them)
        sizes = np.bincount(group)
        novel = np.zeros((sizes.size, values.shape[1]))
        novel_ok = np.zeros(sizes.size, bool)
        novel_idx = np.flatnonzero(~st)
        novel[group[novel_idx]] = vals[novel_idx]
        novel_ok[group[novel_idx]] = usable[a:b][::-1][novel_idx]
        size = sizes[group]
        short = (size - 1 <= max_dup) & (group > 0)  # a novel frame and at most max_dup repeats
        long_tail = (group == 0) & (size > max_dup)
        filled[a:b] = np.where(short[:, None], novel[group] / size[:, None], vals)[::-1]
        valid[a:b] &= ((short & novel_ok[group]) | ~short & ((group > 0) | long_tail))[::-1]
    return filled, valid


def _baseline_cadence(
    still: np.ndarray,
    ok: np.ndarray,
    shots: list[ShotSpan],
    refs: dict[int, tuple[float, int]],
    prm: _Params,
) -> float:
    """Novel-frame fraction of normal footage (1.0, or e.g. 0.5 for 30 fps in a 60 fps file)."""
    win = prm.frames("cadence_window", 0.5, minimum=5)
    values = []
    for shot in shots:
        if shot.index not in refs:
            continue
        a, b = shot.f0 + 1, shot.f1
        m = ok[a:b]
        if m.sum() < win:
            continue
        novel = (~still[a:b] & m).astype(float)
        frac = ndimage.uniform_filter1d(novel, win, mode="nearest") / np.maximum(
            ndimage.uniform_filter1d(m.astype(float), win, mode="nearest"), 1e-6)
        values.append(frac[m])
    if not values:
        return 1.0
    q = float(np.percentile(np.concatenate(values), 95))
    return 1.0 if q >= 0.85 else max(q, 0.2)


# --------------------------------------------------------------------------- speed curve


def _speed_curve(
    filled: np.ndarray, valid: np.ndarray, window: np.ndarray, shot: ShotSpan, prm: _Params
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Footage speed inside each valid run of a shot.

    ``filled`` holds ``(tx, ty, residual)`` per frame. The translation is median
    filtered per component (the median of an oscillation about a level is the
    level, and a step stays a step); frames whose remaining jitter is
    comparable to the speed itself (and above the ``jitter_floor`` of flow
    noise) for at least ``jitter_min_duration`` are unjudgeable and leave
    ``ok``. Returns ``(fine, smooth, ok, parts)``: ``fine`` is the
    median-filtered speed (keeps hard speed cuts sharp for the ramp fit),
    ``smooth`` is additionally Gaussian smoothed (stable levels) and ``parts``
    ``(m, 2)`` holds its camera (translation) and subject (residual) components.
    """
    vals = filled[shot.f0:shot.f1]
    ok = valid[shot.f0:shot.f1].copy()
    win = window[shot.f0:shot.f1]
    fine = np.zeros(ok.size)
    smooth = np.zeros(ok.size)
    parts = np.zeros((ok.size, 2))
    sigma_jit = max(1.0, float(prm("jitter_sigma", 0.1)) * prm.fps)
    sigma = max(1.0, float(prm("smooth_sigma", 0.035)) * prm.fps)
    jitter_ratio = float(prm("jitter_ratio", 0.8))
    jitter_floor = float(prm("jitter_floor", 0.0015))
    jitter_min = prm.frames("jitter_min_duration", 0.2, minimum=3)
    shaky = np.zeros(ok.size, bool)
    for a, b in _runs(ok):
        trans = vals[a:b, :2]
        med = np.zeros((b - a, 3))
        for size in np.unique(win[a:b]):
            sel = win[a:b] == size
            med[sel] = _running_median(vals[a:b], int(size))[sel]
        parts[a:b] = np.column_stack([np.hypot(med[:, 0], med[:, 1]), med[:, 2]])
        x = parts[a:b].sum(axis=1)
        jitter = np.sqrt(ndimage.gaussian_filter1d(((trans - med[:, :2]) ** 2).sum(axis=1), sigma_jit,
                                                   mode="nearest"))
        unsure = jitter > np.maximum(jitter_ratio * x, jitter_floor)
        for s, e in _runs(unsure):
            if e - s >= jitter_min:
                shaky[a + s:a + e] = True
        fine[a:b] = x
    ok &= ~shaky
    for a, b in _runs(ok):
        smooth[a:b] = ndimage.gaussian_filter1d(fine[a:b], sigma, mode="nearest") if b - a >= 3 else fine[a:b]
    return fine, smooth, ok, parts


def _running_median(x: np.ndarray, size: int) -> np.ndarray:
    """Centred running median along axis 0 whose window is truncated at the ends.

    Unlike reflect / nearest padding, truncation never lets one or two outlier
    frames at the start of a run dominate the first windows.
    """
    half = size // 2
    pad = np.full((half,) + x.shape[1:], np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, x, pad]), size, axis=0)
    return np.nanmedian(windows, axis=-1)


def _reference_level(values: np.ndarray, percentile: float) -> float:
    """Typical level of the top of the curve (the shot's normal speed)."""
    top = float(np.percentile(values, percentile))
    if top <= 0:
        return 0.0
    return float(np.median(values[values >= 0.7 * top]))


# --------------------------------------------------------------------------- speed events


def _speed_events(r: np.ndarray, fine: np.ndarray, offset: int, prm: _Params) -> list[_SpeedEvent]:
    """Slow-motion spans and the ramps into / out of them in one continuous run.

    ``r`` is the smoothed relative rate (levels, detection), ``fine`` the
    median-only rate the ramp shapes are fitted to.
    """
    slow_rate = float(prm("slow_rate", 0.6))
    min_rate = float(prm("min_rate", 0.06))
    ramp_min = prm.frames("ramp_min", 0.2, minimum=4)
    slow_min = prm.frames("slowmo_min_duration", 0.2, minimum=3)
    gap_max = prm.frames("slow_gap", 0.15)
    n = r.size
    slow_runs: list[tuple[int, int]] = []
    for a, b in _runs(r < slow_rate):  # a brief rise inside a slow span is noise
        if slow_runs and a - slow_runs[-1][1] <= gap_max:
            slow_runs[-1] = (slow_runs[-1][0], b)
        else:
            slow_runs.append((a, b))
    events: list[_SpeedEvent] = []
    for k, (a, b) in enumerate(slow_runs):
        prev_end = slow_runs[k - 1][1] if k else 0
        next_start = slow_runs[k + 1][0] if k + 1 < len(slow_runs) else n
        seg = r[a:b]
        lo = float(np.median(seg)) if b - a >= slow_min else float(seg.min())
        if lo < min_rate:
            continue  # the camera simply stopped; not a speed effect
        mid = a + int(np.argmin(seg))
        s_slow, e_slow = a, b
        pieces: list[_SpeedEvent] = []
        sharp = False  # entered or left through a ramp or a hard speed cut
        if a > 0:
            ramp = _transition(r, fine, a, mid, lo, prev_end, prm)
            if ramp is not None:
                sharp = True
                if ramp.end - ramp.start >= ramp_min:
                    s_slow = ramp.end
                    pieces.append(_ramp_event("ramp_down", ramp, offset))
                else:  # hard speed cut: the slow span starts mid-transition
                    s_slow = (ramp.start + ramp.end + 1) // 2
        if b < n:
            # Ramp up = ramp down of the time-reversed curve; reversed index x is frame n-1-x,
            # so the reversed span [s, e) is [n - e, n - s) here.
            ramp = _transition(r[::-1], fine[::-1], n - b, n - 1 - mid, lo, n - next_start, prm)
            if ramp is not None:
                sharp = True
                start, end = n - ramp.end, n - ramp.start
                if end - start >= ramp_min:
                    e_slow = start
                    up = _Ramp(start, end, ramp.rate_to, ramp.rate_from, _mirror_easing(ramp.easing), ramp.long)
                    pieces.append(_ramp_event("ramp_up", up, offset))
                else:
                    e_slow = (start + end) // 2
        if not sharp:
            continue  # a gradual drift (natural acceleration / deceleration), not an edit
        if e_slow - s_slow >= slow_min:
            level = float(np.median(r[s_slow:e_slow]))
            conf = 0.4 + 0.2 * float(np.clip((slow_rate - level) / slow_rate, 0.0, 1.0)) + 0.1 * bool(pieces)
            pieces.append(_SpeedEvent("slowmo", offset + s_slow, offset + e_slow, level, level,
                                      confidence=round(conf, 3), params={"method": "motion"}))
        events += pieces
    return events


def _transition(
    r: np.ndarray, fine: np.ndarray, a: int, mid: int, lo: float, bound: int, prm: _Params
) -> Optional[_Ramp]:
    """The edit-like drop into the slow run starting at ``a`` (see :func:`_fit_ramp`), or ``None``.

    A drop of at least ``ramp_ratio`` within ``ramp_max`` seconds qualifies (a
    ramp, or a hard speed cut when very short). A longer, eased drop of up to
    ``ramp_max_long`` seconds qualifies only when it is deep (``ramp_ratio_long``)
    and starts from a steady plateau of at least ``ramp_plateau`` seconds: a
    natural deceleration has no such plateau-ramp-plateau shape.
    """
    ramp_max = prm.frames("ramp_max", 0.6, minimum=3)
    ramp = _fit_ramp(r, fine, a, mid, lo, ramp_max, bound)
    if ramp is not None and ramp.rate_from >= float(prm("ramp_ratio", 1.8)) * ramp.rate_to:
        return ramp
    ramp = _fit_ramp(r, fine, a, mid, lo, prm.frames("ramp_max_long", 1.0, minimum=ramp_max), bound)
    if ramp is None or ramp.rate_from < float(prm("ramp_ratio_long", 2.5)) * ramp.rate_to:
        return None
    size = prm.frames("ramp_plateau", 0.2, minimum=3)
    before = r[max(bound, ramp.start - 1 - size):ramp.start]
    if before.size <= size or np.abs(before / ramp.rate_from - 1.0).max() > float(prm("plateau_tol", 0.15)):
        return None
    ramp.long = ramp.end - ramp.start > ramp_max
    return ramp


def _fit_ramp(
    r: np.ndarray, fine: np.ndarray, a: int, mid: int, lo: float, ramp_max: int, bound: int = 0
) -> Optional[_Ramp]:
    """Fit a high-plateau -> ramp -> low-plateau model ending in ``[a, mid]``.

    ``r[a]`` is the first frame below the slow threshold and ``mid`` the bottom
    of the low run (level ``lo``); the high plateau is searched in
    ``[max(bound, a - ramp_max - 2), a)`` (``bound`` = end of the previous slow
    run). The high level comes from the smoothed ``r``, the shape is fitted to
    ``fine``. The model is the renderer's
    (:func:`editast.render.compiler.source_positions`): ramp frames ``[start, end)``
    take ``fx.interp(hi, low, (k - start + 0.5) / (end - start), easing)``, so a
    rendered ramp is fitted exactly. Exhaustive search over start, end, easing
    and the target level ``low`` (between the bottom and ``lo``: a ramp that runs
    to the end of the shot never settles at its median), error in log-rate.
    Returns ``None`` when there is no higher level to ramp from, or the best
    ramp is too long or not monotonic.
    """
    w0 = max(bound, a - ramp_max - 2)
    if w0 >= a:
        return None
    peak = w0 + int(np.argmax(r[w0:a]))
    hi = float(r[peak])
    w1 = max(mid, a) + 1  # fitted frames are [peak, w1); the ramp ends at or before the bottom
    if hi <= lo or w1 - peak < 2:
        return None
    k = np.arange(peak, w1, dtype=float)
    y = np.log(np.maximum(fine[peak:w1], 1e-3))
    bottom = float(np.clip(np.min(fine[a:w1]), 1e-3, lo))
    lows = np.geomspace(bottom, lo, 6) if bottom < 0.95 * lo else np.array([lo])
    best: Optional[tuple[float, int, int, str, float]] = None
    for i in range(peak + 1, w1):
        ends = np.arange(i + 1, min(w1, i + int(1.5 * ramp_max)) + 1)
        x = np.clip((k[None, :] - i + 0.5) / (ends[:, None] - i), 0.0, 1.0)  # (n_ends, n_frames)
        for name in _RAMP_EASINGS:
            e = np.asarray(fx.ease(x, name))[None]  # (1, n_ends, n_frames)
            model = np.log(np.maximum(hi + (lows[:, None, None] - hi) * e, 1e-3))
            sse = ((model - y) ** 2).sum(axis=2)  # (n_lows, n_ends)
            li, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
            if best is None or sse[li, j] < best[0] - 1e-12:
                best = (float(sse[li, j]), int(i), int(ends[j]), name, float(lows[li]))
    if best is None:
        return None
    _, i, j, name, low = best
    if j - i > ramp_max:
        return None  # gradual change (e.g. natural deceleration), not an edit ramp
    seg = np.diff(fine[i - 1:min(j + 1, fine.size)])
    if seg.size > 2 and (seg <= 0.05 * hi).mean() < 0.75:
        return None  # not monotonic
    return _Ramp(i, j, hi, low, name)


def _through_transitions(ev: _SpeedEvent, shot: ShotSpan, ok: np.ndarray, trans: np.ndarray) -> None:
    """Extend a held-rate event (flow frames) over unjudged frames up to a shot end that a
    transition covers, in place: speed is unobservable under a flash or fade, and edits
    rarely change it there."""
    if ev.kind not in ("slowmo", "fast"):
        return
    a, b = ev.f0 - shot.f0, ev.f1 - shot.f0
    head, tail = slice(1, a), slice(b, ok.size)
    if b < ok.size and not ok[tail].any() and trans[shot.f0 + b:shot.f1].mean() >= 0.5:
        ev.f1 = shot.f1
    if a > 1 and not ok[head].any() and trans[shot.f0 + 1:ev.f0].mean() >= 0.5:
        ev.f0 = shot.f0 + 1


def _to_rate_frames(ev: _SpeedEvent, shot: ShotSpan) -> None:
    """Move a motion-based event from flow frames to rate frames, in place.

    Flow frame ``t`` measures the step ``t-1 -> t``; the renderer's rate at
    frame ``t`` drives the step ``t -> t+1``. The step out of the shot's last
    frame is never observed, so an event that runs to the end of the shot keeps
    running to it.
    """
    ev.f0 = max(shot.f0, ev.f0 - 1)
    if ev.f1 < shot.f1:
        ev.f1 = max(ev.f0 + 1, ev.f1 - 1)


def _to_rate_signal(rate: np.ndarray, shots: list[ShotSpan]) -> np.ndarray:
    """Per-frame rate in rate frames (see :func:`_to_rate_frames`); the last frame of a shot holds."""
    out = rate.copy()
    for shot in shots:
        out[shot.f0:shot.f1 - 1] = rate[shot.f0 + 1:shot.f1]
    return out


def _mirror_easing(name: str) -> str:
    """Easing of a time-reversed curve."""
    return {"ease_in": "ease_out", "ease_out": "ease_in"}.get(name, name)


def _ramp_event(kind: SpeedKind, ramp: _Ramp, offset: int) -> _SpeedEvent:
    ratio = max(ramp.rate_from, ramp.rate_to) / max(min(ramp.rate_from, ramp.rate_to), 1e-3)
    conf = 0.45 + 0.2 * float(np.clip((np.log2(ratio) - 1.0) / 2.0, 0.0, 1.0)) - 0.05 * ramp.long
    return _SpeedEvent(kind, offset + ramp.start, offset + ramp.end, ramp.rate_from, ramp.rate_to,
                       easing=ramp.easing, confidence=round(conf, 3), params={"method": "motion"})


def _add_fast_events(
    events: dict[int, list[_SpeedEvent]],
    refs: dict[int, tuple[float, int]],
    rate: np.ndarray,
    shots: list[ShotSpan],
    prm: _Params,
) -> None:
    """``fast`` for the normal-speed part of shots moving far above the video's typical motion."""
    if len(refs) < int(prm("fast_min_shots", 3)):
        return
    levels = np.array([v[0] for v in refs.values()])
    weights = np.array([v[1] for v in refs.values()], float)
    order = np.argsort(levels)
    cum = np.cumsum(weights[order])
    typical = float(levels[order][np.searchsorted(cum, 0.5 * cum[-1])])
    ratio_min = float(prm("fast_ratio", 3.0))
    min_len = prm.frames("fast_min_duration", 0.5, minimum=6)
    for shot in shots:
        if shot.index not in refs:
            continue
        k = refs[shot.index][0] / typical
        if k < ratio_min:
            continue
        r = rate[shot.f0:shot.f1]
        taken = np.zeros(r.size, bool)
        for ev in events.get(shot.index, []):
            taken[ev.f0 - shot.f0:ev.f1 - shot.f0] = True
        normal = (r >= 0.7) & ~taken
        normal[0] = False
        for a, b in _runs(normal):
            if b - a < min_len:
                continue
            level = float(k * np.median(r[a:b]))
            conf = 0.4 + 0.15 * float(np.clip((k - ratio_min) / ratio_min, 0.0, 1.0))
            events.setdefault(shot.index, []).append(
                _SpeedEvent("fast", shot.f0 + a, shot.f0 + b, level, level, confidence=round(conf, 3),
                            params={"method": "motion", "relative_to": "video"}))


def _merge_duplicate_slowmo(
    events: dict[int, list[_SpeedEvent]],
    dup_regions: list[tuple[int, int, int, float]],
    novel: np.ndarray,
    q_base: float,
    prm: _Params,
) -> None:
    """Use repeated-frame cadence as direct evidence for slow motion.

    A motion-based ``slowmo`` lying inside a repetition region gets its rate from
    the cadence (mean novel-to-novel gap); regions no motion event explains (e.g.
    shots too static to judge by magnitude) become ``slowmo`` nodes of their own.
    """
    max_rate = float(prm("dup_max_rate", 0.75))
    for shot_index, a, b, q in dup_regions:
        evs = events.setdefault(shot_index, [])
        inside = [ev for ev in evs if ev.kind == "slowmo"
                  and min(ev.f1, b) - max(ev.f0, a) >= 0.8 * (ev.f1 - ev.f0)]
        for ev in inside:
            pos = np.flatnonzero(novel[ev.f0:ev.f1])
            if pos.size < 3:
                continue
            rate = 1.0 / float(np.diff(pos).mean()) / q_base
            if rate <= max_rate:
                ev.rate_from = ev.rate_to = rate
                ev.confidence = max(ev.confidence, 0.8)
                ev.params = {"method": "frame_repeat", "repeat": round(1.0 / rate, 2)}
        rate = q / q_base
        if rate <= max_rate and not any(min(ev.f1, b) - max(ev.f0, a) > 0 for ev in evs):
            evs.append(_SpeedEvent("slowmo", a, b, rate, rate, confidence=0.75,
                                   params={"method": "frame_repeat", "repeat": round(1.0 / rate, 2)}))


def _to_speed(ctx: AnalysisContext, ev: _SpeedEvent, rate: np.ndarray, step: int) -> Speed:
    frames = list(range(ev.f0, ev.f1, step))
    if frames[-1] != ev.f1 - 1:
        frames.append(ev.f1 - 1)
    keys = [Keyframe(t=ctx.t(f), v=round(float(rate[f]), 4)) for f in frames]
    if ev.params.get("method") == "frame_repeat" or ev.kind == "fast":
        keys = [Keyframe(t=k.t, v=round(ev.rate_from, 4)) for k in keys]
    return Speed(
        start=ctx.t(ev.f0), end=ctx.t(ev.f1), kind=ev.kind,
        rate_from=round(ev.rate_from, 4), rate_to=round(ev.rate_to, 4), easing=ev.easing,
        keyframes=keys, confidence=ev.confidence, params=ev.params,
    )
