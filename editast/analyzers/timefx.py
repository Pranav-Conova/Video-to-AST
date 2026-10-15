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
   difference, the largest difference of any ~16 px block (a moving subject in
   an otherwise static shot keeps one block busy) and its flow magnitude are
   all at the level of compression noise.
3. **Duplicate-frame slow motion.** Editors without frame interpolation repeat
   source frames: 0.5x shows every frame twice (novel, still, novel, still ...).
   Regular alternations of short still runs and short novel runs over at least
   three cycles form a segment whose rate is the inverse mean novel-to-novel
   gap, divided by the video's baseline cadence, so 30 fps footage in a 60 fps
   container (which repeats every frame everywhere) is not slow motion.
4. **Freezes.** Longer still runs (at least ``max(3 frames, 0.12 s)`` of
   identical displayed frames) inside a shot whose other frames clearly move
   (footage speed, so a flash or a digital zoom does not count as motion). A
   naturally static (tripod) shot never yields a freeze.
5. **Speed curve.** Each novel frame's motion is spread evenly over the repeats
   that follow it (duplicate-frame slow motion then reads as a smooth low rate,
   exactly like interpolated slow motion); freeze frames are dropped. The
   centre translation is projected on its local travel direction and median
   filtered over at least one jitter period (wider inside Shake spans reported
   by the motion analysis): a median removes oscillating shake but keeps hard
   speed cuts sharp. Speed = projected translation + residual. The shot's
   "normal speed" reference is the typical level of its top decile and the
   relative rate is ``r(t) = speed(t) / reference``.
6. **Speed events.** Sustained ``r < slow_rate`` runs become ``slowmo``. The
   transitions into and out of them are fitted with a plateau-ramp-plateau
   model (ramp start, end and easing by exhaustive least squares in log-rate);
   changes of at least 2x within ``ramp_max`` seconds become ``ramp_down`` /
   ``ramp_up`` while quicker ones are hard speed cuts that only delimit the slow
   span. A shot whose normal-speed motion is far above the video's typical
   motion gets a low-confidence ``fast`` node.

Rates are relative to the shot's own normal speed (motion only reveals relative
speed), so confidences stay in the 0.4-0.8 range except where frame repetition
gives direct evidence. ``Speed`` keyframes sample ``r(t)`` every ~0.1 s; a ramp's
``rate_from``/``rate_to``/``easing`` follow :func:`editast.fx.interp` semantics
(rate interpolated linearly in rate space along the easing curve), and a
``Freeze`` over frames ``[a, b)`` means ``b - a`` identical frames, as rendered
by :func:`editast.render.compiler.source_positions`.
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
    """Per-frame global motion (index ``t`` = motion from frame ``t-1`` to ``t``; 0 at shot starts)."""

    trans: np.ndarray  # (n, 2) frame-centre translation, fraction of width, zoom-normalised
    resid: np.ndarray  # (n,) mean non-rigid flow, fraction of width, zoom-normalised
    total: np.ndarray  # (n,) mean flow magnitude, fraction of width
    log_zoom: np.ndarray  # (n,) inter-frame log zoom of the similarity fit

    @property
    def speed(self) -> np.ndarray:
        """Unsmoothed footage speed per frame."""
        return np.hypot(self.trans[:, 0], self.trans[:, 1]) + self.resid


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
    start: int  # first frame of the ramp (index into the analysed curve)
    end: int  # first frame at the target level
    rate_from: float
    rate_to: float
    easing: str


# --------------------------------------------------------------------------- public API


def analyze_time_effects(
    ctx: AnalysisContext,
    seg: ShotSegmentation,
    motion: Optional[MotionAnalysis] = None,
) -> EffectsResult:
    """Detect ``Speed`` (slowmo / fast / ramp_down / ramp_up) and ``Freeze`` effects per shot.

    ``motion`` is optional; its Shake effects widen the jitter filter of the
    speed curve inside their spans. Signals: ``speed_rate`` (relative playback
    rate, 1 where not judged), ``motion_magnitude`` (smoothed footage speed,
    fraction of width per frame) and ``frame_repeat`` (1 where a frame repeats
    its predecessor).
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
    speed = flow.speed
    diff = np.asarray(ctx.features.diff, float)
    block = _block_change(ctx.gray, starts, prm)
    still = (
        usable
        & (diff < float(prm("still_diff_max", 0.003)))
        & (block < float(prm("still_block_max", 0.025)))
        & (flow.total < float(prm("still_flow_max", 0.002)))
    )

    freeze = np.zeros(n, bool)
    jump = np.zeros(n, bool)
    dup_segments: list[tuple[int, int, int, float]] = []  # (shot index, f0, f1, novel fraction)
    for shot in shots:
        dups, freezes = _repeat_structure(shot, still, usable, diff, speed, prm)
        dup_segments += [(shot.index, a, b, q) for a, b, q in dups]
        for a, b in freezes:  # [a, b): the held frame a and its repeats
            freeze[a + 1:b] = True
            hold_diff = float(np.median(diff[a + 1:b]))
            conf = 0.9 if hold_diff < 0.5 * float(prm("freeze_diff_max", 0.003)) else 0.7
            result.add(shot.index, Freeze(start=ctx.t(a), end=ctx.t(b), confidence=conf,
                                          params={"frames": b - a}))
            if b < shot.f1:
                jump[b] = True  # motion into the first frame after a freeze may span the held time

    # Frames whose flow says nothing about playback speed: flashes / inversions, and
    # fast digital zoom punches (dense flow is unreliable at several % zoom per frame).
    unjudged = _photometric_outliers(ctx, shots, prm)
    unjudged[1:] |= unjudged[:-1]  # the flow into frame t also depends on frame t-1
    unjudged |= np.abs(flow.log_zoom) > float(prm("zoom_max", 0.03))
    filled, valid = _fill_repeats(np.column_stack([flow.trans, flow.resid]), still,
                                  usable & ~freeze & ~jump & ~unjudged, shots, n, prm)
    window = _median_windows(ctx, motion, shots, n, prm)
    rate = np.ones(n)
    smooth = np.zeros(n)
    refs: dict[int, tuple[float, int]] = {}
    events: dict[int, list[_SpeedEvent]] = {}
    min_motion = float(prm("min_motion", 0.003))
    min_seg = prm.frames("min_segment", 0.3, minimum=6)
    for shot in shots:
        fine, sm, ok = _speed_curve(filled, valid, window, shot, prm)
        smooth[shot.f0:shot.f1] = np.where(ok, sm, 0.0)
        if ok.sum() < min_seg:
            continue
        ref = _reference_level(sm[ok], float(prm("ref_percentile", 90.0)))
        if ref < min_motion:
            continue  # too little motion to judge speed
        refs[shot.index] = (ref, int(ok.sum()))
        r = np.where(ok, sm / ref, 1.0)
        r_fine = np.where(ok, fine / ref, 1.0)
        rate[shot.f0:shot.f1] = r
        events[shot.index] = []
        for a, b in _runs(ok):
            if b - a >= min_seg:
                events[shot.index] += _speed_events(r[a:b], r_fine[a:b], shot.f0 + a, prm)

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
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
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


def _repeat_structure(
    shot: ShotSpan,
    still: np.ndarray,
    usable: np.ndarray,
    diff: np.ndarray,
    speed: np.ndarray,
    prm: _Params,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int]]]:
    """Duplicate-frame segments ``(f0, f1, novel_fraction)`` and freezes ``(held, end)``.

    Works on the shot interior (frames after the first). Absolute frame indices.
    """
    base = shot.f0 + 1
    st, ok = still[base:shot.f1], usable[base:shot.f1]
    if st.size < 3:
        return [], []
    max_dup = int(prm("max_dup_run", 4))
    max_novel = int(prm("max_novel_run", 2))
    state = np.where(~ok, 2, np.where(st, 1, 0))  # 0 novel, 1 still, 2 unusable
    change = np.flatnonzero(np.diff(state)) + 1
    bounds = np.concatenate(([0], change, [state.size]))
    runs = [(int(state[s]), int(s), int(e)) for s, e in zip(bounds[:-1], bounds[1:])]

    # Duplicate cadence: short still runs separated by short novel runs.
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
    dups: list[tuple[int, int, float]] = []
    min_len = prm.frames("dup_min_duration", 0.2, minimum=4)
    for s, e, cycles in merged:
        if cycles < int(prm("dup_min_cycles", 3)) or e - s < min_len:
            continue
        seg_still = st[s:e] & ok[s:e]
        seg_novel = ~st[s:e] & ok[s:e]
        if not seg_novel.any():
            continue
        d = diff[base + s:base + e]
        if np.median(d[seg_novel]) < 2.5 * max(float(np.median(d[seg_still])), 1e-4):
            continue  # repeats are not clearly separated from novel frames
        # Rate from complete cycles (novel-to-novel gaps); the last cycle may be truncated.
        pos = np.flatnonzero(seg_novel)
        q = 1.0 / float(np.diff(pos).mean()) if pos.size >= 3 else float(seg_novel.sum()) / float(ok[s:e].sum())
        dups.append((base + s, base + e, q))

    # Freezes: long still runs, or short ones outside a duplicate cadence.
    min_hold = max(int(prm("freeze_min_frames", 3)), prm.frames("freeze_min_duration", 0.12))
    min_motion = float(prm("freeze_min_motion", 0.002))
    novel = ok & ~st
    freezes: list[tuple[int, int]] = []
    for kind, s, e in runs:
        if kind != 1 or e - s + 1 < min_hold:
            continue
        if e - s <= max_dup and any(a <= base + s and base + e <= b for a, b, _ in dups):
            continue
        moving = novel.copy()
        moving[s:e] = False
        if moving.sum() < max(3, 0.2 * (st.size - (e - s))):
            continue
        d_move = float(np.median(diff[base:shot.f1][moving]))
        v_move = float(np.median(speed[base:shot.f1][moving]))
        d_hold = float(np.median(diff[base + s:base + e]))
        limit = min(float(prm("freeze_diff_max", 0.003)), float(prm("freeze_rel", 0.3)) * d_move)
        if d_hold <= limit and v_move >= min_motion:
            freezes.append((base + s - 1, base + e))
    return dups, freezes


def _fill_repeats(
    values: np.ndarray,
    still: np.ndarray,
    usable: np.ndarray,
    shots: list[ShotSpan],
    n: int,
    prm: _Params,
) -> tuple[np.ndarray, np.ndarray]:
    """Spread each novel frame's motion ``values`` (n, k) over the repeats that follow it.

    Returns ``(filled, valid)``; long still runs keep their (zero) motion.
    """
    max_dup = int(prm("max_dup_run", 4))
    filled = values.astype(float).copy()
    valid = usable.copy()
    for shot in shots:
        a, b = shot.f0 + 1, shot.f1
        if b - a < 1:
            continue
        st = still[a:b]
        group = np.cumsum(~st)  # group 0 = repeats of the shot's first frame (no motion)
        sizes = np.bincount(group)
        first = np.zeros((sizes.size, values.shape[1]))
        novel_idx = np.flatnonzero(~st)
        first[group[novel_idx]] = values[a:b][novel_idx]
        size = sizes[group]
        short = (size - 1 <= max_dup) & (group > 0)
        filled[a:b] = np.where(short[:, None], first[group] / size[:, None], filled[a:b])
        valid[a:b] &= group > 0
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Footage speed inside each valid run of a shot.

    ``filled`` holds ``(tx, ty, residual)`` per frame. The translation is median
    filtered per component (the median of an oscillation about a level is the
    level, and a step stays a step); frames whose remaining jitter is
    comparable to the speed itself for at least ``jitter_min_duration`` are
    unjudgeable and leave ``ok``. Returns ``(fine, smooth, ok)``: ``fine`` is
    the median-filtered speed (keeps hard speed cuts sharp for the ramp fit),
    ``smooth`` is additionally Gaussian smoothed (stable levels).
    """
    vals = filled[shot.f0:shot.f1]
    ok = valid[shot.f0:shot.f1].copy()
    win = window[shot.f0:shot.f1]
    fine = np.zeros(ok.size)
    smooth = np.zeros(ok.size)
    sigma_jit = max(1.0, float(prm("jitter_sigma", 0.1)) * prm.fps)
    sigma = max(1.0, float(prm("smooth_sigma", 0.035)) * prm.fps)
    jitter_ratio = float(prm("jitter_ratio", 0.8))
    jitter_min = prm.frames("jitter_min_duration", 0.2, minimum=3)
    shaky = np.zeros(ok.size, bool)
    for a, b in _runs(ok):
        trans = vals[a:b, :2]
        med = np.zeros((b - a, 3))
        for size in np.unique(win[a:b]):
            sel = win[a:b] == size
            med[sel] = _running_median(vals[a:b], int(size))[sel]
        x = np.hypot(med[:, 0], med[:, 1]) + med[:, 2]
        jitter = np.sqrt(ndimage.gaussian_filter1d(((trans - med[:, :2]) ** 2).sum(axis=1), sigma_jit,
                                                   mode="nearest"))
        unsure = jitter > jitter_ratio * np.maximum(x, 1e-6)
        for s, e in _runs(unsure):
            if e - s >= jitter_min:
                shaky[a + s:a + e] = True
        fine[a:b] = x
    ok &= ~shaky
    for a, b in _runs(ok):
        smooth[a:b] = ndimage.gaussian_filter1d(fine[a:b], sigma, mode="nearest") if b - a >= 3 else fine[a:b]
    return fine, smooth, ok


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
    ratio_min = float(prm("ramp_ratio", 1.8))
    ramp_max = prm.frames("ramp_max", 0.6, minimum=3)
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
        if a > 0:
            ramp = _fit_ramp(r, fine, a, mid, lo, ramp_max, prev_end)
            if ramp is not None and ramp.rate_from >= ratio_min * ramp.rate_to:
                if ramp.end - ramp.start >= ramp_min:
                    s_slow = ramp.end
                    pieces.append(_ramp_event("ramp_down", ramp, offset))
                else:  # hard speed cut: the slow span starts mid-transition
                    s_slow = (ramp.start + ramp.end + 1) // 2
        if b < n:
            # Ramp up = ramp down of the time-reversed curve; reversed index x is frame n-1-x.
            ramp = _fit_ramp(r[::-1], fine[::-1], n - b, n - 1 - mid, lo, ramp_max, n - next_start)
            if ramp is not None and ramp.rate_from >= ratio_min * ramp.rate_to:
                start, end = n - 1 - ramp.end, n - 1 - ramp.start
                if end - start >= ramp_min:
                    e_slow = start
                    up = _Ramp(start, end, ramp.rate_to, ramp.rate_from, _mirror_easing(ramp.easing))
                    pieces.append(_ramp_event("ramp_up", up, offset))
                else:
                    e_slow = (start + end + 1) // 2
        if e_slow - s_slow >= slow_min:
            level = float(np.median(r[s_slow:e_slow]))
            conf = 0.4 + 0.2 * float(np.clip((slow_rate - level) / slow_rate, 0.0, 1.0)) + 0.1 * bool(pieces)
            pieces.append(_SpeedEvent("slowmo", offset + s_slow, offset + e_slow, level, level,
                                      confidence=round(conf, 3), params={"method": "motion"}))
        events += pieces
    return events


def _fit_ramp(
    r: np.ndarray, fine: np.ndarray, a: int, mid: int, lo: float, ramp_max: int, bound: int = 0
) -> Optional[_Ramp]:
    """Fit a high-plateau -> ramp -> low-plateau model ending in ``[a, mid]``.

    ``r[a]`` is the first frame below the slow threshold and ``mid`` the bottom
    of the low run; the high plateau is searched in ``[max(bound, a - ramp_max - 2), a)``
    (``bound`` = end of the previous slow run). Plateau levels come from the
    smoothed ``r``, the shape is fitted to ``fine``. Exhaustive search over ramp
    start/end and easing, error in log-rate. Returns ``None`` when there is no
    higher level to ramp from.
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
    best: Optional[tuple[float, int, int, str]] = None
    for i in range(peak, w1 - 1):
        ends = np.arange(i + 1, min(w1 - 1, i + int(1.5 * ramp_max)) + 1)
        x = np.clip((k[None, :] - i) / (ends[:, None] - i), 0.0, 1.0)  # (n_ends, n_frames)
        for name in _RAMP_EASINGS:
            model = np.log(np.maximum(fx.interp(hi, lo, x, name), 1e-3))
            sse = ((model - y[None, :]) ** 2).sum(axis=1)
            j = int(np.argmin(sse))
            if best is None or sse[j] < best[0] - 1e-12:
                best = (float(sse[j]), int(i), int(ends[j]), name)
    if best is None:
        return None
    _, i, j, name = best
    if j - i > ramp_max:
        return None  # gradual change (e.g. natural deceleration), not an edit ramp
    seg = np.diff(fine[i:j + 1])
    if seg.size > 2 and (seg <= 0.05 * hi).mean() < 0.75:
        return None  # not monotonic
    return _Ramp(i, j, hi, lo, name)


def _mirror_easing(name: str) -> str:
    """Easing of a time-reversed curve."""
    return {"ease_in": "ease_out", "ease_out": "ease_in"}.get(name, name)


def _ramp_event(kind: SpeedKind, ramp: _Ramp, offset: int) -> _SpeedEvent:
    ratio = max(ramp.rate_from, ramp.rate_to) / max(min(ramp.rate_from, ramp.rate_to), 1e-3)
    conf = 0.45 + 0.2 * float(np.clip((np.log2(ratio) - 1.0) / 2.0, 0.0, 1.0))
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
            conf = 0.35 + 0.15 * float(np.clip((k - ratio_min) / ratio_min, 0.0, 1.0))
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
    keys = [Keyframe(t=round(ctx.t(f), 4), v=round(float(rate[f]), 4)) for f in frames]
    if ev.params.get("method") == "frame_repeat" or ev.kind == "fast":
        keys = [Keyframe(t=k.t, v=round(ev.rate_from, 4)) for k in keys]
    return Speed(
        start=ctx.t(ev.f0), end=ctx.t(ev.f1), kind=ev.kind,
        rate_from=round(ev.rate_from, 4), rate_to=round(ev.rate_to, 4), easing=ev.easing,
        keyframes=keys, confidence=ev.confidence, params=ev.params,
    )
