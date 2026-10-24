"""Camera / digital motion analysis: zoom, pan, rotate and shake.

Algorithm
=========

1. **Inter-frame similarity** ``t-1 -> t``, only inside shots (``t == shot.f0`` and frames
   of non-cut transitions are identity / ``valid=False``). Track points are the strongest
   minimum-eigenvalue pixel (``cv2.cornerMinEigenVal``) of every ``cell x cell`` grid cell
   of the content box (flat letterbox / pillarbox bars and a thin border are excluded).
   Unlike a global corner-quality threshold, the grid keeps points spread over the frame,
   so a burned-in caption or one high-contrast object cannot dominate the fit, and it
   still finds points on smooth, low-corner footage. Points are tracked with pyramidal
   Lucas-Kanade plus a forward-backward check, and a 4-dof similarity is fitted with RANSAC
   (``cv2.estimateAffinePartial2D``). It is decomposed exactly like
   :func:`editast.fx.similarity_matrix`: ``scale``, ``angle = atan2(M01, M00)`` (degrees,
   CCW positive) and the displacement of the frame centre ``(w/2, h/2)`` normalised by
   width / height. A failed pair is retried on contrast-normalised frames when the two
   frames differ in brightness / contrast (flash, fade, exposure) and with the second
   frame inverted when they are anti-correlated (invert effect, glitch). Pairs that still
   fail (a blank frame) are bridged by measuring the frames on either side of the gap and
   splitting that motion into equal steps (used for the curves, ``valid=False``). Without
   any fit the translation comes from a verified ``cv2.phaseCorrelate`` (scale 1,
   ``valid=False``). ``magnitude`` is the mean length of all forward-backward consistent
   tracks, so it stays meaningful under parallax or independent motion.
2. **Integration** per shot: log-scale, angle and centre displacement increments are
   summed into cumulative curves. Short unmeasured stretches (dissolve frames, failed
   tracking) borrow interpolated increments from their neighbours; long ones count as
   no motion. Effects need most of their frames measured (``min_coverage``).
3. **Zoom**: a zig-zag (hysteresis) filter splits the lightly smoothed log-scale curve
   into monotone legs; legs changing by more than ``zoom_threshold`` become
   :class:`~editast.ast.nodes.Zoom` nodes. A leg is tightened to its fast stretches; a
   move that halts for at most ``pause_seconds`` (e.g. under a freeze frame) and resumes
   in the same direction, or wobbles back slightly in between, stays one move. Start, end and easing come from a
   least-squares fit with free endpoints of every easing in :data:`FIT_EASINGS` to the
   cumulative *scale*, because the renderer interpolates scale linearly through the
   easing (:func:`editast.fx.interp`). The centre is the fixed point of the inter-frame
   transforms: the centre displacement is regressed on ``r_t - 1`` against a pan whose
   velocity varies smoothly over the zoom plus a margin (frames of neighbouring pans
   and zooms excluded, so pans are found once with centred zooms first). The
   zoom-induced centre displacement is then removed from the translation curve, so an
   off-centre zoom does not read as a pan. ``digital`` is True for a rigid, cleanly
   eased scale (low residual, high RANSAC inlier fraction) and False under parallax.
4. **Pan / rotate**: the same leg detection on a Savitzky-Golay trend
   (``smooth_seconds``) of the translation (x and y legs are merged into one pan) and
   angle curves, then the same easing fit on the unsmoothed curves.
5. **Shake**: the high-frequency residual of translation and angle is scanned with a
   windowed RMS. The residual is the curve minus the fitted pan / rotate models and
   minus a Savitzky-Golay trend, so fast eased moves do not leak into it. Runs above
   ``shake_threshold`` (per-axis peak, estimated as ``sqrt(2) * RMS`` as for sinusoidal
   jitter) that last ``shake_min_duration``, swing back and forth and are *periodic*
   (one sinusoid explains ``shake_periodicity`` of the energy of their frame-to-frame
   velocity, over at least ``shake_min_cycles`` cycles, which rejects the residue of
   velocity kinks such as a pan halting under a freeze frame) become
   :class:`~editast.ast.nodes.Shake` nodes
   (snapped to a shot edge within ``shake_edge_seconds``). Frequency and per-axis
   amplitude come from a least-squares sinusoid fit (:func:`_fit_jitter`) to the jitter
   before trend removal, with the axes tied by the renderer's frequency ratios
   (:data:`_SHAKE_AXIS_RATIOS`), so they mean the same as ``Shake.frequency`` /
   ``amplitude`` when rendered; unlike a periodogram this stays accurate over the one
   to three cycles of a short shake.
6. **Confidence**: every effect's ``confidence`` combines the evidence for a move of the
   whole picture (``_ShotAnalyzer._quality``): measured share of its frames, rigidity
   (RANSAC inlier share: parallax lowers it, a subject moving on its own lowers it a
   lot), fit residual, consistency of the eased model (``r2``), size relative to the
   detection threshold and number of measured frames. Effects below
   ``min_confidence`` are not reported (they still shape the curves).
7. **Summary**: one :class:`~editast.ast.nodes.MotionSummary` per shot (cumulative zoom,
   pan and rotation, mean flow magnitude and a coarse camera class).

Units follow ``editast.ast.nodes``: translations are fractions of the width (x) or
height (y); shake amplitudes are fractions of the width on *both* axes; angles are
degrees, CCW positive. Zooms within a shot are relative (``scale_from=1``), compose
multiplicatively and hold after they end; ``params["scale_start"]`` is the cumulative
scale (relative to the shot start) at which a zoom begins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import savgol_filter

from editast import fx
from editast.analyzers.types import (
    EffectsResult,
    FrameMotion,
    MotionAnalysis,
    ShotSegmentation,
    ShotSpan,
)
from editast.ast.nodes import CameraMove, EffectBase, Keyframe, MotionSummary, Pan, Rotate, Shake, Zoom
from editast.context import AnalysisContext

NAME = "motion"

#: Easings considered when fitting zoom / pan / rotate curves ("step" is a 1-frame move).
FIT_EASINGS: tuple[str, ...] = ("linear", "ease_in", "ease_out", "ease_in_out", "punch")

_PUNCH_SNAP = 0.2  # fx "punch" reaches its target after 20% of the span
_FAST_SNAP = 0.2  # seconds; a punch snapping faster than this is identified by its hold:
_PUNCH_TOLERANCE = 1.5  # it may have this multiple of the best smooth easing's SSE ...
_SHAPE_TOLERANCE = 0.015  # ... plus RMS deviations of this fraction of the move per frame
_PUNCH_MARGIN = 0.6  # a slower punch must reach this fraction of the best smooth easing's SSE
_N_KEYFRAMES = 8
#: Jitter frequency of the x, y and rotation axes relative to ``Shake.frequency`` as the
#: renderer produces it (y from :func:`editast.fx.shake_offset`, rotation from
#: ``editast.render.compiler``); the spectra are aligned on this before peak picking.
_SHAKE_AXIS_RATIOS: tuple[float, float, float] = (1.0, 1.13, 0.9)
_FLAT_STD = 3.0  # grey levels; frames with less contrast are blank (nothing to track)
#: Transitions that only change intensity; the motion under them stays measurable.
_PHOTOMETRIC: tuple[str, ...] = ("flash", "fade_black", "fade_white")


@dataclass(frozen=True)
class _Params:
    """Tunables, overridable through ``AnalysisConfig.params["motion"]``.

    Pixel quantities refer to the analysis resolution (default long side 320 px).
    """

    # --- tracking
    cell: int = 14  # px, grid cell for point selection
    min_eigen: float = 1e-4  # cornerMinEigenVal(blockSize=7, ksize=3) floor of a trackable point
    min_points: int = 40  # below this many cells pass, the floor is relaxed 5x
    max_points: int = 400
    lk_window: int = 15  # px
    lk_levels: int = 3
    fb_threshold: float = 0.7  # px, forward-backward consistency
    ransac_threshold: float = 1.0  # px
    fb_relaxed: float = 2.0  # px, second attempt when the strict tracks give no fit
    ransac_relaxed: float = 2.0  # px
    min_spread: float = 0.06  # inlier coordinate std per axis, fraction of the tracking area
    min_inliers: int = 12
    min_inlier_ratio: float = 0.3
    min_phase_response: float = 0.05  # phaseCorrelate peak needed for the fallback translation
    min_shift_correlation: float = 0.5  # ZNCC of the frames aligned by the fallback translation
    bridge_seconds: float = 0.4  # longest gap of failed pairs measured across
    fill_seconds: float = 0.5  # longer unmeasured stretches are taken as "no motion"
    min_coverage: float = 0.5  # fraction of an effect's frames that must be measured
    frozen_flow: float = 0.12  # px; slower tracks are static overlays when the rest moves 3x faster
    border: int = 4  # px ignored at the frame / bar edges
    bar_range: float = 12.0  # grey levels: flatter rows / columns at the edges are bars
    # --- curves
    smooth_seconds: float = 0.4  # Savitzky-Golay window of the pan / rotate / shake trend
    zoom_threshold: float = 0.04  # relative scale change
    pan_threshold: float = 0.05  # normalised displacement
    rotate_threshold: float = 3.0  # degrees
    pause_seconds: float = 0.5  # a move halting at most this long and resuming in the same direction is one move
    center_snap: float = 0.05  # zoom centres this close to 0.5 (per axis) snap to 0.5
    center_margin: float = 0.25  # seconds around a zoom that also constrain the pan velocity
    center_min_shift: float = 0.01  # total centre shift (fraction of width) an off-centre zoom must explain
    center_min_seconds: float = 0.3  # measured frames needed to place a zoom off-centre
    digital_residual: float = 0.004  # mean fit residual (width) below which a zoom can be digital
    digital_inliers: float = 0.8  # ... with at least this RANSAC inlier fraction ...
    digital_r2: float = 0.98  # ... and a clean eased scale curve (R^2 of the easing fit)
    parallax_residual: float = 0.01  # above this residual the motion is not one rigid layer
    parallax_inliers: float = 0.6  # below this inlier fraction: parallax / independent motion
    min_move_seconds: float = 0.15  # measured frames for full confidence in a zoom / pan / rotate
    min_confidence: float = 0.3  # effects with less evidence (see ``_ShotAnalyzer._quality``) are dropped
    # --- shake
    shake_threshold: float = 0.004  # per-axis peak offset, fraction of width
    shake_rotation_threshold: float = 0.5  # degrees
    shake_window: float = 0.3  # seconds, windowed RMS
    shake_min_duration: float = 0.25  # seconds
    shake_periodicity: float = 0.9  # share of the jitter velocity's energy one sinusoid must explain
    shake_min_cycles: float = 1.4  # jitter cycles the shake must span
    shake_edge_seconds: float = 0.1  # shakes this close to a shot edge are extended to it

    @classmethod
    def from_context(cls, ctx: AnalysisContext) -> "_Params":
        return cls(**{f.name: type(f.default)(ctx.config.p(NAME, f.name, f.default)) for f in fields(cls)})


# =========================================================================== public API


def analyze_motion(ctx: AnalysisContext, seg: ShotSegmentation) -> MotionAnalysis:
    """Per-frame camera motion plus Zoom / Pan / Rotate / Shake effects and per-shot summaries."""
    prm = _Params.from_context(ctx)
    n = ctx.n_frames
    shots = _shot_spans(seg, n)
    frame_motion, estimates = _estimate_frame_motion(ctx, shots, seg, prm)

    effects = EffectsResult()
    summaries: dict[int, MotionSummary] = {}
    sig = {
        "motion_magnitude": frame_motion.magnitude.copy(),
        "zoom": np.ones(n),
        "pan_x": np.zeros(n),
        "pan_y": np.zeros(n),
        "rotation": np.zeros(n),
        "shake": np.zeros(n),
    }
    for shot in shots:
        result = _ShotAnalyzer(ctx, shot, frame_motion, estimates, prm).run()
        for eff in result.effects:
            effects.add(shot.index, eff)
        summaries[shot.index] = result.summary
        sl = slice(shot.f0, shot.f1)
        sig["zoom"][sl] = np.exp(result.curves.log_scale)
        sig["pan_x"][sl] = result.curves.pan_x
        sig["pan_y"][sl] = result.curves.pan_y
        sig["rotation"][sl] = result.curves.angle
        sig["shake"][sl] = result.shake_level
    effects.signals.update(sig)
    return MotionAnalysis(frame_motion=frame_motion, effects=effects, summaries=summaries)


# =========================================================================== per-frame motion


@dataclass
class _Pair:
    """Similarity estimate between two consecutive frames (pixel units)."""

    scale: float = 1.0
    angle: float = 0.0
    dx: float = 0.0  # displacement of the frame centre
    dy: float = 0.0
    residual: float = 0.0  # mean inlier reprojection error
    inliers: float = 0.0  # fraction of the consistent tracks that fit the similarity
    magnitude: float = 0.0  # mean track length
    flow: bool = False  # whether ``magnitude`` was measured
    quality: int = 0  # 2 = similarity fit, 1 = translation only, 0 = nothing


class _PairEstimator:
    """Grid-sampled LK tracking + RANSAC similarity, with a phase-correlation fallback."""

    def __init__(self, width: int, height: int, prm: _Params) -> None:
        self.w, self.h = width, height
        self.prm = prm
        self.center = np.array([width / 2.0, height / 2.0])
        self.lk = dict(
            winSize=(prm.lk_window, prm.lk_window), maxLevel=prm.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def select(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """Strongest corner-ness pixel per grid cell, as ``(n, 1, 2)`` float32 points."""
        x0, y0, x1, y1 = box
        c = self.prm.cell
        rows, cols = (y1 - y0) // c, (x1 - x0) // c
        if rows < 1 or cols < 1:
            return np.empty((0, 1, 2), np.float32)
        # Centre the grid in the box; cornerMinEigenVal needs a little context around it.
        oy, ox = y0 + ((y1 - y0) - rows * c) // 2, x0 + ((x1 - x0) - cols * c) // 2
        eig = cv2.cornerMinEigenVal(gray[oy:oy + rows * c, ox:ox + cols * c], 7, 3)
        blocks = eig.reshape(rows, c, cols, c).transpose(0, 2, 1, 3).reshape(rows * cols, c * c)
        arg = blocks.argmax(axis=1)
        val = blocks[np.arange(blocks.shape[0]), arg]
        cell_r, cell_c = np.divmod(np.arange(rows * cols), cols)
        dy, dx = np.divmod(arg, c)
        pts = np.stack([ox + cell_c * c + dx, oy + cell_r * c + dy], axis=1).astype(np.float32)
        keep = val >= self.prm.min_eigen
        if keep.sum() < self.prm.min_points:  # low-contrast footage: relax the floor
            keep = val >= self.prm.min_eigen / 5.0
        pts, val = pts[keep], val[keep]
        if len(pts) > self.prm.max_points:
            pts = pts[np.argsort(val)[::-1][: self.prm.max_points]]
        return pts.reshape(-1, 1, 2)

    def __call__(self, g0: np.ndarray, g1: np.ndarray, box: tuple[int, int, int, int]) -> _Pair:
        prm = self.prm
        pair = _Pair()
        p0 = self.select(g0, box)
        if len(p0) >= 3:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **self.lk)
            back, st_b, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **self.lk)
            fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
            a, b = p0.reshape(-1, 2), p1.reshape(-1, 2)
            ok = (st.ravel() == 1) & (st_b.ravel() == 1)
            ok &= (b[:, 0] >= 0) & (b[:, 0] < self.w) & (b[:, 1] >= 0) & (b[:, 1] < self.h)
            # Strict tracks first; noisy, blocky or re-lit (flash / fade) frames get a second
            # try with looser forward-backward and RANSAC tolerances.
            for fb_max, ransac in ((prm.fb_threshold, prm.ransac_threshold), (prm.fb_relaxed, prm.ransac_relaxed)):
                aa, bb = self._drop_frozen(a[ok & (fb < fb_max)], b[ok & (fb < fb_max)])
                if len(aa) < prm.min_inliers:
                    continue
                if not pair.flow:  # enough consistent tracks for a meaningful mean flow
                    pair.magnitude, pair.flow = float(np.linalg.norm(bb - aa, axis=1).mean()), True
                if self._fit(aa, bb, pair, ransac, box):
                    return pair
        (sx, sy), response = self._phase(g0, g1, box)
        if response >= prm.min_phase_response and self._verify_shift(g0, g1, box, sx, sy):
            pair.dx, pair.dy, pair.quality = float(sx), float(sy), 1
            if not pair.flow:
                pair.magnitude, pair.flow = math.hypot(sx, sy), True
        return pair

    def _drop_frozen(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove tracks on burned-in overlays (captions, logos, stickers).

        When the camera moves slowly (per-frame motion below the RANSAC threshold), points
        on static overlays fit the same model as the content and bias the estimate toward
        zero. A track is *frozen* when it does not move at all; frozen tracks are dropped
        only while they are a minority and the other tracks clearly move, so a static
        camera keeps its points.
        """
        if len(a) < 2 * self.prm.min_inliers:
            return a, b
        flow = np.linalg.norm(b - a, axis=1)
        frozen = flow < self.prm.frozen_flow
        n_frozen = int(frozen.sum())
        if n_frozen == 0 or n_frozen >= 0.5 * len(a):
            return a, b
        if float(np.median(flow[~frozen])) < 3.0 * self.prm.frozen_flow:
            return a, b
        return a[~frozen], b[~frozen]

    def _fit(self, a: np.ndarray, b: np.ndarray, pair: _Pair, ransac: float,
             box: tuple[int, int, int, int]) -> bool:
        """RANSAC similarity; inliers must be numerous and spread over the tracking area
        (a fit to a strip or a blob of points is unconstrained across it)."""
        prm = self.prm
        m, inl = cv2.estimateAffinePartial2D(
            a, b, method=cv2.RANSAC, ransacReprojThreshold=ransac,
            maxIters=500, confidence=0.995, refineIters=10,
        )
        if m is None:
            return False
        inl = inl.ravel().astype(bool)
        if inl.sum() < max(prm.min_inliers, prm.min_inlier_ratio * len(a)):
            return False
        spread = a[inl].std(axis=0)
        if spread[0] < prm.min_spread * (box[2] - box[0]) or spread[1] < prm.min_spread * (box[3] - box[1]):
            return False
        d = m[:, :2] @ self.center + m[:, 2] - self.center
        pair.scale = math.hypot(m[0, 0], m[0, 1])
        pair.angle = math.degrees(math.atan2(m[0, 1], m[0, 0]))
        pair.dx, pair.dy = float(d[0]), float(d[1])
        pair.residual = float(np.linalg.norm(a[inl] @ m[:, :2].T + m[:, 2] - b[inl], axis=1).mean())
        pair.inliers = float(inl.mean())
        pair.quality = 2
        return True

    @staticmethod
    def _phase(g0: np.ndarray, g1: np.ndarray, box: tuple[int, int, int, int]) -> tuple[tuple[float, float], float]:
        x0, y0, x1, y1 = box
        if x1 - x0 < 8 or y1 - y0 < 8:
            return (0.0, 0.0), 0.0
        win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
        a = g0[y0:y1, x0:x1].astype(np.float32)
        b = g1[y0:y1, x0:x1].astype(np.float32)
        shift, response = cv2.phaseCorrelate(a, b, win)
        return shift, float(response)

    def _verify_shift(self, g0: np.ndarray, g1: np.ndarray, box: tuple[int, int, int, int],
                      sx: float, sy: float) -> bool:
        """Does translating ``g0`` by ``(sx, sy)`` really explain ``g1``?

        The phase-correlation peak is unreliable exactly when tracking failed: across a
        flash, an inverted frame or a blank frame it returns large bogus shifts. The
        check is a zero-mean normalised cross-correlation of the overlap at half
        resolution (tolerant to sub-pixel error, invariant to brightness / contrast).
        """
        x0, y0, x1, y1 = box
        ix, iy = int(round(sx)), int(round(sy))
        w, h = x1 - x0 - abs(ix), y1 - y0 - abs(iy)
        if w < 16 or h < 16:
            return False
        a = g0[y0 + max(0, -iy):y0 + max(0, -iy) + h, x0 + max(0, -ix):x0 + max(0, -ix) + w]
        b = g1[y0 + max(0, iy):y0 + max(0, iy) + h, x0 + max(0, ix):x0 + max(0, ix) + w]
        a = cv2.pyrDown(a).astype(np.float32).ravel()
        b = cv2.pyrDown(b).astype(np.float32).ravel()
        a -= a.mean()
        b -= b.mean()
        den = math.sqrt(float(a @ a) * float(b @ b))
        return den > 0 and float(a @ b) / den >= self.prm.min_shift_correlation


def _content_box(gray: np.ndarray, prm: _Params) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` tracking area of a shot: flat edge bars and a border removed."""
    h, w = gray.shape[1:]
    sample = gray[:: max(1, len(gray) // 8)].astype(np.int16)
    row_range = (sample.max(axis=2) - sample.min(axis=2)).max(axis=0)
    col_range = (sample.max(axis=1) - sample.min(axis=1)).max(axis=0)
    y0, y1 = _non_flat_extent(row_range < prm.bar_range)
    x0, x1 = _non_flat_extent(col_range < prm.bar_range)
    if y1 - y0 < 0.3 * h or x1 - x0 < 0.3 * w:  # (nearly) blank frames: keep everything
        x0, y0, x1, y1 = 0, 0, w, h
    b = prm.border
    return x0 + b, y0 + b, max(x0 + b, x1 - b), max(y0 + b, y1 - b)


def _non_flat_extent(flat: np.ndarray) -> tuple[int, int]:
    if flat.all():
        return 0, 0
    return int(np.argmin(flat)), len(flat) - int(np.argmin(flat[::-1]))


@dataclass
class _Estimates:
    """Per-frame similarity estimates ``t-1 -> t`` used for integration.

    Unlike :class:`FrameMotion` these include frames of photometric transitions
    (flash / fade), measured on contrast-normalised frames.
    """

    scale: np.ndarray
    rotation: np.ndarray
    tx: np.ndarray  # centre displacement, fraction of width
    ty: np.ndarray  # fraction of height
    residual: np.ndarray  # fraction of width
    inliers: np.ndarray  # RANSAC inlier fraction of the similarity fit
    quality: np.ndarray  # 2 = similarity fit, 1 = translation only, 0 = nothing
    bridged: np.ndarray  # one step of a motion measured across a gap of failed pairs


def _estimate_frame_motion(
    ctx: AnalysisContext, shots: list[ShotSpan], seg: ShotSegmentation, prm: _Params,
) -> tuple[FrameMotion, _Estimates]:
    n = ctx.n_frames
    w, h = ctx.size
    trans, photometric = _transition_masks(seg, n)
    est = _Estimates(np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                     np.zeros(n, np.int8), np.zeros(n, bool))
    magnitude = np.zeros(n)
    measured = np.zeros(n, bool)  # magnitude measured
    attempted = np.zeros(n, bool)  # pair t-1 -> t was measured (not skipped as a transition)
    estimator = _PairEstimator(w, h, prm)
    max_bridge = max(2, int(round(prm.bridge_seconds * ctx.fps)))

    def store(t: int, p: _Pair, steps: int = 1) -> None:
        """Record ``p`` as increment ``t`` (one of ``steps`` equal steps of a bridge)."""
        est.quality[t] = p.quality
        est.residual[t], est.inliers[t] = p.residual / w, p.inliers
        if p.quality == 2:
            est.scale[t], est.rotation[t] = p.scale ** (1.0 / steps), p.angle / steps
        # Centre displacement of one step: the steps compose to the measured displacement
        # d = sum_j z^j d_1 with z the step's complex scale-rotation (image y points down).
        z = complex(est.scale[t] * np.exp(-1j * np.radians(est.rotation[t])))
        series = (z ** steps - 1.0) / (z - 1.0) if abs(z - 1.0) > 1e-9 else float(steps)
        d = complex(p.dx, p.dy) / series
        est.tx[t], est.ty[t] = d.real / w, d.imag / h
        magnitude[t], measured[t] = p.magnitude / w / steps, p.flow
        est.bridged[t] = steps > 1

    for shot in shots:
        box = _content_box(ctx.gray[shot.f0:shot.f1], prm)
        for t in range(shot.f0 + 1, shot.f1):
            # A flash / fade only re-lights the content (I' = a I + b per frame), so the
            # motion is still measurable on contrast-normalised frames. Dissolves mix two
            # shots and whip / zoom / spin transitions add motion of their own: skipped.
            relit = bool(trans[t] or trans[t - 1])
            if relit and not (photometric[t - 1] and photometric[t]):
                continue
            attempted[t] = True
            store(t, _measure(estimator, ctx.gray[t - 1], ctx.gray[t], box, relit))
        # Bridge failed pairs (a flash, glitch or inverted frame, a blank frame inside the
        # shot) by measuring the frames on either side of the gap directly; the motion is
        # split into equal steps.
        sl = slice(shot.f0, shot.f1)
        for a, b in _runs(attempted[sl] & (est.quality[sl] < 2)):
            a, b = a + shot.f0, b + shot.f0
            i, j = a - 1, b - 1  # last frame before / last frame of the gap
            if not 2 <= b - a <= max_bridge or not (photometric[i] and photometric[j]):
                continue
            p = _measure(estimator, ctx.gray[i], ctx.gray[j], box, bool(trans[i] or trans[j]))
            if p.quality == 2:
                for t in range(a, b):
                    store(t, p, b - a)
        _fill_gaps(magnitude[sl], measured[sl])
    # The public per-frame motion is identity / invalid wherever a transition frame is involved.
    clean = ~(trans | np.concatenate([[False], trans[:-1]]))
    fm = FrameMotion(
        scale=np.where(clean, est.scale, 1.0), rotation=np.where(clean, est.rotation, 0.0),
        tx=np.where(clean, est.tx, 0.0), ty=np.where(clean, est.ty, 0.0),
        residual=np.where(clean, est.residual, 0.0),
        valid=clean & (est.quality == 2) & ~est.bridged, magnitude=magnitude,
    )
    return fm, est


def _measure(estimator: _PairEstimator, g0: np.ndarray, g1: np.ndarray, box: tuple[int, int, int, int],
             relit: bool) -> _Pair:
    """Similarity between two frames.

    Contrast-normalised frames are used when ``relit`` (flash / fade transition) and as a
    retry when the plain frames give no fit: in-shot flashes and exposure changes are
    affine in intensity, and an inverted frame (invert effect / glitch) is the same with a
    negative gain, so the last retry inverts the second frame.
    """
    best = _Pair() if relit else estimator(g0, g1, box)
    if best.quality == 2:
        return best
    (m0, s0), (m1, s1) = _photometry(g0), _photometry(g1)
    if min(s0, s1) < _FLAT_STD:
        return best
    n0, n1 = _normalise_contrast(g0, m0, s0), _normalise_contrast(g1, m1, s1)
    # Retry only where normalisation changes something (the frames were re-lit) and with
    # inversion only for anti-correlated frames: failures on unstructured footage (noise,
    # blur) must not cost three tracking passes.
    seconds = []
    if relit or abs(m0 - m1) > 8.0 or abs(math.log(s0 / s1)) > 0.2:
        seconds.append(n1)
    if _correlation(n0, n1) < -0.2:
        seconds.append(255 - n1)
    for second in seconds:
        p = estimator(n0, second, box)
        if p.quality > best.quality:
            best = p
        if best.quality == 2:
            break
    return best


def _photometry(gray: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of a grey frame."""
    mean, std = cv2.meanStdDev(gray)
    return float(mean[0, 0]), float(std[0, 0])


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of two frames at quarter resolution."""
    x, y = (cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            for g in (a, b))
    x -= x.mean()
    y -= y.mean()
    den = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y) / den if den > 0 else 0.0


def _normalise_contrast(gray: np.ndarray, mean: float, std: float) -> np.ndarray:
    """``gray`` (with the given mean / standard deviation) mapped to mean 128, std 48."""
    out = (gray.astype(np.float32) - mean) * (48.0 / max(std, 1e-6)) + 128.0
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _fill_gaps(values: np.ndarray, known: np.ndarray, max_gap: Optional[int] = None, k: int = 3) -> None:
    """In place: fill ``values[1:]`` where not ``known`` (index 0, the shot start, is kept).

    Used for frames without a trustworthy measurement (transition frames, flashes,
    blank frames): over a short gap the motion is better approximated by the
    neighbouring motion than by "no motion" or a bogus estimate. A gap is bridged
    linearly between the means of up to ``k`` known values on either side, so a
    single jittery neighbour (shake) is not held across it. Gaps longer than
    ``max_gap`` keep their values: extrapolating motion into a long unmeasured
    stretch would invent it.
    """
    gaps = ~known
    gaps[0] = False
    if not gaps.any() or not known.any():
        return
    ki = np.flatnonzero(known)
    for a, b in _runs(gaps):
        if max_gap is not None and b - a > max_gap:
            continue
        left, right = ki[ki < a][-k:], ki[ki >= b][:k]
        lv = values[left].mean() if left.size else None
        rv = values[right].mean() if right.size else None
        if lv is None or rv is None:
            values[a:b] = lv if rv is None else rv
        else:
            values[a:b] = lv + (rv - lv) * (np.arange(a, b) - a + 1) / (b - a + 1)


def _shot_spans(seg: ShotSegmentation, n: int) -> list[ShotSpan]:
    if n == 0:
        return []
    spans = [ShotSpan(s.index, max(0, s.f0), min(n, s.f1)) for s in seg.shots if min(n, s.f1) > max(0, s.f0)]
    return spans or [ShotSpan(0, 0, n)]


def _transition_masks(seg: ShotSegmentation, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(trans, photometric)``: frames of non-cut transitions, and frames that are either
    outside every transition or only inside flash / fade transitions."""
    trans = np.zeros(n, bool)
    m = seg.transition_frames()[:n]
    trans[: len(m)] = m
    other = np.zeros(n, bool)
    for tr in seg.transitions:
        if tr.kind != "cut" and tr.kind not in _PHOTOMETRIC and tr.f1 > tr.f0:
            other[max(0, tr.f0):min(n, tr.f1)] = True
    return trans, ~other


# =========================================================================== curve helpers


def _odd_window(seconds: float, fps: float, n: int, minimum: int = 5) -> int:
    """Odd window length of about ``seconds``, at most ``n`` (0 when ``n`` is too short)."""
    win = max(minimum, int(round(seconds * fps)) | 1)
    win = min(win, n if n % 2 else n - 1)
    return win if win >= minimum else 0


def _trend(y: np.ndarray, window: int) -> np.ndarray:
    return savgol_filter(y, window, 2, mode="interp") if window else y.copy()


def _turning_points(y: np.ndarray, hysteresis: float) -> list[int]:
    """Indices of alternating extrema of ``y`` whose swings exceed ``hysteresis`` (zig-zag)."""
    n = len(y)
    pivots = [0]
    direction, lo, hi, ext = 0, 0, 0, 0
    for i in range(1, n):
        v = y[i]
        if direction == 0:
            hi = i if v > y[hi] else hi
            lo = i if v < y[lo] else lo
            if y[hi] - y[lo] >= hysteresis:
                direction, start, ext = (1, lo, hi) if hi > lo else (-1, hi, lo)
                if start > 0:
                    pivots.append(start)
        elif direction > 0:
            if v >= y[ext]:
                ext = i
            elif y[ext] - v >= hysteresis:
                pivots.append(ext)
                direction, ext = -1, i
        else:
            if v <= y[ext]:
                ext = i
            elif v - y[ext] >= hysteresis:
                pivots.append(ext)
                direction, ext = 1, i
    if direction and ext > pivots[-1]:
        pivots.append(ext)
    if pivots[-1] != n - 1:
        pivots.append(n - 1)
    return pivots


def _moves(y: np.ndarray, hysteresis: float, threshold: float, pause: int = 0) -> list[tuple[int, int]]:
    """Monotone moves of ``y`` larger than ``threshold`` as ``(a, b)`` frame ranges.

    A move runs from where it passes 5% to where it passes 95% of its total change,
    tightened to the stretches where the speed stays above 10% of the leg's peak speed,
    so slow drift (e.g. compression bias) or jitter before or after a quick move does
    not stretch it. Fast stretches of one leg separated by at most ``pause`` frames
    (a move that halts briefly, e.g. under a freeze frame, and resumes) are one move;
    farther ones are separate moves when each is large enough on its own. Consecutive
    moves in the same direction with at most ``pause`` frames between them (a small
    wobble back in between) are merged too, so one camera move is not reported as
    fragments.
    """
    out: list[tuple[int, int, float]] = []
    piv = _turning_points(y, hysteresis)
    velocity = np.gradient(y) if len(y) > 1 else np.zeros(len(y))
    for p, q in zip(piv[:-1], piv[1:]):
        delta = y[q] - y[p]
        if abs(delta) < threshold:
            continue
        sign = float(np.sign(delta))
        prog = (y[p:q + 1] - y[p]) / delta
        a = p + int(np.argmax(prog >= 0.05))
        b = q - int(np.argmax(prog[::-1] <= 0.95))
        leg = velocity[p:q + 1] * sign  # positive while moving in the leg's direction
        peak = p + int(np.argmax(leg))
        groups: list[list[int]] = []
        for s, e in _runs(leg >= 0.1 * float(leg.max())):
            if groups and s + p - groups[-1][1] <= pause + 1:
                groups[-1][1] = e - 1 + p
            else:
                groups.append([s + p, e - 1 + p])
        for f0, f1 in groups:
            if not f0 <= peak <= f1 and (y[f1] - y[f0]) * sign < threshold:
                continue  # a minor stretch of the leg, too small to be a move of its own
            lo, hi = (max(a, f0), min(b, f1)) if max(a, f0) < min(b, f1) else (f0, f1)
            out.append((lo, max(hi, lo + 1), sign))
    merged: list[tuple[int, int, float]] = []
    for a, b, sign in out:
        if merged and merged[-1][2] == sign and a - merged[-1][1] <= pause:
            merged[-1] = (merged[-1][0], b, sign)
        else:
            merged.append((a, b, sign))
    return [(a, b) for a, b, _ in merged]


def _merge(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` runs of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _lstsq(x: np.ndarray, y: np.ndarray) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
    """Least squares ``y ~ x @ beta`` (``y`` may be complex: both axes at once).

    Returns ``(beta, sse, inv(x.T @ x))`` or ``None`` when ``x`` is degenerate.
    """
    xtx = x.T @ x
    if np.linalg.cond(xtx) > 1e12:
        return None
    inv = np.linalg.inv(xtx)
    beta = inv @ (x.T @ y)
    return beta, float((np.abs(y - x @ beta) ** 2).sum()), inv


# =========================================================================== easing fit


@dataclass
class _Fit:
    """``channel[i] ~ offsets[c] + amps[c] * ease((i - t0) / (t1 - t0))`` near the move."""

    easing: str
    t0: float  # shot-relative frame positions (continuous)
    t1: float
    offsets: np.ndarray
    amps: np.ndarray
    r2: float

    def progress(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(fx.ease(np.clip((idx - self.t0) / (self.t1 - self.t0), 0.0, 1.0), self.easing))


def _linear_fit(g: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row least squares ``y ~ off + amp * g`` for every channel.

    ``g`` is ``(K, L)`` candidate progress curves, ``ys`` is ``(C, L)``. Returns the summed
    SSE ``(K,)`` and offsets / amplitudes ``(K, C)``.
    """
    n = g.shape[1]
    sg, sgg = g.sum(axis=1), np.einsum("kl,kl->k", g, g)
    det = n * sgg - sg * sg
    ok = det > 1e-9 * n * n
    safe = np.where(ok, det, 1.0)
    sy, syy = ys.sum(axis=1), np.einsum("cl,cl->c", ys, ys)
    sgy = g @ ys.T  # (K, C)
    amp = np.where(ok[:, None], (n * sgy - sg[:, None] * sy[None, :]) / safe[:, None], 0.0)
    off = (sy[None, :] - amp * sg[:, None]) / n
    sse = (syy[None, :] - off * sy[None, :] - amp * sgy).sum(axis=1)
    return np.maximum(sse, 0.0), off, amp


def _fit_move(channels: Sequence[np.ndarray], a: int, b: int, lo: int, hi: int, fps: float) -> Optional[_Fit]:
    """Fit every easing with free start / end to the move detected at ``(a, b)``.

    ``lo``/``hi`` bound the fit window and the span (neighbouring moves or the shot):
    progress 0 is at frame ``t0 >= lo`` and progress 1 at frame ``t1 <= hi - 1``. The
    change start is searched around ``a`` and the change end around ``b``; for "punch"
    the span is five times the snap, clipped to ``hi - 1``.
    """
    d = max(b - a, 1)
    margin = max(2, int(round(0.2 * fps)))
    w_lo = max(lo, int(math.floor(a - 0.6 * d - margin)))
    w_hi = min(hi, int(math.ceil(b + 0.8 * d + margin)) + 1)
    if w_hi - w_lo < 4:
        return None
    t_min, t_max = float(lo), float(hi - 1)
    idx = np.arange(w_lo, w_hi, dtype=float)
    ys = np.stack([np.asarray(c[w_lo:w_hi], float) for c in channels])
    sst = float(((ys - ys.mean(axis=1, keepdims=True)) ** 2).sum())
    fits: dict[str, tuple[float, _Fit]] = {}
    for easing in FIT_EASINGS:
        punch = easing == "punch"
        t0s = np.linspace(max(w_lo, a - 0.45 * d - 2), a + 1.0, 12)
        ends = np.linspace(b - 1.0, min(w_hi - 1.0, b + (0.7 if punch else 0.45) * d + 2), 12)
        for _ in range(2):  # coarse grid, then a finer grid around the best cell
            t0, e = (np.clip(v.ravel(), t_min, t_max) for v in np.meshgrid(t0s, ends, indexing="ij"))
            t1 = np.minimum(t0 + (e - t0) / _PUNCH_SNAP, t_max) if punch else e
            keep = t1 > t0 + 0.5
            if not keep.any():
                break
            t0, e, t1 = t0[keep], e[keep], t1[keep]
            x = np.clip((idx[None, :] - t0[:, None]) / (t1 - t0)[:, None], 0.0, 1.0)
            sse, off, amp = _linear_fit(np.asarray(fx.EASINGS[easing](x)), ys)
            k = int(np.argmin(sse))
            if easing not in fits or sse[k] < fits[easing][0]:
                r2 = 1.0 - float(sse[k]) / sst if sst > 0 else 0.0
                fits[easing] = (float(sse[k]), _Fit(easing, float(t0[k]), float(t1[k]), off[k], amp[k], r2))
            step0 = (t0s[-1] - t0s[0]) / (len(t0s) - 1) if len(t0s) > 1 else 1.0
            step1 = (ends[-1] - ends[0]) / (len(ends) - 1) if len(ends) > 1 else 1.0
            t0s = np.linspace(t0[k] - step0, t0[k] + step0, 9)
            ends = np.linspace(e[k] - step1, e[k] + step1, 9)
    if not fits:
        return None
    smooth = [v for k, v in fits.items() if k != "punch"]
    if not smooth:
        return fits["punch"][1]
    best_sse, best = min(smooth, key=lambda v: v[0])
    if "punch" in fits:
        # A punch is a snap that then *holds* for four times as long. Over the few frames
        # of a fast snap all easings fit equally well and only that hold identifies it, so
        # a fast punch wins when it fits as well as the best smooth easing (up to a noise
        # floor relative to the move size). A slow snap is told apart by its cubic shape,
        # which under noise is easily confused with ease_out: it needs a clear win.
        p_sse, p_fit = fits["punch"]
        if _PUNCH_SNAP * (p_fit.t1 - p_fit.t0) <= _FAST_SNAP * fps:
            floor = ys.shape[1] * (_SHAPE_TOLERANCE * float(np.linalg.norm(best.amps))) ** 2
            better = p_sse <= _PUNCH_TOLERANCE * best_sse + floor
        else:
            better = p_sse < _PUNCH_MARGIN * best_sse
        if better:
            best = p_fit
    return best


# =========================================================================== per-shot analysis


@dataclass
class _Curves:
    """Cumulative motion of one shot (index 0 = its first frame)."""

    log_scale: np.ndarray
    angle: np.ndarray  # degrees
    pan_x: np.ndarray  # zoom-corrected content displacement, fraction of width
    pan_y: np.ndarray  # fraction of height


@dataclass
class _ShotResult:
    effects: list[EffectBase]
    summary: MotionSummary
    curves: _Curves
    shake_level: np.ndarray


class _ShotAnalyzer:
    """Integrates the frame motion of one shot and turns it into effects + a summary."""

    def __init__(self, ctx: AnalysisContext, shot: ShotSpan, fm: FrameMotion, est: _Estimates,
                 prm: _Params) -> None:
        self.ctx, self.shot, self.prm = ctx, shot, prm
        self.fps = ctx.fps
        self.w, self.h = ctx.size
        sl = slice(shot.f0, shot.f1)
        self.n = shot.n_frames
        self.idx = np.arange(self.n, dtype=float)
        self.inside = self.idx >= 1  # frames with an increment (index 0 is the shot boundary)
        self.good = est.quality[sl] == 2  # similarity fit (never at index 0)
        self.has_t = est.quality[sl] >= 1  # measured translation
        self.residual = est.residual[sl]
        self.inliers = est.inliers[sl]
        self.magnitude = fm.magnitude[sl]

        # Per-frame increments; unmeasured frames (dissolves, failed tracking) borrow
        # interpolated increments from their neighbours (see module docstring, step 2).
        log_r, ang = np.log(est.scale[sl]), est.rotation[sl].copy()
        disp = est.tx[sl] * self.w + 1j * est.ty[sl] * self.h  # centre displacement, px
        max_gap = max(1, int(round(prm.fill_seconds * self.fps)))
        for values, known in ((log_r, self.good), (ang, self.good), (disp, self.has_t)):
            _fill_gaps(values, known, max_gap)
        self.inc_ratio = np.exp(log_r)
        self.inc_disp = disp
        self.log_scale = np.cumsum(log_r)
        self.angle = np.cumsum(ang)
        self.trend_window = _odd_window(prm.smooth_seconds, self.fps, self.n)
        self.pause = int(round(prm.pause_seconds * self.fps))

    # ------------------------------------------------------------------ driver

    def run(self) -> _ShotResult:
        zooms, zoom_fits = self._zooms()
        # Pans are first detected with centred zooms. Each zoom centre is then estimated
        # without the frames of pans that do not overlap it, and the pans are detected
        # again with the off-centre displacement removed (it is not camera motion).
        pan_x, pan_y = self._pan_curves(zoom_fits, [(0.5, 0.5)] * len(zoom_fits))
        pans, pan_fits = self._pans(pan_x, pan_y)
        centers = [self._zoom_center(fit, zoom_fits, pan_fits) for fit in zoom_fits]
        if any(c != (0.5, 0.5) for c in centers):
            for zoom, center in zip(zooms, centers):
                zoom.center = center
            pan_x, pan_y = self._pan_curves(zoom_fits, centers)
            pans, pan_fits = self._pans(pan_x, pan_y)
        rotates, rot_fits = self._rotates()
        shakes, level = self._shakes(pan_x, pan_y, pan_fits, rot_fits)
        curves = _Curves(self.log_scale, self.angle, pan_x, pan_y)
        # Every fitted move shapes the curves above (pan correction, shake baseline), but
        # only moves with enough evidence are reported: weak fits are mostly a subject
        # moving on its own, parallax-heavy footage or tracking noise.
        zooms, pans, rotates, shakes = ([e for e in group if e.confidence >= self.prm.min_confidence]
                                        for group in (zooms, pans, rotates, shakes))
        effects = sorted(zooms + pans + rotates + shakes, key=lambda e: (e.start, e.type))
        summary = self._summary(curves, zooms, pans, rotates, shakes, level)
        return _ShotResult(effects, summary, curves, level)

    def _pan_curves(self, zoom_fits: list[_Fit], centers: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative content displacement (x / width, y / height) with the centre
        displacement of off-centre zooms, ``(r_t - 1) (m - c)``, removed."""
        disp = self.inc_disp.copy()
        m = complex(self.w / 2.0, self.h / 2.0)
        for fit, (cx, cy) in zip(zoom_fits, centers):
            k = self._span(fit)
            disp[k] -= (self.inc_ratio[k] - 1.0) * (m - complex(cx * self.w, cy * self.h))
        cum = np.cumsum(disp)
        return cum.real / self.w, cum.imag / self.h

    # ------------------------------------------------------------------ helpers

    def _span(self, fit: _Fit, margin: float = 0.0) -> np.ndarray:
        """Frames whose increment (from the previous frame) lies in a fitted move (+ margin)."""
        return (self.idx > fit.t0 - margin) & (self.idx <= fit.t1 + 1.0 + margin)

    def _time(self, i: float) -> float:
        """Shot-relative (fractional) frame -> absolute seconds, clamped to the shot."""
        return self.ctx.t(self.shot.f0 + float(np.clip(i, 0.0, self.n)))

    def _quality(self, i0: float, i1: float, r2: float, strength: Optional[float] = None) -> float:
        """Confidence that frames ``(i0, i1]`` hold a move of the whole picture.

        A product of the evidence, each factor in ``[0, 1]``:

        * *coverage*: share of the frames with a similarity fit;
        * *rigidity*: the RANSAC inlier share, from ``min_inlier_ratio`` (no evidence) to
          ``digital_inliers`` (one rigid picture). Parallax lowers it moderately, while a
          large subject moving on its own (a character or vehicle filling the frame over
          a static or featureless background) leaves most tracks off the fitted motion;
        * *residual*: mean reprojection error of the inliers (``parallax_residual`` scale);
        * *consistency*: ``r2`` of the eased model over the move (a coherent move vs.
          wandering);
        * *strength*: the change relative to its detection threshold (``strength``; a
          move barely above it is as likely drift or tracking bias as an edit), halved
          at the threshold and full from twice the threshold;
        * *duration*: measured frames relative to ``min_move_seconds``.
        """
        prm = self.prm
        k = (self.idx > i0) & (self.idx <= i1) & self.inside
        if not k.any():
            return 0.0
        valid = float(self.good[k].mean())
        fitted = k & self.good
        if not fitted.any():
            return 0.0
        res = float(self.residual[fitted].mean())
        inl = float(self.inliers[fitted].mean())
        rigid = float(np.clip((inl - prm.min_inlier_ratio) / (prm.digital_inliers - prm.min_inlier_ratio), 0.0, 1.0))
        res_factor = float(np.clip(1.0 - res / (2.0 * prm.parallax_residual), 0.2, 1.0))
        consistency = float(np.clip(r2, 0.0, 1.0))
        strong = 1.0 if strength is None else float(np.clip(0.5 * strength, 0.5, 1.0))
        frames = float(np.clip(fitted.sum() / max(1.0, prm.min_move_seconds * self.fps), 0.0, 1.0))
        return round(valid * rigid * res_factor * consistency * strong * frames, 3)

    def _observed(self, curve: np.ndarray, known: np.ndarray, fit: _Fit) -> float:
        """Change of ``curve`` over a fitted move counting only measured increments.

        Interpolated increments may complete a move seen through a flash, but they
        must not create one: the measured part alone has to be significant.
        """
        inc = np.diff(curve, prepend=curve[:1])
        return float(inc[(self.idx > fit.t0) & (self.idx <= fit.t1) & known].sum())

    def _covered(self, known: np.ndarray, fit: _Fit) -> bool:
        """Were enough of the increments of a fitted move measured to trust it at all?"""
        k = (self.idx > fit.t0) & (self.idx <= fit.t1) & self.inside
        return bool(k.any()) and float(known[k].mean()) >= self.prm.min_coverage

    def _change(self, curve: np.ndarray, fit: _Fit) -> float:
        """Change of ``curve`` between the start and the end of a fitted move."""
        return float(np.interp(fit.t1, self.idx, curve) - np.interp(fit.t0, self.idx, curve))

    def _limits(self, moves: list[tuple[int, int]], k: int) -> tuple[int, int]:
        lo = moves[k - 1][1] if k > 0 else 0
        hi = moves[k + 1][0] if k + 1 < len(moves) else self.n
        return lo, hi

    # ------------------------------------------------------------------ zoom

    def _zooms(self) -> tuple[list[Zoom], list[_Fit]]:
        """Zoom nodes (centred; see :meth:`_zoom_center`) and their fits."""
        prm = self.prm
        if self.n < 4:
            return [], []
        thr = math.log1p(prm.zoom_threshold)
        smooth = gaussian_filter1d(self.log_scale, 1.0, mode="nearest")
        moves = _moves(smooth, 0.5 * thr, thr, self.pause)
        scale = np.exp(self.log_scale)
        zooms, fits = [], []
        for k, (a, b) in enumerate(moves):
            lo, hi = self._limits(moves, k)
            fit = _fit_move([scale], a, b, lo, hi, self.fps)
            if fit is None or fit.offsets[0] <= 0:
                continue
            ratio = float((fit.offsets[0] + fit.amps[0]) / fit.offsets[0])
            if ratio <= 0 or abs(math.log(ratio)) < 0.9 * thr:
                continue
            if abs(self._observed(self.log_scale, self.good, fit)) < 0.75 * thr or not self._covered(self.good, fit):
                continue
            # A digital zoom scales one rigid picture along a clean easing curve: every track
            # fits it. A dolly / camera move shows parallax or independent motion (residual
            # error, many outliers); per frame that can be subtle, so None is common.
            sel = (self.idx > fit.t0) & (self.idx <= fit.t1) & self.good
            res = float(self.residual[sel].mean()) if sel.any() else None
            inl = float(self.inliers[sel].mean()) if sel.any() else None
            digital: Optional[bool] = None
            if res is not None and inl is not None:
                if res > prm.parallax_residual or inl < prm.parallax_inliers:
                    digital = False
                elif res < prm.digital_residual and inl >= prm.digital_inliers and fit.r2 >= prm.digital_r2:
                    digital = True
            kt = np.linspace(fit.t0, fit.t1, _N_KEYFRAMES)
            s0 = np.interp(fit.t0, self.idx, scale)
            keyframes = [Keyframe(t=self._time(t), v=float(np.interp(t, self.idx, scale) / s0)) for t in kt]
            zooms.append(Zoom(
                start=self._time(fit.t0), end=self._time(fit.t1 + 1.0), easing=fit.easing,
                scale_from=1.0, scale_to=ratio, digital=digital, keyframes=keyframes,
                confidence=self._quality(fit.t0, fit.t1, fit.r2, abs(math.log(ratio)) / thr),
                params={"scale_start": float(fit.offsets[0]), "fit_r2": round(fit.r2, 4),
                        "residual": None if res is None else round(res, 5),
                        "inliers": None if inl is None else round(inl, 3)},
            ))
            fits.append(fit)
        return zooms, fits

    def _zoom_center(self, fit: _Fit, zoom_fits: list[_Fit], pan_fits: list[_Fit]) -> tuple[float, float]:
        """Fixed point of the zoom (normalised), separated from a simultaneous camera pan.

        A zoom by ``r_t = s_t / s_{t-1}`` about ``c`` over footage whose content moves
        ``w_t`` px per frame moves the frame centre ``m`` by
        ``d_t = (r_t - 1) (m - c) + s_t w_t`` (the pan is magnified by the zoom). An
        off-centre zoom is exactly a centred zoom plus a pan with velocity profile
        ``r_t - 1``, so the split rests on a prior: the pan velocity varies at most
        linearly over the zoom plus a short margin on either side. A "pan" that starts
        and stops exactly with the zoom and follows its velocity profile is therefore
        read as an off-centre fixed point (the more compact description of the same
        pixels), while a pan that runs through the zoom stays a pan. The centre leaves
        0.5 only when the off-centre term explains most of the displacement that such a
        pan cannot (a model comparison, because the two terms are nearly collinear for
        smooth zooms), the centre shift it implies over the zoom is visible
        (``center_min_shift``) and the offset is significant on each axis.
        """
        prm = self.prm
        own = self._span(fit)
        k = self._span(fit, round(prm.center_margin * self.fps)) & self.good
        # Other zooms, and pans just before / after this zoom, break the prior: skip them.
        for other in zoom_fits:
            if other is not fit:
                k &= own | ~self._span(other)
        for pan in pan_fits:
            span = self._span(pan)
            if not (span & own).any():
                k &= ~span
        m = int(k.sum())
        if m < max(8, round(prm.center_min_seconds * self.fps)):
            return 0.5, 0.5
        t = self.idx[k]
        tau = (t - t.mean()) / max(1.0, float(np.ptp(t)))
        s = np.exp(self.log_scale[k])
        d = self.inc_disp[k]
        pan_only = _lstsq(np.stack([s, s * tau], axis=1), d)
        full = _lstsq(np.stack([self.inc_ratio[k] - 1.0, s, s * tau], axis=1), d)
        if pan_only is None or full is None:
            return 0.5, 0.5
        beta, sse, inv = full  # beta[0] = centre - fixed point, px
        if pan_only[1] - sse < 0.5 * pan_only[1]:
            return 0.5, 0.5
        shift = abs(complex(beta[0])) * abs(float((self.inc_ratio[k] - 1.0).sum()))
        if shift < prm.center_min_shift * self.w:
            return 0.5, 0.5
        se = math.sqrt(sse / (2.0 * max(1, m - 3)) * inv[0, 0])
        c = complex(self.w / 2.0, self.h / 2.0) - complex(beta[0])
        out = []
        for value, size in ((c.real, self.w), (c.imag, self.h)):
            norm = value / size
            if abs(norm - 0.5) < max(prm.center_snap, 3.0 * se / size):
                norm = 0.5
            out.append(norm)
        if not all(0.0 <= v <= 1.0 for v in out):  # a fixed point outside the picture is no zoom centre
            return 0.5, 0.5
        return out[0], out[1]

    # ------------------------------------------------------------------ pan / rotate

    def _pans(self, pan_x: np.ndarray, pan_y: np.ndarray) -> tuple[list[Pan], list[_Fit]]:
        prm = self.prm
        if not self.trend_window:
            return [], []
        thr = prm.pan_threshold
        trends = [_trend(curve, self.trend_window) for curve in (pan_x, pan_y)]
        cand = []
        for trend in trends:
            cand += _moves(trend, 0.4 * thr, 0.6 * thr, self.pause)
        moves = _merge(cand, int(round(0.1 * self.fps)))
        pans, fits = [], []
        for k, (a, b) in enumerate(moves):
            lo, hi = self._limits(moves, k)
            fit = _fit_move([pan_x, pan_y], a, b, lo, hi, self.fps)
            if fit is None or math.hypot(*fit.amps) < 0.9 * thr:
                continue
            if math.hypot(*(self._change(t, fit) for t in trends)) < 0.75 * thr:
                continue  # the raw fit locked onto jitter (shake), the trend barely moves
            if math.hypot(*(self._observed(c, self.has_t, fit) for c in (pan_x, pan_y))) < 0.75 * thr:
                continue
            if not self._covered(self.has_t, fit):
                continue
            pans.append(Pan(
                start=self._time(fit.t0), end=self._time(fit.t1 + 1.0), easing=fit.easing,
                dx=float(fit.amps[0]), dy=float(fit.amps[1]),
                confidence=self._quality(fit.t0, fit.t1, fit.r2, math.hypot(*fit.amps) / thr),
                params={"fit_r2": round(fit.r2, 4)},
            ))
            fits.append(fit)
        return pans, fits

    def _rotates(self) -> tuple[list[Rotate], list[_Fit]]:
        prm = self.prm
        if not self.trend_window:
            return [], []
        thr = prm.rotate_threshold
        trend = _trend(self.angle, self.trend_window)
        moves = _moves(trend, 0.5 * thr, thr, self.pause)
        rotates, fits = [], []
        for k, (a, b) in enumerate(moves):
            lo, hi = self._limits(moves, k)
            fit = _fit_move([self.angle], a, b, lo, hi, self.fps)
            if fit is None or abs(fit.amps[0]) < 0.9 * thr or abs(self._change(trend, fit)) < 0.75 * thr:
                continue
            if abs(self._observed(self.angle, self.good, fit)) < 0.75 * thr or not self._covered(self.good, fit):
                continue
            kt = np.linspace(fit.t0, fit.t1, _N_KEYFRAMES)
            a0 = np.interp(fit.t0, self.idx, self.angle)
            rotates.append(Rotate(
                start=self._time(fit.t0), end=self._time(fit.t1 + 1.0), easing=fit.easing,
                angle_from=0.0, angle_to=float(fit.amps[0]),
                keyframes=[Keyframe(t=self._time(t), v=float(np.interp(t, self.idx, self.angle) - a0)) for t in kt],
                confidence=self._quality(fit.t0, fit.t1, fit.r2, abs(float(fit.amps[0])) / thr),
                params={"fit_r2": round(fit.r2, 4)},
            ))
            fits.append(fit)
        return rotates, fits

    # ------------------------------------------------------------------ shake

    def _shakes(self, pan_x: np.ndarray, pan_y: np.ndarray, pan_fits: list[_Fit],
                rot_fits: list[_Fit]) -> tuple[list[Shake], np.ndarray]:
        """Shake nodes plus the per-frame jitter level (peak per-axis offset, width units)."""
        prm = self.prm
        level = np.zeros(self.n)
        if not self.trend_window:
            return [], level
        aspect = self.h / self.w  # y residual in width units, as the Shake convention requires
        x, y = pan_x, pan_y * aspect
        # The fitted moves follow sharp easings that the SG trend would smear into the
        # residual; remove them first and let the SG trend take only drift and offsets.
        base_x, base_y, base_a = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
        for fit in pan_fits:
            prog = fit.progress(self.idx)
            base_x += fit.amps[0] * prog
            base_y += fit.amps[1] * aspect * prog
        for fit in rot_fits:
            base_a += fit.amps[0] * fit.progress(self.idx)
        dx, dy, da = x - base_x, y - base_y, self.angle - base_a
        rx, ry, ra = (v - _trend(v, self.trend_window) for v in (dx, dy, da))
        rx[~self.has_t] = ry[~self.has_t] = ra[~self.good] = 0.0  # interpolated frames carry no jitter

        win = _odd_window(prm.shake_window, self.fps, self.n, minimum=3) or 3
        level = np.sqrt(2.0 * np.maximum(uniform_filter1d((rx * rx + ry * ry) / 2.0, win, mode="nearest"), 0.0))
        rot_level = np.sqrt(2.0 * np.maximum(uniform_filter1d(ra * ra, win, mode="nearest"), 0.0))
        mask = (level > prm.shake_threshold) | (rot_level > prm.shake_rotation_threshold)
        for a, b in _runs(~mask):  # close short gaps inside a shaky stretch
            if 0 < a and b < self.n and b - a <= win // 2:
                mask[a:b] = True
        min_len = max(3, int(round(prm.shake_min_duration * self.fps)))
        rad = math.pi / 360.0  # degrees -> displacement (width units) at half the width
        shakes = []
        edge = max(1, int(round(prm.shake_edge_seconds * self.fps)))
        for a, b in _runs(mask):
            a, b = self._trim_shake(a, b, rx, ry, ra * rad)
            if b - a < min_len:
                continue
            sx, sy, sa = rx[a:b], ry[a:b], ra[a:b]
            signals = [sx, sy, sa * rad]
            power = [float(np.mean(v * v)) for v in signals]
            if max(power) <= 0:
                continue
            # One-sided excursions (e.g. a nudge) are not shake: some axis carrying a fair
            # share of the jitter must swing back and forth. With 2-D jitter one axis may sit
            # near a zero crossing at both ends of a short shake, so every such axis counts.
            if max(_oscillations(v) for v, e in zip(signals, power) if e >= 0.25 * max(power)) < 3:
                continue
            # It must also be periodic. A sinusoid (tied across axes by the renderer's ratios)
            # is fitted to the jitter before the trend removal, which absorbs part of it
            # unevenly across frequencies and near shot edges, with a quadratic standing in
            # for the trend over the span: its frequency and per-axis amplitudes describe the
            # shake. Periodicity is judged on the frame-to-frame velocity, where one sinusoid
            # explains nearly all of real jitter but not the steps of a move halting and
            # resuming (e.g. under a freeze frame), whose position residue can pass for one
            # slow cycle.
            span = [dx[a:b], dy[a:b], da[a:b] * rad]
            freq, _, fitted = _fit_jitter(span, _SHAKE_AXIS_RATIOS, self.fps, degree=2)
            _, share, _ = _fit_jitter([np.diff(v) for v in span], _SHAKE_AXIS_RATIOS, self.fps)
            if share < prm.shake_periodicity or freq * (b - a) / self.fps < prm.shake_min_cycles:
                continue
            ax, ay, rot_amp = float(fitted[0]), float(fitted[1]), float(fitted[2]) / rad
            amp = math.sqrt((ax * ax + ay * ay) / 2.0)
            if amp < 0.5 * prm.shake_threshold and rot_amp < prm.shake_rotation_threshold:
                continue
            strength = max(amp / prm.shake_threshold, rot_amp / prm.shake_rotation_threshold)
            conf = self._quality(a - 1, b - 1, 1.0) * float(np.clip(0.5 + 0.25 * (strength - 1.0), 0.3, 1.0)) * share
            # The jitter of the first frame is unobservable (it is the reference) and the
            # trim can lose a frame at either end: a shake this close to a cut starts / ends there.
            a = 0 if a <= edge else a
            b = self.n if b >= self.n - edge else b
            shakes.append(Shake(
                start=self._time(a), end=self._time(b), amplitude=amp, frequency=freq,
                rotation_amplitude=rot_amp, confidence=round(conf, 3),
                params={"amplitude_x": round(ax, 5), "amplitude_y": round(ay, 5)},
            ))
        return shakes, level

    @staticmethod
    def _trim_shake(a: int, b: int, *signals: np.ndarray) -> tuple[int, int]:
        """Shrink a windowed-RMS run to where the jitter itself reaches half its peak.

        The jitter envelope is the largest per-axis offset over three frames, so a sample
        that happens to fall on a zero crossing of every axis does not end the shake early.
        """
        inst = maximum_filter1d(np.max(np.abs(np.stack([s[a:b] for s in signals])), axis=0), 3, mode="nearest")
        peak = float(inst.max()) if inst.size else 0.0
        if peak <= 0:
            return a, a
        hit = np.flatnonzero(inst >= 0.5 * peak)
        return a + int(hit[0]), a + int(hit[-1]) + 1

    # ------------------------------------------------------------------ summary

    def _summary(self, curves: _Curves, zooms: list[Zoom], pans: list[Pan], rotates: list[Rotate],
                 shakes: list[Shake], level: np.ndarray) -> MotionSummary:
        prm = self.prm
        meas = self.inside
        magnitude = float(self.magnitude[meas].mean()) if meas.any() else 0.0
        valid = float(self.good[meas].mean()) if meas.any() else 0.0
        has_t = float(self.has_t[meas].mean()) if meas.any() else 0.0
        # Cumulative curves only classify the camera when most of the shot was measured
        # (sub-threshold moves of a slow camera); otherwise only detected effects count.
        sim_ok, t_ok = valid >= prm.min_coverage, has_t >= prm.min_coverage
        comps: list[CameraMove] = []
        if zooms or (sim_ok and np.ptp(curves.log_scale) >= math.log1p(prm.zoom_threshold)):
            comps.append("zoom")
        if pans or (t_ok and max(np.ptp(curves.pan_x), np.ptp(curves.pan_y)) >= prm.pan_threshold):
            comps.append("pan")
        if rotates or (sim_ok and np.ptp(curves.angle) >= prm.rotate_threshold):
            comps.append("rotate")
        duration = self.n / self.fps
        shaky = (sum(s.duration for s in shakes) >= 0.4 * duration
                 or (meas.any() and float(np.median(level[meas])) >= 0.5 * prm.shake_threshold))
        res = float(self.residual[self.good].mean()) if self.good.any() else 0.0
        inl = float(self.inliers[self.good].mean()) if self.good.any() else 0.0
        poor_fit = not sim_ok or res > prm.parallax_residual or inl < prm.parallax_inliers
        camera: CameraMove
        if len(comps) > 1:
            camera = "complex"
        elif comps:
            camera = comps[0]
        elif shaky:
            camera = "handheld"
        elif poor_fit and magnitude > 0.005:
            camera = "complex"
        else:
            camera = "static"
        return MotionSummary(magnitude=magnitude, camera=camera, zoom_ratio=float(np.exp(curves.log_scale[-1])),
                             pan=(float(curves.pan_x[-1]), float(curves.pan_y[-1])),
                             rotation=float(curves.angle[-1]))


# =========================================================================== shake helpers


def _oscillations(signal: np.ndarray) -> int:
    """Sign changes between excursions beyond +-30% of the peak (Schmitt trigger)."""
    peak = float(np.abs(signal).max()) if signal.size else 0.0
    if peak <= 0:
        return 0
    state, count = 0, 0
    for v in signal:
        s = 1 if v > 0.3 * peak else (-1 if v < -0.3 * peak else 0)
        if s and s != state:
            count += state != 0
            state = s
    return count


def _fit_jitter(signals: Sequence[np.ndarray], ratios: Sequence[float], fps: float,
                degree: int = 1, f_min: float = 1.0, step: float = 0.02) -> tuple[float, float, np.ndarray]:
    """Least-squares sinusoid fit of the jitter: ``(frequency, explained share, amplitudes)``.

    Signal ``k`` is modelled as ``a_k cos(2 pi r_k f t) + b_k sin(2 pi r_k f t)`` plus a
    polynomial nuisance trend of ``degree``, with ``r_k = ratios[k]`` (the renderer's axis ratios), and
    the frequency maximising the summed explained energy is returned along with the
    share of the (detrended) energy it explains. This is the maximum-likelihood
    frequency of a sinusoid in white noise: unlike a windowed periodogram it stays
    accurate over the one to three cycles of a short shake, and the share tells real
    oscillation (most of the energy at one frequency) from the residue of velocity
    kinks (a move halting and resuming), which no single sinusoid explains. The
    amplitudes are the fitted per-signal sinusoid amplitudes at that frequency.
    """
    n = len(signals[0])
    amps = np.zeros(len(signals))
    if n < degree + 4:
        return 0.0, 0.0, amps
    t = np.arange(n) / fps
    u = (t - t.mean()) / max(float(np.ptp(t)), 1e-9)
    q, _ = np.linalg.qr(np.stack([u ** d for d in range(degree + 1)], axis=1))

    def deflate(m: np.ndarray) -> np.ndarray:  # remove the nuisance polynomial along the last axis
        return m - (m @ q) @ q.T

    ys = [deflate(np.asarray(s, float)) for s in signals]
    energy = sum(float(y @ y) for y in ys)
    f_max = 0.5 * fps / max(ratios)
    if energy <= 0 or f_max <= f_min:
        return 0.0, 0.0, amps

    def project(f: np.ndarray, y: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Explained energy and cos / sin coefficients of ``y`` at frequencies ``f``."""
        w = 2.0 * np.pi * ratio * f[:, None] * t[None, :]
        c, s = deflate(np.cos(w)), deflate(np.sin(w))
        cc, ss, cs = (c * c).sum(1), (s * s).sum(1), (c * s).sum(1)
        cy, sy = c @ y, s @ y
        det = cc * ss - cs * cs
        ok = det > 1e-6 * np.maximum(cc * ss, 1e-12)
        safe = np.where(ok, det, 1.0)
        ca, sb = np.where(ok, (ss * cy - cs * sy) / safe, 0.0), np.where(ok, (cc * sy - cs * cy) / safe, 0.0)
        return ca * cy + sb * sy, ca, sb

    freqs = np.arange(f_min, f_max, step)
    score = np.zeros(len(freqs))
    for y, ratio in zip(ys, ratios):
        if float(y @ y) > 0:
            score += project(freqs, y, ratio)[0]
    k = int(np.argmax(score))
    freq = float(freqs[k])
    if 0 < k < len(score) - 1:  # parabolic peak interpolation
        p0, p1, p2 = score[k - 1], score[k], score[k + 1]
        den = p0 - 2 * p1 + p2
        if den < 0:
            freq += float(np.clip(0.5 * (p0 - p2) / den, -0.5, 0.5)) * step
    for i, (y, ratio) in enumerate(zip(ys, ratios)):
        _, ca, sb = project(np.array([freq]), y, ratio)
        amps[i] = math.hypot(float(ca[0]), float(sb[0]))
    return freq, float(np.clip(score[k] / energy, 0.0, 1.0)), amps
