"""Shot-boundary and transition detection.

Algorithm
---------
1. **Thumbnails.** Every frame is reduced to a small blurred luma thumbnail
   (long side ~64 px). Pixels that never change over the clip (letterbox bars,
   burned-in static titles) are masked out, and each thumbnail is z-normalised
   over the active pixels. Blending a frame toward a solid colour (flashes,
   fades) is an affine change of intensity, so the normalised thumbnail is
   unchanged by it. A frame is **flat** (no content: a solid flash or black
   frame) when its robust contrast is negligible, or when it is low and its
   normalised structure matches neither neighbour, which is the case for
   compression noise. A dimmed frame of a low-contrast night shot keeps its
   structure and stays content.
2. **Content similarity** of two frames = mean of the global correlation and the
   median per-block SSIM contrast-structure term of their normalised thumbnails.
   The block median ignores local changes such as a caption popping in. When two
   frames disagree, the similarity is re-evaluated after motion compensation:
   phase-correlation translation, then a small search over zoom factors and
   rotations, so pans, shakes, zoom punches and spins still match.
3. **Hard cuts.** ``score[t] = D(previous content frame, t) * (w + (1 - w) * colour
   change)`` with ``D = 1 - similarity`` and the colour change taken from the HSV
   histogram distance. The score is thresholded adaptively (rolling median +
   ``k * MAD`` over ~1 s, clamped to ``[floor, ceiling]``), kept only at local
   maxima, and confirmed by a content check: the frames on either side of the
   candidate must also differ (``t-2`` vs ``t`` and ``t-1`` vs ``t+1``, skipping
   flat frames). A one-frame glitch or in-shot flash fails this check.
4. **Luma excursions.** Runs of flat frames are grown outward over "ramp" frames
   that follow the blend model, ``x' = g * x + (1 - g) * C`` (contrast falls and
   the mean moves toward the flat colour ``C``). The content before and after
   the excursion decides the outcome. If it is the same, this is an in-shot
   flash or dip: no boundary, and any cut candidate inside is discarded. If it
   differs, this is a ``flash`` (short) or a ``fade_black``/``fade_white``
   (long or slowly ramped) transition. Frames that still show structure are
   assigned to the shot they resemble and the content-less ones are split
   evenly (an odd one out starts the incoming shot, as in the compiler whose
   flash envelope peaks on the boundary frame). Excursions touching the first or
   last frame, and ramps from a dimmed first / to a dimmed last frame, become
   fade-in / fade-out transitions. Long ramp-less flat runs become solid shots.
   A hard cut flanked by partial blend ramps (no flat frame) is a flash or fade
   as well.
5. **Dissolves.** Long-lag distances ``D(t-k, t+k)`` (``k`` at a few scales) flag
   gradual change away from known boundaries. A candidate is verified with a
   linear-blend test: each frame is regressed on the two anchor frames,
   ``x ~ a*A + b*B + c``, where the anchors are extrapolated along their own
   shot's motion (constant-velocity translation from phase correlation), so
   dissolves between panning shots still fit. A piecewise-linear ramp is fitted
   to ``alpha = b / (a + b)`` and the extent is then grown or shrunk until
   anchors next to the blend frames are pure. A dissolve needs a ramp-shaped
   monotonic alpha, a fit that clearly beats either anchor alone, and anchors
   that stay different after motion compensation. This rejects pans and zooms.
   The boundary is the middle of the blend frames.
6. **Cut styles.** Remaining hard cuts are compared with calm frames of the same
   shot (never with other shots): median block gradient energy (sharpness) and
   the horizontal/vertical gradient balance. A sudden blur with a directional
   balance shift right at the cut is a ``whip``. An isotropic blur whose frame
   pair shows a large scale change is a ``zoom``, one with a large rotation a
   ``spin`` (wide scale x rotation hypothesis search). Red/blue channel
   misalignment that the shot does not otherwise have is a ``glitch``. Styled
   cuts get a lower confidence. Without strong evidence a boundary stays ``cut``.

Every threshold is a field of :class:`ShotParams` and can be overridden with
``AnalysisConfig.params["shots"][<field name>]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import cv2
import numpy as np

from editast.analyzers.types import ShotSegmentation, ShotSpan, TransitionSpan
from editast.ast.nodes import TransitionKind
from editast.context import AnalysisContext

NAME = "shots"

# Motion hypotheses tried (in order) when two frames disagree: (zoom factor, rotation deg).
# Each is combined with a phase-correlation translation estimate. Every extra hypothesis
# also raises the best chance match between unrelated frames, so combined zoom + rotation
# is only tried as the pairing of the best zoom and the best rotation (see best_motion).
_MOTION_SEARCH: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (1.1, 0.0), (1 / 1.1, 0.0), (1.22, 0.0), (1 / 1.22, 0.0), (1.4, 0.0), (1 / 1.4, 0.0),
    (1.0, 8.0), (1.0, -8.0), (1.0, 16.0), (1.0, -16.0),
)
# Wider search used only to name the style of a blurred cut (zoom / spin transitions).
_STYLE_SEARCH: tuple[tuple[float, float], ...] = tuple(
    (s, a)
    for s in (1.0, 1.15, 1 / 1.15, 1.35, 1 / 1.35, 1.6, 1 / 1.6, 2.0, 1 / 2.0)
    for a in (0.0, 10.0, -10.0, 22.0, -22.0, 35.0, -35.0, 50.0, -50.0)
)


@dataclass(frozen=True)
class ShotParams:
    """Tunable thresholds. Override any field via ``config.params["shots"][name]``."""

    thumb_long_side: int = 64  # px, long side of comparison thumbnails
    block: int = 8  # px, block size for the per-block similarity
    ssim_c: float = 0.02  # stabiliser of the block contrast-structure term (z units)
    static_std: float = 0.012  # temporal std below which a pixel is static (masked)
    noise_std: float = 0.004  # contrast floor used when z-normalising thumbnails
    flat_abs: float = 0.006  # robust contrast (0..1) below which a frame is always flat
    flat_std: float = 0.025  # below this a frame is flat unless it shares a neighbour's structure
    structure_min: float = 0.5  # similarity to a neighbour that keeps a low-contrast frame content
    content_std: float = 0.005  # below this a frame is too flat to tell which shot it shows
    assign_margin: float = 0.2  # similarity margin needed to assign a faded frame to a shot
    match_good: float = 0.8  # similarity at which the motion search stops
    max_shift: float = 0.45  # largest translation tried, fraction of thumbnail size
    align_credit: tuple[float, float] = (0.3, 0.55)  # aligned similarity: no credit .. full credit
    min_shot_frames: int = 2
    bridge_seconds: float = 1.0  # longest flat gap bridged by the cut score
    window_seconds: float = 1.0  # rolling window of the adaptive threshold
    mad_k: float = 4.0
    threshold_floor: float = 0.3
    threshold_ceiling: float = 0.6
    hist_weight: float = 0.5  # share of the score independent of colour change
    hist_saturation: float = 0.3  # histogram distance counted as a full colour change
    content_min: float = 0.2  # min distance between the two sides of a cut ...
    content_ratio: float = 0.35  # ... and at least this share of the cut's own distance
    same_content_max: float = 0.35  # distance below which two frames show the same shot
    ramp_contrast_step: float = 0.04  # min relative contrast drop per ramp frame
    ramp_luma_tol: float = 0.4  # allowed deviation from the blend model (share of the step)
    ramp_max_frames: int = 30
    flash_max_frames: int = 6  # longer excursions are fades
    fade_min_ramp: int = 3  # ramps at least this long make an excursion a fade
    edge_fade_min_ramp: int = 2  # ramp needed for a fade-in/out at the clip ends
    edge_fade_min_depth: float = 0.5  # min contrast loss of a fade that never reaches a flat frame
    solid_shot_seconds: float = 0.4  # ramp-less flat runs this long are shots of their own
    flash_min_alpha: float = 0.3  # min blend amount for a partial flash around a cut
    dissolve_lags: tuple[int, ...] = (4, 8, 16)
    dissolve_min_frames: int = 3
    dissolve_max_frames: int = 30
    dissolve_min_distance: float = 0.5
    dissolve_min_r2: float = 0.75  # mean R^2 of the blend fit over the blend frames
    dissolve_min_gain: float = 0.1  # blend fit must beat the best single anchor by this
    dissolve_max_err: float = 0.12  # rms deviation of alpha from a linear ramp
    classify_cuts: bool = True
    style_ref_seconds: float = 0.5  # calm frames of the same shot used as reference
    style_max_blur: float = 0.55  # gradient energy ratio vs. the shot's calm frames
    whip_min_aniso: float = 0.7  # |log| shift of the horizontal/vertical gradient balance
    whip_min_shift: float = 0.15  # or translation between the blurred frames, fraction of size
    zoom_min_scale: float = 1.12  # scale change between the blurred frames
    spin_min_angle: float = 8.0  # degrees between the blurred frames
    style_min_gain: float = 0.1  # the style hypothesis must beat no motion by this similarity
    glitch_min_shift: float = 0.012  # red/blue misalignment, fraction of width

    @classmethod
    def from_context(cls, ctx: AnalysisContext) -> "ShotParams":
        defaults = cls()
        values = {f.name: ctx.config.p(NAME, f.name, getattr(defaults, f.name)) for f in fields(cls)}
        values["dissolve_lags"] = tuple(int(k) for k in values["dissolve_lags"])
        values["align_credit"] = tuple(float(v) for v in values["align_credit"])
        return cls(**values)


@dataclass
class _Boundary:
    """A boundary candidate before the shot partition is fixed."""

    frame: int
    kind: TransitionKind
    f0: int
    f1: int
    strength: float  # evidence used to rank conflicting candidates
    priority: int  # 2 = explicit transition (flash / fade / dissolve / solid shot), 1 = hard cut
    confidence: float = 1.0
    color: Optional[tuple[int, int, int]] = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Excursion:
    """Frames ``[f0, f1)`` blended toward a solid colour; ``[core0, core1)`` are flat."""

    f0: int
    f1: int
    core0: int
    core1: int

    @property
    def ramp_in(self) -> int:
        return self.core0 - self.f0

    @property
    def ramp_out(self) -> int:
        return self.f1 - self.core1


@dataclass
class _EdgeFade:
    kind: TransitionKind
    f0: int
    f1: int
    color: tuple[int, int, int]
    confidence: float


@dataclass
class _BlendFit:
    """Per-frame linear-blend regression of frames ``frames`` on two (moving) anchors."""

    frames: np.ndarray
    alpha: np.ndarray  # b / (a + b); NaN where the fit has no blend energy
    r2: np.ndarray  # R^2 of the two-anchor fit
    single: np.ndarray  # best R^2 of one anchor alone


# --------------------------------------------------------------------------- helpers


def _rolling(x: np.ndarray, win: int, fn: Any) -> np.ndarray:
    """Centered rolling ``fn`` (e.g. ``np.median``) over an odd window, reflect-padded."""
    half = win // 2
    padded = np.pad(x, half, mode="reflect" if x.size > 1 else "edge")
    return fn(np.lib.stride_tricks.sliding_window_view(padded, win), axis=1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of every run of True values."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _block_sum(x: np.ndarray, bs: int) -> np.ndarray:
    """Sum over ``bs x bs`` blocks of the last two axes (trailing partial blocks dropped)."""
    h, w = x.shape[-2:]
    by, bx = h // bs, w // bs
    x = x[..., : by * bs, : bx * bs]
    return x.reshape(*x.shape[:-2], by, bs, bx, bs).sum(axis=(-3, -1))


def _shift(img: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``img`` translated by ``d = (dx, dy)`` px and the mask of pixels that came from inside."""
    h, w = img.shape
    m = np.float32([[1.0, 0.0, d[0]], [0.0, 1.0, d[1]]])
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    valid = cv2.warpAffine(np.ones((h, w), np.float32), m, (w, h), flags=cv2.INTER_NEAREST) > 0.5
    return out, valid


def _regress(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Fit ``x ~ ca*a + cb*b + c``; return ``(alpha, r2, best single-anchor r2)``.

    ``alpha = cb / (ca + cb)`` is NaN when the fit has no blend energy; the
    single-anchor R^2 only counts positive correlation.
    """
    xc, ac, bc = (v.astype(np.float64) - float(v.mean()) for v in (x, a, b))
    sxx, saa, sbb = float(xc @ xc), float(ac @ ac), float(bc @ bc)
    sab, sxa, sxb = float(ac @ bc), float(xc @ ac), float(xc @ bc)
    if min(sxx, saa, sbb) <= 1e-9:
        return float("nan"), 0.0, 0.0
    single = max(max(sxa, 0.0) ** 2 / (sxx * saa), max(sxb, 0.0) ** 2 / (sxx * sbb))
    det = saa * sbb - sab * sab
    if det <= 1e-9 * saa * sbb:  # collinear anchors
        return float("nan"), single, single
    ca = (sxa * sbb - sxb * sab) / det
    cb = (sxb * saa - sxa * sab) / det
    r2 = (ca * sxa + cb * sxb) / sxx
    tot = ca + cb
    alpha = cb / tot if abs(tot) > 0.2 else float("nan")
    return alpha, r2, single


def _fit_ramp(alpha: np.ndarray, min_len: int, max_len: int) -> tuple[int, int, float]:
    """Best ``[i0, i1)`` (indices into ``alpha``) such that ``alpha`` is 0 before,
    rises linearly through ``i0..i1-1`` and is 1 from ``i1`` on. The ramp must not
    touch the ends. Returns ``(i0, i1, rms error)``."""
    m = alpha.size
    j = np.arange(m)
    best = (0, 0, np.inf)
    for i0 in range(2, m - 2):
        i1 = np.arange(i0 + min_len, min(i0 + max_len, m - 2) + 1)
        if i1.size == 0:
            continue
        model = np.clip((j[None, :] - (i0 - 1)) / (i1[:, None] - i0 + 1), 0.0, 1.0)
        err = np.sqrt(np.mean((model - alpha[None, :]) ** 2, axis=1))
        k = int(np.argmin(err))
        if err[k] < best[2]:
            best = (i0, int(i1[k]), float(err[k]))
    return best


# --------------------------------------------------------------------------- matcher


class _Matcher:
    """Content similarity between frames (z-normalised thumbnails, motion compensated)."""

    def __init__(self, z: np.ndarray, active: np.ndarray, params: ShotParams) -> None:
        self.z = z
        self.params = params
        self.bs = params.block
        self.active = active.astype(np.float32)
        self.n_active = max(float(self.active.sum()), 1.0)
        h, w = z.shape[1:]
        self.size = (w, h)
        self.center = (w / 2.0, h / 2.0)
        self.block_valid = _block_sum(self.active, self.bs) >= 0.5 * self.bs * self.bs
        if self.block_valid.sum() < 4:
            self.block_valid[:] = True
        self.window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self._cache: dict[tuple[int, int], float] = {}

    def identity(self, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
        """Similarity of frame pairs ``(ia[k], ib[k])`` without motion compensation."""
        out = np.empty(len(ia), np.float32)
        c, bs, area = self.params.ssim_c, self.bs, float(self.bs * self.bs)
        for s in range(0, len(ia), 512):
            za, zb = self.z[ia[s:s + 512]], self.z[ib[s:s + 512]]
            ncc = (za * zb).sum(axis=(1, 2)) / self.n_active
            ma, mb = _block_sum(za, bs) / area, _block_sum(zb, bs) / area
            va = _block_sum(za * za, bs) / area - ma * ma
            vb = _block_sum(zb * zb, bs) / area - mb * mb
            cov = _block_sum(za * zb, bs) / area - ma * mb
            cs = (2.0 * cov + c) / (va + vb + c)
            out[s:s + 512] = 0.5 * ncc + 0.5 * np.median(cs[:, self.block_valid], axis=1)
        return out

    def similarity(self, a: int, b: int, base: Optional[float] = None) -> float:
        """Best similarity of frames ``a`` and ``b`` over the motion hypotheses."""
        key = (a, b) if a <= b else (b, a)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        plain = float(self.identity(np.array([a]), np.array([b]))[0]) if base is None else float(base)
        aligned = plain
        if plain < self.params.match_good:
            aligned = max(plain, self.best_motion(a, b, _MOTION_SEARCH, stop=self.params.match_good)[0])
        # The best of many hypotheses also lifts unrelated frames a little (~0.2-0.4), so
        # the alignment gain is credited only as the aligned match becomes convincing.
        lo, hi = self.params.align_credit
        credit = float(np.clip((aligned - lo) / (hi - lo), 0.0, 1.0))
        sim = plain + credit * max(0.0, aligned - plain)
        self._cache[key] = sim
        return sim

    def distance(self, a: int, b: int) -> float:
        return float(np.clip(1.0 - self.similarity(a, b), 0.0, 1.0))

    def best_motion(
        self, a: int, b: int, search: tuple[tuple[float, float], ...], stop: float = np.inf
    ) -> tuple[float, float, float, float, float]:
        """Best ``(similarity, scale, angle, dx, dy)`` mapping frame ``a`` onto ``b``.

        ``dx, dy`` are fractions of the thumbnail width / height. The search ends
        early once a hypothesis reaches ``stop``. When both a pure zoom and a pure
        rotation beat no motion, their combination is tried as well.
        """
        best = (-1.0, 1.0, 0.0, 0.0, 0.0)
        still = zoom = spin = -1.0
        best_scale, best_angle = 1.0, 0.0
        for scale, angle in search:
            sim, dx, dy = self._try(a, b, scale, angle, best[0])
            if sim > best[0]:
                best = (sim, scale, angle, dx, dy)
                if sim >= stop:
                    return best
            if scale == 1.0 and angle == 0.0:
                still = sim
            elif angle == 0.0 and sim > zoom:
                zoom, best_scale = sim, scale
            elif scale == 1.0 and sim > spin:
                spin, best_angle = sim, angle
        if min(zoom, spin) > still and (best_scale, best_angle) not in search:
            sim, dx, dy = self._try(a, b, best_scale, best_angle, best[0])
            if sim > best[0]:
                best = (sim, best_scale, best_angle, dx, dy)
        return best

    def _try(self, a: int, b: int, scale: float, angle: float, beat: float) -> tuple[float, float, float]:
        """Similarity of ``b`` and ``a`` zoomed / rotated about the centre and then shifted by
        phase correlation; returns ``(similarity, dx, dy)`` with shifts as fractions of size."""
        za, zb = self.z[a], self.z[b]
        w, h = self.size
        m = cv2.getRotationMatrix2D(self.center, angle, scale)
        src = za.copy() if (scale == 1.0 and angle == 0.0) else cv2.warpAffine(za, m, (w, h))
        # phaseCorrelate applies the window to its inputs in place: pass copies.
        (dx, dy), _ = cv2.phaseCorrelate(src, zb.copy(), self.window)
        if abs(dx) > self.params.max_shift * w or abs(dy) > self.params.max_shift * h:
            return -1.0, 0.0, 0.0
        m[0, 2] += dx
        m[1, 2] += dy
        warped = cv2.warpAffine(za, m, (w, h))
        mask = cv2.warpAffine(self.active, m, (w, h), flags=cv2.INTER_NEAREST) * self.active
        return self._masked(warped, zb, mask, beat=beat), dx / w, dy / h

    def _masked(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray, beat: float = -np.inf) -> float:
        """Similarity restricted to the ``mask`` pixels (re-normalised over them).

        When even a perfect block term could not lift the result above ``beat``,
        that upper bound is returned without computing the block term.
        """
        wsum = float(mask.sum())
        if wsum < 0.35 * mask.size:
            return -1.0
        da = (a - (a * mask).sum() / wsum) * mask
        db = (b - (b * mask).sum() / wsum) * mask
        sa, sb = np.sqrt((da * da).sum() / wsum), np.sqrt((db * db).sum() / wsum)
        if sa < 1e-6 or sb < 1e-6:
            return -1.0
        da, db = da / sa, db / sb
        ncc = float((da * db).sum() / wsum)
        if 0.5 * ncc + 0.5 <= beat:
            return 0.5 * ncc + 0.5
        bs = self.bs
        bw, sa_, sb_, saa, sbb, sab = _block_sum(np.stack([mask, da, db, da * da, db * db, da * db]), bs)
        ok = (bw >= 0.75 * bs * bs) & self.block_valid
        if ok.sum() < 4:
            return ncc
        bw = np.maximum(bw, 1.0)
        ma, mb = sa_ / bw, sb_ / bw
        va = saa / bw - ma * ma
        vb = sbb / bw - mb * mb
        cov = sab / bw - ma * mb
        c = self.params.ssim_c
        cs = (2.0 * cov + c) / (va + vb + c)
        return 0.5 * ncc + 0.5 * float(np.median(cs[ok]))


# --------------------------------------------------------------------------- detector


class _Detector:
    def __init__(self, ctx: AnalysisContext, params: ShotParams) -> None:
        self.ctx = ctx
        self.p = params
        self.n = ctx.n_frames
        self.fps = float(ctx.fps)
        self._thumbnails()
        content = ~self.flat
        idx = np.arange(self.n)
        last = np.maximum.accumulate(np.where(content, idx, -1))
        self.prev_c = np.concatenate([[-1], last[:-1]])  # last content frame < t
        first = np.minimum.accumulate(np.where(content, idx, self.n)[::-1])[::-1]
        self.next_c = np.concatenate([first[1:], [self.n]])  # first content frame > t
        self.cd = np.zeros(self.n)  # content distance to the previous content frame
        self.score = np.zeros(self.n)
        self.threshold = np.full(self.n, params.threshold_floor)
        self.blocked = np.zeros(self.n, bool)  # frames near known boundaries (set before dissolves)
        self._texture_cache: dict[int, tuple[float, float]] = {}

    # ----------------------------------------------------------------- features

    def _thumbnails(self) -> None:
        p, gray = self.p, self.ctx.gray
        n, h, w = gray.shape
        bs = p.block
        scale = p.thumb_long_side / max(h, w)
        tw = max(bs, int(round(w * scale / bs)) * bs)
        th = max(bs, int(round(h * scale / bs)) * bs)
        thumbs = np.empty((n, th, tw), np.float32)
        for i in range(n):
            small = cv2.resize(gray[i], (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
            thumbs[i] = cv2.GaussianBlur(small, (0, 0), 0.8)
        thumbs /= 255.0
        # Static pixels (bars, fixed titles) carry no shot information.
        active = np.ones((th, tw), bool)
        if n >= 10:
            sample = thumbs[np.linspace(0, n - 1, min(n, 300)).astype(int)]
            moving = sample.std(axis=0) >= p.static_std
            if moving.mean() >= 0.3:
                active = moving
        act = active.astype(np.float32)
        na = float(act.sum())
        mean = (thumbs * act).sum(axis=(1, 2)) / na
        std = np.sqrt((((thumbs - mean[:, None, None]) ** 2) * act).sum(axis=(1, 2)) / na)
        # Robust level / contrast (median, 20-80 percentile spread in std units). A blend
        # toward a solid colour maps every percentile exactly, and captions drawn on top
        # of a flash (up to ~20% of the pixels) do not move them.
        q20, q50, q80 = np.percentile(thumbs[:, active], [20.0, 50.0, 80.0], axis=1)
        self.level = q50
        self.contrast = (q80 - q20) / 1.683
        z = (thumbs - mean[:, None, None]) / np.maximum(std, p.noise_std)[:, None, None]
        self.matcher = _Matcher((z * act).astype(np.float32), active, p)
        self.thumbs = thumbs
        self.active = active
        self.active_full = cv2.resize(act, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        self.flat = self._flat_frames()

    def _flat_frames(self) -> np.ndarray:
        """Frames without content: negligible contrast, or low contrast and no structure in
        common with either neighbour (a dimmed frame keeps its neighbours' structure)."""
        p, n = self.p, self.n
        flat = self.contrast < p.flat_abs
        cand = np.flatnonzero((self.contrast < p.flat_std) & ~flat)
        if cand.size == 0 or n < 2:
            return flat
        sim = np.full(cand.size, -1.0)
        for off in (-1, 1):
            nb = cand + off
            ok = (nb >= 0) & (nb < n)
            sim[ok] = np.maximum(sim[ok], self.matcher.identity(cand[ok], nb[ok]))
        for k in np.flatnonzero(sim < p.structure_min):
            t = int(cand[k])
            for nb in (t - 1, t + 1):
                if 0 <= nb < n and not flat[nb]:
                    sim[k] = max(sim[k], self.matcher.similarity(t, nb))
        flat[cand[sim < p.structure_min]] = True
        return flat

    def _content_distance(self) -> None:
        """Cut score from content + colour change to the previous content frame."""
        p, n = self.p, self.n
        max_gap = max(1, int(round(p.bridge_seconds * self.fps)))
        t = np.arange(n)
        ts = t[(~self.flat) & (self.prev_c >= 0) & (t - self.prev_c <= max_gap)]
        if ts.size == 0:
            return
        ps = self.prev_c[ts]
        sim = self.matcher.identity(ps, ts).astype(np.float64)
        for k in np.flatnonzero(sim < p.match_good):
            sim[k] = self.matcher.similarity(int(ps[k]), int(ts[k]), base=sim[k])
        self.cd[ts] = np.clip(1.0 - sim, 0.0, 1.0)
        hist = self.ctx.features.hist
        hd = 1.0 - np.minimum(hist[ps], hist[ts]).sum(axis=1)
        colour = np.minimum(1.0, np.clip(hd, 0.0, 1.0) / p.hist_saturation)
        self.score[ts] = self.cd[ts] * (p.hist_weight + (1.0 - p.hist_weight) * colour)
        win = max(5, int(round(p.window_seconds * self.fps)) | 1)
        if n >= 3:
            med = _rolling(self.score, win, np.median)
            mad = _rolling(np.abs(self.score - med), win, np.median)
            self.threshold = np.clip(med + p.mad_k * 1.4826 * mad, p.threshold_floor, p.threshold_ceiling)

    # ----------------------------------------------------------------- excursions

    def _blends_toward(self, src: int, dst: int, c: float) -> bool:
        """Whether frame ``dst`` looks like frame ``src`` blended further toward luma ``c``."""
        s, m = self.contrast, self.level
        if s[src] <= 1e-6:
            return False
        g = s[dst] / s[src]
        if g > 1.0 - self.p.ramp_contrast_step:
            return False
        pred = g * m[src] + (1.0 - g) * c
        return abs(m[dst] - pred) <= self.p.ramp_luma_tol * (1.0 - g) * abs(c - m[src]) + 0.01

    def _same_step(self, a: int, b: int) -> bool:
        """Adjacent frames ``a < b`` show the same content (unreliable when nearly flat)."""
        if min(self.contrast[a], self.contrast[b]) < 3.0 * self.p.flat_abs:
            return True
        return self.prev_c[b] == a and self.cd[b] <= self.p.same_content_max

    def _ramp_left(self, end: int, c: float, lo: int, limit: int) -> int:
        """Number of frames ``end, end-1, ...`` (not below ``lo``) each blended further than its predecessor."""
        k, j = 0, end
        while k < limit and j - 1 >= lo and not self.flat[j] and not self.flat[j - 1] \
                and self._blends_toward(j - 1, j, c) and self._same_step(j - 1, j):
            k += 1
            j -= 1
        return k

    def _ramp_right(self, start: int, c: float, hi: int, limit: int) -> int:
        """Number of frames ``start, start+1, ...`` (below ``hi - 1``) each blended further than its successor."""
        k, j = 0, start
        while k < limit and j + 1 < hi and not self.flat[j] and not self.flat[j + 1] \
                and self._blends_toward(j + 1, j, c) and self._same_step(j, j + 1):
            k += 1
            j += 1
        return k

    def _excursions(self) -> list[_Excursion]:
        out: list[_Excursion] = []
        for a0, a1 in _runs(self.flat):
            c = float(np.median(self.level[a0:a1]))
            f0 = a0 - self._ramp_left(a0 - 1, c, 0, self.p.ramp_max_frames) if a0 > 0 else a0
            f1 = a1 + self._ramp_right(a1, c, self.n, self.p.ramp_max_frames) if a1 < self.n else a1
            if out and f0 <= out[-1].f1:  # touching excursions (flicker) merge
                prev = out[-1]
                out[-1] = _Excursion(prev.f0, max(prev.f1, f1), prev.core0, a1)
            else:
                out.append(_Excursion(f0, f1, a0, a1))
        return out

    def _core_color(self, f0: int, f1: int) -> tuple[int, int, int]:
        """Median colour of the flattest frames in ``[f0, f1)`` (active pixels only)."""
        core = np.arange(f0, f1)
        c = self.contrast[core]
        pick = core[c <= 1.5 * c.min() + 1e-3]
        rgb = np.median(self.ctx.frames[pick][:, self.active_full].reshape(-1, 3), axis=0)
        return tuple(int(round(v)) for v in rgb)  # type: ignore[return-value]

    @staticmethod
    def _dip_kind(level: float) -> TransitionKind:
        return "fade_white" if level >= 0.5 else "fade_black"

    def _split_excursion(self, ex: _Excursion, a: int, b: int) -> int:
        """First frame of the incoming shot inside an excursion between anchors ``a`` and ``b``.

        Frames that still show structure are assigned to the anchor they resemble;
        the frames without usable content in between are split evenly, an odd one
        out going to the incoming shot.
        """
        last_a, first_b = ex.f0 - 1, b
        for j in range(ex.f0, ex.f1):
            if self.contrast[j] < self.p.content_std:
                continue
            sa, sb = self.matcher.similarity(a, j), self.matcher.similarity(b, j)
            if abs(sb - sa) < self.p.assign_margin:
                continue
            if sb > sa:
                first_b = j
                break
            last_a = j
        unassigned0 = last_a + 1
        return unassigned0 + (first_b - unassigned0) // 2

    def _edge_ramp(self, at_start: bool) -> Optional[_EdgeFade]:
        """A fade-in from (fade-out to) a colour that the clip never fully reaches."""
        p, n = self.p, self.n
        best: Optional[_EdgeFade] = None
        for c in (0.0, 1.0):
            if at_start:
                k = self._ramp_right(0, c, n, p.ramp_max_frames)
                edge, clean, f0, f1 = 0, k, 0, k
            else:
                k = self._ramp_left(n - 1, c, 0, p.ramp_max_frames)
                edge, clean, f0, f1 = n - 1, n - 1 - k, n - k, n
            if k < p.fade_min_ramp or self.contrast[clean] <= 1e-6:
                continue
            depth = 1.0 - self.contrast[edge] / self.contrast[clean]
            if depth < p.edge_fade_min_depth:
                continue
            conf = float(np.clip(0.3 + 0.5 * depth, 0.5, 0.8))
            if best is None or conf > best.confidence:
                rgb = (255, 255, 255) if c == 1.0 else (0, 0, 0)
                best = _EdgeFade(self._dip_kind(c), f0, f1, rgb, conf)
        return best

    def _resolve_excursions(
        self, events: list[_Excursion]
    ) -> tuple[list[_Boundary], np.ndarray, Optional[_EdgeFade], Optional[_EdgeFade]]:
        """Turn excursions into transitions; returns ``(boundaries, suppressed, fade_in, fade_out)``.

        ``suppressed[t]`` marks frames where hard-cut candidates are not allowed.
        """
        p, n = self.p, self.n
        solid = max(p.min_shot_frames, int(round(p.solid_shot_seconds * self.fps)))
        records: list[_Boundary] = []
        suppressed = np.zeros(n + 1, bool)
        fade_in = fade_out = None
        for ex in events:
            core_len = ex.core1 - ex.core0
            if ex.f0 == 0 and ex.f1 == n:
                suppressed[:] = True
                continue
            color = self._core_color(ex.core0, ex.core1)
            level = float(np.median(self.level[ex.core0:ex.core1]))
            if ex.f0 == 0 or ex.f1 == n:
                at_start = ex.f0 == 0
                ramp = ex.ramp_out if at_start else ex.ramp_in
                suppressed[ex.f0:ex.f1 + 1] = True
                if ramp >= p.edge_fade_min_ramp:
                    conf = float(np.clip(0.5 + 0.1 * ramp, 0.5, 1.0))
                    fade = _EdgeFade(self._dip_kind(level), ex.f0, ex.f1, color, conf)
                    if at_start:
                        fade_in = fade
                    else:
                        fade_out = fade
                elif core_len >= solid:
                    b = ex.core1 if at_start else ex.core0
                    records.append(_Boundary(b, "cut", b, b, 1.0, 2, 0.8, params={"solid": color}))
                continue
            if ex.ramp_in == 0 and ex.ramp_out == 0 and core_len >= solid:
                # Cut to a solid frame held long enough to be a shot of its own.
                for b in (ex.core0, ex.core1):
                    records.append(_Boundary(b, "cut", b, b, 1.0, 2, 0.8, params={"solid": color}))
                    suppressed[b + 1:b + p.min_shot_frames] = True
                suppressed[ex.core0 + 1:ex.core1] = True
                continue
            a, b = ex.f0 - 1, ex.f1
            dist = self.matcher.distance(a, b)
            suppressed[ex.f0:ex.f1 + 1] = True
            if dist < p.same_content_max:
                continue  # in-shot flash / dip: reported by the look analyzer
            boundary = self._split_excursion(ex, a, b)
            long_ramp = max(ex.ramp_in, ex.ramp_out) >= p.fade_min_ramp
            is_fade = (ex.f1 - ex.f0) > p.flash_max_frames or long_ramp
            kind: TransitionKind = self._dip_kind(level) if is_fade else "flash"
            conf = float(np.clip(dist / (2.0 * p.same_content_max), 0.5, 1.0))
            records.append(_Boundary(
                boundary, kind, ex.f0, ex.f1, 1.0 + dist, 2, conf, color,
                {"ramp_in": ex.ramp_in, "ramp_out": ex.ramp_out, "hold": core_len},
            ))
            self.score[boundary] = max(self.score[boundary], dist)
        if fade_in is None and not suppressed[0]:
            fade_in = self._edge_ramp(at_start=True)
            if fade_in is not None:
                suppressed[:fade_in.f1 + 1] = True
        if fade_out is None and not suppressed[n - 1]:
            fade_out = self._edge_ramp(at_start=False)
            if fade_out is not None:
                suppressed[fade_out.f0:] = True
        return records, suppressed[:n], fade_in, fade_out

    # ----------------------------------------------------------------- hard cuts

    def _hard_cuts(self, suppressed: np.ndarray) -> list[_Boundary]:
        p, n = self.p, self.n
        if n < 3:
            return []
        r = max(1, p.min_shot_frames - 1)
        local_max = self.score >= _rolling(self.score, 2 * r + 1, np.max)
        cand = np.flatnonzero((self.score > self.threshold) & local_max & ~self.flat & ~suppressed)
        out: list[_Boundary] = []
        for t in cand.tolist():
            p1 = int(self.prev_c[t])
            if p1 < 0:
                continue
            checks = []
            if self.next_c[t] < n:
                checks.append((p1, int(self.next_c[t])))
            if self.prev_c[p1] >= 0:
                checks.append((int(self.prev_c[p1]), t))
            # Both sides must stay apart (relative to the spike itself): a one-frame
            # glitch or flash has t-1 ~ t+1 although t differs from both.
            need = max(p.content_min, p.content_ratio * float(self.cd[t]))
            if checks and min(self.matcher.distance(a, b) for a, b in checks) < need:
                continue
            thr = float(self.threshold[t])
            conf = float(np.clip(0.5 + 0.5 * (self.score[t] / thr - 1.0), 0.5, 1.0))
            out.append(_Boundary(t, "cut", t, t, float(self.score[t]), 1, conf,
                                 params={"score": round(float(self.score[t]), 3), "threshold": round(thr, 3)}))
        return out

    # ----------------------------------------------------------------- dissolves

    def _velocity(self, f: int, g: int) -> np.ndarray:
        """Thumbnail translation (px) of the content from frame ``f`` to frame ``g``."""
        if not (0 <= f < self.n and 0 <= g < self.n) or self.blocked[f] or self.blocked[g]:
            return np.zeros(2)
        # phaseCorrelate windows its inputs in place: pass copies.
        (dx, dy), response = cv2.phaseCorrelate(self.thumbs[f].copy(), self.thumbs[g].copy(), self.matcher.window)
        w, h = self.matcher.size
        if response < 0.05 or abs(dx) > self.p.max_shift * w or abs(dy) > self.p.max_shift * h:
            return np.zeros(2)
        return np.array([dx, dy])

    def _blend_fit(self, a: int, b: int, frames: np.ndarray) -> Optional[_BlendFit]:
        """Regress ``frames`` on anchors ``a`` and ``b`` moved along their shot's motion.

        Anchor ``a`` is extrapolated with the velocity from ``a-1`` to ``a`` and anchor
        ``b`` with the velocity from ``b+1`` back to ``b`` (zero when those frames are blocked).
        """
        va, vb = self._velocity(a - 1, a), self._velocity(b + 1, b)
        alpha, r2, single = (np.full(frames.size, np.nan) for _ in range(3))
        for i, t in enumerate(frames.tolist()):
            pa, ma = _shift(self.thumbs[a], va * (t - a))
            pb, mb = _shift(self.thumbs[b], vb * (b - t))
            mask = ma & mb & self.active
            if mask.mean() < 0.3:
                return None
            alpha[i], r2[i], single[i] = _regress(self.thumbs[t][mask], pa[mask], pb[mask])
        return _BlendFit(frames, alpha, r2, single)

    def _warped_r2(self, f: int, anchors: tuple[int, ...]) -> float:
        """Best R^2 of frame ``f`` explained by one anchor under a wide similarity-motion search."""
        w, h = self.matcher.size
        x = self.thumbs[f]
        best = 0.0
        for a in anchors:
            _, scale, angle, dx, dy = self.matcher.best_motion(a, f, _STYLE_SEARCH)
            m = cv2.getRotationMatrix2D(self.matcher.center, angle, scale)
            m[0, 2] += dx * w
            m[1, 2] += dy * h
            warped = cv2.warpAffine(self.thumbs[a], m, (w, h))
            inside = cv2.warpAffine(self.matcher.active, m, (w, h), flags=cv2.INTER_NEAREST) > 0.5
            mask = inside & self.active
            if mask.mean() < 0.3:
                continue
            xc = x[mask] - x[mask].mean()
            vc = warped[mask] - warped[mask].mean()
            den = float(np.sqrt((xc @ xc) * (vc @ vc)))
            if den > 1e-9:
                best = max(best, max(0.0, float(xc @ vc) / den) ** 2)
        return best

    def _find_dissolves(self, blocked: np.ndarray) -> list[_Boundary]:
        """Gradual transitions in windows free of other boundaries (``blocked`` frames)."""
        p, n = self.p, self.n
        found: list[_Boundary] = []
        taken = blocked.copy()
        cum = np.concatenate([[0], np.cumsum(blocked)])
        for k in sorted(set(p.dissolve_lags)):
            if 2 * k + 1 > n or k < 2:
                continue
            centres = np.arange(k, n - k)
            ok = (cum[centres + k + 1] - cum[centres - k]) == 0
            dist = np.zeros(centres.size)
            if ok.any():
                dist[ok] = 1.0 - self.matcher.identity(centres[ok] - k, centres[ok] + k)
            peak = (dist >= p.dissolve_min_distance) & (dist >= _rolling(dist, 2 * (k // 2) + 1, np.max))
            for i in np.flatnonzero(peak)[np.argsort(-dist[peak], kind="stable")]:
                c = int(centres[i])
                if taken[c - k:c + k + 1].any():
                    continue
                # Motion-compensated check first: pans and zooms end here cheaply.
                if self.matcher.distance(c - k, c + k) < p.dissolve_min_distance:
                    continue
                rec = self._verify_dissolve(c - k, c + k, blocked)
                if rec is not None:
                    found.append(rec)
                    taken[rec.f0 - 1:rec.f1 + 1] = True
        return found

    def _refine_extent(self, f0: int, f1: int, lo: int, hi: int) -> tuple[int, int]:
        """Grow / shrink the blend frames ``[f0, f1)`` until both anchors are pure.

        With anchors ``f0-1`` and ``f1``, a blended anchor shows up as alphas that keep
        ramping beyond it (below 0 on the left, above 1 on the right); an unblended
        frame inside the span shows up as an alpha stuck at 0 / 1.
        """
        p = self.p
        for _ in range(2 * p.dissolve_max_frames):
            # Alphas of the frame beyond each anchor and of the first / last blend frame.
            probe = np.array([max(lo, f0 - 2), f0, f1 - 1, min(hi, f1 + 1)])
            fit = self._blend_fit(f0 - 1, f1, probe)
            if fit is None:
                break
            out_l, in_l, in_r, out_r = np.nan_to_num(fit.alpha, nan=0.5).tolist()
            step = max(0.04, 0.5 / (f1 - f0 + 1))
            moved = False
            if f0 - 2 >= lo and out_l < -step and f1 - f0 < p.dissolve_max_frames:
                f0, moved = f0 - 1, True
            elif in_l < 0.3 * step and f1 - f0 > p.dissolve_min_frames:
                f0, moved = f0 + 1, True
            if f1 + 1 <= hi and out_r > 1.0 + step and f1 - f0 < p.dissolve_max_frames:
                f1, moved = f1 + 1, True
            elif in_r > 1.0 - 0.3 * step and f1 - f0 > p.dissolve_min_frames:
                f1, moved = f1 - 1, True
            if not moved:
                break
        return f0, f1

    def _verify_dissolve(self, lo: int, hi: int, blocked: np.ndarray) -> Optional[_Boundary]:
        p = self.p
        coarse = self._blend_fit(lo, hi, np.arange(lo, hi + 1))
        if coarse is None or np.isnan(coarse.alpha).any():
            return None
        i0, i1, err = _fit_ramp(np.clip(coarse.alpha, 0.0, 1.0), p.dissolve_min_frames, p.dissolve_max_frames)
        if i1 <= i0 or err > 0.2:
            return None
        # The refined span may reach beyond the window, but not into other boundaries.
        before = np.flatnonzero(blocked[:lo])
        lim_lo = int(before[-1]) + 3 if before.size else 2
        after = np.flatnonzero(blocked[hi + 1:])
        lim_hi = hi + int(after[0]) - 2 if after.size else self.n - 3
        f0, f1 = self._refine_extent(lo + i0, lo + i1, lim_lo, lim_hi)
        a, b = f0 - 1, f1
        if a < 1 or b > self.n - 2 or f1 - f0 < p.dissolve_min_frames:
            return None
        fit = self._blend_fit(a, b, np.arange(a, b + 1))
        if fit is None:
            return None
        inner, rr = fit.alpha[1:-1], fit.r2[1:-1]
        if np.isnan(inner).any() or np.any(np.diff(fit.alpha) < -0.15):
            return None
        if inner[0] > 0.5 or inner[-1] < 0.5:
            return None
        ramp = (np.arange(1, inner.size + 1)) / (inner.size + 1)
        if float(np.sqrt(np.mean((inner - ramp) ** 2))) > p.dissolve_max_err:
            return None
        mid = inner.size // 2
        if rr.mean() < p.dissolve_min_r2 or rr[mid] - fit.single[1 + mid] < p.dissolve_min_gain:
            return None
        dist = self.matcher.distance(a, b)
        if dist < p.dissolve_min_distance:
            return None
        # A zoom or pan between the anchors looks like a blend at low resolution, but its
        # middle frame is a single image: one anchor, warped, explains it as well.
        if rr[mid] - self._warped_r2(a + 1 + mid, (a, b)) < p.dissolve_min_gain:
            return None
        conf = float(np.clip(0.4 + 0.6 * min(1.0, (rr.mean() - 0.6) / 0.35) * min(1.0, dist), 0.4, 1.0))
        boundary = (f0 + f1) // 2
        self.score[boundary] = max(self.score[boundary], dist)
        return _Boundary(boundary, "dissolve", f0, f1, 1.0 + dist, 2, conf,
                         params={"fit_r2": round(float(rr.mean()), 3)})

    # ----------------------------------------------------------------- cut styles

    def _partial_flash(self, b: int, lo: int, hi: int) -> Optional[_Boundary]:
        """A cut flanked by frames blended toward white/black (no flat frame) is a flash."""
        p = self.p
        best: Optional[tuple[float, float, int, int]] = None
        for c in (1.0, 0.0):
            nl = self._ramp_left(b - 1, c, lo, p.ramp_max_frames)
            nr = self._ramp_right(b, c, hi, p.ramp_max_frames)
            al = 1.0 - self.contrast[b - 1] / self.contrast[b - 1 - nl] if nl else 0.0
            ar = 1.0 - self.contrast[b] / self.contrast[b + nr] if nr else 0.0
            alpha = max(al, ar)
            if best is None or alpha > best[1]:
                best = (c, alpha, nl, nr)
        assert best is not None
        c, alpha, nl, nr = best
        if alpha < p.flash_min_alpha:
            return None
        rgb = (255, 255, 255) if c == 1.0 else (0, 0, 0)
        if (nl + nr) > p.flash_max_frames or max(nl, nr) >= p.fade_min_ramp:
            kind: TransitionKind = self._dip_kind(c)
        else:
            kind = "flash"
        return _Boundary(b, kind, b - nl, b + nr, 0.0, 1, float(np.clip(0.4 + alpha, 0.5, 0.9)), rgb,
                         {"peak_alpha": round(float(alpha), 3), "ramp_in": nl, "ramp_out": nr})

    def _texture(self, f: int) -> tuple[float, float]:
        """``(gradient energy, log(Ex / Ey))`` of frame ``f`` at half resolution: medians over
        blocks of active pixels, so captions and other local overlays barely move them."""
        cached = self._texture_cache.get(f)
        if cached is not None:
            return cached
        g = cv2.pyrDown(self.ctx.gray[f]).astype(np.float32)
        h, w = g.shape
        m = cv2.resize(self.active_full.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        bs = max(4, min(h, w) // 8)
        cnt = _block_sum(m, bs)
        ok = cnt >= 0.5 * bs * bs
        if not ok.any():
            ok = cnt > 0
        ex = _block_sum(gx * gx * m, bs)[ok] / cnt[ok]
        ey = _block_sum(gy * gy * m, bs)[ok] / cnt[ok]
        stats = (float(np.median(ex + ey)), float(np.median(np.log((ex + 1.0) / (ey + 1.0)))))
        self._texture_cache[f] = stats
        return stats

    def _blur_side(self, near: list[int], ref: list[int]) -> Optional[tuple[float, float]]:
        """Most blurred frame of ``near`` vs. the calm ``ref`` frames of the same shot:
        ``(energy ratio, anisotropy shift)``, or None without enough references."""
        if len(ref) < 2 or not near:
            return None
        ref = ref[:: max(1, len(ref) // 6)]  # a few calm frames are enough
        ref_e = float(np.median([self._texture(f)[0] for f in ref]))
        ref_a = float(np.median([self._texture(f)[1] for f in ref]))
        if ref_e <= 1e-6:
            return None
        energy, aniso = min((self._texture(f) for f in near), key=lambda t: t[0])
        return energy / ref_e, aniso - ref_a

    def _chroma_shift(self, f: int) -> float:
        """Horizontal red/blue misalignment of frame ``f`` as a fraction of width."""
        frame = self.ctx.frames[f]
        r, b = frame[..., 0].astype(np.float32), frame[..., 2].astype(np.float32)
        if r.std() < 2.0 or b.std() < 2.0:
            return 0.0
        (dx, _), _ = cv2.phaseCorrelate(r, b)
        return abs(dx) / frame.shape[1]

    def _cut_style(self, b: int, lo: int, hi: int) -> Optional[_Boundary]:
        """Name a hard cut at ``b`` between shots ``[lo, b)`` and ``[b, hi)`` (None = plain cut)."""
        p = self.p
        # Glitch: channel misalignment at the cut that the surrounding frames do not have.
        shift = max(self._chroma_shift(b - 1), self._chroma_shift(b))
        if shift >= p.glitch_min_shift:
            refs = [f for f in (b - 4, b - 3, b + 2, b + 3) if lo <= f < hi]
            ref = max((self._chroma_shift(f) for f in refs), default=0.0)
            if refs and ref < 0.5 * shift:
                conf = float(np.clip(0.4 + 10.0 * (shift - p.glitch_min_shift), 0.5, 0.8))
                return _Boundary(b, "glitch", b, b, 0.0, 1, conf, params={"rgb_shift": round(shift, 4)})
        span = max(2, int(round(p.style_ref_seconds * self.fps)))
        out_side = self._blur_side([f for f in (b - 1, b - 2) if f >= lo],
                                   list(range(max(lo, b - 2 - span), b - 2)))
        in_side = self._blur_side([f for f in (b, b + 1) if f < hi],
                                  list(range(b + 2, min(hi, b + 2 + span))))
        sides = [(s, pair) for s, pair in ((out_side, (b - 2, b - 1)), (in_side, (b, b + 1)))
                 if s is not None and s[0] <= p.style_max_blur]
        if not sides:
            return None
        (blur, aniso), pair = min(sides, key=lambda sp: sp[0][0])
        params: dict[str, Any] = {"blur_ratio": round(blur, 3)}
        kind: Optional[TransitionKind] = None
        motion = None
        if lo <= pair[0] and pair[1] < hi:
            sim, scale, angle, dx, dy = self.matcher.best_motion(pair[0], pair[1], _STYLE_SEARCH)
            plain = float(self.matcher.identity(np.array([pair[0]]), np.array([pair[1]]))[0])
            if sim - plain >= p.style_min_gain:
                motion = (scale, angle, dx, dy)
        if abs(aniso) >= p.whip_min_aniso:
            kind = "whip"
            horizontal = aniso < 0  # horizontal motion blur removes horizontal gradients
            d = motion[2] if (motion and horizontal) else (motion[3] if motion else 0.0)
            if horizontal:
                params["direction"] = "right" if d > 0 else "left"
            else:
                params["direction"] = "down" if d > 0 else "up"
        elif motion is not None:
            scale, angle, dx, dy = motion
            zoom = abs(np.log(scale)) / np.log(p.zoom_min_scale)
            spin = abs(angle) / p.spin_min_angle
            if max(zoom, spin) >= 1.0:
                kind = "zoom" if zoom >= spin else "spin"
                params["scale" if kind == "zoom" else "angle"] = round(scale if kind == "zoom" else angle, 3)
            elif np.hypot(dx, dy) >= p.whip_min_shift:
                kind = "whip"
                if abs(dx) >= abs(dy):
                    params["direction"] = "right" if dx > 0 else "left"
                else:
                    params["direction"] = "down" if dy > 0 else "up"
        if kind is None:
            return None
        # The visibly affected frames are the blurred ones adjacent to the cut.
        f0 = b - 1 if out_side is not None and out_side[0] <= p.style_max_blur else b
        f1 = b + 1 if in_side is not None and in_side[0] <= p.style_max_blur else b
        conf = float(np.clip(0.5 + 0.5 * (1.0 - blur / p.style_max_blur), 0.5, 0.8))
        return _Boundary(b, kind, f0, f1, 0.0, 1, conf, params=params)

    # ----------------------------------------------------------------- assembly

    def _merge(self, records: list[_Boundary]) -> list[_Boundary]:
        """Keep the strongest candidates at least ``min_shot_frames`` apart."""
        ms = self.p.min_shot_frames
        taken: list[_Boundary] = []
        for r in sorted(records, key=lambda r: (-r.priority, -r.strength)):
            if r.frame < ms or r.frame > self.n - ms:
                continue
            if any(abs(r.frame - o.frame) < ms for o in taken):
                continue
            taken.append(r)
        return sorted(taken, key=lambda r: r.frame)

    def run(self) -> ShotSegmentation:
        n, p = self.n, self.p
        self._content_distance()
        events = self._excursions()
        records, suppressed, fade_in, fade_out = self._resolve_excursions(events)
        cuts = self._hard_cuts(suppressed)
        blocked = self.flat | suppressed
        for ex in events:
            blocked[ex.f0:ex.f1] = True
        for r in records + cuts:
            blocked[r.f0:r.f1 + 1] = True
            blocked[r.frame] = True
        self.blocked = blocked
        dissolves = self._find_dissolves(blocked)
        accepted = self._merge(records + cuts + dissolves)

        edges = [0] + [r.frame for r in accepted] + [n]
        for i, r in enumerate(accepted):
            if r.kind != "cut" or r.priority != 1:
                continue
            lo, hi = edges[i], edges[i + 2]
            styled = self._partial_flash(r.frame, lo, hi)
            if styled is None and p.classify_cuts:
                styled = self._cut_style(r.frame, lo, hi)
            if styled is not None:
                styled.params = {**r.params, **styled.params}
                accepted[i] = styled

        shots = [ShotSpan(i, edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        transitions: list[TransitionSpan] = []
        if fade_in is not None:
            transitions.append(TransitionSpan(
                fade_in.kind, 0, 0, max(1, min(fade_in.f1, edges[1])), None, 0,
                round(fade_in.confidence, 3), fade_in.color, {"edge": "start"},
            ))
        for i, r in enumerate(accepted):
            f0 = int(np.clip(r.f0, edges[i], r.frame))
            f1 = int(np.clip(r.f1, r.frame, edges[i + 2]))
            if r.kind == "cut":
                f0 = f1 = r.frame
            transitions.append(TransitionSpan(
                r.kind, r.frame, f0, f1, i, i + 1, round(r.confidence, 3), r.color, dict(r.params),
            ))
        if fade_out is not None:
            transitions.append(TransitionSpan(
                fade_out.kind, n, min(n - 1, max(fade_out.f0, edges[-2])), n, len(shots) - 1, None,
                round(fade_out.confidence, 3), fade_out.color, {"edge": "end"},
            ))
        return ShotSegmentation(shots=shots, transitions=transitions, boundary_score=self.score.copy())


def detect_shots(ctx: AnalysisContext) -> ShotSegmentation:
    """Segment ``ctx`` into shots and describe every boundary (see module docstring)."""
    n = ctx.n_frames
    if n == 0:
        return ShotSegmentation(shots=[], transitions=[], boundary_score=np.zeros(0))
    if n < 3:
        return ShotSegmentation(shots=[ShotSpan(0, 0, n)], transitions=[], boundary_score=np.zeros(n))
    return _Detector(ctx, ShotParams.from_context(ctx)).run()
