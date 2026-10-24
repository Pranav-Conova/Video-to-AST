"""Look analysis: per-shot colour grade and layout, plus in-shot frame effects.

Per shot, a handful of representative frames (evenly spaced, avoiding
transition frames and frames altered by detected effects) are measured:

* **Look** (inside the content box): brightness / contrast (luma mean / std),
  HSV saturation, temperature ``(R-B)/(R+B)``, tint ``G-(R+B)/2``, black & white
  (near-zero chroma ``max(RGB)-min(RGB)``), sharpness ``v / (v + K)`` from the
  Laplacian variance ``v`` with a resolution-aware constant ``K``, vignette (the
  :func:`editast.fx.vignette` strength that explains the corner darkening
  relative to both the centre and the edge midpoints of the temporal-mean luma,
  reported only when all four corners agree, so a dark floor or a bright sky
  alone does not read as vignette) and the dominant colour (peak of a smoothed
  coarse RGB histogram).
* **Layout**: letterbox / pillarbox bars are runs of edge lines that match the
  outermost line's colour on most of their pixels (so captions on the bar are
  tolerated), are flat (robust spread at noise level, which a blurred
  background never reaches) and stay constant over the shot. A blur pad is a sharp content
  band between two soft (low Laplacian energy) but non-flat bands that
  correlate with a blurred, cover-scaled copy of the content. A split screen is
  a perfectly straight seam near the middle of the content that persists in
  every sampled frame.

In-shot effects (frames covered by non-cut transitions are skipped, so flash
and fade transitions are not reported twice):

* **Flash**: blending a frame toward a colour ``c`` by ``a`` scales its contrast
  by ``1-a`` (independent of ``c``), so ``a = 1 - std / baseline_std`` against a
  rolling median baseline that excludes candidate frames; the colour follows
  from ``rgb = (1-a) * baseline_rgb + a * c``. A spike must move the mean colour
  clearly away from the scene (``flash_color_shift`` per unit of ``a``, RMS over
  the channels, so a saturated colour flash at the scene's luma counts) and
  last at most ``flash_max_duration``.
* **Blur**: Laplacian variance below ``blur_threshold`` of the shot median for
  at least two frames. The Gaussian sigma is fitted from the radially averaged
  power-spectrum ratio to nearby sharp frames (``log ratio = -4 pi^2 sigma^2
  f^2``, exactly the transfer function of :func:`editast.fx.blur`, insensitive
  to camera motion). Directional (motion) blur is rejected by comparing the
  horizontal and vertical fits, and a collapse of global contrast (a flash or
  fade also removes detail) by comparing the luma std with the reference frames.
* **Invert**: consecutive frames whose coarse thumbnails are strongly
  anti-correlated and whose mean colours sum to ~1 flip a polarity flag; shots
  separated by such a flip share a polarity domain (an inversion often fools
  the cut detector) and the minority polarity of each domain is inverted.
* **RGBSplit**: low-pass weighted horizontal phase correlation between the R
  and B gradient magnitudes; :func:`editast.fx.rgb_split` moves R by
  ``+offset*width`` and B by ``-offset*width``, so ``offset = shift / (2 *
  width)``. Spans are frames whose shift is clearly above the estimation noise
  (or a whole shot with a constant split).

Flash / blur / RGB-split keyframes hold the per-frame intensity, sigma (as a
fraction of width) or offset; ``easing`` describes the envelope shape after the
peak (``1 - ease(x)`` decay, or ``ease(x)`` rise when the peak is at the end).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from editast import fx
from editast.analyzers.types import EffectsResult, LookAnalysis, ShotSegmentation, ShotSpan
from editast.ast.nodes import Blur, EffectBase, Flash, Invert, Keyframe, Layout, Look, RGBSplit
from editast.context import AnalysisContext, FrameFeatures

NAME = "look"
_EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out")
_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class _Params:
    """``ctx.config.p("look", ...)`` with seconds-to-frames conversion."""

    def __init__(self, ctx: AnalysisContext) -> None:
        self._cfg = ctx.config
        self.fps = ctx.fps

    def __call__(self, key: str, default: Any) -> Any:
        return self._cfg.p(NAME, key, default)

    def frames(self, key: str, seconds: float, minimum: int = 1) -> int:
        return max(minimum, int(round(float(self(key, seconds)) * self.fps)))


# --------------------------------------------------------------------------- public API


def analyze_look(ctx: AnalysisContext, seg: ShotSegmentation) -> LookAnalysis:
    """Per-shot ``Look`` and ``Layout`` plus in-shot Flash / Blur / Invert / RGBSplit effects.

    Signals: ``luma``, ``saturation``, ``sharpness_rel`` (downscaled
    Laplacian variance / shot median), ``rgb_shift`` (R-vs-B offset as a
    fraction of width, in :func:`editast.fx.rgb_split` units) and
    ``flash_amount`` (estimated blend toward a flash colour).
    """
    prm = _Params(ctx)
    n = ctx.n_frames
    feats = ctx.features
    shots = [ShotSpan(s.index, max(0, s.f0), min(s.f1, n)) for s in seg.shots if min(s.f1, n) > max(s.f0, 0)]
    effects = EffectsResult()
    analysis = LookAnalysis(effects=effects)
    sharp_rel = np.ones(n)
    flash_amount = np.zeros(n)
    rgb_shift = np.zeros(n)
    if n == 0 or not shots:
        effects.signals = _signals(feats, sharp_rel, rgb_shift, flash_amount)
        return analysis

    w, h = ctx.size
    trans = np.zeros(n, bool)
    tm = np.asarray(seg.transition_frames(), bool)[:n]
    trans[:tm.size] = tm
    shift, response = _rgb_shift(ctx.frames, float(prm("rgb_max_offset", 0.05)), float(prm("rgb_lowpass", 0.12)))
    rgb_shift = shift / (2.0 * w)
    thumbs, texture = _thumbnails(ctx.gray)
    inverted, flips = _inverted_frames(ctx, shots, trans, thumbs, texture, prm)
    altered = inverted.copy()  # frames changed by an in-shot effect (kept out of look sampling)
    half = _downscale(ctx.gray, int(prm("blur_long_side", 160)))
    detail, blocks = _detail(half)
    spectrum = _Spectrum(*half.shape[1:])

    for shot in shots:
        sl = slice(shot.f0, shot.f1)
        usable = ~trans[sl]
        found: list[EffectBase] = []
        flashes, amount, flash_mask = _detect_flashes(ctx, shot, usable, prm)
        flash_amount[sl] = amount
        found += flashes
        blurs, rel, blur_mask = _detect_blurs(ctx, shot, usable & ~flash_mask, half, detail, blocks, spectrum, prm)
        sharp_rel[sl] = rel
        found += blurs
        found += _invert_effects(ctx, shot, inverted, flips)
        splits, split_mask = _detect_rgb_split(ctx, shot, usable & ~flash_mask, shift, response, texture, prm)
        found += splits
        altered[sl] |= flash_mask | blur_mask | split_mask
        for eff in sorted(found, key=lambda e: (e.start, e.end)):
            effects.add(shot.index, eff)

    k = int(prm("samples", 10))
    for shot in shots:
        idx = _sample(shot, ~trans & ~altered, k)
        if idx.size == 0:
            idx = _sample(shot, np.ones(n, bool), k)
        sample = ctx.frames[idx]
        layout = _layout(sample, prm)
        analysis.layouts[shot.index] = layout
        analysis.looks[shot.index] = _look(sample, layout, prm)

    effects.signals = _signals(feats, sharp_rel, rgb_shift, flash_amount)
    return analysis


def _signals(
    feats: FrameFeatures, sharp_rel: np.ndarray, rgb_shift: np.ndarray, flash_amount: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "luma": np.asarray(feats.luma_mean, float),
        "saturation": np.asarray(feats.sat_mean, float),
        "sharpness_rel": sharp_rel,
        "rgb_shift": rgb_shift,
        "flash_amount": flash_amount,
    }


# --------------------------------------------------------------------------- helpers


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index pairs of the True runs of a boolean array."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _sample(shot: ShotSpan, ok: np.ndarray, k: int) -> np.ndarray:
    idx = np.arange(shot.f0, shot.f1)[ok[shot.f0:shot.f1]]
    if idx.size <= k:
        return idx
    return idx[np.linspace(0, idx.size - 1, k).round().astype(int)]


def _baseline(values: np.ndarray, exclude: np.ndarray, size: int) -> np.ndarray:
    """Rolling median of ``values`` with excluded samples interpolated from their neighbours."""
    keep = np.flatnonzero(~exclude)
    if keep.size == 0:
        return np.full(values.shape, float(np.median(values)))
    x = np.interp(np.arange(values.size), keep, values[keep])
    return ndimage.median_filter(x, size=min(size, x.size) | 1, mode="nearest")


def _envelope(values: np.ndarray) -> str:
    """Easing that best describes a peaked envelope (decay after the peak, or the rise).

    A flat envelope (a constant effect switched on and off) is ``"linear"``.
    """
    v = np.asarray(values, float)
    peak = int(np.argmax(v))
    if v[peak] <= 0 or v.min() >= 0.8 * v[peak]:
        return "linear"
    if peak == v.size - 1 and v.size >= 3:
        part, rising = v / v[peak], True
    else:
        part, rising = v[peak:] / v[peak], False
    if part.size < 3:
        return "linear"
    x = np.linspace(0.0, 1.0, part.size)
    errors = {}
    for name in _EASINGS:
        model = fx.ease(x, name) if rising else 1.0 - fx.ease(x, name)
        errors[name] = float(((model - part) ** 2).sum())
    return min(errors, key=errors.get)


# --------------------------------------------------------------------------- flash


def _detect_flashes(
    ctx: AnalysisContext, shot: ShotSpan, usable: np.ndarray, prm: _Params
) -> tuple[list[Flash], np.ndarray, np.ndarray]:
    """Short contrast collapses toward a colour. Returns (effects, per-frame amount, mask)."""
    f = ctx.features
    sl = slice(shot.f0, shot.f1)
    m = shot.f1 - shot.f0
    amount = np.zeros(m)
    mask = np.zeros(m, bool)
    if m < 3 or usable.sum() < 3:
        return [], amount, mask
    std = np.asarray(f.luma_std[sl], float)
    rgb = np.asarray(f.rgb_mean[sl], float)
    size = prm.frames("flash_baseline", 0.8, minimum=9) | 1
    a_min = float(prm("flash_min_intensity", 0.25))
    a_edge = float(prm("flash_edge", 0.08))
    shift_min = float(prm("flash_color_shift", 0.15))
    max_len = prm.frames("flash_max_duration", 0.4, minimum=2)

    # Pass 1 finds candidates, pass 2 re-estimates the baseline without them.
    base = _baseline(std, ~usable, size)
    cand = usable & (1.0 - std / np.maximum(base, 1e-3) > 0.6 * a_min)
    exclude = ~usable | ndimage.binary_dilation(cand, iterations=1)
    base_std = np.maximum(_baseline(std, exclude, size), 1e-3)
    base_rgb = np.stack([_baseline(rgb[:, c], exclude, size) for c in range(3)], axis=1)
    a = np.clip(1.0 - std / base_std, 0.0, 1.0)
    delta = rgb - base_rgb  # colour shift away from the scene
    dist = np.linalg.norm(delta, axis=1) / np.sqrt(3.0)
    a[~usable] = 0.0
    amount[:] = a

    flashes: list[Flash] = []
    hot = (a >= a_min) & (dist >= shift_min * a)
    for r0, r1 in _runs(hot):
        peak = r0 + int(np.argmax(a[r0:r1]))
        if mask[peak]:
            continue
        toward = delta[peak]  # neighbours belong to the flash while they shift the same way
        s0, s1 = peak, peak + 1
        while s0 > 0 and a[s0 - 1] >= a_edge and float(delta[s0 - 1] @ toward) > 0:
            s0 -= 1
        while s1 < m and a[s1] >= a_edge and float(delta[s1] @ toward) > 0:
            s1 += 1
        if s1 - s0 > max_len or s1 - s0 >= 0.8 * m:
            continue  # a lighting change, not a flash
        mask[s0:s1] = True
        strong = np.arange(s0, s1)[a[s0:s1] >= 0.5 * a[peak]]
        colour = base_rgb[strong] + (rgb[strong] - base_rgb[strong]) / a[strong, None]
        c = np.clip(np.median(colour, axis=0), 0.0, 1.0)
        if c.min() >= 0.9:
            c = np.ones(3)
        elif c.max() <= 0.1:
            c = np.zeros(3)
        frames = np.arange(s0, s1)
        flashes.append(Flash(
            start=ctx.t(shot.f0 + s0), end=ctx.t(shot.f0 + s1),
            color=tuple(int(round(v * 255)) for v in c), intensity=round(float(a[peak]), 4),
            easing=_envelope(a[s0:s1]),
            keyframes=[Keyframe(t=ctx.t(shot.f0 + i), v=round(float(a[i]), 4)) for i in frames],
            confidence=round(float(np.clip(0.5 + 0.5 * a[peak], 0.5, 0.95)), 3),
            params={"peak": round(ctx.t(shot.f0 + peak), 4)},
        ))
    return flashes, amount, mask


# --------------------------------------------------------------------------- blur


class _Spectrum:
    """Radially averaged power spectra (all directions, horizontal and vertical sectors)."""

    def __init__(self, h: int, w: int, n_bins: int = 48) -> None:
        self.window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        fy = np.fft.fftfreq(h)[:, None]
        fxx = np.fft.rfftfreq(w)[None, :]
        f = np.sqrt(fy ** 2 + fxx ** 2)
        self.n_bins = n_bins
        self.bins = np.minimum((f / 0.5 * n_bins).astype(int), n_bins).ravel()
        self.freq = (np.arange(n_bins) + 0.5) * 0.5 / n_bins
        tan30 = np.tan(np.radians(30.0))
        self.sectors = {
            "all": np.ones(f.size, bool),
            "x": (np.abs(fy) <= tan30 * fxx).ravel(),  # frequencies along x: blur in x
            "y": (fxx <= tan30 * np.abs(fy)).ravel(),
        }
        self.lowest = 2.0 / min(h, w)  # skip the DC / window-dominated bins

    def power(self, gray: np.ndarray) -> dict[str, np.ndarray]:
        g = gray.astype(np.float32)
        g = (g - g.mean()) * self.window
        p = np.abs(np.fft.rfft2(g)).ravel() ** 2
        out = {}
        for name, sel in self.sectors.items():
            tot = np.bincount(self.bins[sel], weights=p[sel], minlength=self.n_bins + 1)[: self.n_bins]
            cnt = np.bincount(self.bins[sel], minlength=self.n_bins + 1)[: self.n_bins]
            out[name] = tot / np.maximum(cnt, 1)
        return out

    def sigma(self, p: np.ndarray, p_ref: np.ndarray) -> float:
        """Gaussian sigma (px) explaining ``p / p_ref = exp(-4 pi^2 sigma^2 f^2)``."""
        f = self.freq
        ok = (f >= self.lowest) & (f <= 0.45) & (p_ref > 0)
        lr = np.log(np.clip(p / np.maximum(p_ref, 1e-12), 1e-9, 1.0))
        use = ok & (lr > np.log(2e-3))
        # Stop at the first bin that reaches the noise floor.
        below = np.flatnonzero(ok & ~use)
        if below.size:
            use &= np.arange(f.size) < below[0]
        if use.sum() < 2:
            return 0.0 if not below.size else float(np.sqrt(-np.log(2e-3) / (4 * np.pi ** 2)) / max(f[below[0]], 1e-3))
        f2 = f[use] ** 2
        s2 = -float((lr[use] * f2).sum()) / (4 * np.pi ** 2 * float((f2 * f2).sum()))
        return float(np.sqrt(max(s2, 0.0)))


def _downscale(gray: np.ndarray, long_side: int) -> np.ndarray:
    """Area-downscaled copies of the gray frames (grain and compression noise average out)."""
    n, h, w = gray.shape
    s = min(1.0, long_side / max(h, w))
    if s >= 1.0:
        return gray
    size = (max(16, int(round(w * s))), max(16, int(round(h * s))))
    return np.stack([cv2.resize(g, size, interpolation=cv2.INTER_AREA) for g in gray])


def _detail(half: np.ndarray, cells: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Laplacian variance per frame and mean Laplacian energy per grid block ``(n, n_blocks)``.

    The grid has ``cells`` blocks along the short side.
    """
    n, h, w = half.shape
    grid = (cells, max(1, int(round(cells * h / w)))) if w <= h else (max(1, int(round(cells * w / h))), cells)
    detail = np.zeros(n)
    blocks = np.zeros((n, grid[0] * grid[1]))
    for i, g in enumerate(half):
        lap = cv2.Laplacian(g, cv2.CV_32F)
        detail[i] = float(lap.var())
        blocks[i] = cv2.resize(lap * lap, grid, interpolation=cv2.INTER_AREA).ravel()
    return detail, blocks


def _detect_blurs(
    ctx: AnalysisContext,
    shot: ShotSpan,
    usable: np.ndarray,
    half: np.ndarray,
    detail: np.ndarray,
    blocks: np.ndarray,
    spectrum: _Spectrum,
    prm: _Params,
) -> tuple[list[Blur], np.ndarray, np.ndarray]:
    """Global, isotropic sharpness dips. Returns (effects, sharpness relative to the shot, mask).

    ``half`` are downscaled gray frames, ``detail`` their Laplacian variance and
    ``blocks`` its per-block energy: at full analysis resolution the Laplacian
    is dominated by grain, which H.264 keeps in some frames and drops in others.
    A blur softens most of the frame, while a caption or subject leaving the
    frame only removes detail locally (``blur_coverage``). A soft span running
    into a cut must switch on abruptly (a digital blur) or keep softening
    toward the cut; a gradual onset followed by a plateau is a held zoom or a
    focus change.
    """
    m = shot.f1 - shot.f0
    sharp = detail[shot.f0:shot.f1]
    mask = np.zeros(m, bool)
    level = float(np.median(sharp[usable])) if usable.any() else float(np.median(sharp))
    rel = sharp / max(level, 1e-6)
    if m < 3 or usable.sum() < 3:
        return [], rel, mask
    core = usable & (rel < float(prm("blur_threshold", 0.38)))
    soft = usable & (rel < float(prm("blur_edge", 0.65)))
    min_len = int(prm("blur_min_frames", 2))
    w = half.shape[2]  # sigmas are measured in downscaled pixels
    thr = float(prm("blur_threshold", 0.38))
    contrast = np.asarray(ctx.features.luma_std, float)
    blurs: list[Blur] = []
    for s0, s1 in _runs(soft):
        if (core[s0:s1]).sum() < min_len or s1 - s0 >= 0.8 * m:
            continue
        nearby = list(range(s0 - 1, max(-1, s0 - 7), -1)) + list(range(s1, min(m, s1 + 6)))
        refs = [i for i in nearby if usable[i] and rel[i] >= 0.7][:4]
        if not refs:
            continue
        peak = s0 + int(np.argmin(rel[s0:s1]))
        ref_std = float(np.median(contrast[shot.f0 + np.array(refs)]))
        if contrast[shot.f0 + peak] < float(prm("blur_contrast", 0.6)) * ref_std:
            continue  # global contrast collapsed: a flash or fade, not a blur
        ref_blocks = blocks[shot.f0 + np.array(refs)].mean(axis=0)
        textured = ref_blocks > 0.1 * max(float(np.median(ref_blocks)), 1e-6)
        softened = blocks[shot.f0 + peak] < float(prm("blur_block_ratio", 0.6)) * ref_blocks
        if not textured.any() or softened[textured].mean() < float(prm("blur_coverage", 0.6)):
            continue  # local change (caption, subject), not a blur
        into_cut = not usable[s1:].any()
        from_cut = not usable[:s0].any()
        if into_cut or from_cut:
            span = rel[s0:s1]
            edge = span[-3:] if into_cut else span[:3]
            abrupt = (min(rel[s0:s0 + 2]) <= thr) if into_cut else (min(rel[max(s0, s1 - 2):s1]) <= thr)
            if not abrupt and float(np.median(edge)) >= 0.8 * float(np.median(span)):
                continue  # gradual onset, then a plateau up to the cut
        ref_p = [spectrum.power(half[shot.f0 + i]) for i in refs]
        ref = {k: np.mean([p[k] for p in ref_p], axis=0) for k in spectrum.sectors}
        pk = spectrum.power(half[shot.f0 + peak])
        sx, sy = spectrum.sigma(pk["x"], ref["x"]), spectrum.sigma(pk["y"], ref["y"])
        if max(sx, sy) > 0.5 and min(sx, sy) < float(prm("blur_isotropy", 0.5)) * max(sx, sy):
            continue  # directional: motion blur from a fast camera move
        sigmas = np.array([spectrum.sigma(spectrum.power(half[shot.f0 + i])["all"], ref["all"])
                           if i != peak else spectrum.sigma(pk["all"], ref["all"]) for i in range(s0, s1)])
        amounts = sigmas / w
        if amounts.max() < float(prm("blur_min_amount", 0.0015)):
            continue
        mask[s0:s1] = True
        depth = 1.0 - float(rel[peak])
        blurs.append(Blur(
            start=ctx.t(shot.f0 + s0), end=ctx.t(shot.f0 + s1),
            amount=round(float(amounts.max()), 5), easing=_envelope(amounts),
            keyframes=[Keyframe(t=ctx.t(shot.f0 + i), v=round(float(v), 5))
                       for i, v in zip(range(s0, s1), amounts)],
            confidence=round(float(np.clip(0.45 + 0.4 * depth, 0.45, 0.85)), 3),
            params={"peak": round(ctx.t(shot.f0 + peak), 4)},
        ))
    return blurs, rel, mask


# --------------------------------------------------------------------------- invert


def _thumbnails(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean, unit-norm coarse thumbnails (N, d) and their std (texture)."""
    n, h, w = gray.shape
    size = (max(8, w // 8), max(8, h // 8))
    th = np.stack([cv2.resize(g, size, interpolation=cv2.INTER_AREA) for g in gray]).reshape(n, -1)
    th = th.astype(np.float32) / 255.0
    th -= th.mean(axis=1, keepdims=True)
    std = th.std(axis=1)
    th /= np.maximum(np.linalg.norm(th, axis=1, keepdims=True), 1e-6)
    return th, std


def _inverted_frames(
    ctx: AnalysisContext,
    shots: list[ShotSpan],
    trans: np.ndarray,
    thumbs: np.ndarray,
    texture: np.ndarray,
    prm: _Params,
) -> tuple[np.ndarray, np.ndarray]:
    """Frames that are the photographic negative of their context, and polarity flip frames."""
    n = thumbs.shape[0]
    rgb = np.asarray(ctx.features.rgb_mean, float)
    corr_in = float(prm("invert_corr", -0.5))
    corr_cut = float(prm("invert_corr_cut", -0.7))
    tol = float(prm("invert_mean_tol", 0.12))
    min_tex = float(prm("invert_min_texture", 0.02))
    starts = {s.f0 for s in shots[1:]}
    polarity = np.ones(n, np.int8)
    domain = np.zeros(n, int)
    flips = np.zeros(n, bool)
    cur, dom, last = 1, 0, -1
    for t in range(n):
        textured = texture[t] > min_tex
        flip = False
        if last >= 0 and textured and not trans[t]:
            c = float(thumbs[t] @ thumbs[last])
            limit = corr_cut if t in starts else corr_in
            flip = c < limit and float(np.abs(rgb[t] + rgb[last] - 1.0).max()) < tol
        if t in starts and not flip:
            cur, dom = 1, dom + 1  # a real cut starts a new polarity domain
        elif flip:
            cur = -cur
            flips[t] = True
        polarity[t], domain[t] = cur, dom
        if textured:
            last = t
    inverted = np.zeros(n, bool)
    for d in np.unique(domain):
        sel = (domain == d) & ~trans
        neg = int((polarity[sel] < 0).sum())
        if neg == 0:
            continue
        minority = -1 if neg <= int(sel.sum()) - neg else 1
        inverted |= (domain == d) & (polarity == minority)
    return inverted, flips


def _invert_effects(ctx: AnalysisContext, shot: ShotSpan, inverted: np.ndarray, flips: np.ndarray) -> list[Invert]:
    out = []
    for s0, s1 in _runs(inverted[shot.f0:shot.f1]):
        a, b = shot.f0 + s0, shot.f0 + s1
        bounded = bool(flips[a]) + bool(b < len(flips) and flips[b])
        out.append(Invert(start=ctx.t(a), end=ctx.t(b), confidence=round(0.6 + 0.15 * bounded, 3)))
    return out


# --------------------------------------------------------------------------- RGB split


def _rgb_shift(frames: np.ndarray, max_offset: float, cutoff: float, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal shift (px) of the R channel relative to B, and the correlation peak (0..1).

    1-D phase correlation along x of the horizontal gradient *magnitudes*
    (edges sit at the same place in every channel, while the intensities may be
    uncorrelated or even anti-correlated, e.g. a red-to-blue gradient). The
    cross-power spectrum is summed over row bands before normalisation and
    weighted by a Gaussian low-pass (``cutoff`` cycles/px): H.264 keeps chroma
    only at low frequencies, so at high frequencies R and B both carry the
    (unshifted) luma detail and plain phase correlation is biased toward zero.
    Shifts up to ``2 * max_offset * width`` are searched (sub-pixel by a
    parabola through the peak); the peak is normalised so that 1 means all
    weighted phases agree.
    """
    n, h, w = frames.shape[:3]
    size = (w, max(16, h // 4))  # rows are averaged; horizontal resolution is kept
    window = np.hanning(w).astype(np.float32)
    n_fft = 2 * w  # zero padding: no wrap-around
    weight = np.exp(-0.5 * (np.fft.rfftfreq(n_fft) / cutoff) ** 2)
    norm = float(np.fft.irfft(weight, n=n_fft)[0])
    max_lag = int(np.ceil(2.0 * max_offset * w)) + 2
    lags = np.r_[n_fft - max_lag:n_fft, 0:max_lag + 1]  # -max_lag .. max_lag
    shift = np.zeros(n)
    resp = np.zeros(n)
    for c0 in range(0, n, chunk):
        part = frames[c0:c0 + chunk]
        grads = np.stack([cv2.resize(np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)), size,
                                     interpolation=cv2.INTER_AREA) for f in part])  # (k, rows, w, 3)
        grads = grads[..., [2, 0]].transpose(0, 3, 1, 2)  # (k, [B, R], rows, w)
        grads -= grads.mean(axis=3, keepdims=True)
        spec = np.fft.rfft(grads * window, n=n_fft, axis=3)
        # B vs R cross-power summed over rows; float64 from here (complex64 division can overflow).
        cross = (np.conj(spec[:, 0]) * spec[:, 1]).sum(axis=1).astype(np.complex128)
        mag = np.abs(cross)
        floor = 1e-9 * np.maximum(mag.max(axis=1, keepdims=True), 1e-30)
        corr = np.fft.irfft(cross / np.maximum(mag, floor) * weight, n=n_fft, axis=1)[:, lags] / norm
        k = np.argmax(corr, axis=1)
        rows = np.arange(k.size)
        inner = (k > 0) & (k < lags.size - 1)
        left = corr[rows, np.clip(k - 1, 0, None)]
        right = corr[rows, np.clip(k + 1, None, lags.size - 1)]
        mid = corr[rows, k]
        den = left - 2.0 * mid + right
        frac = np.where(inner & (den < 0), 0.5 * (left - right) / np.where(den < 0, den, -1.0), 0.0)
        flat = mag.max(axis=1) <= 1e-12
        shift[c0:c0 + part.shape[0]] = np.where(flat, 0.0, k - max_lag + frac)
        resp[c0:c0 + part.shape[0]] = np.where(flat, 0.0, mid)
    return shift, resp


def _detect_rgb_split(
    ctx: AnalysisContext,
    shot: ShotSpan,
    usable: np.ndarray,
    shift: np.ndarray,
    response: np.ndarray,
    texture: np.ndarray,
    prm: _Params,
) -> tuple[list[RGBSplit], np.ndarray]:
    """R/B misalignment clearly above the estimation noise. Returns (effects, mask).

    Natural footage has R and B aligned, so the reference is zero shift (a shot's
    median would be contaminated when a split covers most of a short shot). The
    noise level comes from frame-to-frame differences, which a step-like split
    barely affects.
    """
    sl = slice(shot.f0, shot.f1)
    m = shot.f1 - shot.f0
    mask = np.zeros(m, bool)
    w = ctx.size[0]
    dx = shift[sl]
    max_px = 2.0 * float(prm("rgb_max_offset", 0.05)) * w  # larger "shifts" are failed correlations
    good = (usable & (response[sl] >= float(prm("rgb_min_response", 0.5))) & (texture[sl] > 0.02)
            & (np.abs(dx) <= max_px))
    if good.sum() < 1:
        return [], mask
    min_px = float(prm("rgb_min_shift_px", 1.2))
    v = dx[good]
    noise = 1.4826 * float(np.median(np.abs(np.diff(v)))) / np.sqrt(2.0) if v.size >= 3 else 0.0
    thr = max(min_px, float(prm("rgb_noise_factor", 4.0)) * noise)
    out: list[RGBSplit] = []
    # A constant split over the whole shot.
    base = float(np.median(v))
    if (abs(base) >= float(prm("rgb_shot_shift_px", 1.5)) and v.size >= 3
            and (np.sign(v) * np.sign(base) * np.abs(v) >= thr).mean() >= 0.7):
        idx = np.flatnonzero(good)
        out.append(RGBSplit(start=ctx.t(shot.f0 + idx[0]), end=ctx.t(shot.f0 + idx[-1] + 1),
                            offset=round(base / (2 * w), 5), confidence=0.6, params={"whole_shot": True}))
        mask[idx[0]:idx[-1] + 1] = True
        return out, mask
    hot = good & (np.abs(dx) >= thr)
    # Chroma subsampling and failed correlations make single frames inside a split read
    # low: bridge short gaps.
    spans: list[list[int]] = []
    gap = int(prm("rgb_gap", 2))
    for s0, s1 in _runs(hot):
        if spans and s0 - spans[-1][1] <= gap:
            spans[-1][1] = s1
        else:
            spans.append([s0, s1])
    for s0, s1 in spans:
        vals = dx[s0:s1][hot[s0:s1]]
        sign = 1.0 if (vals > 0).sum() >= (vals < 0).sum() else -1.0
        level = float(np.median(vals[np.sign(vals) == sign]))
        mask[s0:s1] = True
        offs = np.where(good[s0:s1], dx[s0:s1], level) / (2 * w)
        out.append(RGBSplit(
            start=ctx.t(shot.f0 + s0), end=ctx.t(shot.f0 + s1), offset=round(level / (2 * w), 5),
            easing=_envelope(np.abs(offs)),
            keyframes=[Keyframe(t=ctx.t(shot.f0 + i), v=round(float(o), 5))
                       for i, o in zip(range(s0, s1), offs)],
            confidence=round(float(np.clip(0.5 + 0.1 * abs(level) / thr, 0.5, 0.9)), 3),
        ))
    return out, mask


# --------------------------------------------------------------------------- layout


def _bar_lines(lines: np.ndarray, dev: np.ndarray, prm: _Params) -> int:
    """Number of flat bar lines at the start of ``lines`` ``(L, M, 3)`` (edge first).

    A bar line matches the outermost line's colour on most of its pixels
    (sparse text drawn on the bar is tolerated) and is flat: the median
    absolute deviation along the line stays at compression-noise level, which a
    heavily blurred background (blur pad) does not reach. A band of up to
    ``bar_caption_gap`` of non-flat lines (a full-width caption on the bar) is
    crossed when at least three flat lines of the very same colour as the
    edge (``bar_resume_tolerance``) follow it, which dark scene content next to
    a black bar rarely matches; isolated noisy lines (up to two) are always
    crossed.
    """
    tol = float(prm("bar_tolerance", 14.0))
    px = lines.astype(np.float32)
    colour = np.median(px[0], axis=0)
    near = (np.abs(px - colour).max(axis=2) <= tol).mean(axis=1)
    line_colour = np.median(px, axis=1)
    spread = np.median(np.abs(px - line_colour[:, None]).max(axis=2), axis=1)
    max_spread = float(prm("bar_flatness", 2.5))
    if near[0] < 0.9 or spread[0] > max_spread:
        return 0
    flat = ((near >= float(prm("bar_fraction", 0.6))) & (spread <= max_spread)
            & (dev <= float(prm("bar_temporal_dev", 6.0))))
    same = np.abs(line_colour - colour).max(axis=1) <= float(prm("bar_resume_tolerance", 4.0))
    max_gap = int(round(float(prm("bar_caption_gap", 0.12)) * lines.shape[0]))
    size = 0
    for a, b in _runs(flat):
        gap = a - size
        if gap > 2 and (size == 0 or gap > max_gap or b - a < 3 or not same[a:b].all()):
            break
        size = b
    return size


def _soft_band(profile: np.ndarray, level: float, ratio: float) -> int:
    """Length of the low-detail band at the start of a Laplacian-energy profile."""
    bad = np.concatenate((profile >= ratio * level, [True, True, True]))
    run3 = bad[:-2] & bad[1:-1] & bad[2:]
    return int(np.argmax(run3))


def _layout(sample: np.ndarray, prm: _Params) -> Layout:
    k, h, w = sample.shape[:3]
    med = np.median(sample, axis=0).astype(np.uint8)
    gray = sample.astype(np.float32) @ _LUMA
    dev = np.abs(gray - med.astype(np.float32) @ _LUMA).mean(axis=0)  # (h, w) temporal deviation
    row_dev, col_dev = dev.mean(axis=1), dev.mean(axis=0)
    lo, hi = float(prm("bar_min", 0.02)), float(prm("bar_max", 0.45))

    def bar(lines: np.ndarray, d: np.ndarray, total: int) -> int:
        size = _bar_lines(lines, d, prm)
        return size if lo * total <= size <= hi * total else 0

    top = bar(med, row_dev, h)
    bottom = bar(med[::-1], row_dev[::-1], h)
    left = bar(med.transpose(1, 0, 2), col_dev, w)
    right = bar(med.transpose(1, 0, 2)[::-1], col_dev[::-1], w)
    layout = Layout(letterbox_top=top / h, letterbox_bottom=bottom / h,
                    pillarbox_left=left / w, pillarbox_right=right / w)
    y0, y1, x0, x1 = top, h - bottom, left, w - right
    if y1 - y0 < 8 or x1 - x0 < 8:
        return layout

    # Detail at half resolution: 8-bit quantisation of a smooth blurred pad leaves
    # Laplacian noise at full resolution that area averaging removes.
    rh, rw = y1 - y0, x1 - x0
    lap = np.stack([np.abs(cv2.Laplacian(cv2.resize(g, (max(4, rw // 2), max(4, rh // 2)),
                                                    interpolation=cv2.INTER_AREA), cv2.CV_32F))
                    for g in gray[:, y0:y1, x0:x1]])
    for axis in (0, 1):  # 0: bands above / below the content, 1: left / right
        profile = np.median(lap.mean(axis=2 - axis), axis=0)
        span = profile.size
        level = float(np.median(profile[int(0.4 * span):int(0.6 * span) + 1]))
        if level <= 0:
            continue
        ratio = float(prm("pad_soft_ratio", 0.25))
        a = _soft_band(profile, level, ratio)
        b = _soft_band(profile[::-1], level, ratio)
        if min(a, b) < float(prm("pad_min", 0.04)) * span or span - a - b < 0.2 * span:
            continue
        full = rh if axis == 0 else rw
        box = (int(round(a * full / span)), full - int(round(b * full / span)))
        if _is_blur_pad(sample[:, y0:y1, x0:x1], box, axis, prm):
            if axis == 0:
                y0, y1 = y0 + box[0], y0 + box[1]
            else:
                x0, x1 = x0 + box[0], x0 + box[1]
            layout.blur_pad = True
    layout.content_box = (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
    layout.split = _split(gray[:, y0:y1, x0:x1], prm)
    return layout


def _is_blur_pad(region: np.ndarray, box: tuple[int, int], axis: int, prm: _Params) -> bool:
    """Do the bands outside ``box`` look like a blurred, cover-scaled copy of the content?"""
    k, h, w = region.shape[:3]
    a, b = box
    scores = []
    for frame in region[:: max(1, k // 4)]:
        f = frame.astype(np.float32)
        bands = [f[:a], f[b:]] if axis == 0 else [f[:, :a], f[:, b:]]
        if max(float(np.std(x)) for x in bands) < 3.0:
            return False  # flat bands are bars, not a blur pad
        content = f[a:b] if axis == 0 else f[:, a:b]
        scale = h / content.shape[0] if axis == 0 else w / content.shape[1]
        big = cv2.resize(content, (max(w, int(round(content.shape[1] * scale))),
                                   max(h, int(round(content.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        oy, ox = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
        cover = cv2.GaussianBlur(big[oy:oy + h, ox:ox + w], (0, 0), 0.04 * max(h, w))
        guess = [cover[:a], cover[b:]] if axis == 0 else [cover[:, :a], cover[:, b:]]
        x = np.concatenate([cv2.resize(v, (16, 16), interpolation=cv2.INTER_AREA).ravel() for v in bands])
        y = np.concatenate([cv2.resize(v, (16, 16), interpolation=cv2.INTER_AREA).ravel() for v in guess])
        x, y = x - x.mean(), y - y.mean()
        scores.append(float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-6)))
    return bool(scores) and float(np.median(scores)) >= float(prm("pad_correlation", 0.4))


def _split(gray: np.ndarray, prm: _Params) -> str:
    """``"horizontal"`` (top/bottom panels), ``"vertical"`` (side by side) or ``"none"``."""
    k = gray.shape[0]
    if k < 3:
        return "none"
    if float(np.abs(gray - np.median(gray, axis=0)).mean()) < 2.0:
        return "none"  # static content: a natural edge would persist too
    for axis, name in ((1, "horizontal"), (2, "vertical")):
        d = np.abs(np.diff(gray, axis=axis))  # (k, h-1, w) or (k, h, w-1)
        across = d.mean(axis=3 - axis)  # mean edge strength of each seam line, per frame
        size = across.shape[1]
        lo, hi = int(0.38 * size), int(0.62 * size) + 1
        norm = across / np.maximum(np.median(across, axis=1, keepdims=True), 1e-3)
        persistent = norm[:, lo:hi].min(axis=0)
        line = lo + int(np.argmax(persistent))
        if persistent[line - lo] < float(prm("split_strength", 4.0)):
            continue
        # A digital seam is perfectly straight: the strongest edge near it is on the same line everywhere.
        near = d[:, max(0, line - 3):line + 4] if axis == 1 else d[:, :, max(0, line - 3):line + 4]
        arg = np.argmax(near, axis=axis) + max(0, line - 3)
        if float((arg == line).mean()) >= float(prm("split_straightness", 0.7)):
            return name
    return "none"


# --------------------------------------------------------------------------- look


def _look(sample: np.ndarray, layout: Layout, prm: _Params) -> Look:
    k, h, w = sample.shape[:3]
    bx, by, bw, bh = layout.content_box
    x0, y0 = int(round(bx * w)), int(round(by * h))
    x1, y1 = max(x0 + 2, int(round((bx + bw) * w))), max(y0 + 2, int(round((by + bh) * h)))
    crop = sample[:, y0:y1, x0:x1]
    f = crop.astype(np.float32) / 255.0
    luma = f @ _LUMA
    rgb = np.median(f.reshape(k, -1, 3).mean(axis=1), axis=0)
    sat = np.median([cv2.cvtColor(c, cv2.COLOR_RGB2HSV)[..., 1].mean() / 255.0 for c in crop])
    chroma = float(np.median((f.max(axis=3) - f.min(axis=3)).reshape(k, -1).mean(axis=1)))
    gray = (luma * 255.0).astype(np.float32)
    lap = float(np.median([cv2.Laplacian(g, cv2.CV_32F).var() for g in gray]))
    long_side = max(h, w)
    k_sharp = float(prm("sharpness_k", 250.0)) * (320.0 / long_side) ** 2
    r, g, b = (float(v) for v in rgb)
    return Look(
        brightness=round(float(np.median(luma.reshape(k, -1).mean(axis=1))), 4),
        contrast=round(float(np.median(luma.reshape(k, -1).std(axis=1))), 4),
        saturation=round(float(sat), 4),
        temperature=round((r - b) / max(r + b, 1e-6), 4),
        tint=round(g - 0.5 * (r + b), 4),
        bw=bool(chroma < float(prm("bw_chroma", 0.03)) and sat < float(prm("bw_saturation", 0.12))),
        sharpness=round(lap / (lap + k_sharp), 4),
        vignette=round(_vignette(luma.mean(axis=0), prm), 4),
        dominant_color=_dominant_color(crop),
    )


def _vignette(mean_luma: np.ndarray, prm: _Params) -> float:
    """:func:`editast.fx.vignette` strength explaining the corner darkening (0 = none).

    Each corner patch gets the strength that explains its darkening relative
    to both the centre and the edge midpoints; the result is their mean, but
    only when every corner is darkened (at least ``vignette_agreement`` of the
    mean): a vignette is radially symmetric, while dark scene content (a road,
    a shadow) rarely covers all four corners.
    """
    h, w = mean_luma.shape
    ch, cw = max(2, int(0.12 * h)), max(2, int(0.12 * w))
    corners = np.array([mean_luma[:ch, :cw].mean(), mean_luma[:ch, -cw:].mean(),
                        mean_luma[-ch:, :cw].mean(), mean_luma[-ch:, -cw:].mean()])
    eh, ew = max(2, int(0.12 * h)), max(2, int(0.24 * w))
    vh, vw = max(2, int(0.24 * h)), max(2, int(0.12 * w))
    mx0, mx1, my0, my1 = (w - ew) // 2, (w + ew) // 2, (h - vh) // 2, (h + vh) // 2
    edges = np.mean([mean_luma[:eh, mx0:mx1].mean(), mean_luma[-eh:, mx0:mx1].mean(),
                     mean_luma[my0:my1, :vw].mean(), mean_luma[my0:my1, -vw:].mean()])
    centre = mean_luma[int(0.35 * h):int(0.65 * h) + 1, int(0.35 * w):int(0.65 * w) + 1].mean()
    if centre <= 1e-3 or edges <= 1e-3:
        return 0.0
    # fx.vignette darkens by 1 - s*r^2 with r^2 ~ 0.77 in the corner patches, ~0.39 at the edge midpoints.
    q_c, q_e = corners / centre, corners / edges
    s_centre = (1.0 - q_c) / np.maximum(0.774 - 0.03 * q_c, 1e-3)
    s_edge = (1.0 - q_e) / np.maximum(0.774 - 0.387 * q_e, 1e-3)
    per_corner = np.clip(np.minimum(s_centre, s_edge), 0.0, 1.0)
    s = float(per_corner.mean())
    if per_corner.min() < float(prm("vignette_agreement", 0.25)) * s:
        return 0.0
    return s if s >= float(prm("vignette_min", 0.08)) else 0.0


def _dominant_color(crop: np.ndarray) -> tuple[int, int, int]:
    """Mean colour around the peak of a smoothed 8x8x8 RGB histogram."""
    px = crop.reshape(-1, 3)
    if px.shape[0] > 20000:
        px = px[:: px.shape[0] // 20000 + 1]
    q = (px // 32).astype(int)
    hist = np.zeros((8, 8, 8))
    np.add.at(hist, (q[:, 0], q[:, 1], q[:, 2]), 1.0)
    peak = np.unravel_index(int(np.argmax(ndimage.uniform_filter(hist, size=3, mode="constant"))), hist.shape)
    near = (np.abs(q - np.array(peak)) <= 1).all(axis=1)
    c = px[near].mean(axis=0) if near.any() else px.mean(axis=0)
    return int(round(c[0])), int(round(c[1])), int(round(c[2]))
