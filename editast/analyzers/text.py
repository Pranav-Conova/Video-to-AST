"""Burned-in text overlays: captions, hooks, titles, labels and watermarks.

Overlay text is composited *on top of* the footage, so it stays pixel-still
while the footage underneath moves, cuts and zooms. The analyzer is built
around that invariant:

1. **Static-edge maps** (analysis resolution). At every text sample
   (``config.text_fps``, default 3 Hz) a pixel is a *static edge* when its
   morphological gradient is strong and its 3x3 neighbourhood is unchanged
   ``d`` frames (~0.1 s) earlier or later. Scene edges move away; overlay
   strokes survive. When most edges are static (tripod shot, freeze frame) the
   sample is *frozen* and the cue is uninformative.
2. **Candidates.** Static edges are closed horizontally into line blobs and
   filtered by height (1.2-15 % of the frame), aspect and density.
3. **Detection.** With OCR (RapidOCR, optional), samples whose candidate
   regions changed since the last OCR'd sample are decoded at
   ``config.text_long_side`` and OCR'd on the bands of rows holding the
   candidates (the whole frame when frozen); unchanged samples reuse the
   previous result, and at most ``max_ocr_frames`` frames are OCR'd. Boxes are
   merged into lines and lines into blocks; lines whose edges are not static
   (scene text moving with the camera) are dropped. Without OCR the candidates
   themselves are the detections, provided they are clearly stiller than
   their surroundings.
4. **Tracking.** Detections are linked across samples by place (IoU) and
   identity (normalised edit distance of the OCR text, or pixel agreement in
   both directions).
5. **Refinement.** Each track gets a template (per-pixel median of its sample
   patches) and the mask of its static stroke pixels, split into bright and
   dark classes. Start/end move frame by frame while the template still
   matches: every class agrees with the frame, or (under flashes and fades,
   where only the text's own contrast survives) the frame correlates with
   the template. Flat frames (white flash, black dip) are neutral. Consecutive
   tracks in one place are split at the frame that best separates them.
6. **Probing.** Overlays shorter than the sampling step (word-by-word
   captions) can fall between samples. Extra samples are taken in the gaps
   between consecutive overlays in one place and half a step before / after
   every overlay; whatever they find is tracked and refined with the rest.
7. **Interpretation.** Runs of short consecutive overlays in one place merge
   into one ``word_by_word`` overlay (text growing by prefix: ``typewriter``);
   a scaled copy of the template just before the start means ``pop``, a
   contrast ramp ``fade``. Style comes from the pixels that stay still over
   the overlay's lifetime: the deepest class is the fill (colour, glyph
   height); a band around it that stays dark (light) while the footage moves
   is an outline. ``size`` and ``bbox`` follow the layout of
   :func:`editast.fx.draw_text`, which the renderer uses, so re-rendering an
   analysed overlay reproduces its size and place. Role comes from timing,
   position, size and screen coverage.
"""

from __future__ import annotations

import functools
import math
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np
from scipy.ndimage import median_filter

from editast import fx
from editast.analyzers.types import ShotSegmentation, TextAnalysis
from editast.ast.nodes import RGB, TextAnimation, TextOverlay, TextPosition, TextRole, Word
from editast.context import AnalysisContext
from editast.media import ffmpeg_exe, fit_size

NAME = "text"

_K3 = np.ones((3, 3), np.uint8)
_K5 = np.ones((5, 5), np.uint8)
_OCR_MIN_SIDE = 640  # frames whose long side is smaller are upscaled before OCR
_OCR_TEXT_PX = 40  # small text is upscaled to about this height before OCR
# Scales tried for pop-in frames: geometric steps of _POP_STEP (log scale), 0.3x .. 1.4x,
# including exactly 1 (the full-size text).
_POP_STEP = 0.04
_POP_SCALES = np.exp(_POP_STEP * np.arange(-30, 9))

Box = tuple[float, float, float, float]  # (x0, y0, x1, y1) in analysis-frame pixels, end exclusive


# --------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class _Params:
    stable_frames: int  # frame distance of the static-edge test
    stable_tol: float  # max colour change of a static pixel (3x3 neighbourhood)
    edge_thr: float  # min morphological colour gradient of an edge pixel
    frozen_ratio: float  # static/all footage edge ratio above which a sample is "frozen"
    frozen_diff: float  # mean colour change of edge-poor footage below which a sample is "frozen"
    frozen_change: float  # block-mean grey change that invalidates OCR reuse on frozen samples
    join: float  # horizontal closing width joining characters (fraction of frame height)
    min_h: float  # text line height limits (fraction of frame height)
    max_h: float
    min_stable: float  # min static-edge support of an accepted box
    min_contrast: float  # OCR-less: box support minus surrounding support
    min_score: float  # min OCR confidence
    max_ocr: int  # OCR budget (frames) per video
    max_line_reads: int  # single-line reads of candidates the detector missed, per sample
    presence_tol: float  # grey tolerance of the template presence test
    presence_ratio: float  # presence threshold relative to the in-track level
    presence_floor: float
    min_corr: float  # template correlation that counts as present under flashes / fades
    min_gain: float  # ... with at least this fraction of the template's contrast
    max_gap: int  # samples a track may miss and still continue
    pop_corr: float  # min correlation of a pop-in frame with the scaled template
    pop_scale: float  # min |log scale| of the smallest (largest) pop-in frame
    probe_rounds: int  # rounds of extra samples around / between overlays
    min_duration: float  # seconds
    still_tol: float  # max grey range over an overlay's lifetime of a "still" pixel
    outline_contrast: float  # min value step between fill and outline band
    word_max: float  # seconds; longer items do not form word-by-word runs
    word_gap: float  # seconds; max pause between consecutive words of one caption
    word_run: int  # min items of a word-by-word run
    hook_time: float  # seconds
    watermark_cover: float  # min fraction of the video a watermark is on screen
    watermark_size: float  # max size (line height / frame height) of a watermark
    title_size: float  # min size of a title
    label_size: float  # max size of a label

    @classmethod
    def from_context(cls, ctx: AnalysisContext) -> "_Params":
        def p(key: str, default: Any) -> Any:
            return ctx.config.p(NAME, key, default)

        return cls(
            stable_frames=int(p("stable_frames", max(2, min(6, round(0.1 * ctx.fps))))),
            stable_tol=float(p("stable_tol", 26.0)),
            edge_thr=float(p("edge_thr", 64.0)),
            frozen_ratio=float(p("frozen_ratio", 0.6)),
            frozen_diff=float(p("frozen_diff", 2.0)),
            frozen_change=float(p("frozen_change", 10.0)),
            join=float(p("join", 0.025)),
            min_h=float(p("min_height", 0.012)),
            max_h=float(p("max_height", 0.15)),
            min_stable=float(p("min_stable", 0.4)),
            min_contrast=float(p("min_contrast", 0.3)),
            min_score=float(p("min_score", 0.5)),
            max_ocr=int(p("max_ocr_frames", 200)),
            max_line_reads=int(p("max_line_reads", 3)),
            presence_tol=float(p("presence_tol", 30.0)),
            presence_ratio=float(p("presence_ratio", 0.6)),
            presence_floor=float(p("presence_floor", 0.35)),
            min_corr=float(p("min_corr", 0.8)),
            min_gain=float(p("min_gain", 0.35)),
            max_gap=int(p("max_gap", 2)),
            pop_corr=float(p("pop_corr", 0.6)),
            pop_scale=abs(math.log(float(p("pop_scale", 0.85)))),
            probe_rounds=int(p("probe_rounds", 2)),
            min_duration=float(p("min_duration", 0.15)),
            still_tol=float(p("still_tol", 40.0)),
            outline_contrast=float(p("outline_contrast", 90.0)),
            word_max=float(p("word_max_duration", 1.0)),
            word_gap=float(p("word_gap", 0.35)),
            word_run=int(p("word_min_run", 3)),
            hook_time=float(p("hook_time", 2.0)),
            watermark_cover=float(p("watermark_coverage", 0.7)),
            watermark_size=float(p("watermark_size", 0.05)),
            title_size=float(p("title_size", 0.065)),
            label_size=float(p("label_size", 0.045)),
        )


# --------------------------------------------------------------------------- data


@dataclass(eq=False)
class _Sample:
    frame: int  # analysis frame index
    static: np.ndarray  # bool (h, w): strong edge with an unchanged neighbourhood d frames away
    edges: np.ndarray  # bool (h, w): strong edge
    frozen: bool  # most edges static: stillness does not single out overlays
    cands: list[Box]
    focus: list[Box] = field(default_factory=list)  # OCR reads the rows of these boxes (see _bands)
    grad: Optional[np.ndarray] = None  # uint8 (h, w): colour gradient, max over channels


@dataclass(eq=False)
class _Crop:
    """The part of an OCR'd (usually high-resolution) frame around a detection."""

    image: np.ndarray  # RGB
    x0: float  # origin, analysis px
    y0: float
    sx: float  # crop px per analysis px
    sy: float


@dataclass(eq=False)
class _Det:
    box: Box
    text: Optional[str]
    score: float  # OCR confidence, or static support for OCR-less detections
    lines: list[Box]  # member text-line boxes
    sample: Optional[_Sample] = None
    crop: Optional[_Crop] = None  # OCR'd detections: the frame around them, for the style
    texts: list[str] = field(default_factory=list)  # OCR text of each of ``lines`` (when read)

    @property
    def frame(self) -> int:
        assert self.sample is not None
        return self.sample.frame


@dataclass(eq=False)
class _Track:
    dets: list[_Det]
    misses: int = 0
    tmpl: Optional["_Template"] = None
    start: int = 0  # refined first frame
    end: int = 0  # refined end frame (exclusive)
    core: tuple[int, int] = (0, 0)  # [start, end) fully visible (without pop / fade frames)
    animation: TextAnimation = "static"
    p_in: float = 0.0  # template presence at the track's own samples

    @property
    def first(self) -> int:
        return self.dets[0].frame

    @property
    def last(self) -> int:
        return self.dets[-1].frame

    @property
    def box(self) -> Box:
        return _union([d.box for d in self.dets])


@dataclass
class _Item:
    """An overlay under construction (analysis-pixel geometry, frame timing)."""

    start: int
    end: int
    box: Box  # ink box
    text: Optional[str]
    fill_h: float  # glyph fill height, px
    color: Optional[RGB]
    outline: Optional[bool]
    animation: TextAnimation
    confidence: float
    tmpl: Optional["_Template"] = None  # None for merged items
    words: list[Word] = field(default_factory=list)
    lines: list[tuple[Box, str]] = field(default_factory=list)  # text lines (box, text), when read
    big: float = 0.0  # blocks of several overlays: fill height of the largest, px


class _Budget:
    """Remaining OCR calls."""

    def __init__(self, n: int) -> None:
        self.left = max(0, n)

    def take(self) -> bool:
        if self.left <= 0:
            return False
        self.left -= 1
        return True


# --------------------------------------------------------------------------- geometry helpers


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersect(a: Box, b: Box) -> Box:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def _iou(a: Box, b: Box) -> float:
    inter = _area(_intersect(a, b))
    return inter / max(_area(a) + _area(b) - inter, 1e-9)


def _overlapping(a: Box, b: Box) -> bool:
    """Boxes share at least 30 % of the smaller one (same place on screen)."""
    return _area(_intersect(a, b)) >= 0.3 * min(_area(a), _area(b))


def _same_band(a: Box, b: Box) -> bool:
    """Vertical overlap of at least half the shorter box."""
    return min(a[3], b[3]) - max(a[1], b[1]) >= 0.5 * min(a[3] - a[1], b[3] - b[1])


def _ibox(b: Box, shape: tuple[int, ...], pad: int = 0) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x0 = int(np.clip(math.floor(b[0]) - pad, 0, w - 1))
    y0 = int(np.clip(math.floor(b[1]) - pad, 0, h - 1))
    x1 = int(np.clip(math.ceil(b[2]) + pad, x0 + 1, w))
    y1 = int(np.clip(math.ceil(b[3]) + pad, y0 + 1, h))
    return x0, y0, x1, y1


def _blur(patch: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(patch.astype(np.float32), (3, 3), 0)


def _channel_max(img: np.ndarray) -> np.ndarray:
    """Per-pixel max over the channels of a ``uint8`` image (~30x faster than ``max(axis=2)``)."""
    c = cv2.split(img)
    return cv2.max(cv2.max(c[0], c[1]), c[2])


def _otsu(values: np.ndarray) -> float:
    v = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


@functools.lru_cache(maxsize=1)
def _draw_metrics() -> tuple[float, float, float]:
    """Layout of :func:`fx.draw_text` (the renderer's text primitive) per unit of
    glyph fill height: ``(line height, horizontal box padding, vertical box padding)``.

    The renderer reads ``TextOverlay.size`` as the box height of ``"Hg"`` and
    draws glyphs centred on the bbox centre, so reporting the observed fill
    height and ink box in these units makes a re-render match the original.
    """
    canvas = np.zeros((1000, 500, 3), np.uint8)
    drawn, (bx, by, bw, bh) = fx.draw_text(canvas, "HWH", (0.5, 0.5), 0.05, outline=False)
    _, line = fx.draw_text(canvas, "Hg", (0.5, 0.5), 0.05, outline=False)
    value = drawn.max(axis=2)
    h, w = value.shape
    ys, xs = np.nonzero(value > 0)
    rows = np.flatnonzero((value >= 128).any(axis=1))
    fill_h = float(rows[-1] - rows[0] + 1)
    pad_x = 0.5 * ((xs.min() - bx * w) + ((bx + bw) * w - xs.max() - 1))
    pad_y = 0.5 * ((ys.min() - by * h) + ((by + bh) * h - ys.max() - 1))
    return line[3] * h / fill_h, max(0.0, pad_x) / fill_h, max(0.0, pad_y) / fill_h


# --------------------------------------------------------------------------- text helpers


def _norm(text: Optional[str]) -> str:
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


@functools.lru_cache(maxsize=4096)
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """1 - normalised edit distance of two normalised strings."""
    if not a or not b:
        return 0.0
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b))


# Glyphs OCR confuses on bold caption fonts; folded before comparing two readings.
_CONFUSABLE = str.maketrans("0158", "OISB")


def _same_text(a: Optional[str], b: Optional[str]) -> bool:
    """Are two OCR readings the same text? Up to common glyph confusions (``B00M`` is
    ``BOOM``) and a small edit distance, but numbers must agree: ``PART 16`` and
    ``PART 18`` are successive captions, not one misread caption."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    da, db = "".join(c for c in na if c.isdigit()), "".join(c for c in nb if c.isdigit())
    if da and db and da != db:
        return False
    fa, fb = na.translate(_CONFUSABLE), nb.translate(_CONFUSABLE)
    need = 1.0 if min(len(fa), len(fb)) <= 3 else 0.7
    if abs(len(fa) - len(fb)) > (1.0 - need) * max(len(fa), len(fb)):
        return False  # the length difference alone costs more edits than allowed
    return _similarity(fa, fb) >= need


# --------------------------------------------------------------------------- 1-2. static edges, candidates


def _sample_rate(ctx: AnalysisContext) -> float:
    return float(min(max(ctx.config.text_fps, 1e-3), ctx.fps))


def _static_edges(frames: np.ndarray, f: int, prm: _Params
                  ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """``(static, edges, diffs, grad)`` of RGB frame ``f``: strong colour edges, those
    whose 3x3 neighbourhood is unchanged ``stable_frames`` earlier or later, the
    per-pixel colour change to those reference frames, and the colour gradient.

    Colour (max over channels) rather than grey: yellow captions over orange or
    red footage have little luma contrast but a large blue / green step.
    """
    rgb = frames[f]
    grad = _channel_max(cv2.morphologyEx(rgb, cv2.MORPH_GRADIENT, _K3))
    edges = grad >= prm.edge_thr
    refs = [o for o in (f - prm.stable_frames, f + prm.stable_frames) if 0 <= o < frames.shape[0]]
    if not refs:
        return edges.copy(), edges, [], grad
    stable = np.zeros(edges.shape, bool)
    diffs = [_channel_max(cv2.absdiff(rgb, frames[o])) for o in refs]
    for d in diffs:
        # Dilated difference: an edge whose neighbourhood moved by >= 1 px is not static,
        # even when the pixel itself sits inside a flat region that kept its value.
        stable |= cv2.dilate(d, _K3) <= prm.stable_tol
    return edges & stable, edges, diffs, grad


def _candidates(static: np.ndarray, prm: _Params) -> list[Box]:
    """Text-line blobs of the static-edge map."""
    h, w = static.shape
    kw = max(3, int(round(prm.join * h)))
    m = static.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K3)
    _, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    st = stats[1:, :4]
    bx, bw, bh = st[:, 0], st[:, 2], st[:, 3]
    # Blobs touching the left / right border are footage artefacts (padding, reflections).
    ok = ((bh >= max(prm.min_h * h, 5.0)) & (bh <= prm.max_h * h) & (bw >= 0.8 * bh) & (bw <= 30 * bh)
          & (bx > 0) & (bx + bw < w))
    boxes = []
    for x, y, cw, ch in st[ok]:
        if static[y:y + ch, x:x + cw].mean() >= 0.08:
            boxes.append((float(x), float(y), float(x + cw), float(y + ch)))
    return boxes


def _is_frozen(prm: _Params, static: np.ndarray, edges: np.ndarray, diffs: list[np.ndarray],
               cands: list[Box]) -> bool:
    """Is the footage itself still? Judged outside the candidate boxes, so a large
    caption over plain footage does not make the whole sample look frozen: most
    footage edges are static or, when the footage has hardly any edges, its
    pixels barely change."""
    if not diffs:
        return True
    footage = ~_boxes_mask(cands, static.shape, pad=2)
    n_edges = int(np.count_nonzero(edges & footage))
    if n_edges >= max(50, 0.002 * footage.size):
        return int(np.count_nonzero(static & footage)) >= prm.frozen_ratio * n_edges
    if not footage.any():
        return True
    return min(float(d[footage].mean()) for d in diffs) <= prm.frozen_diff


def _make_sample(ctx: AnalysisContext, prm: _Params, f: int) -> _Sample:
    static, edges, diffs, grad = _static_edges(ctx.frames, f, prm)
    cands = _candidates(static, prm)
    return _Sample(f, static, edges, _is_frozen(prm, static, edges, diffs, cands), cands, grad=grad)


def _scan(ctx: AnalysisContext, prm: _Params) -> list[_Sample]:
    """Regular samples at ``config.text_fps``: sample ``k`` is the analysis frame
    shown at ``k / text_fps`` seconds (and frame ``k`` of a decode at that rate)."""
    n = ctx.n_frames
    step = ctx.fps / _sample_rate(ctx)
    count = int(math.floor((n - 1) / step + 1e-6)) + 1
    return [_make_sample(ctx, prm, min(n - 1, int(round(k * step)))) for k in range(count)]


def _support(s: _Sample, box: Box) -> float:
    """Fraction of the edge pixels inside ``box`` that are static."""
    x0, y0, x1, y1 = _ibox(box, s.static.shape)
    e = int(s.edges[y0:y1, x0:x1].sum())
    return float(s.static[y0:y1, x0:x1].sum()) / e if e else 0.0


def _stiller_than_ring(s: _Sample, box: Box, prm: _Params) -> bool:
    """Is ``box`` clearly stiller than a ring around it (one box height wide)?

    The ring starts 2 px out: footage pixels right next to an overlay's border are
    edges because of the overlay itself. A ring with hardly any edges of its own
    (text over sky, a wall, a blurred background) says nothing about the footage
    there; then the box counts as stiller when the footage moves elsewhere (the
    sample is not frozen). Text on a caption box is tested again with the box:
    its static border lies in the ring of the text alone.
    """
    h = box[3] - box[1]
    shape = s.static.shape
    x0, y0, x1, y1 = _ibox(box, shape, pad=int(math.ceil(0.5 * h)))
    ys, xs = np.nonzero(s.static[y0:y1, x0:x1])
    hull = _union([box, (float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1),
                         float(y0 + ys.max() + 1))]) if xs.size else box
    for b in dict.fromkeys((box, hull)):
        inner = _boxes_mask([b], shape, pad=2)
        ring = _boxes_mask([b], shape, pad=int(math.ceil(h))) & ~inner
        e = int(np.count_nonzero(s.edges & ring))
        if e < max(20, 0.1 * int(np.count_nonzero(s.edges & inner))):
            if not s.frozen:
                return True
        elif _support(s, box) - np.count_nonzero(s.static & ring) / e >= prm.min_contrast:
            return True
    return False


# --------------------------------------------------------------------------- line / block grouping


def _same_line(a: Box, b: Box) -> bool:
    ha, hb = a[3] - a[1], b[3] - b[1]
    lo, hi = min(ha, hb), max(ha, hb)
    v_overlap = min(a[3], b[3]) - max(a[1], b[1])
    gap = max(a[0], b[0]) - min(a[2], b[2])
    return v_overlap >= 0.5 * lo and hi <= 1.6 * lo and gap <= 1.2 * hi


def _stacked(a: _Det, b: _Det) -> bool:
    """``b`` is the next line of the same text block as ``a``."""
    if b.box[1] < a.box[1]:
        a, b = b, a
    ha = float(np.median([ln[3] - ln[1] for ln in a.lines]))
    hb = float(np.median([ln[3] - ln[1] for ln in b.lines]))
    lo, hi = min(ha, hb), max(ha, hb)
    gap = b.box[1] - a.box[3]
    overlap = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
    narrow = min(a.box[2] - a.box[0], b.box[2] - b.box[0])
    return hi <= 1.5 * lo and -0.35 * lo <= gap <= 0.8 * lo and overlap >= 0.3 * narrow


def _join(a: _Det, b: _Det, block: bool) -> _Det:
    axis = 1 if block else 0
    first, second = (a, b) if a.box[axis] <= b.box[axis] else (b, a)
    text = None
    if first.text is not None or second.text is not None:
        # Stacked lines of a block keep their line breaks; pieces of one line join with a space.
        text = ("\n" if block else " ").join(t for t in (first.text, second.text) if t)
    wa, wb = len(_norm(first.text)) or 1, len(_norm(second.text)) or 1
    score = (first.score * wa + second.score * wb) / (wa + wb)
    box = _union([a.box, b.box])
    lines = first.lines + second.lines if block else [box]
    texts = (first.texts + second.texts if block else [text]) if text is not None else []
    return _Det(box=box, text=text, score=score, lines=lines, texts=texts)


def _group(dets: list[_Det], block: bool) -> list[_Det]:
    dets = list(dets)
    merged = True
    while merged:
        merged = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                hit = _stacked(dets[i], dets[j]) if block else _same_line(dets[i].box, dets[j].box)
                if hit:
                    dets[i] = _join(dets[i], dets[j], block)
                    del dets[j]
                    merged = True
                    break
            if merged:
                break
    return dets


def _blocks(lines: list[_Det]) -> list[_Det]:
    """Merge fragments of one text line, then stacked lines into blocks."""
    return _group(_group(lines, block=False), block=True)


# --------------------------------------------------------------------------- 3. detection (OCR / classical)


def _respace(chars: list[str], xs: np.ndarray, crop: np.ndarray) -> str:
    """Join recognised characters (centres ``xs``, crop px), restoring the word spaces
    the recognizer dropped ("COMINGSOON"): a space goes between two characters
    whose centres straddle an ink-free column run that is wide for this line,
    at least 0.3 ink heights and 1.8 times its median gap. Relative to the line's
    own gaps, so widely tracked lettering ("S I N T E L") gets no spaces."""
    text = "".join(chars)
    if len(chars) < 2 or crop.size == 0:
        return text
    # Binarise the channel (grey, B, G or R) that splits most cleanly into two classes
    # (Otsu's between-class / total variance): yellow text on a light sky has almost
    # the sky's grey level but none of its blue.
    planes = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)] + list(cv2.split(crop)) if crop.ndim == 3 else [crop]
    best, bright = -1.0, None
    for plane in planes:
        t, _ = cv2.threshold(plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hi = plane > t
        v = plane.astype(np.float32)
        w1 = float(hi.mean())
        if 0.0 < w1 < 1.0 and float(v.var()) > 0:
            sep = w1 * (1 - w1) * (float(v[hi].mean()) - float(v[~hi].mean())) ** 2 / float(v.var())
            if sep > best:
                best, bright = sep, hi
    if bright is None:
        return text
    ink = bright if bright.mean() < 0.5 else ~bright  # text is the minority class
    h = ink.shape[0]
    cols = ink[int(0.15 * h):max(int(0.15 * h) + 1, int(math.ceil(0.85 * h)))].any(axis=0)
    on = np.flatnonzero(cols)
    ink_h = int(ink.any(axis=1).sum())
    if on.size < 2 or ink_h < 4:
        return text
    gaps = np.flatnonzero(np.diff(on) > 1)
    runs = [(int(on[i]) + 1, int(on[i + 1])) for i in gaps]  # [start, end) of empty columns
    if not runs:
        return text
    thr = max(0.3 * ink_h, 1.8 * float(np.median([e - b for b, e in runs])), 2.0)
    wide = [0.5 * (b + e) for b, e in runs if e - b >= thr]
    out = [chars[0]]
    for i in range(1, len(chars)):
        if chars[i] != " " and chars[i - 1] != " " and any(xs[i - 1] < c < xs[i] for c in wide):
            out.append(" ")
        out.append(chars[i])
    return "".join(out)


def _ctc_text(pred: np.ndarray, charset: list[str], crop: np.ndarray, width: int, height: int) -> tuple[str, float]:
    """Greedy CTC decode of one recognizer output ``(T, classes)`` (as rapidocr does),
    with the characters' positions used to restore word spaces (:func:`_respace`).
    The input was ``crop`` resized to ``height`` and padded to ``width``."""
    k = pred.argmax(axis=1)
    keep = (k != 0) & np.r_[True, k[1:] != k[:-1]]
    ts = np.flatnonzero(keep)
    if ts.size == 0:
        return "", 0.0
    # Mean over n + 1 values, the extra one ~0: rapidocr's score, which the thresholds
    # (min_score) were set against; it discounts short readings.
    score = float(pred.max(axis=1)[ts].sum() / (ts.size + 1))
    h, w = crop.shape[:2]
    resized = max(1, min(width, int(math.ceil(height * w / max(h, 1)))))
    xs = (ts + 0.5) * (width / len(k)) * (w / resized)
    return _respace([charset[int(c)] for c in k[ts]], xs, crop), score


class _Ocr:
    """RapidOCR wrappers: ``det`` finds and reads lines; ``rec`` (optional) reads a
    whole crop as one line, for candidates the detector missed."""

    def __init__(self, det: Any, rec: Any) -> None:
        self.det = det
        self.rec = rec
        # Detection and recognition as separate steps (rapidocr internals), so boxes can
        # be filtered in between; otherwise the whole pipeline runs at once.
        self.staged = all(hasattr(det, a) for a in ("text_detector", "sorted_boxes", "get_crop_img_list",
                                                    "text_recognizer"))

    @property
    def positions(self) -> bool:
        """Can recognition run step by step (rapidocr internals), keeping the
        character positions?"""
        r = getattr(self.det, "text_recognizer", None)
        return (self.staged and all(hasattr(r, a) for a in ("session", "resize_norm_img", "rec_batch_num",
                                                           "postprocess_op"))
                and hasattr(r.postprocess_op, "character"))

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """``[(text, score)]`` of BGR line crops, with word spaces restored when the
        character positions are available."""
        rec = self.det.text_recognizer
        if not self.positions:
            return [(str(t).strip(), float(sc)) for t, sc in rec(crops)[0]]
        charset = rec.postprocess_op.character
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        order = np.argsort([c.shape[1] / max(c.shape[0], 1) for c in crops], kind="stable")
        for b0 in range(0, len(crops), rec.rec_batch_num):
            idx = order[b0:b0 + rec.rec_batch_num]
            ratio = max(crops[i].shape[1] / max(crops[i].shape[0], 1) for i in idx)
            batch = np.concatenate([rec.resize_norm_img(crops[i], ratio)[np.newaxis] for i in idx]).astype(np.float32)
            preds = rec.session(batch)[0]
            for i, pred in zip(idx, preds):
                text, score = _ctc_text(pred, charset, crops[i], batch.shape[3], batch.shape[2])
                out[i] = (text.strip(), score)
        return out

    @staticmethod
    def _run(engine: Any, rgb: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        result, _ = engine(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
        return [(np.asarray(q, float), str(t).strip(), float(s)) for q, t, s in result or []]

    def read(self, rgb: np.ndarray, keep: Optional[Callable[[np.ndarray], bool]] = None
             ) -> list[tuple[np.ndarray, str, float]]:
        """``[(quad (4, 2) px, text, score)]`` for every text line in ``rgb``. Only boxes
        passing ``keep(quad)`` are recognised (recognition dominates the cost when
        busy footage yields many boxes)."""
        if not self.staged:
            return [r for r in self._run(self.det, rgb) if keep is None or keep(r[0])]
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        boxes, _ = self.det.text_detector(bgr)
        if boxes is None or len(boxes) == 0:
            return []
        quads = [q for q in self.det.sorted_boxes(np.asarray(boxes))
                 if keep is None or keep(np.asarray(q, float))]
        if not quads:
            return []
        rec = self.recognize(self.det.get_crop_img_list(bgr, quads))
        return [(np.asarray(q, float), t, sc) for q, (t, sc) in zip(quads, rec)]

    def read_line(self, rgb: np.ndarray) -> tuple[str, float]:
        if self.positions:
            return self.recognize([cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)])[0]
        res = self._run(self.rec, rgb) if self.rec is not None else []
        return (res[0][1], res[0][2]) if res else ("", 0.0)


_OCR: Optional[_Ocr] = None
_OCR_UNAVAILABLE = False


def _load_ocr() -> Optional[_Ocr]:
    """Module-level OCR engines (loaded once, ~1 s), or ``None`` when unavailable."""
    global _OCR, _OCR_UNAVAILABLE
    if _OCR is None and not _OCR_UNAVAILABLE:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception:  # not installed, or onnxruntime failing to load
            _OCR_UNAVAILABLE = True
            return None
        try:
            # Overlays are upright: no angle classifier. Bands are upscaled here, so the
            # detector must not resize them, and must run even on thin bands. (rapidocr's
            # parameter updater needs an explicit ``det_model_path`` to accept det options.)
            det = RapidOCR(det_model_path=None, det_limit_side_len=1600, det_limit_type="max",
                           use_angle_cls=False, min_height=0, width_height_ratio=-1, text_score=0.0)
            rec = RapidOCR(use_text_det=False, use_angle_cls=False, text_score=0.0)
        except Exception:  # other rapidocr versions take other keyword names
            det, rec = RapidOCR(), None
        _OCR = _Ocr(det, rec)
    return _OCR


def _boxes_mask(boxes: list[Box], shape: tuple[int, ...], pad: int = 0) -> np.ndarray:
    m = np.zeros(shape[:2], bool)
    for b in boxes:
        x0, y0, x1, y1 = _ibox(b, shape, pad)
        m[y0:y1, x0:x1] = True
    return m


def _blank(s: _Sample) -> bool:
    """Hardly any edge in the frame (black, white or flat): nothing to read."""
    return int(np.count_nonzero(s.edges)) < 8


def _same_pixels(ctx: AnalysisContext, prm: _Params, r: _Sample, s: _Sample, box: Box) -> bool:
    """Do the static pixels of ``s`` in ``box`` look as they did at ``r``? Judged per
    glyph-sized column chunk: one changed character ("PART 16" -> "PART 18") is a
    small fraction of the line but most of its own chunk."""
    gr, gs = ctx.gray[r.frame], ctx.gray[s.frame]
    x0, y0, x1, y1 = _ibox(box, gs.shape)
    px = s.static[y0:y1, x0:x1]
    if px.sum() < 8:
        return True
    same = cv2.absdiff(gr[y0:y1, x0:x1], gs[y0:y1, x0:x1]) <= prm.stable_tol
    cols = np.arange(0, x1 - x0, max(2, y1 - y0))
    n_px = np.add.reduceat(px.sum(axis=0), cols)
    n_ok = np.add.reduceat((px & same).sum(axis=0), cols)
    dense = n_px >= 8
    return not (n_ok.sum() < 0.85 * n_px.sum() or (dense.any() and (n_ok[dense] / n_px[dense]).min() < 0.7))


def _same_look(ctx: AnalysisContext, r: _Sample, s: _Sample, box: Box, shift: int = 1) -> bool:
    """Does ``box`` show the same content at ``r`` and ``s`` up to a contrast change
    (fade, grade) and ``shift`` px of drift (slow zoom)? Per glyph-sized column
    chunk, the best normalised correlation over the shifts must be high, and a
    chunk keeps or lacks texture at both: a replaced word changes the letters of
    some chunk entirely."""
    gr, gs = ctx.gray[r.frame], ctx.gray[s.frame]
    h, w = gs.shape
    x0, y0, x1, y1 = _ibox(box, gs.shape)
    step = max(4, y1 - y0)
    for cx in range(x0, x1, step):
        cw = min(step, x1 - cx)
        if cw < 3:
            continue
        a = gs[y0:y1, cx:cx + cw].astype(np.float32)
        b = gr[y0:y1, cx:cx + cw].astype(np.float32)
        ta, tb = float(a.std()) >= 8.0, float(b.std()) >= 8.0
        if ta != tb:
            return False
        if not ta:
            continue
        ry0, ry1 = max(0, y0 - shift), min(h, y1 + shift)
        rx0, rx1 = max(0, cx - shift), min(w, cx + cw + shift)
        b = gr[ry0:ry1, rx0:rx1].astype(np.float32)
        corr = np.nan_to_num(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED), nan=0.0)
        if float(corr.max()) < 0.8:
            return False
    return True


# Change test over still footage: an edge appeared (vanished) when its strength exceeds
# twice the strongest edge within 1 px at the other sample plus this margin. Ratio and
# margin let fades (contrast scaling) and compression noise pass; the 1 px slack lets
# slow drift pass.
_CHANGE_RATIO = 2.0
_CHANGE_MARGIN = 32


def _changed(ctx: AnalysisContext, prm: _Params, ref: np.ndarray, p: _Sample, s: _Sample) -> list[Box]:
    """Where text may differ at still sample ``s`` from what was read: text-line blobs
    of the edges that appeared in ``s`` or vanished since the gradient ``ref``
    (each row as it was when last read), and candidate lines of ``s`` that look
    different from the previous sample ``p`` (a word replaced in place adds and
    removes only fragments of letters). The look test is contrast-invariant, so
    it compares neighbours, where drift (a slow zoom) is smallest; the edge test
    is not, so it compares with the last read of each row, letting a slow
    fade-in add up until it matters even while other rows are read.

    Over still footage stillness cannot single out an overlay, but an overlay
    that appears, changes or leaves adds or removes edges of its own shape, while
    drift, fades and a moving subject do not form text lines."""
    assert s.grad is not None
    gr, gs = ref.astype(np.int16), s.grad.astype(np.int16)
    near_r = cv2.dilate(ref, _K3).astype(np.int16)
    near_s = cv2.dilate(s.grad, _K3).astype(np.int16)
    new = (gs >= prm.edge_thr) & (gs > _CHANGE_RATIO * near_r + _CHANGE_MARGIN)
    lost = (gr >= prm.edge_thr) & (gr > _CHANGE_RATIO * near_s + _CHANGE_MARGIN)
    lines = [b for b in _candidates(new | lost, prm) if b[2] - b[0] >= 1.5 * (b[3] - b[1])]
    return lines + [c for c in s.cands if not _same_look(ctx, p, s, c)]


def _spans(boxes: list[Box], ah: int, aw: int = 0) -> Optional[list[tuple[float, float, float, float, float]]]:
    """Bands ``(y0, y1, text height, x0, x1)`` (analysis px) that OCR reads around
    ``boxes``; ``None`` for the whole frame (no boxes, or bands covering most of it).

    A band spans the rows of its boxes plus a line height (at least 3 % of the
    frame) above and below, and their columns plus three line heights (at least
    8 % of the width) to either side, or the full width (``aw`` = 0, or when that
    is most of it): text lines run horizontally, so a line partly found by the
    candidates still fits, while footage elsewhere in the rows is not read.
    """
    if not boxes:
        return None
    spans: list[list[Any]] = []
    for b in sorted(boxes, key=lambda b: b[1]):
        h = b[3] - b[1]
        pad = max(h, 0.03 * ah)
        y0, y1 = max(0.0, b[1] - pad), min(float(ah), b[3] + pad)
        if spans and y0 <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], y1)
            spans[-1][2].append(h)
            spans[-1][3].append(b)
        else:
            spans.append([y0, y1, [h], [b]])
    if sum(y1 - y0 for y0, y1, _, _ in spans) > 0.6 * ah:
        return None
    out = []
    for y0, y1, hs, bs in spans:
        th = float(np.median(hs))
        x0, x1 = 0.0, float(aw)
        if aw > 0:
            pad = max(3.0 * th, 0.08 * aw)
            x0, x1 = max(0.0, min(b[0] for b in bs) - pad), min(float(aw), max(b[2] for b in bs) + pad)
            if x1 - x0 > 0.7 * aw:
                x0, x1 = 0.0, float(aw)
        out.append((float(y0), float(y1), th, x0, x1))
    return out


def _rows(boxes: list[Box], ah: int) -> np.ndarray:
    """Bool ``(ah,)``: analysis rows that a read of ``boxes`` covers."""
    spans = _spans(boxes, ah)
    rows = np.zeros(ah, bool)
    if spans is None:
        rows[:] = True
    for y0, y1, *_ in spans or []:
        rows[int(math.floor(y0)):int(math.ceil(y1))] = True
    return rows


def _hits(box: Box, rows: np.ndarray) -> bool:
    return bool(rows[max(0, int(math.floor(box[1]))):int(math.ceil(box[3]))].any())


def _close(focus: list[Box], pool: list[Box], ah: int) -> list[Box]:
    """``focus`` plus the boxes of ``pool`` its read bands touch (to closure), so a
    partial read never cuts through a text block."""
    focus = list(focus)
    rest = [b for b in pool if b not in focus]
    while focus and rest:
        rows = _rows(focus, ah)
        add = [b for b in rest if _hits(b, rows)]
        if not add:
            break
        focus += add
        rest = [b for b in rest if b not in add]
    return focus


@dataclass
class _Step:
    """How a regular sample gets its detections.

    ``ocr``: read the bands of ``sample.focus`` (or its candidates; the whole frame
    when frozen without focus); outside the read ``rows`` the detections of sample
    ``ref`` still hold (``ref`` < 0: nothing there). ``reuse``: the detections of
    ``ref`` still hold. ``classical``: OCR-less detection. ``none``: no text.
    """

    kind: str
    ref: int = -1
    rows: Optional[np.ndarray] = None


def _plan_ocr(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], budget: _Budget) -> list[_Step]:
    """Read only what may have changed since the last read (the *reference*).

    Moving footage: candidates that the reference read covered and whose static
    pixels are unchanged keep the reference's reading; the others are read.
    Still footage (frozen): candidates say little, so the first sample of a run
    is read whole; later ones read only where text-shaped edges appeared or
    vanished since the reference (:func:`_changed`). Blank frames and moving
    footage without candidates hold no text and end the run. Comparing with the
    last *read* sample (not the previous sample) lets slow fades and drift add
    up until they matter.
    """
    ah = ctx.gray.shape[1]
    plan: list[_Step] = []
    ref: Optional[int] = None
    cover = np.zeros(ah, bool)  # rows the reference's detections describe
    seen: Optional[np.ndarray] = None  # gradient of each row when it was last read
    for k, s in enumerate(samples):
        if (not s.frozen and not s.cands) or (s.frozen and _blank(s)):
            plan.append(_Step("none"))
            ref = None
            continue
        r = samples[ref] if ref is not None else None
        if s.frozen:
            if r is None or not cover.all() or seen is None:
                focus, base = [], None  # whole frame
            else:
                focus, base = _close(_changed(ctx, prm, seen, samples[k - 1], s), s.cands, ah), ref
                if not focus:
                    plan.append(_Step("reuse", ref))
                    continue
        else:
            fresh = [c for c in s.cands
                     if r is None or not (cover[int(c[1]):int(math.ceil(c[3]))].all() and _same_pixels(ctx, prm, r, s, c))]
            if not fresh:
                plan.append(_Step("reuse", ref if ref is not None else -1))
                continue
            focus, base = _close(fresh, s.cands, ah), ref
        rows = _rows(focus, ah)
        if rows.all():
            focus, base = ([] if s.frozen else s.cands), None
        s.focus = focus
        plan.append(_Step("ocr", -1 if base is None else base, rows))
        cover = rows | (cover if base is not None else False)
        assert s.grad is not None
        if base is None or seen is None:
            seen = s.grad.copy()
        else:
            seen[rows] = s.grad[rows]
        ref = k
    todo = [k for k, st in enumerate(plan) if st.kind == "ocr"]
    if len(todo) > budget.left:
        # Over budget: OCR an even spread, the rest falls back to the classical detector
        # (as does every sample whose reading depends on a dropped one, see _detect).
        keep = {todo[i] for i in np.linspace(0, len(todo) - 1, budget.left).round().astype(int)} if budget.left else set()
        for k in set(todo) - keep:
            plan[k] = _Step("classical")
    budget.left -= sum(1 for st in plan if st.kind == "ocr")
    return plan


def _decode_selected(path: str, fps: float, size: tuple[int, int], frames: list[int],
                     duration: float) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame, RGB image at size)`` for the sorted analysis ``frames`` of ``path``.

    The source goes through the same ``fps`` filter as the analysis decode
    (:func:`editast.context.load_context`), so indices match it exactly, and only
    the selected frames are scaled and piped. (Decoding at the sampling rate
    instead would hand out the *last* source frame of each sampling slot, ~0.4
    sampling periods after the analysis frame it stands for.) Stops early when
    ffmpeg fails; callers fall back to the analysis frames for what is missing.
    """
    w, h = size
    pick = "+".join(f"eq(n,{f})" for f in frames)
    args = [ffmpeg_exe(), "-v", "error", "-nostdin", "-i", path, "-t", f"{duration:.4f}",
            "-vf", f"fps={fps:.6g},select='{pick}',scale={w}:{h}:flags=area", "-fps_mode", "passthrough",
            "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return
    assert proc.stdout is not None
    try:
        for f in frames:
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            yield f, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _iter_hires(ctx: AnalysisContext, samples: list[_Sample], wanted: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    """``(k, RGB frame)`` of the wanted samples at ``config.text_long_side`` when the
    source file is available (and sharper than the analysis frames), else analysis frames."""
    done: set[int] = set()
    path = Path(ctx.path)
    if wanted and ctx.info.width > 0 and ctx.info.height > 0 and path.is_file():
        size = fit_size(ctx.info.width, ctx.info.height, ctx.config.text_long_side)
        if max(size) >= 1.2 * max(ctx.size):
            by_frame = {samples[k].frame: k for k in wanted}
            for f, frame in _decode_selected(str(path), ctx.fps, size, sorted(by_frame), ctx.duration):
                done.add(by_frame[f])
                yield by_frame[f], frame
    for k in sorted(wanted - done):
        yield k, ctx.frames[samples[k].frame]


def _bands(s: _Sample, ah: int, aw: int, fh: int, fw: int) -> list[tuple[int, int, int, int, float]]:
    """Bands ``(y0, y1, x0, x1, text height)`` of a frame of size ``fw`` x ``fh`` around
    the focus boxes of ``s`` (its candidates when it has none and is not frozen);
    the whole frame, text height unknown = 0, when a frozen sample has no focus."""
    boxes = s.focus if (s.frozen or s.focus) else s.cands
    spans = _spans(boxes, ah, aw)
    if spans is None:
        return [(0, fh, 0, fw, 0.0)]
    sy, sx = fh / ah, fw / aw
    return [(max(0, int(y0 * sy)), min(fh, int(math.ceil(y1 * sy))), max(0, int(x0 * sx)),
             min(fw, int(math.ceil(x1 * sx))), h * sy) for y0, y1, h, x0, x1 in spans]


def _accept(prm: _Params, s: _Sample, box: Box, text: str, score: float, ah: int) -> bool:
    """OCR line filter: confident, readable, tall enough, and static (scene text
    moves with the camera; overlays do not)."""
    return (score >= prm.min_score and bool(_norm(text)) and box[3] - box[1] >= prm.min_h * ah
            and _support(s, box) >= prm.min_stable)


def _ocr_sample(ctx: AnalysisContext, prm: _Params, s: _Sample, frame: np.ndarray, ocr: _Ocr) -> list[_Det]:
    ah, aw = ctx.gray.shape[1:]
    fh, fw = frame.shape[:2]
    sx, sy = aw / fw, ah / fh
    base_up = max(1.0, _OCR_MIN_SIDE / max(fh, fw))
    bands = _bands(s, ah, aw, fh, fw)
    # The detector misses short words below ~40 px: upscale small text (at most 3x), but
    # never past the cost (pixels) of reading the whole frame.
    ups = [base_up if text_h <= 0 else float(np.clip(_OCR_TEXT_PX / text_h, base_up, 3.0)) for *_, text_h in bands]
    cost = sum((y1 - y0) * (x1 - x0) * up * up for (y0, y1, x0, x1, _), up in zip(bands, ups))
    if cost > fh * fw * base_up * base_up:
        shrink = math.sqrt(fh * fw * base_up * base_up / cost)
        ups = [max(base_up, up * shrink) for up in ups]
    lines: list[_Det] = []
    for (y0, y1, x0, x1, _), up in zip(bands, ups):
        crop = frame[y0:y1, x0:x1]
        if up > 1.0:
            crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)

        def to_box(quad: np.ndarray, up: float = up, y0: int = y0, x0: int = x0) -> Box:
            q = quad / up
            return (float((q[:, 0].min() + x0) * sx), float((q[:, 1].min() + y0) * sy),
                    float((q[:, 0].max() + x0) * sx), float((q[:, 1].max() + y0) * sy))

        def plausible(quad: np.ndarray) -> bool:
            box = to_box(quad)
            return box[3] - box[1] >= prm.min_h * ah and _support(s, box) >= prm.min_stable

        for quad, text, score in ocr.read(crop, keep=plausible):
            box = to_box(quad)
            if _accept(prm, s, box, text, score, ah):
                lines.append(_Det(box=box, text=text, score=score, lines=[box], texts=[text]))
    if not s.frozen:
        # Overlay-like candidates the detector did not cover (short words), largest first:
        # read each as one line. A candidate holding a detected line is covered (it is
        # that line plus footage).
        missed = [b for b in (s.focus or s.cands)
                  if not any(_area(_intersect(b, d.box)) >= 0.5 * min(_area(b), _area(d.box)) for d in lines)
                  and _support(s, b) >= prm.min_stable and _stiller_than_ring(s, b, prm)]
        for b in sorted(missed, key=_area, reverse=True)[:prm.max_line_reads]:
            h = (b[3] - b[1]) / sy
            x0, x1 = max(0, int(b[0] / sx - 0.4 * h)), min(fw, int(math.ceil(b[2] / sx + 0.4 * h)))
            y0, y1 = max(0, int(b[1] / sy - 0.25 * h)), min(fh, int(math.ceil(b[3] / sy + 0.25 * h)))
            crop = frame[y0:y1, x0:x1]
            if crop.shape[0] < _OCR_TEXT_PX:
                f = _OCR_TEXT_PX / crop.shape[0]
                crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            text, score = ocr.read_line(crop)
            if len(_norm(text)) >= 2 and _accept(prm, s, b, text, score, ah):
                lines.append(_Det(box=b, text=text, score=score, lines=[b], texts=[text]))
    blocks = _blocks(lines)
    for d in blocks:
        # The sharpest view of the block, kept for the style estimate (colour, outline, size).
        pad = 0.6 * float(np.median([ln[3] - ln[1] for ln in d.lines]))
        x0, y0 = max(0, int((d.box[0] - pad) / sx)), max(0, int((d.box[1] - pad) / sy))
        x1, y1 = min(fw, int(math.ceil((d.box[2] + pad) / sx))), min(fh, int(math.ceil((d.box[3] + pad) / sy)))
        d.crop = _Crop(frame[y0:y1, x0:x1].copy(), x0 * sx, y0 * sy, 1.0 / sx, 1.0 / sy)
    return blocks


def _classical_sample(prm: _Params, s: _Sample) -> list[_Det]:
    """OCR-less detections: candidates clearly stiller than their surroundings."""
    lines = []
    for b in s.cands:
        sup = _support(s, b)
        if sup >= prm.min_stable and _stiller_than_ring(s, b, prm):
            lines.append(_Det(box=b, text=None, score=sup, lines=[b]))
    return _blocks(lines)


def _stamp(dets: list[_Det], s: _Sample) -> list[_Det]:
    for d in dets:
        d.sample = s
    return dets


def _holds(ctx: AnalysisContext, prm: _Params, d: _Det, s: _Sample) -> bool:
    """Is detection ``d`` (of an earlier sample) still on screen at ``s``? Over still
    footage the change test already said so; over moving footage its static
    pixels must still agree."""
    return s.frozen or _agreement(ctx, prm, d, s.frame) >= 0.7


def _detect(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], ocr: Optional[_Ocr],
            budget: _Budget) -> dict[_Sample, list[_Det]]:
    """Detections of the regular samples (OCR'd, partly or wholly reused, or classical)."""
    if ocr is None:
        plan = [_Step("classical")] * len(samples)
    else:
        plan = _plan_ocr(ctx, prm, samples, budget)
    wanted = {k for k, st in enumerate(plan) if st.kind == "ocr"}
    read: dict[int, list[_Det]] = {}
    if ocr is not None:
        for k, frame in _iter_hires(ctx, samples, wanted):
            read[k] = _stamp(_ocr_sample(ctx, prm, samples[k], frame, ocr), samples[k])
    held: dict[int, list[_Det]] = {}  # per read sample: what is on screen (detections keep their own sample)
    out: dict[_Sample, list[_Det]] = {}
    for k, (s, st) in enumerate(zip(samples, plan)):
        dets: Optional[list[_Det]] = None
        base = held.get(st.ref) if st.ref >= 0 else []
        if st.kind == "ocr" and k in read and base is not None:
            assert st.rows is not None
            dets = read[k] + [d for d in base if not _hits(d.box, st.rows) and _holds(ctx, prm, d, s)]
            held[k] = dets
        elif st.kind == "reuse" and base is not None:
            dets = [d for d in base if _holds(ctx, prm, d, s)]
        elif st.kind == "none":
            dets = []
        if dets is None:
            dets = _classical_sample(prm, s)
        out[s] = _stamp([replace(d, lines=list(d.lines)) for d in dets], s)
    return out


# --------------------------------------------------------------------------- 4. tracking


def _agreement(ctx: AnalysisContext, prm: _Params, d: _Det, frame: int, exclude: Optional[Box] = None) -> float:
    """Fraction of ``d``'s static pixels (outside ``exclude``) whose grey value is
    unchanged in ``frame``; 1.0 when there is nothing to compare."""
    assert d.sample is not None
    shape = ctx.gray.shape[1:]
    x0, y0, x1, y1 = _ibox(d.box, shape)
    px = d.sample.static[y0:y1, x0:x1]
    if exclude is not None:
        px = px & ~_boxes_mask([exclude], shape)[y0:y1, x0:x1]
    if px.sum() < 8:
        return 1.0
    a = _blur(ctx.gray[d.frame][y0:y1, x0:x1])
    b = _blur(ctx.gray[frame][y0:y1, x0:x1])
    return float((np.abs(a - b)[px] <= prm.presence_tol).mean())


def _grown(ctx: AnalysisContext, prm: _Params, a: _Det, b: _Det) -> bool:
    """Is the wider of ``a`` / ``b`` the narrower plus something that was *not* on
    screen at the narrower one's frame (a typewriter step, a longer caption)?

    A detection that is only part of a line (weak edges over similar-coloured
    footage) also looks narrower, but the rest of the line was already there.
    """
    wa, wb = a.box[2] - a.box[0], b.box[2] - b.box[0]
    if max(wa, wb) < 1.08 * min(wa, wb):
        return False
    small, big = (a, b) if wa < wb else (b, a)
    return _agreement(ctx, prm, big, small.frame, exclude=small.box) < 0.5


def _link_score(ctx: AnalysisContext, prm: _Params, a: _Det, b: _Det) -> float:
    """Evidence that detections ``a`` and ``b`` show the same overlay (0 = none)."""
    if _area(_intersect(a.box, b.box)) < 0.6 * min(_area(a.box), _area(b.box)) or _grown(ctx, prm, a, b):
        return 0.0
    iou = _iou(a.box, b.box)
    both = a.text is not None and b.text is not None
    if both and iou >= 0.3 and _same_text(a.text, b.text):
        return 3.0 + iou
    # Same pixels in the same place is the same overlay, whatever OCR made of it. Both
    # directions must agree: one word's fill can cover much of a narrower word's strokes.
    agree = min(_agreement(ctx, prm, a, b.frame), _agreement(ctx, prm, b, a.frame))
    return iou + agree if agree >= (0.8 if both else 0.6) else 0.0


def _absorb(ctx: AnalysisContext, prm: _Params, tracks: list[_Track]) -> list[_Track]:
    """Merge tracks that show the same overlay at the same time: pieces of one line
    detected separately at some samples (weak edges, busy footage) become one track."""
    step = ctx.fps / _sample_rate(ctx)
    tracks = sorted(tracks, key=lambda t: -_area(t.box))  # hosts first
    out: list[_Track] = []
    for tr in tracks:
        for host in out:
            if tr.first > host.last + 1.5 * step or tr.last < host.first - 1.5 * step:
                continue
            if _area(_intersect(tr.box, host.box)) < 0.6 * _area(tr.box):
                continue
            d = tr.dets[len(tr.dets) // 2]
            near = [e for e in host.dets if _area(_intersect(e.box, d.box)) >= 0.6 * _area(d.box)]
            if near and _link_score(ctx, prm, min(near, key=lambda e: abs(e.frame - d.frame)), d) > 0:
                host.dets = sorted(host.dets + tr.dets, key=lambda e: e.frame)
                break
        else:
            out.append(tr)
    return sorted(out, key=lambda t: t.first)


def _track(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], dets: dict[_Sample, list[_Det]]) -> list[_Track]:
    active: list[_Track] = []
    done: list[_Track] = []
    for s in samples:
        cur = dets.get(s, [])
        pairs = sorted(((sc, ti, di) for ti, tr in enumerate(active) for di, d in enumerate(cur)
                        if (sc := _link_score(ctx, prm, tr.dets[-1], d)) > 0), key=lambda p: p[0], reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, ti, di in pairs:
            if ti not in used_t and di not in used_d:
                active[ti].dets.append(cur[di])
                active[ti].misses = 0
                used_t.add(ti)
                used_d.add(di)
        keep: list[_Track] = []
        for ti, tr in enumerate(active):
            if ti in used_t:
                keep.append(tr)
                continue
            last = tr.dets[-1]
            replaced = any(_iou(last.box, d.box) > 0.2 for d in cur)
            if (not replaced and tr.misses < prm.max_gap
                    and _agreement(ctx, prm, last, s.frame) >= 0.7):
                tr.misses += 1  # missed detection, but the text is still there
                keep.append(tr)
            else:
                done.append(tr)
        keep.extend(_Track([d]) for di, d in enumerate(cur) if di not in used_d)
        active = keep
    done.extend(active)
    return _absorb(ctx, prm, done)


# --------------------------------------------------------------------------- 5. refinement


class _Template:
    """Median appearance of a track and the mask of its static stroke pixels,
    split into bright and dark classes (fill vs outline / background side).

    ``extra`` frames (spread over the refined span) join the detection frames in
    the stillness test and the median.
    """

    def __init__(self, ctx: AnalysisContext, prm: _Params, dets: list[_Det], extra: tuple[int, ...] = ()) -> None:
        self.x0, self.y0, self.x1, self.y1 = _ibox(_union([d.box for d in dets]), ctx.gray.shape[1:], pad=2)
        sl = np.s_[self.y0:self.y1, self.x0:self.x1]
        frames = sorted({d.frame for d in dets})
        idx = np.unique(np.linspace(0, len(frames) - 1, min(7, len(frames))).round().astype(int))
        stack = np.stack([_blur(ctx.gray[f][sl]) for f in sorted({frames[i] for i in idx} | set(extra))])
        self.gray = np.median(stack, axis=0)
        sources = list({id(d.sample): d.sample for d in dets if d.sample is not None}.values())
        votes = np.zeros(self.gray.shape, np.int32)
        for s in sources:
            votes += s.static[sl]
        # Static at most samples and unchanged across the whole track: flat footage next
        # to the strokes can look static for a moment but not for the overlay's lifetime.
        self.mask = ((votes >= max(1, math.ceil(0.5 * len(sources))))
                     & (stack.max(axis=0) - stack.min(axis=0) <= prm.still_tol))
        self._polarity(prm.presence_tol)

    def _polarity(self, tol: float) -> None:
        """Bright / dark classes of the stroke mask and the grey tolerance ``self.tol``
        of the presence test: at most half the contrast between the classes, so
        footage that shows one class's value (dark footage under dim text) does
        not pass for the text. Classes at least two tolerances apart are tested
        separately (``split``)."""
        vals = self.gray[self.mask]
        self.bright = np.zeros_like(self.mask)
        self.dark = np.zeros_like(self.mask)
        self.split = False
        self.tol = tol
        if vals.size >= 8:
            t = _otsu(vals)
            self.bright = self.mask & (self.gray > t)
            self.dark = self.mask & ~self.bright
            nb, nd = int(self.bright.sum()), int(self.dark.sum())
            contrast = float(self.gray[self.bright].mean() - self.gray[self.dark].mean()) if nb and nd else 0.0
            self.tol = float(np.clip(0.5 * contrast, min(10.0, tol), tol))
            self.split = min(nb, nd) >= 0.2 * vals.size and contrast >= 2.0 * self.tol

    @property
    def center(self) -> tuple[float, float]:
        return 0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1)

    def _patch(self, gray: np.ndarray, frame: int) -> np.ndarray:
        return _blur(gray[frame][self.y0:self.y1, self.x0:self.x1])

    def _score(self, g: np.ndarray, tmpl: np.ndarray, mask: np.ndarray, bright: np.ndarray,
               dark: np.ndarray, split: bool, tol: float) -> float:
        if not mask.any():
            return 0.0
        ok = np.abs(g - tmpl) <= tol
        if split:
            # Each polarity must agree: a dark or white frame matches only one of them.
            return float(min(ok[bright].mean(), ok[dark].mean()))
        return float(ok[mask].mean())

    def presence(self, gray: np.ndarray, frame: int) -> float:
        g = self._patch(gray, frame)
        return self._score(g, self.gray, self.mask, self.bright, self.dark, self.split, self.tol)

    def visible(self, gray: np.ndarray, frame: int, prm: _Params, thr: float) -> Optional[bool]:
        """Is the template on screen in ``frame``? ``None`` when the patch is flat
        (white flash, dip to black): the text cannot be seen either way."""
        g = self._patch(gray, frame)
        if float(g.std()) < 6.0:
            return None
        if self._score(g, self.gray, self.mask, self.bright, self.dark, self.split, self.tol) >= thr:
            return True
        # Flashes, fades and grades shift the pixel values but keep the text's own contrast.
        gain, corr = self._regress(g)
        return corr >= prm.min_corr and gain >= prm.min_gain

    def scaled_match(self, gray: np.ndarray, frame: int, scales: np.ndarray, shift: int = 2) -> np.ndarray:
        """Correlation of ``frame`` with the template scaled by each of ``scales``
        about its centre, over the scaled stroke mask and the best offset within
        ``+-shift`` px (0 where the scaled mask is too small to judge).

        Correlation, not pixel agreement: a renderer that re-rasterises text at
        each size (rather than scaling a bitmap) changes stroke widths and
        anti-aliasing non-linearly, which breaks exact pixel tests.
        """
        h, w = self.gray.shape
        fh, fw = gray.shape[1:]
        cx, cy = self.center
        reach = int(math.ceil(0.5 * max(w, h) * float(scales.max()))) + shift + 1
        x0, y0 = int(round(cx)) - reach, int(round(cy)) - reach
        x1, y1 = int(round(cx)) + reach, int(round(cy)) + reach
        region = gray[frame][max(0, y0):min(fh, y1), max(0, x0):min(fw, x1)]
        region = cv2.copyMakeBorder(region, max(0, -y0), max(0, y1 - fh), max(0, -x0), max(0, x1 - fw),
                                    cv2.BORDER_REPLICATE)
        region = _blur(region)
        out = np.zeros(len(scales))
        for k, scale in enumerate(scales):
            sw, sh = int(round(w * scale)), int(round(h * scale))
            if sh < 7:
                continue  # a few pixels tall: any bright blob matches
            mask = cv2.resize(self.mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
            if int(mask.sum()) < 30:
                continue
            tmpl = cv2.resize(self.gray, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            # Scaled template centred on the overlay's centre, offsets within +-shift.
            ox, oy = reach - int(round(sw / 2)) - shift, reach - int(round(sh / 2)) - shift
            win = region[oy:oy + sh + 2 * shift, ox:ox + sw + 2 * shift]
            res = cv2.matchTemplate(win, tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED, mask=mask)
            out[k] = float(np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).max())
        return out

    def _regress(self, patch: np.ndarray) -> tuple[float, float]:
        if self.mask.sum() < 8:
            return 0.0, 0.0
        g = patch[self.mask]
        t = self.gray[self.mask]
        g, t = g - g.mean(), t - t.mean()
        tt, gg = float((t * t).sum()), float((g * g).sum())
        if tt < 1e-6 or gg < 1e-6:
            return 0.0, 0.0
        gt = float((g * t).sum())
        return gt / tt, gt / math.sqrt(tt * gg)

    def amplitude(self, gray: np.ndarray, frame: int) -> tuple[float, float]:
        """``(a, r)``: regression gain and correlation of the frame on the template
        over the stroke mask (a ~ 1 when fully visible, ~ 0 when absent)."""
        return self._regress(self._patch(gray, frame))


def _refine(ctx: AnalysisContext, prm: _Params, tracks: list[_Track], neutral: np.ndarray) -> None:
    """Frame-accurate start/end of every track (in place).

    The span grows frame by frame from the first / last detection while the
    template stays visible, up to the neighbouring tracks in the same place,
    so samples the detector missed (busy footage, a cut) do not shorten it.
    Two passes: the first template comes from the detection samples alone, so
    footage that happened to hold still around them (a slow pan, a fade) can
    pass for strokes; the second keeps only pixels that stay still across the
    first span, which matters most for overlays seen by one or two samples.
    """
    n = ctx.n_frames
    boxes = [tr.box for tr in tracks]
    for i, tr in enumerate(tracks):
        lo, hi = 0, n - 1
        for j, o in enumerate(tracks):
            if j != i and _overlapping(boxes[j], boxes[i]):
                if o.last < tr.first:
                    lo = max(lo, o.last + 1)
                elif o.first > tr.last:
                    hi = min(hi, o.first - 1)
        first = _Template(ctx, prm, tr.dets)
        _fit(ctx, prm, tr, first, lo, hi, neutral)
        inset = (tr.end - tr.start) // 8  # interior only: no fade / pop frames
        span = np.linspace(tr.start + inset, tr.end - 1 - inset, 9).round().astype(int)
        second = _Template(ctx, prm, tr.dets, extra=tuple(int(f) for f in span if not neutral[f]))
        if int(second.mask.sum()) >= max(20, 0.3 * int(first.mask.sum())):
            _fit(ctx, prm, tr, second, lo, hi, neutral)
    _split_neighbours(ctx, prm, tracks, boxes)
    _close_neutral_gaps(tracks, boxes, neutral)
    for i, tr in enumerate(tracks):
        tr.core = (tr.start, tr.end)
        same = [o for j, o in enumerate(tracks) if j != i and _area(_intersect(boxes[j], boxes[i])) > 0]
        # Entrance / exit animations never reach into another overlay in the same place.
        floor = max([o.end for o in same if o.start < tr.start], default=0)
        ceil = min([o.start for o in same if o.end > tr.end], default=n)
        if tr.start > floor + 2:
            _entrance(ctx, prm, tr, floor)
        if tr.end < ceil - 2:
            _exit(ctx, tr, ceil)


def _fit(ctx: AnalysisContext, prm: _Params, tr: _Track, tmpl: "_Template", lo: int, hi: int,
         neutral: np.ndarray) -> None:
    """Give ``tr`` the template ``tmpl`` and grow its span within ``[lo, hi]`` from the
    detections while the template stays visible (two misses end it; neutral
    frames neither extend nor end it)."""
    tr.tmpl = tmpl
    tr.p_in = float(np.median([tmpl.presence(ctx.gray, d.frame) for d in tr.dets]))
    thr = max(prm.presence_floor, prm.presence_ratio * tr.p_in)

    def scan(frames: range) -> Optional[int]:
        best, misses = None, 0
        for f in frames:
            seen = tmpl.visible(ctx.gray, f, prm, thr)
            if seen:
                best, misses = f, 0
            elif seen is not None and not neutral[f]:
                misses += 1
                if misses > 1:
                    break
        return best

    s = scan(range(tr.first - 1, lo - 1, -1))
    e = scan(range(tr.last + 1, hi + 1))
    tr.start = tr.first if s is None else s
    tr.end = (tr.last if e is None else e) + 1


def _split_neighbours(ctx: AnalysisContext, prm: _Params, tracks: list[_Track], boxes: list[Box]) -> None:
    """Consecutive tracks in one place that overlap in time after refinement (e.g. a
    typewriter state contained in the next) are cut where their templates swap."""
    for i, a in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            b = tracks[j]
            if b.first <= a.last or a.end <= b.start or not _overlapping(boxes[i], boxes[j]):
                continue
            lo, hi = a.last + 1, b.first
            if hi <= lo:
                a.end, b.start = lo, lo
                continue
            assert a.tmpl is not None and b.tmpl is not None
            diff = np.array([a.tmpl.presence(ctx.gray, f) - b.tmpl.presence(ctx.gray, f)
                             for f in range(lo, hi)])
            # Cut c maximises sum(diff[:c]) - sum(diff[c:]).
            gain = np.concatenate([[0.0], np.cumsum(diff)])
            c = int(np.argmax(2 * gain - gain[-1]))
            a.end = lo + c
            b.start = lo + c


def _close_neutral_gaps(tracks: list[_Track], boxes: list[Box], neutral: np.ndarray) -> None:
    """Consecutive tracks in one place separated only by neutral frames (flash,
    transition): the switch happened where it cannot be seen, so split the gap."""
    order = sorted(range(len(tracks)), key=lambda i: tracks[i].start)
    for pos, i in enumerate(order):
        a = tracks[i]
        for j in order[pos + 1:]:
            b = tracks[j]
            if b.start >= a.end and _overlapping(boxes[i], boxes[j]):
                if b.start > a.end and neutral[a.end:b.start].all():
                    a.end = b.start = (a.end + b.start + 1) // 2
                break


def _pop_run(ctx: AnalysisContext, prm: _Params, tr: _Track, frames: range) -> list[int]:
    """Frames (walking back from the start) that show the template scaled, heading
    away from full size, and preceded by no text: a pop. Else empty."""
    tmpl = tr.tmpl
    assert tmpl is not None
    devs_grid = np.abs(np.log(_POP_SCALES))
    full = devs_grid < 0.5 * _POP_STEP
    run: list[int] = []
    devs: list[float] = []
    for f in frames:
        corr = tmpl.scaled_match(ctx.gray, f, _POP_SCALES)
        k = int(np.argmax(np.where(full, -1.0, corr)))
        # A pop frame matches a clearly scaled copy better than the full-size template.
        if corr[k] < prm.pop_corr or corr[k] < float(corr[full].max()) + 0.05:
            break
        run.append(f)
        devs.append(float(devs_grid[k]))
    else:
        if not run or run[-1] > 0:
            return []  # scaled matches all through the window: footage, not an animation
    need = 1 if ctx.fps <= 15 else 2
    if len(run) < need or max(devs) < prm.pop_scale - 0.5 * _POP_STEP:
        return []
    # The scale must head away from 1 walking back in time (grid-step wobble allowed).
    if (any(d1 > d2 + _POP_STEP + 1e-9 for d1, d2 in zip(devs, devs[1:]))
            or (len(run) > 1 and devs[-1] - devs[0] < _POP_STEP - 1e-9)):
        return []
    return run


def _fade_run(ctx: AnalysisContext, tmpl: _Template, frames: range) -> list[int]:
    """Frames (walking away from the visible span) where the template shows at
    steadily falling contrast before it vanishes: a fade. Else empty."""
    run: list[int] = []
    gains: list[float] = []
    for f in frames:
        gain, corr = tmpl.amplitude(ctx.gray, f)
        if not (0.2 <= gain <= 0.9 and corr >= 0.3) or (gains and gain > gains[-1] + 0.05):
            break
        run.append(f)
        gains.append(gain)
    else:
        return []  # still partly visible at the end of the window: not a fade
    return run if len(run) >= 2 and gains[0] - gains[-1] >= 0.2 else []


def _entrance(ctx: AnalysisContext, prm: _Params, tr: _Track, floor: int) -> None:
    """Pop (scale-in) or fade-in just before ``tr.start`` (not before ``floor``): label
    it and extend the start."""
    assert tr.tmpl is not None
    window = max(2, int(round(0.3 * ctx.fps)))
    before = range(tr.start - 1, max(floor - 1, tr.start - 1 - window), -1)
    pop = _pop_run(ctx, prm, tr, before)
    if pop:
        tr.animation, tr.start = "pop", pop[-1]
        return
    fade = _fade_run(ctx, tr.tmpl, before)
    if fade:
        tr.animation, tr.start = "fade", fade[-1]


def _exit(ctx: AnalysisContext, tr: _Track, ceil: int) -> None:
    """Fade-out just after ``tr.end`` (ending by ``ceil``): extend the end (and label a
    static overlay ``fade``)."""
    assert tr.tmpl is not None
    window = max(2, int(round(0.3 * ctx.fps)))
    fade = _fade_run(ctx, tr.tmpl, range(tr.end, min(ceil, tr.end + window)))
    if fade:
        tr.end = fade[-1] + 1
        if tr.animation == "static":
            tr.animation = "fade"


def _disturbed(ctx: AnalysisContext) -> np.ndarray:
    """Frames whose overall brightness jumps away from their surroundings (flash, dip
    to black): overlay pixels there are unreliable evidence either way."""
    luma = ctx.features.luma_mean
    size = 2 * max(2, int(round(0.25 * ctx.fps))) + 1
    return np.abs(luma - median_filter(luma, size=size, mode="nearest")) > 0.12


def _same_overlay(ctx: AnalysisContext, prm: _Params, a: _Track, b: _Track) -> bool:
    """Do two consecutive tracks in one place show the same overlay (the tracker lost
    it in between)? Each template must correlate with the other's samples, and
    neither may be the other plus text that appeared later. Correlation rather than
    pixel agreement: the footage beside the strokes differs between the two."""
    if _iou(a.box, b.box) < 0.6 or _grown(ctx, prm, a.dets[-1], b.dets[0]):
        return False
    for x, y in ((a, b), (b, a)):
        assert x.tmpl is not None
        corr = float(np.median([x.tmpl.amplitude(ctx.gray, d.frame)[1] for d in y.dets]))
        if corr < prm.min_corr:
            return False
    return True


def _merge_consecutive(ctx: AnalysisContext, prm: _Params, tracks: list[_Track]) -> list[_Track]:
    out: list[_Track] = []
    for tr in sorted(tracks, key=lambda t: t.start):
        prev = next((o for o in reversed(out) if tr.start <= o.end + 2 and _overlapping(o.box, tr.box)), None)
        if prev is not None and _same_overlay(ctx, prm, prev, tr):
            # Keep the refined span so a later piece can chain onto the merged track.
            out[out.index(prev)] = _Track(sorted(prev.dets + tr.dets, key=lambda d: d.frame),
                                          start=min(prev.start, tr.start), end=max(prev.end, tr.end))
        else:
            out.append(tr)
    return out


def _build_tracks(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], dets: dict[_Sample, list[_Det]],
                  neutral: np.ndarray) -> list[_Track]:
    tracks = _track(ctx, prm, samples, dets)
    _refine(ctx, prm, tracks, neutral)
    merged = _merge_consecutive(ctx, prm, tracks)
    if len(merged) < len(tracks):
        _refine(ctx, prm, merged, neutral)  # new templates and spans for the merged tracks
    return _drop_nested(merged)


def _drop_nested(tracks: list[_Track]) -> list[_Track]:
    """Drop tracks inside another track's box for most of their time: two overlays
    cannot share pixels, so these are pieces of the bigger one."""
    def nested(t: _Track, host: _Track) -> bool:
        both = min(t.end, host.end) - max(t.start, host.start)
        return (_area(host.box) > _area(t.box) and both >= 0.6 * (t.end - t.start)
                and _area(_intersect(t.box, host.box)) >= 0.7 * _area(t.box))

    return [t for t in tracks if not any(nested(t, o) for o in tracks if o is not t)]


# --------------------------------------------------------------------------- 6. probing


def _probes(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], tracks: list[_Track],
            read_frozen: bool) -> list[_Sample]:
    """Extra samples where a short overlay may hide between the regular samples: the
    middle of each gap between consecutive tracks in one place, and half a sampling
    step before / after every track. Probes showing static text candidates in the
    band of the track that motivated them are kept; over still footage (frozen),
    where candidates say nothing, probes are kept for OCR of that band when
    ``read_frozen``. Frozen regular samples do not block probes: whole-frame OCR
    misses short words ("IS") that a band read finds.
    """
    n = ctx.n_frames
    half = max(1, int(round(0.5 * ctx.fps / _sample_rate(ctx))))
    taken = {s.frame for s in samples}
    near = np.zeros(n + 2, bool)  # frames already sampled with a stillness cue (+-1)
    for s in samples:
        if not s.frozen:
            near[max(0, s.frame - 1):s.frame + 2] = True
    boxes = [tr.box for tr in tracks]
    order = sorted(range(len(tracks)), key=lambda i: tracks[i].start)

    def covered(f: int, box: Box) -> bool:
        return any(tr.start <= f < tr.end and _overlapping(b, box) for tr, b in zip(tracks, boxes))

    wanted: dict[int, Box] = {}
    for pos, i in enumerate(order):
        tr, box = tracks[i], boxes[i]
        frames = [tr.start - half, tr.end - 1 + half]
        nxt = next((j for j in order[pos + 1:] if tracks[j].start >= tr.end and _overlapping(boxes[j], box)), None)
        if nxt is not None and tracks[nxt].start - tr.end >= 2:
            frames.append((tr.end + tracks[nxt].start) // 2)
        for f in frames:
            f = next((g for g in (f, f + 1, f - 1) if g not in taken), f)
            if 0 <= f < n and f not in taken and not near[f] and not covered(f, box):
                wanted.setdefault(f, box)
                near[max(0, f - 1):f + 2] = True
    out = []
    for f in sorted(wanted):
        s = _make_sample(ctx, prm, f)
        s.cands = [c for c in s.cands if _same_band(c, wanted[f])]
        if s.frozen:
            if read_frozen:
                s.focus = [wanted[f]]
                out.append(s)
        elif s.cands:
            out.append(s)
    return out


def _known(ctx: AnalysisContext, prm: _Params, s: _Sample, samples: list[_Sample],
           dets: dict[_Sample, list[_Det]]) -> Optional[list[_Det]]:
    """What a read of probe ``s`` would find when that is known already: nothing on a
    blank band, else the detections of a neighbouring sample showing the same
    thing in the band (see :func:`_changed`, :func:`_same_pixels`). A still
    neighbour that found nothing there does not count: its whole-frame read may
    have missed a short word that a band read finds."""
    ah = ctx.gray.shape[1]
    rows = _rows(s.focus or s.cands, ah)
    if s.frozen and int(np.count_nonzero(s.edges[rows])) < 8:
        return []
    i = int(np.searchsorted([o.frame for o in samples], s.frame))
    for n in sorted(samples[max(0, i - 1):i + 1], key=lambda o: abs(o.frame - s.frame)):
        if n.frame == s.frame or n not in dets:
            continue
        mine = [d for d in dets[n] if _hits(d.box, rows)]
        if s.frozen:
            assert n.grad is not None
            if not mine or any(_hits(b, rows) for b in _changed(ctx, prm, n.grad, n, s)):
                continue
        elif not all(_same_pixels(ctx, prm, n, s, c) for c in s.cands) or                 not all(_agreement(ctx, prm, d, s.frame) >= 0.7 for d in mine):
            continue
        return [replace(d, lines=list(d.lines)) for d in mine]
    return None


def _detect_probes(ctx: AnalysisContext, prm: _Params, probes: list[_Sample], ocr: Optional[_Ocr],
                   budget: _Budget, samples: list[_Sample], dets: dict[_Sample, list[_Det]]
                   ) -> dict[_Sample, list[_Det]]:
    """Probes are OCR'd on the analysis frames (no extra decode) while the budget lasts,
    unless what they would find is known (:func:`_known`)."""
    out: dict[_Sample, list[_Det]] = {}
    for s in probes:
        found = _known(ctx, prm, s, samples, dets) if ocr is not None else None
        if found is not None:
            pass
        elif ocr is not None and budget.take():
            found = _ocr_sample(ctx, prm, s, ctx.frames[s.frame], ocr)
        elif s.frozen:
            found = []  # over still footage stillness cannot single out an overlay
        else:
            found = _classical_sample(prm, s)
        out[s] = _stamp(found, s)
    return out


# --------------------------------------------------------------------------- 7. interpretation


def _line_height(mask: np.ndarray) -> float:
    """Median height of the row runs of ``mask`` (one run per text line)."""
    prof = mask.sum(axis=1)
    if prof.size == 0 or prof.max() <= 0:
        return 0.0
    on = prof >= 0.15 * prof.max()
    runs, start = [], None
    for y, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append(y - start)
            start = None
    runs = [r for r in runs if r >= 2]
    return float(np.median(runs)) if runs else 0.0


def _style(ctx: AnalysisContext, prm: _Params, tr: _Track) -> tuple[Optional[RGB], Optional[bool], float, Box]:
    """``(fill colour, outline, glyph fill height, fill ink box)`` of a refined track, in
    analysis px.

    Stillness comes from up to 12 analysis frames spread over the fully visible
    span: over moving footage the overlay is what holds still. Appearance comes
    from the sharpest view available: the high-resolution crops kept from OCR
    when they cover the overlay, else those analysis frames (where strokes are
    a couple of pixels wide and outlines blur into the fill). See :func:`_appearance`.
    """
    box = tr.box
    lh = float(np.median([ln[3] - ln[1] for d in tr.dets for ln in d.lines]))
    pad = max(3.0, 0.5 * lh)
    fh, fw = ctx.gray.shape[1:]
    x0, y0 = max(0, int(box[0] - pad)), max(0, int(box[1] - pad))
    x1, y1 = min(fw, int(math.ceil(box[2] + pad))), min(fh, int(math.ceil(box[3] + pad)))
    c0, c1 = tr.core
    span = max(1, c1 - c0)
    margin = span // 10
    frames = np.unique(np.linspace(c0 + margin, c0 + span - 1 - margin, max(1, min(12, span - 2 * margin)))
                       .round().astype(int))
    gray = ctx.gray[frames, y0:y1, x0:x1].astype(np.float32)
    still = gray.max(axis=0) - gray.min(axis=0) <= prm.still_tol
    tight = (int(box[0]) - x0, int(box[1]) - y0, int(math.ceil(box[2])) - x0, int(math.ceil(box[3])) - y0)
    zone = _boxes_mask([tuple(float(v) for v in tight)], still.shape, pad=1)
    moving = frames.size >= 3 and bool((~zone).any()) and float(still[~zone].mean()) <= 0.5
    rgb, sx, sy = _hires_views(tr, (x0, y0, x1, y1))
    if rgb is None:
        rgb, sx, sy = ctx.frames[frames, y0:y1, x0:x1], 1.0, 1.0
    gh, gw = rgb.shape[1:3]
    if (gh, gw) != still.shape:
        still = cv2.resize(still.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
    tight_g = (int(round(tight[0] * sx)), int(round(tight[1] * sy)), int(round(tight[2] * sx)),
               int(round(tight[3] * sy)))
    found = _appearance(prm, rgb.astype(np.float32), tight_g, still, moving)
    if found is None:
        return None, None, 0.7 * lh, box  # OCR / edge boxes are ~1.4 glyph heights tall
    color, outline, fill_h, ink = found
    fill_h /= sy
    if fill_h < 0.35 * lh:
        # Thin or dim lettering can leave only a sliver of "fill"; the text cannot be much
        # smaller than the line boxes OCR / the edges found for it.
        fill_h = 0.7 * lh
    return color, outline, fill_h, (x0 + ink[0] / sx, y0 + ink[1] / sy, x0 + ink[2] / sx, y0 + ink[3] / sy)


def _hires_views(tr: _Track, region: tuple[int, int, int, int]) -> tuple[Optional[np.ndarray], float, float]:
    """The track's OCR crops at the highest resolution (>= 1.2x the analysis frames)
    that cover ``region`` (analysis px), cut to it: ``(stack (K, h, w, 3), sx, sy)``."""
    crops = list({id(d.crop): d.crop for d in tr.dets if d.crop is not None}.values())
    if not crops:
        return None, 1.0, 1.0
    sx = max(c.sx for c in crops)
    if sx < 1.2:
        return None, 1.0, 1.0
    crops = [c for c in crops if abs(c.sx - sx) <= 0.01 * sx]
    sy = crops[0].sy
    x0, y0, x1, y1 = region
    gw, gh = int(round((x1 - x0) * sx)), int(round((y1 - y0) * sy))
    views = []
    for c in crops:
        ox, oy = int(round((x0 - c.x0) * sx)), int(round((y0 - c.y0) * sy))
        if ox >= 0 and oy >= 0 and ox + gw <= c.image.shape[1] and oy + gh <= c.image.shape[0]:
            views.append(c.image[oy:oy + gh, ox:ox + gw])
    return (np.stack(views), sx, sy) if views else (None, 1.0, 1.0)


def _appearance(prm: _Params, rgb: np.ndarray, tight: tuple[int, int, int, int], still: np.ndarray,
                moving: bool) -> Optional[tuple[RGB, Optional[bool], float, Box]]:
    """``(fill colour, outline, glyph fill height, fill ink box)`` (grid px) of the text in
    the ``tight`` box of a stack of views ``rgb`` ``(K, h, w, 3)`` of the region around it.

    Text and what lies between its strokes are split by Otsu on the colour value
    (max channel). Light text is the rule (white / yellow captions); dark text is
    assumed on a flat light panel that extends past the text (a caption box), or
    when over moving footage only the dark class holds still. Over moving
    footage the fill is what holds still; over still footage, fill pixels
    connected to the outside of the box are footage (bright sky between
    letters), unless that is nearly all of them (no outline separates text and
    footage). The fill is then snapped to the pixels of its colour at full
    resolution. An outline is a band hugging the fill, far darker than the fill
    (for light text) and darker than the footage just beyond it.
    """
    med = np.median(rgb, axis=0)
    value = med.max(axis=2)
    h, w = value.shape
    tx0, ty0, tx1, ty1 = tight
    inside = np.zeros((h, w), bool)
    inside[max(0, ty0):ty1, max(0, tx0):tx1] = True
    if int(inside.sum()) < 16:
        return None
    lit = value > _otsu(value[inside])
    ring = ~cv2.dilate(inside.astype(np.uint8), _K5).astype(bool)
    boxed = int(ring.sum()) >= 16 and float(lit[ring].mean()) >= 0.9 and float(value[ring].std()) < 15.0
    light = not boxed
    if light and moving and (inside & ~lit).any() and (inside & lit).any():
        light = float(still[inside & ~lit].mean()) - float(still[inside & lit].mean()) < 0.3
    cls = lit if light else ~lit
    fill = inside & cls
    if moving and (fill & still).sum() >= 0.3 * fill.sum():
        fill = fill & still  # over moving footage the overlay is what holds still
    else:
        # Still footage: footage of the fill's polarity reaches into the box from outside.
        _, labels = cv2.connectedComponents(cls.astype(np.uint8), connectivity=8)
        outer = np.unique(labels[ring & cls])
        kept = fill & ~np.isin(labels, outer[outer > 0])
        if kept.sum() >= 0.3 * fill.sum():
            fill = kept
    if int(fill.sum()) < 12:
        return None
    depth = cv2.distanceTransform(fill.astype(np.uint8), cv2.DIST_L2, 3)
    core = fill & (depth >= np.median(depth[fill]))
    color_f = np.median(med[core], axis=0)
    # Snap the fill to the pixels of its colour next to it (the stillness mask comes
    # from coarser analysis frames): at least half-way from what surrounds the
    # strokes to the fill, the 50 % ink level that sizes follow (see _draw_metrics).
    dist = np.abs(med - color_f).max(axis=2)
    grown = cv2.dilate(fill.astype(np.uint8), _K5).astype(bool)
    around = grown & ~fill
    step = float(np.median(dist[around])) if around.any() else 255.0
    snapped = inside & grown & (dist <= 0.5 * max(step, 30.0))
    if snapped.sum() >= 0.5 * fill.sum():
        fill = snapped
    color = tuple(int(round(c)) for c in color_f)
    ys, xs = np.nonzero(fill)
    fill_h = _line_height(fill) or float(ys.max() - ys.min() + 1)
    ink = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    # An outline is a band hugging the fill that is far darker (lighter) than the fill,
    # with brighter (darker) footage beyond it within ~0.3 glyph heights: a luma
    # profile over the distance from the fill. Luma, not colour value: 4:2:0 chroma
    # bleeds the footage's colour into a thin dark outline. Brightest (light text) /
    # darkest (dark text) over time: moving footage lights up (darkens) a plain border,
    # an outline stays put.
    luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    fill_l = float(np.median(np.median(luma, axis=0)[core]))
    extreme = luma.max(axis=0) if light else luma.min(axis=0)
    if not light:
        extreme, fill_l = 255.0 - extreme, 255.0 - fill_l  # dark text: mirror to the light case
    dist = cv2.distanceTransform((~fill).astype(np.uint8), cv2.DIST_L2, 3)
    r = max(1, int(round(0.05 * fill_h)))
    profile = [float(np.median(extreme[(dist > d - 1) & (dist <= d)])) if ((dist > d - 1) & (dist <= d)).any()
               else math.nan for d in range(1, max(r + 2, int(round(0.3 * fill_h))) + 1)]
    outline: Optional[bool] = None
    hug = float(np.nanmin(profile[:r])) if not all(math.isnan(v) for v in profile[:r]) else math.nan
    if not math.isnan(hug):
        if fill_l - hug < prm.outline_contrast:
            outline = False
        elif np.nanmax(profile[r:]) - hug >= 40.0:
            outline = True
        # else: the footage around is as dark (light) as the band, so an outline cannot be told apart.
    return color, outline, fill_h, ink


def _best_text(dets: list[_Det]) -> Optional[str]:
    votes: dict[str, float] = {}
    best: dict[str, _Det] = {}
    seen: set[int] = set()
    for d in dets:
        # One vote per OCR read: samples that reuse a reading share its crop.
        read = id(d.crop) if d.crop is not None else id(d)
        if read in seen:
            continue
        seen.add(read)
        # Readings that differ only by confusable glyphs vote together; the most
        # confident one of the winning group is reported.
        key = _norm(d.text).translate(_CONFUSABLE)
        if not key:
            continue
        votes[key] = votes.get(key, 0.0) + d.score
        if key not in best or d.score > best[key].score:
            best[key] = d
    if not votes:
        return None
    return best[max(votes, key=votes.__getitem__)].text


def _item(ctx: AnalysisContext, prm: _Params, tr: _Track) -> _Item:
    color, outline, fill_h, ink = _style(ctx, prm, tr)
    conf = float(np.clip(np.mean([d.score for d in tr.dets]) * min(1.0, tr.p_in / 0.8), 0.0, 1.0))
    text = _best_text(tr.dets)
    src = next((d for d in tr.dets if d.text == text and len(d.texts) == len(d.lines)), None)
    lines = list(zip(src.lines, src.texts)) if src is not None and text else []
    return _Item(start=tr.start, end=tr.end, box=ink, text=text, fill_h=fill_h, color=color,
                 outline=outline, animation=tr.animation, confidence=conf, tmpl=tr.tmpl, lines=lines)


def _same_place(a: _Item, b: _Item) -> bool:
    ha, hb = a.box[3] - a.box[1], b.box[3] - b.box[1]
    lo, hi = sorted((max(ha, 1.0), max(hb, 1.0)))
    cya, cyb = 0.5 * (a.box[1] + a.box[3]), 0.5 * (b.box[1] + b.box[3])
    overlap = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
    return hi <= 1.6 * lo and abs(cya - cyb) <= 0.5 * hi and overlap > -0.5 * hi


def _starts_with(a: "_Template", b: "_Template") -> bool:
    """Does the wider ``b`` begin with ``a``'s glyphs (OCR-less typewriter test)?

    The edge maps of the two templates are correlated over horizontal (and
    +-2 px vertical) shifts; the best match must be strong and sit where
    ``a``'s first glyph lines up with ``b``'s first glyph. A shorter word that
    merely resembles part of a longer one matches further inside it.
    """
    ca, cb = np.flatnonzero(a.mask.any(axis=0)), np.flatnonzero(b.mask.any(axis=0))
    if ca.size < 3 or cb.size < 3 or a.x1 - a.x0 >= b.x1 - b.x0:
        return False
    y0, y1 = min(a.y0, b.y0) - 2, max(a.y1, b.y1) + 2
    canvas = np.zeros((y1 - y0, b.x1 - b.x0), np.float32)
    canvas[b.y0 - y0:b.y1 - y0] = cv2.morphologyEx(b.gray.astype(np.float32), cv2.MORPH_GRADIENT, _K3)
    ta = cv2.morphologyEx(a.gray.astype(np.float32), cv2.MORPH_GRADIENT, _K3)
    _, score, _, (x, _) = cv2.minMaxLoc(cv2.matchTemplate(canvas, ta, cv2.TM_CCOEFF_NORMED))
    return score >= 0.75 and abs(x - (cb[0] - ca[0])) <= max(2.0, 0.08 * (ca[-1] - ca[0]))


def _prefix(a: _Item, b: _Item) -> bool:
    """``b`` shows ``a``'s text plus more (successive typewriter states)."""
    na, nb = _norm(a.text), _norm(b.text)
    if na and nb:
        return len(nb) > len(na) and nb.startswith(na)
    if a.text is None and b.text is None and a.tmpl is not None and b.tmpl is not None:
        return b.box[2] - b.box[0] >= 1.08 * (a.box[2] - a.box[0]) and _starts_with(a.tmpl, b.tmpl)
    return False


def _chains(items: list[_Item], fps: float, prm: _Params) -> list[_Item]:
    """Merge runs of consecutive same-place items into typewriter / word-by-word overlays."""
    gap = max(2, int(round(prm.word_gap * fps)))

    def wordlike(it: _Item) -> bool:
        return (it.end - it.start) / fps <= prm.word_max and len((it.text or "").split()) <= 2

    order = sorted(range(len(items)), key=lambda i: items[i].start)
    used: set[int] = set()
    out: list[_Item] = []
    for i in order:
        if i in used:
            continue
        for kind in ("typewriter", "word_by_word"):
            if kind == "word_by_word" and not wordlike(items[i]):
                continue
            chain = [i]
            while True:
                cur = items[chain[-1]]
                nxt = [j for j in order if j not in used and j not in chain
                       and -2 <= items[j].start - cur.end <= gap and _same_place(cur, items[j])]
                nxt = [j for j in nxt if (_prefix(cur, items[j]) if kind == "typewriter" else wordlike(items[j]))]
                if not nxt:
                    break
                chain.append(min(nxt, key=lambda j: abs(items[j].start - cur.end)))
            if len(chain) >= (2 if kind == "typewriter" else prm.word_run):
                used.update(chain)
                out.append(_merge(items, chain, kind, fps))
                break
        else:
            used.add(i)
            out.append(items[i])
    return sorted(out, key=lambda it: it.start)


def _merge(items: list[_Item], chain: list[int], kind: TextAnimation, fps: float) -> _Item:
    parts = [items[j] for j in chain]
    end = parts[-1].end
    if kind == "typewriter":
        # Word i of the final text starts with the first state that shows part of it.
        text = parts[-1].text
        words = []
        for i, tok in enumerate((text or "").split()):
            first = next(p for p in parts if len((p.text or "").split()) > i or p is parts[-1])
            words.append(Word(text=tok, start=first.start / fps, end=end / fps))
    else:
        words = [Word(text=p.text or "", start=p.start / fps, end=p.end / fps) for p in parts]
        text = " ".join(p.text for p in parts if p.text) or None
    colors = [p.color for p in parts if p.color is not None]
    outlines = [p.outline for p in parts if p.outline is not None]
    return _Item(
        start=parts[0].start, end=end, box=_union([p.box for p in parts]), text=text,
        fill_h=float(np.median([p.fill_h for p in parts])),
        color=tuple(int(c) for c in np.median(np.array(colors), axis=0)) if colors else None,
        outline=(sum(outlines) * 2 >= len(outlines)) if outlines else None,
        animation=kind, confidence=float(np.mean([p.confidence for p in parts])), words=words,
    )


def _reading_order(lines: list[tuple[Box, str]]) -> str:
    """Text of ``lines`` (box, text): rows top to bottom (lines sharing a band form a
    row, read left to right), joined by newlines."""
    rows: list[list[tuple[Box, str]]] = []
    for box, text in sorted(lines, key=lambda bt: 0.5 * (bt[0][1] + bt[0][3])):
        if rows and any(_same_band(box, b) for b, _ in rows[-1]):
            rows[-1].append((box, text))
        else:
            rows.append([(box, text)])
    return "\n".join(" ".join(t for _, t in sorted(row, key=lambda bt: bt[0][0])) for row in rows)


def _stack(items: list[_Item], fps: float) -> list[_Item]:
    """Merge overlays that appear and disappear together and sit next to each other
    (the stacked lines of a credit card, a title over its subtitle) into one block:
    text in reading order (rows joined by newlines), bbox the union.

    Together: start and end agree within 0.1 s. Next to each other: stacked
    (vertical gap up to 2.5 glyph heights, as credits are spaced; overlapping
    columns) or on one row (horizontal gap up to 3 line heights). Word-by-word
    and typewriter overlays never join."""
    tol = max(2, int(round(0.1 * fps)))
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def near(a: _Item, b: _Item) -> bool:
        gh = max(a.fill_h, b.fill_h, 1.0)
        vgap = max(a.box[1], b.box[1]) - min(a.box[3], b.box[3])
        hgap = max(a.box[0], b.box[0]) - min(a.box[2], b.box[2])
        return (vgap <= 2.5 * gh and hgap < 0) or (_same_band(a.box, b.box) and hgap <= 4.2 * gh)

    loose = [i for i, it in enumerate(items) if it.animation not in ("word_by_word", "typewriter")]
    for x, i in enumerate(loose):
        for j in loose[x + 1:]:
            a, b = items[i], items[j]
            if abs(a.start - b.start) <= tol and abs(a.end - b.end) <= tol and near(a, b):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(items[members[0]])
            continue
        parts = [items[i] for i in members]
        lines = [ln for pt in parts for ln in (pt.lines or ([(pt.box, pt.text)] if pt.text else []))]
        colors = [pt.color for pt in parts if pt.color is not None]
        outlines = [pt.outline for pt in parts if pt.outline is not None]
        anims = {pt.animation for pt in parts}
        out.append(_Item(
            start=min(pt.start for pt in parts), end=max(pt.end for pt in parts), box=_union([pt.box for pt in parts]),
            text=_reading_order(lines) if lines else None,
            fill_h=float(np.median([pt.fill_h for pt in parts])),
            color=tuple(int(c) for c in np.median(np.array(colors), axis=0)) if colors else None,
            outline=(sum(outlines) * 2 >= len(outlines)) if outlines else None,
            animation=anims.pop() if len(anims) == 1 else "static",
            confidence=float(np.mean([pt.confidence for pt in parts])), lines=lines,
            big=max(max(pt.big, pt.fill_h) for pt in parts),
        ))
    return sorted(out, key=lambda it: it.start)


def _position(cy: float) -> TextPosition:
    if cy < 0.2:
        return "top"
    if cy < 0.4:
        return "upper"
    if cy < 0.6:
        return "center"
    if cy < 0.8:
        return "lower"
    return "bottom"


def _role(it: _Item, items: list[_Item], prm: _Params, ctx: AnalysisContext, size: float) -> TextRole:
    """Role from timing, place, size and company.

    Captions are word-by-word, or one of a series (3+) of overlays in the same
    place and style; titles are large, long, or a title card (centred and alone
    on screen); a block of several overlays is a title when it holds a
    title-size line, else a label (credits)."""
    h, w = ctx.gray.shape[1:]
    line, _, _ = _draw_metrics()
    cx, cy = 0.5 * (it.box[0] + it.box[2]) / w, 0.5 * (it.box[1] + it.box[3]) / h
    pos = _position(cy)
    cover = (it.end - it.start) / max(ctx.n_frames, 1)
    corner = (cx < 0.3 or cx > 0.7) and (cy < 0.2 or cy > 0.8)
    series = sum(1 for o in items if o is it or _same_place(o, it)) >= 3
    alone = not any(o is not it and o.start < it.end and it.start < o.end for o in items)
    block = it.big > 0
    if size <= prm.watermark_size and corner and cover >= prm.watermark_cover:
        return "watermark"
    if it.animation in ("word_by_word", "typewriter") and pos != "top":
        return "caption"
    if ctx.t(it.start) <= prm.hook_time and pos in ("top", "upper", "center"):
        return "hook"
    if block:
        return "title" if it.big * line / h >= prm.title_size else "label"
    if series and pos in ("center", "lower", "bottom"):
        return "caption"
    if size >= prm.title_size or cover >= 0.3 or (alone and abs(cx - 0.5) <= 0.1 and pos in ("upper", "center", "lower")):
        return "title"
    if size <= prm.label_size:
        return "label"
    return "unknown"


def _overlays(items: list[_Item], prm: _Params, ctx: AnalysisContext) -> list[TextOverlay]:
    h, w = ctx.gray.shape[1:]
    line, pad_x, pad_y = _draw_metrics()
    out = []
    for it in items:
        size = it.fill_h * line / h
        px, py = pad_x * it.fill_h, pad_y * it.fill_h
        x0, y0 = max(0.0, it.box[0] - px), max(0.0, it.box[1] - py)
        x1, y1 = min(float(w), it.box[2] + px), min(float(h), it.box[3] + py)
        cy = 0.5 * (it.box[1] + it.box[3]) / h
        out.append(TextOverlay(
            start=ctx.t(it.start), end=ctx.t(it.end), text=it.text,
            bbox=(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h), position=_position(cy), size=float(size),
            color=it.color, outline=it.outline, animation=it.animation, role=_role(it, items, prm, ctx, size),
            words=it.words, confidence=it.confidence,
        ))
    return out


# --------------------------------------------------------------------------- entry point


def detect_text(ctx: AnalysisContext, seg: Optional[ShotSegmentation] = None) -> TextAnalysis:
    """Detect burned-in text overlays and interpret them as :class:`TextOverlay` nodes.

    ``seg`` (optional) marks transition frames as neutral during refinement, so
    an overlay that stays on screen through a dissolve or flash keeps its span.
    Signals: ``text_present`` (0/1 per analysis frame).
    """
    n = ctx.n_frames
    if n == 0:
        return TextAnalysis()
    prm = _Params.from_context(ctx)
    ocr = _load_ocr() if ctx.config.ocr else None
    budget = _Budget(prm.max_ocr)
    neutral = _disturbed(ctx)
    if seg is not None:
        mask = seg.transition_frames()
        neutral[:min(n, mask.size)] |= mask[:n]
    samples = _scan(ctx, prm)
    dets = _detect(ctx, prm, samples, ocr, budget)
    tracks = _build_tracks(ctx, prm, samples, dets, neutral)
    for _ in range(prm.probe_rounds):
        probes = _probes(ctx, prm, samples, tracks, read_frozen=ocr is not None and budget.left > 0)
        if not probes:
            break
        dets.update(_detect_probes(ctx, prm, probes, ocr, budget, samples, dets))
        samples = sorted(samples + probes, key=lambda s: s.frame)
        tracks = _build_tracks(ctx, prm, samples, dets, neutral)
    min_frames = max(2, int(round(prm.min_duration * ctx.fps)))
    items = [_item(ctx, prm, tr) for tr in tracks if tr.end - tr.start >= min_frames]
    overlays = _overlays(_stack(_chains(items, ctx.fps, prm), ctx.fps), prm, ctx)
    present = np.zeros(n, np.float64)
    for ov in overlays:
        present[ctx.f(ov.start):min(n, int(round(ov.end * ctx.fps)))] = 1.0
    return TextAnalysis(overlays=overlays, signals={"text_present": present},
                        ocr_engine="rapidocr" if ocr is not None else None)
