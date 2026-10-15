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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np
from scipy.ndimage import median_filter

from editast import fx
from editast.analyzers.types import ShotSegmentation, TextAnalysis
from editast.ast.nodes import RGB, TextAnimation, TextOverlay, TextPosition, TextRole, Word
from editast.context import AnalysisContext
from editast.media import MediaError, fit_size, iter_video

NAME = "text"

_K3 = np.ones((3, 3), np.uint8)
_K5 = np.ones((5, 5), np.uint8)
_OCR_MIN_SIDE = 640  # frames whose long side is smaller are upscaled before OCR
_OCR_TEXT_PX = 40  # small text is upscaled to about this height before OCR
_POP_SCALES = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3)

Box = tuple[float, float, float, float]  # (x0, y0, x1, y1) in analysis-frame pixels, end exclusive


# --------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class _Params:
    stable_frames: int  # frame distance of the static-edge test
    stable_tol: float  # max grey change of a static pixel (3x3 neighbourhood)
    edge_thr: float  # min morphological gradient of an edge pixel
    frozen_ratio: float  # static/all edge ratio above which a sample is "frozen"
    frozen_change: float  # block-mean grey change that invalidates OCR reuse on frozen samples
    join: float  # horizontal closing width joining characters (fraction of frame height)
    min_h: float  # text line height limits (fraction of frame height)
    max_h: float
    min_stable: float  # min static-edge support of an accepted box
    min_contrast: float  # OCR-less: box support minus surrounding support
    min_score: float  # min OCR confidence
    max_ocr: int  # OCR budget (frames) per video
    presence_tol: float  # grey tolerance of the template presence test
    presence_ratio: float  # presence threshold relative to the in-track level
    presence_floor: float
    min_corr: float  # template correlation that counts as present under flashes / fades
    min_gain: float  # ... with at least this fraction of the template's contrast
    max_gap: int  # samples a track may miss and still continue
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
            frozen_change=float(p("frozen_change", 10.0)),
            join=float(p("join", 0.025)),
            min_h=float(p("min_height", 0.012)),
            max_h=float(p("max_height", 0.15)),
            min_stable=float(p("min_stable", 0.4)),
            min_contrast=float(p("min_contrast", 0.3)),
            min_score=float(p("min_score", 0.5)),
            max_ocr=int(p("max_ocr_frames", 200)),
            presence_tol=float(p("presence_tol", 30.0)),
            presence_ratio=float(p("presence_ratio", 0.6)),
            presence_floor=float(p("presence_floor", 0.35)),
            min_corr=float(p("min_corr", 0.8)),
            min_gain=float(p("min_gain", 0.35)),
            max_gap=int(p("max_gap", 2)),
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


@dataclass(eq=False)
class _Det:
    box: Box
    text: Optional[str]
    score: float  # OCR confidence, or static support for OCR-less detections
    lines: list[Box]  # member text-line boxes
    sample: Optional[_Sample] = None

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


def _otsu(values: np.ndarray) -> float:
    v = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    flood[1:-1, 1:-1] = mask
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return mask.astype(bool) | (flood[1:-1, 1:-1] == 0)


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


def _levenshtein(a: str, b: str) -> int:
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


# --------------------------------------------------------------------------- 1-2. static edges, candidates


def _sample_rate(ctx: AnalysisContext) -> float:
    return float(min(max(ctx.config.text_fps, 1e-3), ctx.fps))


def _static_edges(gray: np.ndarray, f: int, prm: _Params) -> tuple[np.ndarray, np.ndarray]:
    g = gray[f]
    edges = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, _K3) >= prm.edge_thr
    refs = [o for o in (f - prm.stable_frames, f + prm.stable_frames) if 0 <= o < gray.shape[0]]
    if not refs:
        return edges.copy(), edges
    stable = np.zeros(g.shape, bool)
    for o in refs:
        # Dilated difference: an edge whose neighbourhood moved by >= 1 px is not static,
        # even when the pixel itself sits inside a flat region that kept its value.
        stable |= cv2.dilate(cv2.absdiff(g, gray[o]), _K3) <= prm.stable_tol
    return edges & stable, edges


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


def _make_sample(ctx: AnalysisContext, prm: _Params, f: int) -> _Sample:
    static, edges = _static_edges(ctx.gray, f, prm)
    n_edges = int(edges.sum())
    frozen = n_edges > 0 and int(static.sum()) >= prm.frozen_ratio * n_edges
    return _Sample(f, static, edges, frozen, _candidates(static, prm))


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


def _ring_support(s: _Sample, box: Box) -> float:
    """Static support of a ring around ``box`` (one box height wide)."""
    h = box[3] - box[1]
    x0, y0, x1, y1 = _ibox(box, s.static.shape)
    X0, Y0, X1, Y1 = _ibox(box, s.static.shape, pad=int(math.ceil(h)))
    e = int(s.edges[Y0:Y1, X0:X1].sum()) - int(s.edges[y0:y1, x0:x1].sum())
    st = int(s.static[Y0:Y1, X0:X1].sum()) - int(s.static[y0:y1, x0:x1].sum())
    return st / e if e > 0 else 0.0


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
        text = " ".join(t for t in (first.text, second.text) if t)
    wa, wb = len(_norm(first.text)) or 1, len(_norm(second.text)) or 1
    score = (first.score * wa + second.score * wb) / (wa + wb)
    box = _union([a.box, b.box])
    lines = first.lines + second.lines if block else [box]
    return _Det(box=box, text=text, score=score, lines=lines)


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


class _Ocr:
    """RapidOCR wrappers: ``det`` finds and reads lines; ``rec`` (optional) reads a
    whole crop as one line, for candidates the detector missed."""

    def __init__(self, det: Any, rec: Any) -> None:
        self.det = det
        self.rec = rec

    @staticmethod
    def _run(engine: Any, rgb: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        result, _ = engine(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
        return [(np.asarray(q, float), str(t).strip(), float(s)) for q, t, s in result or []]

    def read(self, rgb: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        """``[(quad (4, 2) px, text, score)]`` for every text line in ``rgb``."""
        return self._run(self.det, rgb)

    def read_line(self, rgb: np.ndarray) -> tuple[str, float]:
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


def _boxes_mask(boxes: list[Box], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, bool)
    for b in boxes:
        x0, y0, x1, y1 = _ibox(b, shape)
        m[y0:y1, x0:x1] = True
    return m


def _unchanged(ctx: AnalysisContext, prm: _Params, a: _Sample, b: _Sample) -> bool:
    """Would OCR of sample ``b`` return what it returned for ``a``?"""
    ga, gb = ctx.gray[a.frame], ctx.gray[b.frame]
    if a.frozen or b.frozen:
        if not (a.frozen and b.frozen):
            return False
        # Largest mean change over ~20x20 blocks: a caption appearing is local.
        h, w = ga.shape
        bs = max(4.0, max(h, w) / 20.0)
        grid = cv2.resize(cv2.absdiff(ga, gb), (max(1, round(w / bs)), max(1, round(h / bs))),
                          interpolation=cv2.INTER_AREA)
        return float(grid.max()) <= prm.frozen_change
    ma, mb = _boxes_mask(a.cands, ga.shape), _boxes_mask(b.cands, gb.shape)
    union = int((ma | mb).sum())
    if union == 0 or (ma & mb).sum() < 0.6 * union:
        return False
    px = ma & a.static
    if not px.any():
        return True
    return float((cv2.absdiff(ga, gb)[px] <= prm.stable_tol).mean()) >= 0.85


def _plan_ocr(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], budget: _Budget) -> list[tuple[str, int]]:
    """Per sample: ``("ocr", k)``, ``("reuse", j)``, ``("classical", -1)`` or ``("none", -1)``."""
    plan: list[tuple[str, int]] = []
    last: Optional[int] = None
    for k, s in enumerate(samples):
        if not s.frozen and not s.cands:
            plan.append(("none", -1))
        elif last is not None and _unchanged(ctx, prm, samples[last], s):
            plan.append(("reuse", last))
        else:
            plan.append(("ocr", k))
            last = k
    todo = [k for k, (kind, _) in enumerate(plan) if kind == "ocr"]
    if len(todo) > budget.left:
        # Over budget: OCR an even spread, the rest falls back to the classical detector.
        keep = {todo[i] for i in np.linspace(0, len(todo) - 1, budget.left).round().astype(int)} if budget.left else set()
        dropped = set(todo) - keep
        plan = [("classical", -1) if (kind == "ocr" and k in dropped) or (kind == "reuse" and ref in dropped)
                else (kind, ref) for k, (kind, ref) in enumerate(plan)]
    budget.left -= sum(1 for kind, _ in plan if kind == "ocr")
    return plan


def _iter_hires(ctx: AnalysisContext, samples: list[_Sample], wanted: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    """RGB frames of the wanted regular samples at ``config.text_long_side`` when the
    source file is available (and sharper than the analysis frames), else analysis frames."""
    done: set[int] = set()
    path = Path(ctx.path)
    if wanted and ctx.info.width > 0 and ctx.info.height > 0 and path.is_file():
        size = fit_size(ctx.info.width, ctx.info.height, ctx.config.text_long_side)
        if max(size) >= 1.2 * max(ctx.size):
            last = max(wanted)
            try:
                for k, frame in enumerate(iter_video(path, fps=_sample_rate(ctx), size=size, duration=ctx.duration)):
                    if k > last:
                        break
                    if k in wanted:
                        done.add(k)
                        yield k, frame
            except MediaError:
                pass  # fall back to the analysis frames below
    for k in sorted(wanted - done):
        yield k, ctx.frames[samples[k].frame]


def _bands(s: _Sample, ah: int, fh: int) -> list[tuple[int, int, float]]:
    """Row bands ``(y0, y1, text height)`` of a frame of height ``fh`` around the
    candidates of ``s`` (the whole frame, text height unknown = 0, when frozen)."""
    if s.frozen or not s.cands:
        return [(0, fh, 0.0)]
    sy = fh / ah
    spans: list[list[Any]] = []
    for b in sorted(s.cands, key=lambda b: b[1]):
        h = (b[3] - b[1]) * sy
        pad = max(h, 0.03 * fh)
        y0, y1 = max(0, int(b[1] * sy - pad)), min(fh, int(math.ceil(b[3] * sy + pad)))
        if spans and y0 <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], y1)
            spans[-1][2].append(h)
        else:
            spans.append([y0, y1, [h]])
    if sum(y1 - y0 for y0, y1, _ in spans) > 0.6 * fh:
        return [(0, fh, 0.0)]
    return [(int(y0), int(y1), float(np.median(hs))) for y0, y1, hs in spans]


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
    lines: list[_Det] = []
    for y0, y1, text_h in _bands(s, ah, fh):
        # The detector misses short words below ~40 px: upscale small text (at most 3x).
        up = base_up if text_h <= 0 else float(np.clip(_OCR_TEXT_PX / text_h, base_up, 3.0))
        crop = frame[y0:y1]
        if up > 1.0:
            crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        for quad, text, score in ocr.read(crop):
            q = quad / up
            box = (float(q[:, 0].min() * sx), float((q[:, 1].min() + y0) * sy),
                   float(q[:, 0].max() * sx), float((q[:, 1].max() + y0) * sy))
            if _accept(prm, s, box, text, score, ah):
                lines.append(_Det(box=box, text=text, score=score, lines=[box]))
    if not s.frozen:
        # Candidates the detector did not cover: read the candidate box as one line.
        for b in s.cands:
            if any(_area(_intersect(b, d.box)) >= 0.5 * _area(b) for d in lines):
                continue
            h = (b[3] - b[1]) / sy
            x0, x1 = max(0, int(b[0] / sx - 0.4 * h)), min(fw, int(math.ceil(b[2] / sx + 0.4 * h)))
            y0, y1 = max(0, int(b[1] / sy - 0.25 * h)), min(fh, int(math.ceil(b[3] / sy + 0.25 * h)))
            crop = frame[y0:y1, x0:x1]
            if crop.shape[0] < _OCR_TEXT_PX:
                f = _OCR_TEXT_PX / crop.shape[0]
                crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            text, score = ocr.read_line(crop)
            if len(_norm(text)) >= 2 and _accept(prm, s, b, text, score, ah):
                lines.append(_Det(box=b, text=text, score=score, lines=[b]))
    return _blocks(lines)


def _classical_sample(prm: _Params, s: _Sample) -> list[_Det]:
    """OCR-less detections: candidates clearly stiller than their surroundings."""
    lines = []
    for b in s.cands:
        sup = _support(s, b)
        if sup >= prm.min_stable and sup - _ring_support(s, b) >= prm.min_contrast:
            lines.append(_Det(box=b, text=None, score=sup, lines=[b]))
    return _blocks(lines)


def _stamp(dets: list[_Det], s: _Sample) -> list[_Det]:
    for d in dets:
        d.sample = s
    return dets


def _detect(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], ocr: Optional[_Ocr],
            budget: _Budget) -> dict[_Sample, list[_Det]]:
    """Detections of the regular samples (OCR'd, reused or classical)."""
    if ocr is None:
        plan = [("classical", -1)] * len(samples)
    else:
        plan = _plan_ocr(ctx, prm, samples, budget)
    wanted = {k for k, (kind, _) in enumerate(plan) if kind == "ocr"}
    read: dict[int, list[_Det]] = {}
    if ocr is not None:
        for k, frame in _iter_hires(ctx, samples, wanted):
            read[k] = _ocr_sample(ctx, prm, samples[k], frame, ocr)
    out: dict[_Sample, list[_Det]] = {}
    for s, (kind, ref) in zip(samples, plan):
        if kind in ("ocr", "reuse"):
            dets = [replace(d, lines=list(d.lines)) for d in read.get(ref, [])]
        elif kind == "classical":
            dets = _classical_sample(prm, s)
        else:
            dets = []
        out[s] = _stamp(dets, s)
    return out


# --------------------------------------------------------------------------- 4. tracking


def _agreement(ctx: AnalysisContext, prm: _Params, d: _Det, frame: int) -> float:
    """Fraction of ``d``'s static pixels whose grey value is unchanged in ``frame``."""
    assert d.sample is not None
    x0, y0, x1, y1 = _ibox(d.box, ctx.gray.shape[1:])
    px = d.sample.static[y0:y1, x0:x1]
    if not px.any():
        return 0.0
    a = _blur(ctx.gray[d.frame][y0:y1, x0:x1])
    b = _blur(ctx.gray[frame][y0:y1, x0:x1])
    return float((np.abs(a - b)[px] <= prm.presence_tol).mean())


def _grows(a: _Det, b: _Det) -> bool:
    """Text grew (or shrank) by a prefix: successive typewriter states, not one overlay."""
    na, nb = _norm(a.text), _norm(b.text)
    if not na or not nb or na == nb:
        return False
    short, long_ = (na, nb) if len(na) < len(nb) else (nb, na)
    wa, wb = a.box[2] - a.box[0], b.box[2] - b.box[0]
    return (long_.startswith(short) and len(long_) - len(short) >= max(2, math.ceil(0.15 * len(long_)))
            and max(wa, wb) >= 1.08 * min(wa, wb))


def _link_score(ctx: AnalysisContext, prm: _Params, a: _Det, b: _Det) -> float:
    iou = _iou(a.box, b.box)
    if iou < 0.3 or _grows(a, b):
        return 0.0
    both = a.text is not None and b.text is not None
    if both:
        na, nb = _norm(a.text), _norm(b.text)
        sim = _similarity(na, nb)
        if sim >= (1.0 if min(len(na), len(nb)) <= 3 else 0.7):
            return 2.0 + iou + sim
    if iou < 0.5:
        return 0.0
    # Same pixels in the same place is the same overlay, whatever OCR made of it. Both
    # directions must agree: one word's fill can cover much of a narrower word's strokes.
    agree = min(_agreement(ctx, prm, a, b.frame), _agreement(ctx, prm, b, a.frame))
    return iou + agree if agree >= (0.8 if both else 0.6) else 0.0


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
    return sorted(done, key=lambda t: t.first)


# --------------------------------------------------------------------------- 5. refinement


class _Template:
    """Median appearance of a track and the mask of its static stroke pixels,
    split into bright and dark classes (fill vs outline / background side)."""

    def __init__(self, ctx: AnalysisContext, dets: list[_Det]) -> None:
        self.x0, self.y0, self.x1, self.y1 = _ibox(_union([d.box for d in dets]), ctx.gray.shape[1:], pad=2)
        sl = np.s_[self.y0:self.y1, self.x0:self.x1]
        frames = sorted({d.frame for d in dets})
        idx = np.unique(np.linspace(0, len(frames) - 1, min(7, len(frames))).round().astype(int))
        self.gray = np.median(np.stack([_blur(ctx.gray[frames[i]][sl]) for i in idx]), axis=0)
        sources = list({id(d.sample): d.sample for d in dets if d.sample is not None}.values())
        votes = np.zeros(self.gray.shape, np.int32)
        for s in sources:
            votes += s.static[sl]
        self.mask = votes >= max(1, math.ceil(0.5 * len(sources)))
        self._polarity()

    def _polarity(self) -> None:
        vals = self.gray[self.mask]
        self.bright = np.zeros_like(self.mask)
        self.dark = np.zeros_like(self.mask)
        self.split = False
        if vals.size >= 8:
            t = _otsu(vals)
            self.bright = self.mask & (self.gray > t)
            self.dark = self.mask & ~self.bright
            nb, nd = int(self.bright.sum()), int(self.dark.sum())
            self.split = (min(nb, nd) >= 0.2 * vals.size
                          and float(self.gray[self.bright].mean() - self.gray[self.dark].mean()) >= 60.0)

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

    def presence(self, gray: np.ndarray, frame: int, tol: float) -> float:
        g = self._patch(gray, frame)
        return self._score(g, self.gray, self.mask, self.bright, self.dark, self.split, tol)

    def visible(self, gray: np.ndarray, frame: int, prm: _Params, thr: float) -> Optional[bool]:
        """Is the template on screen in ``frame``? ``None`` when the patch is flat
        (white flash, dip to black): the text cannot be seen either way."""
        g = self._patch(gray, frame)
        if float(g.std()) < 6.0:
            return None
        if self._score(g, self.gray, self.mask, self.bright, self.dark, self.split, prm.presence_tol) >= thr:
            return True
        # Flashes, fades and grades shift the pixel values but keep the text's own contrast.
        gain, corr = self._regress(g)
        return corr >= prm.min_corr and gain >= prm.min_gain

    def presence_scaled(self, gray: np.ndarray, frame: int, scale: float, tol: float) -> float:
        """Presence of the template scaled by ``scale`` about its centre."""
        h, w = self.gray.shape
        sw, sh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        tg = cv2.resize(self.gray, (sw, sh), interpolation=interp)
        planes = [cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
                  for m in (self.mask, self.bright, self.dark)]
        cx, cy = self.center
        fh, fw = gray.shape[1:]
        x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
        cx0, cy0 = max(0, -x0), max(0, -y0)
        cx1, cy1 = min(sw, fw - x0), min(sh, fh - y0)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            return 0.0
        g = _blur(gray[frame][y0 + cy0:y0 + cy1, x0 + cx0:x0 + cx1])
        crop = np.s_[cy0:cy1, cx0:cx1]
        return self._score(g, tg[crop], planes[0][crop], planes[1][crop], planes[2][crop], self.split, tol)

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
    """Frame-accurate start/end of every track (in place)."""
    n = ctx.n_frames
    step = ctx.fps / _sample_rate(ctx)
    reach = int(math.ceil((prm.max_gap + 1) * step))
    for tr in tracks:
        tr.tmpl = _Template(ctx, tr.dets)
        tr.p_in = float(np.median([tr.tmpl.presence(ctx.gray, d.frame, prm.presence_tol) for d in tr.dets]))
    boxes = [tr.box for tr in tracks]
    for i, tr in enumerate(tracks):
        tmpl = tr.tmpl
        assert tmpl is not None
        thr = max(prm.presence_floor, prm.presence_ratio * tr.p_in)
        lo, hi = max(0, tr.first - reach), min(n - 1, tr.last + reach)
        for j, o in enumerate(tracks):
            if j != i and _overlapping(boxes[j], boxes[i]):
                if o.last < tr.first:
                    lo = max(lo, o.last + 1)
                elif o.first > tr.last:
                    hi = min(hi, o.first - 1)

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
    _split_neighbours(ctx, prm, tracks, boxes)
    _close_neutral_gaps(tracks, boxes, neutral)
    for i, tr in enumerate(tracks):
        tr.core = (tr.start, tr.end)
        same = [o for j, o in enumerate(tracks) if j != i and _overlapping(boxes[j], boxes[i])]
        if not any(o.start < tr.start <= o.end + 2 for o in same):
            _entrance(ctx, prm, tr)
        if not any(o.end > tr.end >= o.start - 2 for o in same):
            _exit(ctx, tr)


def _split_neighbours(ctx: AnalysisContext, prm: _Params, tracks: list[_Track], boxes: list[Box]) -> None:
    """Consecutive tracks in one place that overlap in time after refinement (e.g. a
    typewriter state contained in the next) are cut where their templates swap."""
    tol = prm.presence_tol
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
            diff = np.array([a.tmpl.presence(ctx.gray, f, tol) - b.tmpl.presence(ctx.gray, f, tol)
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
    """Frames (walking back from the start) that show the template scaled, converging
    to full size towards the start, and preceded by no text: a pop. Else empty."""
    tmpl = tr.tmpl
    assert tmpl is not None
    thr = max(prm.presence_floor, prm.presence_ratio * tr.p_in)
    tol = 1.5 * prm.presence_tol
    run: list[int] = []
    devs: list[float] = []
    for f in frames:
        scores = {sc: tmpl.presence_scaled(ctx.gray, f, sc, tol) for sc in _POP_SCALES}
        sc = max(scores, key=scores.__getitem__)
        if scores[sc] < thr:
            break
        run.append(f)
        devs.append(abs(math.log(sc)))
    else:
        return []  # scaled matches all through the window: footage, not an animation
    need = 1 if ctx.fps <= 15 else 2
    if len(run) < need or max(devs) < abs(math.log(0.85)) - 1e-9:
        return []
    # The scale must head away from 1 walking back in time (and actually change).
    if any(d1 > d2 + 1e-9 for d1, d2 in zip(devs, devs[1:])) or (len(run) > 1 and devs[-1] - devs[0] < 0.04):
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


def _entrance(ctx: AnalysisContext, prm: _Params, tr: _Track) -> None:
    """Pop (scale-in) or fade-in just before ``tr.start``: label it and extend the start."""
    assert tr.tmpl is not None
    window = max(2, int(round(0.3 * ctx.fps)))
    before = range(tr.start - 1, max(-1, tr.start - 1 - window), -1)
    pop = _pop_run(ctx, prm, tr, before)
    if pop:
        tr.animation, tr.start = "pop", pop[-1]
        return
    fade = _fade_run(ctx, tr.tmpl, before)
    if fade:
        tr.animation, tr.start = "fade", fade[-1]


def _exit(ctx: AnalysisContext, tr: _Track) -> None:
    """Fade-out just after ``tr.end``: extend the end (and label a static overlay ``fade``)."""
    assert tr.tmpl is not None
    window = max(2, int(round(0.3 * ctx.fps)))
    fade = _fade_run(ctx, tr.tmpl, range(tr.end, min(ctx.n_frames, tr.end + window)))
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


def _build_tracks(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], dets: dict[_Sample, list[_Det]],
                  neutral: np.ndarray) -> list[_Track]:
    tracks = _track(ctx, prm, samples, dets)
    _refine(ctx, prm, tracks, neutral)
    return tracks


# --------------------------------------------------------------------------- 6. probing


def _probes(ctx: AnalysisContext, prm: _Params, samples: list[_Sample], tracks: list[_Track]) -> list[_Sample]:
    """Extra samples where a short overlay may hide between the regular samples: the
    middle of each gap between consecutive tracks in one place, and half a sampling
    step before / after every track. Only probes showing static text candidates in
    the band of the track that motivated them are kept."""
    n = ctx.n_frames
    half = max(1, int(round(0.5 * ctx.fps / _sample_rate(ctx))))
    near = np.zeros(n + 2, bool)  # frames already sampled (+-1)
    for s in samples:
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
            if 0 <= f < n and not near[f] and not covered(f, box):
                wanted.setdefault(f, box)
                near[max(0, f - 1):f + 2] = True
    out = []
    for f in sorted(wanted):
        s = _make_sample(ctx, prm, f)
        s.cands = [c for c in s.cands if _same_band(c, wanted[f])]
        if s.cands and not s.frozen:
            out.append(s)
    return out


def _detect_probes(ctx: AnalysisContext, prm: _Params, probes: list[_Sample], ocr: Optional[_Ocr],
                   budget: _Budget) -> dict[_Sample, list[_Det]]:
    """Probes are OCR'd on the analysis frames (no extra decode) while the budget lasts."""
    out: dict[_Sample, list[_Det]] = {}
    for s in probes:
        if ocr is not None and budget.take():
            dets = _ocr_sample(ctx, prm, s, ctx.frames[s.frame], ocr)
        else:
            dets = _classical_sample(prm, s)
        out[s] = _stamp(dets, s)
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
    """``(fill colour, outline, glyph fill height px, ink box)`` of a refined track.

    Up to 12 frames spread over the fully visible span are stacked. Over moving footage the
    overlay is what stays *still* (small grey range): split those pixels into
    bright and dark classes; the class deepest inside the still area is the
    fill (an outline wraps it). Over static footage everything is still, and
    the classes come from the stroke region alone. An outline is a band next
    to the fill whose brightest (for light text) value over time stays well
    below the fill and below the footage just outside the text; it is only
    judged when the footage moves.
    """
    tmpl = tr.tmpl
    assert tmpl is not None
    fh, fw = ctx.gray.shape[1:]
    x0, y0 = max(0, tmpl.x0 - 3), max(0, tmpl.y0 - 3)
    x1, y1 = min(fw, tmpl.x1 + 3), min(fh, tmpl.y1 + 3)
    stroke = np.zeros((y1 - y0, x1 - x0), np.uint8)
    stroke[tmpl.y0 - y0:tmpl.y1 - y0, tmpl.x0 - x0:tmpl.x1 - x0] = tmpl.mask
    region = _fill_holes(cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, _K3))
    ys, xs = np.nonzero(region)
    if xs.size == 0:
        return None, None, float(tmpl.y1 - tmpl.y0), tr.box
    ink = (float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1), float(y0 + ys.max() + 1))
    fallback_h = max(1.0, float(ys.max() - ys.min() - 1))  # edge maps reach ~1 px past the glyphs
    c0, c1 = tr.core
    span = max(1, c1 - c0)
    margin = span // 10
    count = max(1, min(12, span - 2 * margin))
    frames = np.unique(np.linspace(c0 + margin, c0 + span - 1 - margin, count).round().astype(int))
    sl = np.s_[y0:y1, x0:x1]
    rgb = np.stack([ctx.frames[f][sl] for f in frames]).astype(np.float32)
    gray = np.stack([ctx.gray[f][sl] for f in frames]).astype(np.float32)
    value = rgb.max(axis=3)
    zone = cv2.dilate(region.astype(np.uint8), _K3).astype(bool)
    still = gray.max(axis=0) - gray.min(axis=0) <= prm.still_tol
    moving = frames.size >= 3 and bool((~zone).any()) and float(still[~zone].mean()) <= 0.5
    px = zone & still if moving else region
    if px.sum() < 12:
        return None, None, fallback_h, ink
    med = np.median(gray, axis=0)
    vmed = np.median(value, axis=0)
    t = _otsu(med[px])
    bright, dark = px & (med > t), px & (med <= t)
    outside = cv2.dilate(zone.astype(np.uint8), _K5).astype(bool) & ~zone
    ref = float(np.median(vmed[outside])) if outside.any() else 128.0
    if min(bright.sum(), dark.sum()) >= 4 and float(med[bright].mean() - med[dark].mean()) >= 60.0:
        if moving:
            # Only overlay pixels are still: the outline wraps the fill, so the fill lies deeper.
            depth = cv2.distanceTransform(px.astype(np.uint8), cv2.DIST_L2, 3)
            light = float(depth[bright].mean()) >= float(depth[dark].mean())
        else:
            # The region also holds footage (letter counters): the fill contrasts with the
            # surroundings; when both classes do (fill + outline), captions are light on dark.
            cb = abs(float(np.median(vmed[bright])) - ref)
            cd = abs(float(np.median(vmed[dark])) - ref)
            light = cb >= cd or min(cb, cd) >= 60.0
        fill = bright if light else dark
    else:
        fill = px
        light = float(np.median(vmed[fill])) >= ref
    depth = cv2.distanceTransform(fill.astype(np.uint8), cv2.DIST_L2, 3)
    core = fill & (depth >= np.median(depth[fill]))
    color = tuple(int(round(c)) for c in np.median(np.median(rgb, axis=0)[core], axis=0))
    outline: Optional[bool] = None
    near = cv2.dilate(fill.astype(np.uint8), _K3).astype(bool) & ~fill
    if outside.any() and near.any():
        # Brightest (light text) / darkest (dark text) value over time: moving footage
        # lights up (darkens) a plain border, an outline stays put.
        fill_v = float(np.median(vmed[core]))
        extreme = value.max(axis=0) if light else value.min(axis=0)
        near_v, out_v = float(np.median(extreme[near])), float(np.median(extreme[outside]))
        band, sep = (fill_v - near_v, out_v - near_v) if light else (near_v - fill_v, near_v - out_v)
        if band < prm.outline_contrast:
            outline = False
        elif sep >= 25.0:
            outline = True
        # else: the footage around is as dark (light) as the band, so an outline cannot be told apart.
    return color, outline, _line_height(fill) or fallback_h, ink


def _best_text(dets: list[_Det]) -> Optional[str]:
    votes: dict[str, float] = {}
    best: dict[str, _Det] = {}
    for d in dets:
        key = _norm(d.text)
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
    return _Item(start=tr.start, end=tr.end, box=ink, text=_best_text(tr.dets), fill_h=fill_h, color=color,
                 outline=outline, animation=tr.animation, confidence=conf, tmpl=tr.tmpl)


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
    h, w = ctx.gray.shape[1:]
    cx, cy = 0.5 * (it.box[0] + it.box[2]) / w, 0.5 * (it.box[1] + it.box[3]) / h
    pos = _position(cy)
    cover = (it.end - it.start) / max(ctx.n_frames, 1)
    corner = (cx < 0.3 or cx > 0.7) and (cy < 0.2 or cy > 0.8)
    frequent = sum(1 for o in items if abs(0.5 * (o.box[1] + o.box[3]) / h - cy) <= 0.08) >= 3
    if size <= prm.watermark_size and corner and cover >= prm.watermark_cover:
        return "watermark"
    if it.animation in ("word_by_word", "typewriter") and pos != "top":
        return "caption"
    if ctx.t(it.start) <= prm.hook_time and pos in ("top", "upper", "center"):
        return "hook"
    if frequent and pos in ("center", "lower", "bottom"):
        return "caption"
    if size >= prm.title_size or cover >= 0.3:
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
        probes = _probes(ctx, prm, samples, tracks)
        if not probes:
            break
        dets.update(_detect_probes(ctx, prm, probes, ocr, budget))
        samples = sorted(samples + probes, key=lambda s: s.frame)
        tracks = _build_tracks(ctx, prm, samples, dets, neutral)
    min_frames = max(2, int(round(prm.min_duration * ctx.fps)))
    items = [_item(ctx, prm, tr) for tr in tracks if tr.end - tr.start >= min_frames]
    overlays = _overlays(_chains(items, ctx.fps, prm), prm, ctx)
    present = np.zeros(n, np.float64)
    for ov in overlays:
        present[ctx.f(ov.start):min(n, int(round(ov.end * ctx.fps)))] = 1.0
    return TextAnalysis(overlays=overlays, signals={"text_present": present},
                        ocr_engine="rapidocr" if ocr is not None else None)
