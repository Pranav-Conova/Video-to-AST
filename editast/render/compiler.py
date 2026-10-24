"""Compile an Edit AST into a new video — the inverse of analysis.

A reverse-engineered viral edit is reused as a *template*: its shot timing,
transitions, effects and captions are replayed over the user's own footage
(and optionally retimed to a new song with :func:`retime_edit`).

Per output frame ``f`` (shown at ``f / fps`` seconds):

1. **Shot lookup** — shots partition the timeline; shot ``i`` owns output
   frames ``[round(start_i * fps), round(start_{i+1} * fps))``.
2. **Source timing** — :func:`source_positions` integrates the eased playback
   rate of the shot's Speed effects (``rate_from -> rate_to``) and holds the
   picture during Freeze spans, giving a (fractional) source position per
   output frame. The :class:`~editast.render.sources.ClipSource` delivers the
   footage; fractional positions are sampled exactly by a ``SubframeSource``
   and frame-blended otherwise.
3. **Geometry** — Zoom (scale), Rotate (angle) and Pan (``tx, ty``) are eased
   over their span and hold their end value afterwards, Shake adds
   :func:`editast.fx.shake_offset` jitter inside its span; all of it is one
   :func:`editast.fx.transform` call per frame. A constant per-shot cover
   magnification keeps moves from exposing the frame border without
   changing any observable (relative) zoom, pan or shake.
4. **Pixel effects** — blur, rgb_split, invert, then in-shot flash (linear
   rise and fall, exactly ``intensity`` on the middle frame of the span);
   then the shot look (B&W) and layout bars (letterbox / pillarbox).
5. **Transitions** — flash / fade_black / fade_white blend toward a colour
   with an envelope peaking on the boundary frame; dissolve and slide mix the
   outgoing and incoming shot, which are rendered with *handles* (extra
   frames past their cut) for that; whip / zoom / spin / glitch are
   approximated with fast motion + blur around the boundary. Fade-in
   (``from_shot=None``) and fade-out (``to_shot=None``) dip from / to colour.
6. **Text overlays** — :func:`editast.fx.draw_text` centred on the bbox
   centre; ``word_by_word`` overlays show each Word in its own span. Text is
   a layer above the picture: transitions and fades do not affect it.

Frame quantisation (shared with :mod:`editast.bench.generate`): a span
``[start, end)`` covers frames ``round(start*fps) .. round(end*fps) - 1``, at
least one frame. Geometry progress runs from 0 on the first to 1 on the last
frame of the span, so the observed end/start zoom ratio is exactly
``scale_to / scale_from``; envelopes (flash, speed) use frame centres.

Effects with ``params["source"] == "camera"`` describe motion that is already
in the footage (e.g. an analysed camera pan) and are not re-applied.

Frames are streamed into :class:`editast.media.VideoWriter`. Source footage
is fetched lazily per shot (streamed from the decoder or sampled in chunks,
see ``_Footage``), so memory does not grow with shot or video length.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import cv2
import numpy as np

from editast import fx
from editast.ast.nodes import (
    BBox,
    Blur,
    Edit,
    EffectBase,
    Flash,
    Freeze,
    Invert,
    Pan,
    RGBSplit,
    Rotate,
    Shake,
    Shot,
    Speed,
    TextOverlay,
    Timed,
    Transition,
    Zoom,
    assign_ids,
)
from editast.media import VideoWriter, fit_size
from editast.render.sources import ClipSource, Size, StreamingSource, SubframeSource, VideoClipSource

ProgressFn = Callable[[str], None]
Sources = Union[ClipSource, str, Path, Sequence[Union[str, Path]]]

MAX_LONG_SIDE = 1280
DEFAULT_SIZE: Size = (720, 1280)
DEFAULT_LINE_HEIGHT = 0.07  # text line height (fraction of frame height) when unknown
_HANDLE_KINDS = ("dissolve", "slide")  # transitions that show both shots at once
_PIXEL_ORDER = {"blur": 0, "rgb_split": 1, "invert": 2, "flash": 3}
_POSITION_Y = {"top": 0.1, "upper": 0.3, "center": 0.5, "lower": 0.7, "bottom": 0.88}
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# --------------------------------------------------------------------------- timing helpers


def frame_span(start: float, end: float, fps: float, min_frames: int = 1) -> tuple[int, int]:
    """Output frames ``[f0, f1)`` covered by the span ``[start, end)`` seconds."""
    f0 = int(round(start * fps))
    return f0, max(int(round(end * fps)), f0 + min_frames)


def _is_descriptive(effect: EffectBase) -> bool:
    """Effects that describe the footage itself (camera moves) rather than a post effect."""
    return effect.params.get("source") == "camera"


def source_positions(shot: Shot, fps: float, frames: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Source position of every output frame of ``shot``, in source frames since its in-point.

    ``frames`` is the shot's output frame range (default: rounded start/end).
    The position advances by the eased Speed rate per frame (1 outside Speed
    spans; several overlapping Speed effects multiply) and holds during a
    Freeze span, so a freeze over frames ``[a, b)`` shows ``b - a`` identical
    frames and the footage resumes where it stopped.
    """
    f0, f1 = frames if frames is not None else (int(round(shot.start * fps)), int(round(shot.end * fps)))
    n = max(0, f1 - f0)
    if n == 0:
        return np.zeros(0)
    idx = np.arange(f0, f1)
    rate = np.ones(n)
    frozen = np.zeros(n, bool)
    for eff in shot.effects:
        if _is_descriptive(eff) or not isinstance(eff, (Speed, Freeze)):
            continue
        es, ee = frame_span(eff.start, eff.end, fps)
        inside = (idx >= es) & (idx < ee)
        if not inside.any():
            continue
        if isinstance(eff, Freeze):
            frozen |= inside
        else:
            p = (idx[inside] - es + 0.5) / (ee - es)
            rate[inside] *= np.maximum(0.0, fx.interp(eff.rate_from, eff.rate_to, p, eff.easing))
    step = rate.copy()
    step[frozen & np.append(frozen[1:], False)] = 0.0  # hold, then resume on the frame after the span
    return np.concatenate([[0.0], np.cumsum(step[:-1])])


def output_format(edit: Edit, size: Optional[Size] = None, fps: Optional[float] = None) -> tuple[Size, float, int]:
    """``((width, height), fps, n_frames)`` of the rendered video.

    Default size: the edit's media size scaled so the long side is at most
    1280 px (even dimensions); default fps: ``media.analysis_fps``.
    """
    if size is None:
        if edit.media.width > 0 and edit.media.height > 0:
            size = fit_size(edit.media.width, edit.media.height, MAX_LONG_SIDE)
        else:
            size = DEFAULT_SIZE
    w, h = int(size[0]) // 2 * 2, int(size[1]) // 2 * 2
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid output size {size}")
    fps = float(fps or edit.media.analysis_fps or 30.0)
    shots = edit.shots()
    duration = max(edit.media.duration, shots[-1].end if shots else 0.0)
    n = int(round(duration * fps))
    if n <= 0:
        raise ValueError("the edit has no duration to render")
    return (w, h), fps, n


# --------------------------------------------------------------------------- text


@functools.lru_cache(maxsize=1)
def _line_ratio() -> float:
    """Drawn line height (text bbox height) per unit of :func:`fx.draw_text` cap size."""
    _, box = fx.draw_text(np.zeros((1000, 4000, 3), np.uint8), "Hg", (0.5, 0.5), 0.05)
    return box[3] / 0.05


def text_box(text: str, center: tuple[float, float], line_height: float, size: Size) -> BBox:
    """Normalised bbox the renderer draws ``text`` into (``line_height`` = fraction of frame height)."""
    w, h = size
    _, box = fx.draw_text(np.zeros((h, w, 3), np.uint8), text, center, line_height / _line_ratio())
    return box


@dataclass
class _OverlayPlan:
    f0: int
    f1: int
    segments: list[tuple[int, int, str]]  # (first frame, end frame, text shown; "\n" separates lines)
    center: tuple[float, float]
    cap: float
    color: tuple[int, int, int]
    outline: bool
    animation: str
    n_lines: int = 1  # line slots of the layout, stacked around ``center``
    pitch: float = 0.0  # distance between line centres, fraction of frame height

    def draw(self, frame: np.ndarray, f: int, fps: float) -> np.ndarray:
        for s0, s1, text in self.segments:
            if not (s0 <= f < s1) or not text:
                continue
            cap = self.cap
            if self.animation == "pop":  # quick scale-up over the first 0.15 s
                cap *= 0.75 + 0.25 * min(1.0, (f - self.f0 + 1) / max(1.0, 0.15 * fps))
            drawn = frame
            for k, line in enumerate(text.split("\n")):
                cy = self.center[1] + (k - (self.n_lines - 1) / 2.0) * self.pitch
                if line:
                    drawn, _ = fx.draw_text(drawn, line, (self.center[0], cy), cap, self.color, self.outline)
            if self.animation == "fade":
                ramp = max(1.0, 0.2 * fps)
                alpha = min(1.0, (f - self.f0 + 1) / ramp, (self.f1 - f) / ramp)
                drawn = fx.blend(frame, drawn, alpha)
            frame = drawn
        return frame


def _wrap(words: list[str], n_lines: int) -> list[list[str]]:
    """Split ``words`` into at most ``n_lines`` lines, minimising the longest line (in characters)."""
    n, m = min(n_lines, len(words)), len(words)
    if n <= 1:
        return [words]
    ends = np.concatenate([[0], np.cumsum([len(w) + 1 for w in words])])  # ends[j] - ends[i] - 1 = width

    # best[k][j]: smallest possible longest line when the first j words fill k lines.
    best = np.full((n + 1, m + 1), np.inf)
    cut = np.zeros((n + 1, m + 1), int)
    best[0, 0] = 0.0
    for k in range(1, n + 1):
        for j in range(k, m + 1):
            i = np.arange(k - 1, j)
            cost = np.maximum(best[k - 1, i], ends[j] - ends[i] - 1)
            cut[k, j] = i[int(np.argmin(cost))]
            best[k, j] = cost.min()
    lines, j = [], m
    for k in range(n, 0, -1):
        i = int(cut[k, j])
        lines.append(words[i:j])
        j = i
    return lines[::-1]


def _layout(lines: list[list[str]], n_words: Optional[int] = None) -> str:
    """The first ``n_words`` words of a wrapped text, keeping their line breaks."""
    out, left = [], sum(len(line) for line in lines) if n_words is None else n_words
    for line in lines:
        out.append(" ".join(line[:max(0, left)]))
        left -= len(line)
    return "\n".join(out).rstrip("\n")


def _plan_overlay(ov: TextOverlay, fps: float) -> Optional[_OverlayPlan]:
    """Frame segments and layout of one overlay.

    A bbox several ``size`` lines tall (a caption block, as the text analyzer
    reports it) is laid out as that many balanced lines at a fixed pitch.
    """
    f0, f1 = frame_span(ov.start, ov.end, fps)
    x, y, bw, bh = ov.bbox
    has_box = bw > 0 and bh > 0
    line_h = ov.size if ov.size > 0 else (bh if has_box else DEFAULT_LINE_HEIGHT)
    n_lines = max(1, int(round(bh / line_h))) if has_box and ov.size > 0 else 1
    text = " ".join((ov.text or "").split())
    words = sorted((w for w in ov.words if w.text.strip()), key=lambda w: w.start)
    tokens = [w.text.strip() for w in words] if words else text.split()
    lines = _wrap(tokens, n_lines)
    segments: list[tuple[int, int, str]] = []
    if words and ov.animation == "word_by_word":  # one word at a time, centred in the box
        for w in words:
            a, b = frame_span(w.start, w.end, fps)
            segments.append((max(a, f0), min(b, f1), w.text.strip()))
        lines = [tokens]
    elif words and ov.animation == "typewriter":
        for i, w in enumerate(words):
            a = frame_span(w.start, w.end, fps)[0]
            b = int(round(words[i + 1].start * fps)) if i + 1 < len(words) else f1
            segments.append((max(a, f0), min(b, f1), _layout(lines, i + 1)))
    elif text and ov.animation == "typewriter":
        full = _layout(lines)
        reveal = max(1, int(0.6 * (f1 - f0)))
        cuts = [f0 + reveal * i // len(full) for i in range(len(full))] + [f1]
        segments = [(cuts[i], cuts[i + 1], full[: i + 1]) for i in range(len(full)) if cuts[i + 1] > cuts[i]]
    elif tokens:
        segments.append((f0, f1, _layout(lines)))
    if not segments:
        return None
    n_lines = len(lines)
    center = (x + bw / 2.0, y + bh / 2.0) if has_box else (0.5, _POSITION_Y.get(ov.position, 0.5))
    pitch = bh / n_lines if has_box else line_h
    color = tuple(int(c) for c in ov.color) if ov.color is not None else WHITE
    outline = True if ov.outline is None else bool(ov.outline)
    return _OverlayPlan(f0, f1, segments, center, line_h / _line_ratio(), color, outline, ov.animation,
                        n_lines, pitch)


# --------------------------------------------------------------------------- shots


@dataclass
class Geometry:
    """Per-frame similarity transform of a shot (arguments of :func:`fx.transform`)."""

    scale: np.ndarray
    angle: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    center: tuple[float, float] = (0.5, 0.5)


def shot_geometry(shot: Shot, fps: float, frames: np.ndarray, size: Size) -> Geometry:
    """Combined Zoom / Rotate / Pan / Shake transform of ``shot`` at output ``frames``.

    Zoom scales multiply, angles and translations add. Zoom, Rotate and Pan
    hold their start value before and their end value after their span;
    Shake only acts inside its span. A constant cover scale keeps the frame
    filled (see :func:`_cover`); pass all frames the shot will render.
    """
    w, h = size
    frames = np.asarray(frames)
    k = frames.size
    geo = Geometry(np.ones(k), np.zeros(k), np.zeros(k), np.zeros(k))
    zoom_seen = False
    for order, eff in enumerate(shot.effects):
        if _is_descriptive(eff) or not isinstance(eff, (Zoom, Rotate, Pan, Shake)):
            continue
        es, ee = frame_span(eff.start, eff.end, fps)
        p = np.clip((frames - es) / max(1, ee - es - 1), 0.0, 1.0)
        if isinstance(eff, Zoom):
            geo.scale *= fx.interp(eff.scale_from, eff.scale_to, p, eff.easing)
            if not zoom_seen:
                geo.center, zoom_seen = (float(eff.center[0]), float(eff.center[1])), True
        elif isinstance(eff, Rotate):
            geo.angle += fx.interp(eff.angle_from, eff.angle_to, p, eff.easing)
        elif isinstance(eff, Pan):
            geo.tx += fx.interp(0.0, eff.dx, p, eff.easing)
            geo.ty += fx.interp(0.0, eff.dy, p, eff.easing)
        else:
            seed = int(eff.params.get("seed", 31 * shot.index + order))
            phase = float(np.random.default_rng(seed + 1).uniform(0.0, 2.0 * np.pi))
            for i in np.flatnonzero((frames >= es) & (frames < ee)):
                t = (frames[i] - es) / fps
                dx, dy = fx.shake_offset(t, eff.amplitude, eff.frequency, seed)
                geo.tx[i] += dx
                geo.ty[i] += dy * w / h  # amplitude is a fraction of the width on both axes
                if eff.rotation_amplitude:
                    geo.angle[i] += eff.rotation_amplitude * math.sin(2 * math.pi * 0.9 * eff.frequency * t + phase)
    _cover(geo, size)
    return geo


def _cover(geo: Geometry, size: Size) -> None:
    """Magnify the whole shot just enough that no frame shows the picture's border.

    Pans, rotations, shake and zoom-outs would otherwise expose mirrored
    edges. The extra scale ``k`` is constant over the shot (so it is not an
    observable zoom) and is folded into the same similarity transform as a
    pre-scale about the frame centre: for ``q = s R (p - c) + c + t`` that is
    scale ``s k`` and translation ``t + s (k - 1) R (c - c0)``. Translations
    keep their on-screen size, so observed pan and shake are unchanged.
    """
    w, h = size
    s, theta = geo.scale, np.radians(geo.angle)
    cos, sin = np.cos(theta), np.sin(theta)
    cx, cy = geo.center[0] * w, geo.center[1] * h
    # Forward map q = A p + b (cv2.getRotationMatrix2D layout); A^-1 = [[a, -b], [b, a]] / s^2.
    a, b = s * cos, s * sin
    bx = (1.0 - a) * cx - b * cy + geo.tx * w
    by = b * cx + (1.0 - a) * cy + geo.ty * h
    need = np.ones_like(s)
    for qx, qy in ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h)):  # output corners, mapped back to the source
        ux, uy = qx - bx, qy - by
        px, py = (a * ux - b * uy) / s ** 2, (b * ux + a * uy) / s ** 2
        need = np.maximum(need, np.maximum(np.abs(px - w / 2) / (w / 2), np.abs(py - h / 2) / (h / 2)))
    k = float(need.max()) if need.size else 1.0
    if k <= 1.0 + 1e-6:
        return
    dx, dy = cx - w / 2, cy - h / 2
    geo.tx = geo.tx + s * (k - 1.0) * (cos * dx + sin * dy) / w
    geo.ty = geo.ty + s * (k - 1.0) * (-sin * dx + cos * dy) / h
    geo.scale = s * k


class _Footage:
    """Source pictures of one shot, fetched lazily as its output frames advance.

    ``positions[i]`` is the (fractional) source position of the shot's i-th
    rendered frame; frames are requested in non-decreasing order. A
    :class:`SubframeSource` is sampled exactly, chunk by chunk; a
    :class:`StreamingSource` is read through a sliding window of decoded
    frames; any other source delivers the whole shot at once. Fractional
    positions of frame-based sources are frame-blended (slow motion).
    """

    CHUNK_BYTES = 48_000_000  # memory budget of one SubframeSource chunk

    def __init__(self, shot: Shot, positions: np.ndarray, source: ClipSource, size: Size) -> None:
        self.shot, self.positions, self.source, self.size = shot, positions, source, size
        w, h = size
        self._shape = (h, w, 3)
        self._chunk_len = int(np.clip(self.CHUNK_BYTES // (w * h * 3), 4, 64))
        self._chunk: Optional[np.ndarray] = None
        self._c0 = 0
        self._buf: list[np.ndarray] = []  # _buf[k] is source frame _base + k
        self._base = 0
        self._iter: Optional[Iterator[np.ndarray]] = None
        n_src = int(math.ceil(positions[-1] - 1e-6)) + 1 if positions.size else 0
        if isinstance(source, SubframeSource):
            self._subframe = True
        else:
            self._subframe = False
            if isinstance(source, StreamingSource):
                self._iter = iter(source.iter_shot(shot, n_src, size))
            else:
                frames = source.frames_for_shot(shot, n_src, size)
                self._check(frames, n_src)
                self._buf = list(frames)
            if not self._fill(0):
                raise ValueError(f"source returned no frames for shot {shot.index}")

    def _check(self, frames: np.ndarray, n: int) -> None:
        if frames.ndim != 4 or frames.shape[1:] != self._shape or frames.shape[0] < min(n, 1):
            raise ValueError(f"source returned frames of shape {frames.shape}, expected (n, *{self._shape})")

    def _fill(self, k: int) -> bool:
        """Read the stream up to source frame ``k``; False if nothing is available at all."""
        while self._iter is not None and self._base + len(self._buf) <= k:
            frame = next(self._iter, None)
            if frame is None:
                self._iter = None
                break
            if frame.shape != self._shape:
                raise ValueError(f"source returned a frame of shape {frame.shape}, expected {self._shape}")
            self._buf.append(frame)
        return bool(self._buf)

    def _frame(self, k: int) -> np.ndarray:
        self._fill(k)
        k = int(np.clip(k - self._base, 0, len(self._buf) - 1))  # a short source holds its last frame
        return self._buf[k]

    def at(self, i: int) -> np.ndarray:
        """Picture for the shot's ``i``-th rendered frame."""
        if self._subframe:
            if self._chunk is None or not (self._c0 <= i < self._c0 + self._chunk.shape[0]):
                self._c0 = i
                positions = self.positions[i:i + self._chunk_len]
                chunk = self.source.sample_shot(self.shot, positions, self.size)  # type: ignore[attr-defined]
                self._check(chunk, 1)
                self._chunk = chunk
            return self._chunk[i - self._c0]
        pos = float(self.positions[i])
        k0 = int(math.floor(pos + 1e-6))
        if self._iter is not None and k0 > self._base:  # positions never go back: drop older frames
            drop = min(k0 - self._base, len(self._buf) - 1)
            del self._buf[:drop]
            self._base += drop
        a = self._frame(k0)
        frac = pos - k0
        if frac < 1e-3:
            return a
        b = self._frame(k0 + 1)
        return a if b is a else fx.blend(a, b, frac)

    def close(self) -> None:
        close = getattr(self._iter, "close", None)
        if close is not None:
            close()  # stops a decoder that was not read to the end
        self._iter, self._buf, self._chunk = None, [], None


class _ShotRenderer:
    """Renders the output frames of one shot, including handle frames beyond its cut."""

    def __init__(self, shot: Shot, own: tuple[int, int], pre: int, post: int,
                 source: ClipSource, size: Size, fps: float) -> None:
        self.shot = shot
        self.size = size
        self.first = own[0] - pre
        self.last = own[1] + post  # exclusive
        pos = source_positions(shot, fps, own)
        tail = (pos[-1] + 1.0 if pos.size else 0.0) + np.arange(post, dtype=float)
        positions = np.concatenate([np.arange(-pre, 0, dtype=float), pos, tail])
        positions -= positions[0]  # non-decreasing, so the first position is the in-point
        self._footage = _Footage(shot, positions, source, size)
        self._plan_effects(fps)

    def _plan_effects(self, fps: float) -> None:
        self.geometry = shot_geometry(self.shot, fps, np.arange(self.first, self.last), self.size)
        self.pixel: list[tuple[int, int, EffectBase]] = []
        for eff in self.shot.effects:
            if isinstance(eff, (Blur, RGBSplit, Invert, Flash)) and not _is_descriptive(eff):
                self.pixel.append((*frame_span(eff.start, eff.end, fps), eff))
        self.pixel.sort(key=lambda item: _PIXEL_ORDER[item[2].type])

    def close(self) -> None:
        self._footage.close()

    def frame(self, f: int) -> np.ndarray:
        i = int(np.clip(f - self.first, 0, self.last - self.first - 1))
        g = self.geometry
        img = fx.transform(self._footage.at(i), float(g.scale[i]), float(g.angle[i]),
                           float(g.tx[i]), float(g.ty[i]), g.center)
        for es, ee, eff in self.pixel:
            if not (es <= f < ee):
                continue
            if isinstance(eff, Blur):
                img = fx.blur(img, eff.amount)
            elif isinstance(eff, RGBSplit):
                img = fx.rgb_split(img, eff.offset)
            elif isinstance(eff, Invert):
                img = fx.invert(img)
            elif isinstance(eff, Flash):  # peaks at exactly ``intensity`` on the middle frame
                env = _peak(f, es, es + (ee - es - 1) // 2, ee)
                img = fx.flash(img, eff.intensity * env, tuple(int(c) for c in eff.color))
        look, layout = self.shot.look, self.shot.layout
        if look is not None and look.bw:
            img = fx.to_bw(img)
        if layout is not None:
            if layout.letterbox_top > 0 or layout.letterbox_bottom > 0:
                img = fx.letterbox(img, layout.letterbox_top, layout.letterbox_bottom)
            w = img.shape[1]
            left, right = int(round(layout.pillarbox_left * w)), int(round(layout.pillarbox_right * w))
            if left or right:
                img = img.copy()
                img[:, :left] = 0
                img[:, w - right:] = 0
        return img


# --------------------------------------------------------------------------- timeline


@dataclass
class _TransitionPlan:
    kind: str
    a: Optional[int]  # outgoing shot (position in the shot list), None for a fade-in
    b: Optional[int]  # incoming shot, None for a fade-out
    f0: int
    fb: int  # boundary: first frame of the incoming shot
    f1: int
    color: tuple[int, int, int]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Timeline:
    shots: list[Shot]
    own: list[tuple[int, int]]  # output frame range owned by each shot
    pre: list[int]  # handle frames needed before / after each shot
    post: list[int]
    transitions: list[_TransitionPlan]
    overlays: list[_OverlayPlan]
    n_frames: int


def _default_color(kind: str) -> tuple[int, int, int]:
    return WHITE if kind in ("flash", "fade_white") else BLACK


def _plan_timeline(edit: Edit, fps: float, n: int) -> _Timeline:
    shots: list[Shot] = []
    raw_transitions: list[tuple[Transition, Optional[int], Optional[int]]] = []
    for sec in edit.timeline.sections:
        for item in sec.items:
            if isinstance(item, Shot):
                shots.append(item)
            else:
                raw_transitions.append((item, len(shots) - 1 if shots else None, len(shots)))
    if not shots:
        shots = [Shot(index=0, start=0.0, end=n / fps)]
    starts = [0]
    for s in shots[1:]:
        starts.append(min(n, max(starts[-1], int(round(s.start * fps)))))
    own = [(starts[i], starts[i + 1] if i + 1 < len(shots) else n) for i in range(len(shots))]
    by_id = {s.id: i for i, s in enumerate(shots) if s.id}
    pre, post = [0] * len(shots), [0] * len(shots)

    plans: list[_TransitionPlan] = []
    for tr, prev, nxt in raw_transitions:
        if tr.kind in ("cut", "unknown"):
            continue
        nxt = nxt if nxt is not None and nxt < len(shots) else None
        if (tr.from_shot in by_id) or (tr.to_shot in by_id):  # ids are authoritative
            a = by_id.get(tr.from_shot, prev) if tr.from_shot else None
            b = by_id.get(tr.to_shot, nxt) if tr.to_shot else None
        else:
            a, b = prev, nxt
        if (a is None and b is None) or any(i is not None and own[i][1] <= own[i][0] for i in (a, b)):
            continue
        f0, f1 = frame_span(tr.start, tr.end, fps, min_frames=0)
        if b is None:  # fade-out at the end
            fb = n
            f0, f1 = min(f0, n - 1), n
        elif a is None:  # fade-in at the start
            fb = own[b][0]
            f0, f1 = fb, max(f1, fb + 1)
        else:
            fb = own[b][0]
            f0, f1 = max(min(f0, fb), own[a][0]), min(max(f1, fb), own[b][1])
            if f1 - f0 < 2:  # a non-cut transition is at least visible around the cut
                f0, f1 = max(own[a][0], fb - 1), min(own[b][1], fb + 1)
            if tr.kind in _HANDLE_KINDS:
                post[a] = max(post[a], f1 - fb)
                pre[b] = max(pre[b], fb - f0)
        if f1 <= f0:
            continue
        color = tuple(int(c) for c in tr.color) if tr.color is not None else _default_color(tr.kind)
        plans.append(_TransitionPlan(tr.kind, a, b, f0, fb, f1, color, dict(tr.params)))
    overlays = [p for p in (_plan_overlay(ov, fps) for ov in edit.overlays) if p is not None]
    return _Timeline(shots, own, pre, post, plans, overlays, n)


def _peak(f: int, f0: int, fb: int, f1: int) -> float:
    """Envelope in (0, 1] rising over ``[f0, fb)`` and falling over ``[fb, f1)``; 1 on ``fb``."""
    if f < fb:
        return (f - f0 + 1) / (fb - f0 + 1)
    return (f1 - f) / max(1, f1 - fb)


def _motion_blur(img: np.ndarray, length: int, horizontal: bool) -> np.ndarray:
    if length <= 1:
        return img
    return cv2.blur(img, (length, 1) if horizontal else (1, length))


def _compose(plan: _TransitionPlan, f: int, get: Callable[[int], _ShotRenderer], owner: int) -> np.ndarray:
    kind = plan.kind
    if plan.a is None or plan.b is None:  # fade from / to colour at the ends of the video
        img = get(owner).frame(f)
        span = max(1, plan.f1 - plan.f0)
        env = (plan.f1 - f) / span if plan.a is None else (f - plan.f0 + 1) / span
        return fx.flash(img, fx.ease(env, "ease_in_out") if kind != "flash" else env, plan.color)
    if kind in _HANDLE_KINDS:
        out_img, in_img = get(plan.a).frame(f), get(plan.b).frame(f)
        p = (f - plan.f0 + 0.5) / (plan.f1 - plan.f0)
        if kind == "dissolve":
            return fx.blend(out_img, in_img, p)
        return _slide(out_img, in_img, p, str(plan.params.get("direction", "left")))
    img = get(owner).frame(f)
    e = _peak(f, plan.f0, plan.fb, plan.f1)
    before = f < plan.fb
    if kind == "flash":
        return fx.flash(img, e * float(plan.params.get("intensity", 1.0)), plan.color)
    if kind in ("fade_black", "fade_white"):
        return fx.flash(img, fx.ease(e, "ease_in_out"), plan.color)
    h, w = img.shape[:2]
    if kind == "whip":
        direction = str(plan.params.get("direction", "left"))
        horizontal = direction in ("left", "right")
        sign = -1.0 if direction in ("left", "up") else 1.0
        shift = sign * 0.5 * e * e * (1.0 if before else -1.0)
        moved = fx.transform(img, tx=shift if horizontal else 0.0, ty=0.0 if horizontal else shift)
        return _motion_blur(moved, int(e * 0.15 * (w if horizontal else h)), horizontal)
    if kind == "zoom":
        return fx.blur(fx.transform(img, scale=1.0 + e * e), 0.012 * e)
    if kind == "spin":  # rotate out and back in, magnified so the corners stay filled
        geo = Geometry(np.ones(1), np.array([(120.0 if before else -120.0) * e * e]), np.zeros(1), np.zeros(1))
        _cover(geo, (w, h))
        return fx.blur(fx.transform(img, float(geo.scale[0]), float(geo.angle[0])), 0.01 * e)
    if kind == "glitch":
        out = fx.rgb_split(img, 0.03 * e)
        rng = np.random.default_rng(f)
        for _ in range(4):
            y0 = int(rng.integers(0, h))
            y1 = min(h, y0 + int(rng.integers(2, max(3, h // 8))))
            out[y0:y1] = np.roll(out[y0:y1], int(rng.normal(0.0, 0.06 * w * e)), axis=1)
        return out
    return img


def _slide(out_img: np.ndarray, in_img: np.ndarray, p: float, direction: str) -> np.ndarray:
    """The incoming shot pushes the outgoing one out of frame (``direction`` = motion of the content)."""
    horizontal = direction in ("left", "right")
    extent = out_img.shape[1] if horizontal else out_img.shape[0]
    s = int(round(np.clip(p, 0.0, 1.0) * extent))
    axis = 1 if horizontal else 0
    if direction in ("right", "down"):  # mirror the problem so the code below always pushes left / up
        out_img, in_img = np.flip(out_img, axis), np.flip(in_img, axis)
    a = np.take(out_img, np.arange(s, extent), axis=axis)
    b = np.take(in_img, np.arange(0, s), axis=axis)
    res = np.concatenate([a, b], axis=axis)
    return np.ascontiguousarray(np.flip(res, axis) if direction in ("right", "down") else res)


# --------------------------------------------------------------------------- public API


def as_source(sources: Sources) -> ClipSource:
    """A :class:`ClipSource` instance, or a path / list of paths wrapped in :class:`VideoClipSource`."""
    if isinstance(sources, ClipSource):
        return sources
    if isinstance(sources, (str, Path)):
        return VideoClipSource([sources])
    return VideoClipSource(list(sources))


def render_frames(
    edit: Edit,
    sources: Sources,
    size: Optional[Size] = None,
    fps: Optional[float] = None,
    progress: Optional[ProgressFn] = None,
) -> Iterator[np.ndarray]:
    """Yield the rendered RGB frames of ``edit`` one by one (see :func:`render_edit`)."""
    say = progress or (lambda _msg: None)
    size, fps, n = output_format(edit, size, fps)
    source = as_source(sources)
    source.prepare(edit, size, fps)
    tl = _plan_timeline(edit, fps, n)
    starts = np.array([o[0] for o in tl.own])
    owner = np.clip(np.searchsorted(starts, np.arange(n), side="right") - 1, 0, len(tl.shots) - 1)
    active = np.full(n, -1)
    for i, plan in enumerate(tl.transitions):
        active[plan.f0:plan.f1] = i
    first_needed = [o[0] - p for o, p in zip(tl.own, tl.pre)]
    last_needed = [o[1] + p for o, p in zip(tl.own, tl.post)]
    live: dict[int, _ShotRenderer] = {}
    created = -1

    def get(i: int) -> _ShotRenderer:
        nonlocal created
        while created < i:  # create renderers in shot order so sources see shots in sequence
            created += 1
            own = tl.own[created]
            if own[1] - own[0] + tl.pre[created] + tl.post[created] > 0:
                say(f"rendering shot {created + 1}/{len(tl.shots)}")
                live[created] = _ShotRenderer(tl.shots[created], own, tl.pre[created], tl.post[created],
                                              source, size, fps)
        return live[i]

    try:
        for f in range(n):
            for i in range(created + 1, len(tl.shots)):  # shots whose handles start here (dissolve)
                if first_needed[i] <= f:
                    get(i)
                else:
                    break
            t_idx = int(active[f])
            if t_idx >= 0:
                frame = _compose(tl.transitions[t_idx], f, get, int(owner[f]))
            else:
                frame = get(int(owner[f])).frame(f)
            for ov in tl.overlays:
                if ov.f0 <= f < ov.f1:
                    frame = ov.draw(frame, f, fps)
            yield frame
            for i in [i for i in live if last_needed[i] <= f + 1]:  # free footage no longer on screen
                live.pop(i).close()
    finally:
        for renderer in live.values():
            renderer.close()


def render_edit(
    edit: Edit,
    sources: Sources,
    out_path: Union[str, Path],
    music: Union[str, Path, None] = None,
    size: Optional[Size] = None,
    fps: Optional[float] = None,
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Render ``edit`` with footage from ``sources`` into an H.264 MP4 at ``out_path``.

    ``sources`` is a :class:`ClipSource` or a list of video paths (cut round-robin
    by :class:`VideoClipSource`); ``music`` is muxed (trimmed to the video).
    ``size`` is ``(width, height)``; defaults are described in :func:`output_format`.
    """
    if music is not None and not Path(music).exists():
        raise FileNotFoundError(f"music file not found: {music}")
    size, fps, _ = output_format(edit, size, fps)
    out_path = Path(out_path)
    with VideoWriter(out_path, size[0], size[1], fps, audio_path=music) as writer:
        for frame in render_frames(edit, sources, size, fps, progress):
            writer.write(frame)
    if progress is not None:
        progress(f"wrote {writer.frames_written} frames to {out_path}")
    return out_path


# --------------------------------------------------------------------------- retiming


def retime_edit(edit: Edit, bpm: float, offset: float = 0.0) -> Edit:
    """Deep copy of ``edit`` moved from its music grid onto a new tempo grid.

    Every time maps ``t -> offset + (t - old_offset) * old_bpm / bpm`` where
    ``old_offset`` is the edit's first beat, so cuts that landed on beats of the
    original song land on beats of the new one. Durations, rates and frame
    indices (``media.analysis_fps``) follow; shot lengths *in beats* do not
    change. Times that would fall before 0 are clamped (the first shot always
    starts at 0, and shots squeezed to nothing are dropped). Edits without a
    music BPM come back as an unchanged copy.
    """
    out = edit.model_copy(deep=True)
    music = out.music
    if music is None or not music.bpm or bpm <= 0:
        return out
    old_offset = music.beat_offset if music.beat_offset is not None else (music.beats[0] if music.beats else 0.0)
    k = music.bpm / bpm

    def raw(t: float) -> float:
        return offset + (t - old_offset) * k

    def m(t: float) -> float:
        return max(0.0, raw(t))

    def events(times: list[float]) -> list[float]:
        return [raw(t) for t in times if raw(t) >= 0.0]

    fps_a = out.media.analysis_fps or 30.0
    for node in out.walk():
        if isinstance(node, Timed):
            node.start, node.end = m(node.start), m(node.end)
        if isinstance(node, EffectBase):
            for kf in node.keyframes:
                kf.t = m(kf.t)
        if isinstance(node, (Shot, Transition)) and node.beat_offset is not None:
            node.beat_offset *= k
        if isinstance(node, TextOverlay):
            for word in node.words:
                word.start, word.end = m(word.start), m(word.end)
            node.words = [w for w in node.words if w.end > w.start]
    for sec in out.timeline.sections:
        sec.cuts_per_second /= k
        sec.mean_shot_duration *= k
    for pat in out.patterns:
        pat.start = m(pat.start) if pat.start is not None else None
        pat.end = m(pat.end) if pat.end is not None else None

    music.bpm = float(bpm)
    music.beat_offset = raw(old_offset)
    music.beats, music.downbeats, music.onsets = events(music.beats), events(music.downbeats), events(music.onsets)
    music.drops = [d.model_copy(update={"time": raw(d.time)}) for d in music.drops if raw(d.time) >= 0.0]
    music.sections = [s for s in music.sections if s.end > s.start]
    if music.energy is not None:
        music.energy.t0, music.energy.dt = raw(music.energy.t0), music.energy.dt * k

    if out.stats is not None:
        st = out.stats
        st.mean_shot_duration *= k
        st.median_shot_duration *= k
        st.min_shot_duration *= k
        st.max_shot_duration *= k
        st.cuts_per_minute /= k
        st.hook_duration *= k
        st.first_cut_time = m(st.first_cut_time) if st.first_cut_time is not None else None
        if st.pacing is not None:
            st.pacing.t0, st.pacing.dt = raw(st.pacing.t0), st.pacing.dt * k
            st.pacing.values = [v / k for v in st.pacing.values]

    _drop_collapsed(out)
    shots = out.shots()
    if shots:
        shots[0].start = 0.0
        out.timeline.sections[0].start = 0.0
        for s in shots:
            s.frame_start, s.frame_end = int(round(s.start * fps_a)), int(round(s.end * fps_a))
        out.media.duration = max(m(edit.media.duration), shots[-1].end)
    else:
        out.media.duration = m(edit.media.duration)
    out.media.n_frames = int(round(out.media.duration * fps_a))
    out.provenance = {**out.provenance, "retimed": {"from_bpm": edit.music.bpm, "to_bpm": float(bpm),
                                                    "offset": float(offset)}}
    return out


def _drop_collapsed(edit: Edit) -> None:
    """Remove nodes squeezed to zero length by clamping at t=0 and re-index shots."""
    removed: set[str] = set()
    for sec in edit.timeline.sections:
        kept = []
        for item in sec.items:
            if isinstance(item, Shot):
                item.effects = [e for e in item.effects if e.end > e.start]
                if item.end <= item.start:
                    removed.add(item.id)
                    continue
            kept.append(item)
        sec.items = kept
    edit.overlays = [ov for ov in edit.overlays if ov.end > ov.start]
    if not removed:
        return
    for sec in edit.timeline.sections:
        sec.items = [it for it in sec.items if not (isinstance(it, Transition) and
                                                     (it.from_shot in removed or it.to_shot in removed))]
    edit.timeline.sections = [s for s in edit.timeline.sections if s.items]
    had_ids = any(s.id for s in edit.shots())
    old_ids = [s.id for s in edit.shots()]
    for i, s in enumerate(edit.shots()):
        s.index = i
    if had_ids:
        new_ids = {old: f"s{i + 1}" for i, old in enumerate(old_ids)}
        for tr in edit.transitions():
            tr.from_shot = new_ids.get(tr.from_shot, tr.from_shot) if tr.from_shot else None
            tr.to_shot = new_ids.get(tr.to_shot, tr.to_shot) if tr.to_shot else None
        assign_ids(edit)
