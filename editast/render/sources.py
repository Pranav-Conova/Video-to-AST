"""Footage providers for the renderer.

The compiler (:mod:`editast.render.compiler`) walks the shots of an Edit in
playback order and asks a :class:`ClipSource` for the raw footage behind each
one: ``frames_for_shot(shot, n, size)`` must return ``n`` *consecutive* source
frames at the output frame rate, already cropped to the output size. The
compiler decides how many it needs (speed ramps and freezes consume source
footage at a different rate than the timeline advances, and dissolves need a
few frames of handle before/after a shot).

Two optional extensions let the compiler avoid holding a whole shot in memory:

* :class:`StreamingSource` — ``iter_shot`` yields the same frames one by one;
  the compiler keeps only a sliding window of them.
* :class:`SubframeSource` — ``sample_shot`` returns frames at *fractional*
  source positions. Procedural footage can do this exactly, which keeps slow
  motion smooth instead of blending neighbours; the compiler samples long
  shots in chunks.

Sources provided here:

* :class:`VideoClipSource` — the user's own clips. Only the segment a shot
  needs is decoded (``ffmpeg -ss``) and streamed; frames are scaled and
  centre-cropped to fill the output aspect (no letterbox). Shots sharing a
  ``Shot.source_group`` label map to the same clip, other shots are assigned
  round-robin (or randomly), and every clip keeps an in-point that advances
  with use so a reused clip shows new footage.
* :class:`SyntheticSource` — procedural camera moves over textures from
  :mod:`editast.testing.synth`; each source group / shot gets its own texture
  and an optional per-shot camera velocity. Used by the benchmark, where the
  footage must be known exactly.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from typing import Iterator, Literal, Optional, Protocol, Sequence, Union, runtime_checkable

import cv2
import numpy as np

from editast.ast.nodes import Edit, Shot
from editast.media import MediaError, Probe, iter_video, probe
from editast.testing import synth

Size = tuple[int, int]  # (width, height), as everywhere in editast.media
PathLike = Union[str, Path]


@runtime_checkable
class ClipSource(Protocol):
    """Provides the raw footage for every shot of an edit.

    The compiler calls :meth:`prepare` once, then asks for the footage of each
    rendered shot exactly once, in playback order (``frames_for_shot``, or the
    extension methods below when the source implements them).
    """

    def prepare(self, edit: Edit, size: Size, fps: float) -> None:
        """Called once before rendering with the output ``(width, height)`` and fps."""

    def frames_for_shot(self, shot: Shot, n_source_frames: int, size: Size) -> np.ndarray:
        """``(n_source_frames, h, w, 3)`` uint8 RGB consecutive frames at the output fps."""


@runtime_checkable
class StreamingSource(ClipSource, Protocol):
    """A source that can deliver a shot's frames one at a time."""

    def iter_shot(self, shot: Shot, n_source_frames: int, size: Size) -> Iterator[np.ndarray]:
        """Yield the frames ``frames_for_shot`` would return (the in-point is claimed on call)."""


@runtime_checkable
class SubframeSource(ClipSource, Protocol):
    """A source that can render arbitrary (fractional) source positions."""

    def sample_shot(self, shot: Shot, positions: np.ndarray, size: Size) -> np.ndarray:
        """Frames at ``positions`` (source frames since the shot's in-point, >= 0, non-decreasing).

        Must be deterministic per ``(shot, position)``: the compiler may call it
        several times per shot with consecutive chunks of positions.
        """


def _group_key(shot: Shot) -> str:
    """Shots with the same ``source_group`` share footage; others are unique."""
    return f"group:{shot.source_group}" if shot.source_group else f"shot:{shot.index}"


# --------------------------------------------------------------------------- user clips


class VideoClipSource:
    """Cut the user's clips into the shots of an edit.

    ``assign``: ``"round_robin"`` cycles through the clips in shot order,
    ``"random"`` picks clips with a seeded RNG. Either way shots with equal
    ``source_group`` labels reuse one clip. A clip that runs out of fresh
    footage starts again from its beginning; a clip shorter than a shot holds
    its last frame.
    """

    def __init__(
        self,
        paths: Union[PathLike, Sequence[PathLike]],
        assign: Literal["round_robin", "random"] = "round_robin",
        seed: int = 0,
    ) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("VideoClipSource needs at least one clip")
        if assign not in ("round_robin", "random"):
            raise ValueError(f"unknown assign mode {assign!r}")
        self.assign = assign
        self.seed = seed
        self.fps = 30.0
        self._probes: list[Probe] = []
        self._inpoints: list[float] = []
        self._by_key: dict[str, int] = {}
        self._next = 0
        self._rng = np.random.default_rng(seed)

    def prepare(self, edit: Edit, size: Size, fps: float) -> None:
        self.fps = fps
        self._probes = [probe(p) for p in self.paths]
        for path, info in zip(self.paths, self._probes):
            if not info.has_video or info.width <= 0 or info.height <= 0:
                raise MediaError(f"no video stream in {path}")
        self._inpoints = [0.0] * len(self.paths)
        self._by_key = {}
        self._next = 0
        self._rng = np.random.default_rng(self.seed)
        for shot in edit.shots():
            self.clip_for_shot(shot)

    def clip_for_shot(self, shot: Shot) -> int:
        """Index into ``paths`` of the clip used for ``shot`` (stable per source group)."""
        key = _group_key(shot)
        if key not in self._by_key:
            if self.assign == "random":
                self._by_key[key] = int(self._rng.integers(len(self.paths)))
            else:
                self._by_key[key] = self._next % len(self.paths)
                self._next += 1
        return self._by_key[key]

    def frames_for_shot(self, shot: Shot, n_source_frames: int, size: Size) -> np.ndarray:
        w, h = size
        if n_source_frames <= 0:
            return np.zeros((0, h, w, 3), np.uint8)
        return np.stack(list(self.iter_shot(shot, n_source_frames, size)))

    def iter_shot(self, shot: Shot, n_source_frames: int, size: Size) -> Iterator[np.ndarray]:
        if not self._probes:
            raise MediaError("VideoClipSource.prepare() must be called before rendering")
        if n_source_frames <= 0:
            return iter(())
        clip = self.clip_for_shot(shot)
        need = n_source_frames / self.fps
        start = self._inpoints[clip]
        duration = self._probes[clip].duration
        if duration > 0 and start + need > duration:
            start = 0.0  # out of fresh footage: wrap around to the start of the clip
        self._inpoints[clip] = start + need  # claimed now, so later shots get fresh footage
        return self._stream(clip, start, need, n_source_frames, size)

    def _stream(self, clip: int, start: float, need: float, n: int, size: Size) -> Iterator[np.ndarray]:
        w, h = size
        info = self._probes[clip]
        dw, dh = _cover_size(info.width, info.height, w, h)
        x0, y0 = (dw - w) // 2, (dh - h) // 2
        count = 0
        last: Optional[np.ndarray] = None
        for begin in ((start, 0.0) if start > 0 else (0.0,)):  # retry from 0 if the seek found nothing
            try:
                with closing(iter_video(self.paths[clip], fps=self.fps, size=(dw, dh), start=begin,
                                        duration=need + 1.0 / self.fps, chunk_frames=8)) as frames:
                    for frame in frames:
                        if count >= n:
                            break
                        last = np.ascontiguousarray(frame[y0:y0 + h, x0:x0 + w])
                        count += 1
                        yield last
            except MediaError:
                if last is None and begin == 0.0:
                    raise
            if last is not None:
                break
        if last is None:
            raise MediaError(f"no frames decoded from {self.paths[clip]}")
        for _ in range(n - count):  # clip shorter than the shot: hold the last frame
            yield last


def _bounce(s: float, span: float) -> float:
    """Triangle wave: ``s`` folded back and forth into ``[0, span]``."""
    return span - abs(s % (2.0 * span) - span)


def _cover_size(src_w: int, src_h: int, w: int, h: int) -> Size:
    """Smallest aspect-preserving size of the source that covers ``(w, h)``."""
    if src_w <= 0 or src_h <= 0:
        return w, h
    scale = max(w / src_w, h / src_h)
    return max(w, int(math.ceil(src_w * scale - 1e-6))), max(h, int(math.ceil(src_h * scale - 1e-6)))


# --------------------------------------------------------------------------- procedural footage


class SyntheticSource:
    """Procedural footage: a virtual camera moving over a synthetic texture.

    Each source group (or each shot without a group) gets its own texture,
    with consecutive groups using consecutive texture seeds so neighbouring
    shots have clearly different palettes. ``motion`` maps a shot index to
    the camera velocity in output px per source frame (content moves the
    opposite way); missing shots are static. A shot that reuses a group
    starts where the previous shot of that group stopped, shifted to another
    part of the texture, so the reuse is a new view of the same "clip".

    The texture is mirror-tiled into a seamless periodic canvas, so the camera
    can travel arbitrarily far without hitting an edge (which would reverse
    the apparent motion of reflected borders).

    With ``subjects=True`` every shot also shows a small moving subject (an
    elliptical patch covering ~1 % of the frame that drives back and forth
    inside it). A static shot then always has *some* life, so it cannot be
    mistaken for a freeze frame, while the global (camera) motion that pan /
    zoom analyzers measure is unaffected. The patch is the colour negative of
    a crop of the shot's own canvas: its detail scale matches the background,
    so it does not dominate frequency- or corner-based global motion
    estimates, yet it clearly stands out in frame differences. ``grain`` adds
    luminance noise (std in 8-bit levels) that changes with the source
    position, like sensor noise: held (frozen) frames keep identical grain,
    live frames do not.
    """

    #: Texture tile size relative to the frame (bigger = mirror seams show up later).
    margin = 1.5
    #: Textures are synthesised with this long side at most and upscaled (speed).
    detail_long_side = 640
    #: Canvases kept in memory (least recently used ones are dropped).
    cache_size = 3
    #: Subject ellipse size (fractions of frame width, height) and speed (fraction of width per frame).
    subject_size = (0.16, 0.09)
    subject_speed = (0.015, 0.03)

    def __init__(self, seed: int = 0, motion: Optional[dict[int, tuple[float, float]]] = None,
                 subjects: bool = True, grain: float = 2.0) -> None:
        self.seed = seed
        self.motion = dict(motion or {})
        self.subjects = subjects
        self.grain = grain
        self._grain_bank: list[tuple[np.ndarray, np.ndarray]] = []
        self._tex_seed: dict[str, int] = {}
        self._canvas: OrderedDict[str, np.ndarray] = OrderedDict()
        self._camera: dict[str, tuple[float, float]] = {}  # group -> camera at its last sampled frame
        self._origin: dict[int, tuple[float, float]] = {}  # shot index -> camera at position 0
        self._sprite: Optional[tuple[int, Size, np.ndarray, np.ndarray]] = None

    def prepare(self, edit: Edit, size: Size, fps: float) -> None:
        self._grain_bank = self._make_grain(size)
        self._canvas = OrderedDict()
        self._camera = {}
        self._origin = {}
        self._tex_seed = {}
        self._sprite = None
        for shot in edit.shots():
            self.texture_seed(shot)

    def texture_seed(self, shot: Shot) -> int:
        """Seed of the texture behind ``shot`` (consecutive per new source group)."""
        key = _group_key(shot)
        if key not in self._tex_seed:
            self._tex_seed[key] = 1009 * self.seed + 1 + len(self._tex_seed)
        return self._tex_seed[key]

    def frames_for_shot(self, shot: Shot, n_source_frames: int, size: Size) -> np.ndarray:
        return self.sample_shot(shot, np.arange(max(0, n_source_frames), dtype=float), size)

    def sample_shot(self, shot: Shot, positions: np.ndarray, size: Size) -> np.ndarray:
        w, h = size
        positions = np.asarray(positions, dtype=float)
        canvas = self._canvas_for(shot, size)
        ch, cw = canvas.shape[:2]
        key = _group_key(shot)
        vx, vy = self.motion.get(shot.index, (0.0, 0.0))
        if shot.index not in self._origin:
            if key in self._camera:  # reused clip: continue, then cut to another part of the texture
                px, py = self._camera[key]
                self._origin[shot.index] = (px + 0.43 * cw, py + 0.37 * ch)
            else:
                self._origin[shot.index] = (cw / 4.0, ch / 4.0)  # centre of the unflipped tile
        cx, cy = self._origin[shot.index]
        out = np.empty((positions.size, h, w, 3), np.uint8)
        for i, p in enumerate(positions):
            # Wrap into the canvas period: warpAffine stores coordinates as int16.
            x = (cx + vx * p) % cw
            y = (cy + vy * p) % ch
            m = np.array([[1.0, 0.0, w / 2.0 - x], [0.0, 1.0, h / 2.0 - y]], np.float32)
            out[i] = cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        if self.subjects:
            self._draw_subject(out, shot, positions, size, canvas)
        if self.grain > 0:
            if not self._grain_bank or self._grain_bank[0][0].shape != (h, w, 3):
                self._grain_bank = self._make_grain(size)
            for i, p in enumerate(positions):
                up, down = self._grain_bank[int(math.floor(3.0 * p + 0.5)) % len(self._grain_bank)]
                out[i] = cv2.subtract(cv2.add(out[i], up), down)
        if positions.size:
            last = float(positions[-1])
            self._camera[key] = (cx + vx * last, cy + vy * last)
        return out

    def _make_grain(self, size: Size, n: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
        """``n`` grain frames as (positive, negative) uint8 parts for saturating add / subtract."""
        if self.grain <= 0:
            return []
        w, h = size
        rng = np.random.default_rng([self.seed, 99])
        bank = []
        for _ in range(n):
            g = np.round(rng.standard_normal((h, w)) * self.grain).clip(-64, 64)
            g3 = np.repeat(g[..., None], 3, axis=2)
            bank.append((np.maximum(g3, 0).astype(np.uint8), np.maximum(-g3, 0).astype(np.uint8)))
        return bank

    def _subject_sprite(self, shot: Shot, size: Size, canvas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Padded float RGB patch and soft elliptical alpha mask of the subject of ``shot``."""
        if self._sprite is not None and self._sprite[:2] == (shot.index, size):
            return self._sprite[2], self._sprite[3]
        w, h = size
        rng = np.random.default_rng([self.seed, shot.index, 11])
        sw, sh = max(4, int(self.subject_size[0] * w)), max(4, int(self.subject_size[1] * h))
        ch, cw = canvas.shape[:2]
        # Colour negative of an upside-down crop of the canvas: same detail scale, distinct look.
        y0, x0 = int(rng.integers(0, ch // 2 - sh)), int(rng.integers(0, cw // 2 - sw))
        patch = 255.0 - canvas[ch - y0 - sh:ch - y0, x0:x0 + sw][::-1].astype(np.float32)
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        mask = (((xx + 0.5 - sw / 2) / (sw / 2)) ** 2 + ((yy + 0.5 - sh / 2) / (sh / 2)) ** 2 <= 1.0)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)
        patch = cv2.copyMakeBorder(patch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        self._sprite = (shot.index, size, patch, mask)
        return patch, mask

    def _draw_subject(self, frames: np.ndarray, shot: Shot, positions: np.ndarray, size: Size,
                      canvas: np.ndarray) -> None:
        """Composite the subject, bouncing back and forth inside the frame, in place.

        It never leaves the picture, so a live static shot always has motion
        somewhere and only a real freeze holds every pixel.
        """
        w, h = size
        sprite, mask = self._subject_sprite(shot, size, canvas)
        sh, sw = sprite.shape[0] - 4, sprite.shape[1] - 4
        size_p = (sprite.shape[1], sprite.shape[0])
        rng = np.random.default_rng([self.seed, shot.index, 7])
        x_lo, y_lo = sw / 2 + 2, 0.25 * h
        x_span, y_span = max(1.0, w - sw - 4), 0.55 * h  # ranges of the subject centre
        x0, y0 = rng.uniform(0.0, x_span), rng.uniform(0.1, 0.9) * y_span
        u = float(rng.choice([-1.0, 1.0])) * rng.uniform(*self.subject_speed) * w  # px per source frame
        v = rng.uniform(-0.002, 0.002) * h
        for i, p in enumerate(positions):
            cx = x_lo + _bounce(x0 + u * p, x_span)
            cy = y_lo + _bounce(y0 + v * p, y_span)
            x, y = cx - sw / 2 - 2, cy - sh / 2 - 2  # top-left of the padded sprite
            ix, iy = int(math.floor(x)), int(math.floor(y))
            fx0, fy0 = max(0, ix), max(0, iy)
            fx1, fy1 = min(w, ix + size_p[0]), min(h, iy + size_p[1])
            if fx1 <= fx0 or fy1 <= fy0:
                continue
            m = np.array([[1.0, 0.0, x - ix], [0.0, 1.0, y - iy]], np.float32)  # sub-pixel shift
            spr = cv2.warpAffine(sprite, m, size_p, flags=cv2.INTER_LINEAR)
            alpha = cv2.warpAffine(mask, m, size_p, flags=cv2.INTER_LINEAR)[..., None]
            sx, sy = fx0 - ix, fy0 - iy
            a = alpha[sy:sy + fy1 - fy0, sx:sx + fx1 - fx0]
            roi = frames[i, fy0:fy1, fx0:fx1].astype(np.float32)
            roi = roi * (1.0 - a) + spr[sy:sy + fy1 - fy0, sx:sx + fx1 - fx0] * a
            frames[i, fy0:fy1, fx0:fx1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)

    def _canvas_for(self, shot: Shot, size: Size) -> np.ndarray:
        key = _group_key(shot)
        tw, th = int(round(size[0] * self.margin)), int(round(size[1] * self.margin))
        canvas = self._canvas.get(key)
        if canvas is None or canvas.shape[:2] != (2 * th, 2 * tw):
            k = min(1.0, self.detail_long_side / max(tw, th))
            tile = synth.texture(max(8, int(round(th * k))), max(8, int(round(tw * k))), self.texture_seed(shot))
            if k < 1.0:
                tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_CUBIC)
            top = np.concatenate([tile, tile[:, ::-1]], axis=1)
            canvas = np.ascontiguousarray(np.concatenate([top, top[::-1]], axis=0))
            self._canvas[key] = canvas
            while len(self._canvas) > self.cache_size:
                self._canvas.popitem(last=False)
        self._canvas.move_to_end(key)
        return canvas
