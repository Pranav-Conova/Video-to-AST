"""Footage providers for the renderer.

The compiler (:mod:`editast.render.compiler`) walks the shots of an Edit in
playback order and asks a :class:`ClipSource` for the raw footage behind each
one: ``frames_for_shot(shot, n, size)`` must return ``n`` *consecutive* source
frames at the output frame rate, already cropped to the output size. The
compiler decides how many it needs (speed ramps and freezes consume source
footage at a different rate than the timeline advances, and dissolves need a
few frames of handle before/after a shot).

Sources may additionally implement :class:`SubframeSource` — ``sample_shot``
returns frames at *fractional* source positions. Procedural footage can do
this exactly, which keeps slow motion smooth instead of blending neighbours.

* :class:`VideoClipSource` — the user's own clips. Only the segment a shot
  needs is decoded (``ffmpeg -ss``); frames are scaled and centre-cropped to
  fill the output aspect (no letterbox). Shots sharing a
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
from pathlib import Path
from typing import Literal, Optional, Protocol, Sequence, Union, runtime_checkable

import cv2
import numpy as np

from editast.ast.nodes import Edit, Shot
from editast.media import MediaError, Probe, probe, read_video
from editast.testing import synth

Size = tuple[int, int]  # (width, height), as everywhere in editast.media
PathLike = Union[str, Path]


@runtime_checkable
class ClipSource(Protocol):
    """Provides the raw footage for every shot of an edit.

    The compiler calls :meth:`prepare` once, then ``frames_for_shot`` (or
    ``sample_shot`` for a :class:`SubframeSource`) exactly once per rendered
    shot, in playback order.
    """

    def prepare(self, edit: Edit, size: Size, fps: float) -> None:
        """Called once before rendering with the output ``(width, height)`` and fps."""

    def frames_for_shot(self, shot: Shot, n_source_frames: int, size: Size) -> np.ndarray:
        """``(n_source_frames, h, w, 3)`` uint8 RGB consecutive frames at the output fps."""


@runtime_checkable
class SubframeSource(ClipSource, Protocol):
    """A source that can render arbitrary (fractional) source positions."""

    def sample_shot(self, shot: Shot, positions: np.ndarray, size: Size) -> np.ndarray:
        """Frames at ``positions`` (source frames since the shot's in-point, >= 0, non-decreasing)."""


def _group_key(shot: Shot) -> str:
    """Shots with the same ``source_group`` share footage; others are unique."""
    return f"group:{shot.source_group}" if shot.source_group else f"shot:{shot.index}"


# --------------------------------------------------------------------------- user clips


class VideoClipSource:
    """Cut the user's clips into the shots of an edit.

    ``assign``: ``"round_robin"`` cycles through the clips in shot order,
    ``"random"`` picks clips with a seeded RNG. Either way shots with equal
    ``source_group`` labels reuse one clip.
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
        if not self._probes:
            raise MediaError("VideoClipSource.prepare() must be called before rendering")
        clip = self.clip_for_shot(shot)
        info = self._probes[clip]
        need = n_source_frames / self.fps
        start = self._inpoints[clip]
        if info.duration > 0 and start + need > info.duration:
            start = 0.0  # out of fresh footage: wrap around to the start of the clip
        frames = self._decode(clip, start, need, size)
        if frames.shape[0] == 0 and start > 0:
            start = 0.0
            frames = self._decode(clip, start, need, size)
        if frames.shape[0] == 0:
            raise MediaError(f"no frames decoded from {self.paths[clip]}")
        self._inpoints[clip] = start + need
        if frames.shape[0] < n_source_frames:  # clip shorter than the shot: hold the last frame
            pad = np.repeat(frames[-1:], n_source_frames - frames.shape[0], axis=0)
            frames = np.concatenate([frames, pad])
        return frames[:n_source_frames]

    def _decode(self, clip: int, start: float, duration: float, size: Size) -> np.ndarray:
        w, h = size
        info = self._probes[clip]
        dw, dh = _cover_size(info.width, info.height, w, h)
        try:
            frames = read_video(self.paths[clip], fps=self.fps, size=(dw, dh), start=start,
                                duration=duration + 1.0 / self.fps)
        except MediaError:
            return np.zeros((0, h, w, 3), np.uint8)
        x0, y0 = (dw - w) // 2, (dh - h) // 2
        return np.ascontiguousarray(frames[:, y0:y0 + h, x0:x0 + w])


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
    opposite way); missing shots are static. Shots of a reused group resume
    the camera where it stopped and then jump to a different part of the
    texture, so the reuse is a new view of the same "clip".

    The texture is mirror-tiled into a seamless periodic canvas, so the camera
    can travel arbitrarily far without hitting an edge (which would reverse
    the apparent motion of reflected borders).

    With ``subjects=True`` every shot also shows a small moving subject (a
    textured blob covering ~1.5 % of the frame that drifts across it, like a
    car driving by). A static shot then still has *some* life, so it cannot be
    mistaken for a freeze frame, while the global (camera) motion that
    pan / zoom analyzers measure is unaffected. ``grain`` adds luminance noise
    (std in 8-bit levels) that changes with the source position, like sensor
    noise: held (frozen) frames keep identical grain, live frames do not.
    """

    #: Texture tile size relative to the frame (bigger = mirror seams show up later).
    margin = 1.5
    #: Textures are synthesised with this long side at most and upscaled (speed).
    detail_long_side = 640

    def __init__(self, seed: int = 0, motion: Optional[dict[int, tuple[float, float]]] = None,
                 subjects: bool = True, grain: float = 2.0) -> None:
        self.seed = seed
        self.motion = dict(motion or {})
        self.subjects = subjects
        self.grain = grain
        self._grain_bank: list[tuple[np.ndarray, np.ndarray]] = []
        self._tex_seed: dict[str, int] = {}
        self._canvas: dict[str, np.ndarray] = {}
        self._camera: dict[str, tuple[float, float]] = {}
        self._last_use: dict[str, int] = {}

    def prepare(self, edit: Edit, size: Size, fps: float) -> None:
        self._grain_bank = self._make_grain(size)
        self._canvas = {}
        self._camera = {}
        self._tex_seed = {}
        self._last_use = {}
        for shot in edit.shots():
            self.texture_seed(shot)
            self._last_use[_group_key(shot)] = shot.index

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
        if key in self._camera:  # reused clip: continue, then cut to another part of the texture
            px, py = self._camera[key]
            cx, cy = px + 0.43 * cw, py + 0.37 * ch
        else:
            cx, cy = cw / 4.0, ch / 4.0  # centre of the unflipped tile
        out = np.empty((positions.size, h, w, 3), np.uint8)
        for i, p in enumerate(positions):
            # Keep the translation small: warpAffine stores coordinates as int16.
            x = (cx + vx * p) % cw
            y = (cy + vy * p) % ch
            m = np.array([[1.0, 0.0, w / 2.0 - x], [0.0, 1.0, h / 2.0 - y]], np.float32)
            out[i] = cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        if self.subjects:
            self._draw_subject(out, shot, positions, size)
        if self.grain > 0:
            if not self._grain_bank or self._grain_bank[0][0].shape != (h, w, 3):
                self._grain_bank = self._make_grain(size)
            for i, p in enumerate(positions):
                up, down = self._grain_bank[int(math.floor(3.0 * p + 0.5)) % len(self._grain_bank)]
                out[i] = cv2.subtract(cv2.add(out[i], up), down)
        last = float(positions[-1]) if positions.size else 0.0
        self._camera[key] = (cx + vx * last, cy + vy * last)
        if shot.index >= self._last_use.get(key, shot.index):
            self._canvas.pop(key, None)  # group not used again: free the canvas
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

    def _draw_subject(self, frames: np.ndarray, shot: Shot, positions: np.ndarray, size: Size) -> None:
        """Composite a small textured blob moving across the frame (wraps around), in place."""
        w, h = size
        rng = np.random.default_rng([self.seed, shot.index, 7])
        sw, sh = max(4, int(0.16 * w)), max(4, int(0.09 * h))
        sprite = synth.texture(sh, sw, self.texture_seed(shot) + 7919).astype(np.float32)
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        mask = (((xx + 0.5 - sw / 2) / (sw / 2)) ** 2 + ((yy + 0.5 - sh / 2) / (sh / 2)) ** 2 <= 1.0)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)
        pad = 2
        sprite = cv2.copyMakeBorder(sprite, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = rng.uniform(0.15, 0.85) * w, rng.uniform(0.3, 0.75) * h
        u = float(rng.choice([-1.0, 1.0])) * rng.uniform(0.006, 0.012) * w  # px per source frame
        v = rng.uniform(-0.002, 0.002) * h
        span_x, span_y = w + 2 * sw, h + 2 * sh
        for i, p in enumerate(positions):
            x = (x0 + u * p + sw) % span_x - sw - sw / 2 - pad  # top-left of the padded sprite
            y = (y0 + v * p + sh) % span_y - sh - sh / 2 - pad
            ix, iy = int(math.floor(x)), int(math.floor(y))
            m = np.array([[1.0, 0.0, x - ix], [0.0, 1.0, y - iy]], np.float32)  # sub-pixel shift
            size_p = (sprite.shape[1], sprite.shape[0])
            spr = cv2.warpAffine(sprite, m, size_p, flags=cv2.INTER_LINEAR)
            alpha = cv2.warpAffine(mask, m, size_p, flags=cv2.INTER_LINEAR)[..., None]
            fx0, fy0 = max(0, ix), max(0, iy)
            fx1, fy1 = min(w, ix + size_p[0]), min(h, iy + size_p[1])
            if fx1 <= fx0 or fy1 <= fy0:
                continue
            sx, sy = fx0 - ix, fy0 - iy
            a = alpha[sy:sy + fy1 - fy0, sx:sx + fx1 - fx0]
            roi = frames[i, fy0:fy1, fx0:fx1].astype(np.float32)
            roi = roi * (1.0 - a) + spr[sy:sy + fy1 - fy0, sx:sx + fx1 - fx0] * a
            frames[i, fy0:fy1, fx0:fx1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)

    def _canvas_for(self, shot: Shot, size: Size) -> np.ndarray:
        key = _group_key(shot)
        tw, th = int(round(size[0] * self.margin)), int(round(size[1] * self.margin))
        if key not in self._canvas or self._canvas[key].shape[:2] != (2 * th, 2 * tw):
            k = min(1.0, self.detail_long_side / max(tw, th))
            tile = synth.texture(max(8, int(round(th * k))), max(8, int(round(tw * k))), self.texture_seed(shot))
            if k < 1.0:
                tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_CUBIC)
            top = np.concatenate([tile, tile[:, ::-1]], axis=1)
            self._canvas[key] = np.ascontiguousarray(np.concatenate([top, top[::-1]], axis=0))
        return self._canvas[key]
