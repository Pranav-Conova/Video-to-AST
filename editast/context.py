"""Analysis context shared by every analyzer.

The video is decoded **once** at a reduced, constant-rate "analysis" resolution
(default long side 320 px). Frame ``i`` is shown at ``i / ctx.fps`` seconds.
Cheap per-frame signals that several analyzers need are computed once in
:class:`FrameFeatures`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from editast.ast.nodes import MediaInfo
from editast.media import aspect_string, fit_size, probe, read_audio, read_video

ProgressFn = Callable[[str], None]


@dataclass
class AnalysisConfig:
    analysis_long_side: int = 320  # px, long side of analysis frames
    target_fps: Optional[float] = None  # None = source fps, capped at max_fps
    max_fps: float = 60.0
    max_duration: float = 300.0  # seconds analysed from the start
    audio_sr: int = 22050
    beat_tolerance: float = 0.07  # seconds; an event this close to a beat is "on beat"
    ocr: bool = True  # use RapidOCR for overlay text when installed
    text_fps: float = 3.0  # sampling rate for text detection
    text_long_side: int = 720  # resolution used for text detection / OCR
    # Analyzer switches: shots, motion, timefx, look, text, audio
    enabled: dict[str, bool] = field(default_factory=dict)
    # Per-analyzer parameter overrides, e.g. {"shots": {"threshold": 0.4}}
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def is_enabled(self, name: str) -> bool:
        return self.enabled.get(name, True)

    def p(self, analyzer: str, key: str, default: Any) -> Any:
        """Parameter lookup with a default: ``cfg.p("shots", "threshold", 0.35)``."""
        return self.params.get(analyzer, {}).get(key, default)


@dataclass
class FrameFeatures:
    """Per-frame signals, all arrays of length ``N`` (index = analysis frame)."""

    luma_mean: np.ndarray  # 0..1
    luma_std: np.ndarray  # 0..1
    sat_mean: np.ndarray  # 0..1 (HSV saturation)
    rgb_mean: np.ndarray  # (N, 3) 0..1
    hist: np.ndarray  # (N, 128) joint HSV histogram (8x4x4), each row sums to 1
    diff: np.ndarray  # mean |gray[t] - gray[t-1]| / 255; diff[0] = 0
    hist_dist: np.ndarray  # 1 - histogram intersection with previous frame; [0] = 0
    sharpness: np.ndarray  # variance of the Laplacian of gray (raw, scale-dependent)
    edge_density: np.ndarray  # fraction of Canny edge pixels


@dataclass
class AnalysisContext:
    path: str
    info: MediaInfo
    fps: float  # analysis frame rate
    frames: np.ndarray  # (N, H, W, 3) uint8 RGB
    gray: np.ndarray  # (N, H, W) uint8
    audio: np.ndarray  # (S,) float32 mono, may be empty
    sr: int
    features: FrameFeatures
    config: AnalysisConfig

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        """Analysis ``(width, height)``."""
        return int(self.frames.shape[2]), int(self.frames.shape[1])

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    def t(self, frame: float) -> float:
        """Frame index -> seconds."""
        return float(frame) / self.fps

    def f(self, seconds: float) -> int:
        """Seconds -> nearest frame index (clamped)."""
        return int(np.clip(round(seconds * self.fps), 0, max(0, self.n_frames - 1)))


def to_gray(frames: np.ndarray) -> np.ndarray:
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])


def compute_features(frames: np.ndarray, gray: Optional[np.ndarray] = None) -> FrameFeatures:
    if gray is None:
        gray = to_gray(frames)
    n = frames.shape[0]
    luma_mean = gray.reshape(n, -1).mean(axis=1) / 255.0
    luma_std = gray.reshape(n, -1).std(axis=1) / 255.0
    rgb_mean = frames.reshape(n, -1, 3).mean(axis=1) / 255.0
    sat_mean = np.empty(n)
    hist = np.empty((n, 128))
    sharp = np.empty(n)
    edges = np.empty(n)
    for i in range(n):
        hsv = cv2.cvtColor(frames[i], cv2.COLOR_RGB2HSV)
        sat_mean[i] = hsv[..., 1].mean() / 255.0
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        hist[i] = h / max(h.sum(), 1.0)
        sharp[i] = cv2.Laplacian(gray[i], cv2.CV_64F).var()
        edges[i] = (cv2.Canny(gray[i], 100, 200) > 0).mean()
    diff = np.zeros(n)
    hist_dist = np.zeros(n)
    if n > 1:
        g = gray.astype(np.int16)
        diff[1:] = np.abs(g[1:] - g[:-1]).reshape(n - 1, -1).mean(axis=1) / 255.0
        hist_dist[1:] = 1.0 - np.minimum(hist[1:], hist[:-1]).sum(axis=1)
    return FrameFeatures(
        luma_mean=luma_mean, luma_std=luma_std, sat_mean=sat_mean, rgb_mean=rgb_mean,
        hist=hist, diff=diff, hist_dist=np.clip(hist_dist, 0.0, 1.0),
        sharpness=sharp, edge_density=edges,
    )


def context_from_arrays(
    frames: np.ndarray,
    fps: float,
    audio: Optional[np.ndarray] = None,
    sr: int = 22050,
    config: Optional[AnalysisConfig] = None,
    source: str = "<memory>",
) -> AnalysisContext:
    """Build a context from in-memory frames (used by tests and the benchmark)."""
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    gray = to_gray(frames)
    audio = np.zeros(0, np.float32) if audio is None else np.asarray(audio, np.float32)
    n, h, w = frames.shape[:3]
    info = MediaInfo(
        source=source, duration=n / fps, fps=fps, width=w, height=h, n_frames=n,
        analysis_fps=fps, has_audio=audio.size > 0, aspect=aspect_string(w, h),
    )
    return AnalysisContext(
        path=source, info=info, fps=fps, frames=frames, gray=gray, audio=audio, sr=sr,
        features=compute_features(frames, gray), config=config or AnalysisConfig(),
    )


def load_context(
    path: str | Path,
    config: Optional[AnalysisConfig] = None,
    progress: Optional[ProgressFn] = None,
) -> AnalysisContext:
    """Probe + decode ``path`` into an :class:`AnalysisContext`."""
    config = config or AnalysisConfig()
    say = progress or (lambda _msg: None)
    path = str(path)
    p = probe(path)
    fps = config.target_fps or min(p.fps, config.max_fps)
    size = fit_size(p.width, p.height, config.analysis_long_side)
    limit = config.max_duration if (p.duration == 0 or p.duration > config.max_duration) else None
    say(f"decoding video at {size[0]}x{size[1]} @ {fps:.3g} fps")
    frames = read_video(path, fps=fps, size=size, duration=limit)
    gray = to_gray(frames)
    audio = np.zeros(0, np.float32)
    if p.has_audio:
        say("decoding audio")
        audio = read_audio(path, sr=config.audio_sr)
        if limit is not None:
            audio = audio[: int(limit * config.audio_sr)]
    say("computing frame features")
    features = compute_features(frames, gray)
    n = frames.shape[0]
    info = MediaInfo(
        source=path, duration=n / fps, fps=p.fps, width=p.width, height=p.height, n_frames=n,
        analysis_fps=fps, has_audio=audio.size > 0, aspect=aspect_string(p.width, p.height),
    )
    return AnalysisContext(
        path=path, info=info, fps=fps, frames=frames, gray=gray, audio=audio,
        sr=config.audio_sr, features=features, config=config,
    )
