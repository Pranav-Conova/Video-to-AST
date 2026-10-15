"""Result contracts between analyzers and the AST builder.

Every analyzer is a pure function of the :class:`~editast.context.AnalysisContext`
(plus earlier results) and returns one of these dataclasses. Effects are
returned as ready-made AST effect nodes (``editast.ast.nodes``) with absolute
times in seconds; the builder (``editast.build``) only places them in the tree.

Pipeline order and signatures (see PLAN.md):

    seg    = shots.detect_shots(ctx)                          -> ShotSegmentation
    motion = motion.analyze_motion(ctx, seg)                  -> MotionAnalysis
    timefx = timefx.analyze_time_effects(ctx, seg, motion)    -> EffectsResult
    look   = look.analyze_look(ctx, seg)                      -> LookAnalysis
    text   = text.detect_text(ctx, seg)                       -> TextAnalysis
    audio  = audio.analyze_audio(ctx)                         -> AudioAnalysis
    edit   = build.build_edit(ctx, seg, motion, timefx, look, text, audio) -> Edit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from editast.ast.nodes import (
    EffectBase,
    Layout,
    Look,
    MotionSummary,
    Music,
    TextOverlay,
    TransitionKind,
)

# --------------------------------------------------------------------------- shots


@dataclass
class ShotSpan:
    """Frames ``[f0, f1)`` of one shot. Shots partition ``[0, N)`` contiguously."""

    index: int
    f0: int
    f1: int

    @property
    def n_frames(self) -> int:
        return self.f1 - self.f0


@dataclass
class TransitionSpan:
    """A boundary between shot ``from_shot`` and shot ``to_shot`` (0-based indices).

    ``boundary`` is the first frame of the incoming shot (== ``shots[to_shot].f0``).
    ``[f0, f1)`` are the frames visibly affected by the transition; for a hard
    cut ``f0 == f1 == boundary``. A fade-in at the start of the video has
    ``from_shot=None`` and a fade-out at the end ``to_shot=None``
    (their ``boundary`` is 0 / N respectively).
    """

    kind: TransitionKind
    boundary: int
    f0: int
    f1: int
    from_shot: Optional[int]
    to_shot: Optional[int]
    confidence: float = 1.0
    color: Optional[tuple[int, int, int]] = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShotSegmentation:
    shots: list[ShotSpan]
    transitions: list[TransitionSpan]  # one per boundary (incl. plain "cut"), plus fades at the ends
    boundary_score: np.ndarray  # (N,) per-frame boundary score used for detection (for reports)

    def shot_at(self, frame: int) -> int:
        """Index of the shot containing ``frame``."""
        for s in self.shots:
            if s.f0 <= frame < s.f1:
                return s.index
        return self.shots[-1].index if self.shots else 0

    def transition_frames(self) -> np.ndarray:
        """Boolean mask ``(N,)`` of frames covered by non-cut transitions."""
        n = self.shots[-1].f1 if self.shots else 0
        mask = np.zeros(n, bool)
        for t in self.transitions:
            if t.kind != "cut" and t.f1 > t.f0:
                mask[max(0, t.f0):min(n, t.f1)] = True
        return mask


# --------------------------------------------------------------------------- effects


@dataclass
class EffectsResult:
    """Effects keyed by shot index, plus named per-frame signals for reports/debug."""

    by_shot: dict[int, list[EffectBase]] = field(default_factory=dict)
    signals: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, shot_index: int, effect: EffectBase) -> None:
        self.by_shot.setdefault(shot_index, []).append(effect)


@dataclass
class FrameMotion:
    """Inter-frame similarity transform from frame ``t-1`` to ``t`` (index 0 = identity).

    ``scale`` is multiplicative (1 = none); ``rotation`` degrees (CCW positive);
    ``tx``/``ty`` content translation as a fraction of frame width / height;
    ``residual`` mean reprojection error (fraction of width) of the fit;
    ``valid`` whether the estimate is trustworthy (enough tracked points);
    ``magnitude`` mean optical-flow magnitude (fraction of width per frame).
    Values across a shot boundary (t = shot.f0) must be identity / invalid.
    """

    scale: np.ndarray
    rotation: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    residual: np.ndarray
    valid: np.ndarray
    magnitude: np.ndarray


@dataclass
class MotionAnalysis:
    frame_motion: FrameMotion
    effects: EffectsResult  # zoom / pan / rotate / shake
    summaries: dict[int, MotionSummary] = field(default_factory=dict)  # by shot index


@dataclass
class LookAnalysis:
    effects: EffectsResult  # flash (in-shot) / blur / invert / rgb_split
    looks: dict[int, Look] = field(default_factory=dict)  # by shot index
    layouts: dict[int, Layout] = field(default_factory=dict)  # by shot index


@dataclass
class TextAnalysis:
    overlays: list[TextOverlay] = field(default_factory=list)
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    ocr_engine: Optional[str] = None  # e.g. "rapidocr" or None when unavailable


@dataclass
class AudioAnalysis:
    music: Optional[Music]  # None when there is no (or silent) audio
    signals: dict[str, np.ndarray] = field(default_factory=dict)  # e.g. "onset_env", "rms"
    hop_seconds: float = 0.0  # time step of the arrays in ``signals``
