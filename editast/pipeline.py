"""End-to-end analysis: a video file or URL -> analyzers -> Edit AST.

:func:`analyze` runs the stages of PLAN.md in order::

    fetch (URLs only) -> decode (AnalysisContext) -> shots -> motion -> timefx
    -> look -> text -> audio -> build

Every analyzer module is imported lazily, by name, when its stage runs
(:data:`STAGES`), so a missing, half-written or crashing analyzer costs only
its own stage: the failure becomes a warning, the stage result is ``None``
and the remaining stages continue. The shot segmentation is repaired to a
valid contiguous partition (:func:`editast.build.repair_segmentation`)
before the other analyzers see it; if shot detection is unavailable the
whole video is treated as a single shot. Stages can be switched off with
``AnalysisConfig.enabled`` (e.g. ``{"text": False}``). Decoding errors and
an unreadable source are fatal.

:func:`signals_for_report` collects the per-frame / per-hop signals of a
result for the HTML report, and :func:`summarize_edits` aggregates many edits
(what do these viral edits have in common?).
"""

from __future__ import annotations

import importlib
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from editast.analyzers.types import (
    AudioAnalysis,
    EffectsResult,
    LookAnalysis,
    MotionAnalysis,
    ShotSegmentation,
    ShotSpan,
    TextAnalysis,
)
from editast.ast.nodes import Edit, EditStats
from editast.build import beats_phrase, build_edit, repair_segmentation
from editast.context import AnalysisConfig, AnalysisContext, ProgressFn, load_context
from editast.fetch import DOWNLOAD_DIR, download, is_url

logger = logging.getLogger(__name__)

# stage -> (module, function, expected result type), in pipeline order
STAGES: dict[str, tuple[str, str, type]] = {
    "shots": ("editast.analyzers.shots", "detect_shots", ShotSegmentation),
    "motion": ("editast.analyzers.motion", "analyze_motion", MotionAnalysis),
    "timefx": ("editast.analyzers.timefx", "analyze_time_effects", EffectsResult),
    "look": ("editast.analyzers.look", "analyze_look", LookAnalysis),
    "text": ("editast.analyzers.text", "detect_text", TextAnalysis),
    "audio": ("editast.analyzers.audio", "analyze_audio", AudioAnalysis),
}


@dataclass
class AnalysisResult:
    """The Edit plus everything it was derived from (for reports and debugging)."""

    edit: Edit
    ctx: AnalysisContext
    seg: ShotSegmentation
    motion: Optional[MotionAnalysis] = None
    timefx: Optional[EffectsResult] = None
    look: Optional[LookAnalysis] = None
    text: Optional[TextAnalysis] = None
    audio: Optional[AudioAnalysis] = None
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage, plus "total"
    warnings: list[str] = field(default_factory=list)


# =========================================================================== analysis


def resolve_source(source: Union[str, Path], progress: Optional[ProgressFn] = None,
                   download_dir: Union[str, Path] = DOWNLOAD_DIR) -> Path:
    """Local path of ``source``, downloading it first when it is a URL."""
    if is_url(str(source)):
        if progress is not None:
            progress(f"downloading {source}")
        return download(str(source), download_dir)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"no such video file: {path}")
    return path


def analyze(
    source: Union[str, Path],
    config: Optional[AnalysisConfig] = None,
    progress: Optional[ProgressFn] = None,
) -> AnalysisResult:
    """Analyse a video file or URL into an :class:`AnalysisResult` (see the module docstring)."""
    config = config or AnalysisConfig()
    say = progress or (lambda _msg: None)
    run = _Runner(config, say)
    t_start = time.perf_counter()

    if is_url(str(source)):
        with run.timed("fetch"):
            path = resolve_source(source, say)
    else:
        path = resolve_source(source, say)
    with run.timed("decode"):
        ctx = load_context(path, config, say)

    seg = run.stage("shots", ctx)
    if seg is not None:
        # Later analyzers rely on the segmentation contract; repair it once, here.
        try:
            seg, _, repairs = repair_segmentation(seg, ctx.n_frames)
            for note in repairs:
                run.warn(f"shots: {note}")
        except Exception as exc:
            run.status["shots"] = "invalid result"
            run.warn(f"shots: unusable segmentation ({type(exc).__name__}: {exc})")
            seg = None
    if seg is None:
        if run.status.get("shots") != "disabled":
            run.warn("shots: treating the whole video as a single shot")
        seg = single_shot(ctx.n_frames)
    motion = run.stage("motion", ctx, seg)
    timefx = run.stage("timefx", ctx, seg, motion)
    look = run.stage("look", ctx, seg)
    text = run.stage("text", ctx, seg)
    if text is not None:
        run.status["text"] = f"ok (ocr: {text.ocr_engine})" if text.ocr_engine else "ok (no ocr)"
    if ctx.audio.size == 0 and config.is_enabled("audio"):
        run.status["audio"] = "no audio track"
        audio = None
    else:
        audio = run.stage("audio", ctx)

    say("building the edit tree")
    with run.timed("build"):
        edit = _build(run, ctx, seg, motion, timefx, look, text, audio)
    run.timings["total"] = time.perf_counter() - t_start
    edit.provenance["timings"] = {k: round(v, 4) for k, v in run.timings.items()}
    return AnalysisResult(edit=edit, ctx=ctx, seg=seg, motion=motion, timefx=timefx, look=look,
                          text=text, audio=audio, timings=dict(run.timings), warnings=list(run.warnings))


def single_shot(n_frames: int) -> ShotSegmentation:
    """Fallback segmentation: one shot covering every frame."""
    return ShotSegmentation(shots=[ShotSpan(0, 0, n_frames)], transitions=[], boundary_score=np.zeros(n_frames))


def _build(run: "_Runner", ctx: AnalysisContext, seg: ShotSegmentation, *results: Any) -> Edit:
    """Build the Edit; if a malformed analyzer result breaks it, retry from the shots alone."""
    try:
        return build_edit(ctx, seg, *results, analyzers=run.status, warnings=run.warnings, timings=run.timings)
    except Exception as exc:
        logger.debug("build failed", exc_info=True)
        run.warn(f"build: failed with all analyzer results ({type(exc).__name__}: {exc}); "
                 "keeping only the shot structure")
        for name in STAGES:
            if name != "shots" and run.status.get(name, "").startswith("ok"):
                run.status[name] = "dropped (build failed)"
        return build_edit(ctx, seg, analyzers=run.status, warnings=run.warnings, timings=run.timings)


class _Runner:
    """Runs stages in isolation and records status, timings and warnings."""

    def __init__(self, config: AnalysisConfig, say: Callable[[str], None]) -> None:
        self.config = config
        self.say = say
        self.status: dict[str, str] = {}
        self.timings: dict[str, float] = {}
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        logger.info(message)  # returned to the caller in AnalysisResult.warnings
        self.warnings.append(message)

    def timed(self, name: str) -> "_Timer":
        return _Timer(self.timings, name)

    def stage(self, name: str, *args: Any) -> Any:
        if not self.config.is_enabled(name):
            self.status[name] = "disabled"
            return None
        module_name, func_name, expected = STAGES[name]
        try:
            func = getattr(importlib.import_module(module_name), func_name)
        except Exception as exc:  # missing module, syntax error, missing function ...
            self.status[name] = "unavailable"
            self.warn(f"{name}: analyzer unavailable ({type(exc).__name__}: {exc}); continuing without it")
            return None
        self.say(f"{name}: running")
        try:
            with self.timed(name):
                result = func(*args)
        except Exception as exc:
            logger.debug("stage %s failed", name, exc_info=True)
            self.status[name] = "failed"
            self.warn(f"{name}: analyzer failed ({type(exc).__name__}: {exc}); continuing without it")
            return None
        if not isinstance(result, expected):
            self.status[name] = "invalid result"
            self.warn(f"{name}: analyzer returned {type(result).__name__}, expected {expected.__name__}; ignoring it")
            return None
        self.status[name] = "ok"
        return result


class _Timer:
    def __init__(self, sink: dict[str, float], name: str) -> None:
        self.sink, self.name = sink, name

    def __enter__(self) -> "_Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.sink[self.name] = self.sink.get(self.name, 0.0) + time.perf_counter() - self.t0


# =========================================================================== report signals


def signals_for_report(result: AnalysisResult) -> dict[str, tuple[np.ndarray, float]]:
    """Named 1-D signals ``name -> (values, seconds per sample)`` for plots in the HTML report.

    Video signals are per analysis frame (``dt = 1 / fps``); audio signals use
    the audio analyzer's hop. Signals of stages that did not run are omitted.
    """
    ctx = result.ctx
    frame_dt = 1.0 / ctx.fps
    out: dict[str, tuple[np.ndarray, float]] = {}

    def put(name: str, values: Any, dt: float) -> None:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1 and arr.size > 1 and dt > 0 and np.isfinite(arr).any():
            out[name] = (np.nan_to_num(arr), float(dt))

    put("boundary_score", result.seg.boundary_score, frame_dt)
    put("frame_diff", ctx.features.diff, frame_dt)
    put("luma", ctx.features.luma_mean, frame_dt)
    put("saturation", ctx.features.sat_mean, frame_dt)
    if result.motion is not None:
        put("motion_magnitude", result.motion.frame_motion.magnitude, frame_dt)
        for key in ("zoom", "shake"):
            if key in result.motion.effects.signals:
                put(key, result.motion.effects.signals[key], frame_dt)
    if result.timefx is not None:
        for key in ("speed_rate", "frame_repeat"):
            if key in result.timefx.signals:
                put(key, result.timefx.signals[key], frame_dt)
    if result.look is not None:
        for key in ("flash_amount", "sharpness_rel"):
            if key in result.look.effects.signals:
                put(key, result.look.effects.signals[key], frame_dt)
    if result.text is not None and "text_present" in result.text.signals:
        put("text_present", result.text.signals["text_present"], frame_dt)
    if result.audio is not None and result.audio.hop_seconds > 0:
        for key, name in (("onset_env", "onset_envelope"), ("energy", "audio_energy")):
            if key in result.audio.signals:
                put(name, result.audio.signals[key], result.audio.hop_seconds)
    return out


# =========================================================================== batch summary

_STAT_FIELDS = ("n_shots", "mean_shot_duration", "median_shot_duration", "cuts_per_minute", "cut_on_beat_ratio",
                "hook_duration", "first_cut_time", "text_coverage", "n_transitions")
_PHRASES = {
    "cut_on_beat": "cut on the beat",
    "cut_every_n_beats": "cut {beats} in at least one section",
    "pacing_accelerates_at_drop": "cut faster once the drop hits",
    "flash_on_cut": "punctuate cuts with flashes",
    "hook_text": "open with on-screen hook text in the first second",
    "speed_ramp_on_beat": "land speed ramps on beats",
    "recurring_source": "cut back to the same clip more than once",
}


def _pattern_key(rule: str, params: dict[str, Any]) -> str:
    return f"{rule}[n={params['n']:g}]" if rule == "cut_every_n_beats" and "n" in params else rule


def _phrase(key: str) -> str:
    rule, _, variant = key.partition("[")
    if rule == "cut_every_n_beats" and variant.startswith("n="):
        return _PHRASES[rule].format(beats=beats_phrase(float(variant[2:-1])))
    if rule.startswith("effect_on_beat:"):
        return f"put {rule.split(':', 1)[1].replace('_', ' ')} effects on beats"
    if rule.startswith("effect_on_drop:"):
        what = rule.split(":", 1)[1].replace("_", " ")
        return f"hit {'an' if what[:1] in 'aeiou' else 'a'} {what} on the drop"
    return _PHRASES.get(rule, rule.replace("_", " "))


def _describe(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    return {"mean": round(statistics.fmean(values), 4), "median": round(statistics.median(values), 4),
            "min": round(min(values), 4), "max": round(max(values), 4), "n": len(values)}


def summarize_edits(edits: Sequence[tuple[str, Edit]], common_fraction: float = 0.5) -> dict[str, Any]:
    """Aggregate stats, structures and pattern frequencies over many analysed edits.

    ``edits`` are ``(source name, edit)`` pairs. ``common`` lists the traits
    shared by at least ``common_fraction`` of the videos as plain sentences.
    """
    n = len(edits)
    videos: list[dict[str, Any]] = []
    pattern_hits: dict[str, list[float]] = {}
    effect_videos: dict[str, list[int]] = {}
    transition_videos: dict[str, list[int]] = {}
    structures: dict[str, int] = {}
    for name, edit in edits:
        st = edit.stats or EditStats()
        roles = " > ".join(sec.role for sec in edit.timeline.sections)
        structures[roles] = structures.get(roles, 0) + 1
        best: dict[str, float] = {}
        for pat in edit.patterns:
            key = _pattern_key(pat.rule, pat.params)
            best[key] = max(best.get(key, 0.0), pat.support)
        for key, support in best.items():
            pattern_hits.setdefault(key, []).append(support)
        for kind, count in st.effect_counts.items():
            effect_videos.setdefault(kind, []).append(count)
        for kind, count in st.transition_counts.items():
            if kind != "cut":
                transition_videos.setdefault(kind, []).append(count)
        row = {"source": name, "duration": round(edit.media.duration, 3),
               "bpm": round(edit.music.bpm, 2) if edit.music is not None and edit.music.bpm else None,
               "structure": roles, "patterns": sorted(best)}
        row.update({f: getattr(st, f) for f in _STAT_FIELDS})
        videos.append(row)

    stats = {f: _describe([v[f] for v in videos if v[f] is not None]) for f in ("duration", "bpm", *_STAT_FIELDS)}
    patterns = {k: {"videos": len(v), "fraction": round(len(v) / n, 4), "mean_support": round(statistics.fmean(v), 4),
                    "phrase": _phrase(k)}
                for k, v in sorted(pattern_hits.items(), key=lambda kv: (-len(kv[1]), kv[0]))}

    def usage(table: dict[str, list[int]]) -> dict[str, dict[str, float]]:
        return {k: {"videos": len(v), "fraction": round(len(v) / n, 4), "mean_per_video": round(sum(v) / n, 4)}
                for k, v in sorted(table.items(), key=lambda kv: (-len(kv[1]), kv[0]))}

    common: list[str] = []
    if n:
        for key, info in patterns.items():
            if info["fraction"] >= common_fraction:
                common.append(f"{info['videos']}/{n} videos {info['phrase']}.")
        for kind, info in usage(effect_videos).items():
            if info["fraction"] >= common_fraction:
                common.append(f"{info['videos']}/{n} videos use {kind.replace('_', ' ')} effects "
                              f"({info['mean_per_video']:.1f} per video on average).")
        for kind, info in usage(transition_videos).items():
            if info["fraction"] >= common_fraction:
                common.append(f"{info['videos']}/{n} videos use {kind.replace('_', ' ')} transitions.")
        cpm = stats["cuts_per_minute"]
        msd = stats["mean_shot_duration"]
        if cpm and msd:
            common.append(f"Median pace: {cpm['median']:.0f} cuts per minute "
                          f"(mean shot length {msd['median']:.2f} s).")
        if structures:
            top, count = max(structures.items(), key=lambda kv: kv[1])
            if count / n >= common_fraction:
                common.append(f"{count}/{n} videos follow the structure {top}.")
    return {
        "n_videos": n,
        "stats": stats,
        "patterns": patterns,
        "effects": usage(effect_videos),
        "transitions": usage(transition_videos),
        "structures": dict(sorted(structures.items(), key=lambda kv: -kv[1])),
        "common": common,
        "videos": videos,
    }
