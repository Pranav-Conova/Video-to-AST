"""Structure synthesis: assemble analyzer results into an :class:`~editast.ast.Edit`.

The analyzers each describe one aspect of the video (shot boundaries, camera
motion, time effects, look, text, music). :func:`build_edit` places those
fragments in one tree and derives the *editing structure* that no single
analyzer sees:

1. **Shots** — :func:`repair_segmentation` turns ``seg`` into a contiguous
   partition of the analysis frames (sorted, gaps and overlaps closed, an
   empty segmentation becomes a single shot). Shot ``i`` gets
   ``id = shot_id(i)``.
2. **Transitions** — exactly one per shot boundary (the most confident;
   missing boundaries get a plain ``cut``), plus an optional fade-in
   (``from_shot=None``) and fade-out (``to_shot=None``). A hard cut has
   ``start == end == boundary``; other kinds keep their visible span, widened
   to contain the boundary.
3. **Effects** — motion, time and look effects are copied onto their shot,
   clamped to the shot span (an effect keyed to a shot it does not overlap
   moves to the shot it overlaps most). Degenerate spans are dropped,
   overlapping effects of one type are de-duplicated (higher confidence wins;
   a marginal overlap is trimmed instead), and in-shot flashes that merely
   repeat a flash/fade transition are removed.
4. **Beat alignment** — against the music beat grid (gaps filled with the beat
   period, extended one beat past either end), every shot start and transition
   boundary gets its signed offset to the nearest beat; ``on_beat`` means
   ``|offset| <= config.beat_tolerance``; ``Shot.beats`` = duration / period.
5. **Source groups** — shots are clustered by their mean colour histogram
   (``ctx.features.hist`` outside transition and pixel-effect frames;
   histogram intersection, average-linkage agglomeration down to
   ``group_threshold``). Labels ``A``, ``B``, ... follow first appearance.
   Colour only: a regraded (e.g. black & white) reuse of a clip is not found.
6. **Sections** — always whole shots. The *hook* runs to the first shot
   boundary at or after ``hook_min`` (if that is within ``hook_max``), extended
   to cover an opening text overlay. With a music drop: *build* up to the drop
   (snapped to the nearest shot boundary, so a shot straddling the drop goes to
   the side holding most of it), *drop* to the end of the high-energy audio
   sections that follow it, an optional *break* before a later drop, and an
   *outro* for a trailing low-energy part of at least ``outro_min``. Without a
   drop the rest is split recursively where the mean shot duration changes by
   more than ``pacing_ratio`` (binary segmentation); a slow part followed by a
   fast one becomes build -> drop, everything else body.
7. **Patterns** — editing rules that are actually supported by the data
   (cut on beat, cut every N beats, faster cutting at the drop, effects on
   beats / on the drop, flash cuts, hook text, speed ramps on beats, recurring
   source clips), each with a one-sentence description for creators.
8. **Stats and provenance** — pacing curve (cuts per second in 1 s windows
   every 0.5 s), shot-length statistics, effect / transition counts, text
   coverage; the provenance records the tool version, analyzer status,
   warnings, timings and the analysis configuration.

Tunable parameters are read with ``ctx.config.p("build", <name>, default)``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union, get_args

import numpy as np
from scipy import ndimage

import editast
from editast.analyzers.types import (
    AudioAnalysis,
    EffectsResult,
    LookAnalysis,
    MotionAnalysis,
    ShotSegmentation,
    ShotSpan,
    TextAnalysis,
    TransitionSpan,
)
from editast.ast.nodes import (
    EFFECT_TYPES,
    Curve,
    Edit,
    EditStats,
    EffectBase,
    Music,
    Pattern,
    Section,
    Shot,
    TextOverlay,
    Timeline,
    Transition,
    TransitionKind,
    assign_ids,
    shot_id,
)
from editast.context import AnalysisConfig, AnalysisContext

NAME = "build"
BEAT_QUANTA = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)  # shot lengths (in beats) a pattern can name
_FLASHY_TRANSITIONS = ("flash", "fade_white", "fade_black")
_PIXEL_EFFECTS = ("flash", "invert", "rgb_split", "blur")
_TRANSITION_KINDS = frozenset(get_args(TransitionKind))
_EPS = 1e-6

TimelineItem = Union[Shot, Transition]


@dataclass(frozen=True)
class _Settings:
    """Thresholds of the builder (``config.params["build"]`` overrides the defaults)."""

    beat_tolerance: float
    hook_min: float  # s, shortest hook
    hook_max: float  # s, longest hook
    hook_text_window: float  # s, an overlay starting this early is "opening text"
    hook_text_tolerance: float  # s, a cut this much before the text ends still covers it
    drop_snap: float  # s, an audio drop section starting this close to the drop is "the drop"
    drop_keep: float  # energy ratio that keeps a section part of the drop
    outro_min: float  # s, shortest outro
    pacing_ratio: float  # mean shot duration change that splits sections
    min_section_shots: int
    min_section_duration: float  # s
    group_threshold: float  # histogram intersection that joins source groups
    dedupe_overlap: float  # overlap / shorter span above which same-type effects are duplicates
    min_support: float  # pattern support threshold
    beat_quant_tolerance: float  # relative tolerance of "n beats"
    accel_ratio: float  # cuts/s after vs before the drop
    drop_window: float  # s, "on the drop"
    flash_cut_window: float  # s, an in-shot flash this close to a cut is a flash cut
    hook_text_time: float  # s, the hook_text pattern needs text starting this early
    min_pattern_count: int  # instances a frequency pattern needs

    @classmethod
    def from_config(cls, cfg: AnalysisConfig) -> "_Settings":
        p = functools.partial(cfg.p, NAME)
        return cls(
            beat_tolerance=float(cfg.beat_tolerance),
            hook_min=float(p("hook_min", 1.0)),
            hook_max=float(p("hook_max", 3.5)),
            hook_text_window=float(p("hook_text_window", 1.0)),
            hook_text_tolerance=float(p("hook_text_tolerance", 0.15)),
            drop_snap=float(p("drop_snap", 0.35)),
            drop_keep=float(p("drop_keep", 0.7)),
            outro_min=float(p("outro_min", 1.0)),
            pacing_ratio=float(p("pacing_ratio", 1.8)),
            min_section_shots=int(p("min_section_shots", 3)),
            min_section_duration=float(p("min_section_duration", 2.0)),
            group_threshold=float(p("group_threshold", 0.7)),
            dedupe_overlap=float(p("dedupe_overlap", 0.2)),
            min_support=float(p("min_support", 0.6)),
            beat_quant_tolerance=float(p("beat_quant_tolerance", 0.15)),
            accel_ratio=float(p("accel_ratio", 1.5)),
            drop_window=float(p("drop_window", 0.15)),
            flash_cut_window=float(p("flash_cut_window", 0.1)),
            hook_text_time=float(p("hook_text_time", 1.0)),
            min_pattern_count=int(p("min_pattern_count", 2)),
        )


# =========================================================================== public API


def build_edit(
    ctx: AnalysisContext,
    seg: ShotSegmentation,
    motion: Optional[MotionAnalysis] = None,
    timefx: Optional[EffectsResult] = None,
    look: Optional[LookAnalysis] = None,
    text: Optional[TextAnalysis] = None,
    audio: Optional[AudioAnalysis] = None,
    *,
    analyzers: Optional[Mapping[str, str]] = None,
    warnings: Sequence[str] = (),
    timings: Optional[Mapping[str, float]] = None,
) -> Edit:
    """Assemble the analyzer results into an Edit AST (see the module docstring).

    Any result except ``seg`` may be ``None``. ``analyzers`` (stage -> status),
    ``warnings`` and ``timings`` are recorded in ``edit.provenance``; the
    pipeline passes them, direct callers may omit them. Analyzer results are
    never modified (effects and overlays are copied).
    """
    n = ctx.n_frames
    if n <= 0:
        raise ValueError("cannot build an edit from a video without frames")
    cfg = _Settings.from_config(ctx.config)
    notes = list(warnings)
    total = ctx.t(n)

    clean, index_map, repairs = repair_segmentation(seg, n)
    notes += [f"build: {r}" for r in repairs]
    shots = [Shot(index=s.index, frame_start=s.f0, frame_end=s.f1, start=ctx.t(s.f0), end=ctx.t(s.f1))
             for s in clean.shots]
    fade_in, cuts, fade_out = _transitions(ctx, clean)
    boundary_transitions = [t for t in (fade_in, *cuts.values(), fade_out) if t is not None]

    _attach_attributes(shots, index_map, motion, look)
    effect_sources = [motion.effects if motion else None, timefx, look.effects if look else None]
    _attach_effects(ctx, shots, index_map, [r for r in effect_sources if r is not None],
                    boundary_transitions, cfg, notes)

    music = audio.music.model_copy(deep=True) if audio is not None and audio.music is not None else None
    grid = _BeatGrid.from_music(music, cfg.beat_tolerance)
    if grid is not None:
        _align_to_beats(grid, shots, fade_in, cuts, fade_out, total)
    _group_sources(ctx, clean, shots, cfg)

    overlays = _overlays(text, total)
    sections = _sections(shots, fade_in, cuts, fade_out, music, overlays, total, cfg)
    edit = Edit(
        media=ctx.info.model_copy(deep=True),
        music=music,
        timeline=Timeline(sections=sections),
        overlays=overlays,
    )
    edit.media.duration = total
    edit.media.n_frames = n
    edit.media.analysis_fps = ctx.fps
    edit.patterns = _patterns(edit, grid, cfg)
    edit.stats = _stats(edit, total)
    edit.provenance = {
        "tool": "editast",
        "version": editast.__version__,
        "analyzers": dict(analyzers) if analyzers is not None else _default_status(
            motion=motion, timefx=timefx, look=look, text=text, audio=audio),
        "warnings": notes,
        "timings": {k: round(float(v), 4) for k, v in (timings or {}).items()},
        "config": _json_safe(dataclasses.asdict(ctx.config)),
    }
    return assign_ids(edit)


def rebeat(edit: Edit, config: Optional[AnalysisConfig] = None) -> Edit:
    """Re-derive what depends on the music after ``edit.music`` was replaced (in place).

    Beat alignment (step 4), section energies, patterns and stats are computed
    again against the new music; shots, transitions, effects and section roles
    are kept. :mod:`editast.fit` uses it after moving an edit onto another song.
    """
    cfg = _Settings.from_config(config or AnalysisConfig())
    shots = edit.shots()
    total = edit.media.duration
    index = {s.id: i for i, s in enumerate(shots)}
    fade_in = fade_out = None
    cuts: dict[int, Transition] = {}
    for tr in edit.transitions():
        if tr.from_shot is None:
            fade_in = tr
        elif tr.to_shot is None:
            fade_out = tr
        elif tr.to_shot in index:
            cuts[index[tr.to_shot]] = tr
    grid = _BeatGrid.from_music(edit.music, cfg.beat_tolerance)
    if grid is not None:
        _align_to_beats(grid, shots, fade_in, cuts, fade_out, total)
    else:
        for node in [*shots, *edit.transitions()]:
            node.beat_offset = node.on_beat = None
        for shot in shots:
            shot.beats = None
    for sec in edit.timeline.sections:
        sec.energy = _mean_energy(edit.music, sec.start, sec.end)
    edit.patterns = _patterns(edit, grid, cfg)
    edit.stats = _stats(edit, total)
    return assign_ids(edit)


# =========================================================================== shots & transitions


def repair_segmentation(seg: ShotSegmentation, n_frames: int) -> tuple[ShotSegmentation, dict[int, int], list[str]]:
    """A copy of ``seg`` that satisfies the :class:`ShotSegmentation` contract exactly.

    Shots become a contiguous partition of ``[0, n_frames)`` (sorted, gaps and
    overlaps closed, duplicate starts merged, nothing -> one shot); every shot
    boundary gets exactly one transition (the most confident one, or a plain
    cut when none was reported); fades are kept only at the very start / end;
    transition spans are clamped and contain their boundary, and
    ``boundary_score`` has length ``n_frames``.

    Returns ``(clean, index_map, notes)``: ``index_map`` maps the original shot
    indices to the new ones (to re-key results computed on ``seg``) and
    ``notes`` says what had to be repaired (empty for a valid segmentation).
    """
    n = n_frames
    notes: list[str] = []
    starts: list[int] = []
    index_map: dict[int, int] = {}
    for span in sorted(seg.shots, key=lambda s: (s.f0, s.f1)):
        f0 = int(np.clip(span.f0, 0, n))
        if f0 >= n:
            continue
        if starts and f0 <= starts[-1]:
            index_map.setdefault(span.index, len(starts) - 1)  # duplicate start: same shot
            continue
        index_map.setdefault(span.index, len(starts))
        starts.append(f0)
    if not starts:
        notes.append("the shot segmentation was empty; using a single shot")
        starts = [0]
        index_map = {s.index: 0 for s in seg.shots}
    starts[0] = 0
    shots = [ShotSpan(i, f0, starts[i + 1] if i + 1 < len(starts) else n) for i, f0 in enumerate(starts)]
    if seg.shots and sorted((s.f0, s.f1) for s in seg.shots) != [(s.f0, s.f1) for s in shots]:
        notes.append(f"repaired the shot spans ({len(seg.shots)} spans -> {len(shots)} contiguous shots)")

    k = len(shots)
    first = np.array([s.f0 for s in shots])
    best: dict[Union[int, str], TransitionSpan] = {}
    dropped = 0
    for ts in seg.transitions:
        to = index_map.get(ts.to_shot) if ts.to_shot is not None else None
        if ts.to_shot is not None and to is None:  # unknown shot index: use the nearest boundary
            to = int(np.argmin(np.abs(first - ts.boundary)))
        key: Union[int, str]
        if ts.to_shot is None and ts.from_shot is None:
            key = ""
        elif to is None:
            key, frm, boundary = "out", k - 1, n
        elif to == 0:
            key, frm, boundary = ("in" if ts.from_shot is None else ""), None, 0
        else:
            key, frm, boundary = to, to - 1, shots[to].f0
        kind = ts.kind if ts.kind in _TRANSITION_KINDS else "unknown"
        if not key or (kind == "cut" and key in ("in", "out")):
            dropped += 1  # not attached to a boundary, or a "cut" from / to nothing
            continue
        if kind == "cut":
            f0 = f1 = boundary
        else:
            lo, hi = sorted((int(np.clip(ts.f0, 0, n)), int(np.clip(ts.f1, 0, n))))
            f0, f1 = min(lo, boundary), max(hi, boundary)
        fixed = dataclasses.replace(ts, kind=kind, boundary=boundary, f0=f0, f1=f1, from_shot=frm,  # type: ignore[arg-type]
                                    to_shot=None if key == "out" else to)
        if key in best:
            dropped += 1
            if fixed.confidence <= best[key].confidence:
                continue
        best[key] = fixed
    missing = [i for i in range(1, k) if i not in best]
    for i in missing:
        b = shots[i].f0
        best[i] = TransitionSpan("cut", b, b, b, i - 1, i)
    if missing and seg.transitions:
        notes.append(f"added {len(missing)} missing cut transition(s)")
    if dropped:
        notes.append(f"dropped {dropped} duplicate or unattached transition(s)")
    ordered = [best["in"]] if "in" in best else []
    ordered += [best[i] for i in range(1, k)]
    ordered += [best["out"]] if "out" in best else []

    score = np.zeros(n)
    raw = np.nan_to_num(np.asarray(seg.boundary_score, float).ravel())
    score[:min(n, raw.size)] = raw[:n]
    return ShotSegmentation(shots, ordered, score), index_map, notes


def _transitions(
    ctx: AnalysisContext, seg: ShotSegmentation,
) -> tuple[Optional[Transition], dict[int, Transition], Optional[Transition]]:
    """``(fade_in, {to_shot index: transition}, fade_out)`` from a repaired segmentation."""
    fade_in: Optional[Transition] = None
    fade_out: Optional[Transition] = None
    cuts: dict[int, Transition] = {}
    for ts in seg.transitions:
        tr = Transition(
            kind=ts.kind, start=ctx.t(ts.f0), end=ctx.t(ts.f1),
            from_shot=None if ts.from_shot is None else shot_id(ts.from_shot),
            to_shot=None if ts.to_shot is None else shot_id(ts.to_shot),
            color=tuple(int(c) for c in ts.color) if ts.color is not None else None,  # type: ignore[arg-type]
            confidence=float(ts.confidence), params=dict(ts.params),
        )
        if ts.from_shot is None:
            fade_in = tr
        elif ts.to_shot is None:
            fade_out = tr
        else:
            cuts[ts.to_shot] = tr
    return fade_in, cuts, fade_out


def _attach_attributes(
    shots: list[Shot],
    index_map: dict[int, int],
    motion: Optional[MotionAnalysis],
    look: Optional[LookAnalysis],
) -> None:
    """Per-shot look / layout / motion summary (the first value wins for merged shots)."""
    tables: list[tuple[str, Mapping[int, Any]]] = []
    if look is not None:
        tables += [("look", look.looks), ("layout", look.layouts)]
    if motion is not None:
        tables.append(("motion", motion.summaries))
    for attr, table in tables:
        for key, value in sorted(table.items()):
            i = index_map.get(key)
            if i is not None and value is not None and getattr(shots[i], attr) is None:
                setattr(shots[i], attr, value.model_copy(deep=True))


def _attach_effects(
    ctx: AnalysisContext,
    shots: list[Shot],
    index_map: dict[int, int],
    results: Sequence[EffectsResult],
    transitions: Sequence[Transition],
    cfg: _Settings,
    notes: list[str],
) -> None:
    starts = np.array([s.start for s in shots])
    ends = np.array([s.end for s in shots])
    min_len = 0.5 / ctx.fps
    pending: list[list[EffectBase]] = [[] for _ in shots]
    unknown: set[str] = set()
    for result in results:
        for key, effects in result.by_shot.items():
            keyed = index_map.get(key)
            for eff in effects:
                kind = getattr(eff, "type", None)
                if kind not in EFFECT_TYPES:
                    unknown.add(str(kind or type(eff).__name__))
                    continue
                overlap = np.minimum(ends, eff.end) - np.maximum(starts, eff.start)
                host = keyed if keyed is not None and overlap[keyed] > 0 else int(np.argmax(overlap))
                if overlap[host] <= 0:
                    continue
                a, b = max(eff.start, shots[host].start), min(eff.end, shots[host].end)
                if b - a < min_len:
                    continue
                copy = eff.model_copy(deep=True)
                copy.start, copy.end = a, b
                copy.keyframes = [kf for kf in copy.keyframes if a - min_len <= kf.t <= b + min_len]
                pending[host].append(copy)
    if unknown:
        notes.append(f"build: ignored effects of unknown type: {', '.join(sorted(unknown))}")
    pad = 1.0 / ctx.fps
    flash_spans = [(t.start - pad, t.end + pad) for t in transitions if t.kind in _FLASHY_TRANSITIONS]
    for shot, effects in zip(shots, pending):
        effects = [e for e in effects if not (e.type == "flash" and _repeats_transition(e, flash_spans))]
        shot.effects = _dedupe(effects, cfg.dedupe_overlap, min_len)  # type: ignore[assignment]


def _repeats_transition(eff: EffectBase, spans: Sequence[tuple[float, float]]) -> bool:
    """True if an in-shot flash lies mostly inside a flash / fade transition."""
    for a, b in spans:
        if eff.overlap(a, b) >= 0.5 * max(eff.duration, _EPS):
            return True
    return False


def _dedupe(effects: list[EffectBase], max_overlap: float, min_len: float) -> list[EffectBase]:
    """Keep the more confident of overlapping same-type effects; trim marginal overlaps."""
    kept: list[EffectBase] = []
    for eff in sorted(effects, key=lambda e: (-e.confidence, e.start)):
        for other in kept:
            if other.type != eff.type:
                continue
            ov = eff.overlap(other.start, other.end)
            if ov <= 0:
                continue
            if ov > max_overlap * min(eff.duration, other.duration):
                break  # duplicate of a more confident effect
            if eff.start < other.start:
                eff.end = min(eff.end, other.start)
            else:
                eff.start = max(eff.start, other.end)
            if eff.duration < min_len:
                break
        else:
            kept.append(eff)
    return sorted(kept, key=lambda e: (e.start, e.end, e.type))


def _overlays(text: Optional[TextAnalysis], total: float) -> list[TextOverlay]:
    out = []
    for ov in (text.overlays if text is not None else []):
        copy = ov.model_copy(deep=True)
        copy.start, copy.end = max(0.0, copy.start), min(total, copy.end)
        if copy.end > copy.start:
            out.append(copy)
    return sorted(out, key=lambda o: (o.start, o.end))


# =========================================================================== beats


class _BeatGrid:
    """Beat times with gaps filled, extended one period before the first / after the last beat."""

    def __init__(self, beats: np.ndarray, period: float, tolerance: float) -> None:
        filled = [beats[:1]]
        for a, b in zip(beats[:-1], beats[1:]):
            gap = int(round((b - a) / period))
            if gap > 1:  # the tracker lost a few beats: fill them in on the period grid
                filled.append(a + (b - a) * np.arange(1, gap) / gap)
            filled.append(np.array([b]))
        grid = np.concatenate(filled)
        self.times = np.concatenate([[grid[0] - period], grid, [grid[-1] + period]])
        self.period = period
        self.tolerance = tolerance

    @classmethod
    def from_music(cls, music: Optional[Music], tolerance: float) -> Optional["_BeatGrid"]:
        if music is None or not music.beats:
            return None
        beats = np.unique(np.asarray([b for b in music.beats if math.isfinite(b)], float))
        if beats.size == 0:
            return None
        if music.bpm and music.bpm > 0:
            period = 60.0 / music.bpm
        elif beats.size >= 2:
            period = float(np.median(np.diff(beats)))
        else:
            return None
        return cls(beats, period, tolerance) if period > 0 else None

    def offset(self, t: float) -> Optional[float]:
        """Signed seconds from the nearest beat, or ``None`` outside the music."""
        if t < self.times[0] - 0.5 * self.period or t > self.times[-1] + 0.5 * self.period:
            return None
        i = int(np.argmin(np.abs(self.times - t)))
        return float(t - self.times[i])

    def on_beat(self, t: float) -> Optional[bool]:
        off = self.offset(t)
        return None if off is None else abs(off) <= self.tolerance + _EPS


def _align_to_beats(
    grid: _BeatGrid,
    shots: list[Shot],
    fade_in: Optional[Transition],
    cuts: dict[int, Transition],
    fade_out: Optional[Transition],
    total: float,
) -> None:
    """Beat offsets of shot starts and transition boundaries (fades: 0 and the end)."""
    events: list[tuple[Union[Shot, Transition], float]] = [(s, s.start) for s in shots]
    events += [(tr, shots[i].start) for i, tr in cuts.items()]
    events += [(tr, t) for tr, t in ((fade_in, 0.0), (fade_out, total)) if tr is not None]
    for node, t in events:
        off = grid.offset(t)
        node.beat_offset = None if off is None else round(off, 4)
        node.on_beat = None if off is None else abs(off) <= grid.tolerance + _EPS
    for shot in shots:
        shot.beats = round(shot.duration / grid.period, 2)


# =========================================================================== source groups


def _group_sources(ctx: AnalysisContext, seg: ShotSegmentation, shots: list[Shot], cfg: _Settings) -> None:
    """Label shots that look like the same source clip (average-linkage on colour histograms)."""
    hist = np.asarray(ctx.features.hist, float)
    if hist.ndim != 2 or hist.shape[0] < ctx.n_frames:
        return
    # Frames whose colours say little about the footage: transitions (two shots mixed)
    # and pixel effects (flash, invert, ...).
    busy = seg.transition_frames()
    for shot in shots:
        for eff in shot.effects:
            if eff.type in _PIXEL_EFFECTS:
                busy[int(round(eff.start * ctx.fps)):int(round(eff.end * ctx.fps))] = True
    means = np.empty((len(shots), hist.shape[1]))
    for i, s in enumerate(shots):
        rows = hist[s.frame_start:s.frame_end]
        clean = rows[~busy[s.frame_start:s.frame_end]]
        means[i] = (clean if len(clean) else rows).mean(axis=0)
    means /= np.maximum(means.sum(axis=1, keepdims=True), _EPS)
    labels = _cluster(means, cfg.group_threshold)
    for shot, lab in zip(shots, labels):
        shot.source_group = lab


def _cluster(hists: np.ndarray, threshold: float) -> list[str]:
    k = len(hists)
    sim = np.stack([np.minimum(h, hists).sum(axis=1) for h in hists]) if k else np.zeros((0, 0))
    np.fill_diagonal(sim, -np.inf)
    sizes = np.ones(k)
    members = [[i] for i in range(k)]
    while k > 1:
        flat = int(np.argmax(sim))
        i, j = divmod(flat, k)
        if sim[i, j] < threshold:
            break
        merged = (sizes[i] * sim[i] + sizes[j] * sim[j]) / (sizes[i] + sizes[j])
        sim[i], sim[:, i] = merged, merged
        sim[j], sim[:, j] = -np.inf, -np.inf
        sim[i, i] = -np.inf
        sizes[i] += sizes[j]
        members[i] += members[j]
        members[j] = []
    order = sorted((min(m), m) for m in members if m)
    labels = [""] * k
    for rank, (_, group) in enumerate(order):
        for idx in group:
            labels[idx] = _letters(rank)
    return labels


def _letters(k: int) -> str:
    """Spreadsheet-style labels: 0 -> A, 25 -> Z, 26 -> AA."""
    out = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        out = chr(65 + r) + out
    return out


# =========================================================================== sections


def _sections(
    shots: list[Shot],
    fade_in: Optional[Transition],
    cuts: dict[int, Transition],
    fade_out: Optional[Transition],
    music: Optional[Music],
    overlays: list[TextOverlay],
    total: float,
    cfg: _Settings,
) -> list[Section]:
    k = len(shots)
    bounds = np.array([s.start for s in shots] + [total])
    durations = np.diff(bounds)
    plan: list[tuple[str, int, int]] = []  # (role, first shot, end shot exclusive)

    drops = sorted(music.drops, key=lambda d: d.time) if music is not None else []
    drop_first = _snap(bounds, drops[0].time) if drops else None
    hook_end = _hook_end(bounds, overlays, total, cfg, limit=drop_first)
    pos = hook_end or 0
    if hook_end:
        plan.append(("hook", 0, hook_end))

    runs: list[tuple[int, int]] = []
    for drop in drops:
        di = _snap(bounds, drop.time)
        floor = runs[-1][1] if runs else pos
        if runs and di < floor:
            continue  # inside the high-energy run of an earlier drop
        di = max(di, floor)
        if di >= k:
            continue
        ei = _snap(bounds, _high_energy_end(music, drop.time, total, cfg))  # type: ignore[arg-type]
        runs.append((di, int(np.clip(ei, di + 1, k))))
    for r, (di, ei) in enumerate(runs):
        if di > pos:
            plan.append(("build" if r == 0 else "break", pos, di))
        plan.append(("drop", di, ei))
        pos = ei
    if runs:
        if pos < k:
            if total - bounds[pos] >= cfg.outro_min - _EPS:
                plan.append(("outro", pos, k))
            else:
                role, a, _ = plan[-1]
                plan[-1] = (role, a, k)
    elif pos < k:
        plan += _pacing_sections(durations, pos, k, cfg)

    merged: list[tuple[str, int, int]] = []
    for role, a, b in plan:
        if b <= a:
            continue
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1], b)
        else:
            merged.append((role, a, b))

    sections = []
    for role, a, b in merged:
        items: list[TimelineItem] = []
        for shot in shots[a:b]:
            if shot.index == 0 and fade_in is not None:
                items.append(fade_in)
            if shot.index > 0:
                items.append(cuts[shot.index])
            items.append(shot)
        if b == k and fade_out is not None:
            items.append(fade_out)
        start, end = float(bounds[a]), float(bounds[b])
        n_cuts = sum(1 for s in shots[a:b] if s.index > 0)
        sections.append(Section(
            role=role, start=start, end=end, items=items,  # type: ignore[arg-type]
            cuts_per_second=round(n_cuts / max(end - start, _EPS), 4),
            mean_shot_duration=round(float(durations[a:b].mean()), 4),
            energy=_mean_energy(music, start, end),
        ))
    return sections


def _snap(bounds: np.ndarray, t: float) -> int:
    """Index of the shot boundary nearest to ``t`` (``len(shots)`` = the end)."""
    return int(np.argmin(np.abs(bounds - t)))


def _hook_end(bounds: np.ndarray, overlays: list[TextOverlay], total: float, cfg: _Settings,
              limit: Optional[int]) -> Optional[int]:
    """Shot index where the hook ends, or ``None`` when the opening shot is too long for a hook."""
    k = len(bounds) - 1
    candidates = [i for i in range(1, k + 1) if bounds[i] >= cfg.hook_min - _EPS]
    if not candidates:
        return k  # the whole video is shorter than a hook
    end = candidates[0]
    if bounds[end] > cfg.hook_max + _EPS:
        return None
    capped = [i for i in range(end, k + 1) if bounds[i] <= cfg.hook_max + _EPS]
    if limit is not None and limit > end:
        capped = [i for i in capped if i <= limit]
    opening = [ov for ov in overlays if ov.start <= cfg.hook_text_window and ov.role != "watermark"
               and ov.duration < 0.8 * total]
    if opening and capped:
        target = max(ov.end for ov in opening) - cfg.hook_text_tolerance
        covering = [i for i in capped if bounds[i] >= target - _EPS]
        end = covering[0] if covering else capped[-1]
    return end


def _high_energy_end(music: Music, t_drop: float, total: float, cfg: _Settings) -> float:
    """End time of the high-energy audio run that starts at the drop."""
    secs = sorted(music.sections, key=lambda s: s.start)
    if secs:
        i = next((j for j, s in enumerate(secs) if s.start <= t_drop < s.end), None)
        if i is None:
            i = int(np.argmin([abs(s.start - t_drop) for s in secs]))
        if secs[i].label != "drop" and i + 1 < len(secs) and secs[i + 1].start - t_drop <= cfg.drop_snap:
            i += 1
        ref = secs[i].energy
        while i + 1 < len(secs):
            nxt = secs[i + 1]
            if nxt.label == "drop" or (nxt.label == "body" and nxt.energy >= cfg.drop_keep * ref):
                i += 1
            else:
                break
        return secs[i].end
    if music.energy is not None and len(music.energy.values) > 1 and music.energy.dt > 0:
        # No sections: the run ends at the first lull (energy smoothed over ~1 s, so
        # kick pulses do not count) that lasts at least outro_min.
        dt = music.energy.dt
        smooth = ndimage.uniform_filter1d(np.asarray(music.energy.values, float),
                                          max(1, int(round(1.0 / dt))), mode="nearest")
        times = music.energy.t0 + dt * np.arange(smooth.size)
        after = times >= t_drop
        ref = smooth[after & (times < t_drop + 2.0)]
        if ref.size:
            low = np.concatenate([[False], after & (smooth < cfg.drop_keep * float(ref.mean())), [False]])
            edges = np.flatnonzero(np.diff(low.astype(np.int8)))
            for a, b in zip(edges[::2], edges[1::2]):
                if (b - a) * dt >= cfg.outro_min - _EPS:
                    return float(times[a])
    return total


def _pacing_sections(durations: np.ndarray, lo: int, hi: int, cfg: _Settings) -> list[tuple[str, int, int]]:
    """Split shots ``[lo, hi)`` where the mean shot duration changes by > ``pacing_ratio``."""
    cum = np.concatenate([[0.0], np.cumsum(durations)])
    splits = _split_pacing(cum, lo, hi, cfg)
    edges = [lo, *splits, hi]
    parts = list(zip(edges[:-1], edges[1:]))
    means = [(cum[b] - cum[a]) / (b - a) for a, b in parts]
    roles = ["body"] * len(parts)
    for i in range(len(parts) - 1):
        if means[i] > cfg.pacing_ratio * means[i + 1]:
            if roles[i] == "body":
                roles[i] = "build"
            roles[i + 1] = "drop"
    return [(role, a, b) for role, (a, b) in zip(roles, parts)]


def _split_pacing(cum: np.ndarray, lo: int, hi: int, cfg: _Settings) -> list[int]:
    m = cfg.min_section_shots
    ks = np.arange(lo + m, hi - m + 1)
    if ks.size == 0:
        return []
    left, right = cum[ks] - cum[lo], cum[hi] - cum[ks]
    ok = (left >= cfg.min_section_duration) & (right >= cfg.min_section_duration)
    if not ok.any():
        return []
    ratio = np.abs(np.log((left / (ks - lo)) / (right / (hi - ks))))
    ratio[~ok] = -np.inf
    best = int(np.argmax(ratio))
    if ratio[best] < math.log(cfg.pacing_ratio):
        return []
    k = int(ks[best])
    return _split_pacing(cum, lo, k, cfg) + [k] + _split_pacing(cum, k, hi, cfg)


def _mean_energy(music: Optional[Music], start: float, end: float) -> Optional[float]:
    if music is None or music.energy is None or not music.energy.values:
        return None
    values = np.asarray(music.energy.values, float)
    times = music.energy.t0 + music.energy.dt * np.arange(values.size)
    inside = (times >= start) & (times < end)
    if inside.any():
        return round(float(values[inside].mean()), 4)
    return round(float(np.interp(0.5 * (start + end), times, values)), 4)


# =========================================================================== patterns


def _hit_time(eff: EffectBase) -> float:
    """When an effect "hits": flashes peak mid-span, everything else lands at its start."""
    return 0.5 * (eff.start + eff.end) if eff.type == "flash" else eff.start


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def beats_phrase(n: float) -> str:
    """Plain-English rhythm: ``1 -> "every beat"``, ``0.5 -> "every half beat"``, ``4 -> "every 4 beats (one bar)"``."""
    named = {0.25: "quarter beat", 0.5: "half beat", 1.0: "beat"}
    if n in named:
        return f"every {named[n]}"
    return f"every {n:g} beats" + (" (one bar)" if n == 4 else "")


def _patterns(edit: Edit, grid: Optional[_BeatGrid], cfg: _Settings) -> list[Pattern]:
    out: list[Pattern] = []
    shots = edit.shots()
    cuts = shots[1:]
    music = edit.music
    drops = [d.time for d in music.drops] if music is not None else []
    # Pans mostly describe camera movement in the footage, not an editing choice.
    effects = [e for _, e in edit.effects() if e.type != "pan" and e.params.get("source") != "camera"]
    tol_ms = f"{cfg.beat_tolerance * 1000:.0f} ms"

    if grid is not None:
        flags = [s.on_beat for s in cuts if s.on_beat is not None]
        if len(flags) >= 3 and np.mean(flags) >= cfg.min_support:
            ratio = float(np.mean(flags))
            out.append(Pattern(
                rule="cut_on_beat", support=round(ratio, 3),
                description=f"{_pct(ratio)} of cuts land on the beat (within {tol_ms}), so the edit is cut to the music.",
                params={"cuts": len(flags), "on_beat": int(sum(flags)), "tolerance": cfg.beat_tolerance},
            ))
        bpm = 60.0 / grid.period
        for sec in edit.timeline.sections:
            beats = np.array([s.beats for s in sec.shots() if s.beats], float)
            if beats.size < 2:
                continue
            counts = [int(np.sum(np.abs(beats - q) <= cfg.beat_quant_tolerance * q)) for q in BEAT_QUANTA]
            median = float(np.median(beats))
            best = max(range(len(BEAT_QUANTA)),
                       key=lambda i: (counts[i], -abs(math.log(BEAT_QUANTA[i] / median))))
            n_beats, hits = BEAT_QUANTA[best], counts[best]
            support = hits / beats.size
            if hits >= cfg.min_pattern_count and support >= cfg.min_support:
                out.append(Pattern(
                    rule="cut_every_n_beats", start=sec.start, end=sec.end, support=round(support, 3),
                    description=(f"In the {sec.role}, the edit cuts {beats_phrase(n_beats)} "
                                 f"({n_beats * grid.period:.2f} s at {bpm:.0f} BPM); {hits} of {beats.size} shots follow it."),
                    params={"n": n_beats, "seconds": round(n_beats * grid.period, 4), "section": sec.role,
                            "shots": int(beats.size), "matching": hits},
                ))

    out += _drop_patterns(edit, drops, effects, cfg)

    if grid is not None:
        by_type: dict[str, list[EffectBase]] = {}
        for e in effects:
            if e.type != "speed":  # speed changes have their own pattern
                by_type.setdefault(e.type, []).append(e)
        for kind, group in sorted(by_type.items()):
            flags = [grid.on_beat(_hit_time(e)) for e in group]
            known = [f for f in flags if f is not None]
            if len(known) >= cfg.min_pattern_count and np.mean(known) >= cfg.min_support:
                ratio = float(np.mean(known))
                out.append(Pattern(
                    rule=f"effect_on_beat:{kind}", support=round(ratio, 3),
                    description=f"{kind.replace('_', ' ').capitalize()} effects hit on the beat ({int(sum(known))} of {len(known)}).",
                    params={"effect": kind, "count": len(known), "on_beat": int(sum(known))},
                ))

    starts = {s.id or shot_id(s.index): s.start for s in shots}
    boundary = [t for t in edit.transitions() if t.from_shot is not None and t.to_shot in starts]
    if boundary:
        flash_fx = [_hit_time(e) for e in effects if e.type == "flash"]
        flashy = 0
        for tr in boundary:
            b = starts[tr.to_shot]  # type: ignore[index]
            if tr.kind in ("flash", "fade_white") or any(abs(t - b) <= cfg.flash_cut_window for t in flash_fx):
                flashy += 1
        if flashy >= cfg.min_pattern_count:
            ratio = flashy / len(boundary)
            out.append(Pattern(
                rule="flash_on_cut", support=round(ratio, 3),
                description=f"A flash punctuates {flashy} of {len(boundary)} cuts ({_pct(ratio)}).",
                params={"count": flashy, "cuts": len(boundary)},
            ))

    total = edit.media.duration
    opening = [ov for ov in edit.overlays
               if ov.start <= cfg.hook_text_time and ov.role != "watermark" and ov.duration < 0.8 * total]
    if opening:
        ov = opening[0]
        quoted = f' ("{ov.text}")' if ov.text else ""
        out.append(Pattern(
            rule="hook_text", start=ov.start, end=ov.end, support=1.0,
            description=f"Opens with on-screen text{quoted} in the first second to hook viewers.",
            params={"text": ov.text, "overlay_start": round(ov.start, 4), "overlay_end": round(ov.end, 4)},
        ))

    if grid is not None:
        ramps = [e for e in effects if e.type == "speed" and getattr(e, "kind", "constant") != "constant"]
        flags = [bool(grid.on_beat(e.start)) or bool(grid.on_beat(e.end)) for e in ramps]
        if len(flags) >= cfg.min_pattern_count and np.mean(flags) >= cfg.min_support:
            ratio = float(np.mean(flags))
            out.append(Pattern(
                rule="speed_ramp_on_beat", support=round(ratio, 3),
                description=f"Speed ramps start or land on a beat ({int(sum(flags))} of {len(flags)}).",
                params={"count": len(flags), "on_beat": int(sum(flags))},
            ))

    recurring = _recurring_groups(shots)
    if recurring:
        members = sum(1 for s in shots if s.source_group in recurring)
        returns = sum(len(runs) - 1 for runs in recurring.values())
        names = ", ".join(sorted(recurring))
        out.append(Pattern(
            rule="recurring_source", support=round(members / len(shots), 3),
            description=(f"The edit cuts away and comes back to the same clip {returns} "
                         f"time{'s' if returns != 1 else ''} (source group{'s' if len(recurring) > 1 else ''} {names})."),
            params={"groups": sorted(recurring), "returns": returns},
        ))
    return out


def _drop_patterns(edit: Edit, drops: list[float], effects: list[EffectBase], cfg: _Settings) -> list[Pattern]:
    if not drops:
        return []
    out = []
    sections = edit.timeline.sections
    drop_secs = [i for i, s in enumerate(sections) if s.role == "drop"]
    cut_times = np.asarray(edit.cut_times(), float)
    accelerating, details = 0, []
    for i in drop_secs:
        sec = sections[i]
        before_start = sections[i - 1].start if i > 0 and sections[i - 1].duration >= 1.5 else 0.0
        if sec.start - before_start <= _EPS:
            continue
        before = float(np.sum((cut_times >= before_start) & (cut_times < sec.start)))
        after = float(np.sum((cut_times >= sec.start - _EPS) & (cut_times < sec.end - _EPS)))
        rate_b, rate_a = before / (sec.start - before_start), after / max(sec.duration, _EPS)
        ok = after >= 2 and (rate_b == 0 or rate_a >= cfg.accel_ratio * rate_b)
        accelerating += int(ok)
        details.append((rate_b, rate_a, ok))
    if details and accelerating / len(details) >= cfg.min_support and accelerating:
        rate_b, rate_a, _ = next(d for d in details if d[2])
        speedup = f" ({rate_a / rate_b:.1f}x faster)" if rate_b > 0 else ""
        out.append(Pattern(
            rule="pacing_accelerates_at_drop", support=round(accelerating / len(details), 3),
            description=f"Cutting speeds up when the drop hits: {rate_b:.2f} -> {rate_a:.2f} cuts per second{speedup}.",
            params={"before": round(rate_b, 4), "after": round(rate_a, 4),
                    "ratio": round(rate_a / rate_b, 3) if rate_b > 0 else None},
        ))

    hits: dict[str, set[int]] = {}
    sources: dict[str, str] = {}
    w = cfg.drop_window
    for k, t in enumerate(drops):
        for e in effects:
            times = [_hit_time(e)] + ([e.end] if e.type == "speed" else [])
            if any(abs(x - t) <= w for x in times):
                hits.setdefault(e.type, set()).add(k)
                sources.setdefault(e.type, "effect")
        for tr in edit.transitions():
            if tr.kind != "cut" and tr.start - w <= t <= tr.end + w:
                hits.setdefault(tr.kind, set()).add(k)
                sources.setdefault(tr.kind, "transition")
    for kind, ks in sorted(hits.items()):
        support = len(ks) / len(drops)
        if support < cfg.min_support:
            continue
        what = kind.replace("_", " ")
        noun = "transition" if sources[kind] == "transition" else "effect"
        when = ", ".join(f"{drops[k]:.2f} s" for k in sorted(ks))
        out.append(Pattern(
            rule=f"effect_on_drop:{kind}", support=round(support, 3),
            description=f"{_article(what).capitalize()} {what} {noun} lands right on the drop ({when}).",
            params={"kind": kind, "source": sources[kind], "drops": [round(drops[k], 4) for k in sorted(ks)]},
        ))
    return out


def _recurring_groups(shots: list[Shot]) -> dict[str, list[int]]:
    """Source groups that come back after other footage: group -> start shot of each run."""
    runs: dict[str, list[int]] = {}
    prev: Optional[str] = None
    for s in shots:
        if s.source_group is not None and s.source_group != prev:
            runs.setdefault(s.source_group, []).append(s.index)
        prev = s.source_group
    return {g: r for g, r in runs.items() if len(r) >= 2}


# =========================================================================== stats & provenance


def _stats(edit: Edit, total: float) -> EditStats:
    shots = edit.shots()
    durs = np.array([s.duration for s in shots]) if shots else np.zeros(1)
    cuts = np.asarray(edit.cut_times(), float)
    on = [s.on_beat for s in shots[1:] if s.on_beat is not None]
    effect_counts: dict[str, int] = {}
    for _, eff in edit.effects():
        effect_counts[eff.type] = effect_counts.get(eff.type, 0) + 1
    transition_counts: dict[str, int] = {}
    for tr in edit.transitions():
        transition_counts[tr.kind] = transition_counts.get(tr.kind, 0) + 1
    sections = edit.timeline.sections
    return EditStats(
        n_shots=len(shots),
        n_transitions=sum(1 for t in edit.transitions() if t.kind != "cut"),
        mean_shot_duration=float(durs.mean()),
        median_shot_duration=float(np.median(durs)),
        min_shot_duration=float(durs.min()),
        max_shot_duration=float(durs.max()),
        cuts_per_minute=60.0 * cuts.size / total if total > 0 else 0.0,
        cut_on_beat_ratio=float(np.mean(on)) if on else None,
        hook_duration=sections[0].duration if sections and sections[0].role == "hook" else 0.0,
        first_cut_time=float(cuts[0]) if cuts.size else None,
        effect_counts=dict(sorted(effect_counts.items())),
        transition_counts=dict(sorted(transition_counts.items())),
        text_coverage=_coverage([(o.start, o.end) for o in edit.overlays], total),
        pacing=_pacing(cuts, total),
    )


def _coverage(spans: list[tuple[float, float]], total: float) -> float:
    """Fraction of ``[0, total)`` covered by the union of ``spans``."""
    if total <= 0:
        return 0.0
    covered, reach = 0.0, 0.0
    for a, b in sorted((max(0.0, a), min(total, b)) for a, b in spans):
        if b <= reach:
            continue
        covered += b - max(a, reach)
        reach = b
    return covered / total


def _pacing(cuts: np.ndarray, total: float, window: float = 1.0, step: float = 0.5) -> Curve:
    """Cuts per second in ``window``-second windows every ``step`` seconds (centre times)."""
    if total < window:
        return Curve(t0=total / 2, dt=step, values=[float(cuts.size / total) if total > 0 else 0.0])
    centres = np.arange(window / 2, total - window / 2 + _EPS, step)
    lo = np.searchsorted(cuts, centres - window / 2 - _EPS, side="left")
    hi = np.searchsorted(cuts, centres + window / 2 - _EPS, side="left")
    return Curve(t0=window / 2, dt=step, values=[float(v) / window for v in hi - lo])


def _default_status(**results: Any) -> dict[str, str]:
    status = {"shots": "ok"}
    for name, result in results.items():
        status[name] = "ok" if result is not None else "not run"
    text = results.get("text")
    if isinstance(text, TextAnalysis):
        status["text"] = f"ok (ocr: {text.ocr_engine})" if text.ocr_engine else "ok (no ocr)"
    return status


def _json_safe(value: Any) -> Any:
    """Round-trip through JSON so the provenance never holds numpy scalars or tuples."""
    def default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    return json.loads(json.dumps(value, default=default))
