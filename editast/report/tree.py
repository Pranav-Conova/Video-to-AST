"""Terminal views of an Edit: a colour-coded :mod:`rich` tree and a plain-text summary.

:func:`build_tree` renders one line per node (sections, shots, transitions,
effects, overlays, patterns, music) with a fixed colour per node type, e.g.::

    shot s3 1.20-1.65s (0.45s, 1 beat, on-beat) group B

Plain hard cuts are implied by shot boundaries and hidden unless
``show_cuts=True``. :func:`edit_metrics` computes the headline numbers shared
by :func:`summary_text` and the HTML report: values are measured from the tree
itself, with :class:`~editast.ast.nodes.EditStats` used only where the tree
cannot answer (the analyzer's own on-beat ratio and hook length).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Optional

from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from editast.ast.nodes import EFFECT_TYPES, Edit, EffectBase, Music, Pattern, Section, Shot, TextOverlay, Transition

__all__ = ["EditMetrics", "edit_metrics", "build_tree", "print_tree", "summary_text", "STYLES"]

STYLES: dict[str, str] = {
    "edit": "bold",
    "music": "bold red",
    "audio_section": "red",
    "drop": "bright_red",
    "section": "bold magenta",
    "shot": "cyan",
    "transition": "yellow",
    "cut": "dim yellow",
    "effect": "green",
    "text": "blue",
    "pattern": "bright_magenta",
    "stats": "bold white",
    "detail": "dim",
}


# --------------------------------------------------------------------------- metrics


@dataclass
class EditMetrics:
    """Headline numbers of an edit (``None`` = unknown)."""

    duration: float = 0.0
    n_shots: int = 0
    mean_shot: Optional[float] = None
    median_shot: Optional[float] = None
    cuts_per_minute: Optional[float] = None
    on_beat_ratio: Optional[float] = None
    hook: Optional[float] = None
    first_cut: Optional[float] = None
    text_coverage: Optional[float] = None
    bpm: Optional[float] = None
    n_beats: int = 0
    drops: list[float] = field(default_factory=list)
    effect_counts: dict[str, int] = field(default_factory=dict)
    transition_counts: dict[str, int] = field(default_factory=dict)  # excluding plain cuts
    n_cuts: int = 0  # plain hard cuts


def _coverage(spans: Iterable[tuple[float, float]], duration: float) -> float:
    """Fraction of ``[0, duration)`` covered by the union of ``spans``."""
    covered, cur_end = 0.0, 0.0
    for start, end in sorted((max(0.0, s), min(duration, e)) for s, e in spans):
        if end <= cur_end:
            continue
        covered += end - max(start, cur_end)
        cur_end = end
    return covered / duration if duration > 0 else 0.0


def edit_duration(edit: Edit) -> float:
    """Timeline length: the media duration, or the latest end of any timed node."""
    ends = [edit.media.duration]
    ends += [s.end for s in edit.shots()] + [o.end for o in edit.overlays]
    ends += [sec.end for sec in edit.timeline.sections]
    return max(0.0, *ends)


def edit_metrics(edit: Edit) -> EditMetrics:
    shots = edit.shots()
    stats = edit.stats
    m = EditMetrics(duration=edit_duration(edit), n_shots=len(shots))
    durations = [s.duration for s in shots]
    if durations:
        m.mean_shot = sum(durations) / len(durations)
        m.median_shot = median(durations)
        if len(shots) > 1:
            m.first_cut = shots[1].start
    elif stats is not None and stats.n_shots:
        m.n_shots, m.mean_shot, m.median_shot = stats.n_shots, stats.mean_shot_duration, stats.median_shot_duration
        m.first_cut = stats.first_cut_time
    if m.duration > 0 and m.n_shots:
        m.cuts_per_minute = 60.0 * (m.n_shots - 1) / m.duration
    if stats is not None and stats.cut_on_beat_ratio is not None:
        m.on_beat_ratio = stats.cut_on_beat_ratio
    else:
        flags = [s.on_beat for s in shots[1:] if s.on_beat is not None]
        m.on_beat_ratio = sum(flags) / len(flags) if flags else None
    hooks = [sec for sec in edit.timeline.sections if sec.role == "hook"]
    if stats is not None and stats.hook_duration > 0:
        m.hook = stats.hook_duration
    elif hooks:
        m.hook = hooks[0].duration
    if edit.overlays and m.duration > 0:
        m.text_coverage = _coverage(((o.start, o.end) for o in edit.overlays), m.duration)
    elif stats is not None:
        m.text_coverage = stats.text_coverage
    if edit.music is not None:
        m.bpm = edit.music.bpm
        m.n_beats = len(edit.music.beats)
        m.drops = [d.time for d in edit.music.drops]
    effects = Counter(e.type for _, e in edit.effects())
    m.effect_counts = {t: effects[t] for t in EFFECT_TYPES if effects[t]}
    kinds = Counter(t.kind for t in edit.transitions() if not _is_plain_cut(t))
    m.transition_counts = dict(kinds.most_common())
    m.n_cuts = sum(1 for t in edit.transitions() if _is_plain_cut(t))
    if not effects and stats is not None:
        m.effect_counts = dict(stats.effect_counts)
    return m


def _is_plain_cut(t: Transition) -> bool:
    return t.kind == "cut" and t.end - t.start < 1e-6


# --------------------------------------------------------------------------- label helpers


def _span(start: float, end: float) -> str:
    return f"{start:.2f}-{end:.2f}s"


def _num(x: float, digits: int = 2) -> str:
    """``x`` with at most ``digits`` decimals and no trailing zeros (``2.50`` -> ``2.5``)."""
    text = f"{x:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _beats(n: float) -> str:
    return f"{_num(n)} beat" + ("" if abs(n - 1.0) < 1e-9 else "s")


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _hex(rgb: Optional[tuple[int, int, int]]) -> str:
    if rgb is None:
        return ""
    if all(0 <= c <= 255 for c in rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "rgb{}".format(tuple(rgb))


def _label(kind: str, node_id: str, *rest: str, detail: str = "") -> Text:
    """``kind id rest...`` styled by node kind, followed by a dimmed detail."""
    style = STYLES.get(kind, STYLES["effect"])
    text = Text()
    text.append(kind, style=style + " bold")
    if node_id:
        text.append(" " + node_id, style=style)
    for part in rest:
        if part:
            text.append(" " + part)
    if detail:
        text.append(" " + detail, style=STYLES["detail"])
    return text


def shot_label(shot: Shot) -> Text:
    info = [f"{shot.duration:.2f}s"]
    if shot.beats is not None:
        info.append(_beats(shot.beats))
    if shot.on_beat is not None:
        info.append("on-beat" if shot.on_beat else "off-beat")
    extra = []
    if shot.source_group:
        extra.append(f"group {shot.source_group}")
    if shot.motion is not None and shot.motion.camera != "static":
        extra.append(shot.motion.camera)
    if shot.look is not None and shot.look.bw:
        extra.append("b&w")
    if shot.layout is not None and (shot.layout.letterbox_top or shot.layout.letterbox_bottom):
        extra.append("letterbox")
    if shot.layout is not None and shot.layout.split != "none":
        extra.append(f"{shot.layout.split} split")
    return _label("shot", shot.id, _span(shot.start, shot.end), f"({', '.join(info)})", *extra)


def effect_desc(eff: EffectBase) -> str:
    """Short human description of an effect's parameters."""
    t = eff.type
    parts: list[str] = []
    if t == "zoom":
        parts.append(f"{eff.scale_from:.2f}→{eff.scale_to:.2f}")
        if eff.center != (0.5, 0.5):
            parts.append(f"@({eff.center[0]:.2f},{eff.center[1]:.2f})")
        if eff.digital is not None:
            parts.append("digital" if eff.digital else "optical")
    elif t == "pan":
        parts.append(f"by ({eff.dx:+.2f}, {eff.dy:+.2f})")
    elif t == "rotate":
        parts.append(f"{_num(eff.angle_from, 1)}°→{_num(eff.angle_to, 1)}°")
    elif t == "shake":
        parts.append(f"amp {eff.amplitude:.3f} @ {eff.frequency:.1f} Hz")
        if eff.rotation_amplitude:
            parts.append(f"rot {_num(eff.rotation_amplitude, 1)}°")
    elif t == "speed":
        parts.append(f"{eff.kind} {eff.rate_from:.2f}×→{eff.rate_to:.2f}×")
    elif t == "flash":
        parts.append(f"{_hex(eff.color)} ×{eff.intensity:.2f}")
    elif t == "blur":
        parts.append(f"σ {eff.amount:.3f}")
    elif t == "rgb_split":
        parts.append(f"offset {eff.offset:.3f}")
    if eff.easing != "linear":
        parts.append(eff.easing)
    return " ".join(parts)


def effect_label(eff: EffectBase) -> Text:
    detail = f"conf {eff.confidence:.2f}" if eff.confidence < 1.0 else ""
    return _label(eff.type, "", _span(eff.start, eff.end), effect_desc(eff), detail=detail)


def transition_label(t: Transition) -> Text:
    style_kind = "cut" if _is_plain_cut(t) else "transition"
    link = f"{t.from_shot or '∅'}→{t.to_shot or '∅'}"
    extra = [_hex(t.color)] if t.color is not None else []
    if t.on_beat is not None:
        extra.append("on-beat" if t.on_beat else "off-beat")
    text = Text()
    text.append("transition", style=STYLES[style_kind] + " bold")
    if t.id:
        text.append(" " + t.id, style=STYLES[style_kind])
    text.append(f" {t.kind} {_span(t.start, t.end)} {link}")
    for part in extra:
        text.append(" " + part)
    return text


def text_label(ov: TextOverlay) -> Text:
    shown = "(unread)" if ov.text is None else f'"{_clip(ov.text, 48)}"'
    detail = " · ".join(x for x in (ov.position, ov.animation, ov.role) if x and x != "unknown")
    return _label("text", ov.id, shown, _span(ov.start, ov.end), detail=detail)


def pattern_label(p: Pattern) -> Text:
    params = ", ".join(f"{k}={v}" for k, v in p.params.items())
    rest = [p.rule, f"{100 * p.support:.0f}%"]
    if params:
        rest.append(f"({_clip(params, 60)})")
    return _label("pattern", p.id, *rest, detail=f"— {p.description}" if p.description else "")


def section_label(sec: Section) -> Text:
    shots = sec.shots()
    info = [f"{sec.duration:.2f}s", f"{len(shots)} shot" + ("" if len(shots) == 1 else "s")]
    if sec.cuts_per_second:
        info.append(f"{sec.cuts_per_second:.2f} cuts/s")
    if sec.energy is not None:
        info.append(f"energy {sec.energy:.2f}")
    return _label("section", sec.id, sec.role, _span(sec.start, sec.end), detail=f"({', '.join(info)})")


def music_label(music: Music) -> Text:
    rest = [f"{music.bpm:.1f} BPM" if music.bpm is not None else "tempo unknown", f"{music.meter}/4"]
    detail = [f"{len(music.beats)} beats"]
    if music.downbeats:
        detail.append(f"{len(music.downbeats)} downbeats")
    if music.has_speech:
        detail.append("speech")
    return _label("music", "", *rest, detail=" · ".join(detail))


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --------------------------------------------------------------------------- tree


def build_tree(edit: Edit, max_shots: Optional[int] = None, show_cuts: bool = False) -> Tree:
    """A :class:`rich.tree.Tree` with one colour-coded line per node.

    ``max_shots`` limits the number of shots shown (the rest are summarised);
    plain hard cuts are hidden unless ``show_cuts``.
    """
    m = edit_metrics(edit)
    media = edit.media
    head = [f'"{media.source}"' if media.source else "", f"{m.duration:.2f}s"]
    if media.width and media.height:
        head.append(f"{media.width}x{media.height}")
    if media.aspect:
        head.append(media.aspect)
    head.append(f"@{_num(media.fps)}fps")
    root = Tree(_label("edit", "", *head))

    if edit.music is not None:
        music = root.add(music_label(edit.music))
        for drop in edit.music.drops:
            music.add(_label("drop", "", f"{drop.time:.2f}s", detail=f"strength {drop.strength:.2f}"))
        for sec in edit.music.sections:
            music.add(_label("audio_section", sec.id, sec.label, _span(sec.start, sec.end),
                             detail=f"energy {sec.energy:.2f}"))

    shown = 0
    hidden = 0
    for sec in edit.timeline.sections:
        branch = root.add(section_label(sec))
        for item in sec.items:
            if isinstance(item, Shot):
                if max_shots is not None and shown >= max_shots:
                    hidden += 1
                    continue
                shown += 1
                node = branch.add(shot_label(item))
                for eff in item.effects:
                    node.add(effect_label(eff))
            elif (show_cuts or not _is_plain_cut(item)) and (max_shots is None or shown < max_shots):
                branch.add(transition_label(item))
    if hidden:
        root.add(Text(f"… {hidden} more shot" + ("" if hidden == 1 else "s"), style=STYLES["detail"]))

    if edit.overlays:
        texts = root.add(Text(f"overlays ({len(edit.overlays)})", style=STYLES["text"] + " bold"))
        for ov in edit.overlays:
            texts.add(text_label(ov))
    if edit.patterns:
        pats = root.add(Text(f"patterns ({len(edit.patterns)})", style=STYLES["pattern"] + " bold"))
        for p in edit.patterns:
            pats.add(pattern_label(p))
    stats = Text("stats", style=STYLES["stats"])
    stats.append(" " + _stats_line(m), style=STYLES["detail"])
    root.add(stats)
    return root


def _stats_line(m: EditMetrics) -> str:
    parts = [f"{m.n_shots} shots"]
    if m.mean_shot is not None:
        parts.append(f"mean {m.mean_shot:.2f}s")
    if m.cuts_per_minute is not None:
        parts.append(f"{m.cuts_per_minute:.1f} cuts/min")
    if m.on_beat_ratio is not None:
        parts.append(f"{_pct(m.on_beat_ratio)} cuts on beat")
    return " · ".join(parts)


def print_tree(edit: Edit, console: Optional[Console] = None, max_shots: Optional[int] = None,
               show_cuts: bool = False) -> None:
    """Print :func:`build_tree` to ``console`` (a new stdout console by default)."""
    (console or Console()).print(build_tree(edit, max_shots=max_shots, show_cuts=show_cuts))


# --------------------------------------------------------------------------- summary


def _mix(counts: dict[str, int], limit: int = 6) -> str:
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    text = " · ".join(f"{k} {v}" for k, v in items[:limit])
    return text + (f" · +{len(items) - limit} more" if len(items) > limit else "")


def summary_text(edit: Edit) -> str:
    """Short plain-text summary: format, tempo, pacing, hook, effect mix, top patterns."""
    m = edit_metrics(edit)
    media = edit.media
    lines = []
    if media.source:
        lines.append(media.source)
    fmt = [f"{m.duration:.2f}s"]
    if media.aspect or (media.width and media.height):
        size = f"{media.width}x{media.height}" if media.width and media.height else ""
        fmt.append(" ".join(x for x in (media.aspect, f"({size})" if size and media.aspect else size) if x))
    fmt.append(f"{_num(media.fps)} fps")
    if media.has_audio:
        fmt.append("audio")
    lines.append(" · ".join(fmt))
    if edit.music is not None:
        music = [f"{m.bpm:.1f} BPM" if m.bpm is not None else "tempo unknown"]
        if m.n_beats:
            music.append(f"{m.n_beats} beats")
        if m.drops:
            music.append("drop at " + ", ".join(f"{t:.2f}s" for t in m.drops[:3]))
        lines.append("music: " + " · ".join(music))
    shots = [f"{m.n_shots}"]
    if m.mean_shot is not None:
        shots.append(f"mean {m.mean_shot:.2f}s (median {m.median_shot:.2f}s)")
    if m.cuts_per_minute is not None:
        shots.append(f"{m.cuts_per_minute:.1f} cuts/min")
    if m.on_beat_ratio is not None:
        shots.append(f"{_pct(m.on_beat_ratio)} cuts on beat")
    lines.append("shots: " + " · ".join(shots))
    pace = []
    if m.hook is not None:
        pace.append(f"hook {m.hook:.2f}s")
    if m.first_cut is not None:
        pace.append(f"first cut {m.first_cut:.2f}s")
    if m.text_coverage:
        pace.append(f"text on screen {_pct(m.text_coverage)}")
    if pace:
        lines.append(" · ".join(pace))
    if m.effect_counts:
        lines.append("effects: " + _mix(m.effect_counts))
    if m.transition_counts or m.n_cuts:
        mix = _mix(m.transition_counts) if m.transition_counts else "none"
        cuts = f" (+{m.n_cuts} hard cut{'' if m.n_cuts == 1 else 's'})" if m.n_cuts else ""
        lines.append(f"transitions: {mix}{cuts}")
    if edit.patterns:
        top = sorted(edit.patterns, key=lambda p: -p.support)[:3]
        lines.append("patterns: " + " · ".join(f"{p.rule} {_pct(p.support)}" for p in top))
    return "\n".join(lines)
