"""Self-contained HTML report for an Edit.

One HTML file with inline CSS, JS and SVG (no external resources): a header
with the key numbers, a horizontally scrollable timeline whose lanes share one
time axis (sections, shots with optional thumbnails, transitions, one lane per
effect type, text overlays, beats/downbeats/drops, the music energy curve, the
pacing curve and any extra signals), the inferred patterns, the effect and
transition mix, and collapsible compact EditScript and JSON views.

Layout: the time scale (px per second) is chosen so the median shot is about
60 px wide (clamped), and the timeline is at least ~900 px wide. Lane labels
sit in a fixed column left of the scrolling area. Every mark carries a
``data-tip`` attribute; a tiny script shows it as a tooltip, and curve lanes
get a crosshair readout of the nearest sample. Colours are CSS tokens with a
light and a dark set (``prefers-color-scheme``), categorical hues in a fixed
order. Thumbnails (``ctx`` given) are the middle frame of each shot from
``ctx.frames`` (RGB), JPEG-encoded at ~96 px tall and inlined as base64.
"""

from __future__ import annotations

import base64
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import cv2
import numpy as np

from editast.ast.nodes import EFFECT_TYPES, Edit, EffectBase, Shot, Transition
from editast.lang.printer import to_dsl
from editast.report.tree import EditMetrics, edit_duration, edit_metrics, effect_desc

if TYPE_CHECKING:
    from editast.context import AnalysisContext

__all__ = ["write_report", "report_html"]

THUMB_HEIGHT = 96
MAX_THUMBS = 600  # beyond this many shots, thumbnails are skipped to keep the file small
LABEL_WIDTH = 92
SECTION_ROLES = ("hook", "intro", "build", "drop", "body", "break", "outro")

Signals = Mapping[str, tuple[Sequence[float] | np.ndarray, float]]


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _tip(lines: Sequence[str]) -> str:
    """Escaped ``data-tip`` value: non-empty lines joined by newlines (first line = title)."""
    return _e("\n".join(line for line in lines if line))


def _f(x: float) -> str:
    """Compact SVG coordinate."""
    return f"{x:.1f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- thumbnails


def _thumbnail(ctx: "AnalysisContext", t: float) -> Optional[str]:
    """Base64 JPEG of the analysis frame at ``t`` seconds, ``THUMB_HEIGHT`` px tall."""
    if ctx.n_frames == 0:
        return None
    frame = ctx.frames[ctx.f(t)]
    h, w = frame.shape[:2]
    size = (max(1, round(w * THUMB_HEIGHT / h)), THUMB_HEIGHT)
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if h > THUMB_HEIGHT else cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


# --------------------------------------------------------------------------- timeline


@dataclass
class _Lane:
    label: str
    height: int
    body: str  # SVG markup, drawn with the lane's top at y=0 (wrapped in a translated <g>)
    note: str = ""  # tooltip of the label


class _Timeline:
    """Builds the lanes of the timeline SVG for one edit."""

    def __init__(self, edit: Edit, ctx: Optional["AnalysisContext"], signals: Optional[Signals]) -> None:
        self.edit = edit
        self.ctx = ctx
        self.signals = signals or {}
        self.duration = edit_duration(edit)
        shots = edit.shots()
        typical = median([s.duration for s in shots if s.duration > 0] or [1.0])
        pps = min(320.0, max(60.0, 60.0 / max(typical, 1e-3)))
        if self.duration > 0:
            pps = max(pps, 900.0 / self.duration)
        self.pps = pps
        self.width = max(320, int(math.ceil(self.duration * pps)) + 24)

    def x(self, t: float) -> float:
        return 8.0 + t * self.pps

    def span(self, start: float, end: float, min_w: float = 2.0) -> tuple[float, float]:
        x0 = self.x(start)
        return x0, max(min_w, self.x(end) - x0)

    # ---- lanes ---------------------------------------------------------------

    def ruler(self) -> _Lane:
        step = next((s for s in (0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300) if s * self.pps >= 56), 600)
        parts = []
        n = int(self.duration // step) + 1
        for k in range(n + 1):
            t = k * step
            if t > self.duration + 1e-9:
                break
            x = _f(self.x(t))
            label = f"{t:g}s"
            parts.append(f'<line class="tick" x1="{x}" x2="{x}" y1="14" y2="20"/>'
                         f'<text class="axis" x="{x}" y="11" text-anchor="middle">{label}</text>')
        return _Lane("time", 22, "".join(parts))

    def sections(self) -> Optional[_Lane]:
        secs = self.edit.timeline.sections
        if not secs:
            return None
        parts = []
        for sec in secs:
            x, w = self.span(sec.start, sec.end)
            slot = SECTION_ROLES.index(sec.role) + 1 if sec.role in SECTION_ROLES else 8
            tip = [f"section {sec.id} · {sec.role}".strip(), f"{sec.start:.2f}–{sec.end:.2f}s ({sec.duration:.2f}s)",
                   f"{len(sec.shots())} shots · {sec.cuts_per_second:.2f} cuts/s"]
            if sec.energy is not None:
                tip.append(f"energy {sec.energy:.2f}")
            label = f'<text class="in" x="{_f(x + 6)}" y="17">{_e(sec.role)}</text>' if w > 7 * len(sec.role) + 12 else ""
            parts.append(f'<g data-tip="{_tip(tip)}"><rect class="sec c{slot}" x="{_f(x + 1)}" y="2" '
                         f'width="{_f(max(1, w - 2))}" height="22" rx="4"/>'
                         f'<rect class="bar c{slot}" x="{_f(x + 1)}" y="2" width="{_f(max(1, w - 2))}" height="3"/>'
                         f"{label}</g>")
        return _Lane("sections", 26, "".join(parts))

    def shots(self) -> Optional[_Lane]:
        shots = self.edit.shots()
        if not shots:
            return None
        thumbs = self.ctx is not None and len(shots) <= MAX_THUMBS
        img_h = THUMB_HEIGHT if thumbs else 30
        parts = []
        for i, shot in enumerate(shots):
            x, w = self.span(shot.start, shot.end)
            inner_w = max(1.0, w - 2)
            body = [f'<rect class="shot s{i % 2}" x="{_f(x + 1)}" y="0" width="{_f(inner_w)}" '
                    f'height="{img_h + 16}" rx="3"/>']
            if thumbs:
                b64 = _thumbnail(self.ctx, 0.5 * (shot.start + shot.end))
                if b64:
                    body.append(f'<image x="{_f(x + 1)}" y="0" width="{_f(inner_w)}" height="{img_h}" '
                                f'preserveAspectRatio="xMidYMid slice" href="data:image/jpeg;base64,{b64}"/>')
            if inner_w > 8 * len(shot.id) + 4:
                body.append(f'<text class="lbl" x="{_f(x + 4)}" y="{img_h + 12}">{_e(shot.id)}</text>')
            if shot.on_beat is not None:
                cls = "beatdot on" if shot.on_beat else "beatdot off"
                body.append(f'<circle class="{cls}" cx="{_f(x + 6)}" cy="6" r="4"/>')
            parts.append(f'<g data-tip="{_e(_shot_tip(shot))}">{"".join(body)}</g>')
        return _Lane("shots", img_h + 16, "".join(parts), "beat dot: filled = starts on a beat, hollow = off-beat")

    def transitions(self) -> Optional[_Lane]:
        items = self.edit.transitions()
        if not items:
            return None
        parts = []
        for t in items:
            tip = _transition_tip(t)
            if t.kind == "cut" and t.end - t.start < 1e-6:
                x = _f(self.x(t.start))
                parts.append(f'<g data-tip="{_e(tip)}"><rect class="hit" x="{_f(self.x(t.start) - 3)}" y="0" '
                             f'width="6" height="20"/><line class="cut" x1="{x}" x2="{x}" y1="4" y2="16"/></g>')
                continue
            x, w = self.span(t.start, t.end, 4.0)
            label = f'<text class="in" x="{_f(x + 3)}" y="14">{_e(t.kind)}</text>' if w > 7 * len(t.kind) + 6 else ""
            parts.append(f'<g data-tip="{_e(tip)}"><rect class="tr" x="{_f(x)}" y="3" width="{_f(w)}" '
                         f'height="14" rx="3"/>{label}</g>')
        return _Lane("transitions", 20, "".join(parts))

    def effect_lanes(self) -> list[_Lane]:
        by_type: dict[str, list[tuple[Shot, EffectBase]]] = {}
        for shot, eff in self.edit.effects():
            by_type.setdefault(eff.type, []).append((shot, eff))
        lanes = []
        for etype in EFFECT_TYPES:
            if etype not in by_type:
                continue
            parts = []
            for shot, eff in by_type[etype]:
                x, w = self.span(eff.start, eff.end, 3.0)
                tip = [f"{eff.type} {eff.id}".strip(), f"{eff.start:.2f}–{eff.end:.2f}s in {shot.id or 'shot'}",
                       effect_desc(eff), f"confidence {eff.confidence:.2f}"]
                if eff.params:
                    tip.append(_clip(json.dumps(eff.params, ensure_ascii=False), 120))
                opacity = 0.35 + 0.6 * min(1.0, max(0.0, eff.confidence))
                parts.append(f'<rect class="fx" data-tip="{_tip(tip)}" x="{_f(x)}" '
                             f'y="3" width="{_f(w)}" height="12" rx="3" fill-opacity="{opacity:.2f}"/>')
            lanes.append(_Lane(etype.replace("_", " "), 18, "".join(parts), f"{len(by_type[etype])} {etype} effects"))
        return lanes

    def text_lane(self) -> Optional[_Lane]:
        overlays = sorted(self.edit.overlays, key=lambda o: o.start)
        if not overlays:
            return None
        rows: list[float] = []
        placed = []
        for ov in overlays:
            row = next((r for r, end in enumerate(rows) if end <= ov.start + 1e-9), None)
            if row is None:
                row = len(rows)
                rows.append(0.0)
            rows[row] = ov.end
            placed.append((row, ov))
        n_rows = min(len(rows), 6)
        parts = []
        for row, ov in placed:
            y = 2 + min(row, n_rows - 1) * 18
            x, w = self.span(ov.start, ov.end, 4.0)
            shown = ov.text if ov.text else ("(unread text)" if ov.text is None else "")
            tip = [f"text {ov.id}".strip(), f"“{_clip(shown, 160)}”", f"{ov.start:.2f}–{ov.end:.2f}s",
                   f"{ov.position} · {ov.animation} · {ov.role}"]
            if ov.words:
                tip.append(f"{len(ov.words)} timed words")
            chars = int((w - 8) // 6.5)
            label = (f'<text class="in" x="{_f(x + 5)}" y="{y + 12}">{_e(_clip(shown, chars))}</text>'
                     if chars >= 3 and shown else "")
            parts.append(f'<g data-tip="{_tip(tip)}"><rect class="txt" x="{_f(x)}" y="{y}" '
                         f'width="{_f(w)}" height="16" rx="3"/>{label}</g>')
        return _Lane("text", 4 + 18 * n_rows, "".join(parts))

    def beats(self) -> Optional[_Lane]:
        music = self.edit.music
        if music is None or not (music.beats or music.drops or music.onsets):
            return None
        down = {round(t, 3) for t in music.downbeats}
        parts = [f'<circle class="onset" cx="{_f(self.x(t))}" cy="23" r="1.6"/>' for t in music.onsets]
        # Downbeats are tall ticks; they are drawn even if missing from the beat list.
        ticks = {round(t, 3): False for t in music.beats} | {t: True for t in down}
        for t, is_down in sorted(ticks.items()):
            x = _f(self.x(t))
            y1 = 4 if is_down else 10
            parts.append(f'<line class="{"down" if is_down else "beat"}" x1="{x}" x2="{x}" y1="{y1}" y2="20"/>')
        for drop in music.drops:
            x = self.x(drop.time)
            tip = f"drop at {drop.time:.2f}s\nstrength {drop.strength:.2f}"
            parts.append(f'<path class="drop" data-tip="{_e(tip)}" d="M{_f(x - 6)},0 L{_f(x + 6)},0 L{_f(x)},9 Z"/>')
        note = "beats (tall = downbeat, dots = onsets, triangles = drops)"
        if music.bpm is not None:
            note = f"{music.bpm:.1f} BPM · " + note
        return _Lane("beats", 26, "".join(parts), note)

    def curve_lane(self, label: str, values: Sequence[float] | np.ndarray, t0: float, dt: float,
                   cls: str, unit: str = "", height: int = 44) -> Optional[_Lane]:
        vals = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if vals.size == 0 or dt <= 0:
            return None
        lo, hi = float(vals.min()), float(vals.max())
        # Decimate to <= 2 samples per pixel with a block maximum, so short peaks stay visible.
        stride = max(1, int(1.0 / (2.0 * dt * self.pps)))
        if stride > 1:
            pad = (-vals.size) % stride
            vals = np.concatenate([vals, np.full(pad, vals[-1])]).reshape(-1, stride).max(axis=1)
        span = hi - lo if hi > lo else 1.0
        top, bottom = 4.0, height - 4.0
        ys = bottom - (vals - lo) / span * (bottom - top)
        xs = self.x(t0) + np.arange(vals.size) * dt * stride * self.pps
        line = " ".join(f"{_f(a)},{_f(b)}" for a, b in zip(xs, ys))
        area = f"M{_f(xs[0])},{_f(bottom)} L{line.replace(' ', ' L')} L{_f(xs[-1])},{_f(bottom)} Z"
        data = ",".join(f"{v:.4g}" for v in vals)
        body = (f'<g class="curve" data-name="{_e(label)}" data-unit="{_e(unit)}" data-t0="{t0:.6g}" '
                f'data-dt="{dt * stride:.6g}" data-vals="{data}">'
                f'<rect class="hit" x="{_f(xs[0])}" y="0" width="{_f(max(1.0, xs[-1] - xs[0]))}" height="{height}"/>'
                f'<path class="area {cls}" d="{area}"/><polyline class="line {cls}" points="{line}"/>'
                f'<line class="guide" x1="0" x2="0" y1="0" y2="{height}" visibility="hidden"/></g>')
        note = f"{label}: min {lo:.3g}, max {hi:.3g}"
        return _Lane(label, height, body, note)

    # ---- assembly --------------------------------------------------------------

    def lanes(self) -> list[_Lane]:
        lanes = [self.ruler(), self.sections(), self.shots(), self.transitions(), *self.effect_lanes(),
                 self.text_lane(), self.beats()]
        music = self.edit.music
        if music is not None and music.energy is not None:
            c = music.energy
            lanes.append(self.curve_lane("energy", c.values, c.t0, c.dt, "k1"))
        stats = self.edit.stats
        if stats is not None and stats.pacing is not None:
            c = stats.pacing
            lanes.append(self.curve_lane("pacing", c.values, c.t0, c.dt, "k2", " cuts/s"))
        for name, (values, dt) in self.signals.items():
            lanes.append(self.curve_lane(str(name), values, 0.0, float(dt), "k7", height=36))
        return [lane for lane in lanes if lane is not None]

    def render(self) -> str:
        if self.duration <= 0:
            return '<p class="empty">This edit has no timeline yet.</p>'
        lanes = self.lanes()
        gap = 6
        total = sum(l.height for l in lanes) + gap * (len(lanes) - 1)
        labels, bodies, grid = [], [], []
        y = 0
        for k, lane in enumerate(lanes):
            note = f' data-tip="{_e(lane.note)}"' if lane.note else ""
            labels.append(f'<text class="lane"{note} x="{LABEL_WIDTH - 10}" y="{y + lane.height / 2 + 4:.0f}" '
                          f'text-anchor="end">{_e(lane.label)}</text>')
            if k % 2 == 1:
                bodies.append(f'<rect class="stripe" x="0" y="{y - gap / 2:.0f}" width="{self.width}" '
                              f'height="{lane.height + gap}"/>')
            bodies.append(f'<g transform="translate(0,{y})">{lane.body}</g>')
            y += lane.height + gap
        step = next((s for s in (0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300) if s * self.pps >= 56), 600)
        t = 0.0
        while t <= self.duration + 1e-9:
            grid.append(f'<line class="grid" x1="{_f(self.x(t))}" x2="{_f(self.x(t))}" y1="22" y2="{total}"/>')
            t += step
        return (f'<div class="tl"><svg class="labels" width="{LABEL_WIDTH}" height="{total}" '
                f'viewBox="0 0 {LABEL_WIDTH} {total}" role="img" aria-label="lane names">{"".join(labels)}</svg>'
                f'<div class="scroll"><svg class="lanes" width="{self.width}" height="{total}" '
                f'viewBox="0 0 {self.width} {total}" data-pps="{self.pps:.6g}" data-x0="8" role="img" '
                f'aria-label="edit timeline">{"".join(grid)}{"".join(bodies)}</svg></div></div>')


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    if n <= 1:
        return ""
    return text if len(text) <= n else text[: n - 1] + "…"


def _shot_tip(shot: Shot) -> str:
    lines = [f"shot {shot.id}".strip(), f"{shot.start:.2f}–{shot.end:.2f}s ({shot.duration:.2f}s) · "
             f"frames {shot.frame_start}–{shot.frame_end}"]
    beat = []
    if shot.beats is not None:
        beat.append(f"{shot.beats:g} beat" + ("" if shot.beats == 1 else "s"))
    if shot.on_beat is not None:
        beat.append("on-beat" if shot.on_beat else "off-beat")
    if shot.beat_offset is not None:
        beat.append(f"offset {shot.beat_offset:+.3f}s")
    if beat:
        lines.append(" · ".join(beat))
    if shot.source_group:
        lines.append(f"source group {shot.source_group}")
    if shot.effects:
        lines.append("effects: " + ", ".join(e.type for e in shot.effects))
    if shot.motion is not None:
        mo = shot.motion
        lines.append(f"camera {mo.camera} · motion {mo.magnitude:.3f} · zoom ×{mo.zoom_ratio:.2f}")
    if shot.look is not None:
        lk = shot.look
        lines.append(f"look: brightness {lk.brightness:.2f} · contrast {lk.contrast:.2f} · "
                     f"saturation {lk.saturation:.2f}" + (" · B&W" if lk.bw else ""))
    if shot.layout is not None and (shot.layout.letterbox_top or shot.layout.letterbox_bottom or shot.layout.blur_pad):
        lo = shot.layout
        lines.append(f"layout: letterbox {lo.letterbox_top:.2f}/{lo.letterbox_bottom:.2f}"
                     + (" · blur pad" if lo.blur_pad else ""))
    lines.append(f"confidence {shot.confidence:.2f}")
    return "\n".join(lines)


def _transition_tip(t: Transition) -> str:
    lines = [f"transition {t.id} · {t.kind}".replace("  ", " "),
             f"{t.start:.2f}–{t.end:.2f}s · {t.from_shot or 'start'} → {t.to_shot or 'end'}"]
    if t.color is not None:
        lines.append("colour #{:02x}{:02x}{:02x}".format(*(min(255, max(0, c)) for c in t.color)))
    if t.on_beat is not None:
        lines.append(("on-beat" if t.on_beat else "off-beat")
                     + (f" ({t.beat_offset:+.3f}s)" if t.beat_offset is not None else ""))
    lines.append(f"confidence {t.confidence:.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- other sections


def _tiles(edit: Edit, m: EditMetrics) -> str:
    """Stat tiles: label, value and an optional caption with secondary facts."""
    media = edit.media

    def fmt(x: Optional[float], spec: str, suffix: str = "") -> str:
        return "–" if x is None else f"{x:{spec}}{suffix}"

    size = f"{media.width}×{media.height}" if media.width and media.height else ""
    tempo_caption = f"{m.n_beats} beats" if m.n_beats else ""
    tiles = [
        ("Duration", f"{m.duration:.2f}s", f"{media.fps:g} fps"),
        ("Format", media.aspect or size or "–", size if media.aspect else ""),
        ("Tempo", fmt(m.bpm, ".1f", " BPM"), tempo_caption),
        ("Shots", str(m.n_shots), f"median {m.median_shot:.2f}s" if m.median_shot is not None else ""),
        ("Mean shot", fmt(m.mean_shot, ".2f", "s"), ""),
        ("Cuts / min", fmt(m.cuts_per_minute, ".1f"), ""),
        ("Cuts on beat", "–" if m.on_beat_ratio is None else f"{100 * m.on_beat_ratio:.0f}%", ""),
        ("Hook", fmt(m.hook, ".2f", "s"), f"first cut {m.first_cut:.2f}s" if m.first_cut is not None else ""),
    ]
    return "".join(
        f'<div class="tile"><div class="tl-label">{_e(k)}</div><div class="tl-value">{_e(v)}</div>'
        + (f'<div class="tl-cap">{_e(c)}</div>' if c else "") + "</div>"
        for k, v, c in tiles)


def _patterns(edit: Edit) -> str:
    if not edit.patterns:
        return '<p class="empty">No patterns inferred.</p>'
    rows = []
    for p in sorted(edit.patterns, key=lambda p: -p.support):
        params = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in p.params.items())
        span = f"{p.start:.2f}–{p.end:.2f}s" if p.start is not None and p.end is not None else ""
        pct = max(0.0, min(1.0, p.support))
        meta = " · ".join(x for x in (params and f"<code>{_e(_clip(params, 120))}</code>", _e(span)) if x)
        rows.append(
            f'<li><div class="p-head"><code class="rule">{_e(p.rule)}</code>'
            f'<span class="meter" role="meter" aria-valuenow="{pct:.2f}" aria-valuemin="0" aria-valuemax="1">'
            f'<span style="width:{100 * pct:.0f}%"></span></span><span class="pct">{100 * p.support:.0f}%</span></div>'
            + (f'<div class="p-desc">{_e(p.description)}</div>' if p.description else "")
            + (f'<div class="p-meta">{meta}</div>' if meta else "") + "</li>")
    return f'<ul class="patterns">{"".join(rows)}</ul>'


def _mix(m: EditMetrics) -> str:
    groups = [("Effects", m.effect_counts, "k1"), ("Transitions", m.transition_counts, "k2")]
    out = []
    for title, counts, cls in groups:
        if not counts:
            out.append(f'<h3>{title}</h3><p class="empty">None detected.</p>')
            continue
        top = max(counts.values())
        rows = "".join(
            f'<div class="bar-row" data-tip="{_e(f"{k}: {v}")}"><span class="bar-k">{_e(k)}</span>'
            f'<span class="bar-track"><span class="bar-fill {cls}" style="width:{100 * v / top:.1f}%"></span></span>'
            f'<span class="bar-v">{v}</span></div>'
            for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        out.append(f"<h3>{title}</h3>{rows}")
    if m.n_cuts:
        out.append(f'<p class="note">plus {m.n_cuts} hard cut{"s" if m.n_cuts != 1 else ""}</p>')
    return "".join(out)


_HL_LINE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)")
_HL_TOKEN = re.compile(
    r'(?P<str>"(?:[^"\\\n]|\\.)*")'
    r"|(?P<col>#[0-9A-Fa-f]{6}(?![0-9A-Za-z_]))"
    r"|(?P<com>#[^\n]*)"
    r"|(?P<num>(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[A-Za-z]*)"
)
_HL_CLASS = {"str": "hs", "col": "hn", "com": "hm", "num": "hn"}


def _highlight(src: str) -> str:
    """Escape EditScript and wrap statement keywords, strings, numbers and comments in spans."""
    out = []
    for line in src.split("\n"):
        pieces, pos = [], 0
        m = _HL_LINE.match(line)
        if m:
            pieces.append(_e(m.group(1)) + f'<span class="hk">{_e(m.group(2))}</span>')
            pos = m.end()
        for tok in _HL_TOKEN.finditer(line, pos):
            pieces.append(_e(line[pos:tok.start()]))
            pieces.append(f'<span class="{_HL_CLASS[tok.lastgroup]}">{_e(tok.group())}</span>')
            pos = tok.end()
        pieces.append(_e(line[pos:]))
        out.append("".join(pieces))
    return "\n".join(out)


# --------------------------------------------------------------------------- page

_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--stripe:#f3f2ee;--shot0:#ecebe6;--shot1:#e1e0d9;
--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#008300;--c7:#6250d6;--c8:#e34948;
--track:#cde2fb;--tip:#0b0b0b;--tipink:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--stripe:#20201e;--shot0:#2a2a28;--shot1:#333331;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;
--c6:#008300;--c7:#9085e9;--c8:#e66767;--track:#184f95;--tip:#f0efec;--tipink:#0b0b0b}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--stripe:#20201e;--shot0:#2a2a28;
--shot1:#333331;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300;
--c7:#9085e9;--c8:#e66767;--track:#184f95;--tip:#f0efec;--tipink:#0b0b0b}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 48px}
header .eyebrow{color:var(--muted);font-size:12px;letter-spacing:.06em;text-transform:uppercase}
h1{font-size:22px;margin:2px 0 4px;overflow-wrap:anywhere}
h2{font-size:15px;margin:0 0 12px}
h3{font-size:13px;margin:12px 0 6px;color:var(--ink2)}
.sub{color:var(--ink2);margin:0 0 16px;white-space:pre-line}
.tiles{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.tile{flex:1 1 120px;min-width:0;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.tl-label{font-size:12px;color:var(--muted)}
.tl-value{font-size:20px;font-weight:600}
.tl-cap{font-size:12px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:16px}
.grid2{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px}
.grid2>.card{margin-bottom:0}
@media (max-width:760px){.grid2{grid-template-columns:minmax(0,1fr)}.tl-value{font-size:17px}}
.tl{display:flex;align-items:flex-start}
.labels{flex:0 0 auto}
.scroll{overflow-x:auto;flex:1 1 auto;min-width:0;border-left:1px solid var(--axis)}
svg text{font:11px system-ui,-apple-system,"Segoe UI",sans-serif}
.lane{fill:var(--ink2)}.axis{fill:var(--muted)}.tick{stroke:var(--axis)}
.grid{stroke:var(--grid);stroke-width:1}.stripe{fill:var(--stripe)}
.lbl{fill:var(--ink2)}.in{fill:var(--ink);pointer-events:none}
.sec{fill-opacity:.18}.c1{fill:var(--c1)}.c2{fill:var(--c2)}.c3{fill:var(--c3)}.c4{fill:var(--c4)}
.c5{fill:var(--c5)}.c6{fill:var(--c6)}.c7{fill:var(--c7)}.c8{fill:var(--c8)}
.shot.s0{fill:var(--shot0)}.shot.s1{fill:var(--shot1)}
.beatdot{stroke:var(--surface);stroke-width:2}.beatdot.on{fill:var(--c3)}
.beatdot.off{fill:var(--surface);stroke:var(--c3);stroke-width:2}
.cut{stroke:var(--muted);stroke-width:1.5}.hit{fill:transparent}
.tr{fill:var(--c2);fill-opacity:.85}.fx{fill:var(--c1)}.txt{fill:var(--c7);fill-opacity:.22;stroke:var(--c7);stroke-opacity:.6}
.beat{stroke:var(--muted);stroke-width:1}.down{stroke:var(--ink);stroke-width:1.5}.onset{fill:var(--muted)}
.drop{fill:var(--c8)}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.area{stroke:none;fill-opacity:.1}
.line.k1{stroke:var(--c1)}.area.k1{fill:var(--c1)}.line.k2{stroke:var(--c2)}.area.k2{fill:var(--c2)}
.line.k7{stroke:var(--c7)}.area.k7{fill:var(--c7)}
.guide{stroke:var(--ink2);stroke-width:1}
[data-tip]{cursor:default}
.hint,.note,.empty{color:var(--muted);font-size:12px;margin:8px 0 0}
.patterns{list-style:none;margin:0;padding:0}
.patterns li{padding:8px 0;border-top:1px solid var(--grid)}.patterns li:first-child{border-top:0;padding-top:0}
.p-head{display:flex;align-items:center;gap:10px}
.rule{font-weight:600;overflow-wrap:anywhere;flex:1 1 auto}
.meter{flex:0 0 96px;height:6px;border-radius:3px;background:var(--track);overflow:hidden}
.meter span{display:block;height:100%;background:var(--c1);border-radius:3px}
.pct{flex:0 0 40px;text-align:right;font-variant-numeric:tabular-nums;color:var(--ink2)}
.p-desc{color:var(--ink2)}.p-meta{color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.bar-row{display:flex;align-items:center;gap:8px;margin:3px 0}
.bar-k{flex:0 0 84px;color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-track{flex:1 1 auto;height:10px}
.bar-fill{display:block;height:100%;border-radius:0 4px 4px 0}.bar-fill.k1{background:var(--c1)}
.bar-fill.k2{background:var(--c2)}
.bar-v{flex:0 0 32px;text-align:right;font-variant-numeric:tabular-nums}
details.card>summary{cursor:pointer;font-weight:600}
pre{margin:12px 0 0;overflow:auto;max-height:520px;font:12px/1.45 ui-monospace,"Cascadia Mono",Consolas,monospace;
color:var(--ink2);white-space:pre}
.hk{color:var(--c7);font-weight:600}.hs{color:var(--c3)}.hn{color:var(--c1)}.hm{color:var(--muted)}
#tip{position:fixed;z-index:10;max-width:340px;padding:8px 10px;border-radius:8px;background:var(--tip);
color:var(--tipink);font-size:12px;line-height:1.45;white-space:pre-line;pointer-events:none;
box-shadow:0 4px 16px rgba(0,0,0,.18)}
#tip strong{display:block;font-weight:600}
"""

_JS = """
(() => {
  const tip = document.getElementById('tip');
  const show = (text, ev) => {
    const [head, ...rest] = text.split('\\n');
    tip.replaceChildren();
    const strong = document.createElement('strong');
    strong.textContent = head;
    tip.append(strong, document.createTextNode(rest.join('\\n')));
    tip.hidden = false;
    move(ev);
  };
  const move = (ev) => {
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
  };
  const hide = () => { tip.hidden = true; };
  const lanes = document.querySelector('svg.lanes');
  const pps = lanes ? parseFloat(lanes.dataset.pps) : 1, x0 = lanes ? parseFloat(lanes.dataset.x0) : 0;
  document.addEventListener('pointermove', (ev) => {
    const curve = ev.target.closest && ev.target.closest('g.curve');
    document.querySelectorAll('g.curve line.guide').forEach(g => {
      if (!curve || g.parentNode !== curve) g.setAttribute('visibility', 'hidden');
    });
    if (curve) {
      const px = ev.clientX - lanes.getBoundingClientRect().left;  // the SVG is drawn 1:1
      const vals = curve.dataset.vals.split(',').map(Number);
      const t0 = parseFloat(curve.dataset.t0), dt = parseFloat(curve.dataset.dt);
      const i = Math.max(0, Math.min(vals.length - 1, Math.round(((px - x0) / pps - t0) / dt)));
      const t = t0 + i * dt, gx = x0 + t * pps;
      const guide = curve.querySelector('line.guide');
      guide.setAttribute('x1', gx); guide.setAttribute('x2', gx); guide.setAttribute('visibility', 'visible');
      show(curve.dataset.name + '\\n' + vals[i].toPrecision(3) + curve.dataset.unit + ' at ' + t.toFixed(2) + 's', ev);
      return;
    }
    const el = ev.target.closest && ev.target.closest('[data-tip]');
    if (el) show(el.dataset.tip, ev); else hide();
  });
  document.documentElement.addEventListener('pointerleave', hide);
  addEventListener('scroll', hide, {passive: true});
})();
"""


def report_html(edit: Edit, ctx: Optional["AnalysisContext"] = None, signals: Optional[Signals] = None) -> str:
    """The report as an HTML string (see :func:`write_report`)."""
    m = edit_metrics(edit)
    media = edit.media
    name = Path(media.source).name if media.source else "Untitled edit"
    sub = []
    if media.source and media.source != name:
        sub.append(media.source)
    facts = [f"{m.duration:.2f}s", f"{media.fps:g} fps"]
    if media.has_audio:
        facts.append("audio")
    if m.drops:
        facts.append("drop at " + ", ".join(f"{t:.2f}s" for t in m.drops[:3]))
    if m.text_coverage:
        facts.append(f"text on screen {100 * m.text_coverage:.0f}%")
    sub.append(" · ".join(facts))
    timeline = _Timeline(edit, ctx, signals).render()
    dsl = _highlight(to_dsl(edit, compact=True))
    data = _e(edit.to_json(indent=2))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(name)} · editast</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
<div class="eyebrow">editast · edit report</div>
<h1>{_e(name)}</h1>
<p class="sub">{_tip(sub)}</p>
<div class="tiles">{_tiles(edit, m)}</div>
</header>
<section class="card">
<h2>Timeline</h2>
{timeline}
<p class="hint">Scroll sideways for the full edit; hover any block, tick or curve for details.</p>
</section>
<div class="grid2">
<section class="card"><h2>Patterns</h2>{_patterns(edit)}</section>
<section class="card"><h2>Effect &amp; transition mix</h2>{_mix(m)}</section>
</div>
<details class="card" style="margin-top:16px"><summary>EditScript (compact)</summary><pre>{dsl}</pre></details>
<details class="card"><summary>AST JSON</summary><pre>{data}</pre></details>
</main>
<div id="tip" role="tooltip" hidden></div>
<script>{_JS}</script>
</body>
</html>
"""


def write_report(edit: Edit, path: str | Path, ctx: Optional["AnalysisContext"] = None,
                 signals: Optional[Signals] = None) -> Path:
    """Write a self-contained HTML report for ``edit`` to ``path``.

    ``ctx`` adds a thumbnail per shot; ``signals`` maps a name to
    ``(values, dt_seconds)`` drawn as extra sparkline lanes (sample ``i`` at ``i * dt``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_html(edit, ctx, signals), encoding="utf-8")
    return path
