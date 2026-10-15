"""Synthetic benchmark cases: random viral-style edits with a known AST.

:func:`generate_case` builds a random *truth* Edit that follows the
conventions of :mod:`editast.ast.nodes`, renders it through the compiler over
procedural footage (:class:`~editast.render.sources.SyntheticSource`) with a
click-track song, and saves ``<name>.mp4`` next to ``<name>.truth.json``.
Analysing the mp4 and scoring the result against the truth
(:func:`editast.bench.evaluate.compare`) measures how well the analyzers
recover an edit.

How a case is built (:func:`build_case`):

1. **Music grid** — BPM 90-160, first beat in the first ~0.25 s, 8-14 s long
   ending on a beat, a drop on a (down)beat at 40-60 % of the runtime.
2. **Cuts** — walking back from the drop every 2 beats (sometimes 1 or 4)
   down to a >= 1 s hook shot, and forward from it every beat (sometimes two
   half-beat cuts or a 2-beat hold); one or two cuts are nudged off the beat.
3. **Transitions** — mostly hard cuts, plus flashes (often on the drop),
   dissolves and dips to black before the drop; some cases fade in from black
   (and a few fade out). Each seed forces one of dissolve / fade / flash so a
   suite covers all of them.
4. **Camera** — ~45 % of shots get a camera move of 12-30 % of the frame
   width over the shot (well above the 5 % Pan threshold); they carry a
   ``Pan(params={"source": "camera"})`` in the truth. Other shots drift
   by < 1.5 % of the width, so no pan is expected there.
5. **Effects** — zoom (ease_out / punch / linear, ratio 1.15-1.4), shake
   (amplitude 0.006-0.02, 5-10 Hz, mostly right after the drop), speed
   (slow-mo 0.3-0.5x or ramps, only on moving shots and never over a whole
   shot, because a constant rate is indistinguishable from a slower camera),
   freeze (0.2-0.5 s on moving shots, without zoom or shake so the held frame
   is really static), in-shot flash on a beat, rgb_split, a short invert,
   and one black & white shot. Pixel effects avoid transition spans.
6. **Text** — a hook title in the first ~1.5 s and a word-by-word caption,
   with bboxes measured from the renderer's own text layout.

Every time in the truth is quantised to the frame grid, so it describes what
can actually be observed in the rendered video.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from editast.ast.nodes import (
    AudioSection,
    BBox,
    Curve,
    Drop,
    Edit,
    EditStats,
    EffectBase,
    Flash,
    Freeze,
    Invert,
    Look,
    MediaInfo,
    Music,
    Pan,
    Pattern,
    RGBSplit,
    Section,
    Shake,
    Shot,
    Speed,
    TextOverlay,
    TextPosition,
    Timeline,
    Transition,
    Word,
    Zoom,
    assign_ids,
    shot_id,
)
from editast.media import aspect_string, write_wav
from editast.render.compiler import render_edit, shot_geometry, source_positions, text_box
from editast.render.sources import SyntheticSource
from editast.testing import synth

HOOKS = ("WAIT FOR IT", "POV: YOU HEARD IT FIRST", "POV: 0-100 IN 3S", "WATCH TILL THE END", "SOUND ON")
CAPTIONS = (
    ("THIS", "CAR", "IS", "INSANE"),
    ("NO", "WAY", "THIS", "IS", "REAL"),
    ("BUILT", "NOT", "BOUGHT"),
    ("PURE", "SOUND", "PURE", "SPEED"),
)
BEAT_TOLERANCE = 0.07  # seconds; same default as AnalysisConfig.beat_tolerance
MIN_SHOT = 0.25  # seconds
PAN_RANGE = (0.12, 0.30)  # camera travel of moving shots, fraction of width
MAX_DRIFT = 0.012  # camera travel of "static" shots, fraction of width (< 2 % even under a 1.4x zoom)
MAX_SPEED = 0.035  # camera speed cap, fraction of width per source frame


@dataclass
class BenchCase:
    """A truth Edit plus everything needed to render it."""

    edit: Edit
    motion: dict[int, tuple[float, float]]  # shot index -> camera px / source frame
    audio: np.ndarray
    sr: int
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


class _Grid:
    """Beat grid + frame quantisation."""

    def __init__(self, bpm: float, beat0: float, fps: float) -> None:
        self.bpm, self.beat0, self.fps = bpm, beat0, fps
        self.period = 60.0 / bpm

    def t(self, k: float) -> float:
        return self.beat0 + k * self.period

    def q(self, t: float) -> float:
        return round(t * self.fps) / self.fps

    def offset(self, t: float) -> float:
        """Signed seconds from the nearest beat."""
        k = round((t - self.beat0) / self.period)
        return t - self.t(k)


def _position(cy: float) -> TextPosition:
    for limit, name in ((0.2, "top"), (0.4, "upper"), (0.6, "center"), (0.8, "lower")):
        if cy < limit:
            return name  # type: ignore[return-value]
    return "bottom"


def _union(boxes: list[BBox]) -> BBox:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _free(spans: list[tuple[float, float]], start: float, end: float, margin: float) -> bool:
    return all(end + margin <= s or start >= e + margin for s, e in spans)


# --------------------------------------------------------------------------- structure


def _plan_cuts(rng: np.random.Generator, g: _Grid, duration: float, kd: int) -> list[float]:
    """Cut times (frame-quantised) around the drop at beat ``kd``."""
    ks: list[float] = [kd]
    k: float = kd
    while True:
        step = float(rng.choice([2, 2, 2, 2, 1, 4]))
        if g.t(k - step) < 1.0:
            break
        k -= step
        ks.append(k)
    half_ok = 0.5 * g.period >= MIN_SHOT
    tail = max(g.period, 0.5)
    k = kd
    done = False
    while not done:
        r = rng.random()
        steps = [0.5, 0.5] if (half_ok and r < 0.2) else ([2.0] if r > 0.9 else [1.0])
        for st in steps:
            if g.t(k + st) > duration - tail + 1e-9:
                done = True
                break
            k += st
            ks.append(k)
    cuts = sorted({g.q(g.t(x)) for x in ks})
    # Nudge one or two cuts off the beat (never the drop cut).
    drop_t = g.q(g.t(kd))
    candidates = [i for i, c in enumerate(cuts) if c != drop_t]
    for i in rng.permutation(candidates)[: int(rng.integers(1, 3))]:
        shift = float(rng.choice([-1.0, 1.0])) * max(0.11, rng.uniform(0.22, 0.35) * g.period)
        new = g.q(cuts[i] + shift)
        prev = cuts[i - 1] if i > 0 else 0.0
        nxt = cuts[i + 1] if i + 1 < len(cuts) else duration
        if new - prev >= MIN_SHOT and nxt - new >= MIN_SHOT and new >= 1.0:
            cuts[i] = new
    return cuts


def _plan_transitions(rng: np.random.Generator, seed: int, g: _Grid, bounds: list[float],
                      drop_t: float) -> dict[int, tuple[str, float, float]]:
    """Non-cut transitions keyed by the index of the incoming shot -> (kind, start, end)."""
    out: dict[int, tuple[str, float, float]] = {}

    def span(kind: str, i: int) -> Optional[tuple[float, float]]:
        b = bounds[i]
        prev_len, next_len = b - bounds[i - 1], bounds[i + 1] - b
        if kind == "flash":
            return b - 2 / g.fps, b + int(rng.choice([2, 3])) / g.fps
        d = g.q(rng.uniform(0.3, 0.5))
        if min(prev_len, next_len) < d / 2 + 0.15:
            return None
        return g.q(b - d / 2), g.q(b + d / 2)

    for i in range(1, len(bounds) - 1):
        b = bounds[i]
        if abs(b - drop_t) < 1e-6:
            kind = "flash" if rng.random() < 0.6 else "cut"
        elif b < drop_t:
            kind = str(rng.choice(["cut", "flash", "dissolve", "fade_black"], p=[0.62, 0.12, 0.14, 0.12]))
        else:
            kind = "flash" if rng.random() < 0.15 else "cut"
        sp = span(kind, i) if kind != "cut" else None
        if sp is not None:
            out[i] = (kind, *sp)
    forced = ("dissolve", "fade_black", "flash")[seed % 3]
    if all(k != forced for k, _, _ in out.values()):
        options = [i for i in range(1, len(bounds) - 1) if i not in out and (forced == "flash" or bounds[i] < drop_t)]
        for i in rng.permutation(options):
            sp = span(forced, int(i))
            if sp is not None:
                out[int(i)] = (forced, *sp)
                break
    return out


# --------------------------------------------------------------------------- effects


def _pick_effects(rng: np.random.Generator, g: _Grid, shots: list[Shot], moving: list[bool],
                  drop_t: float, blocked: list[tuple[float, float]], bw_shot: int) -> None:
    """Sample effects into ``shot.effects`` (spans frame-quantised, inside their shot)."""
    q, fps = g.q, g.fps
    pixel_spans: list[tuple[float, float]] = []
    time_hosts = [i for i, s in enumerate(shots) if moving[i] and s.duration >= 0.7]
    freeze_left = 2
    time_kind: dict[int, str] = {}
    for i in time_hosts:
        r = rng.random()
        if r < 0.5:
            time_kind[i] = "speed"
        elif r < 0.75 and freeze_left:
            time_kind[i] = "freeze"
            freeze_left -= 1
    if time_hosts and not time_kind:
        time_kind[time_hosts[0]] = "speed"

    def pixel_slot(shot: Shot, length: float, anchor: Optional[float] = None) -> Optional[tuple[float, float]]:
        lo, hi = shot.start + 0.1, shot.end - 0.1 - length
        if hi <= lo:
            return None
        start = q(anchor) if anchor is not None and lo <= anchor <= hi else q(rng.uniform(lo, hi))
        end = start + max(1, round(length * fps)) / fps
        if start < shot.start or end > shot.end or not _free(blocked + pixel_spans, start, end, 2 / fps):
            return None
        pixel_spans.append((start, end))
        return start, end

    drop_idx = next(i for i, s in enumerate(shots) if s.end > drop_t + 1e-6)
    for i, shot in enumerate(shots):
        L = shot.duration
        kind = time_kind.get(i)
        effects: list[EffectBase] = []
        if kind == "speed":
            r = round(float(rng.uniform(0.3, 0.5)), 2)
            style = str(rng.choice(["slowmo", "ramp_down", "ramp_up"]))
            if style == "slowmo":
                effects.append(Speed(start=q(shot.start + rng.uniform(0.35, 0.5) * L), end=shot.end,
                                     kind="slowmo", rate_from=r, rate_to=r))
            elif style == "ramp_down":
                effects.append(Speed(start=q(shot.start + rng.uniform(0.25, 0.4) * L), end=shot.end,
                                     kind="ramp_down", rate_from=1.0, rate_to=r, easing="ease_out"))
            else:
                effects.append(Speed(start=shot.start, end=q(shot.start + rng.uniform(0.5, 0.7) * L),
                                     kind="ramp_up", rate_from=r, rate_to=1.0, easing="ease_in"))
        elif kind == "freeze":
            d = q(min(rng.uniform(0.2, 0.5), 0.5 * L))
            if rng.random() < 0.6:
                effects.append(Freeze(start=shot.end - d, end=shot.end))
            else:
                s0 = q(shot.start + 0.3 * L)
                effects.append(Freeze(start=s0, end=s0 + d))
        if kind != "freeze" and rng.random() < (0.6 if i == drop_idx else 0.3):
            effects.append(_zoom(rng, g, shot))
        near_drop = drop_t - 1e-6 <= shot.start < drop_t + 2.5
        if kind is None and L >= 0.3 and rng.random() < (0.6 if near_drop else 0.12):
            effects.append(_shake(rng, g, shot))
        if L >= 0.6 and rng.random() < 0.15:
            k0 = math.floor((shot.start - g.beat0) / g.period)
            inner = [g.t(k) for k in range(k0, k0 + int(L / g.period) + 2) if shot.start + 0.15 < g.t(k) < shot.end - 0.2]
            slot = pixel_slot(shot, int(rng.choice([3, 4, 5])) / fps, float(rng.choice(inner)) if inner else None)
            if slot:
                effects.append(Flash(start=slot[0], end=slot[1], intensity=round(float(rng.uniform(0.6, 1.0)), 2)))
        if i != bw_shot and rng.random() < (0.3 if shot.start >= drop_t - 1e-6 else 0.1):
            slot = pixel_slot(shot, q(rng.uniform(0.2, 0.4)))
            if slot:
                effects.append(RGBSplit(start=slot[0], end=slot[1], offset=round(float(rng.uniform(0.01, 0.02)), 4)))
        shot.effects = effects

    # Guarantees: every case has a zoom, a shake and one short invert after the drop.
    if not any(isinstance(e, Zoom) for s in shots for e in s.effects):
        host = max((i for i in range(len(shots)) if time_kind.get(i) != "freeze"), key=lambda i: shots[i].duration)
        shots[host].effects.append(_zoom(rng, g, shots[host]))
    if not any(isinstance(e, Shake) for s in shots for e in s.effects):
        hosts = [i for i in range(drop_idx, len(shots)) if i not in time_kind and shots[i].duration >= 0.3]
        hosts = hosts or [i for i in range(len(shots)) if i not in time_kind and shots[i].duration >= 0.3]
        if hosts:
            shots[hosts[0]].effects.append(_shake(rng, g, shots[hosts[0]]))
    for i in rng.permutation(range(drop_idx, len(shots))):
        shot = shots[int(i)]
        if shot.duration >= 0.4 and int(i) not in time_kind:
            slot = pixel_slot(shot, int(rng.choice([3, 4, 5, 6])) / fps)
            if slot:
                shot.effects.append(Invert(start=slot[0], end=slot[1]))
                break


def _zoom(rng: np.random.Generator, g: _Grid, shot: Shot) -> Zoom:
    easing = str(rng.choice(["ease_out", "punch", "linear"]))
    end = shot.end
    if shot.duration >= 0.8 and rng.random() < 0.4:
        end = g.q(shot.start + rng.uniform(0.5, 0.9) * shot.duration)
    return Zoom(start=shot.start, end=end, scale_from=1.0, scale_to=round(float(rng.uniform(1.15, 1.4)), 3),
                easing=easing, digital=True)  # type: ignore[arg-type]


def _shake(rng: np.random.Generator, g: _Grid, shot: Shot) -> Shake:
    end = shot.end if shot.duration <= 1.0 else g.q(shot.start + rng.uniform(0.5, 1.0))
    return Shake(start=shot.start, end=end, amplitude=round(float(rng.uniform(0.006, 0.02)), 4),
                 frequency=round(float(rng.uniform(5.0, 10.0)), 1),
                 params={"seed": int(rng.integers(0, 1_000_000))})


def _camera(rng: np.random.Generator, shots: list[Shot], moving: list[bool], size: tuple[int, int],
            fps: float) -> dict[int, tuple[float, float]]:
    """Camera velocities; moving shots also get their (observable) Pan in the truth.

    The Pan describes the content displacement seen in the output: camera travel
    (which follows the shot's speed ramps and freezes) magnified by any digital zoom.
    """
    w, h = size
    motion: dict[int, tuple[float, float]] = {}
    for i, shot in enumerate(shots):
        pos = source_positions(shot, fps)
        travel = float(pos[-1] - pos[0]) if pos.size > 1 else 0.0
        if travel <= 0:
            continue
        scale = shot_geometry(shot, fps, np.arange(shot.frame_start, shot.frame_end), size).scale
        seen = float(np.sum(np.diff(pos) * scale[1:]))  # travel as it appears on screen
        r = rng.random()
        if r < 0.7:
            u = (float(rng.choice([-1.0, 1.0])), 0.0)
        elif r < 0.85:
            u = (0.0, float(rng.choice([-1.0, 1.0])))
        else:
            u = (float(rng.choice([-1.0, 1.0])) * math.sqrt(0.5), float(rng.choice([-1.0, 1.0])) * math.sqrt(0.5))
        if moving[i]:
            frac = min(float(rng.uniform(*PAN_RANGE)), MAX_SPEED * travel)
        else:
            frac = float(rng.uniform(0.003, MAX_DRIFT))
        v = frac * w / travel
        motion[i] = (v * u[0], v * u[1])
        if moving[i]:
            # Content moves opposite to the camera; dx is a fraction of width, dy of height.
            shot.effects.insert(0, Pan(start=shot.start, end=shot.end, dx=round(-v * u[0] * seen / w, 4),
                                       dy=round(-v * u[1] * seen / h, 4), params={"source": "camera"}))
    return motion


# --------------------------------------------------------------------------- text / music / stats


def _overlays(rng: np.random.Generator, g: _Grid, size: tuple[int, int], duration: float) -> list[TextOverlay]:
    q = g.q
    hook_text = str(rng.choice(HOOKS))
    center = (0.5, round(float(rng.uniform(0.22, 0.38)), 3))
    hook_box = text_box(hook_text, center, round(float(rng.uniform(0.045, 0.06)), 4), size)
    hook_end = q(rng.uniform(1.0, 1.5))
    hook = TextOverlay(start=0.0, end=hook_end, text=hook_text, bbox=hook_box, position=_position(center[1]),
                       size=round(hook_box[3], 4), color=(255, 255, 255), outline=True,
                       animation="pop" if rng.random() < 0.3 else "static", role="hook")
    words = list(CAPTIONS[int(rng.integers(len(CAPTIONS)))])
    k0 = math.ceil((hook_end + 0.2 - g.beat0) / g.period)
    while k0 > 0 and g.t(k0 + len(words)) > duration - 0.3:
        k0 -= 1
    center = (0.5, round(float(rng.uniform(0.68, 0.78)), 3))
    line = round(float(rng.uniform(0.05, 0.065)), 4)
    spans = [(q(g.t(k0 + j)), min(duration, q(g.t(k0 + j + 1)))) for j in range(len(words))]
    keep = max(1, sum(1 for s, e in spans if e - s > 0.1))  # only the tail can be cut short by the end
    words, spans = words[:keep], spans[:keep]
    boxes = [text_box(wd, center, line, size) for wd in words]
    box = _union(boxes)
    color = (255, 255, 255) if rng.random() < 0.6 else (255, 221, 0)
    caption = TextOverlay(start=spans[0][0], end=spans[-1][1], text=" ".join(words), bbox=box,
                          position=_position(center[1]), size=round(max(b[3] for b in boxes), 4), color=color,
                          outline=True, animation="word_by_word", role="caption",
                          words=[Word(text=wd, start=s, end=e) for wd, (s, e) in zip(words, spans)])
    return [hook, caption]


def _music(g: _Grid, duration: float, drop_t: float, audio: np.ndarray, sr: int) -> Music:
    hop = int(0.1 * sr)
    n = max(1, audio.size // hop)
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1))
    rms /= max(float(rms.max()), 1e-9)
    d = min(n, int(drop_t / 0.1))
    before = float(rms[:d].mean()) if d > 0 else 0.0
    after = float(rms[d:].mean()) if d < n else 0.0
    beats = [round(g.t(k), 4) for k in range(int((duration - g.beat0) / g.period) + 1) if g.t(k) < duration - 1e-6]
    return Music(
        bpm=g.bpm, beat_offset=g.beat0, meter=4, beats=beats, downbeats=beats[::4],
        drops=[Drop(time=round(drop_t, 4), strength=round((after - before) / max(after, 1e-9), 3))],
        sections=[AudioSection(start=0.0, end=round(drop_t, 4), label="build", energy=round(before, 3)),
                  AudioSection(start=round(drop_t, 4), end=duration, label="drop", energy=round(after, 3))],
        energy=Curve(t0=0.05, dt=0.1, values=[round(float(v), 4) for v in rms]),
        has_speech=False,
    )


def edit_stats(edit: Edit) -> EditStats:
    """Pacing summary of an edit (used for the truth; analyzers produce their own)."""
    shots = edit.shots()
    durs = np.array([s.duration for s in shots]) if shots else np.zeros(1)
    total = max(edit.media.duration, shots[-1].end if shots else 0.0)
    cuts = edit.cut_times()
    on = [s.on_beat for s in shots[1:] if s.on_beat is not None]
    effect_counts: dict[str, int] = {}
    for _, eff in edit.effects():
        effect_counts[eff.type] = effect_counts.get(eff.type, 0) + 1
    transition_counts: dict[str, int] = {}
    for tr in edit.transitions():
        transition_counts[tr.kind] = transition_counts.get(tr.kind, 0) + 1
    covered = np.zeros(int(math.ceil(total * 100)) + 1, bool)
    for ov in edit.overlays:
        covered[int(ov.start * 100):int(ov.end * 100)] = True
    n_bins = max(1, int(math.ceil(total)))
    pacing = np.histogram(cuts, bins=n_bins, range=(0.0, float(n_bins)))[0] if cuts else np.zeros(n_bins)
    sections = edit.timeline.sections
    return EditStats(
        n_shots=len(shots),
        n_transitions=sum(1 for t in edit.transitions() if t.kind != "cut"),
        mean_shot_duration=float(durs.mean()), median_shot_duration=float(np.median(durs)),
        min_shot_duration=float(durs.min()), max_shot_duration=float(durs.max()),
        cuts_per_minute=60.0 * len(cuts) / total if total > 0 else 0.0,
        cut_on_beat_ratio=float(np.mean(on)) if on else None,
        hook_duration=sections[0].duration if sections and sections[0].role == "hook" else 0.0,
        first_cut_time=cuts[0] if cuts else None,
        effect_counts=effect_counts, transition_counts=transition_counts,
        text_coverage=float(covered[: int(total * 100)].mean()) if total > 0 else 0.0,
        pacing=Curve(t0=0.5, dt=1.0, values=[float(v) for v in pacing]),
    )


def _patterns(edit: Edit) -> list[Pattern]:
    out = []
    for sec in edit.timeline.sections:
        beats = [s.beats for s in sec.shots() if s.beats is not None]
        if sec.role == "hook" or len(beats) < 2:
            continue
        n = float(np.median(beats))
        n = round(n * 2) / 2
        support = float(np.mean([abs(b - n) < 0.15 for b in beats]))
        label = f"{n:g} beat" + ("" if n == 1 else "s")
        out.append(Pattern(rule="cut_every_n_beats", description=f"cut every {label} in the {sec.role}",
                           start=sec.start, end=sec.end, support=round(support, 3), params={"n": n}))
    return out


# --------------------------------------------------------------------------- public API


def build_case(seed: int, size: tuple[int, int] = (360, 640), fps: float = 30.0,
               duration: Optional[float] = None, sr: int = 22050) -> BenchCase:
    """Build (but do not render) a random truth Edit; see the module docstring."""
    rng = np.random.default_rng(seed)
    bpm = float(round(rng.uniform(90.0, 160.0), 1))
    beat0 = round(float(rng.uniform(0.0, min(0.25, 30.0 / bpm))), 3)
    g = _Grid(bpm, beat0, fps)
    if duration is None:
        total = g.q(g.t(int(round((rng.uniform(8.0, 14.0) - beat0) / g.period))))
    else:
        total = g.q(duration)
    lo = math.ceil((0.4 * total - beat0) / g.period)
    hi = math.floor((0.6 * total - beat0) / g.period)
    cands = [k for k in range(max(lo, 2), hi + 1)] or [max(2, int(round((0.5 * total - beat0) / g.period)))]
    downs = [k for k in cands if k % 4 == 0]
    kd = int(rng.choice(downs or cands))
    drop_exact = g.t(kd)
    drop_t = g.q(drop_exact)

    bounds = [0.0] + _plan_cuts(rng, g, total, kd) + [total]
    shots = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        off = g.offset(start)
        shots.append(Shot(index=i, start=start, end=end, frame_start=int(round(start * fps)),
                          frame_end=int(round(end * fps)), beats=round((end - start) / g.period, 3),
                          on_beat=abs(off) <= BEAT_TOLERANCE, beat_offset=round(off, 4)))
    n = len(shots)

    # Source clips: mostly fresh footage, sometimes a non-adjacent earlier clip again.
    labels: list[str] = []
    for i in range(n):
        reuse = [lab for lab in dict.fromkeys(labels[:-1]) if lab != (labels[-1] if labels else None)]
        if reuse and rng.random() < 0.15:
            labels.append(str(rng.choice(reuse)))
        else:
            labels.append(_label(len(set(labels))))
    for shot, lab in zip(shots, labels):
        shot.source_group = lab

    trans = _plan_transitions(rng, seed, g, bounds, drop_t)
    fade_in = (seed % 2 == 0 or rng.random() < 0.3) and shots[0].duration >= 0.8
    fade_in_end = g.q(rng.uniform(0.3, 0.5)) if fade_in else 0.0
    fade_out_len = g.q(rng.uniform(0.3, 0.5)) if rng.random() < 0.25 else 0.0
    if fade_out_len and shots[-1].duration < fade_out_len + 0.2:
        fade_out_len = 0.0
    blocked = [(s, e) for _, s, e in trans.values()]
    if fade_in:
        blocked.append((0.0, fade_in_end))
    if fade_out_len:
        blocked.append((total - fade_out_len, total))

    moving = [s.duration >= 0.3 and rng.random() < 0.45 for s in shots]
    if not any(m and s.duration >= 0.7 for m, s in zip(moving, shots)):
        hosts = [i for i, s in enumerate(shots) if s.duration >= 0.7] or [int(np.argmax([s.duration for s in shots]))]
        moving[hosts[int(rng.integers(len(hosts)))]] = True
    bw_options = [i for i in range(1, n) if shots[i].end <= drop_t + 1e-6] or list(range(1, n)) or [0]
    bw_shot = int(rng.choice(bw_options))
    shots[bw_shot].look = Look(bw=True)
    _pick_effects(rng, g, shots, moving, drop_t, blocked, bw_shot)
    motion = _camera(rng, shots, moving, size, fps)

    # Timeline: hook | build | drop sections. Items are anchored at the start of the
    # shot they belong to (a transition belongs with its incoming shot).
    items: list[tuple[float, Union[Shot, Transition]]] = []
    if fade_in:
        items.append((0.0, Transition(kind="fade_black", start=0.0, end=fade_in_end, from_shot=None,
                                      to_shot=shot_id(0), color=(0, 0, 0),
                                      on_beat=abs(g.offset(0.0)) <= BEAT_TOLERANCE, beat_offset=round(g.offset(0.0), 4))))
    for i, shot in enumerate(shots):
        if i > 0:
            kind, ts, te = trans.get(i, ("cut", shot.start, shot.start))
            color = {"flash": (255, 255, 255), "fade_black": (0, 0, 0)}.get(kind)
            items.append((shot.start, Transition(kind=kind, start=ts, end=te, from_shot=shot_id(i - 1),  # type: ignore[arg-type]
                                                 to_shot=shot_id(i), color=color, on_beat=shot.on_beat,
                                                 beat_offset=shot.beat_offset)))
        items.append((shot.start, shot))
    if fade_out_len:
        items.append((shots[-1].start, Transition(kind="fade_black", start=total - fade_out_len, end=total,
                                                  from_shot=shot_id(n - 1), to_shot=None, color=(0, 0, 0))))
    sections = _sections(items, shots[0].end, drop_t, total)

    audio = synth.click_track(bpm, total, sr=sr, offset=beat0, drop_at=drop_exact - 1e-6, seed=seed)
    edit = Edit(
        media=MediaInfo(duration=total, fps=fps, width=size[0], height=size[1], n_frames=int(round(total * fps)),
                        analysis_fps=fps, has_audio=True, aspect=aspect_string(*size)),
        music=_music(g, total, drop_exact, audio, sr),
        timeline=Timeline(sections=sections),
        overlays=_overlays(rng, g, size, total),
        provenance={"generator": "editast.bench.generate", "seed": seed,
                    "camera_px_per_frame": {str(k): [round(v[0], 4), round(v[1], 4)] for k, v in motion.items()}},
    )
    edit.patterns = _patterns(edit)
    edit.stats = edit_stats(edit)
    assign_ids(edit)
    return BenchCase(edit=edit, motion=motion, audio=audio, sr=sr, seed=seed,
                     meta={"drop_time": drop_t, "moving": [i for i, m in enumerate(moving) if m]})


def _label(k: int) -> str:
    """Spreadsheet-style labels A..Z, AA.. for source groups."""
    out = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        out = chr(65 + r) + out
    return out


def _sections(items: list[tuple[float, Union[Shot, Transition]]], hook_end: float, drop_t: float,
              total: float) -> list[Section]:
    bounds = [("hook", 0.0, hook_end)]
    if drop_t > hook_end + 1e-6:
        bounds.append(("build", hook_end, drop_t))
    bounds.append(("drop", max(hook_end, drop_t), total))
    sections = []
    for role, start, end in bounds:
        members = [it for anchor, it in items if start - 1e-6 <= anchor < end - 1e-6]
        shot_durs = [s.duration for s in members if isinstance(s, Shot)]
        n_cuts = sum(1 for s in members if isinstance(s, Shot) and s.index > 0)
        sections.append(Section(role=role, start=start, end=end, items=members,  # type: ignore[arg-type]
                                cuts_per_second=round(n_cuts / max(end - start, 1e-6), 4),
                                mean_shot_duration=round(float(np.mean(shot_durs)), 4) if shot_durs else 0.0))
    return [s for s in sections if s.items]


def generate_case(
    seed: int,
    out_dir: Union[str, Path],
    size: tuple[int, int] = (360, 640),
    fps: float = 30.0,
    duration: Optional[float] = None,
) -> tuple[Path, Edit]:
    """Build, render and save one benchmark case; returns ``(mp4 path, truth edit)``.

    Writes ``case_<seed>.mp4`` (with the click-track music) and
    ``case_<seed>.truth.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = build_case(seed, size=size, fps=fps, duration=duration)
    name = f"case_{seed:04d}"
    mp4 = out_dir / f"{name}.mp4"
    case.edit.media.source = str(mp4)
    with tempfile.TemporaryDirectory(prefix="editast_bench_") as tmp:
        wav = write_wav(Path(tmp) / "music.wav", case.audio, case.sr)
        render_edit(case.edit, SyntheticSource(seed, case.motion), mp4, music=wav, size=size, fps=fps)
    case.edit.save(out_dir / f"{name}.truth.json")
    return mp4, case.edit


def generate_suite(out_dir: Union[str, Path], n: int = 5, seed: int = 0, **kwargs: Any) -> list[tuple[Path, Edit]]:
    """``n`` cases with seeds ``seed .. seed + n - 1`` (``kwargs`` go to :func:`generate_case`)."""
    return [generate_case(seed + i, out_dir, **kwargs) for i in range(n)]
