"""Fit edit templates to a new song: from "pick a song" to an Edit AST for it.

A template is the Edit AST of an analysed video. :func:`fit_edit` fits one to a
song:

1. It moves the template onto the song's beat grid with
   :func:`~editast.render.compiler.warp_edit`, at the song's tempo or at half or
   double it, whichever stretches the template least. Times map through the
   detected beats of both (:class:`BeatClock`), so a cut on beat 12 of the
   template lands on the song's beat, wherever the song's beat really is.
2. It chooses where in the song the edit starts, so that the template's first
   drop lands on one of the song's drops, in whole beats. When the song's drop
   comes earlier than the template's, the start of the template is trimmed;
   when either has no drop, the edit starts on the song's first beat.
3. It replaces the template's music with that stretch of the song and
   re-derives beat alignment, section energies, patterns and stats
   (:func:`~editast.build.rebeat`), so the AST describes the new song.

:func:`best_fit` fits several templates and ranks them by a cost: how much the
template is stretched (``|log2|`` of the tempo ratio), the share of it trimmed
or running past the end of the song, and 0.25 when a drop could not be aligned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from editast.ast.nodes import Edit, Music
from editast.build import rebeat
from editast.context import AnalysisConfig, context_from_arrays
from editast.render.compiler import warp_edit

UNALIGNED_DROP_COST = 0.25
_EPS = 1e-6


@dataclass
class Song:
    """A song's music analysis and length in seconds."""

    music: Music
    duration: float
    source: str = ""


@dataclass
class Fit:
    """A template fitted to a song (see the module docstring)."""

    edit: Edit  # the template on the song's grid, carrying the song's music from song_start
    song_start: float  # seconds into the song where the edit starts
    tempo: float  # beats per minute the template was moved to (the song's, half or double)
    drop: Optional[float]  # song time the template's first drop lands on, if aligned
    trimmed: float  # seconds cut from the start of the template
    overrun: float  # seconds of the edit past the end of the song
    cost: float
    template: str = ""
    notes: list[str] = field(default_factory=list)


def analyze_song(path: str | Path, config: Optional[AnalysisConfig] = None) -> Song:
    """Tempo, beats, drops, sections and energy of an audio (or video) file."""
    from editast.analyzers.audio import analyze_audio
    from editast.media import read_audio

    cfg = config or AnalysisConfig()
    samples = read_audio(path, sr=cfg.audio_sr)
    if samples.size == 0:
        raise ValueError(f"{path} has no audio")
    # A one-frame dummy video: the audio analyzer then uses the whole song as its timeline.
    ctx = context_from_arrays(np.zeros((1, 16, 16, 3), np.uint8), fps=30.0, audio=samples, sr=cfg.audio_sr,
                              source=str(path), config=cfg)
    music = analyze_audio(ctx).music
    if music is None or not music.bpm:
        raise ValueError(f"could not detect a tempo in {path}")
    return Song(music=music, duration=samples.size / cfg.audio_sr, source=str(path))


class BeatClock:
    """Times <-> beat positions along a song's detected beats.

    Beat ``i`` of the (gap-filled) beat list is at position ``i``. Between beats
    positions interpolate; before the first and after the last they continue at
    the tempo. Following the detected beats instead of a straight tempo grid
    keeps cuts on the beat when a song drifts or its first beat is off the grid.
    """

    def __init__(self, music: Music) -> None:
        if not music.bpm:
            raise ValueError("music without a tempo")
        self.period = 60.0 / music.bpm
        beats = np.unique(np.asarray([b for b in music.beats if math.isfinite(b)], float))
        kept: list[float] = []
        for b in beats:
            if kept and b - kept[-1] < 0.5 * self.period:  # a doubled detection
                continue
            if kept:
                gap = int(round((b - kept[-1]) / self.period))
                kept += [kept[-1] + (b - kept[-1]) * j / gap for j in range(1, gap)]  # missed beats
            kept.append(float(b))
        if not kept:
            kept = [float(music.beat_offset or 0.0)]
        self.times = np.asarray(kept)
        self.index = np.arange(self.times.size, dtype=float)

    def pos(self, t: float) -> float:
        if t <= self.times[0]:
            return (t - self.times[0]) / self.period
        if t >= self.times[-1]:
            return self.index[-1] + (t - self.times[-1]) / self.period
        return float(np.interp(t, self.times, self.index))

    def time(self, x: float) -> float:
        if x <= 0.0:
            return float(self.times[0] + x * self.period)
        if x >= self.index[-1]:
            return float(self.times[-1] + (x - self.index[-1]) * self.period)
        return float(np.interp(x, self.index, self.times))


def song_window(music: Music, start: float, duration: float) -> Music:
    """The part of ``music`` from ``start`` to ``start + duration``, with times from 0."""
    end = start + duration

    def times(ts: Sequence[float]) -> list[float]:
        return [round(t - start, 4) for t in ts if start - _EPS <= t <= end + _EPS]

    out = music.model_copy(deep=True)
    out.beats, out.downbeats, out.onsets = times(music.beats), times(music.downbeats), times(music.onsets)
    out.beat_offset = out.beats[0] if out.beats else None
    out.drops = [d.model_copy(update={"time": round(d.time - start, 4)})
                 for d in music.drops if start - _EPS <= d.time <= end + _EPS]
    out.sections = [s.model_copy(update={"start": max(0.0, s.start - start), "end": min(duration, s.end - start)})
                    for s in music.sections if s.end > start and s.start < end]
    if music.energy is not None and music.energy.values and music.energy.dt > 0:
        c = music.energy
        i0 = max(0, math.ceil((start - c.t0) / c.dt - _EPS))
        i1 = max(i0, math.floor((end - c.t0) / c.dt + _EPS) + 1)
        out.energy = c.model_copy(update={"t0": c.t0 + i0 * c.dt - start, "values": list(c.values[i0:i1])})
    return out


def fit_edit(template: Edit, song: Song, name: str = "", config: Optional[AnalysisConfig] = None) -> Fit:
    """Fit ``template`` to ``song`` (see the module docstring)."""
    tm, sm = template.music, song.music
    if tm is None or not tm.bpm:
        raise ValueError(f"template {name} has no tempo" if name else "the template has no tempo")
    if not sm.bpm:
        raise ValueError("the song has no tempo")
    f = min((0.5, 1.0, 2.0), key=lambda f: abs(math.log(sm.bpm * f / tm.bpm)))
    tempo = sm.bpm * f  # one template beat becomes 1 / f song beats
    k = tm.bpm / tempo  # template seconds -> fitted seconds, overall
    src, dst = BeatClock(tm), BeatClock(sm)
    x0, x1 = src.pos(0.0) / f, src.pos(template.media.duration) / f  # the template's span in song beats
    length = template.media.duration * k

    def place(shift: float, drop: Optional[float]) -> tuple[float, ...]:
        """Cost and placement when template beat x lands on song beat x / f + shift."""
        start = dst.time(x0 + shift)  # song time of the template's first frame
        trimmed = max(0.0, -start)
        overrun = max(0.0, dst.time(x1 + shift) - song.duration)
        missed = UNALIGNED_DROP_COST if drop is None and (tm.drops or sm.drops) else 0.0
        cost = abs(math.log2(k)) + (trimmed + overrun) / max(length, _EPS) + missed
        return cost, shift, max(0.0, start), trimmed, overrun, math.nan if drop is None else drop

    # Without a drop to align: the first whole-beat shift that starts the template inside the song.
    candidates = [place(float(math.ceil(dst.pos(0.0) - x0 - _EPS)), None)]
    if tm.drops:
        xd = round(src.pos(min(d.time for d in tm.drops))) / f
        for d in sm.drops:
            yd = round(dst.pos(d.time))
            candidates.append(place(yd - xd, dst.time(yd)))
    cost, shift, s, trimmed, overrun, drop = min(candidates, key=lambda c: c[0])

    edit = warp_edit(template, lambda t: dst.time(src.pos(t) / f + shift) - s, k)
    edit.music = song_window(sm, s, edit.media.duration)
    rebeat(edit, config)
    notes = []
    if math.isnan(drop) and tm.drops:
        notes.append("the template's drop is not aligned: " + ("the song has no drop" if not sm.drops
                                                                else "no song drop fitted better"))
    if trimmed > _EPS:
        notes.append(f"the song's drop comes early: the first {trimmed:.2f} s of the template are cut")
    if overrun > _EPS:
        notes.append(f"the song ends {overrun:.2f} s before the edit")
    if abs(math.log2(k)) > 0.3:
        notes.append(f"the template is stretched {1 / k:.2f}x ({tm.bpm:.1f} -> {tempo:.1f} BPM)")
    edit.provenance = {**edit.provenance, "fit": {
        "template": name, "song": song.source, "song_start": round(s, 4), "tempo": round(tempo, 3),
        "drop": None if math.isnan(drop) else round(drop, 4), "cost": round(cost, 4)}}
    edit.media.source = song.source or edit.media.source
    return Fit(edit=edit, song_start=s, tempo=tempo, drop=None if math.isnan(drop) else drop,
               trimmed=trimmed, overrun=overrun, cost=cost, template=name, notes=notes)


def best_fit(templates: Sequence[tuple[str, Edit]], song: Song,
             config: Optional[AnalysisConfig] = None) -> list[Fit]:
    """Fit every template that has a tempo; the best (lowest cost) first."""
    fits = [fit_edit(edit, song, name, config) for name, edit in templates if edit.music and edit.music.bpm]
    return sorted(fits, key=lambda f: f.cost)
