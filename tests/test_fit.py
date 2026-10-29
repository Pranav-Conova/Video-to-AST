"""Tests for fitting edit templates to a new song (editast.fit, build.rebeat, music_start)."""

import math

import numpy as np
import pytest

from editast.ast import Curve, Drop, Edit, MediaInfo, Music, Section, Shot, Timeline, Transition, assign_ids, shot_id
from editast.build import rebeat
from editast.cli import main
from editast.fit import BeatClock, Song, analyze_song, best_fit, fit_edit, song_window
from editast.media import VideoWriter, read_audio, write_wav
from editast.testing import synth

SR = 22050


def template(bpm: float = 120.0, first: float = 0.1, drop_beat: int = 8, n_beats: int = 16) -> Edit:
    """Cuts every 2 beats up to the drop and on every beat after it."""
    p = 60.0 / bpm
    beats = [first + k * p for k in range(n_beats)]
    cut_beats = list(range(2, drop_beat, 2)) + list(range(drop_beat, n_beats))
    bounds = [0.0] + [beats[k] for k in cut_beats] + [first + n_beats * p]
    shots = [Shot(index=i, start=a, end=b) for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    items: list = []
    for i, s in enumerate(shots):
        if i:
            items.append(Transition(kind="cut", start=s.start, end=s.start, from_shot=shot_id(i - 1), to_shot=shot_id(i)))
        items.append(s)
    edit = Edit(
        media=MediaInfo(duration=bounds[-1], width=360, height=640, analysis_fps=30.0),
        music=Music(bpm=bpm, beat_offset=first, beats=beats, drops=[Drop(time=beats[drop_beat])]),
        timeline=Timeline(sections=[Section(role="build", start=0.0, end=bounds[-1], items=items)]),
    )
    return assign_ids(edit)


def drifting_song(bpm: float = 100.0, n: int = 80, drop_beat: int | None = 40, duration: float | None = None) -> Song:
    """Beats that wander around the tempo grid, with an off-grid first beat."""
    p = 60.0 / bpm
    steps = p * (1.0 + 0.04 * np.sin(np.arange(n) / 3.0))
    beats = list(0.37 + np.concatenate([[0.0], np.cumsum(steps[:-1])]))
    beats[0] -= 0.12
    drops = [Drop(time=beats[drop_beat] + 0.02)] if drop_beat is not None else []
    music = Music(bpm=bpm, beat_offset=beats[0], beats=beats, drops=drops,
                  energy=Curve(t0=0.0, dt=0.5, values=[0.2] * 40 + [0.9] * 200))
    return Song(music=music, duration=duration or beats[-1] + 1.0, source="song.wav")


def song_times_of_cuts(fit) -> list[float]:
    return [s.start + fit.song_start for s in fit.edit.shots()[1:]]


def near_a_beat(t: float, beats: list[float], tol: float = 1e-6) -> bool:
    return min(abs(t - b) for b in beats) <= tol


# --------------------------------------------------------------------------- BeatClock


def test_beat_clock_follows_detected_beats_and_extrapolates():
    music = Music(bpm=60.0, beats=[1.0, 2.1, 3.0, 5.0, 5.4, 6.0])  # a missed beat at 4 and a doubled one at 5.4
    clock = BeatClock(music)
    assert clock.times.tolist() == pytest.approx([1.0, 2.1, 3.0, 4.0, 5.0, 6.0])
    assert clock.pos(2.1) == pytest.approx(1.0) and clock.pos(2.55) == pytest.approx(1.5)
    assert clock.pos(0.5) == pytest.approx(-0.5) and clock.pos(7.0) == pytest.approx(6.0)
    for x in (-1.3, 0.0, 0.4, 2.5, 5.0, 8.2):
        assert clock.pos(clock.time(x)) == pytest.approx(x)


# --------------------------------------------------------------------------- fit_edit


def test_fit_puts_the_drop_on_the_songs_drop_and_cuts_on_its_beats():
    song = drifting_song()
    fit = fit_edit(template(), song, "t")
    beats = song.music.beats
    assert fit.tempo == pytest.approx(100.0)
    assert fit.drop == pytest.approx(beats[40])
    assert fit.trimmed == 0.0 and fit.overrun == 0.0 and fit.notes == []
    # The template's drop cut (its 4th cut) lands on the song's drop beat; every cut lands on a song beat.
    assert song_times_of_cuts(fit)[3] == pytest.approx(beats[40])
    assert all(near_a_beat(t, beats) for t in song_times_of_cuts(fit))
    edit = fit.edit
    assert all(s.on_beat for s in edit.shots()[1:])
    assert [s.beats for s in edit.shots()[4:]] == pytest.approx([1.0] * 8, abs=0.06)
    # The edit carries the song's music from song_start, and says where it came from.
    assert edit.music.bpm == 100.0 and edit.music.drops[0].time == pytest.approx(beats[40] + 0.02 - fit.song_start, abs=1e-3)
    assert edit.music.energy.values[0] == 0.9 and edit.media.source == "song.wav"
    assert edit.provenance["fit"]["template"] == "t" and edit.provenance["fit"]["song_start"] == pytest.approx(fit.song_start)
    assert edit.timeline.sections[0].energy == pytest.approx(0.9)
    assert edit.stats.n_shots == 12 and edit.shots()[0].start == 0.0


def test_an_early_song_drop_trims_the_start_of_the_template():
    song = drifting_song(drop_beat=6)  # two beats earlier than the template's drop
    fit = fit_edit(template(), song)
    assert fit.song_start == 0.0 and fit.drop == pytest.approx(song.music.beats[6])
    assert fit.trimmed > 0.0 and "first" in fit.notes[0]
    assert song_times_of_cuts(fit)[-8] == pytest.approx(song.music.beats[6])  # the drop cut
    assert all(near_a_beat(t, song.music.beats) for t in song_times_of_cuts(fit))
    assert fit.edit.shots()[0].end == pytest.approx(song.music.beats[0])  # the first cut, on song beat 0
    # Much earlier, trimming would cost more than leaving the drop unaligned.
    assert fit_edit(template(), drifting_song(drop_beat=2)).drop is None


def test_without_a_song_drop_the_edit_starts_on_the_first_beats():
    song = drifting_song(drop_beat=None)
    fit = fit_edit(template(), song)
    assert fit.drop is None and fit.notes == ["the template's drop is not aligned: the song has no drop"]
    assert 0.0 <= fit.song_start < 60.0 / 100.0
    assert all(near_a_beat(t, song.music.beats) for t in song_times_of_cuts(fit))
    assert fit.cost == pytest.approx(math.log2(1.2) + 0.25)


def test_fit_uses_half_or_double_tempo_when_it_stretches_less():
    fit = fit_edit(template(bpm=120.0), drifting_song(bpm=230.0))
    assert fit.tempo == pytest.approx(115.0)  # every template beat spans two song beats
    assert all(near_a_beat(t, drifting_song(bpm=230.0).music.beats) for t in song_times_of_cuts(fit))
    assert [s.beats for s in fit.edit.shots()[4:]] == pytest.approx([2.0] * 8, abs=0.12)


def test_a_short_song_is_reported_as_overrun():
    fit = fit_edit(template(), drifting_song(n=10, drop_beat=None))
    assert fit.overrun > 0.0 and fit.notes[-1].startswith("the song ends")


def test_best_fit_ranks_by_stretch_and_skips_templates_without_tempo():
    song = drifting_song()
    no_tempo = template()
    no_tempo.music = None
    fits = best_fit([("fast", template(bpm=150.0)), ("none", no_tempo), ("close", template(bpm=104.0))], song)
    assert [f.template for f in fits] == ["close", "fast"]
    with pytest.raises(ValueError, match="none has no tempo"):
        fit_edit(no_tempo, song, "none")


def test_song_window_shifts_and_crops_the_music():
    music = drifting_song().music
    music.sections = []
    window = song_window(music, 10.0, 5.0)
    assert window.beats[0] >= 0.0 and window.beats[-1] <= 5.0
    assert window.beat_offset == window.beats[0]
    assert window.energy.t0 == pytest.approx(0.0) and len(window.energy.values) == 11
    assert music.beats[0] < 1.0  # the input is untouched


def test_rebeat_realigns_cuts_with_new_music():
    edit = template()
    rebeat(edit)
    assert all(s.on_beat for s in edit.shots()[1:])
    edit.music = Music(bpm=120.0, beats=[0.35 + 0.5 * k for k in range(16)])  # a quarter beat later
    rebeat(edit)
    assert not any(s.on_beat for s in edit.shots()[1:])
    assert edit.shots()[1].beat_offset == pytest.approx(-0.25)
    edit.music = None
    rebeat(edit)
    assert edit.shots()[1].on_beat is None and edit.shots()[1].beats is None


# --------------------------------------------------------------------------- audio and CLI


def test_video_writer_starts_the_music_later(tmp_path):
    clicks = np.zeros(3 * SR, np.float32)
    clicks[SR : SR + 200] = 0.8  # one burst at 1.0 s
    song = write_wav(tmp_path / "song.wav", clicks, SR)
    with VideoWriter(tmp_path / "out.mp4", 32, 32, 10.0, audio_path=song, audio_start=0.6) as vw:
        for _ in range(20):
            vw.write(np.zeros((32, 32, 3), np.uint8))
    audio = read_audio(tmp_path / "out.mp4", sr=SR)
    assert np.argmax(np.abs(audio)) / SR == pytest.approx(0.4, abs=0.03)


def test_cli_fit_writes_the_best_template_for_a_song(tmp_path, capsys):
    song = write_wav(tmp_path / "track.wav", synth.click_track(100.0, 30.0, SR, offset=0.2, drop_at=12.2), SR)
    analysed = analyze_song(song)
    assert analysed.music.bpm == pytest.approx(100.0, abs=1.0) and analysed.music.drops
    template().save(tmp_path / "templates" / "close.edit.json")
    template(bpm=75.0).save(tmp_path / "templates" / "far.edit.json")  # 133% or 67% of the song's tempo
    assert main(["fit", str(song), str(tmp_path / "templates"), "-o", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert out.index("close") < out.index("far")
    fitted = Edit.load(tmp_path / "out" / "track.close.edit.json")
    assert (tmp_path / "out" / "track.close.edit").is_file() and (tmp_path / "out" / "track.close.report.html").is_file()
    assert fitted.provenance["fit"]["drop"] == pytest.approx(analysed.music.drops[0].time, abs=0.35)
    on = [s.on_beat for s in fitted.shots()[1:]]
    assert sum(on) >= 0.8 * len(on)
