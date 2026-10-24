"""Tests for the music-structure analyzer (editast.analyzers.audio)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from editast.analyzers.audio import analyze_audio, mel_filterbank
from editast.ast.nodes import Music
from editast.context import AnalysisConfig, AnalysisContext, context_from_arrays, load_context
from editast.media import VideoWriter, write_wav
from editast.testing import synth

SR = 22050


def make_ctx(audio: np.ndarray, sr: int = SR, video_seconds: float | None = None,
             config: AnalysisConfig | None = None) -> AnalysisContext:
    """A context with tiny frames; the timeline is ``video_seconds`` long (default: the audio)."""
    seconds = audio.size / sr if video_seconds is None else video_seconds
    n = 4
    frames = np.zeros((n, 8, 8, 3), np.uint8)
    return context_from_arrays(frames, n / seconds, audio=audio, sr=sr, config=config)


def f_measure(est: list[float] | np.ndarray, ref: np.ndarray, tol: float = 0.07) -> float:
    """Beat-tracking F-measure with greedy one-to-one matching within ``tol`` seconds."""
    est = np.asarray(est, float)
    if est.size == 0 or ref.size == 0:
        return 0.0
    used = np.zeros(est.size, bool)
    hits = 0
    for r in ref:
        d = np.where(used, np.inf, np.abs(est - r))
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            hits += 1
    precision, recall = hits / est.size, hits / ref.size
    return 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 1) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    noise = np.random.default_rng(seed).standard_normal(audio.size) * rms * 10 ** (-snr_db / 20)
    return (audio + noise).astype(np.float32)


def phonk_song(bpm: float, offset: float = 0.25, seed: int = 0, noise_db: float | None = None,
               ) -> tuple[np.ndarray, np.ndarray, list[float], list[tuple[str, float]]]:
    """A song with known structure: intro (pads) / build (soft kicks, riser, snare
    roll) / drop (kick, clap on 2+4, 16th hats, 808 on 1+3) / break (pads, hats) /
    drop / outro (fading kicks), soft-clipped like a mastered track.

    Returns ``(audio, beat times, drop times, [(label, start), ...])``.
    """
    rng = np.random.default_rng(seed)
    plan = [("intro", 4), ("build", 4), ("drop", 8), ("break", 4), ("drop", 8), ("outro", 4)]
    beat = 60.0 / bpm
    out = np.zeros(int((offset + sum(b for _, b in plan) * 4 * beat + 1.0) * SR))

    def place(sig: np.ndarray, t: float) -> None:
        i = int(round(t * SR))
        out[i:i + sig.size] += sig[: out.size - i]

    def decay(seconds: float, tau: float) -> np.ndarray:
        return np.exp(-np.arange(int(seconds * SR)) / (tau * SR))

    tt = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-tt * 35)) / SR) * np.exp(-tt * 12)
    t808 = np.arange(int(1.5 * beat * SR)) / SR
    bass = 0.6 * np.tanh(2 * np.sin(2 * np.pi * 49 * t808)) * np.minimum(1, t808 / 0.005)
    bass *= np.exp(-t808 / beat)  # 808: saturated 49 Hz sine sustaining into the next beat
    tp = np.arange(int(4 * beat * SR)) / SR
    chord = sum(np.sin(2 * np.pi * f * tp + i) for i, f in enumerate((220, 262, 330)))
    pad = chord * np.minimum(1, tp / 0.3) * np.minimum(1, (tp[-1] - tp) / 0.3)

    def noise(seconds: float, tau: float, amp: float) -> np.ndarray:
        n = decay(seconds, tau)
        return amp * np.diff(rng.standard_normal(n.size + 1)) * n

    beats, drops, bounds, t0 = [], [], [], offset
    for label, bars in plan:
        bounds.append((label, t0))
        drops += [t0] if label == "drop" else []
        for b in range(bars):
            fade = 1.0 - b / bars if label == "outro" else 1.0
            place((0.03 if label == "drop" else 0.05) * fade * pad, t0 + 4 * b * beat)
            for k in range(4):
                t = t0 + (4 * b + k) * beat
                beats.append(t)
                if label == "build":
                    place(0.6 * kick, t)
                    if b == bars - 1:  # snare roll into the drop
                        for s in range(4):
                            place(noise(0.15, 0.04, 0.15 + 0.1 * s), t + s * beat / 4)
                elif label == "drop":
                    place((1.0 if k == 0 else 0.85) * kick, t)
                    if k in (1, 3):
                        place(noise(0.15, 0.04, 0.5), t)
                    if k in (0, 2):
                        place(bass, t)
                    for s in range(4):
                        place(noise(0.05, 0.012, 0.18 if s % 2 == 0 else 0.12), t + s * beat / 4)
                elif label == "break":
                    for s in range(2):
                        place(noise(0.05, 0.012, 0.08), t + s * beat / 2)
                elif label == "outro":
                    place(0.5 * fade * kick, t)
                    place(noise(0.05, 0.012, 0.08 * fade), t + beat / 2)
        if label == "build":  # rising white-noise riser
            n = int(bars * 4 * beat * SR)
            place(0.15 * np.diff(rng.standard_normal(n + 1)) * (np.arange(n) / n) ** 2, t0)
        t0 += bars * 4 * beat
    audio = out[: int(t0 * SR)]
    audio = np.tanh(2.5 * audio / np.abs(audio).max()) / np.tanh(2.5)
    audio = (0.9 * audio).astype(np.float32)
    if noise_db is not None:
        audio = add_noise(audio, -noise_db, seed=seed + 1)
    return audio, np.array(beats), drops, bounds


def check_music_invariants(music: Music, duration: float) -> None:
    beats = np.asarray(music.beats)
    assert np.all(np.diff(beats) > 0)
    assert beats.size == 0 or (beats[0] >= 0 and beats[-1] <= duration)
    assert music.beat_offset == (music.beats[0] if music.beats else None)
    assert set(music.downbeats) <= set(music.beats)
    assert np.all(np.diff(music.onsets) > 0)
    # Sections tile [0, duration] contiguously.
    secs = music.sections
    assert secs and secs[0].start == pytest.approx(0.0) and secs[-1].end == pytest.approx(duration)
    for a, b in zip(secs[:-1], secs[1:]):
        assert a.end == pytest.approx(b.start)
        assert a.label != b.label
    assert all(0.0 <= s.energy <= 1.0 for s in secs)
    curve = music.energy
    assert curve is not None and curve.dt == pytest.approx(0.05)
    assert abs(len(curve.values) * curve.dt - duration) <= curve.dt + 1e-9
    assert 0.0 <= min(curve.values) and max(curve.values) <= 1.0
    assert 0.0 <= music.confidence <= 1.0


# --------------------------------------------------------------------------- tempo / beats


@pytest.mark.parametrize("bpm,offset", [(92, 0.31), (128, 0.17), (150, 0.43)])
def test_tempo_beats_and_downbeats(bpm, offset):
    duration = 20.0
    audio = synth.click_track(bpm, duration, sr=SR, offset=offset)
    res = analyze_audio(make_ctx(audio))
    music = res.music
    assert music is not None
    check_music_invariants(music, duration)
    assert music.bpm == pytest.approx(bpm, rel=0.02)
    ref = synth.beat_times(bpm, duration, offset)
    assert f_measure(music.beats, ref) >= 0.9
    assert music.beat_offset == pytest.approx(offset, abs=0.07)
    assert music.meter == 4
    assert f_measure(music.downbeats, ref[::4]) >= 0.8
    assert music.drops == []
    # A steady loop is one section.
    assert [s.label for s in music.sections] == ["body"]
    # Strong hits include (at least) the kicks.
    assert f_measure(music.onsets, ref) >= 0.8 or len(music.onsets) >= ref.size
    # Debug signals share one time base.
    assert {"onset_env", "rms"} <= set(res.signals)
    assert res.hop_seconds == pytest.approx(512 / SR)
    n = res.signals["onset_env"].size
    assert res.signals["rms"].size == n
    assert abs(n * res.hop_seconds - duration) < 0.1


def test_slow_track_75_bpm():
    """75 BPM (kick per beat, light off-beat hats) is reported at 75, not 150.

    Octave behaviour: the tempo prior is centred at 120 BPM, so below ~72 BPM
    the same kick/hat pattern is reported at double tempo (it is identical to
    a 2x-tempo kick/clap alternation); see the module docstring.
    """
    duration = 20.0
    audio = synth.click_track(75, duration, sr=SR, offset=0.2)
    music = analyze_audio(make_ctx(audio)).music
    assert music is not None
    assert music.bpm == pytest.approx(75, rel=0.02)
    ref = synth.beat_times(75, duration, 0.2)
    assert f_measure(music.beats, ref) >= 0.9
    assert f_measure(music.downbeats, ref[::4]) >= 0.8


@pytest.mark.parametrize("bpm", [140, 170])
def test_fast_phonk_tempi_no_half_time(bpm):
    duration = 16.0
    audio = synth.click_track(bpm, duration, sr=SR, offset=0.05, seed=3)
    music = analyze_audio(make_ctx(audio)).music
    assert music is not None
    assert music.bpm == pytest.approx(bpm, rel=0.02)
    assert f_measure(music.beats, synth.beat_times(bpm, duration, 0.05)) >= 0.9


@pytest.mark.parametrize("bpm,offset", [(128, 0.17), (150, 0.43)])
def test_white_noise_robustness(bpm, offset):
    duration = 16.0
    audio = add_noise(synth.click_track(bpm, duration, sr=SR, offset=offset), snr_db=20.0)
    music = analyze_audio(make_ctx(audio)).music
    assert music is not None
    assert music.bpm == pytest.approx(bpm, rel=0.02)
    ref = synth.beat_times(bpm, duration, offset)
    assert f_measure(music.beats, ref) >= 0.9
    assert f_measure(music.downbeats, ref[::4]) >= 0.75


@pytest.mark.parametrize("sr", [16000, 44100])
def test_other_sample_rates(sr):
    duration = 12.0
    audio = synth.click_track(128, duration, sr=sr, offset=0.17)
    res = analyze_audio(make_ctx(audio, sr=sr))
    assert res.music is not None and res.music.bpm == pytest.approx(128, rel=0.02)
    assert f_measure(res.music.beats, synth.beat_times(128, duration, 0.17)) >= 0.9


# --------------------------------------------------------------------------- drops / sections


@pytest.mark.parametrize("bpm,offset", [(92, 0.31), (128, 0.17), (150, 0.43)])
def test_drop_and_sections(bpm, offset):
    duration, drop_at = 16.0, 6.0
    audio = synth.click_track(bpm, duration, sr=SR, offset=offset, drop_at=drop_at)
    music = analyze_audio(make_ctx(audio)).music
    assert music is not None
    check_music_invariants(music, duration)
    assert music.bpm == pytest.approx(bpm, rel=0.02)
    assert len(music.drops) == 1
    drop = music.drops[0]
    assert drop.time == pytest.approx(drop_at, abs=0.3)
    beats = synth.beat_times(bpm, duration, offset)
    first_loud = beats[beats >= drop_at][0]
    assert drop.time == pytest.approx(first_loud, abs=0.07)  # snapped onto the beat
    assert drop.strength > 1.5
    labels = [s.label for s in music.sections]
    assert labels[-1] == "drop"
    drop_sec = music.sections[-1]
    assert drop_sec.start == pytest.approx(drop.time, abs=0.05)
    assert labels[0] in ("build", "intro")
    assert drop_sec.energy > music.sections[0].energy


@pytest.mark.parametrize("bpm,noise_db", [(140, None), (150, -20.0), (90, None)])
def test_song_structure(bpm, noise_db):
    audio, beats, drops, bounds = phonk_song(bpm, noise_db=noise_db)
    duration = audio.size / SR
    music = analyze_audio(make_ctx(audio)).music
    assert music is not None
    check_music_invariants(music, duration)
    assert music.bpm == pytest.approx(bpm, rel=0.01)
    drum_beats = beats[beats >= bounds[1][1]]  # the pad-only intro has no beat evidence
    assert f_measure(music.beats, drum_beats) >= 0.9
    assert f_measure(music.downbeats, drum_beats[::4]) >= 0.8
    # Both drops, on the beat they start on (not early, despite riser and snare roll).
    assert [d.time for d in music.drops] == pytest.approx(drops, abs=0.07)
    assert music.drops[1].strength > 2.0
    assert [s.label for s in music.sections] == [label for label, _ in bounds]
    for sec, (_, true_start) in zip(music.sections[1:], bounds[1:]):
        assert sec.start == pytest.approx(true_start, abs=0.4)
    drop_secs = [s for s in music.sections if s.label == "drop"]
    assert [s.start for s in drop_secs] == pytest.approx(drops, abs=0.07)
    assert min(s.energy for s in drop_secs) > max(s.energy for s in music.sections if s.label != "drop")


def test_excerpt_starting_inside_a_drop():
    """Short-form clips often start mid-song: a loud opening is a drop, not a build."""
    audio, beats, drops, bounds = phonk_song(150, offset=0.0)
    start, length = 20.0, 25.0
    clip = audio[int(start * SR): int((start + length) * SR)]
    music = analyze_audio(make_ctx(clip)).music
    assert music is not None
    assert [s.label for s in music.sections] == ["drop", "break", "drop"]
    assert music.sections[1].start == pytest.approx(bounds[3][1] - start, abs=0.4)
    assert [d.time for d in music.drops] == pytest.approx([drops[1] - start], abs=0.07)
    ref = beats[(beats >= start) & (beats < start + length)] - start
    assert f_measure(music.beats, ref) >= 0.9


def test_short_bursts_are_not_drops():
    """A single loud bar inside a steady loop is not a sustained energy jump."""
    duration = 16.0
    audio = synth.click_track(128, duration, sr=SR, offset=0.1)
    i0 = int(7.0 * SR)
    burst = synth.click_track(128, 0.5, sr=SR, offset=0.0, seed=5) * 0.4
    audio[i0:i0 + burst.size] += burst
    music = analyze_audio(make_ctx(audio)).music
    assert music is not None
    assert music.drops == []


# --------------------------------------------------------------------------- silence / timeline


def test_silent_or_missing_audio_gives_no_music():
    assert analyze_audio(make_ctx(np.zeros(0, np.float32), video_seconds=5.0)).music is None
    assert analyze_audio(make_ctx(np.zeros(5 * SR, np.float32))).music is None
    hiss = 1e-5 * np.random.default_rng(0).standard_normal(5 * SR).astype(np.float32)
    assert analyze_audio(make_ctx(hiss)).music is None
    # Too short to analyse.
    assert analyze_audio(make_ctx(np.ones(100, np.float32) * 0.1, video_seconds=0.5)).music is None


def test_audio_shorter_than_video_ends_in_silence():
    audio = synth.click_track(128, 10.0, sr=SR, offset=0.17)
    music = analyze_audio(make_ctx(audio, video_seconds=14.0)).music
    assert music is not None
    check_music_invariants(music, 14.0)
    assert music.sections[-1].label == "silence"
    assert music.sections[-1].start == pytest.approx(10.0, abs=0.3)
    assert max(music.beats) < 10.3  # no beats invented in the silent tail
    assert music.energy is not None and max(music.energy.values[-40:]) < 0.05


def test_leading_silence_then_music():
    lead = np.zeros(3 * SR, np.float32)
    audio = np.concatenate([lead, synth.click_track(128, 12.0, sr=SR, offset=0.0)])
    music = analyze_audio(make_ctx(audio)).music
    assert music is not None
    assert music.sections[0].label == "silence"
    assert music.beat_offset == pytest.approx(3.0, abs=0.07)
    assert f_measure(music.beats, 3.0 + synth.beat_times(128, 12.0, 0.0)) >= 0.9


def test_standalone_music_file_uses_audio_length():
    """A context with a single frame (e.g. a music file) is timed by its audio."""
    audio = synth.click_track(128, 10.0, sr=SR, offset=0.17)
    ctx = context_from_arrays(np.zeros((1, 8, 8, 3), np.uint8), 30.0, audio=audio, sr=SR)
    music = analyze_audio(ctx).music
    assert music is not None
    check_music_invariants(music, 10.0)
    assert music.bpm == pytest.approx(128, rel=0.02)


# --------------------------------------------------------------------------- real codec round trip


def test_encode_decode_roundtrip(tmp_path):
    """H.264 + AAC through ffmpeg and back, as a downloaded video would arrive."""
    bpm, offset, duration, fps = 140, 0.12, 12.0, 10
    audio = synth.click_track(bpm, duration, sr=SR, offset=offset, drop_at=5.0)
    wav = write_wav(tmp_path / "music.wav", audio, SR)
    frames, _ = synth.cut_sequence([40, 40, 40], height=64, width=36)
    with VideoWriter(tmp_path / "clip.mp4", 36, 64, fps, audio_path=wav) as vw:
        for f in frames:
            vw.write(f)
    ctx = load_context(tmp_path / "clip.mp4", AnalysisConfig(analysis_long_side=64))
    music = analyze_audio(ctx).music
    assert music is not None
    assert music.bpm == pytest.approx(bpm, rel=0.02)
    ref = synth.beat_times(bpm, duration, offset)
    assert f_measure(music.beats, ref) >= 0.9
    assert any(abs(d.time - 5.0) <= 0.3 for d in music.drops)


# --------------------------------------------------------------------------- config / helpers / speed


def test_config_overrides():
    audio = synth.click_track(128, 12.0, sr=SR, offset=0.17)
    cfg = AnalysisConfig(params={"audio": {"min_clarity": 1.01, "energy_dt": 0.1}})
    music = analyze_audio(make_ctx(audio, config=cfg)).music
    assert music is not None
    assert music.bpm is None and music.beats == [] and music.downbeats == []
    assert music.energy is not None and music.energy.dt == pytest.approx(0.1)
    assert music.onsets  # onsets do not need a beat grid


def test_noise_without_pulse_has_no_confident_grid():
    noise = 0.3 * np.random.default_rng(2).standard_normal(12 * SR).astype(np.float32)
    music = analyze_audio(make_ctx(noise)).music
    assert music is not None  # audible, so there is music (energy, sections) ...
    assert music.bpm is None or music.confidence < 0.3  # ... but no trustworthy tempo
    assert music.drops == []
    assert len(music.onsets) <= 3


def test_steady_tone_has_no_onsets_or_tempo():
    t = np.arange(10 * SR) / SR
    music = analyze_audio(make_ctx((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))).music
    assert music is not None
    assert music.bpm is None and music.beats == [] and music.onsets == []
    assert [s.label for s in music.sections] == ["body"]


def test_clip_too_short_for_a_tempo():
    audio = synth.click_track(128, 1.0, sr=SR, offset=0.1)
    music = analyze_audio(make_ctx(audio)).music
    assert music is not None and music.bpm is None and music.beats == []
    assert music.energy is not None and len(music.energy.values) == 20


def test_mel_filterbank_shape_and_coverage():
    fb, centers = mel_filterbank(SR, 2048, 64, 30.0, 11025.0)
    assert fb.shape == (64, 1025)
    assert np.all(np.diff(centers) > 0)
    assert np.all(fb.sum(axis=1) > 0) and fb.min() >= 0.0


def test_runtime_60s():
    audio = synth.click_track(140, 60.0, sr=SR, offset=0.1, drop_at=20.0)
    ctx = make_ctx(audio)
    t0 = time.perf_counter()
    music = analyze_audio(ctx).music
    elapsed = time.perf_counter() - t0
    assert music is not None and music.bpm == pytest.approx(140, rel=0.02)
    assert elapsed < 5.0
