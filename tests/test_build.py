"""Tests for editast.build: structure synthesis from fabricated analyzer results."""

from __future__ import annotations

import numpy as np
import pytest

from editast.analyzers.types import (
    AudioAnalysis,
    EffectsResult,
    FrameMotion,
    LookAnalysis,
    MotionAnalysis,
    ShotSegmentation,
    ShotSpan,
    TextAnalysis,
    TransitionSpan,
)
from editast.ast.nodes import (
    AudioSection,
    Curve,
    Drop,
    Edit,
    Flash,
    Look,
    MotionSummary,
    Music,
    Shake,
    Shot,
    Speed,
    TextOverlay,
    Transition,
    Zoom,
)
from editast.build import build_edit, repair_segmentation
from editast.context import AnalysisConfig, context_from_arrays, load_context
from editast.media import VideoWriter
from editast.testing import synth

FPS = 30.0


def make_ctx(lengths: list[int], config: AnalysisConfig | None = None):
    frames, _ = synth.cut_sequence(lengths, height=36, width=64)
    return context_from_arrays(frames, FPS, config=config)


def make_seg(lengths: list[int], kinds: dict[int, TransitionSpan] | None = None,
             extra: list[TransitionSpan] = ()) -> ShotSegmentation:
    shots, transitions, pos = [], [], 0
    for i, n in enumerate(lengths):
        shots.append(ShotSpan(i, pos, pos + n))
        pos += n
    for i in range(1, len(shots)):
        b = shots[i].f0
        transitions.append((kinds or {}).get(i) or TransitionSpan("cut", b, b, b, i - 1, i))
    return ShotSegmentation(shots, transitions + list(extra), np.zeros(pos))


def music_120(duration: float, sections: list[AudioSection], drops: list[float]) -> AudioAnalysis:
    beats = np.arange(0.0, duration, 0.5).tolist()
    values = []
    for t in np.arange(0.0, duration, 0.1):
        values.append(next((s.energy for s in sections if s.start <= t < s.end), 0.0))
    music = Music(bpm=120.0, beat_offset=0.0, beats=beats, downbeats=beats[::4],
                  drops=[Drop(time=t, strength=1.0) for t in drops], sections=sections,
                  energy=Curve(t0=0.0, dt=0.1, values=values))
    return AudioAnalysis(music=music, hop_seconds=0.1)


def drop_music(duration: float = 8.0) -> AudioAnalysis:
    return music_120(duration, [AudioSection(start=0.0, end=4.0, label="build", energy=0.4),
                                AudioSection(start=4.0, end=duration, label="drop", energy=0.9)], [4.0])


def ids(edit: Edit) -> list[str]:
    return [n.id for n in edit.walk()]


# --------------------------------------------------------------------------- beats & sections


def test_half_second_cuts_at_120_bpm_are_on_beat_with_hook_build_drop():
    lengths = [15] * 16  # 0.5 s shots, 8 s
    ctx = make_ctx(lengths)
    edit = build_edit(ctx, make_seg(lengths), audio=drop_music())

    shots = edit.shots()
    assert len(shots) == 16
    assert all(s.on_beat for s in shots)
    assert all(s.beat_offset == pytest.approx(0.0, abs=1e-6) for s in shots)
    assert all(s.beats == pytest.approx(1.0) for s in shots)
    assert all(t.on_beat for t in edit.transitions())
    assert [s.id for s in shots] == [f"s{i}" for i in range(1, 17)]
    assert shots[3].start == pytest.approx(1.5) and shots[3].frame_start == 45

    roles = [(sec.role, round(sec.start, 3), round(sec.end, 3)) for sec in edit.timeline.sections]
    assert roles == [("hook", 0.0, 1.0), ("build", 1.0, 4.0), ("drop", 4.0, 8.0)]
    drop = edit.timeline.sections[2]
    assert drop.cuts_per_second == pytest.approx(2.0)
    assert drop.mean_shot_duration == pytest.approx(0.5)
    assert drop.energy == pytest.approx(0.9)
    assert edit.timeline.sections[1].energy == pytest.approx(0.4)

    rules = {p.rule: p for p in edit.patterns}
    assert rules["cut_on_beat"].support == pytest.approx(1.0)
    every = [p for p in edit.patterns if p.rule == "cut_every_n_beats"]
    assert {p.params["section"] for p in every} == {"hook", "build", "drop"}
    assert all(p.params["n"] == 1.0 and p.support == pytest.approx(1.0) for p in every)
    assert "pacing_accelerates_at_drop" not in rules  # same pace on both sides
    assert all(p.description for p in edit.patterns)

    st = edit.stats
    assert st.n_shots == 16 and st.n_transitions == 0
    assert st.cut_on_beat_ratio == pytest.approx(1.0)
    assert st.hook_duration == pytest.approx(1.0)
    assert st.first_cut_time == pytest.approx(0.5)
    assert st.cuts_per_minute == pytest.approx(15 / 8 * 60)
    assert st.transition_counts == {"cut": 15}
    assert st.pacing.dt == pytest.approx(0.5) and st.pacing.t0 == pytest.approx(0.5)
    assert len(st.pacing.values) == 15 and st.pacing.values[0] == pytest.approx(1.0)
    assert edit.music.bpm == 120.0
    assert edit.provenance["tool"] == "editast" and "config" in edit.provenance


def test_pacing_acceleration_and_n_beats_per_section():
    lengths = [30, 30, 30, 30] + [15] * 8  # 1 s shots until the drop at 4 s, then 0.5 s
    ctx = make_ctx(lengths)
    edit = build_edit(ctx, make_seg(lengths), audio=drop_music())
    assert [s.role for s in edit.timeline.sections] == ["hook", "build", "drop"]
    rules = {p.rule: p for p in edit.patterns}
    acc = rules["pacing_accelerates_at_drop"]
    assert acc.params["before"] == pytest.approx(1.0) and acc.params["after"] == pytest.approx(2.0)
    every = {p.params["section"]: p.params["n"] for p in edit.patterns if p.rule == "cut_every_n_beats"}
    assert every == {"build": 2.0, "drop": 1.0}


def test_drop_snaps_to_nearest_cut_and_outro():
    # cuts every 0.5 s except around the drop (3.9 s); trailing low-energy audio from 7 s.
    lengths = [15] * 7 + [12] + [18] + [15] * 10  # boundaries ..., 3.5, 3.9, 4.5, ...
    ctx = make_ctx(lengths)
    audio = music_120(9.5, [AudioSection(start=0.0, end=4.0, label="build", energy=0.4),
                            AudioSection(start=4.0, end=7.0, label="drop", energy=0.9),
                            AudioSection(start=7.0, end=9.5, label="outro", energy=0.2)], [4.0])
    edit = build_edit(ctx, make_seg(lengths), audio=audio)
    roles = [(sec.role, round(sec.start, 2), round(sec.end, 2)) for sec in edit.timeline.sections]
    assert roles == [("hook", 0.0, 1.0), ("build", 1.0, 3.9), ("drop", 3.9, 7.0), ("outro", 7.0, 9.5)]
    assert not edit.shots()[8].on_beat  # starts at 3.9 s, 0.1 s before the beat
    assert edit.shots()[8].beat_offset == pytest.approx(-0.1)

    # Without audio sections the end of the drop comes from the energy curve.
    audio.music.sections = []
    edit = build_edit(ctx, make_seg(lengths), audio=audio)
    roles = [(sec.role, round(sec.start, 2), round(sec.end, 2)) for sec in edit.timeline.sections]
    assert roles == [("hook", 0.0, 1.0), ("build", 1.0, 3.9), ("drop", 3.9, 7.0), ("outro", 7.0, 9.5)]


def test_without_music_sections_follow_pacing_changes():
    lengths = [36] + [60] * 4 + [12] * 12  # 1.2 s hook, four 2 s shots, twelve 0.4 s shots
    ctx = make_ctx(lengths)
    edit = build_edit(ctx, make_seg(lengths))
    roles = [(sec.role, round(sec.start, 2), round(sec.end, 2)) for sec in edit.timeline.sections]
    assert roles == [("hook", 0.0, 1.2), ("build", 1.2, 9.2), ("drop", 9.2, 14.0)]
    assert edit.music is None and edit.stats.cut_on_beat_ratio is None
    assert all(s.on_beat is None and s.beats is None for s in edit.shots())
    assert not [p for p in edit.patterns if "beat" in p.rule]


def test_hook_extends_over_opening_text_and_hook_text_pattern():
    lengths = [15] * 16
    ctx = make_ctx(lengths)
    text = TextAnalysis(overlays=[
        TextOverlay(start=4.2, end=5.0, text="LATER"),
        TextOverlay(start=0.1, end=1.6, text="WAIT FOR IT", role="hook"),
    ])
    edit = build_edit(ctx, make_seg(lengths), text=text, audio=drop_music())
    assert edit.timeline.sections[0].role == "hook"
    assert edit.timeline.sections[0].end == pytest.approx(1.5)
    assert [o.text for o in edit.overlays] == ["WAIT FOR IT", "LATER"]
    hook = next(p for p in edit.patterns if p.rule == "hook_text")
    assert "WAIT FOR IT" in hook.description
    assert edit.stats.text_coverage == pytest.approx((1.5 + 0.8) / 8.0)


# --------------------------------------------------------------------------- transitions & effects


def test_transitions_are_placed_before_their_incoming_shot():
    lengths = [15] * 16
    ctx = make_ctx(lengths)
    flash = TransitionSpan("flash", 120, 118, 122, 7, 8, confidence=0.8, color=(255, 255, 255))
    weak_dup = TransitionSpan("dissolve", 120, 116, 124, 7, 8, confidence=0.3)
    fade_in = TransitionSpan("fade_black", 0, 0, 9, None, 0, color=(0, 0, 0))
    fade_out = TransitionSpan("fade_black", 240, 230, 240, 15, None, color=(0, 0, 0))
    seg = make_seg(lengths, kinds={8: flash}, extra=[weak_dup, fade_in, fade_out])
    seg.transitions = [t for t in seg.transitions if not (t.kind == "cut" and t.boundary == 30)]  # missing cut
    edit = build_edit(ctx, seg, audio=drop_music())

    first = edit.timeline.sections[0].items
    assert isinstance(first[0], Transition) and first[0].from_shot is None and first[0].to_shot == "s1"
    assert first[0].start == pytest.approx(0.0) and first[0].end == pytest.approx(0.3)
    assert isinstance(first[1], Shot) and first[1].index == 0

    last = edit.timeline.sections[-1].items
    assert isinstance(last[-1], Transition) and last[-1].to_shot is None and last[-1].from_shot == "s16"
    assert last[-1].end == pytest.approx(8.0)

    drop = edit.timeline.sections[2]
    assert drop.role == "drop"
    tr, shot = drop.items[0], drop.items[1]
    assert isinstance(tr, Transition) and tr.kind == "flash" and tr.color == (255, 255, 255)
    assert (tr.from_shot, tr.to_shot) == ("s8", "s9") and shot.id == "s9"
    assert tr.start == pytest.approx(118 / FPS) and tr.end == pytest.approx(122 / FPS)
    assert tr.on_beat and tr.confidence == pytest.approx(0.8)

    # every boundary has exactly one transition, cuts are instantaneous, items alternate
    boundary = [t for t in edit.transitions() if t.from_shot and t.to_shot]
    assert len(boundary) == 15
    for t in boundary:
        if t.kind == "cut":
            assert t.start == t.end == pytest.approx(edit.find(t.to_shot).start)
    for sec in edit.timeline.sections:
        for a, b in zip(sec.items, sec.items[1:]):
            if isinstance(a, Transition) and a.to_shot is not None:
                assert isinstance(b, Shot) and b.id == a.to_shot
    assert edit.stats.n_transitions == 3
    assert edit.stats.transition_counts == {"cut": 14, "fade_black": 2, "flash": 1}
    assert any(p.rule == "effect_on_drop:flash" for p in edit.patterns)


def test_effects_are_clamped_deduped_and_rehomed():
    lengths = [15] * 16
    ctx = make_ctx(lengths)
    motion_fx = EffectsResult()
    motion_fx.add(2, Zoom(start=0.9, end=1.6, scale_to=1.3, confidence=0.8,
                          keyframes=[{"t": 0.9, "v": 1.0}, {"t": 1.2, "v": 1.2}]))  # shot 2 = [1.0, 1.5)
    motion_fx.add(2, Zoom(start=1.2, end=1.2))  # degenerate
    motion_fx.add(3, Shake(start=1.5, end=1.9, amplitude=0.01, confidence=0.5))
    motion_fx.add(3, Shake(start=1.55, end=1.95, amplitude=0.02, confidence=0.9))  # duplicate, wins
    motion_fx.add(0, Flash(start=2.6, end=2.7))  # keyed to the wrong shot -> shot 5 [2.5, 3.0)
    timefx = EffectsResult()
    timefx.add(9, Speed(start=4.5, end=5.0, kind="ramp_down", rate_from=1.0, rate_to=0.4))
    timefx.add(11, Speed(start=5.5, end=6.0, kind="ramp_up", rate_from=0.4, rate_to=1.0))
    look_fx = EffectsResult()
    look_fx.add(7, Flash(start=3.92, end=3.98))  # repeats the flash transition at 4 s (shot 7 [3.5, 4.0))
    n = sum(lengths)
    fm = FrameMotion(*(np.zeros(n) for _ in range(7)))
    motion = MotionAnalysis(frame_motion=fm, effects=motion_fx,
                            summaries={2: MotionSummary(magnitude=0.01, camera="zoom", zoom_ratio=1.3)})
    look = LookAnalysis(effects=look_fx, looks={4: Look(bw=True)})
    flash = TransitionSpan("flash", 120, 119, 121, 7, 8)
    edit = build_edit(ctx, make_seg(lengths, kinds={8: flash}), motion=motion, timefx=timefx,
                      look=look, audio=drop_music())

    s = edit.shots()
    zooms = s[2].effects
    assert len(zooms) == 1 and zooms[0].start == pytest.approx(1.0) and zooms[0].end == pytest.approx(1.5)
    assert [kf.t for kf in zooms[0].keyframes] == [1.2]
    assert s[2].motion.camera == "zoom" and s[4].look.bw
    shakes = s[3].effects
    assert len(shakes) == 1 and shakes[0].amplitude == pytest.approx(0.02)
    assert shakes[0].end == pytest.approx(1.95)
    assert [e.type for e in s[5].effects] == ["flash"] and not s[0].effects
    assert not s[7].effects  # the in-shot flash duplicated the flash transition
    for shot in s:
        for e in shot.effects:
            assert shot.start - 1e-9 <= e.start < e.end <= shot.end + 1e-9
    assert edit.stats.effect_counts == {"flash": 1, "shake": 1, "speed": 2, "zoom": 1}
    assert edit.find("s3.zoom1") is not None
    assert any(p.rule == "speed_ramp_on_beat" for p in edit.patterns)
    # the analyzer results were not modified
    assert motion_fx.by_shot[2][0].start == pytest.approx(0.9)


def test_json_round_trip_and_unique_ids():
    lengths = [15] * 16
    ctx = make_ctx(lengths)
    fx = EffectsResult()
    fx.add(4, Zoom(start=2.0, end=2.4, scale_to=1.2))
    fx.add(4, Shake(start=2.0, end=2.4, amplitude=0.01))
    seg = make_seg(lengths, extra=[TransitionSpan("fade_white", 0, 0, 6, None, 0)])
    text = TextAnalysis(overlays=[TextOverlay(start=0.0, end=1.0, text="SOUND ON")])
    edit = build_edit(ctx, seg, motion=MotionAnalysis(FrameMotion(*(np.zeros(240) for _ in range(7))), fx),
                      text=text, audio=drop_music(), warnings=["x: test warning"], timings={"decode": 1.23456})
    data = edit.to_json()
    again = Edit.from_json(data)
    assert again.to_json() == data
    assert len(again.shots()) == 16 and again.find("s5.shake1") is not None
    all_ids = ids(edit)
    assert all(all_ids) and len(all_ids) == len(set(all_ids))
    assert edit.provenance["warnings"] == ["x: test warning"]
    assert edit.provenance["timings"] == {"decode": 1.2346}


def test_degenerate_segmentations_are_repaired():
    lengths = [30] * 4
    ctx = make_ctx(lengths)
    empty = ShotSegmentation([], [], np.zeros(120))
    edit = build_edit(ctx, empty)
    assert len(edit.shots()) == 1 and edit.shots()[0].end == pytest.approx(4.0)
    assert any("single shot" in w for w in edit.provenance["warnings"])

    gappy = ShotSegmentation([ShotSpan(0, 0, 25), ShotSpan(1, 30, 70), ShotSpan(2, 70, 100)], [], np.zeros(120))
    edit = build_edit(ctx, gappy)
    shots = edit.shots()
    assert [(s.frame_start, s.frame_end) for s in shots] == [(0, 30), (30, 70), (70, 120)]
    assert all(a.end == b.start for a, b in zip(shots, shots[1:]))
    assert [t.kind for t in edit.transitions()] == ["cut", "cut"]


def test_repair_segmentation_enforces_the_contract():
    seg = ShotSegmentation(
        shots=[ShotSpan(5, 40, 70), ShotSpan(3, 0, 25), ShotSpan(4, 30, 40), ShotSpan(9, 30, 35)],
        transitions=[TransitionSpan("dissolve", 30, 26, 34, 3, 4, confidence=0.4),
                     TransitionSpan("flash", 30, 29, 31, 3, 4, confidence=0.9),  # more confident: wins
                     TransitionSpan("cut", 0, 0, 0, None, 3),  # a cut from nothing
                     TransitionSpan("wipe", 40, 40, 44, 4, 5),  # not a TransitionKind
                     TransitionSpan("fade_black", 100, 95, 130, 5, None)],  # runs past the end
        boundary_score=np.ones(50))
    clean, index_map, notes = repair_segmentation(seg, 100)
    assert [(s.index, s.f0, s.f1) for s in clean.shots] == [(0, 0, 30), (1, 30, 40), (2, 40, 100)]
    assert index_map == {3: 0, 4: 1, 9: 1, 5: 2}
    assert [(t.kind, t.from_shot, t.to_shot, t.f0, t.f1, t.boundary) for t in clean.transitions] == [
        ("flash", 0, 1, 29, 31, 30), ("unknown", 1, 2, 40, 44, 40), ("fade_black", 2, None, 95, 100, 100)]
    assert clean.boundary_score.shape == (100,)
    assert any("contiguous" in n for n in notes) and any("dropped 2" in n for n in notes)
    again, _, notes_again = repair_segmentation(clean, 100)
    assert notes_again == []
    assert [(t.kind, t.f0, t.f1) for t in again.transitions] == [(t.kind, t.f0, t.f1) for t in clean.transitions]


def test_source_groups_survive_h264_round_trip(tmp_path):
    w, h = 96, 160
    a = synth.scene(60, h, w, seed=1, velocity=(2.0, 1.0))
    parts = [a[:20], synth.scene(20, h, w, seed=2), a[40:60], synth.scene(20, h, w, seed=3)]
    frames = np.concatenate(parts)
    rng = np.random.default_rng(0)
    noisy = np.clip(frames.astype(np.int16) + rng.normal(0, 6, frames.shape).astype(np.int16), 0, 255).astype(np.uint8)
    path = tmp_path / "groups.mp4"
    with VideoWriter(path, w, h, 30, crf=30) as vw:
        for f in noisy:
            vw.write(f)
    ctx = load_context(path, AnalysisConfig(analysis_long_side=160))
    assert ctx.n_frames == 80
    edit = build_edit(ctx, make_seg([20, 20, 20, 20]))
    assert [s.source_group for s in edit.shots()] == ["A", "B", "A", "C"]
    rec = next(p for p in edit.patterns if p.rule == "recurring_source")
    assert rec.params["groups"] == ["A"] and rec.params["returns"] == 1


def test_build_is_fast_on_a_minute_long_edit():
    import time

    n_shots, shot_len = 200, 9  # 1800 frames at 30 fps
    rng = np.random.default_rng(1)
    hist = rng.dirichlet(np.ones(128), size=n_shots).repeat(shot_len, axis=0)
    ctx = make_ctx([30])
    n = n_shots * shot_len
    ctx.frames = np.zeros((n, 2, 2, 3), np.uint8)
    ctx.features.hist = hist
    lengths = [shot_len] * n_shots
    fx = EffectsResult()
    for i in range(n_shots):
        fx.add(i, Zoom(start=i * 0.3, end=i * 0.3 + 0.2))
    music = music_120(60.0, [AudioSection(start=0.0, end=20.0, label="build", energy=0.4),
                             AudioSection(start=20.0, end=60.0, label="drop", energy=0.9)], [20.0])
    t0 = time.perf_counter()
    edit = build_edit(ctx, make_seg(lengths), motion=MotionAnalysis(FrameMotion(*(np.zeros(n) for _ in range(7))), fx),
                      audio=music)
    assert time.perf_counter() - t0 < 5.0
    assert len(edit.shots()) == n_shots
