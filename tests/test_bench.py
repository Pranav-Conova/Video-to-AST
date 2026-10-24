"""Tests for the synthetic benchmark: case generation (editast.bench.generate) and scoring
(editast.bench.evaluate)."""

import json
import math

import numpy as np
import pytest

from editast.ast import (
    Drop, Edit, Look, MediaInfo, Music, Pan, Section, Shake, Shot, Speed, TextOverlay, Timeline,
    Transition, Word, Zoom, assign_ids, shot_id,
)
from editast.bench.evaluate import boundaries, compare, format_report, similarity
from editast.bench.generate import HOOKS, build_case, generate_case, generate_suite
from editast.context import AnalysisConfig, load_context
from editast.media import probe
from editast.render.compiler import render_frames, retime_edit
from editast.render.sources import SyntheticSource

SIZE = (180, 320)  # (width, height)
FPS = 30.0
SEED = 4  # has a fade-in, a dissolve, flashes, a freeze and speed ramps


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return generate_case(SEED, tmp_path_factory.mktemp("bench"), size=SIZE, fps=FPS)


# --------------------------------------------------------------------------- generator


def test_generate_case_writes_playable_video_and_truth(case):
    mp4, truth = case
    assert mp4.name == f"case_{SEED:04d}.mp4" and mp4.exists()
    saved = mp4.with_name(f"case_{SEED:04d}.truth.json")
    assert Edit.load(saved).to_dict() == truth.to_dict()
    p = probe(mp4)
    assert (p.width, p.height) == SIZE and p.fps == pytest.approx(FPS) and p.has_audio
    assert p.duration == pytest.approx(truth.media.duration, abs=0.1)


def test_truth_follows_the_ast_conventions(case):
    _, truth = case
    shots = truth.shots()
    duration = truth.media.duration
    assert 8.0 - 1e-6 <= duration <= 14.0 + 1e-6
    assert truth.media.n_frames == round(duration * FPS)
    # Shots partition the timeline; frame indices match the times.
    assert shots[0].start == 0.0 and shots[-1].end == pytest.approx(duration)
    for i, s in enumerate(shots):
        assert s.index == i and s.id == shot_id(i) and s.end > s.start
        assert (s.frame_start, s.frame_end) == (round(s.start * FPS), round(s.end * FPS))
        if i + 1 < len(shots):
            assert s.end == pytest.approx(shots[i + 1].start)
        for k, eff in enumerate(s.effects):
            assert s.start - 1e-6 <= eff.start < eff.end <= s.end + 1e-6, eff
            assert eff.id.startswith(f"{s.id}.{eff.type}")
    # Sections partition the timeline as well and hold their items in playback order.
    secs = truth.timeline.sections
    assert [sec.role for sec in secs][0] == "hook" and "drop" in [sec.role for sec in secs]
    assert secs[0].start == 0.0 and secs[-1].end == pytest.approx(duration)
    assert all(a.end == pytest.approx(b.start) for a, b in zip(secs, secs[1:]))
    # Transitions reference existing shots and sit on their boundary.
    by_id = {s.id: s for s in shots}
    kinds = {t.kind for t in truth.transitions()}
    assert kinds <= {"cut", "flash", "dissolve", "fade_black"} and "cut" in kinds
    for tr in truth.transitions():
        assert tr.id.startswith("t") and tr.start <= tr.end
        if tr.from_shot and tr.to_shot:
            a, b = by_id[tr.from_shot], by_id[tr.to_shot]
            assert b.index == a.index + 1 and tr.start <= b.start <= tr.end
            assert a.start < tr.start + 1e-6 and tr.end < b.end + 1e-6
            if tr.kind == "cut":
                assert tr.start == tr.end == b.start
        else:  # fade-in / fade-out
            fade_in = tr.from_shot is None and tr.start == 0.0
            assert fade_in or (tr.to_shot is None and tr.end == pytest.approx(duration))
    # Music: tempo grid, drop at 40-60 % of the runtime, sections covering the song.
    m = truth.music
    assert 90.0 <= m.bpm <= 160.0
    assert np.allclose(np.diff(m.beats), 60.0 / m.bpm, atol=2e-4) and m.beats[0] == pytest.approx(m.beat_offset)
    assert set(m.downbeats) <= set(m.beats)
    assert 0.4 * duration - 0.1 <= m.drops[0].time <= 0.6 * duration + 0.1
    assert m.sections[0].start == 0.0 and m.sections[-1].end == pytest.approx(duration)
    # Beat bookkeeping on shots: most cuts on the beat, some off.
    period = 60.0 / m.bpm
    for s in shots:
        assert s.beats == pytest.approx(s.duration / period, abs=2e-3)
        assert s.on_beat == (abs(s.beat_offset) <= 0.07)
    on = [s.on_beat for s in shots[1:]]
    assert 0.6 <= np.mean(on) < 1.0
    # Text: a hook title in the first ~1.5 s and a word-by-word caption.
    hook, caption = truth.overlays
    assert hook.role == "hook" and hook.start == 0.0 and hook.end <= 1.5 + 1e-6 and hook.text in HOOKS
    assert caption.animation == "word_by_word" and caption.text == " ".join(w.text for w in caption.words)
    assert caption.words[0].start == caption.start and caption.words[-1].end == caption.end
    assert all(a.end <= b.start + 1e-6 for a, b in zip(caption.words, caption.words[1:]))
    assert [o.id for o in truth.overlays] == ["txt1", "txt2"]
    # Exactly one black & white shot; source groups label every shot.
    assert sum(bool(s.look and s.look.bw) for s in shots) == 1
    assert all(s.source_group for s in shots)


def test_truth_pans_describe_the_camera_motion(case):
    """Moving shots carry an observable camera Pan; the other shots barely drift."""
    _, truth = case
    speeds = {int(k): v for k, v in truth.provenance["camera_px_per_frame"].items()}
    w = SIZE[0]
    n_pans = 0
    for s in truth.shots():
        pans = [e for e in s.effects if isinstance(e, Pan)]
        n_frames = s.frame_end - s.frame_start
        vx, vy = speeds.get(s.index, (0.0, 0.0))
        if pans:
            n_pans += 1
            (pan,) = pans
            assert pan.params == {"source": "camera"} and (pan.start, pan.end) == (s.start, s.end)
            assert math.hypot(pan.dx, pan.dy * SIZE[1] / w) >= 0.05  # above the pan threshold
            # Content moves against the camera.
            assert np.sign(pan.dx) == -np.sign(vx) and np.sign(pan.dy) == -np.sign(vy)
        else:
            assert math.hypot(vx, vy) * n_frames < 0.02 * w
    assert n_pans >= 1


def test_rendered_case_shows_the_truth_after_h264_roundtrip(case):
    mp4, truth = case
    ctx = load_context(mp4, AnalysisConfig(analysis_long_side=320))
    assert abs(ctx.n_frames - truth.media.n_frames) <= 1 and ctx.info.has_audio
    hist_dist, luma, sat = ctx.features.hist_dist, ctx.features.luma_mean, ctx.features.sat_mean
    shots = truth.shots()
    special = {truth.find(t.to_shot).frame_start: t.kind for t in truth.transitions()
               if t.kind != "cut" and t.from_shot and t.to_shot}
    # Hard cuts are clear colour-histogram jumps, unlike frames inside a shot.
    peaks = [hist_dist[b - 1:b + 2].max() for b in (s.frame_start for s in shots[1:]) if b not in special]
    inner = np.concatenate([hist_dist[s.frame_start + 2:s.frame_end - 1] for s in shots])
    assert min(peaks) > 0.5 and np.median(inner) < 0.05
    # Flash transitions peak white on the boundary frame; the B&W shot has no chroma.
    flashes = [b for b, kind in special.items() if kind == "flash"]
    assert flashes and all(luma[b] > 0.9 for b in flashes)
    bw = next(s for s in shots if s.look and s.look.bw)
    others = [np.median(sat[s.frame_start:s.frame_end]) for s in shots if s is not bw]
    assert np.median(sat[bw.frame_start + 2:bw.frame_end - 2]) < 0.03 < 0.3 < min(others)
    # Fade-in from black at the start.
    if any(t.from_shot is None for t in truth.transitions()):
        assert luma[0] < 0.05 < 0.2 < luma[truth.shots()[0].frame_end - 2]


def test_text_truth_matches_the_drawn_pixels():
    """Truth boxes are fx.draw_text layout boxes (what the text analyzer reports): they hold the
    drawn ink, whose height is the cap height plus outline (the box also spans the descender)."""
    case = build_case(SEED, size=SIZE, fps=FPS)
    edit = case.edit
    with_text = list(render_frames(edit, SyntheticSource(SEED, case.motion), size=SIZE, fps=FPS))
    plain = edit.model_copy(deep=True)
    plain.overlays = []
    without = list(render_frames(plain, SyntheticSource(SEED, case.motion), size=SIZE, fps=FPS))
    w, h = SIZE
    for ov in edit.overlays:
        span = ov.words[0] if ov.words else ov
        f = int(round(0.5 * (span.start + span.end) * FPS))
        ys, xs = np.nonzero(np.any(with_text[f] != without[f], axis=2))
        drawn = (xs.min() / w, ys.min() / h, (xs.max() + 1 - xs.min()) / w, (ys.max() + 1 - ys.min()) / h)
        x, y, bw, bh = ov.bbox
        assert x - 2 / w <= drawn[0] and drawn[0] + drawn[2] <= x + bw + 2 / w  # inside the truth box
        assert y - 2 / h <= drawn[1] and drawn[1] + drawn[3] <= y + bh + 2 / h
        assert drawn[3] > 0.5 * bh  # cap-height ink
        if not ov.words:
            assert drawn[2] > 0.85 * bw  # a whole line spans the box
    f_after = int(round(edit.overlays[-1].end * FPS)) + 1
    assert np.array_equal(with_text[f_after], without[f_after])


def test_seeds_cover_the_effect_and_transition_vocabulary(tmp_path):
    cases = [build_case(seed, size=SIZE).edit for seed in range(10)]
    effects = set().union(*({e.type for _, e in c.effects()} for c in cases))
    assert {"zoom", "pan", "shake", "speed", "freeze", "flash", "rgb_split", "invert"} <= effects
    kinds = set().union(*({t.kind for t in c.transitions()} for c in cases))
    assert {"cut", "flash", "dissolve", "fade_black"} <= kinds
    fade_in = [any(t.from_shot is None for t in c.transitions()) for c in cases]
    assert any(fade_in) and not all(fade_in)
    mixes = {tuple(sorted(c.stats.effect_counts.items())) for c in cases}
    assert len(mixes) == len(cases)  # every seed is a different edit
    speeds = [(s, e) for c in cases for s, e in c.effects() if isinstance(e, Speed)]
    assert {e.kind for _, e in speeds} == {"slowmo", "ramp_down", "ramp_up"}
    assert all(any(isinstance(x, Pan) for x in s.effects) for s, _ in speeds)  # only on moving shots
    assert all(0.3 - 1e-9 <= min(e.rate_from, e.rate_to) <= 0.5 + 1e-9 for _, e in speeds)
    assert build_case(3, size=SIZE).edit == build_case(3, size=SIZE).edit  # deterministic

    suite = generate_suite(tmp_path, n=2, seed=20, size=SIZE, duration=8.0)
    assert [p.name for p, _ in suite] == ["case_0020.mp4", "case_0021.mp4"]
    assert all((tmp_path / f"case_00{k}.truth.json").exists() for k in (20, 21))
    assert all(t.media.duration == pytest.approx(8.0) for _, t in suite)


# --------------------------------------------------------------------------- evaluator


def small_edit() -> Edit:
    """Four 1 s shots at 120 BPM: flash, cut, dissolve; zoom, shake, pan, slow motion; B&W shot 2."""
    shots = [Shot(index=i, start=float(i), end=float(i + 1), frame_start=30 * i, frame_end=30 * (i + 1))
             for i in range(4)]
    shots[0].effects = [Zoom(start=0.0, end=1.0, scale_to=1.3, easing="ease_out")]
    shots[1].look = Look(bw=True)
    shots[2].effects = [Pan(start=2.0, end=3.0, dx=0.1, params={"source": "camera"}),
                        Shake(start=2.0, end=2.5, amplitude=0.01, frequency=8.0)]
    shots[3].effects = [Speed(start=3.2, end=4.0, kind="slowmo", rate_from=0.5, rate_to=0.5)]
    items = [shots[0], Transition(kind="flash", start=0.93, end=1.07, from_shot="s1", to_shot="s2"),
             shots[1], Transition(kind="cut", start=2.0, end=2.0, from_shot="s2", to_shot="s3"),
             shots[2], Transition(kind="dissolve", start=2.8, end=3.2, from_shot="s3", to_shot="s4"),
             shots[3]]
    edit = Edit(
        media=MediaInfo(duration=4.0, width=180, height=320, analysis_fps=FPS, n_frames=120),
        music=Music(bpm=120.0, beat_offset=0.0, beats=[0.5 * k for k in range(8)], downbeats=[0.0, 2.0],
                    drops=[Drop(time=2.0)]),
        timeline=Timeline(sections=[Section(role="hook", start=0.0, end=4.0, items=items)]),
        overlays=[TextOverlay(start=0.0, end=1.0, text="WAIT FOR IT", bbox=(0.1, 0.2, 0.8, 0.06), role="hook"),
                  TextOverlay(start=2.0, end=3.0, text="NO WAY", bbox=(0.3, 0.7, 0.4, 0.05),
                              animation="word_by_word",
                              words=[Word(text="NO", start=2.0, end=2.5), Word(text="WAY", start=2.5, end=3.0)])],
    )
    return assign_ids(edit)


def move_cut(edit: Edit, i: int, dt: float) -> None:
    """Move the boundary between shot ``i - 1`` and shot ``i`` by ``dt`` seconds."""
    shots = edit.shots()
    shots[i - 1].end += dt
    shots[i].start += dt


def test_compare_identical_edits_is_perfect(case):
    for truth in (small_edit(), case[1]):
        report = compare(truth, truth.model_copy(deep=True))
        assert report["overall"] == pytest.approx(1.0)
        assert report["cuts"]["f1"] == 1.0 and report["cuts"]["mean_offset"] == 0.0
        assert report["transitions"]["accuracy"] == 1.0 and report["transitions"]["false_positives"] == 0
        assert report["effects"]["f1"] == 1.0
        assert all(e["f1"] == 1.0 for e in report["effects"]["by_type"].values())
        mu = report["music"]
        assert mu["beat_f"] == 1.0 and mu["bpm_ok"] and mu["drop_error"] == 0.0
        tx = report["text"]
        assert tx["f1"] == 1.0 and tx["bbox_iou"] == pytest.approx(1.0) and tx["ocr_similarity"] == 1.0
        assert report["look"]["bw_accuracy"] == 1.0
        json.dumps(report)  # the CLI stores reports as JSON
    text = format_report(report)
    lines = text.splitlines()
    assert lines[0].split()[:4] == ["metric", "P", "R", "F1"] and lines[-1].split() == ["overall", "1.000"]
    assert {"cuts", "transition", "effects", "beats", "text", "look"} <= {ln.split()[0] for ln in lines[2:]}
    f1_col = lines[0].index("F1") + 2
    assert all(ln[f1_col - 5:f1_col] in ("1.000", "    -") for ln in lines[2:])  # aligned columns


def test_compare_scores_degraded_predictions():
    truth = small_edit()
    # Cuts: one frame late is still a hit, 0.1 s late is a miss.
    pred = small_edit()
    for i in (1, 2, 3):
        move_cut(pred, i, 1 / FPS)
    cuts = compare(truth, pred)["cuts"]
    assert cuts["f1"] == 1.0 and cuts["mean_offset"] == pytest.approx(1 / FPS)
    move_cut(pred, 2, 0.1)
    cuts = compare(truth, pred)["cuts"]
    assert (cuts["precision"], cuts["recall"]) == pytest.approx((2 / 3, 2 / 3))

    # Transitions: the flash predicted as a hard cut, the dissolve recognised.
    pred = small_edit()
    pred.timeline.sections[0].items[1].kind = "cut"
    tr = compare(truth, pred)["transitions"]
    assert tr["matched"] == 2 and tr["accuracy"] == 0.5 and tr["detected"] == 0.5
    assert tr["confusion"] == {"flash": {"cut": 1}, "dissolve": {"dissolve": 1}}
    pred.timeline.sections[0].items[3].kind = "fade_black"  # a dip on a truth hard cut
    assert compare(truth, pred)["transitions"]["false_positives"] == 1

    # Effects: a missing zoom, a wrong pan direction, a sped-up "slow motion", a stray blip.
    pred = small_edit()
    shots = pred.shots()
    shots[0].effects = []
    shots[2].effects[0].dx = -0.1
    shots[3].effects[0].rate_from = shots[3].effects[0].rate_to = 1.5
    shots[3].effects.append(Shake(start=3.9, end=4.0, amplitude=0.01, frequency=8.0))  # IoU 0 with the truth shake
    ef = compare(truth, pred)["effects"]
    assert ef["by_type"]["zoom"]["recall"] == 0.0 and ef["by_type"]["zoom"]["n_pred"] == 0
    assert ef["by_type"]["pan"]["direction_agreement"] == 0.0
    assert ef["by_type"]["speed"]["slow_fast_agreement"] == 0.0
    assert ef["by_type"]["shake"]["precision"] == 0.5 and ef["by_type"]["shake"]["frequency_error_hz"] == 0.0
    assert (ef["precision"], ef["recall"]) == pytest.approx((3 / 4, 3 / 4))
    # A zoom recovered at 70 % of its span with a 10 % ratio error still matches.
    shots[0].effects = [Zoom(start=0.3, end=1.0, scale_from=1.0, scale_to=1.43)]
    zoom = compare(truth, pred)["effects"]["by_type"]["zoom"]
    assert zoom["f1"] == 1.0 and zoom["ratio_rel_error"] == pytest.approx(0.1)

    # Music: double tempo is wrong but octave-equivalent; half the beats are extra.
    pred = small_edit()
    pred.music.bpm = 240.0
    pred.music.beats = [0.25 * k for k in range(16)]
    pred.music.drops = [Drop(time=2.25)]
    mu = compare(truth, pred)["music"]
    assert not mu["bpm_ok"] and mu["bpm_ok_octave"] and mu["bpm_rel_error"] == pytest.approx(1.0)
    assert (mu["beat_precision"], mu["beat_recall"]) == (0.5, 1.0) and mu["drop_error"] == pytest.approx(0.25)
    pred.music = None
    assert compare(truth, pred)["music"]["beat_f"] == 0.0

    # Text and look: an OCR typo, a shifted box, a lost caption and a lost B&W grade.
    pred = small_edit()
    pred.overlays[0].text = "WAIT F0R IT"
    pred.overlays[0].bbox = (0.1, 0.21, 0.8, 0.06)
    del pred.overlays[1]
    pred.shots()[1].look = None
    report = compare(truth, pred)
    tx = report["text"]
    assert (tx["precision"], tx["recall"]) == (1.0, 0.5)
    assert 0.8 < tx["ocr_similarity"] < 1.0 and 0.6 < tx["bbox_iou"] < 0.9
    assert report["look"]["bw_accuracy"] == 0.75 and report["look"]["bw_recall"] == 0.0
    assert report["overall"] < 1.0
    assert "WAIT" not in format_report(report)  # the table reports scores, not content


def test_boundaries_include_fades_and_transition_kinds():
    edit = small_edit()
    edit.timeline.sections[0].items.insert(0, Transition(kind="fade_black", start=0.0, end=0.3, to_shot="s1"))
    assign_ids(edit)
    assert [(round(t, 3), k) for t, k, _ in boundaries(edit)] == [
        (0.0, "fade_black"), (1.0, "flash"), (2.0, "cut"), (3.0, "dissolve")]


def test_similarity_compares_editing_grammar():
    a = build_case(1, size=SIZE).edit
    b = build_case(2, size=SIZE).edit
    assert similarity(a, a) == 1.0
    assert similarity(a, b) == similarity(b, a) < 1.0
    # Same edit on a faster song: shot lengths in beats and the effect mix are unchanged.
    faster = retime_edit(a, a.music.bpm * 1.2)
    assert similarity(a, faster) > similarity(a, b)
    # A slideshow (three long shots, no effects, no music) shares little with a velocity edit.
    slides = [Shot(index=i, start=4.0 * i, end=4.0 * (i + 1)) for i in range(3)]
    slideshow = assign_ids(Edit(media=MediaInfo(duration=12.0, width=180, height=320),
                                timeline=Timeline(sections=[Section(start=0.0, end=12.0, items=slides)])))
    assert similarity(a, slideshow) < 0.5 < similarity(a, b)
    assert 0.0 <= similarity(slideshow, Edit()) <= 1.0  # degenerate edits do not crash
