"""Tests for the shared foundation: AST, media I/O, context, fx, synth."""

import numpy as np
import pytest

from editast import fx
from editast.ast import (
    Edit, Flash, Look, MediaInfo, Music, NodeVisitor, Section, Shot, TextOverlay,
    Timeline, Transition, Zoom, assign_ids, shot_id,
)
from editast.context import AnalysisConfig, context_from_arrays, load_context
from editast.media import VideoWriter, aspect_string, fit_size, probe, read_audio, read_video, write_wav
from editast.testing import synth


def make_edit() -> Edit:
    s0 = Shot(index=0, start=0.0, end=1.0, frame_start=0, frame_end=30, look=Look(),
              effects=[Zoom(start=0.0, end=1.0, scale_to=1.3, easing="ease_out")])
    tr = Transition(kind="flash", start=0.95, end=1.05, from_shot=shot_id(0), to_shot=shot_id(1),
                    color=(255, 255, 255))
    s1 = Shot(index=1, start=1.0, end=2.0, frame_start=30, frame_end=60,
              effects=[Flash(start=1.5, end=1.6)])
    edit = Edit(
        media=MediaInfo(source="x.mp4", duration=2.0, width=1080, height=1920, aspect="9:16"),
        music=Music(bpm=120.0, beats=[0.0, 0.5, 1.0, 1.5]),
        timeline=Timeline(sections=[Section(role="hook", start=0.0, end=2.0, items=[s0, tr, s1])]),
        overlays=[TextOverlay(start=0.0, end=1.0, text="WAIT", bbox=(0.1, 0.1, 0.8, 0.1))],
    )
    return assign_ids(edit)


def test_ast_roundtrip_and_queries():
    edit = make_edit()
    again = Edit.from_json(edit.to_json())
    assert again == edit
    assert [s.id for s in edit.shots()] == ["s1", "s2"]
    assert edit.transitions()[0].id == "t1"
    assert edit.find("s1.zoom1").scale_to == pytest.approx(1.3)
    assert edit.cut_times() == [1.0]
    assert "properties" in Edit.json_schema()

    class Counter(NodeVisitor):
        def __init__(self):
            self.types = []

        def generic_visit(self, node):
            self.types.append(node.node_type)
            super().generic_visit(node)

    c = Counter()
    c.visit(edit)
    assert c.types.count("shot") == 2 and "zoom" in c.types and "flash" in c.types


def test_ast_rejects_unknown_fields():
    with pytest.raises(Exception):
        Shot(index=0, bogus=1)


def test_easing_endpoints():
    for name in fx.EASINGS:
        assert fx.ease(1.0, name) == pytest.approx(1.0)
        if name != "step":
            assert fx.ease(0.0, name) == pytest.approx(0.0)
    assert fx.ease(0.5, "ease_out") > 0.5 > fx.ease(0.5, "ease_in")


def test_media_roundtrip(tmp_path):
    w, h, fps = 160, 288, 30
    frames, bounds = synth.cut_sequence([20, 25, 15], height=h, width=w)
    audio = synth.click_track(120, frames.shape[0] / fps, sr=22050)
    wav = write_wav(tmp_path / "a.wav", audio, 22050)
    out = tmp_path / "v.mp4"
    with VideoWriter(out, w, h, fps, audio_path=wav) as vw:
        for f in frames:
            vw.write(f)
    p = probe(out)
    assert (p.width, p.height) == (w, h)
    assert p.fps == pytest.approx(30)
    assert p.has_audio and p.duration == pytest.approx(2.0, abs=0.1)
    back = read_video(out)
    assert back.shape[0] == frames.shape[0]
    assert np.abs(back.astype(int) - frames.astype(int)).mean() < 12
    a = read_audio(out)
    assert a.size > 0.9 * audio.size

    ctx = load_context(out, AnalysisConfig(analysis_long_side=144))
    assert ctx.size == (80, 144)
    assert ctx.n_frames == frames.shape[0]
    assert ctx.info.aspect == aspect_string(w, h)
    top = sorted(np.argsort(ctx.features.hist_dist)[-2:].tolist())
    assert top == bounds


def test_features_detect_cuts_in_memory():
    frames, bounds = synth.cut_sequence([12, 12, 12])
    ctx = context_from_arrays(frames, fps=24)
    top = sorted(np.argsort(ctx.features.hist_dist)[-2:].tolist())
    assert top == bounds
    assert ctx.f(0.5) == 12 and ctx.t(12) == pytest.approx(0.5)


def test_fit_size_and_aspect():
    assert fit_size(1080, 1920, 320) == (180, 320)
    assert aspect_string(1080, 1920) == "9:16"
    assert aspect_string(1920, 1080) == "16:9"
