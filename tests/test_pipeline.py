"""Tests for editast.pipeline, editast.fetch and editast.cli (analyzers are stubbed)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

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
from editast.ast.nodes import Edit, Look, MotionSummary, Music, TextOverlay, Zoom
from editast.cli import main
from editast.context import AnalysisConfig
from editast.fetch import FetchError, download, is_url
from editast.media import VideoWriter, write_wav
from editast.pipeline import STAGES, analyze, signals_for_report, summarize_edits
from editast.testing import synth


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """2 s, 3-shot H.264 clip with a 120 BPM click track (a real encode / decode round trip)."""
    root = tmp_path_factory.mktemp("clip")
    w, h, fps = 96, 160, 30
    frames, _ = synth.cut_sequence([20, 20, 20], height=h, width=w)
    wav = write_wav(root / "music.wav", synth.click_track(120, 2.0), 22050)
    path = root / "velocity.mp4"
    with VideoWriter(path, w, h, fps, audio_path=wav) as vw:
        for f in frames:
            vw.write(f)
    return path


def _detect_shots(ctx):
    n = ctx.n_frames
    b = [0, 20, 40, n]
    shots = [ShotSpan(i, b[i], b[i + 1]) for i in range(3)]
    transitions = [TransitionSpan("cut", 20, 20, 20, 0, 1),
                   TransitionSpan("flash", 40, 39, 41, 1, 2, confidence=0.9, color=(255, 255, 255))]
    return ShotSegmentation(shots, transitions, ctx.features.hist_dist.copy())


def _analyze_motion(ctx, seg):
    n = ctx.n_frames
    fm = FrameMotion(scale=np.ones(n), rotation=np.zeros(n), tx=np.zeros(n), ty=np.zeros(n),
                     residual=np.zeros(n), valid=np.ones(n, bool), magnitude=np.full(n, 0.01))
    fx = EffectsResult(signals={"zoom": np.ones(n), "shake": np.zeros(n)})
    fx.add(1, Zoom(start=0.7, end=1.2, scale_to=1.25, easing="punch"))
    return MotionAnalysis(fm, fx, {0: MotionSummary(camera="pan", magnitude=0.01)})


def _analyze_time_effects(ctx, seg, motion):
    return EffectsResult(signals={"speed_rate": np.ones(ctx.n_frames)})


def _analyze_look(ctx, seg):
    return LookAnalysis(EffectsResult(signals={"flash_amount": np.zeros(ctx.n_frames)}),
                        looks={i: Look(bw=i == 2) for i in range(3)})


def _detect_text(ctx, seg):
    present = np.zeros(ctx.n_frames)
    present[:24] = 1
    return TextAnalysis(overlays=[TextOverlay(start=0.0, end=0.8, text="WAIT FOR IT", role="hook")],
                        signals={"text_present": present}, ocr_engine=None)


def _analyze_audio(ctx):
    beats = np.arange(0.0, 2.0, 0.5).tolist()
    music = Music(bpm=120.0, beat_offset=0.0, beats=beats, downbeats=beats[::4])
    return AudioAnalysis(music=music, signals={"onset_env": np.random.default_rng(0).random(80)}, hop_seconds=0.025)


STUBS = {
    "shots": _detect_shots, "motion": _analyze_motion, "timefx": _analyze_time_effects,
    "look": _analyze_look, "text": _detect_text, "audio": _analyze_audio,
}


@pytest.fixture
def stubs(monkeypatch):
    """Install stub analyzer modules; returns ``(calls, install)`` to override single stages."""
    calls: list[str] = []

    def install(name: str, func=None, missing: bool = False) -> None:
        module_name, func_name, _ = STAGES[name]
        if missing:
            monkeypatch.setitem(sys.modules, module_name, None)  # import raises ImportError
            return
        impl = func or STUBS[name]

        def wrapper(*args):
            calls.append(name)
            return impl(*args)

        module = types.ModuleType(module_name)
        setattr(module, func_name, wrapper)
        monkeypatch.setitem(sys.modules, module_name, module)

    for stage in STAGES:
        install(stage)
    return calls, install


# --------------------------------------------------------------------------- pipeline


def test_analyze_runs_every_stage_in_order(clip, stubs):
    calls, _ = stubs
    messages: list[str] = []
    result = analyze(clip, progress=messages.append)
    assert calls == ["shots", "motion", "timefx", "look", "text", "audio"]
    assert result.warnings == []
    edit = result.edit
    assert [s.id for s in edit.shots()] == ["s1", "s2", "s3"]
    assert [t.kind for t in edit.transitions()] == ["cut", "flash"]
    assert edit.shots()[1].effects[0].type == "zoom" and edit.shots()[2].look.bw
    assert edit.overlays[0].text == "WAIT FOR IT" and edit.music.bpm == 120.0
    assert edit.media.n_frames == 60 and edit.media.duration == pytest.approx(2.0)
    assert set(STAGES) | {"decode", "build", "total"} <= set(result.timings)
    prov = edit.provenance
    assert prov["analyzers"] == {"shots": "ok", "motion": "ok", "timefx": "ok", "look": "ok",
                                 "text": "ok (no ocr)", "audio": "ok"}
    assert prov["timings"]["total"] >= prov["timings"]["decode"] > 0
    assert any("running" in m for m in messages)
    assert Edit.from_json(edit.to_json()).to_json() == edit.to_json()


def test_failing_stage_becomes_a_warning(clip, stubs):
    calls, install = stubs

    def boom(ctx, seg):
        raise RuntimeError("tracker exploded")

    seen = {}

    def timefx(ctx, seg, motion):
        seen["motion"] = motion
        return EffectsResult()

    install("motion", boom)
    install("timefx", timefx)
    install("look", lambda ctx, seg: "not a LookAnalysis")
    result = analyze(clip)
    assert result.motion is None and seen["motion"] is None and result.look is None
    assert any("motion" in w and "tracker exploded" in w for w in result.warnings)
    assert any(w.startswith("look:") and "expected LookAnalysis" in w for w in result.warnings)
    status = result.edit.provenance["analyzers"]
    assert status["motion"] == "failed" and status["look"] == "invalid result" and status["text"] == "ok (no ocr)"
    assert len(result.edit.shots()) == 3 and not result.edit.shots()[1].effects
    assert result.edit.provenance["warnings"] == result.warnings


def test_missing_shot_detector_falls_back_to_one_shot(clip, stubs):
    calls, install = stubs
    install("shots", missing=True)
    install("audio", lambda ctx: (_ for _ in ()).throw(ValueError("bad audio")))
    result = analyze(clip)
    assert "shots" not in calls and "motion" in calls
    assert len(result.edit.shots()) == 1 and result.edit.shots()[0].end == pytest.approx(2.0)
    assert result.seg.shots[0].f1 == 60
    assert any("shots: analyzer unavailable" in w for w in result.warnings)
    assert any("single shot" in w for w in result.warnings)
    assert any("audio" in w and "bad audio" in w for w in result.warnings)
    assert result.edit.music is None


def test_segmentation_is_repaired_before_other_stages(clip, stubs):
    _, install = stubs
    seen = {}
    install("shots", lambda ctx: ShotSegmentation([ShotSpan(0, 0, 15), ShotSpan(1, 20, 40), ShotSpan(2, 40, 60)],
                                                  [], np.zeros(10)))

    def motion(ctx, seg):
        seen.update(spans=[(s.f0, s.f1) for s in seg.shots], kinds=[t.kind for t in seg.transitions],
                    score=seg.boundary_score.size)
        return _analyze_motion(ctx, seg)

    install("motion", motion)
    result = analyze(clip)
    assert seen == {"spans": [(0, 20), (20, 40), (40, 60)], "kinds": ["cut", "cut"], "score": 60}
    assert any(w.startswith("shots: repaired") for w in result.warnings)
    assert result.edit.provenance["analyzers"]["shots"] == "ok"


def test_disabled_stages_are_skipped(clip, stubs):
    calls, _ = stubs
    cfg = AnalysisConfig(enabled={"text": False, "audio": False, "timefx": False})
    result = analyze(clip, cfg)
    assert calls == ["shots", "motion", "look"]
    assert result.warnings == [] and result.text is None and result.audio is None
    status = result.edit.provenance["analyzers"]
    assert status["text"] == status["audio"] == status["timefx"] == "disabled"


def test_signals_for_report(clip, stubs):
    result = analyze(clip)
    sig = signals_for_report(result)
    for key in ("boundary_score", "frame_diff", "luma", "motion_magnitude", "zoom", "speed_rate",
                "flash_amount", "text_present", "onset_envelope"):
        assert key in sig, key
        values, dt = sig[key]
        assert values.ndim == 1 and values.size > 1 and dt > 0
    assert sig["luma"][1] == pytest.approx(1 / result.ctx.fps)
    assert sig["onset_envelope"][1] == pytest.approx(0.025)
    assert sig["luma"][0].size == 60


def test_missing_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        analyze("definitely/not/here.mp4")


def test_summarize_edits(clip, stubs):
    edit = analyze(clip).edit
    summary = summarize_edits([("a.mp4", edit), ("b.mp4", edit.model_copy(deep=True))])
    assert summary["n_videos"] == 2
    assert summary["patterns"]["hook_text"]["videos"] == 2
    assert summary["transitions"]["flash"]["fraction"] == 1.0
    assert summary["stats"]["n_shots"]["median"] == 3
    assert any("hook text" in line for line in summary["common"])
    json.dumps(summary)


# --------------------------------------------------------------------------- fetch


def test_is_url():
    assert is_url("https://www.tiktok.com/@user/video/123")
    assert is_url("http://youtu.be/abc")
    assert not is_url("C:\\videos\\clip.mp4") and not is_url("clips/a.mp4") and not is_url(Path("x.mp4"))


def test_download_without_yt_dlp_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    with pytest.raises(FetchError, match="yt-dlp"):
        download("https://example.com/v/1", tmp_path)


# --------------------------------------------------------------------------- cli


def _fake_module(monkeypatch, name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_cli_schema(tmp_path, capsys):
    out = tmp_path / "schema.json"
    assert main(["schema", "-o", str(out)]) == 0
    assert "properties" in json.loads(out.read_text(encoding="utf-8"))
    capsys.readouterr()
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "Edit"


def test_cli_show_json_and_missing_module(clip, stubs, tmp_path, capsys, monkeypatch):
    path = analyze(clip).edit.save(tmp_path / "e.edit.json")
    assert main(["show", str(path), "--format", "json"]) == 0
    shown = Edit.from_json(capsys.readouterr().out)
    assert len(shown.shots()) == 3
    monkeypatch.setitem(sys.modules, "editast.report.tree", None)
    assert main(["show", str(path)]) == 1
    err = capsys.readouterr().err
    assert "tree view is unavailable" in err and "editast.report.tree" in err
    assert main(["show", str(tmp_path / "missing.json")]) == 1


def test_cli_analyze_writes_outputs(clip, stubs, tmp_path, capsys, monkeypatch):
    def write_report(edit, path, ctx=None, signals=None):
        assert ctx is not None and "luma" in signals
        Path(path).write_text("<html></html>", encoding="utf-8")
        return Path(path)

    _fake_module(monkeypatch, "editast.lang.printer", to_dsl=lambda edit, compact=False: "edit {}\n")
    _fake_module(monkeypatch, "editast.report.html", write_report=write_report)
    _fake_module(monkeypatch, "editast.report.tree", print_tree=lambda edit, console=None: console.print("TREE"),
                 summary_text=lambda edit: "SUMMARY")
    out = tmp_path / "out"
    assert main(["analyze", str(clip), "-o", str(out), "--no-ocr"]) == 0
    for name in ("velocity.edit.json", "velocity.edit", "velocity.report.html"):
        assert (out / name).is_file(), name
    printed = capsys.readouterr().out
    assert "TREE" in printed and "SUMMARY" in printed
    assert Edit.load(out / "velocity.edit.json").provenance["config"]["ocr"] is False

    # Broken optional modules only skip their own outputs.
    monkeypatch.setitem(sys.modules, "editast.lang.printer", None)
    monkeypatch.setitem(sys.modules, "editast.report.tree", None)
    out2 = tmp_path / "out2"
    assert main(["analyze", str(clip), "-o", str(out2)]) == 0
    assert (out2 / "velocity.edit.json").is_file() and not (out2 / "velocity.edit").exists()
    captured = capsys.readouterr()
    assert "skipped velocity.edit" in captured.err and "pattern:" in captured.out


def test_cli_batch_and_compare(clip, stubs, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "editast.lang.printer", None)
    out = tmp_path / "batch"
    assert main(["batch", str(clip), str(clip), str(tmp_path / "nope.mp4"), "-o", str(out), "-q"]) == 1
    assert "no such file" in capsys.readouterr().err
    assert main(["batch", str(clip), str(clip), "-o", str(out), "-q"]) == 0
    summary = json.loads((out / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["n_videos"] == 2 and summary["failed"] == []
    assert (out / "velocity.edit.json").is_file() and (out / "velocity_2.edit.json").is_file()
    assert "hook_text" in capsys.readouterr().out

    _fake_module(monkeypatch, "editast.bench.evaluate", similarity=lambda a, b: 0.5,
                 compare=lambda a, b: {}, format_report=lambda r: "REPORT")
    a = str(out / "velocity.edit.json")
    assert main(["compare", a, a]) == 0
    assert "similarity: 0.500" in capsys.readouterr().out
    assert main(["compare", a, a, "--truth"]) == 0
    assert "REPORT" in capsys.readouterr().out


def test_cli_version_and_usage_errors(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        main(["render", "x.json"])  # --clips is required
    assert exc.value.code == 2
    assert main(["render", "missing.json", "--clips", "."]) == 1
