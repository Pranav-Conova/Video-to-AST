"""Tests for the human-facing outputs: rich tree, plain summary and HTML report."""

from __future__ import annotations

import base64
import io
import re
from html.parser import HTMLParser

import cv2
import numpy as np
import pytest
from rich.console import Console
from rich.tree import Tree

from editast.ast import (
    Curve, Drop, Edit, EditStats, Flash, MediaInfo, Music, Pattern, Section, Shake, Shot, TextOverlay, Timeline,
    Transition, Zoom, assign_ids, shot_id,
)
from editast.context import AnalysisConfig, context_from_arrays, load_context
from editast.media import VideoWriter
from editast.report.html import report_html, write_report
from editast.report.tree import build_tree, edit_metrics, print_tree, summary_text
from editast.testing import synth

from test_lang import rich_edit

VOID = {"meta", "br", "img", "input", "link", "hr", "source", "wbr"}


class _Checker(HTMLParser):
    """Collects tags and verifies that non-void elements are properly nested."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: list[tuple[str, dict]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()


def check_html(text: str) -> _Checker:
    checker = _Checker()
    checker.feed(text)
    checker.close()
    assert not checker.errors, checker.errors[:5]
    assert not checker.stack, checker.stack
    return checker


def assert_self_contained(text: str) -> None:
    assert "http://" not in text and "https://" not in text
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?!data:|#)""", text)
    assert "@import" not in text and "url(" not in text


def cut_edit(lengths: list[int], fps: float) -> Edit:
    """A simple edit whose shots follow ``lengths`` (frames) with beats every shot start."""
    items, f = [], 0
    for i, n in enumerate(lengths):
        start, end = f / fps, (f + n) / fps
        effects = [Zoom(start=start, end=end, scale_to=1.2)] if i % 2 == 0 else [Shake(start=start, end=end)]
        items.append(Shot(index=i, start=start, end=end, frame_start=f, frame_end=f + n, beats=1.0, on_beat=True,
                          effects=effects))
        if i + 1 < len(lengths):
            items.append(Transition(kind="cut", start=end, end=end, from_shot=shot_id(i), to_shot=shot_id(i + 1)))
        f += n
    duration = f / fps
    music = Music(bpm=120.0, beats=[k * 0.5 for k in range(int(duration / 0.5) + 1)], downbeats=[0.0, 2.0],
                  drops=[Drop(time=1.0, strength=1.8)], energy=Curve(dt=0.1, values=list(np.linspace(0, 1, 20))))
    return assign_ids(Edit(media=MediaInfo(source="cuts.mp4", duration=duration, width=180, height=320,
                                           aspect="9:16"),
                           music=music, timeline=Timeline(sections=[Section(role="hook", start=0, end=duration,
                                                                            items=items)])))


# --------------------------------------------------------------------------- tree & summary


def render_tree(tree: Tree) -> str:
    console = Console(record=True, width=220, file=io.StringIO(), color_system=None)
    console.print(tree)
    return console.export_text()


def test_tree_and_summary_on_empty_edit():
    edit = Edit()
    tree = build_tree(edit)
    assert isinstance(tree, Tree)
    assert "edit" in render_tree(tree)
    buf = io.StringIO()
    print_tree(edit, console=Console(file=buf, width=120))
    assert "stats 0 shots" in buf.getvalue()
    text = summary_text(edit)
    assert isinstance(text, str) and "0.00s" in text and "shots: 0" in text
    m = edit_metrics(edit)
    assert m.n_shots == 0 and m.cuts_per_minute is None and m.on_beat_ratio is None


def test_tree_one_line_per_node():
    edit = rich_edit()
    text = render_tree(build_tree(edit))
    assert "shot s1 0.00-1.00s (1.00s, 2 beats, on-beat) group A handheld b&w letterbox" in text
    assert "shot s2 1.00-1.50s (0.50s, 1 beat, off-beat)" in text
    assert "section sec1 hook 0.00-1.00s" in text and "music 128.0 BPM" in text
    assert "zoom 0.00-0.50s 1.00→1.30" in text and "speed 0.50-1.00s ramp_down 1.00×→0.30×" in text
    assert "transition t2 flash 0.95-1.05s s1→s2" in text and "∅→s1" in text
    assert 'text txt1 "WAIT FOR IT' in text and "pattern p1 cut_every_n_beats 83%" in text
    # The hard cut is hidden by default and shown on request.
    assert "t3 cut" not in text
    assert "transition t3 cut 1.50-1.50s s2→s8" in render_tree(build_tree(edit, show_cuts=True))
    n_effects = sum(len(s.effects) for s in edit.shots())
    effect_lines = [ln for ln in text.splitlines() if re.search(r"── (zoom|pan|rotate|shake|speed|freeze|flash|blur|"
                                                                r"invert|rgb_split) \d+\.\d\d-", ln)]
    assert len(effect_lines) == n_effects


def test_tree_max_shots_and_markup_safety():
    edit = rich_edit()
    edit.overlays[0].text = "[bold red]not markup[/]"
    text = render_tree(build_tree(edit, max_shots=1))
    assert "shot s1" in text and "shot s2" not in text and "… 2 more shots" in text
    assert "[bold red]not markup[/]" in text  # OCR text is shown literally, never as rich markup


def test_summary_text_rich_edit():
    text = summary_text(rich_edit())
    for fragment in ("2.00s · 9:16 (1080x1920) · 29.97 fps · audio", "128.0 BPM", "shots: 3 · mean 0.67s",
                     "60.0 cuts/min", "50% cuts on beat", "hook 1.00s", "effects: zoom 2",
                     "transitions: fade_black 1 · fade_white 1 · flash 1 (+1 hard cut)",
                     "patterns: zoom on drop 100% · cut_every_n_beats 83%"):
        assert fragment in text, fragment
    assert len(text.splitlines()) <= 10


def test_metrics_fall_back_to_computed_values_without_stats():
    edit = cut_edit([15, 15, 30], fps=30)
    m = edit_metrics(edit)
    assert m.n_shots == 3 and m.duration == pytest.approx(2.0)
    assert m.cuts_per_minute == pytest.approx(60.0) and m.on_beat_ratio == pytest.approx(1.0)
    assert m.hook == pytest.approx(2.0) and m.n_cuts == 2 and m.effect_counts == {"zoom": 2, "shake": 1}


# --------------------------------------------------------------------------- HTML


def test_report_without_ctx(tmp_path):
    edit = rich_edit()
    edit.overlays[1].text = "<script>alert('x')</script> & \"quotes\""
    path = write_report(edit, tmp_path / "sub" / "report.html")
    text = path.read_text(encoding="utf-8")
    assert path.exists() and "<svg" in text and text.startswith("<!doctype html>")
    assert_self_contained(text)
    checker = check_html(text)
    assert "<script>alert" not in text and "&lt;script&gt;alert" in text
    lanes = {t for tag, attrs in checker.tags if tag == "text" for t in [attrs.get("class")]}
    assert "lane" in lanes
    for label in ("sections", "shots", "transitions", "zoom", "rgb split", "text", "beats", "energy", "pacing"):
        assert f">{label}</text>" in text, label
    assert "<image" not in text  # no thumbnails without a context
    assert "cut_every_n_beats" in text and "EditScript (compact)" in text and "AST JSON" in text
    assert "prefers-color-scheme:dark" in text
    tips = [attrs["data-tip"] for _, attrs in checker.tags if "data-tip" in attrs]
    assert any(t.startswith("shot s1\n") for t in tips) and any(t.startswith("drop at 1.00s") for t in tips)


def test_report_with_ctx_thumbnails_and_signals(tmp_path):
    lengths = [20, 25, 15, 30]
    frames, _ = synth.cut_sequence(lengths, height=320, width=180)
    ctx = context_from_arrays(frames, fps=30)
    edit = cut_edit(lengths, fps=30)
    signals = {"hist_dist": (ctx.features.hist_dist, 1 / 30), "noisy": (np.array([0.0, np.nan, 1.0, np.inf]), 0.5)}
    path = write_report(edit, tmp_path / "r.html", ctx=ctx, signals=signals)
    text = path.read_text(encoding="utf-8")
    assert "<svg" in text
    assert_self_contained(text)
    checker = check_html(text)
    images = [attrs["href"] for tag, attrs in checker.tags if tag == "image"]
    assert len(images) == len(lengths)
    for href, shot in zip(images, edit.shots()):
        assert href.startswith("data:image/jpeg;base64,")
        thumb = cv2.imdecode(np.frombuffer(base64.b64decode(href.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        assert thumb.shape[0] == 96 and thumb.shape[1] == 54
        # Thumbnails are the shot's middle frame with correct channel order (RGB in ctx -> BGR for JPEG).
        mid = ctx.frames[ctx.f(0.5 * (shot.start + shot.end))]
        expected = cv2.resize(mid, (54, 96), interpolation=cv2.INTER_AREA).reshape(-1, 3).mean(0)
        got = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB).reshape(-1, 3).mean(0)
        assert np.abs(got - expected).max() < 8
    assert ">hist_dist</text>" in text and ">noisy</text>" in text
    assert len(text) < 200_000


def test_report_from_real_encoded_video(tmp_path):
    """Thumbnails come from an H.264 encode -> decode round trip, with a letterboxed, noisy source."""
    lengths = [18, 24, 18]
    frames, _ = synth.cut_sequence(lengths, height=256, width=144)
    frames = frames.copy()
    frames[:, :24] = 0  # letterbox bars
    frames[:, -24:] = 0
    rng = np.random.default_rng(0)
    frames = np.clip(frames.astype(np.int16) + rng.integers(-12, 13, frames.shape), 0, 255).astype(np.uint8)
    video = tmp_path / "clip.mp4"
    with VideoWriter(video, 144, 256, 30) as writer:
        for frame in frames:
            writer.write(frame)
    ctx = load_context(video, AnalysisConfig(analysis_long_side=256))
    edit = cut_edit(lengths, fps=ctx.fps)
    edit.media = ctx.info
    html_text = report_html(edit, ctx=ctx, signals={"diff": (ctx.features.diff, 1 / ctx.fps)})
    check_html(html_text)
    assert_self_contained(html_text)
    assert html_text.count("<image") == len(lengths)
    assert "clip.mp4" in html_text


def test_report_on_empty_and_partial_edits(tmp_path):
    for edit in (Edit(), Edit(music=Music(bpm=None)), Edit(stats=EditStats())):
        text = write_report(edit, tmp_path / "e.html").read_text(encoding="utf-8")
        check_html(text)
        assert_self_contained(text)
        assert "Untitled edit" in text and "no timeline" in text
    # Overlays only (no shots) still draw a timeline.
    only_text = Edit(media=MediaInfo(duration=3.0), overlays=[TextOverlay(start=0, end=3, text="hi")],
                     patterns=[Pattern(rule="x", support=0.5)])
    text = report_html(only_text)
    assert "<svg" in text and ">text</text>" in text


def test_report_scales_to_a_long_busy_edit():
    rng = np.random.default_rng(3)
    lengths = list(rng.integers(6, 20, size=150))
    edit = cut_edit([int(n) for n in lengths], fps=30)
    for shot in edit.shots()[::3]:
        shot.effects.append(Flash(start=shot.start, end=min(shot.end, shot.start + 0.05)))
    edit.overlays = [TextOverlay(start=k * 1.0, end=k * 1.0 + 2.5, text=f"caption {k}") for k in range(40)]
    text = report_html(edit)
    check_html(text)
    assert text.count('class="shot ') == 150
    assert len(text) < 1_500_000
