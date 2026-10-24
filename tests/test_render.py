"""Tests for the AST compiler (editast.render) and its footage sources."""

import cv2
import numpy as np
import pytest

from editast import fx
from editast.ast import (
    Curve, Drop, Edit, EditStats, Freeze, Keyframe, Look, MediaInfo, Music, Pan, Rotate, Section, Shake,
    Shot, Speed, TextOverlay, Timeline, Transition, Word, Zoom, assign_ids, shot_id,
)
from editast.ast.nodes import AudioSection, Pattern
from editast.context import AnalysisConfig, load_context
from editast.media import VideoWriter, probe, read_video, write_wav
from editast.render.compiler import (
    output_format, render_edit, render_frames, retime_edit, shot_geometry, source_positions, text_box,
)
from editast.render.sources import SyntheticSource, VideoClipSource
from editast.testing import synth

FPS = 30.0
SIZE = (180, 320)  # (width, height)
COLORS = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40)]


def three_shot_edit(fade_in: bool = False, fade_out: bool = False) -> Edit:
    """Zoom shot -> flash -> B&W shot -> dissolve -> shot, with a hook title."""
    s0 = Shot(index=0, start=0.0, end=1.0, frame_start=0, frame_end=30,
              effects=[Zoom(start=0.0, end=1.0, scale_to=1.3, easing="ease_out")])
    s1 = Shot(index=1, start=1.0, end=2.0, frame_start=30, frame_end=60, look=Look(bw=True))
    s2 = Shot(index=2, start=2.0, end=3.0, frame_start=60, frame_end=90)
    items = [s0,
             Transition(kind="flash", start=0.9, end=1.1, from_shot=shot_id(0), to_shot=shot_id(1)),
             s1,
             Transition(kind="dissolve", start=1.8, end=2.2, from_shot=shot_id(1), to_shot=shot_id(2)),
             s2]
    if fade_in:
        items.insert(0, Transition(kind="fade_black", start=0.0, end=0.3, from_shot=None, to_shot=shot_id(0)))
    if fade_out:
        items.append(Transition(kind="fade_white", start=2.7, end=3.0, from_shot=shot_id(2), to_shot=None))
    edit = Edit(
        media=MediaInfo(duration=3.0, width=360, height=640, analysis_fps=FPS, n_frames=90),
        music=Music(bpm=120.0, beat_offset=0.0, beats=[0.5 * k for k in range(6)]),
        timeline=Timeline(sections=[Section(role="hook", start=0.0, end=3.0, items=items)]),
        overlays=[TextOverlay(start=0.0, end=1.0, text="WAIT FOR IT", bbox=(0.1, 0.2, 0.8, 0.06))],
    )
    return assign_ids(edit)


class SolidSource:
    """Each shot is a flat colour, so compositing can be checked exactly."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def prepare(self, edit, size, fps):
        self.calls = []

    def frames_for_shot(self, shot, n_source_frames, size):
        self.calls.append((shot.index, n_source_frames))
        w, h = size
        return np.broadcast_to(np.uint8(COLORS[shot.index % len(COLORS)]), (n_source_frames, h, w, 3)).copy()


def chroma(frame: np.ndarray) -> float:
    f = frame.astype(np.int16)
    return float(np.abs(f[..., 0] - f[..., 1]).mean() + np.abs(f[..., 1] - f[..., 2]).mean())


# --------------------------------------------------------------------------- compositing


def test_compositing_is_exact_with_flat_sources():
    edit = three_shot_edit(fade_in=True, fade_out=True)
    src = SolidSource()
    frames = np.stack(list(render_frames(edit, src, size=SIZE)))
    assert frames.shape == (90, 320, 180, 3)
    # Shot 1 renders 6 handle frames after its cut, shot 2 six before (dissolve 1.8-2.2 s).
    assert src.calls == [(0, 30), (1, 36), (2, 36)]
    assert np.all(frames[0, 120:] == 0)  # fade-in starts from black (the title is a layer on top)
    assert (frames[0, 60:90] == 255).all(axis=2).sum() > 30
    assert np.all(frames[15, 300] == COLORS[0])  # below the title; zooming a flat colour changes nothing
    assert np.all(frames[30] == 255)  # flash transition peaks on the boundary frame
    expected27 = fx.flash(frames[27].copy() * 0 + np.uint8(COLORS[0]), 0.25)
    assert np.abs(frames[27, 300].astype(int) - expected27[300].astype(int)).max() <= 1
    gray = fx.to_bw(np.full((1, 1, 3), COLORS[1], np.uint8))[0, 0]
    assert np.all(frames[45] == gray)  # B&W shot: equal channels
    for f in (55, 59, 60, 64):  # dissolve: (f - 54 + 0.5) / 12 blend of the two shots
        alpha = (f - 54 + 0.5) / 12
        expected = (1 - alpha) * gray.astype(float) + alpha * np.array(COLORS[2], float)
        assert np.abs(frames[f, 160, 90].astype(float) - expected).max() <= 1.0
    assert np.all(frames[-1] == 255)  # fade-out to white


def test_text_overlay_and_word_by_word():
    edit = three_shot_edit()
    edit.overlays.append(TextOverlay(start=1.0, end=2.0, text="FAST CARS", bbox=(0.2, 0.7, 0.6, 0.06),
                                     animation="word_by_word", color=(255, 220, 0),
                                     words=[Word(text="FAST", start=1.0, end=1.5),
                                            Word(text="CARS", start=1.5, end=2.0)]))
    with_text = np.stack(list(render_frames(edit, SolidSource(), size=SIZE)))
    edit.overlays = []
    plain = np.stack(list(render_frames(edit, SolidSource(), size=SIZE)))
    h, w = SIZE[1], SIZE[0]
    changed = np.any(with_text != plain, axis=3)
    # Hook title: drawn inside its bbox only, white with a dark outline.
    box = text_box("WAIT FOR IT", (0.5, 0.23), 0.06, SIZE)
    x0, y0 = int(box[0] * w) - 2, int(box[1] * h) - 2
    x1, y1 = int((box[0] + box[2]) * w) + 2, int((box[1] + box[3]) * h) + 2
    assert changed[10, y0:y1, x0:x1].mean() > 0.1
    outside = changed[10].copy()
    outside[y0:y1, x0:x1] = False
    assert not outside.any()
    assert (with_text[10, y0:y1, x0:x1] == 255).all(axis=2).sum() > 30
    assert not changed[35, : h // 2].any()  # title gone after 1 s
    # Word by word: a different word in each half, yellow.
    lower = slice(int(0.65 * h), int(0.82 * h))
    a, b = changed[40, lower], changed[50, lower]
    assert a.sum() > 30 and b.sum() > 30
    assert np.any(with_text[40, lower] != with_text[50, lower], axis=2).sum() > 100  # FAST vs CARS
    assert np.array_equal(with_text[40], with_text[44]) and not changed[65].any()  # same word; gone after 2 s
    yellow = with_text[40, lower]
    assert ((yellow[..., 0] > 200) & (yellow[..., 1] > 180) & (yellow[..., 2] < 120)).sum() > 20


def test_caption_block_is_laid_out_on_several_lines():
    """A box two ``size`` lines tall (how the text analyzer reports a caption block) gets two lines."""
    edit = three_shot_edit()
    edit.overlays = [TextOverlay(start=0.0, end=1.0, text="THIS CAR IS INSANE", bbox=(0.1, 0.4, 0.8, 0.12),
                                 size=0.06)]
    with_text = np.stack(list(render_frames(edit, SolidSource(), size=SIZE)))
    edit.overlays = []
    plain = np.stack(list(render_frames(edit, SolidSource(), size=SIZE)))
    rows = np.flatnonzero(np.any(with_text[10] != plain[10], axis=(1, 2)))
    h = SIZE[1]
    assert 0.4 * h - 2 <= rows.min() and rows.max() <= 0.52 * h + 2  # inside the box
    bands = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    assert len(bands) == 2 and all(len(b) >= 8 for b in bands)  # two text lines


def test_explicit_line_breaks_are_kept():
    """A stacked block ("A" over "BIG RED CAR") keeps its breaks instead of being rebalanced."""
    edit = three_shot_edit()
    edit.overlays = [TextOverlay(start=0.0, end=1.0, text="A\nBIG RED CAR", bbox=(0.1, 0.4, 0.8, 0.12),
                                 size=0.06)]
    with_text = np.stack(list(render_frames(edit, SolidSource(), size=SIZE)))
    edit.overlays = []
    plain = np.stack(list(render_frames(edit, SolidSource(), size=SIZE)))
    changed = np.any(with_text[10] != plain[10], axis=2)
    rows = np.flatnonzero(changed.any(axis=1))
    bands = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    assert len(bands) == 2
    widths = [np.ptp(np.flatnonzero(changed[b].any(axis=0))) for b in bands]
    assert widths[0] < 0.4 * widths[1]  # "A" alone on the first line


# --------------------------------------------------------------------------- timing


def test_source_positions_speed_and_freeze():
    shot = Shot(index=0, start=0.0, end=1.0, effects=[
        Freeze(start=0.3, end=0.5),
        Speed(start=0.5, end=1.0, kind="slowmo", rate_from=0.5, rate_to=0.5),
    ])
    pos = source_positions(shot, FPS)
    assert pos.size == 30 and pos[0] == 0
    assert np.all(np.diff(pos[:9]) == 1)  # normal speed
    assert np.all(pos[9:15] == pos[9])  # frames 9..14 show the held picture
    assert pos[15] == pos[14] + 1  # footage resumes where it stopped
    assert np.allclose(np.diff(pos[15:]), 0.5)  # 2x slow motion
    ramp = Shot(index=0, start=0.0, end=1.0, effects=[
        Speed(start=0.0, end=1.0, kind="ramp_down", rate_from=1.0, rate_to=0.25, easing="ease_in_out")])
    steps = np.diff(source_positions(ramp, FPS))
    assert steps[0] > 0.95 and steps[-1] < 0.3 and np.all(np.diff(steps) <= 1e-9)


def test_speed_and_freeze_are_observable_on_moving_footage():
    shot = Shot(index=0, start=0.0, end=2.0, effects=[
        Speed(start=0.5, end=1.0, kind="slowmo", rate_from=0.4, rate_to=0.4),
        Freeze(start=1.3, end=1.6),
    ])
    edit = assign_ids(Edit(media=MediaInfo(duration=2.0, width=180, height=320, analysis_fps=FPS),
                           timeline=Timeline(sections=[Section(start=0, end=2, items=[shot])])))
    # No moving subject / grain: phase correlation then measures the camera alone.
    src = SyntheticSource(5, motion={0: (4.0, 0.0)}, subjects=False, grain=0)
    frames = np.stack(list(render_frames(edit, src, size=SIZE)))
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    win = cv2.createHanningWindow((SIZE[0], SIZE[1]), cv2.CV_32F)
    shift = np.array([cv2.phaseCorrelate(gray[i - 1].copy(), gray[i].copy(), win)[0][0]
                      for i in range(1, len(gray))])
    assert np.median(shift[2:14]) == pytest.approx(-4.0, abs=0.3)  # content moves left at 4 px/frame
    assert np.median(shift[16:29]) == pytest.approx(-1.6, abs=0.3)  # 0.4x slow motion
    assert all(np.array_equal(frames[39], frames[k]) for k in range(40, 48))  # freeze frames 39..47
    assert not np.array_equal(frames[47], frames[48])


class GradientSource:
    """Red encodes the source column and green the source row, so mirrored borders show up
    as a gradient reversal."""

    def prepare(self, edit, size, fps):
        pass

    def frames_for_shot(self, shot, n_source_frames, size):
        w, h = size
        img = np.empty((h, w, 3), np.uint8)
        img[..., 0] = np.linspace(0, 255, w)[None, :]
        img[..., 1] = np.linspace(0, 255, h)[:, None]
        img[..., 2] = 128
        return np.broadcast_to(img, (n_source_frames, h, w, 3)).copy()


def test_moves_never_expose_the_frame_border():
    shot = Shot(index=0, start=0.0, end=1.0, effects=[
        Pan(start=0.0, end=0.5, dx=0.2, dy=-0.1),
        Rotate(start=0.2, end=0.8, angle_to=12.0),
        Zoom(start=0.5, end=1.0, scale_from=1.0, scale_to=0.8),  # a zoom out
        Shake(start=0.6, end=1.0, amplitude=0.02, frequency=8.0),
    ])
    edit = assign_ids(Edit(media=MediaInfo(duration=1.0, width=180, height=320, analysis_fps=FPS),
                           timeline=Timeline(sections=[Section(start=0, end=1, items=[shot])])))
    frames = np.stack(list(render_frames(edit, GradientSource(), size=SIZE))).astype(int)
    assert (np.diff(frames[..., 0], axis=2) >= -2).all() and (np.diff(frames[..., 1], axis=1) >= -2).all()
    # The cover magnification is constant, so the observable moves are unchanged.
    geo = shot_geometry(shot, FPS, np.arange(30), SIZE)
    assert geo.scale[0] > 1.2 and np.allclose(geo.scale[:15], geo.scale[0])
    assert geo.scale[29] / geo.scale[15] == pytest.approx(0.8)
    assert (geo.tx[14] - geo.tx[0], geo.ty[14] - geo.ty[0]) == pytest.approx((0.2, -0.1))
    plain = Shot(index=0, start=0.0, end=1.0, effects=[Zoom(start=0.0, end=1.0, scale_to=1.3)])
    assert shot_geometry(plain, FPS, np.arange(30), SIZE).scale[0] == 1.0  # nothing to cover


def max_block_diff(frames: np.ndarray, block: int = 16) -> np.ndarray:
    """Largest mean |gray difference| of any block between consecutive frames."""
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]).astype(np.int16)
    d = np.abs(np.diff(g, axis=0))
    n, h, w = d.shape
    d = d[:, :h // block * block, :w // block * block].reshape(n, h // block, block, w // block, block)
    return d.mean(axis=(2, 4)).reshape(n, -1).max(axis=1)


def test_static_synthetic_shot_stays_alive_until_frozen(tmp_path):
    """The moving subject keeps a static shot visibly live after H.264, a freeze holds every
    pixel, and the subject does not read as camera motion."""
    shot = Shot(index=0, start=0.0, end=2.0, effects=[Freeze(start=1.0, end=1.4)])
    edit = assign_ids(Edit(media=MediaInfo(duration=2.0, width=180, height=320, analysis_fps=FPS),
                           timeline=Timeline(sections=[Section(start=0, end=2, items=[shot])])))
    frames = read_video(render_edit(edit, SyntheticSource(7), tmp_path / "s.mp4", size=SIZE))
    block = max_block_diff(frames)  # block[i] compares frames i and i + 1; frames 30..41 are held
    assert block[np.r_[0:29, 42:59]].min() > 6.0 and block[30:41].max() < 1.5
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    win = cv2.createHanningWindow(SIZE, cv2.CV_32F)
    # phaseCorrelate windows its inputs in place: pass copies.
    drift = sum(cv2.phaseCorrelate(gray[i - 1].copy(), gray[i].copy(), win)[0][0] for i in range(1, 30))
    # Even this naive global estimate (pulled by the subject; seed 7 is a low-detail texture, the
    # worst case) stays below the 5 % pan threshold; robust (RANSAC) estimates see far less.
    assert abs(drift) < 0.04 * SIZE[0]


# --------------------------------------------------------------------------- end to end


def test_render_edit_synthetic_roundtrip(tmp_path):
    edit = three_shot_edit()
    wav = write_wav(tmp_path / "music.wav", synth.click_track(120, 3.0, sr=22050), 22050)
    out = render_edit(edit, SyntheticSource(1, motion={2: (2.0, 0.0)}), tmp_path / "out.mp4", music=wav, size=SIZE)
    p = probe(out)
    assert (p.width, p.height) == SIZE and p.has_audio
    assert p.duration == pytest.approx(3.0, abs=0.1)
    frames = read_video(out)
    assert abs(frames.shape[0] - round(3.0 * FPS)) <= 1
    luma = frames.mean(axis=(1, 2, 3))
    assert int(np.argmax(luma)) == 30 and luma[30] > 235  # bright flash frame on the cut
    assert chroma(frames[45]) < 3.0 < 15.0 < chroma(frames[75])  # B&W shot vs colour shot
    c_mid, c_bw, c_col = chroma(frames[60]), chroma(frames[50]), chroma(frames[70])
    assert c_bw + 0.2 * (c_col - c_bw) < c_mid < c_bw + 0.8 * (c_col - c_bw)  # dissolve blends the shots
    # Through the analysis decoder as well (H.264 encode -> decode round trip).
    ctx = load_context(out, AnalysisConfig(analysis_long_side=160))
    assert abs(ctx.n_frames - 90) <= 1 and ctx.info.has_audio
    assert int(np.argmax(ctx.features.luma_mean)) == 30


def _write_clip(path, frames, fps=FPS):
    with VideoWriter(path, frames.shape[2], frames.shape[1], fps) as vw:
        for f in frames:
            vw.write(f)
    return path


def test_video_clip_source_groups_inpoints_and_crop(tmp_path):
    landscape = synth.scene(60, 90, 160, seed=11, velocity=(1.5, 0.0))  # 2 s, 16:9
    portrait = synth.scene(60, 160, 90, seed=12, velocity=(0.0, 1.0))  # 2 s, 9:16
    a = _write_clip(tmp_path / "a.mp4", landscape)
    b = _write_clip(tmp_path / "b.mp4", portrait)
    shots = [Shot(index=i, start=0.4 * i, end=0.4 * (i + 1), source_group=g)
             for i, g in enumerate(["x", "y", "x", None])]
    edit = assign_ids(Edit(media=MediaInfo(duration=1.6, width=90, height=160, analysis_fps=FPS),
                           timeline=Timeline(sections=[Section(start=0, end=1.6, items=shots)])))
    src = VideoClipSource([a, b])
    out = render_edit(edit, [a, b], tmp_path / "out.mp4", size=(90, 160))
    frames = read_video(out)
    assert frames.shape[1:] == (160, 90, 3) and abs(frames.shape[0] - 48) <= 1
    src.prepare(edit, (90, 160), FPS)
    assert [src.clip_for_shot(s) for s in edit.shots()] == [0, 1, 0, 0]
    # Shot 3 continues clip "a" where shot 1 stopped (12 source frames in), cropped to fill.
    ref = read_video(a, fps=FPS, size=(284, 160))[:, :, 97:187]  # 16:9 scaled to cover 90x160
    first_of_shot3 = frames[24].astype(float)
    err = [np.abs(first_of_shot3 - r.astype(float)).mean() for r in ref[:30]]
    assert int(np.argmin(err)) == 12 and min(err) < 8
    assert frames[:, :4].mean() > 30 and frames[:, -4:].mean() > 30  # no pillar/letterbox bars
    hist = lambda f: cv2.calcHist([f], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3).ravel() / f[..., 0].size
    near_a = np.minimum(hist(frames[2]), hist(frames[26])).sum()  # both from clip a
    near_b = np.minimum(hist(frames[2]), hist(frames[14])).sum()  # clip a vs clip b
    assert near_a > near_b


class WholeShotSource:
    """Wraps a VideoClipSource but only exposes the basic protocol (whole shots as arrays)."""

    def __init__(self, inner):
        self.inner = inner

    def prepare(self, edit, size, fps):
        self.inner.prepare(edit, size, fps)

    def frames_for_shot(self, shot, n_source_frames, size):
        return self.inner.frames_for_shot(shot, n_source_frames, size)


def test_streamed_footage_matches_whole_shot_delivery(tmp_path):
    """Slow motion (frame blending), freeze and dissolve handles read through the sliding
    decoder window give exactly the frames of the decode-everything path."""
    clip = _write_clip(tmp_path / "c.mp4", synth.scene(45, 160, 90, seed=3, velocity=(1.0, 0.5)))
    shots = [Shot(index=0, start=0.0, end=0.8, effects=[
                 Speed(start=0.2, end=0.8, kind="ramp_down", rate_from=1.0, rate_to=0.3, easing="ease_out")]),
             Shot(index=1, start=0.8, end=1.6, effects=[Freeze(start=1.0, end=1.2)])]
    items = [shots[0], Transition(kind="dissolve", start=0.7, end=0.9, from_shot="s1", to_shot="s2"), shots[1]]
    edit = assign_ids(Edit(media=MediaInfo(duration=1.6, width=90, height=160, analysis_fps=FPS),
                           timeline=Timeline(sections=[Section(start=0, end=1.6, items=items)])))
    streamed = np.stack(list(render_frames(edit, VideoClipSource([clip]), size=(90, 160))))
    whole = np.stack(list(render_frames(edit, WholeShotSource(VideoClipSource([clip])), size=(90, 160))))
    assert streamed.shape == (48, 160, 90, 3) and np.array_equal(streamed, whole)
    assert np.array_equal(streamed[31], streamed[35]) and not np.array_equal(streamed[35], streamed[36])


def test_output_format_defaults():
    edit = three_shot_edit()
    edit.media = MediaInfo(duration=3.0, width=1080, height=1920, analysis_fps=25.0)
    assert output_format(edit) == ((720, 1280), 25.0, 75)
    edit.media = MediaInfo(duration=3.0, width=1920, height=1080, analysis_fps=30.0)
    assert output_format(edit)[0] == (1280, 720)
    edit.media = MediaInfo(duration=0.0, width=0, height=0)
    assert output_format(edit, fps=30.0) == ((720, 1280), 30.0, 90)  # duration from the shots


# --------------------------------------------------------------------------- retime


def test_retime_edit_doubles_times_from_120_to_60_bpm():
    edit = three_shot_edit()
    shot = edit.shots()[0]
    shot.beats = 2.0
    shot.effects.append(Pan(start=0.2, end=0.6, dx=0.1, keyframes=[Keyframe(t=0.4, v=0.05)]))
    edit.overlays[0].words = [Word(text="WAIT", start=0.0, end=0.5)]
    edit.music.downbeats = [0.0, 2.0]
    edit.music.drops = [Drop(time=2.0)]
    edit.music.sections = [AudioSection(start=0.0, end=2.0, label="build"), AudioSection(start=2.0, end=3.0)]
    edit.music.energy = Curve(t0=0.05, dt=0.1, values=[0.1] * 30)
    edit.patterns = [Pattern(rule="cut_every_n_beats", start=0.0, end=3.0, params={"n": 2})]
    edit.stats = EditStats(n_shots=3, mean_shot_duration=1.0, cuts_per_minute=40.0, first_cut_time=1.0,
                           pacing=Curve(t0=0.5, dt=1.0, values=[0.0, 1.0, 1.0]))
    out = retime_edit(edit, 60.0)
    assert out is not edit and edit.music.bpm == 120.0  # the input is untouched
    assert out.music.bpm == 60.0 and out.media.duration == pytest.approx(6.0)
    assert out.media.n_frames == 180
    for a, b in zip(edit.shots(), out.shots()):
        assert (b.start, b.end) == pytest.approx((2 * a.start, 2 * a.end))
        assert (b.frame_start, b.frame_end) == (2 * a.frame_start, 2 * a.frame_end)
        assert b.beats == a.beats
        for ea, eb in zip(a.effects, b.effects):
            assert (eb.start, eb.end) == pytest.approx((2 * ea.start, 2 * ea.end))
    assert out.shots()[0].effects[1].keyframes[0].t == pytest.approx(0.8)
    assert [(t.start, t.end) for t in out.transitions()] == pytest.approx([(1.8, 2.2), (3.6, 4.4)])
    assert (out.overlays[0].start, out.overlays[0].end) == pytest.approx((0.0, 2.0))
    assert out.overlays[0].words[0].end == pytest.approx(1.0)
    assert out.music.beats == pytest.approx([2 * b for b in edit.music.beats])
    assert out.music.downbeats == pytest.approx([0.0, 4.0]) and out.music.drops[0].time == pytest.approx(4.0)
    assert out.music.sections[1].start == pytest.approx(4.0)
    assert out.music.energy.dt == pytest.approx(0.2) and out.music.energy.t0 == pytest.approx(0.1)
    assert (out.patterns[0].start, out.patterns[0].end) == pytest.approx((0.0, 6.0))
    assert out.stats.mean_shot_duration == pytest.approx(2.0) and out.stats.cuts_per_minute == pytest.approx(20.0)
    assert out.stats.first_cut_time == pytest.approx(2.0) and out.stats.pacing.dt == pytest.approx(2.0)
    # Re-rendering the retimed edit takes twice as many frames.
    assert output_format(out, size=SIZE)[2] == 180


def test_retime_edit_offset_and_no_music():
    edit = three_shot_edit()
    edit.music.beat_offset = 0.5
    out = retime_edit(edit, 120.0, offset=0.25)  # same tempo, first beat moves 0.25 s earlier
    assert out.shots()[1].start == pytest.approx(0.75)
    assert out.shots()[0].start == 0.0  # the timeline still starts at 0
    assert out.music.beat_offset == pytest.approx(0.25)
    plain = three_shot_edit()
    plain.music = None
    again = retime_edit(plain, 90.0)
    assert again == plain and again is not plain
    no_bpm = three_shot_edit()
    no_bpm.music.bpm = None
    assert retime_edit(no_bpm, 90.0) == no_bpm


def test_retime_drops_shots_squeezed_before_zero():
    edit = three_shot_edit()
    edit.music.beat_offset = 1.5  # shot 1 lies entirely before the first beat
    out = retime_edit(edit, 120.0, offset=0.0)
    assert [s.id for s in out.shots()] == ["s1", "s2"]
    assert out.shots()[0].start == 0.0 and out.shots()[0].end == pytest.approx(0.5)
    tr = out.transitions()
    assert len(tr) == 1 and (tr[0].from_shot, tr[0].to_shot) == ("s1", "s2")
