"""Tests for the look analyzer (grade, layout, flash / blur / invert / RGB split).

Effects are rendered with the same primitives the renderer uses
(:mod:`editast.fx`), so the tests also pin down parameter conventions: Flash
``intensity`` is the blend amount at the peak, Blur ``amount`` the Gaussian
sigma as a fraction of width, RGBSplit ``offset`` the per-channel shift as a
fraction of width.
"""

import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from editast import fx
from editast.analyzers.look import analyze_look
from editast.analyzers.types import LookAnalysis, ShotSegmentation, ShotSpan, TransitionSpan
from editast.context import AnalysisConfig, context_from_arrays, load_context
from editast.media import VideoWriter
from editast.testing import synth

H, W, FPS = 320, 180, 30.0  # portrait analysis frames


def scene(n: int = 90, seed: int = 7, velocity=(1.5, 0.5), h: int = H, w: int = W) -> np.ndarray:
    return synth.scene(n, h, w, seed=seed, velocity=velocity)


def segmentation(lengths, transitions=()) -> ShotSegmentation:
    b = np.cumsum([0, *lengths])
    shots = [ShotSpan(i, int(b[i]), int(b[i + 1])) for i in range(len(lengths))]
    return ShotSegmentation(shots=shots, transitions=list(transitions), boundary_score=np.zeros(int(b[-1])))


def analyze(frames, seg=None, noise: float = 0.0) -> LookAnalysis:
    if noise:
        rng = np.random.default_rng(0)
        frames = np.clip(frames + rng.normal(0.0, noise, frames.shape), 0, 255).astype(np.uint8)
    return analyze_look(context_from_arrays(frames, FPS), seg or segmentation([len(frames)]))


def effects(res: LookAnalysis, kind: str, shot: int = 0) -> list:
    return [e for e in res.effects.by_shot.get(shot, []) if e.type == kind]


def all_effects(res: LookAnalysis) -> list:
    return [e for effs in res.effects.by_shot.values() for e in effs]


def assert_span(effect, f0: int, f1: int, tol: float = 1.0) -> None:
    a, b = effect.start * FPS, effect.end * FPS
    assert abs(a - f0) <= tol and abs(b - f1) <= tol, (a, b, f0, f1)


def apply(frames: np.ndarray, frame_range, fn) -> np.ndarray:
    out = frames.copy()
    for k, i in enumerate(frame_range):
        out[i] = fn(out[i], k)
    return out


# --------------------------------------------------------------------------- flash


@pytest.mark.parametrize("color", [(255, 255, 255), (0, 0, 0), (255, 0, 0)])
def test_flash_colour_span_and_intensity(color):
    env = (0.4, 0.9, 0.7, 0.3)
    frames = apply(scene(), range(30, 34), lambda f, k: fx.flash(f, env[k], color))
    res = analyze(frames)
    (fl,) = effects(res, "flash")
    assert_span(fl, 30, 34)
    assert fl.intensity == pytest.approx(0.9, abs=0.12)
    assert np.abs(np.array(fl.color) - color).max() <= 40
    assert fl.keyframes and max(k.v for k in fl.keyframes) == pytest.approx(fl.intensity)
    assert not effects(res, "blur")
    assert res.effects.signals["flash_amount"][31] == pytest.approx(0.9, abs=0.12)


def test_flash_inside_transition_is_not_repeated():
    frames = apply(scene(), range(30, 34), lambda f, k: fx.flash(f, (0.5, 1.0, 0.6, 0.2)[k]))
    seg = segmentation([90], [TransitionSpan("flash", 32, 29, 35, 0, 0)])
    assert not all_effects(analyze(frames, seg))


def test_lighting_change_is_not_a_flash():
    base = scene()
    brighten = np.stack([fx.grade(f, brightness=0.25 * i / 89) for i, f in enumerate(base)])
    assert not all_effects(analyze(brighten))


# --------------------------------------------------------------------------- blur / invert / rgb split


def test_blur_span_and_sigma():
    frames = apply(scene(), range(40, 46), lambda f, k: fx.blur(f, 0.012))
    res = analyze(frames)
    (bl,) = effects(res, "blur")
    assert_span(bl, 40, 46)
    assert bl.amount == pytest.approx(0.012, rel=0.3)
    assert res.effects.signals["sharpness_rel"][42] < 0.38
    assert not effects(res, "flash")


def test_invert_span():
    frames = apply(scene(), range(50, 56), lambda f, k: fx.invert(f))
    res = analyze(frames)
    (inv,) = effects(res, "invert")
    assert_span(inv, 50, 56)
    assert inv.confidence >= 0.6
    assert len(all_effects(res)) == 1


def test_rgb_split_span_and_offset():
    frames = apply(scene(), range(60, 68), lambda f, k: fx.rgb_split(f, 0.015))
    res = analyze(frames)
    (rs,) = effects(res, "rgb_split")
    assert_span(rs, 60, 68)
    truth = round(0.015 * W) / W  # fx.rgb_split shifts whole pixels
    assert rs.offset == pytest.approx(truth, rel=0.2)
    assert res.effects.signals["rgb_shift"][64] == pytest.approx(truth, rel=0.2)
    assert abs(res.effects.signals["rgb_shift"][10]) < 0.002
    # The opposite direction gives a negative offset.
    neg = analyze(apply(scene(), range(60, 68), lambda f, k: fx.rgb_split(f, -0.015)))
    assert effects(neg, "rgb_split")[0].offset == pytest.approx(-truth, rel=0.2)


def test_no_false_positives_on_clean_shots():
    n = 90
    clips = [
        scene(n),  # pan
        scene(n, seed=3, velocity=(0.0, 0.0)),  # static
        np.stack([fx.transform(f, 1.0 + 0.004 * i) for i, f in enumerate(scene(n, seed=4))]),  # slow zoom
        np.stack([fx.transform(f, tx=dx, ty=dy) for f, (dx, dy) in  # handheld shake
                  zip(scene(n, seed=5), (fx.shake_offset(i / FPS, 0.01, 6.0) for i in range(n)))]),
    ]
    for clip in clips:
        assert not all_effects(analyze(clip, noise=2.0))


# --------------------------------------------------------------------------- look


def test_look_attributes():
    base = scene(60)
    plain = analyze(base).looks[0]
    bw = analyze(np.stack([fx.to_bw(f) for f in base])).looks[0]
    warm = analyze(np.stack([fx.grade(f, temperature=0.6) for f in base])).looks[0]
    bright = analyze(np.stack([fx.grade(f, brightness=0.2) for f in base])).looks[0]
    flat = analyze(np.stack([fx.grade(f, contrast=0.5) for f in base])).looks[0]
    soft = analyze(np.stack([fx.blur(f, 0.008) for f in base])).looks[0]
    vig = analyze(np.stack([fx.vignette(f, 0.7) for f in base])).looks[0]

    assert not plain.bw and plain.saturation > 0.4 and plain.vignette == 0.0
    assert bw.bw and bw.saturation < 0.02 and abs(bw.temperature) < 0.02
    assert warm.temperature > plain.temperature + 0.2
    assert bright.brightness == pytest.approx(plain.brightness + 0.2, abs=0.05)
    assert flat.contrast == pytest.approx(0.5 * plain.contrast, rel=0.2)
    assert 0.0 <= soft.sharpness < 0.5 * plain.sharpness <= 1.0
    assert vig.vignette == pytest.approx(0.7, abs=0.2)
    for look in (plain, bw, warm, bright, flat, soft, vig):
        assert 0.0 <= look.brightness <= 1.0 and 0.0 <= look.sharpness <= 1.0
    # Dominant colour of a mostly red picture.
    red = np.zeros((20, H, W, 3), np.uint8)
    red[:] = (200, 30, 40)
    red[:, :60] = scene(20, seed=2)[:, :60]
    assert np.abs(np.array(analyze(red).looks[0].dominant_color) - (200, 30, 40)).max() <= 20


# --------------------------------------------------------------------------- layout


def test_letterbox_and_pillarbox():
    lb = np.stack([fx.letterbox(f, 0.12, 0.12) for f in scene(40)])
    lay = analyze(lb).layouts[0]
    assert lay.letterbox_top == pytest.approx(0.12, abs=0.01)
    assert lay.letterbox_bottom == pytest.approx(0.12, abs=0.01)
    assert lay.pillarbox_left == lay.pillarbox_right == 0.0
    assert lay.content_box == pytest.approx((0.0, 0.12, 1.0, 0.76), abs=0.015)
    assert not lay.blur_pad and lay.split == "none"

    land = np.zeros((40, 180, 320, 3), np.uint8)
    land[:, :, 70:250] = scene(40, seed=3, h=180, w=180)
    lay = analyze(land).layouts[0]
    assert lay.pillarbox_left == pytest.approx(70 / 320, abs=0.01)
    assert lay.pillarbox_right == pytest.approx(70 / 320, abs=0.01)
    assert lay.letterbox_top == lay.letterbox_bottom == 0.0

    clean = analyze(scene(40)).layouts[0]
    assert clean.content_box == (0.0, 0.0, 1.0, 1.0) and not clean.blur_pad


def test_bar_with_caption_and_blur_pad():
    n = 40
    band = scene(n, seed=4, velocity=(1.0, 0.2), h=102, w=W)
    # Black bars with a caption on the bottom bar.
    boxed = np.zeros((n, H, W, 3), np.uint8)
    boxed[:, 109:211] = band
    boxed = np.stack([fx.draw_text(f, "NO WAY", (0.5, 0.83), 0.04)[0] for f in boxed])
    lay = analyze(boxed).layouts[0]
    assert lay.letterbox_top == pytest.approx(109 / H, abs=0.01)
    assert lay.letterbox_bottom == pytest.approx(109 / H, abs=0.01)
    # Landscape footage over a blurred, cover-scaled copy of itself (TikTok "blur pad").
    pad = np.empty((n, H, W, 3), np.uint8)
    for i in range(n):
        big = cv2.resize(band[i], (int(round(W * H / 102)), H), interpolation=cv2.INTER_LINEAR)
        x0 = (big.shape[1] - W) // 2
        pad[i] = (cv2.GaussianBlur(big[:, x0:x0 + W], (0, 0), 12) * 0.7).astype(np.uint8)
        pad[i, 109:211] = band[i]
    lay = analyze(pad).layouts[0]
    assert lay.blur_pad
    assert lay.letterbox_top == lay.letterbox_bottom == 0.0
    assert lay.content_box == pytest.approx((0.0, 109 / H, 1.0, 102 / H), abs=0.02)


def test_split_screen():
    top = scene(40, seed=5, velocity=(1.5, 0.0), h=160)
    bottom = scene(40, seed=6, velocity=(-1.5, 0.0), h=160)
    assert analyze(np.concatenate([top, bottom], axis=1)).layouts[0].split == "horizontal"


# --------------------------------------------------------------------------- integration


def test_per_shot_results_and_signals():
    shots = [scene(40, seed=1), np.stack([fx.to_bw(f) for f in scene(40, seed=2)]), scene(40, seed=3)]
    shots[2] = apply(shots[2], range(15, 18), lambda f, k: fx.flash(f, (0.6, 1.0, 0.5)[k]))
    res = analyze(np.concatenate(shots), segmentation([40, 40, 40]))
    assert set(res.looks) == set(res.layouts) == {0, 1, 2}
    assert [res.looks[i].bw for i in range(3)] == [False, True, False]
    (fl,) = effects(res, "flash", 2)
    assert 80 / FPS <= fl.start < fl.end <= 120 / FPS
    assert not res.effects.by_shot.get(0) and not res.effects.by_shot.get(1)
    for key in ("luma", "saturation", "sharpness_rel", "rgb_shift", "flash_amount"):
        assert res.effects.signals[key].shape == (120,)
    empty = analyze_look(context_from_arrays(scene(1), FPS), segmentation([1]))
    assert set(empty.looks) == {0} and not all_effects(empty)


def test_h264_roundtrip(tmp_path):
    w, h = 360, 640
    base = synth.scene(150, h, w, seed=8, velocity=(3.0, 1.0))
    frames = apply(base, range(20, 24), lambda f, k: fx.flash(f, (0.5, 1.0, 0.7, 0.3)[k]))
    frames = apply(frames, range(45, 51), lambda f, k: fx.blur(f, 0.01))
    frames = apply(frames, range(75, 81), lambda f, k: fx.invert(f))
    frames = apply(frames, range(105, 113), lambda f, k: fx.rgb_split(f, 0.015))
    graded = np.stack([fx.letterbox(fx.to_bw(f), 0.1, 0.1) for f in synth.scene(60, h, w, seed=9, velocity=(3, 0))])
    out = tmp_path / "look.mp4"
    with VideoWriter(out, w, h, FPS, crf=23) as vw:
        for frame in np.concatenate([frames, graded]):
            vw.write(frame)
    ctx = load_context(out, AnalysisConfig())
    res = analyze_look(ctx, segmentation([150, 60]))

    (fl,) = effects(res, "flash")
    assert_span(fl, 20, 24)
    assert fl.color == (255, 255, 255) and fl.intensity > 0.8
    (bl,) = effects(res, "blur")
    assert_span(bl, 45, 51)
    assert bl.amount == pytest.approx(0.01, rel=0.35)
    (inv,) = effects(res, "invert")
    assert_span(inv, 75, 81)
    (rs,) = effects(res, "rgb_split")
    assert_span(rs, 105, 113)
    assert rs.offset == pytest.approx(round(0.015 * w) / w, rel=0.35)
    assert len(all_effects(res)) == 4
    look, lay = res.looks[1], res.layouts[1]
    assert look.bw and not res.looks[0].bw
    assert lay.letterbox_top == pytest.approx(0.1, abs=0.015) and lay.letterbox_bottom == pytest.approx(0.1, abs=0.015)


def test_performance_one_minute_clip():
    """60 s at 30 fps, 180x320: one 10 s shot repeated (features tiled to save test time)."""
    base = context_from_arrays(scene(300), FPS)
    reps = 6
    f = base.features
    tiled = type(f)(**{k: np.concatenate([getattr(f, k)] * reps) for k in vars(f)})
    ctx = replace(base, frames=np.concatenate([base.frames] * reps), gray=np.concatenate([base.gray] * reps),
                  features=tiled)
    t0 = time.perf_counter()
    res = analyze_look(ctx, segmentation([300] * reps))
    assert time.perf_counter() - t0 < 15.0
    assert len(res.looks) == reps
