"""Tests for the shot-boundary / transition detector.

Ground truth is built from :mod:`editast.testing.synth` scenes and the
:mod:`editast.fx` primitives the renderer uses, so a "flash" or "fade" here is
the same pixel operation the compiler applies. Frames are portrait analysis
frames (width 180, height 320) at 30 fps unless stated otherwise.
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from editast import fx
from editast.analyzers.shots import ShotParams, detect_shots
from editast.analyzers.types import ShotSegmentation
from editast.context import AnalysisConfig, context_from_arrays, load_context
from editast.media import VideoWriter
from editast.testing import synth

H, W, FPS = 320, 180, 30.0
WHITE, BLACK = (255, 255, 255), (0, 0, 0)


# --------------------------------------------------------------------------- helpers


def scene(n: int, seed: int, velocity: tuple[float, float] = (1.0, 0.0), zoom_rate: float = 1.0) -> np.ndarray:
    return synth.scene(n, H, W, seed=seed, velocity=velocity, zoom_rate=zoom_rate)


def detect(frames: np.ndarray, config: AnalysisConfig | None = None) -> ShotSegmentation:
    return detect_shots(context_from_arrays(frames, FPS, config=config))


def internal(seg: ShotSegmentation) -> list:
    return [t for t in seg.transitions if t.from_shot is not None and t.to_shot is not None]


def boundaries(seg: ShotSegmentation) -> list[int]:
    return [s.f0 for s in seg.shots[1:]]


def check_contract(seg: ShotSegmentation, n: int) -> None:
    """Invariants of ``ShotSegmentation`` (see ``editast.analyzers.types``)."""
    assert seg.boundary_score.shape == (n,)
    assert np.isfinite(seg.boundary_score).all()
    assert seg.shots[0].f0 == 0 and seg.shots[-1].f1 == n
    for i, s in enumerate(seg.shots):
        assert s.index == i and s.f1 > s.f0
        if i:
            assert s.f0 == seg.shots[i - 1].f1
    inner = internal(seg)
    assert len(inner) == len(seg.shots) - 1
    for t in inner:
        assert t.to_shot == t.from_shot + 1
        assert t.boundary == seg.shots[t.to_shot].f0
        assert seg.shots[t.from_shot].f0 <= t.f0 <= t.boundary <= t.f1 <= seg.shots[t.to_shot].f1
        assert 0.0 < t.confidence <= 1.0
        if t.kind == "cut":
            assert t.f0 == t.f1 == t.boundary
    for t in seg.transitions:
        if t.from_shot is None:
            assert t.to_shot == 0 and t.boundary == 0 and t.f0 == 0 < t.f1 <= seg.shots[0].f1
        if t.to_shot is None:
            assert t.from_shot == len(seg.shots) - 1 and t.boundary == t.f1 == n
            assert seg.shots[-1].f0 <= t.f0 < n
    assert sum(t.from_shot is None for t in seg.transitions) <= 1
    assert sum(t.to_shot is None for t in seg.transitions) <= 1


def blend_toward(frames: np.ndarray, amounts: list[float], color: tuple[int, int, int]) -> np.ndarray:
    return np.stack([fx.flash(f, a, color) for f, a in zip(frames, amounts)])


def add_noise(frames: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(0.0, sigma, frames.shape)
    return np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def match(truth: list[int], pred: list[int], tol: int = 1) -> int:
    """Number of one-to-one matches within ``tol`` frames."""
    used: set[int] = set()
    hits = 0
    for b in truth:
        cands = [p for p in pred if abs(p - b) <= tol and p not in used]
        if cands:
            used.add(min(cands, key=lambda p: abs(p - b)))
            hits += 1
    return hits


# --------------------------------------------------------------------------- hard cuts


def test_hard_cuts_exact_frames():
    frames, truth = synth.cut_sequence([20, 15, 25, 18, 22], height=H, width=W)
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert boundaries(seg) == truth
    assert all(t.kind == "cut" for t in seg.transitions)
    peaks = np.sort(np.argsort(seg.boundary_score)[-len(truth):])
    assert peaks.tolist() == truth


def test_landscape_cut_sequence_defaults():
    frames, truth = synth.cut_sequence([12, 12, 12, 30])
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert boundaries(seg) == truth


def test_three_frame_shots():
    lengths = [10, 3, 4, 3, 5, 3, 3, 12]
    frames, truth = synth.cut_sequence(lengths, height=H, width=W, velocity=(3.0, 1.0))
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert boundaries(seg) == truth
    assert [s.n_frames for s in seg.shots] == lengths


def test_cut_between_similar_palettes_with_motion():
    # Same texture palette (same seed) but a different region: a cut within one location.
    a = scene(20, seed=7, velocity=(2.0, 0.0))
    b = synth.scene(20, H, W, seed=7, velocity=(-2.0, 0.0), margin=2.6)
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert boundaries(seg) == [20]


def test_min_shot_frames_param():
    frames, truth = synth.cut_sequence([10, 2, 10], height=H, width=W)
    assert boundaries(detect(frames)) == truth
    seg = detect(frames, AnalysisConfig(params={"shots": {"min_shot_frames": 4}}))
    check_contract(seg, len(frames))
    assert len(seg.shots) == 2 and all(s.n_frames >= 4 for s in seg.shots)


# --------------------------------------------------------------------------- flashes


def test_flash_transition_between_shots():
    a, b = scene(16, seed=1), scene(16, seed=2, velocity=(-1.0, 1.0))
    a[-2:] = blend_toward(a[-2:], [0.5, 1.0], WHITE)
    b[:1] = blend_toward(b[:1], [0.5], WHITE)
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 2
    (tr,) = seg.transitions
    assert tr.kind == "flash"
    assert abs(tr.boundary - 16) <= 1  # first frame of new content is 16
    assert tr.f0 <= 14 and tr.f1 >= 16
    assert tr.color is not None and min(tr.color) >= 230


def test_rendered_style_flash_peaking_on_boundary():
    # Compiler envelope: rises over two frames, full colour on the boundary frame, falls over two.
    a, b = scene(18, seed=11), scene(18, seed=12)
    a[-2:] = blend_toward(a[-2:], [1 / 3, 2 / 3], WHITE)
    b[:3] = blend_toward(b[:3], [1.0, 2 / 3, 1 / 3], WHITE)
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "flash" and abs(tr.boundary - 18) <= 1


def test_partial_flash_without_solid_frame():
    a, b = scene(15, seed=21), scene(15, seed=22)
    a[-1:] = blend_toward(a[-1:], [0.6], WHITE)
    b[:2] = blend_toward(b[:2], [0.7, 0.35], WHITE)
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.boundary == 15 and tr.kind == "flash"


def test_black_flash_transition():
    a, b = scene(15, seed=31), scene(15, seed=32)
    a[-1:] = blend_toward(a[-1:], [1.0], BLACK)
    b[:1] = blend_toward(b[:1], [1.0], BLACK)
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert abs(tr.boundary - 15) <= 1
    assert tr.kind in ("flash", "fade_black")
    assert tr.color is not None and max(tr.color) <= 25


def test_in_shot_flash_is_not_a_boundary():
    frames = scene(36, seed=3, velocity=(2.0, 0.0))
    frames[14:17] = blend_toward(frames[14:17], [0.5, 1.0, 0.5], WHITE)
    frames[28:30] = blend_toward(frames[28:30], [0.7, 0.3], WHITE)  # partial flash, no solid frame
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 1 and seg.transitions == []


# --------------------------------------------------------------------------- dissolves & fades


def test_dissolve_over_ten_frames():
    a, b = scene(30, seed=4, velocity=(1.0, 0.0)), scene(30, seed=5, velocity=(-1.0, 0.0))
    mixed = np.stack([fx.blend(a[20 + j], b[j], (j + 1) / 11) for j in range(10)])
    frames = np.concatenate([a[:20], mixed, b[10:]])  # blend frames are [20, 30)
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "dissolve"
    assert abs(tr.boundary - 25) <= 1
    assert abs(tr.f0 - 20) <= 1 and abs(tr.f1 - 30) <= 1


def test_short_dissolve_between_static_shots():
    a, b = scene(25, seed=41, velocity=(0.0, 0.0)), scene(25, seed=42, velocity=(0.0, 0.0))
    mixed = np.stack([fx.blend(a[20 + j], b[j], (j + 1) / 6) for j in range(5)])
    frames = np.concatenate([a[:20], mixed, b[5:]])  # blend frames are [20, 25)
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "dissolve" and abs(tr.boundary - 22) <= 1


def test_fade_to_black_dip_between_shots():
    a, b = scene(24, seed=6), scene(24, seed=8, velocity=(0.0, 1.5))
    a[-4:] = blend_toward(a[-4:], [0.2, 0.4, 0.6, 0.8], BLACK)
    b[:4] = blend_toward(b[:4], [1.0, 0.75, 0.5, 0.25], BLACK)
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "fade_black"
    assert abs(tr.boundary - 24) <= 1
    assert tr.f0 <= 21 and tr.f1 >= 26
    assert tr.color is not None and max(tr.color) <= 25


def test_fade_to_white_dip_between_shots():
    a, b = scene(24, seed=16), scene(24, seed=18)
    ramp = list(np.linspace(0.15, 1.0, 6))
    a[-6:] = blend_toward(a[-6:], ramp, WHITE)
    b[:6] = blend_toward(b[:6], ramp[::-1], WHITE)
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "fade_white" and abs(tr.boundary - 24) <= 1


def test_fade_in_and_fade_out_at_the_ends():
    frames = scene(60, seed=9, velocity=(1.5, 0.0))
    frames[:8] = blend_toward(frames[:8], [1.0 - k / 8 for k in range(8)], BLACK)
    frames[-6:] = blend_toward(frames[-6:], [(k + 1) / 6 for k in range(6)], BLACK)
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 1
    fin = [t for t in seg.transitions if t.from_shot is None]
    fout = [t for t in seg.transitions if t.to_shot is None]
    assert len(fin) == 1 and len(fout) == 1 and len(seg.transitions) == 2
    assert fin[0].kind == "fade_black" and fin[0].boundary == 0 and abs(fin[0].f1 - 8) <= 1
    assert fout[0].kind == "fade_black" and fout[0].boundary == 60 and abs(fout[0].f0 - 54) <= 1


def test_fade_in_then_cuts():
    frames, truth = synth.cut_sequence([20, 10, 15], height=H, width=W)
    frames[:6] = blend_toward(frames[:6], [1.0 - k / 6 for k in range(6)], BLACK)
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert boundaries(seg) == truth
    assert [t.kind for t in seg.transitions] == ["fade_black", "cut", "cut"]


def test_text_card_between_fades_is_its_own_shot():
    """Fade to black, a title card on black, fade back in: two transitions, not one long dip."""
    a, b = scene(30, seed=71), scene(30, seed=72, velocity=(-1.0, 0.5))
    a[-8:] = blend_toward(a[-8:], [(k + 1) / 8 for k in range(8)], BLACK)
    b[:8] = blend_toward(b[:8], [1 - k / 8 for k in range(8)], BLACK)
    card = fx.draw_text(np.zeros((H, W, 3), np.uint8), "PRESENTS", center=(0.5, 0.5), size=0.05)[0]
    black = np.zeros((6, H, W, 3), np.uint8)
    frames = np.concatenate([a, black[:2], np.stack([card] * 24), black, b])  # card frames [32, 56)
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 3, [(t.kind, t.boundary) for t in seg.transitions]
    assert seg.shots[1].f0 <= 32 and seg.shots[1].f1 >= 56
    assert [t.kind for t in seg.transitions] == ["fade_black", "fade_black"]
    first, second = seg.transitions
    assert 22 <= first.boundary <= 32 and 56 <= second.boundary <= 62


def test_caption_over_a_black_dip_is_not_a_card():
    """A caption drawn over the shots and the black hold between them is an overlay."""
    a, b = scene(30, seed=73), scene(30, seed=74, velocity=(0.0, 1.0))
    a[-6:] = blend_toward(a[-6:], [(k + 1) / 6 for k in range(6)], BLACK)
    b[:6] = blend_toward(b[:6], [1 - k / 6 for k in range(6)], BLACK)
    frames = np.concatenate([a, np.zeros((16, H, W, 3), np.uint8), b])
    frames = np.stack([fx.draw_text(f, "WAIT FOR IT", center=(0.5, 0.75), size=0.06)[0] if i >= 12 else f
                       for i, f in enumerate(frames)])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "fade_black" and 30 <= tr.boundary <= 46


def test_foggy_letterboxed_fade_through_white():
    """Low-contrast letterboxed shots dipping through (almost) white."""
    def fog(f: np.ndarray) -> np.ndarray:
        return np.clip(170 + (f.astype(np.float32) - 128) * 0.45, 0, 255).astype(np.uint8)
    a = np.stack([fog(f) for f in scene(30, seed=75, velocity=(0.5, 0.0))])
    b = np.stack([fog(f) for f in scene(30, seed=76, velocity=(0.0, 0.5))])
    amounts = [0.3, 0.55, 0.75, 0.9, 0.95, 0.95]
    a[-6:] = blend_toward(a[-6:], amounts, WHITE)
    b[:6] = blend_toward(b[:6], amounts[::-1], WHITE)
    frames = np.stack([fx.letterbox(f, 0.12, 0.12) for f in np.concatenate([a, b])])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "fade_white" and abs(tr.boundary - 30) <= 2


def test_relative_dip_frames_are_flat():
    """A held near-white frame pair whose contrast is a small fraction of both shots is flat,
    even though the two identical frames trivially share their structure."""
    a, b = scene(30, seed=33), scene(30, seed=78, velocity=(-1.0, 0.0))
    peak = fx.flash(a[-1], 0.92, WHITE)
    frames = np.concatenate([a[:-2], np.stack([peak, peak]), b])
    from editast.analyzers.shots import _Detector
    det = _Detector(context_from_arrays(frames, FPS), ShotParams())
    assert det.contrast[28] > ShotParams().flat_abs  # not flat by the absolute test alone
    assert det.flat[28] and det.flat[29]
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 2 and abs(seg.shots[1].f0 - 29) <= 1


def test_long_dissolve_between_moving_shots():
    """A two-second (60 frame) cross-dissolve between two panning shots."""
    a, b = scene(80, seed=81, velocity=(1.5, 0.0)), scene(80, seed=82, velocity=(-1.0, 0.5))
    mixed = np.stack([fx.blend(a[20 + j], b[j], (j + 1) / 61) for j in range(60)])
    frames = np.concatenate([a[:20], mixed, b[60:]])  # blend frames are [20, 80)
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "dissolve"
    assert 20 <= tr.boundary < 80 and abs(tr.f1 - 80) <= 2


def test_dissolve_into_bw_with_zoom_and_leaving_caption():
    """Colour shot zooming in, dissolving into a B&W shot; the caption leaves mid-dissolve."""
    a = scene(45, seed=91, velocity=(2.0, 0.0), zoom_rate=1.012)
    b = np.stack([fx.to_bw(f) for f in scene(45, seed=92, velocity=(-1.5, 0.5), zoom_rate=0.99)])
    mixed = np.stack([fx.blend(a[25 + j], b[j], (j + 1) / 15) for j in range(14)])
    frames = np.concatenate([a[:25], mixed, b[14:]])  # blend frames are [25, 39)
    frames = np.stack([fx.draw_text(f, "WATCH TILL THE END", center=(0.5, 0.7), size=0.06)[0] if i < 32 else f
                       for i, f in enumerate(frames)])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.kind == "dissolve" and abs(tr.boundary - 32) <= 2


# --------------------------------------------------------------------------- no false cuts


@pytest.mark.parametrize("velocity", [(8.0, 0.0), (0.0, 8.0), (6.0, -6.0), (14.0, 3.0)])
def test_fast_pan_has_no_cuts(velocity):
    frames = scene(45, seed=10, velocity=velocity)
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 1, boundaries(seg)


@pytest.mark.parametrize("zoom_rate", [1.02, 0.98, 1.05])
def test_steady_zoom_has_no_cuts(zoom_rate):
    frames = scene(45, seed=12, velocity=(0.5, 0.0), zoom_rate=zoom_rate)
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 1, boundaries(seg)


def test_shake_has_no_cuts():
    base = scene(60, seed=13, velocity=(0.5, 0.0))
    frames = []
    for i, f in enumerate(base):
        dx, dy = fx.shake_offset(i / FPS, 0.04, 9.0, seed=2)
        frames.append(fx.transform(f, angle=3.0 * np.sin(i), tx=dx, ty=dy * W / H))
    frames = np.stack(frames)
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 1, boundaries(seg)


def test_in_shot_edits_have_no_cuts():
    """Zoom punch, spin, rgb-split glitch, caption pop-in and a B&W switch inside one shot."""
    base = scene(80, seed=14, velocity=(2.0, 0.0))
    out = []
    for i, f in enumerate(base):
        if 10 <= i < 20:  # zoom punch: 1.0 -> 1.4 within two frames, then hold
            f = fx.transform(f, scale=float(fx.interp(1.0, 1.4, (i - 10) / 10, "punch")))
        if 25 <= i < 35:  # fast spin out and back (9 deg/frame)
            f = fx.transform(f, angle=9.0 * min(i - 25, 35 - i), scale=1.25)
        if i in (40, 41):
            f = fx.rgb_split(f, 0.03)
        if i >= 45:
            f, _ = fx.draw_text(f, "WAIT FOR IT", center=(0.5, 0.7), size=0.07)
        if i >= 60:
            f = fx.to_bw(f)
        out.append(f)
    frames = np.stack(out)
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 1, boundaries(seg)


def test_invert_flash_is_not_a_cut():
    """A short photographic-negative flash inside a moving shot (an invert effect)."""
    frames = scene(60, seed=15, velocity=(3.0, 0.0))
    frames[25:29] = [fx.invert(f) for f in frames[25:29]]
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert len(seg.shots) == 1, boundaries(seg)


def test_cut_into_an_inverted_shot_is_still_a_cut():
    frames = np.concatenate([scene(30, seed=16), np.stack([fx.invert(f) for f in scene(30, seed=17)])])
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert boundaries(seg) == [30]


def test_letterboxed_cuts():
    frames, truth = synth.cut_sequence([15, 6, 20, 4, 15], height=H, width=W, velocity=(5.0, 0.0))
    frames = np.stack([fx.letterbox(f, 0.12, 0.12) for f in frames])
    seg = detect(frames)
    check_contract(seg, len(frames))
    assert boundaries(seg) == truth


# --------------------------------------------------------------------------- cut styles


def _whip_frames(img: np.ndarray, e: float, before: bool) -> np.ndarray:
    """Horizontal whip-pan look (shift + motion blur), as the compiler draws it."""
    shift = -0.5 * e * e * (1.0 if before else -1.0)
    moved = fx.transform(img, tx=shift)
    length = int(e * 0.15 * W)
    return cv2.blur(moved, (length, 1)) if length > 1 else moved


def test_whip_cut_is_classified():
    a, b = scene(20, seed=51), scene(20, seed=52)
    a[-2:] = np.stack([_whip_frames(a[-2 + k], e, True) for k, e in enumerate((2 / 3, 1.0))])
    b[:2] = np.stack([_whip_frames(b[k], e, False) for k, e in enumerate((1.0, 0.5))])
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert abs(tr.boundary - 20) <= 1
    assert tr.kind == "whip"
    assert tr.confidence < 1.0


def test_glitch_cut_is_classified():
    a, b = scene(20, seed=61), scene(20, seed=62)
    a[-1] = fx.rgb_split(a[-1], 0.03)
    b[0] = fx.rgb_split(b[0], 0.03)
    frames = np.concatenate([a, b])
    seg = detect(frames)
    check_contract(seg, len(frames))
    (tr,) = seg.transitions
    assert tr.boundary == 20 and tr.kind == "glitch"


# --------------------------------------------------------------------------- robustness


def _viral_sequence(rng: np.random.Generator) -> tuple[np.ndarray, list[int], dict[int, str]]:
    """Short shots with pans/zooms/shakes, one flash, one fade dip, an in-shot flash and captions."""
    lengths = [14, 4, 9, 3, 20, 6, 12, 5, 16, 3, 10, 18]
    shots = []
    for k, n in enumerate(lengths):
        vel = tuple(rng.uniform(-9, 9, 2))
        zoom = float(rng.choice([1.0, 1.02, 0.985]))
        shots.append(scene(n, seed=100 + 13 * k, velocity=vel, zoom_rate=zoom))
    kinds = {4: "flash", 8: "fade_black"}  # keyed by incoming shot index
    shots[3][-1:] = blend_toward(shots[3][-1:], [1.0], WHITE)
    shots[4][:2] = blend_toward(shots[4][:2], [0.6, 0.3], WHITE)
    shots[7][-3:] = blend_toward(shots[7][-3:], [0.4, 0.7, 1.0], BLACK)
    shots[8][:3] = blend_toward(shots[8][:3], [0.8, 0.5, 0.25], BLACK)
    shots[4][9:11] = blend_toward(shots[4][9:11], [1.0, 0.5], WHITE)  # in-shot flash
    for i in range(len(shots[11])):  # shake
        dx, dy = fx.shake_offset(i / FPS, 0.03, 8.0, seed=4)
        shots[11][i] = fx.transform(shots[11][i], tx=dx, ty=dy)
    frames = np.concatenate(shots)
    words = ["WAIT", "FOR", "IT", "SEND", "THIS"]
    frames = np.stack([
        fx.draw_text(f, words[(i // 7) % len(words)], center=(0.5, 0.75), size=0.06)[0] for i, f in enumerate(frames)
    ])
    truth = np.cumsum(lengths)[:-1].tolist()
    return frames, truth, {truth[k - 1]: v for k, v in kinds.items()}


def test_h264_roundtrip_with_noise(tmp_path):
    frames, truth, kinds = _viral_sequence(np.random.default_rng(5))
    frames = add_noise(frames, 6.0, seed=1)
    path = tmp_path / "edit.mp4"
    with VideoWriter(path, W, H, FPS, crf=30) as vw:
        for f in frames:
            vw.write(f)
    ctx = load_context(path)
    assert ctx.n_frames == len(frames)
    seg = detect_shots(ctx)
    check_contract(seg, ctx.n_frames)
    pred = boundaries(seg)
    hits = match(truth, pred, tol=1)
    precision, recall = hits / max(len(pred), 1), hits / len(truth)
    assert recall == 1.0 and precision == 1.0, (truth, pred)
    by_boundary = {t.boundary: t.kind for t in internal(seg)}
    for b, kind in kinds.items():
        got = [k for f, k in by_boundary.items() if abs(f - b) <= 1]
        assert got == [kind], (b, kind, by_boundary)
    plain = [by_boundary[p] for p in pred if all(abs(p - b) > 1 for b in kinds)]
    assert plain.count("cut") >= len(plain) - 1  # at most one cut mislabelled as a style


def test_heavy_noise_in_memory():
    frames, truth = synth.cut_sequence([12, 5, 16, 3, 14], height=H, width=W, velocity=(6.0, 2.0))
    seg = detect(add_noise(frames, 14.0, seed=3))
    check_contract(seg, len(frames))
    assert boundaries(seg) == truth


# --------------------------------------------------------------------------- edge cases & performance


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_tiny_clips(n):
    frames = scene(n, seed=15)
    seg = detect(frames)
    check_contract(seg, n)
    assert len(seg.shots) == 1


def test_constant_and_black_clips():
    black = np.zeros((30, H, W, 3), np.uint8)
    seg = detect(black)
    check_contract(seg, 30)
    assert len(seg.shots) == 1
    still = np.repeat(scene(1, seed=16), 30, axis=0)
    seg = detect(still)
    check_contract(seg, 30)
    assert len(seg.shots) == 1 and seg.transitions == []


def test_params_are_overridable():
    p = ShotParams.from_context(context_from_arrays(scene(3, seed=17), FPS, config=AnalysisConfig(
        params={"shots": {"mad_k": 6.0, "dissolve_lags": [5, 10]}})))
    assert p.mad_k == 6.0 and p.dissolve_lags == (5, 10)


def test_runtime_one_minute_clip():
    rng = np.random.default_rng(0)
    lengths: list[int] = []
    while sum(lengths) < 1800:
        lengths.append(int(rng.choice([3, 4, 6, 8, 12, 20, 30, 45])))
    lengths[-1] -= sum(lengths) - 1800
    if lengths[-1] < 3:
        lengths[-2] += lengths.pop()
    parts, truth, pos = [], [], 0
    for k, n in enumerate(lengths):
        vel = tuple(rng.uniform(-8, 8, 2))
        parts.append(scene(n, seed=1000 + k, velocity=vel, zoom_rate=float(rng.choice([1.0, 1.015]))))
        if k:
            truth.append(pos)
        pos += n
    frames = add_noise(np.concatenate(parts), 4.0)
    ctx = context_from_arrays(frames, FPS)
    t0 = time.perf_counter()
    seg = detect_shots(ctx)
    elapsed = time.perf_counter() - t0
    check_contract(seg, 1800)
    pred = boundaries(seg)
    hits = match(truth, pred, tol=1)
    assert hits / len(truth) >= 0.97 and hits / max(len(pred), 1) >= 0.97
    assert elapsed < 15.0, elapsed


# --------------------------------------------------------------------------- real footage (opt-in)

_LABELS = Path(__file__).parent / "data" / "sintel_trailer_boundaries.json"


def _score_against_labels(seg: ShotSegmentation, fps: float, labels: dict, tol: int = 2) -> tuple[float, float]:
    """Precision / recall of the shot boundaries: cuts match within ``tol`` frames, gradual
    transitions anywhere inside their labelled span (+- ``tol``)."""
    pred = boundaries(seg)
    used: set[int] = set()
    hits = 0
    for b in labels["boundaries"]:
        f = int(round(b["time"] * fps))
        if b["kind"] == "cut":
            lo, hi = f - tol, f + tol
        else:
            lo, hi = int(round(b["start"] * fps)) - tol, int(round(b["end"] * fps)) + tol
        cands = [p for p in pred if lo <= p <= hi and p not in used]
        if cands:
            used.add(min(cands, key=lambda p: abs(p - f)))
            hits += 1
    return hits / max(len(pred), 1), hits / len(labels["boundaries"])


@pytest.mark.skipif(os.environ.get("EDITAST_REAL_TESTS") != "1", reason="set EDITAST_REAL_TESTS=1 (downloads ~4 MB)")
def test_sintel_trailer_boundaries():
    """Sintel trailer (Blender Foundation, CC-BY 3.0) against hand-labelled boundaries:
    fades through black and white, title cards, dark cuts and long dissolves."""
    labels = json.loads(_LABELS.read_text(encoding="utf-8"))
    cache = Path.home() / ".cache" / "editast" / "test-media"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "sintel_trailer-480p.mp4"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(labels["source"], tmp)
        tmp.replace(path)
    ctx = load_context(path)
    assert abs(ctx.fps - labels["fps"]) < 0.01
    seg = detect_shots(ctx)
    check_contract(seg, ctx.n_frames)
    precision, recall = _score_against_labels(seg, ctx.fps, labels)
    assert recall >= 0.9 and precision >= 0.9, (precision, recall, [round(b / ctx.fps, 2) for b in boundaries(seg)])
