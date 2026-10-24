"""Tests for the motion analyzer (zoom / pan / rotate / shake).

Ground truth is rendered with the same geometry the renderer uses
(:func:`editast.fx.similarity_matrix` / :func:`editast.fx.transform`), so these tests
pin down the sign and unit conventions as well as the detection itself.
"""

import time

import cv2
import numpy as np
import pytest

from editast import fx
from editast.analyzers.motion import _fit_jitter, analyze_motion
from editast.analyzers.types import MotionAnalysis, ShotSegmentation, ShotSpan, TransitionSpan
from editast.context import AnalysisConfig, context_from_arrays, load_context
from editast.media import VideoWriter
from editast.testing import synth

H, W, FPS = 320, 180, 30.0  # portrait analysis frames (width 180, height 320)


def render(n: int, params, seed: int = 0, margin: float = 2.0) -> np.ndarray:
    """``n`` frames of ``fx.similarity_matrix(**params(i))`` applied to an "infinite" texture.

    Same matrix as ``fx.transform`` on the output frame, but the content comes from a
    larger canvas, so no reflected borders enter the view.
    """
    canvas = synth.texture(int(H * margin), int(W * margin), seed)
    off = np.array([(canvas.shape[1] - W) / 2.0, (canvas.shape[0] - H) / 2.0])
    out = np.empty((n, H, W, 3), np.uint8)
    for i in range(n):
        m = fx.similarity_matrix(W, H, **params(i))
        m[:, 2] -= m[:, :2] @ off
        out[i] = cv2.warpAffine(canvas, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out


def single_shot(n: int) -> ShotSegmentation:
    return ShotSegmentation(shots=[ShotSpan(0, 0, n)], transitions=[], boundary_score=np.zeros(n))


def analyze(frames: np.ndarray, seg: ShotSegmentation | None = None) -> MotionAnalysis:
    ctx = context_from_arrays(frames, FPS)
    return analyze_motion(ctx, seg or single_shot(len(frames)))


def effects(ma: MotionAnalysis, kind: str, shot: int = 0) -> list:
    return [e for e in ma.effects.by_shot.get(shot, []) if e.type == kind]


def shake_params(i: int, amplitude: float = 0.01, frequency: float = 6.0, seed: int = 0) -> dict:
    """fx.shake_offset jitter; both axes in width units (Shake convention)."""
    dx, dy = fx.shake_offset(i / FPS, amplitude, frequency, seed=seed)
    return {"tx": dx, "ty": dy * W / H}


def add_noise(frames: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(0.0, sigma, frames.shape)
    return np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- conventions


@pytest.mark.parametrize("kw, field, expected", [
    ({"scale": 1.06}, "scale", 1.06),
    ({"scale": 0.95}, "scale", 0.95),
    ({"angle": 3.0}, "rotation", 3.0),
    ({"angle": -2.0}, "rotation", -2.0),
    ({"tx": 0.04}, "tx", 0.04),
    ({"ty": -0.03}, "ty", -0.03),
])
def test_frame_motion_matches_fx_transform(kw, field, expected):
    base = synth.texture(H, W, seed=5)
    ma = analyze(np.stack([base, fx.transform(base, **kw)]))
    fm = ma.frame_motion
    assert not fm.valid[0] and fm.scale[0] == 1.0  # frame 0 = shot start: identity
    assert fm.valid[1]
    assert getattr(fm, field)[1] == pytest.approx(expected, abs=0.002 if field != "rotation" else 0.05)
    assert fm.magnitude[1] > 0


# --------------------------------------------------------------------------- (a) static


def test_static_scene_has_no_effects():
    frames = add_noise(render(60, lambda i: {}, seed=1), 2.0)
    ma = analyze(frames)
    assert ma.effects.by_shot.get(0, []) == []
    s = ma.summaries[0]
    assert s.camera == "static"
    assert s.zoom_ratio == pytest.approx(1.0, abs=0.01)
    assert abs(s.pan[0]) < 0.01 and abs(s.pan[1]) < 0.01
    fm = ma.frame_motion
    assert fm.valid[1:].mean() > 0.9
    assert np.abs(fm.tx).max() < 0.003 and np.abs(fm.rotation).max() < 0.2


# --------------------------------------------------------------------------- (b) zoom


def test_ease_out_zoom():
    # 1.0 -> 1.3 over 1 s starting at 0.5 s in a 2.5 s shot.
    frames = render(75, lambda i: {"scale": fx.interp(1.0, 1.3, (i - 15) / 30, "ease_out")}, seed=2)
    ma = analyze(frames)
    (zoom,) = effects(ma, "zoom")
    assert zoom.scale_from == 1.0
    assert zoom.scale_to == pytest.approx(1.3, rel=0.03)
    assert zoom.easing == "ease_out"
    assert zoom.start == pytest.approx(0.5, abs=0.1) and zoom.end == pytest.approx(1.5, abs=0.15)
    assert zoom.center == pytest.approx((0.5, 0.5), abs=0.03)
    assert zoom.digital is True
    assert zoom.confidence > 0.7
    values = [k.v for k in zoom.keyframes]
    assert 6 <= len(values) <= 10 and values[0] == pytest.approx(1.0, abs=0.01)
    assert np.all(np.diff(values) > -1e-3) and values[-1] == pytest.approx(1.3, rel=0.03)
    assert not effects(ma, "pan") and not effects(ma, "rotate") and not effects(ma, "shake")
    assert ma.summaries[0].camera == "zoom"
    assert ma.summaries[0].zoom_ratio == pytest.approx(1.3, rel=0.03)


@pytest.mark.parametrize("easing, center, noise", [
    ("linear", (0.3, 0.65), 0.0),
    ("ease_out", (0.7, 0.3), 4.0),
    ("punch", (0.4, 0.4), 4.0),
])
def test_off_center_zoom_recovers_center_without_pan(easing, center, noise):
    frames = render(60, lambda i: {"scale": fx.interp(1.0, 1.3, (i - 10) / 30, easing), "center": center}, seed=3)
    ma = analyze(add_noise(frames, noise) if noise else frames)
    (zoom,) = effects(ma, "zoom")
    assert zoom.center == pytest.approx(center, abs=0.05)
    assert zoom.easing == easing
    assert not effects(ma, "pan")


def test_camera_pan_next_to_zoom_keeps_zoom_centred():
    # Camera pans (content translation magnified by the digital zoom on top), stops, and
    # 0.17 s later a centred zoom starts: the pan must not leak into the zoom centre.
    def params(i):
        scale = fx.interp(1.0, 1.25, (i - 35) / 20)
        return {"scale": scale, "tx": scale * fx.interp(0.0, 0.15, (i - 5) / 25)}

    ma = analyze(add_noise(render(80, params, seed=5), 3.0))
    (zoom,) = effects(ma, "zoom")
    assert zoom.center == (0.5, 0.5)
    assert zoom.scale_to == pytest.approx(1.25, rel=0.03)
    (pan,) = effects(ma, "pan")
    assert pan.dx == pytest.approx(0.15, rel=0.07) and pan.end <= zoom.start + 0.05


def test_zoom_in_then_out_gives_two_relative_zooms():
    def params(i):
        if i < 45:
            return {"scale": fx.interp(1.0, 1.25, (i - 10) / 20, "ease_out")}
        return {"scale": fx.interp(1.25, 1.0, (i - 50) / 25, "ease_in_out")}

    ma = analyze(render(90, params, seed=4))
    zin, zout = sorted(effects(ma, "zoom"), key=lambda z: z.start)
    assert zin.scale_to == pytest.approx(1.25, rel=0.03)
    assert zout.scale_to == pytest.approx(1.0 / 1.25, rel=0.03)
    assert zout.params["scale_start"] == pytest.approx(1.25, rel=0.03)
    assert zin.end <= zout.start


# --------------------------------------------------------------------------- (c) pan


@pytest.mark.parametrize("dx, dy, easing", [(0.2, 0.0, "linear"), (-0.1, -0.12, "ease_in_out")])
def test_pan(dx, dy, easing):
    def params(i):
        p = fx.ease((i - 10) / 45, easing)
        return {"tx": dx * p, "ty": dy * p}

    ma = analyze(add_noise(render(75, params, seed=6), 3.0))
    (pan,) = effects(ma, "pan")
    assert pan.dx == pytest.approx(dx, abs=0.1 * max(abs(dx), abs(dy)))
    assert pan.dy == pytest.approx(dy, abs=0.1 * max(abs(dx), abs(dy)))
    assert pan.easing == easing
    assert pan.start == pytest.approx(10 / FPS, abs=0.1) and pan.end == pytest.approx(55 / FPS, abs=0.1)
    assert not effects(ma, "zoom") and not effects(ma, "rotate") and not effects(ma, "shake")
    assert ma.summaries[0].camera == "pan"


def test_synth_scene_camera_velocity_is_opposite_content_motion():
    # synth.scene's velocity moves the *camera* right, so content moves left: dx < 0.
    ma = analyze(synth.scene(60, H, W, seed=3, velocity=(2.0, 0.0)))
    (pan,) = effects(ma, "pan")
    assert pan.dx == pytest.approx(-2.0 * 59 / W, rel=0.05)
    assert ma.frame_motion.magnitude[1:].mean() == pytest.approx(2.0 / W, rel=0.1)


def test_fast_eased_pan_is_not_shake():
    ma = analyze(render(75, lambda i: {"tx": fx.interp(0.0, 0.3, (i - 15) / 15, "ease_out")}, seed=7))
    (pan,) = effects(ma, "pan")
    assert pan.dx == pytest.approx(0.3, rel=0.05)
    assert pan.easing == "ease_out"
    assert not effects(ma, "shake")


# --------------------------------------------------------------------------- (d) rotate


@pytest.mark.parametrize("angle", [10.0, -8.0])
def test_rotation(angle):
    ma = analyze(render(60, lambda i: {"angle": fx.interp(0.0, angle, (i - 10) / 40)}, seed=8))
    (rot,) = effects(ma, "rotate")
    assert rot.angle_from == 0.0
    assert rot.angle_to == pytest.approx(angle, rel=0.1)
    assert rot.easing == "linear"
    assert not effects(ma, "zoom") and not effects(ma, "pan")
    assert ma.summaries[0].camera == "rotate"
    assert ma.summaries[0].rotation == pytest.approx(angle, rel=0.1)


# --------------------------------------------------------------------------- (e) shake


def test_shake_amplitude_and_frequency():
    ma = analyze(render(90, shake_params, seed=9))
    (shake,) = effects(ma, "shake")
    assert shake.amplitude == pytest.approx(0.01, rel=0.4)
    assert shake.frequency == pytest.approx(6.0, rel=0.25)
    assert shake.rotation_amplitude < 0.2
    assert shake.end - shake.start > 2.0
    assert not effects(ma, "zoom") and not effects(ma, "pan") and not effects(ma, "rotate")
    assert ma.summaries[0].camera == "handheld"


def test_shake_is_localised():
    # Jitter only between 0.8 s and 2.0 s of a 3 s static shot.
    ma = analyze(render(90, lambda i: shake_params(i, 0.008, 9.0, seed=1) if 24 <= i < 60 else {}, seed=10))
    (shake,) = effects(ma, "shake")
    assert shake.start == pytest.approx(0.8, abs=0.2) and shake.end == pytest.approx(2.0, abs=0.2)
    assert shake.amplitude == pytest.approx(0.008, rel=0.4)
    assert shake.frequency == pytest.approx(9.0, rel=0.25)


def test_short_shake_frequency_and_amplitude():
    # Two cycles of jitter filling a 0.37 s shot: too short for a periodogram, but the
    # sinusoid fit recovers frequency and amplitude, and the shake is not dropped for
    # lack of zero crossings on one axis.
    ma = analyze(render(11, lambda i: shake_params(i, 0.019, 5.4, seed=310362), seed=17))
    (shake,) = effects(ma, "shake")
    assert shake.start == 0.0 and shake.end == pytest.approx(11 / FPS)
    assert shake.frequency == pytest.approx(5.4, abs=0.3)
    assert shake.amplitude == pytest.approx(0.019, rel=0.15)


@pytest.mark.parametrize("frequency", [5.0, 7.3, 9.6])
def test_shake_frequency_is_precise(frequency):
    ma = analyze(add_noise(render(16, lambda i: shake_params(i, 0.01, frequency, seed=3), seed=18), 2.0))
    (shake,) = effects(ma, "shake")
    assert shake.frequency == pytest.approx(frequency, abs=0.15)
    assert shake.amplitude == pytest.approx(0.01, rel=0.1)


def test_fit_jitter_tells_oscillation_from_kinks():
    fps, ratios = 30.0, (1.0, 1.13, 0.9)
    t = np.arange(12) / fps
    x = 0.01 * np.sin(2 * np.pi * 6.2 * t + 0.4) + 0.004 * t
    y = 0.01 * np.sin(2 * np.pi * 6.2 * 1.13 * t + 2.0)
    zero = np.zeros_like(t)
    freq, share, amps = _fit_jitter([x, y, zero], ratios, fps, degree=2)
    assert freq == pytest.approx(6.2, abs=0.05) and share > 0.99
    assert amps[:2] == pytest.approx([0.01, 0.01], rel=0.02) and amps[2] == pytest.approx(0.0, abs=1e-9)
    _, share, _ = _fit_jitter([np.diff(v) for v in (x, y, zero)], ratios, fps)
    assert share > 0.95  # periodicity is judged on the velocity
    # A pan that halts and resumes (velocity steps): no single sinusoid explains its velocity.
    for pause in (3, 5, 9, 13):
        path = np.concatenate([np.arange(8), np.full(pause, 7.0), 7 + np.arange(1, 9)]) * 0.016
        v = np.diff(path)
        _, share, _ = _fit_jitter([v, np.zeros_like(v), np.zeros_like(v)], ratios, fps)
        assert share < 0.9


def test_pan_halting_under_freeze_is_one_pan_without_shake():
    # A camera pan whose footage freezes for 0.3 s midway (a freeze-frame effect): one
    # move with a pause, not a pan plus a "shake" made of the two velocity kinks.
    k = [min(i, 8) if i < 17 else i - 8 for i in range(25)]
    frames = render(25, lambda i: {"tx": 0.016 * k[i]}, seed=19)
    ma = analyze(add_noise(frames, 2.0))
    (pan,) = effects(ma, "pan")
    assert pan.start < 0.1 and pan.end > 0.75
    assert pan.dx > 0.15 and abs(pan.dy) < 0.02
    assert not effects(ma, "shake")


def test_same_direction_fragments_merge_into_one_pan():
    # Pan right, a small wobble back (above the zig-zag hysteresis, below the threshold),
    # then on to the right: one camera move, not two fragments.
    def tx(i):
        if i < 30:
            return fx.interp(0.0, 0.12, (i - 10) / 20)
        if i < 34:
            return fx.interp(0.12, 0.095, (i - 30) / 4)
        return fx.interp(0.095, 0.23, (i - 34) / 20)

    ma = analyze(add_noise(render(70, lambda i: {"tx": tx(i)}, seed=20), 2.0))
    (pan,) = effects(ma, "pan")
    assert pan.dx == pytest.approx(0.23, rel=0.15)
    assert pan.start < 0.45 and pan.end > 1.7


def test_independent_subject_motion_is_not_a_camera_move():
    # Three large parts of the picture moving on their own (a subject crossing to the
    # right while growing as it approaches, one sinking, one crossing to the left), like
    # a creature filling the frame: no single motion explains most of the picture, so no
    # camera zoom / pan / rotate is reported.
    layers = [synth.texture(H, 4 * W, seed=s) for s in (21, 22, 23)]
    top, middle, bottom = slice(0, int(0.4 * H)), slice(int(0.4 * H), int(0.65 * H)), slice(int(0.65 * H), H)
    frames = []
    for i in range(45):
        f = np.empty((H, W, 3), np.uint8)
        m = fx.similarity_matrix(W, H, scale=fx.interp(1.0, 1.5, i / 44), center=(0.5, 0.2))
        m[:, 2] += [6.0 * i - 1.5 * W, 0.0]  # moves right by 6 px per frame
        f[top] = cv2.warpAffine(layers[0], m, (W, H), flags=cv2.INTER_LINEAR)[top]
        f[middle] = np.roll(layers[1][:, :W], 4 * i, axis=0)[middle]  # moves down by 4 px per frame
        f[bottom] = layers[2][bottom, 6 * i:6 * i + W]  # moves left by 6 px per frame
        frames.append(f)
    ma = analyze(add_noise(np.stack(frames), 2.0))
    assert not effects(ma, "zoom") and not effects(ma, "pan") and not effects(ma, "rotate")


def test_confidence_reflects_evidence():
    # A clean digital move is confident; the same move barely above the threshold is less so.
    big = analyze(render(60, lambda i: {"tx": fx.interp(0.0, 0.2, (i - 10) / 30)}, seed=24))
    small = analyze(add_noise(render(60, lambda i: {"tx": fx.interp(0.0, 0.055, (i - 10) / 30)}, seed=24), 3.0))
    (pb,) = effects(big, "pan")
    (ps,) = effects(small, "pan")
    assert pb.confidence > 0.85
    assert 0.3 <= ps.confidence < pb.confidence - 0.2


# --------------------------------------------------------------------------- (f) shot boundaries


def test_no_motion_across_cuts_and_transitions():
    a = render(40, lambda i: {"tx": fx.interp(0.0, 0.15, i / 39)}, seed=11)
    b = render(40, lambda i: {}, seed=12)
    c = render(40, lambda i: {"angle": fx.interp(0.0, -6.0, i / 39)}, seed=13)
    frames = np.concatenate([a, b, c])
    for k in range(5):  # dissolve b -> c over frames 78..82
        frames[78 + k] = fx.blend(b[min(39, 36 + k)], c[k], (k + 1) / 6)
    seg = ShotSegmentation(
        shots=[ShotSpan(0, 0, 40), ShotSpan(1, 40, 80), ShotSpan(2, 80, 120)],
        transitions=[TransitionSpan("cut", 40, 40, 40, 0, 1), TransitionSpan("dissolve", 80, 78, 83, 1, 2)],
        boundary_score=np.zeros(120),
    )
    ma = analyze(frames, seg)
    fm = ma.frame_motion
    for t in (0, 40, 80):
        assert not fm.valid[t] and fm.scale[t] == 1.0 and fm.tx[t] == 0.0 and fm.magnitude[t] == 0.0
    assert not fm.valid[78:83].any() and (fm.tx[78:83] == 0).all() and (fm.rotation[78:83] == 0).all()
    assert len(fm.scale) == len(fm.magnitude) == 120

    (pan,) = effects(ma, "pan", 0)
    assert pan.dx == pytest.approx(0.15, rel=0.1)
    assert ma.effects.by_shot.get(1, []) == [] and ma.summaries[1].camera == "static"
    (rot,) = effects(ma, "rotate", 2)
    assert rot.angle_to < -4.0
    for shot in seg.shots:  # effect spans lie within their shot
        for e in ma.effects.by_shot.get(shot.index, []):
            assert shot.f0 / FPS - 1e-9 <= e.start < e.end <= shot.f1 / FPS + 1e-9


def test_zoom_survives_in_shot_flash_invert_and_blank_frames():
    # A linear zoom with a strong flash, an inverted frame and a white frame inside it:
    # flashes / inversions are measured on contrast-normalised (inverted) frames and the
    # blank frame is bridged, so the cumulative zoom stays exact.
    frames = render(60, lambda i: {"scale": fx.interp(1.0, 1.3, (i - 10) / 35)}, seed=16)
    frames[20:23] = [fx.flash(f, 0.75) for f in frames[20:23]]
    frames[28] = fx.invert(frames[28])
    frames[36] = 255
    ma = analyze(add_noise(frames, 2.0))
    fm = ma.frame_motion
    assert fm.valid[[20, 21, 23, 28, 29]].all()
    assert not fm.valid[[36, 37]].any()  # bridged across the blank frame: used, but not "valid"
    assert np.prod(fm.scale[1:]) == pytest.approx(1.3, rel=0.02)
    (zoom,) = effects(ma, "zoom")
    assert zoom.scale_to == pytest.approx(1.3, rel=0.02) and zoom.easing == "linear"
    assert not effects(ma, "pan") and not effects(ma, "shake")


def test_unstructured_frames_give_no_effects():
    # Independent noise frames (nothing trackable): no invented motion, bounded runtime.
    rng = np.random.default_rng(3)
    frames = np.stack([cv2.GaussianBlur(f, (0, 0), 1.0) for f in rng.integers(0, 256, (90, H, W, 3), np.uint8)])
    ma = analyze(frames)
    assert ma.effects.by_shot.get(0, []) == []
    assert ma.frame_motion.valid.mean() < 0.1
    assert ma.summaries[0].camera in ("static", "complex")


def test_degenerate_shots_and_blank_frames():
    frames = render(12, lambda i: {"tx": 0.01 * i}, seed=14)
    frames[6:9] = 0  # black frames
    seg = ShotSegmentation(
        shots=[ShotSpan(0, 0, 1), ShotSpan(1, 1, 3), ShotSpan(2, 3, 12)],
        transitions=[], boundary_score=np.zeros(12),
    )
    ma = analyze(frames, seg)
    fm = ma.frame_motion
    assert set(ma.summaries) == {0, 1, 2}
    assert not fm.valid[[0, 1, 3, 7, 8]].any()
    for arr in (fm.scale, fm.rotation, fm.tx, fm.ty, fm.residual, fm.magnitude):
        assert len(arr) == 12 and np.isfinite(arr).all()
    for name, sig in ma.effects.signals.items():
        assert len(sig) == 12 and np.isfinite(sig).all(), name


# --------------------------------------------------------------------------- (g) real codec


def test_h264_roundtrip_zoom_pan_shake(tmp_path):
    def params(i):
        p = {"scale": fx.interp(1.0, 1.25, (i - 10) / 30, "ease_out"),
             "tx": fx.interp(0.0, 0.15, (i - 45) / 30)}
        if 80 <= i < 120:
            j = shake_params(i, 0.008, 8.0, seed=2)
            p["tx"] += j["tx"]
            p["ty"] = j["ty"]
        return p

    path = tmp_path / "combo.mp4"
    with VideoWriter(path, W, H, FPS, crf=26) as vw:
        for frame in render(130, params, seed=4):
            vw.write(frame)
    ctx = load_context(path, AnalysisConfig())
    assert ctx.size == (W, H)
    ma = analyze_motion(ctx, single_shot(ctx.n_frames))

    (zoom,) = effects(ma, "zoom")
    assert zoom.scale_to == pytest.approx(1.25, rel=0.04)
    assert zoom.easing == "ease_out" and zoom.end < 1.6
    (pan,) = effects(ma, "pan")
    assert pan.dx == pytest.approx(0.15, rel=0.1) and abs(pan.dy) < 0.02
    (shake,) = effects(ma, "shake")
    assert shake.amplitude == pytest.approx(0.008, rel=0.4)
    assert shake.frequency == pytest.approx(8.0, rel=0.25)
    assert shake.start == pytest.approx(80 / FPS, abs=0.25) and shake.end == pytest.approx(120 / FPS, abs=0.25)
    assert ma.summaries[0].camera == "complex"


def test_captions_and_letterbox_do_not_bias_slow_pan():
    def dress(frame):
        out, _ = fx.draw_text(frame, "WAIT FOR IT", center=(0.5, 0.75), size=0.05)
        out, _ = fx.draw_text(out, "THIS CAR IS INSANE", center=(0.5, 0.2), size=0.04)
        return fx.letterbox(out, 0.1, 0.1)

    frames = render(75, lambda i: {"tx": fx.interp(0.0, -0.2, (i - 10) / 45)}, seed=6)
    ma = analyze(np.stack([dress(f) for f in frames]))
    (pan,) = effects(ma, "pan")
    assert pan.dx == pytest.approx(-0.2, rel=0.07)
    assert not effects(ma, "zoom")


# --------------------------------------------------------------------------- performance


def test_performance_60s_clip():
    n = 1800

    def params(i):
        return {"scale": 1.0 + 0.1 * np.sin(i / 50), "tx": 0.1 * np.sin(i / 37), "angle": 3.0 * np.sin(i / 60)}

    frames = render(n, params, seed=15)
    bounds = list(range(0, n, 90)) + [n]
    seg = ShotSegmentation(
        shots=[ShotSpan(k, bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)],
        transitions=[], boundary_score=np.zeros(n),
    )
    ctx = context_from_arrays(frames, FPS)
    t0 = time.perf_counter()
    ma = analyze_motion(ctx, seg)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0, f"motion analysis took {elapsed:.1f}s for {n} frames"
    assert len(ma.frame_motion.valid) == n and len(ma.summaries) == len(seg.shots)
