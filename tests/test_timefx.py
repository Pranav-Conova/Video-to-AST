"""Tests for the time-effects analyzer (speed ramps, slow motion, freezes).

Ground truth follows the renderer's timing convention
(:func:`editast.render.compiler.source_positions`): a per-frame playback rate
``rate[i]`` advances the source by ``rate[i]`` frames from output frame ``i``
to ``i + 1``, so a Speed span ``[a, b)`` covers the steps out of frames
``a .. b-1``, and a Freeze over ``[a, b)`` shows ``b - a`` identical frames.
"""

import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from editast import fx
from editast.analyzers.timefx import analyze_time_effects
from editast.analyzers.types import (
    EffectsResult, FrameMotion, MotionAnalysis, ShotSegmentation, ShotSpan, TransitionSpan,
)
from editast.ast.nodes import Shake
from editast.context import AnalysisConfig, context_from_arrays, load_context
from editast.media import VideoWriter
from editast.testing import synth

H, W, FPS = 320, 180, 30.0  # portrait analysis frames


def camera(rate, seed: int = 0, v=(5.0, 1.0), size=(W, H), shake: float = 0.0, zoom=None) -> np.ndarray:
    """Frames of a camera travelling over a large texture at ``v`` px per source frame.

    ``rate[i]`` is the playback rate of the step from output frame ``i`` to ``i + 1``.
    """
    rate = np.asarray(rate, float)
    w, h = size
    pos = np.concatenate([[0.0], np.cumsum(rate[:-1])])
    travel = np.abs(np.asarray(v)) * max(pos.max(), 1.0)
    canvas = synth.texture(int(h * 1.6 + travel[1]), int(w * 1.6 + travel[0]), seed)
    out = np.empty((rate.size, h, w, 3), np.uint8)
    for i, p in enumerate(pos):
        dx, dy = fx.shake_offset(i / FPS, shake, 6.0, seed=seed) if shake else (0.0, 0.0)
        s = 1.0 if zoom is None else float(zoom[i])
        x = 0.3 * w + p * v[0] - dx * w
        y = 0.3 * h + p * v[1] - dy * w
        m = np.float32([[s, 0, w / 2 - s * (x + w / 2)], [0, s, h / 2 - s * (y + h / 2)]])
        out[i] = cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out


def segmentation(lengths, transitions=()) -> ShotSegmentation:
    b = np.cumsum([0, *lengths])
    shots = [ShotSpan(i, int(b[i]), int(b[i + 1])) for i in range(len(lengths))]
    return ShotSegmentation(shots=shots, transitions=list(transitions), boundary_score=np.zeros(int(b[-1])))


def analyze(frames, seg=None, motion=None, noise: float = 0.0, **params) -> EffectsResult:
    if noise:
        rng = np.random.default_rng(0)
        frames = np.clip(frames + rng.normal(0.0, noise, frames.shape), 0, 255).astype(np.uint8)
    ctx = context_from_arrays(frames, FPS)
    if params:
        ctx = replace(ctx, config=AnalysisConfig(params={"timefx": params}))
    return analyze_time_effects(ctx, seg or segmentation([len(frames)]), motion)


def effects(res: EffectsResult, kind: str, shot: int = 0) -> list:
    return [e for e in res.by_shot.get(shot, []) if e.type == kind]


def all_effects(res: EffectsResult) -> list:
    return [e for effs in res.by_shot.values() for e in effs]


def frames_of(effect) -> tuple[float, float]:
    return effect.start * FPS, effect.end * FPS


def assert_span(effect, f0: int, f1: int, tol: float = 2.0) -> None:
    a, b = frames_of(effect)
    assert abs(a - f0) <= tol and abs(b - f1) <= tol, (a, b, f0, f1)


def eased(n: int, a: float, b: float, easing: str) -> np.ndarray:
    """Rates of an ``n``-frame ramp, sampled at frame centres like the renderer."""
    return fx.interp(a, b, (np.arange(n) + 0.5) / n, easing)


# --------------------------------------------------------------------------- speed


def test_slowmo_from_resampled_motion():
    rate = np.r_[np.ones(40), np.full(40, 0.25), np.ones(40)]
    res = analyze(camera(rate))
    slow = effects(res, "speed")
    assert [e.kind for e in slow] == ["slowmo"]
    assert_span(slow[0], 40, 80)
    assert slow[0].rate_from == pytest.approx(0.25, rel=0.35)
    assert 0.4 <= slow[0].confidence <= 0.8
    assert slow[0].keyframes and all(slow[0].start <= k.t < slow[0].end for k in slow[0].keyframes)
    assert not effects(res, "freeze")
    r = res.signals["speed_rate"]
    assert np.median(r[45:75]) == pytest.approx(0.25, rel=0.35)
    assert np.median(r[5:35]) == pytest.approx(1.0, rel=0.15)


def test_ramp_down_and_up():
    rate = np.r_[np.ones(40), eased(12, 1.0, 0.25, "ease_out"), np.full(28, 0.25),
                 eased(12, 0.25, 1.0, "ease_in"), np.ones(28)]
    res = analyze(camera(rate))
    sp = effects(res, "speed")
    assert [e.kind for e in sp] == ["ramp_down", "slowmo", "ramp_up"]
    down, slow, up = sp
    assert_span(down, 40, 52)
    assert_span(slow, 52, 80)
    assert_span(up, 80, 92)
    assert down.rate_from == pytest.approx(1.0, rel=0.25) and down.rate_to == pytest.approx(0.25, rel=0.35)
    assert up.rate_from == pytest.approx(0.25, rel=0.35) and up.rate_to == pytest.approx(1.0, rel=0.25)
    assert down.easing == "ease_out" and up.easing == "ease_in"
    assert slow.rate_from == pytest.approx(0.25, rel=0.35)
    # Ramps are monotonic and their keyframes sample the rate curve.
    v = [k.v for k in down.keyframes]
    assert v[0] > 0.7 and v[-1] < 0.5


def test_long_ramp_from_plateau_and_natural_deceleration():
    # A 1 s eased ramp from a steady plateau is an edit ...
    rate = np.r_[np.ones(15), eased(30, 1.0, 0.3, "ease_out")]
    sp = effects(analyze(camera(rate)), "speed")
    assert sp and sp[0].kind == "ramp_down"
    assert frames_of(sp[0])[0] == pytest.approx(15, abs=4)
    # ... a camera gradually slowing down (no plateau-ramp-plateau shape) is not.
    assert not all_effects(analyze(camera(np.linspace(1.5, 0.3, 60))))
    assert not all_effects(analyze(camera(np.linspace(0.3, 1.5, 60))))


def test_long_ramp_up_from_shot_start():
    """A ramp may start on the shot's first frame and take most of the shot (no slow plateau
    before it); the normal-speed plateau after it anchors it as an edit."""
    rate = np.r_[eased(45, 0.4, 1.0, "ease_in"), np.ones(30)]
    sp = effects(analyze(camera(rate)), "speed")
    ups = [e for e in sp if e.kind == "ramp_up"]
    assert len(ups) == 1, [e.kind for e in sp]
    assert frames_of(ups[0])[1] == pytest.approx(45, abs=6)
    assert ups[0].rate_from < 0.65 and ups[0].rate_to == pytest.approx(1.0, rel=0.25)
    assert all(e.start < ups[0].end for e in sp)  # nothing after the ramp: normal speed


def camera_path(vel, seed: int = 0, size=(W, H)) -> np.ndarray:
    """Frames of a camera moving with per-frame velocity ``vel`` (n, 2) px over a large texture."""
    vel = np.asarray(vel, float)
    w, h = size
    pos = np.vstack([[0.0, 0.0], np.cumsum(vel[:-1], axis=0)])
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    canvas = synth.texture(int(h * 1.6 + hi[1] - lo[1]), int(w * 1.6 + hi[0] - lo[0]), seed)
    out = np.empty((len(vel), h, w, 3), np.uint8)
    for i, (px, py) in enumerate(pos - lo):
        m = np.float32([[1, 0, -(0.3 * w + px)], [0, 1, -(0.3 * h + py)]])
        out[i] = cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out


def test_speed_dip_with_changed_motion_is_natural():
    """A playback-rate change scales the motion field; a camera that slows down while turning
    (the field changes direction) is natural motion, not slow motion."""
    fast = np.tile([5.0, 1.0], (40, 1))
    turned = np.r_[fast, np.tile([0.3, 1.6], (25, 1)), fast]
    assert not effects(analyze(camera_path(turned), noise=1.0), "speed")
    # Control: the same dip keeping the direction of travel is slow motion.
    scaled = np.r_[fast, np.tile([1.5, 0.3], (25, 1)), fast]
    sp = effects(analyze(camera_path(scaled), noise=1.0), "speed")
    assert [e.kind for e in sp] == ["slowmo"]
    assert sp[0].params["agreement"] > 0.9


def test_duplicate_frame_slow_motion():
    src = np.r_[np.arange(40), 40 + np.repeat(np.arange(20), 2), 60 + np.arange(40)]
    res = analyze(camera(np.ones(100))[src])
    sp = effects(res, "speed")
    assert len(sp) == 1 and sp[0].kind == "slowmo"
    assert_span(sp[0], 40, 80)
    assert sp[0].rate_from == pytest.approx(0.5, abs=0.05)
    assert sp[0].params["method"] == "frame_repeat" and sp[0].confidence >= 0.75
    assert not effects(res, "freeze")


def test_duplicate_cadence_overrides_overlapping_motion_estimates():
    """Motion-based events straddling a repeated-frame region give way to the cadence
    (the Sintel trailer's 0.5x shot came out as a motion ramp_down plus a stub)."""
    from editast.analyzers.timefx import _merge_duplicate_slowmo, _Params, _SpeedEvent

    prm = _Params(context_from_arrays(np.zeros((2, 8, 8, 3), np.uint8), FPS))
    novel = np.zeros(200, bool)
    novel[40:80:2] = True
    lead = _SpeedEvent("ramp_down", 20, 50, 1.0, 0.5, params={"method": "motion"})  # 20 of 30 frames outside
    ramp = _SpeedEvent("ramp_down", 44, 70, 1.0, 0.5, params={"method": "motion"})  # inside
    stub = _SpeedEvent("slowmo", 70, 84, 0.5, 0.5, params={"method": "motion"})  # mostly inside
    events = {0: [lead, ramp, stub]}
    _merge_duplicate_slowmo(events, [(0, 40, 80, 0.5)], novel, 1.0, prm)
    evs = events[0]
    assert [(ev.kind, ev.f0, ev.f1) for ev in evs] == [("ramp_down", 20, 40), ("slowmo", 40, 80)]
    assert evs[0].rate_to == pytest.approx(1.0 - 0.5 * 20 / 30)
    assert evs[1].params["method"] == "frame_repeat" and evs[1].rate_from == pytest.approx(0.5)


def test_uniform_frame_doubling_is_not_slow_motion():
    """30 fps footage in a 60 fps container repeats every frame everywhere."""
    frames = np.repeat(camera(np.ones(60), v=(8.0, 2.0)), 2, axis=0)
    assert not all_effects(analyze(frames))


def test_fast_shot_relative_to_video():
    shots = [camera(np.ones(45), seed=s, v=(2.0, 0.5)) for s in range(4)]
    shots.append(camera(np.ones(45), seed=9, v=(9.0, 2.0)))
    frames, seg = np.concatenate(shots), segmentation([45] * 5)
    # A shot moving faster than the others is no evidence of fast-forward by default ...
    assert not all_effects(analyze(frames, seg))
    # ... but the comparison across shots can be switched on.
    res = analyze(frames, seg, fast_relative_to_video=True)
    fast = effects(res, "speed", shot=4)
    assert len(fast) == 1 and fast[0].kind == "fast"
    assert fast[0].rate_from == pytest.approx(4.5, rel=0.35)
    assert 180 / FPS <= fast[0].start and fast[0].end <= 225 / FPS
    assert all(not res.by_shot.get(i) for i in range(4))


# --------------------------------------------------------------------------- freeze


def test_freeze_inside_moving_shot():
    frames = camera(np.ones(100))
    frames[50:58] = frames[50]  # frame 50 held for 8 frames
    res = analyze(frames)
    fz = effects(res, "freeze")
    assert len(fz) == 1
    assert_span(fz[0], 50, 58, tol=1)
    assert fz[0].params["frames"] == 8 and fz[0].confidence >= 0.7
    assert not effects(res, "speed")
    assert res.signals["frame_repeat"][51:58].all()


def test_freeze_with_pixel_effect_switching_on_during_the_hold():
    frames = camera(np.ones(100))
    frames[50:62] = frames[50]  # frame 50 held for 12 frames
    for i in range(55, 70):  # an RGB split switches on in the middle of the hold
        frames[i] = fx.rgb_split(frames[i], 0.02)
    fz = effects(analyze(frames), "freeze")
    assert len(fz) == 1
    assert_span(fz[0], 50, 62, tol=1)


def test_freeze_at_shot_end_and_start():
    a = camera(np.ones(60), seed=1)
    a[48:] = a[48]
    b = camera(np.ones(50), seed=2)
    b[:10] = b[0]
    res = analyze(np.concatenate([a, b]), segmentation([60, 50]))
    (fa,) = effects(res, "freeze", 0)
    (fb,) = effects(res, "freeze", 1)
    assert_span(fa, 48, 60, tol=1)
    assert_span(fb, 60, 70, tol=1)
    assert fa.end <= 60 / FPS + 1e-9 and fb.start >= 60 / FPS - 1e-9


def subject_clip(x) -> np.ndarray:
    """Static (tripod) background with a box at sub-pixel horizontal positions ``x``."""
    bg = synth.texture(H, W, 11)
    frames = np.repeat(bg[None], len(x), axis=0)
    for i, xi in enumerate(x):
        q = int(round(16 * xi))  # 4 fractional bits (cv2 ``shift``)
        corners = np.array([[q, 140 * 16], [q + 640, 140 * 16], [q + 640, 200 * 16], [q, 200 * 16]], np.int32)
        cv2.fillConvexPoly(frames[i], corners, (230, 40, 40), cv2.LINE_AA, 4)
        cv2.circle(frames[i], (q + 320, 170 * 16), 160, (20, 20, 200), -1, cv2.LINE_AA, 4)
    return frames


def test_static_shots_are_not_freezes():
    # Nothing moves at all (tripod shot).
    assert not all_effects(analyze(camera(np.zeros(90)), noise=1.0))
    # A subject that comes to rest naturally (decelerates), then stays still.
    steps = np.r_[np.linspace(3.0, 0.0, 40), np.zeros(50)]
    assert not all_effects(analyze(subject_clip(20 + np.cumsum(steps)), noise=1.0))
    # A subject that drives out of the frame, leaving the static background.
    for step in (4.0, 9.0):
        assert not all_effects(analyze(subject_clip(20 + step * np.arange(90)), noise=1.0))


def test_freeze_with_moving_subject_on_static_background():
    frames = subject_clip(20 + 1.5 * np.arange(90))
    frames[50:60] = frames[50]
    fz = effects(analyze(frames), "freeze")
    assert len(fz) == 1
    assert_span(fz[0], 50, 60, tol=1)


# --------------------------------------------------------------------------- robustness


def test_no_false_positives_on_continuous_shots():
    n = 90
    z = np.r_[np.ones(30), np.linspace(1.0, 1.35, 8), np.full(52, 1.35)]
    clips = [
        camera(np.ones(n)),  # constant pan
        camera(np.ones(n), v=(1.5, 0.0), shake=0.01, seed=3),  # handheld jitter on a slow pan
        camera(np.ones(n), zoom=z, seed=4),  # digital zoom punch
    ]
    for clip in clips:
        assert not all_effects(analyze(clip, noise=2.0))
    # Motion analysis reporting the shake widens the jitter filter; still nothing.
    shaky = camera(np.ones(n), v=(1.5, 0.0), shake=0.01, seed=3)
    zeros = np.zeros(n)
    motion = MotionAnalysis(FrameMotion(np.ones(n), zeros, zeros, zeros, zeros, np.ones(n, bool), zeros),
                            EffectsResult())
    motion.effects.add(0, Shake(start=0.0, end=n / FPS, amplitude=0.01, frequency=6.0))
    assert not all_effects(analyze(shaky, motion=motion))


def test_flash_and_cuts_do_not_read_as_speed_changes():
    a, b = camera(np.ones(60), seed=1), camera(np.ones(60), seed=2, v=(4.0, -1.0))
    for i, amount in zip(range(20, 25), (0.4, 0.9, 1.0, 0.7, 0.3)):
        a[i] = fx.flash(a[i], amount)
    # A flash transition across the cut covers frames 57..62.
    tr = TransitionSpan("flash", 60, 57, 63, 0, 1)
    for i in range(57, 63):
        frames_i = a if i < 60 else b
        frames_i[i % 60] = fx.flash(frames_i[i % 60], 1.0 - abs(i - 59.5) / 3.5)
    res = analyze(np.concatenate([a, b]), segmentation([60, 60], [tr]))
    assert not all_effects(res)


def test_signals_and_degenerate_inputs():
    rate = np.r_[np.ones(40), np.full(40, 0.25), np.ones(40)]
    res = analyze(camera(rate), segmentation([60, 60]))
    for key in ("speed_rate", "motion_magnitude", "frame_repeat"):
        assert res.signals[key].shape == (120,)
    for idx, effs in res.by_shot.items():
        lo, hi = (0.0, 2.0) if idx == 0 else (2.0, 4.0)
        assert all(lo - 1e-9 <= e.start < e.end <= hi + 1e-9 for e in effs)
    # One frame, or a still image: nothing to detect, but signals are well-formed.
    one = analyze(camera(np.ones(1)))
    assert not all_effects(one) and one.signals["speed_rate"].shape == (1,)
    still = analyze(np.repeat(camera(np.ones(1)), 30, axis=0))
    assert not all_effects(still) and np.all(still.signals["speed_rate"] == 1.0)


def test_h264_roundtrip(tmp_path):
    w, h = 360, 640
    size = (w, h)
    v = (10.0, 2.0)  # px per source frame at 360 px
    slow = camera(np.r_[np.ones(25), np.full(25, 0.25), np.ones(20)], seed=1, v=v, size=size)
    ramp = camera(np.r_[np.ones(20), eased(12, 1.0, 0.3, "ease_out"), np.full(18, 0.3),
                        eased(12, 0.3, 1.0, "ease_in"), np.ones(8)], seed=2, v=v, size=size)
    frozen = camera(np.ones(60), seed=3, v=v, size=size)
    frozen[30:38] = frozen[30]
    static = camera(np.zeros(40), seed=4, v=v, size=size)
    plain = camera(np.ones(50), seed=5, v=v, size=size)
    clips = [slow, ramp, frozen, static, plain]
    out = tmp_path / "timefx.mp4"
    with VideoWriter(out, w, h, FPS, crf=23) as vw:
        for frame in np.concatenate(clips):
            vw.write(frame)
    ctx = load_context(out, AnalysisConfig())
    assert ctx.size == (180, 320)
    lengths = [len(c) for c in clips]
    res = analyze_time_effects(ctx, segmentation(lengths), None)

    (s0,) = effects(res, "speed", 0)
    assert s0.kind == "slowmo" and s0.rate_from == pytest.approx(0.25, rel=0.35)
    assert_span(s0, 25, 50)
    kinds = [e.kind for e in effects(res, "speed", 1)]
    assert kinds == ["ramp_down", "slowmo", "ramp_up"]
    ramp_down = effects(res, "speed", 1)[0]
    assert_span(ramp_down, 70 + 20, 70 + 32)
    (fz,) = effects(res, "freeze", 2)
    assert_span(fz, 140 + 30, 140 + 38, tol=1)
    assert not effects(res, "speed", 2)
    assert not res.by_shot.get(3) and not res.by_shot.get(4)


def test_performance_one_minute_clip():
    """60 s at 30 fps, 180x320: one 10 s shot repeated (features tiled to save test time)."""
    base = context_from_arrays(camera(np.r_[np.ones(150), np.full(150, 0.3)], v=(3.0, 1.0)), FPS)
    reps = 6
    f = base.features
    tiled = type(f)(**{k: np.concatenate([getattr(f, k)] * reps) for k in vars(f)})
    ctx = replace(base, frames=np.concatenate([base.frames] * reps), gray=np.concatenate([base.gray] * reps),
                  features=tiled)
    t0 = time.perf_counter()
    res = analyze_time_effects(ctx, segmentation([300] * reps), None)
    assert time.perf_counter() - t0 < 15.0
    assert all(any(e.kind == "slowmo" for e in res.by_shot.get(i, [])) for i in range(reps))
