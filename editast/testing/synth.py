"""Synthetic media with known ground truth, for unit tests and the benchmark.

* :func:`texture` — a colourful, feature-rich image whose palette depends on ``seed``
  (different seeds give clearly different colour histograms, like different shots).
* :func:`scene` — ``n`` frames of a virtual camera moving/zooming over a texture.
* :func:`cut_sequence` — several scenes concatenated with hard cuts.
* :func:`click_track` / :func:`beat_times` — a drum-machine track at a known BPM,
  optionally with a "drop" where energy and density jump.
"""

from __future__ import annotations

import cv2
import numpy as np

GOLDEN = 0.6180339887


def texture(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Feature-rich RGB texture (multi-scale noise + sharp shapes)."""
    rng = np.random.default_rng(seed)
    base_hue = (seed * GOLDEN * 180.0) % 180.0
    field = np.zeros((height, width), np.float32)
    for sigma, weight in ((width / 6, 1.0), (width / 25, 0.6), (width / 90, 0.3)):
        n = rng.standard_normal((height, width)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), max(sigma, 0.5))
        field += weight * n / (n.std() + 1e-6)
    sat_field = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), width / 10)
    sat_field /= sat_field.std() + 1e-6
    hsv = np.empty((height, width, 3), np.float32)
    hsv[..., 0] = (base_hue + 18.0 * np.tanh(0.6 * field)) % 180.0
    hsv[..., 1] = 170.0 + 60.0 * np.tanh(0.5 * sat_field)
    hsv[..., 2] = 135.0 + 85.0 * np.tanh(0.5 * field)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    # Sharp shapes give the motion tracker corners to lock onto.
    n_shapes = max(12, (height * width) // 2500)
    for _ in range(n_shapes):
        hue = (base_hue + rng.uniform(-30, 30)) % 180
        val = rng.choice([40, 230])
        color = cv2.cvtColor(np.uint8([[[hue, rng.uniform(120, 255), val]]]), cv2.COLOR_HSV2RGB)[0, 0]
        color = tuple(int(c) for c in color)
        x, y = int(rng.uniform(0, width)), int(rng.uniform(0, height))
        size = int(rng.uniform(0.01, 0.06) * min(height, width)) + 2
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + size, y + int(size * rng.uniform(0.5, 2))), color, -1)
        else:
            cv2.circle(img, (x, y), size // 2 + 1, color, -1, cv2.LINE_AA)
    return img


def scene(
    n: int,
    height: int,
    width: int,
    seed: int = 0,
    velocity: tuple[float, float] = (0.0, 0.0),
    zoom_rate: float = 1.0,
    margin: float = 1.6,
) -> np.ndarray:
    """``n`` frames of a camera over a large texture.

    ``velocity`` is the camera motion in output px/frame (content moves the
    opposite way); ``zoom_rate`` is the per-frame magnification factor.
    """
    ch, cw = int(height * margin), int(width * margin)
    canvas = texture(ch, cw, seed)
    frames = np.empty((n, height, width, 3), np.uint8)
    cx0, cy0 = cw / 2.0, ch / 2.0
    s = 1.0
    for i in range(n):
        cx = cx0 + velocity[0] * i
        cy = cy0 + velocity[1] * i
        m = np.array([[s, 0.0, width / 2.0 - s * cx], [0.0, s, height / 2.0 - s * cy]], np.float32)
        frames[i] = cv2.warpAffine(canvas, m, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
        s *= zoom_rate
    return frames


def cut_sequence(
    shot_lengths: list[int],
    height: int = 180,
    width: int = 320,
    seed: int = 0,
    velocity: tuple[float, float] = (1.0, 0.0),
) -> tuple[np.ndarray, list[int]]:
    """Concatenate scenes with hard cuts. Returns ``(frames, boundaries)`` where
    ``boundaries`` are the first frame index of every shot after the first."""
    parts, bounds, pos = [], [], 0
    for k, length in enumerate(shot_lengths):
        parts.append(scene(length, height, width, seed=seed + 101 * k + 1, velocity=velocity))
        if k:
            bounds.append(pos)
        pos += length
    return np.concatenate(parts), bounds


# --------------------------------------------------------------------------- audio


def beat_times(bpm: float, duration: float, offset: float = 0.0) -> np.ndarray:
    period = 60.0 / bpm
    return np.arange(offset, duration, period)


def _kick(sr: int, amp: float) -> np.ndarray:
    t = np.arange(int(0.18 * sr)) / sr
    freq = 50.0 + 90.0 * np.exp(-t * 30.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return (amp * np.sin(phase) * np.exp(-t * 18.0)).astype(np.float32)


def _noise_hit(sr: int, amp: float, length: float, rng: np.random.Generator, hp: bool = True) -> np.ndarray:
    t = np.arange(int(length * sr)) / sr
    noise = rng.standard_normal(t.size).astype(np.float32)
    if hp:
        noise = np.diff(noise, prepend=0.0).astype(np.float32)
    return (amp * noise * np.exp(-t / (length / 4))).astype(np.float32)


def click_track(
    bpm: float,
    duration: float,
    sr: int = 22050,
    offset: float = 0.0,
    drop_at: float | None = None,
    meter: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Kick on every beat (accented downbeats), hats on off-beats; after
    ``drop_at`` the mix gets louder and adds snare + an 8th-note bass line."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(duration * sr) + sr, np.float32)
    period = 60.0 / bpm

    def place(sig: np.ndarray, t: float) -> None:
        i = int(round(t * sr))
        if 0 <= i < out.size:
            j = min(out.size, i + sig.size)
            out[i:j] += sig[: j - i]

    for k, bt in enumerate(beat_times(bpm, duration, offset)):
        after = drop_at is not None and bt >= drop_at
        gain = 1.0 if (drop_at is None or after) else 0.45
        place(_kick(sr, gain * (1.0 if k % meter == 0 else 0.75)), bt)
        place(_noise_hit(sr, 0.12 * gain, 0.04, rng), bt + period / 2)
        if after:
            if k % 2 == 1:
                place(_noise_hit(sr, 0.35, 0.12, rng, hp=False), bt)
            for half in (0.0, 0.5):
                t0 = bt + half * period
                n = int(0.45 * period * sr)
                tt = np.arange(n) / sr
                saw = 2.0 * ((55.0 * tt) % 1.0) - 1.0
                place((0.18 * saw * np.exp(-tt * 6.0)).astype(np.float32), t0)
    out = out[: int(duration * sr)]
    peak = float(np.abs(out).max()) or 1.0
    return (0.9 * out / peak).astype(np.float32)
