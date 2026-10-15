"""Frame-level effect primitives and easing curves.

Shared by the renderer (``editast.render``), the synthetic benchmark and the
analyzer tests, so that "an ease_out zoom from 1.0 to 1.3" means exactly the
same thing when it is rendered and when it is recovered.

All functions take/return RGB ``uint8`` frames ``(h, w, 3)`` and never modify
their input in place.
"""

from __future__ import annotations

from typing import Callable, Union

import cv2
import numpy as np

ArrayOrFloat = Union[float, np.ndarray]

# --------------------------------------------------------------------------- easing


def _linear(x: ArrayOrFloat) -> ArrayOrFloat:
    return x


def _ease_in(x: ArrayOrFloat) -> ArrayOrFloat:
    return x * x


def _ease_out(x: ArrayOrFloat) -> ArrayOrFloat:
    return 1.0 - (1.0 - x) ** 2


def _ease_in_out(x: ArrayOrFloat) -> ArrayOrFloat:
    return x * x * (3.0 - 2.0 * x)


def _punch(x: ArrayOrFloat) -> ArrayOrFloat:
    """Snap to the target within the first 20% of the span, then hold."""
    u = np.minimum(1.0, np.asarray(x, dtype=float) / 0.2)
    out = 1.0 - (1.0 - u) ** 3
    return float(out) if np.ndim(out) == 0 else out


def _step(x: ArrayOrFloat) -> ArrayOrFloat:
    out = (np.asarray(x, dtype=float) > 0).astype(float)
    return float(out) if np.ndim(out) == 0 else out


EASINGS: dict[str, Callable[[ArrayOrFloat], ArrayOrFloat]] = {
    "linear": _linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
    "punch": _punch,
    "step": _step,
}


def ease(x: ArrayOrFloat, easing: str = "linear") -> ArrayOrFloat:
    """Map progress ``x`` in [0, 1] through the named easing curve."""
    x = np.clip(x, 0.0, 1.0)
    out = EASINGS[easing](x)
    return float(out) if np.ndim(out) == 0 else out


def interp(a: float, b: float, x: ArrayOrFloat, easing: str = "linear") -> ArrayOrFloat:
    """Eased interpolation from ``a`` to ``b`` at progress ``x``."""
    return a + (b - a) * ease(x, easing)


# --------------------------------------------------------------------------- geometry


def similarity_matrix(
    width: int,
    height: int,
    scale: float = 1.0,
    angle: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    center: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """2x3 matrix: scale + rotate (deg, CCW) about ``center`` (normalised), then
    translate content by ``(tx * width, ty * height)``."""
    m = cv2.getRotationMatrix2D((center[0] * width, center[1] * height), angle, scale)
    m[0, 2] += tx * width
    m[1, 2] += ty * height
    return m


def transform(
    frame: np.ndarray,
    scale: float = 1.0,
    angle: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    center: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """Apply a similarity transform to the content (borders reflect)."""
    if scale == 1.0 and angle == 0.0 and tx == 0.0 and ty == 0.0:
        return frame.copy()
    h, w = frame.shape[:2]
    m = similarity_matrix(w, h, scale, angle, tx, ty, center)
    return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def shake_offset(t: float, amplitude: float, frequency: float, seed: int = 0) -> tuple[float, float]:
    """Deterministic jitter (per-axis peak ``amplitude``) at time ``t`` seconds."""
    rng = np.random.default_rng(seed)
    px, py = rng.uniform(0, 2 * np.pi, 2)
    w = 2 * np.pi * frequency
    dx = amplitude * np.sin(w * t + px)
    dy = amplitude * np.sin(w * 1.13 * t + py)
    return float(dx), float(dy)


# --------------------------------------------------------------------------- colour / pixel


def solid(height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    out = np.empty((height, width, 3), np.uint8)
    out[:] = np.asarray(color, np.uint8)
    return out


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """``(1 - alpha) * a + alpha * b`` (dissolve)."""
    if alpha <= 0.0:
        return a.copy()
    if alpha >= 1.0:
        return b.copy()
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)


def flash(frame: np.ndarray, amount: float, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Blend toward a solid colour; ``amount=1`` gives the solid colour."""
    return blend(frame, solid(frame.shape[0], frame.shape[1], color), float(np.clip(amount, 0.0, 1.0)))


def grade(
    frame: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
) -> np.ndarray:
    """Simple colour grade. ``brightness`` is an offset in 0..1 units,
    ``temperature`` > 0 warms (more red, less blue)."""
    f = frame.astype(np.float32) / 255.0
    luma = f @ np.array([0.299, 0.587, 0.114], np.float32)
    f = luma[..., None] + (f - luma[..., None]) * saturation
    f = (f - 0.5) * contrast + 0.5 + brightness
    if temperature:
        f[..., 0] += 0.15 * temperature
        f[..., 2] -= 0.15 * temperature
    return (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_bw(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def blur(frame: np.ndarray, amount: float) -> np.ndarray:
    """Gaussian blur with sigma ``amount * width``."""
    sigma = amount * frame.shape[1]
    if sigma <= 0.05:
        return frame.copy()
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def invert(frame: np.ndarray) -> np.ndarray:
    return 255 - frame


def rgb_split(frame: np.ndarray, offset: float) -> np.ndarray:
    """Shift red right and blue left by ``offset * width`` px (chromatic aberration)."""
    px = int(round(offset * frame.shape[1]))
    if px == 0:
        return frame.copy()
    out = frame.copy()
    out[..., 0] = np.roll(frame[..., 0], px, axis=1)
    out[..., 2] = np.roll(frame[..., 2], -px, axis=1)
    return out


def vignette(frame: np.ndarray, strength: float) -> np.ndarray:
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    mask = 1.0 - strength * np.clip(r, 0, 1) ** 2
    return (frame.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)


def letterbox(frame: np.ndarray, top: float, bottom: float, color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Cover the top/bottom fractions of the frame with flat bars."""
    out = frame.copy()
    h = frame.shape[0]
    t, b = int(round(top * h)), int(round(bottom * h))
    if t:
        out[:t] = color
    if b:
        out[h - b:] = color
    return out


# --------------------------------------------------------------------------- text


def draw_text(
    frame: np.ndarray,
    text: str,
    center: tuple[float, float] = (0.5, 0.5),
    size: float = 0.05,
    color: tuple[int, int, int] = (255, 255, 255),
    outline: bool = True,
    max_width: float = 0.9,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Draw bold caption-style text; ``size`` = cap height as a fraction of frame height.

    Returns ``(frame, bbox)`` with the normalised ``(x, y, w, h)`` box of the text.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    (_, base_h), _ = cv2.getTextSize("H", font, 1.0, 1)
    scale = max(0.2, size * h / max(base_h, 1))
    thickness = max(1, int(round(scale * 2.2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    if tw > max_width * w:
        scale *= max_width * w / tw
        thickness = max(1, int(round(scale * 2.2)))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(round(center[0] * w - tw / 2))
    y = int(round(center[1] * h + th / 2))
    out = frame.copy()
    if outline:
        cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), thickness + max(2, thickness), cv2.LINE_AA)
    cv2.putText(out, text, (x, y), font, scale, tuple(int(c) for c in color), thickness, cv2.LINE_AA)
    pad = thickness
    bx, by = max(0, x - pad), max(0, y - th - pad)
    bw, bh = min(w, x + tw + pad) - bx, min(h, y + baseline + pad) - by
    return out, (bx / w, by / h, bw / w, bh / h)
