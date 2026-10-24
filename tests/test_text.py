"""Tests for the burned-in text overlay analyzer.

Captions are drawn with :func:`editast.fx.draw_text` (the renderer's text
primitive) over moving :func:`editast.testing.synth.scene` footage, so the
ground truth (frames, bbox, words) is exact. Frames are 360x640 portrait at
30 fps, a resolution where OCR reads the captions reliably.
"""

import difflib
import time

import cv2
import numpy as np
import pytest

from editast import fx
from editast.analyzers.text import detect_text
from editast.analyzers.types import TextAnalysis
from editast.ast.nodes import TextOverlay
from editast.context import AnalysisConfig, context_from_arrays, load_context
from editast.media import VideoWriter
from editast.testing import synth

H, W, FPS = 640, 360, 30.0
BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------- helpers


def scene(n: int, seed: int, velocity: tuple[float, float] = (2.0, 1.0), height: int = H,
          width: int = W) -> np.ndarray:
    return synth.scene(n, height, width, seed=seed, velocity=velocity)


def caption(frames: np.ndarray, text: str, f0: int, f1: int, center: tuple[float, float] = (0.5, 0.5),
            size: float = 0.05, **kw) -> BBox:
    """Burn ``text`` into ``frames[f0:f1]`` (in place); returns its normalised bbox."""
    box: BBox = (0.0, 0.0, 0.0, 0.0)
    for i in range(f0, f1):
        frames[i], box = fx.draw_text(frames[i], text, center, size, **kw)
    return box


def detect(frames: np.ndarray, ocr: bool = True, **params) -> TextAnalysis:
    config = AnalysisConfig(ocr=ocr, params={"text": params} if params else {})
    return detect_text(context_from_arrays(frames, FPS, config=config), None)


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def similar(a: str | None, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").upper(), b.upper()).ratio()


def frames_of(ov: TextOverlay) -> tuple[float, float]:
    return ov.start * FPS, ov.end * FPS


def check_contract(res: TextAnalysis, n: int) -> None:
    present = res.signals["text_present"]
    assert present.shape == (n,) and set(np.unique(present)) <= {0.0, 1.0}
    for ov in res.overlays:
        assert 0.0 <= ov.start < ov.end <= n / FPS + 1e-9
        x, y, w, h = ov.bbox
        assert 0.0 <= x and 0.0 <= y and w > 0 and h > 0 and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9
        assert 0.0 < ov.size < 0.5 and 0.0 <= ov.confidence <= 1.0
        assert present[int(round(ov.start * FPS)):int(round(ov.end * FPS))].all()
        for wd in ov.words:
            assert ov.start - 1e-6 <= wd.start < wd.end <= ov.end + 1e-6


@pytest.fixture(scope="module")
def wait_clip() -> tuple[np.ndarray, BBox]:
    """(a) "WAIT FOR IT" on frames 10..40 of a 60-frame moving scene, upper third."""
    frames = scene(60, seed=3)
    box = caption(frames, "WAIT FOR IT", 10, 41, center=(0.5, 0.25))
    return frames, box


@pytest.fixture(scope="module")
def word_clip() -> tuple[np.ndarray, list[str]]:
    """(c) Word-by-word caption, 0.4 s per word, in the lower third."""
    frames = scene(90, seed=5, velocity=(-2.0, 1.5))
    words = ["THIS", "CAR", "IS", "INSANE"]
    for k, word in enumerate(words):
        caption(frames, word, 15 + 12 * k, 27 + 12 * k, center=(0.5, 0.72), size=0.06, color=(255, 230, 0))
    return frames, words


# --------------------------------------------------------------------------- (a) / (b) single caption


def test_single_caption_timing_box_and_position(wait_clip):
    frames, truth = wait_clip
    res = detect(frames)
    check_contract(res, len(frames))
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    start, end = frames_of(ov)
    assert abs(start - 10) <= 2 and abs(end - 41) <= 2
    assert iou(ov.bbox, truth) > 0.5
    assert ov.position == "upper"
    assert ov.animation == "static"
    assert ov.role == "hook"  # on screen within the first 2 s, upper third
    assert ov.size == pytest.approx(0.05 * 1.4, rel=0.35)  # line height ~ cap height * 1.4
    assert ov.color is not None and min(ov.color) >= 200  # white fill
    present = res.signals["text_present"]
    assert present[15:36].all() and not present[:8].any() and not present[43:].any()


def test_ocr_reads_caption(wait_clip):
    pytest.importorskip("rapidocr_onnxruntime")
    frames, _ = wait_clip
    res = detect(frames)
    assert res.ocr_engine == "rapidocr"
    assert len(res.overlays) == 1
    assert similar(res.overlays[0].text, "WAIT FOR IT") > 0.7


# --------------------------------------------------------------------------- (c) word by word


def test_word_by_word_caption_is_merged(word_clip):
    pytest.importorskip("rapidocr_onnxruntime")
    frames, words = word_clip
    res = detect(frames)
    check_contract(res, len(frames))
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    assert ov.animation == "word_by_word"
    assert len(ov.words) == 4
    assert ov.position == "lower" and ov.role == "caption"
    start, end = frames_of(ov)
    assert abs(start - 15) <= 2 and abs(end - 63) <= 2
    for k, (wd, truth) in enumerate(zip(ov.words, words)):
        assert similar(wd.text, truth) > 0.7
        assert abs(wd.start * FPS - (15 + 12 * k)) <= 2
    assert similar(ov.text, " ".join(words)) > 0.7
    assert ov.color is not None and ov.color[0] > 200 and ov.color[1] > 180 and ov.color[2] < 80  # yellow


def test_word_by_word_without_ocr(word_clip):
    frames, _ = word_clip
    res = detect(frames, ocr=False)
    assert res.ocr_engine is None
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    assert ov.animation == "word_by_word" and len(ov.words) == 4


# --------------------------------------------------------------------------- (d) negatives


def test_plain_moving_scene_has_no_overlays():
    frames = scene(90, seed=7, velocity=(2.5, -1.0))
    for ocr in (True, False):
        res = detect(frames, ocr=ocr)
        check_contract(res, len(frames))
        assert res.overlays == []
        assert not res.signals["text_present"].any()


def test_scene_text_moving_with_camera_is_ignored():
    """Text painted onto the footage (a sign) moves with the camera: not an overlay."""
    ch, cw = int(H * 1.6), int(W * 1.6)
    canvas = synth.texture(ch, cw, seed=11)
    canvas, _ = fx.draw_text(canvas, "SPEED", (0.5, 0.5), 0.05)
    frames = np.stack([
        cv2.warpAffine(canvas, np.float32([[1, 0, W / 2 - (cw / 2 + 2.5 * i)], [0, 1, H / 2 - ch / 2]]),
                       (W, H), borderMode=cv2.BORDER_REFLECT_101)
        for i in range(60)
    ])
    for ocr in (True, False):
        assert detect(frames, ocr=ocr).overlays == []


# --------------------------------------------------------------------------- (e) classical fallback


def test_detection_without_ocr(wait_clip):
    frames, truth = wait_clip
    res = detect(frames, ocr=False)
    check_contract(res, len(frames))
    assert res.ocr_engine is None
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    assert ov.text is None
    start, end = frames_of(ov)
    assert abs(start - 10) <= 2 and abs(end - 41) <= 2
    assert iou(ov.bbox, truth) > 0.5 and ov.position == "upper"


# --------------------------------------------------------------------------- animation / role


def test_pop_in_caption():
    frames = scene(60, seed=9, velocity=(1.5, 2.0))
    for i, size in zip(range(12, 15), (0.025, 0.035, 0.045)):  # scales up over 3 frames
        caption(frames, "BOOM", i, i + 1, center=(0.5, 0.5), size=size)
    caption(frames, "BOOM", 15, 45, center=(0.5, 0.5), size=0.055)
    res = detect(frames)
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    assert ov.animation == "pop"
    start, end = frames_of(ov)
    assert abs(start - 12) <= 2 and abs(end - 45) <= 2
    assert ov.position == "center"


def test_watermark_in_corner():
    frames = scene(150, seed=13, velocity=(2.0, -1.0))
    caption(frames, "@carclips", 0, 150, center=(0.8, 0.94), size=0.022)
    res = detect(frames, ocr=False)
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    assert ov.role == "watermark" and ov.position == "bottom"
    assert ov.start <= 2 / FPS and ov.end >= 148 / FPS


def test_cut_under_caption_keeps_one_overlay():
    """The caption stays on screen through a hard cut: one overlay, not two."""
    frames, _ = synth.cut_sequence([40, 40], height=H, width=W, seed=2, velocity=(2.0, 1.0))
    caption(frames, "NO WAY", 20, 60, center=(0.5, 0.5), size=0.05)
    res = detect(frames)
    assert len(res.overlays) == 1
    start, end = frames_of(res.overlays[0])
    assert abs(start - 20) <= 2 and abs(end - 60) <= 2


# --------------------------------------------------------------------------- (f) real encode round trip


def test_h264_round_trip_uses_source_file(tmp_path):
    frames = scene(75, seed=17, velocity=(3.0, 1.0))
    truth = caption(frames, "FULL SEND", 20, 55, center=(0.5, 0.7), size=0.045)
    path = tmp_path / "caption.mp4"
    with VideoWriter(path, W, H, FPS, crf=28) as vw:
        for f in frames:
            vw.write(f)
    ctx = load_context(path)  # analysis frames are 180x320; OCR samples are decoded at 360x640
    assert ctx.size == (180, 320)
    res = detect_text(ctx, None)
    check_contract(res, ctx.n_frames)
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    start, end = frames_of(ov)
    assert abs(start - 20) <= 2 and abs(end - 55) <= 2
    assert iou(ov.bbox, truth) > 0.5 and ov.position == "lower"
    if res.ocr_engine is not None:
        assert similar(ov.text, "FULL SEND") > 0.7


def test_noisy_frames_without_ocr(wait_clip):
    frames, truth = wait_clip
    rng = np.random.default_rng(0)
    noisy = np.clip(frames.astype(np.int16) + rng.normal(0, 8, frames.shape).round().astype(np.int16), 0, 255)
    res = detect(noisy.astype(np.uint8), ocr=False)
    assert len(res.overlays) == 1
    start, end = frames_of(res.overlays[0])
    assert abs(start - 10) <= 2 and abs(end - 41) <= 2
    assert iou(res.overlays[0].bbox, truth) > 0.5


# --------------------------------------------------------------------------- performance


def test_performance_one_minute_clip():
    """60 s at 30 fps, analysis size 180x320, with captions changing every 2 s."""
    n = 1800
    frames = synth.scene(n, 320, 180, seed=21, velocity=(1.5, 0.5))
    for k in range(0, n // 60, 2):
        caption(frames, f"PART {k}", 60 * k, 60 * k + 60, center=(0.5, 0.75), size=0.05)
    ctx = context_from_arrays(frames, FPS, config=AnalysisConfig())
    t0 = time.perf_counter()
    res = detect_text(ctx, None)
    elapsed = time.perf_counter() - t0
    check_contract(res, n)
    assert elapsed < 30.0, f"text analysis took {elapsed:.1f}s"
    assert 12 <= len(res.overlays) <= 18


# --------------------------------------------------------------------------- OCR economy, spacing, blocks, roles


def count_reads(monkeypatch) -> list[int]:
    """Patch the OCR wrapper to count detector runs; returns the (mutable) counter."""
    from editast.analyzers import text as T

    calls = [0]
    orig = T._Ocr.read

    def read(self, rgb, keep=None):
        calls[0] += 1
        return orig(self, rgb, keep)

    monkeypatch.setattr(T._Ocr, "read", read)
    return calls


def black(n: int) -> np.ndarray:
    return np.zeros((n, H, W, 3), np.uint8)


def test_respace_restores_dropped_word_spaces():
    """The recognizer sometimes drops word spaces ("COMINGSOON"); they come back from
    the gaps between the characters, but evenly tracked lettering gets none."""
    from editast.analyzers.text import _respace

    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2

    def render(text: str, tracking: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
        crop = np.zeros((48, 40 + 60 * len(text), 3), np.uint8)
        x, chars, xs = 10, [], []
        for ch in text:
            (cw, _), _ = cv2.getTextSize(ch, font, scale, th)
            if ch != " ":
                cv2.putText(crop, ch, (x, 38), font, scale, (255, 255, 255), th, cv2.LINE_AA)
                chars.append(ch)
                xs.append(x + cw / 2)
            x += cw + tracking
        return crop, chars, np.array(xs)

    crop, chars, xs = render("COMING SOON")
    assert "".join(chars) == "COMINGSOON"
    assert _respace(chars, xs, crop) == "COMING SOON"
    crop, chars, xs = render("HEARD IT FIRST")
    assert _respace(chars, xs, crop) == "HEARD IT FIRST"
    crop, chars, xs = render("SINTEL", tracking=22)  # widely tracked title lettering
    assert _respace(chars, xs, crop) == "SINTEL"
    crop, chars, xs = render("INSANE")
    assert _respace(chars, xs, crop) == "INSANE"


def test_blank_frames_are_not_ocrd(monkeypatch):
    """Black frames hold nothing to read; a title card on black is read once, not at
    every sample, and is a title (centred, alone on screen, after the hook)."""
    pytest.importorskip("rapidocr_onnxruntime")
    calls = count_reads(monkeypatch)
    frames = black(150)
    caption(frames, "THE END", 75, 140, center=(0.5, 0.5), size=0.035, outline=False)
    res = detect(frames)
    check_contract(res, len(frames))
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    assert similar(ov.text, "THE END") > 0.7
    start, end = frames_of(ov)
    assert abs(start - 75) <= 2 and abs(end - 140) <= 2
    assert ov.role == "title"
    assert calls[0] <= 4, f"{calls[0]} OCR runs"


def test_title_card_fading_in_is_read_once(monkeypatch):
    """A slowly fading title over still footage changes every pixel but not its text:
    it is read about once, not at every sample."""
    pytest.importorskip("rapidocr_onnxruntime")
    calls = count_reads(monkeypatch)
    base = scene(1, seed=23, velocity=(0.0, 0.0))[0]
    titled = base.copy()
    titled, _ = fx.draw_text(titled, "SUNSET", (0.5, 0.4), 0.06)
    alpha = np.clip((np.arange(150) - 30) / 60.0, 0.0, 1.0)  # fades in over 2 s from frame 30
    frames = np.stack([np.round(base * (1 - a) + titled * a).astype(np.uint8) for a in alpha])
    res = detect(frames)
    assert len(res.overlays) == 1
    assert similar(res.overlays[0].text, "SUNSET") > 0.7
    assert calls[0] <= 7, f"{calls[0]} OCR runs"


def test_word_replaced_in_place_over_still_footage():
    """Over still footage stillness says nothing, so reads follow changes: a caption
    word replaced by another in the same place must still be read."""
    pytest.importorskip("rapidocr_onnxruntime")
    frames = scene(90, seed=29, velocity=(0.0, 0.0))
    words = ["THIS", "CAR", "IS", "INSANE"]
    for k, word in enumerate(words):
        caption(frames, word, 15 + 12 * k, 27 + 12 * k, center=(0.5, 0.72), size=0.06)
    res = detect(frames)
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    assert ov.animation == "word_by_word" and len(ov.words) == 4
    for wd, truth in zip(ov.words, words):
        assert similar(wd.text, truth) > 0.7


def test_stacked_lines_sharing_timing_form_one_block():
    """Credit lines that appear and disappear together are one overlay: lines joined
    by newlines in reading order, bbox their union, a label (small text)."""
    pytest.importorskip("rapidocr_onnxruntime")
    frames = black(150)
    lines = ["DIRECTED BY", "JANE DOE", "MUSIC BY KO"]
    boxes = [caption(frames, line, 75, 140, center=(0.5, 0.42 + 0.06 * k), size=0.025, outline=False)
             for k, line in enumerate(lines)]
    res = detect(frames)
    check_contract(res, len(frames))
    assert len(res.overlays) == 1
    ov = res.overlays[0]
    got = (ov.text or "").split("\n")
    assert len(got) == 3 and all(similar(g, t) > 0.7 for g, t in zip(got, lines))
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    union = (x0, y0, max(b[0] + b[2] for b in boxes) - x0, max(b[1] + b[3] for b in boxes) - y0)
    assert iou(ov.bbox, union) > 0.5
    assert ov.role == "label"
