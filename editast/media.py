"""Media I/O through ffmpeg (bundled via imageio-ffmpeg, so no system install is needed).

* :func:`probe` — container facts parsed from ``ffmpeg -i`` (no ffprobe required).
* :func:`read_video` / :func:`iter_video` — decode to RGB ``uint8`` arrays at a fixed
  (CFR) rate and size, so frame ``i`` is at ``i / fps`` seconds.
* :func:`read_audio` — decode to mono ``float32`` in ``[-1, 1]``.
* :class:`VideoWriter` — encode RGB frames to H.264 MP4, optionally muxing an audio file.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


class MediaError(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    """Path to ffmpeg: ``$EDITAST_FFMPEG``, then PATH, then the imageio-ffmpeg binary."""
    env = os.environ.get("EDITAST_FFMPEG")
    if env:
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------- probe


@dataclass
class Probe:
    path: str
    duration: float
    fps: float
    width: int  # display size, rotation already applied (ffmpeg autorotates on decode)
    height: int
    rotation: int
    has_video: bool
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""

    @property
    def n_frames_estimate(self) -> int:
        return int(round(self.duration * self.fps))


_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO = re.compile(r"Stream #\d+:\d+.*?: Video: (\w+)(.*)")
_SIZE = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")
_FPS = re.compile(r"(\d+(?:\.\d+)?)(k?) fps")
_TBR = re.compile(r"(\d+(?:\.\d+)?)(k?) tbr")
_AUDIO = re.compile(r"Stream #\d+:\d+.*?: Audio: (\w+)")
_ROT_TAG = re.compile(r"rotate\s*:\s*(-?\d+)")
_ROT_MATRIX = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")


def _num(value: str, kilo: str) -> float:
    return float(value) * (1000.0 if kilo else 1.0)


def probe(path: str | Path) -> Probe:
    path = str(path)
    if not Path(path).exists():
        raise MediaError(f"file not found: {path}")
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    info = proc.stderr
    duration = 0.0
    m = _DUR.search(info)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    width = height = 0
    fps = 0.0
    vcodec = ""
    vm = _VIDEO.search(info)
    if vm:
        vcodec = vm.group(1)
        rest = vm.group(2)
        sm = _SIZE.search(rest)
        if sm:
            width, height = int(sm.group(1)), int(sm.group(2))
        fm = _FPS.search(rest) or _TBR.search(rest)
        if fm:
            fps = _num(fm.group(1), fm.group(2))
    rotation = 0
    rm = _ROT_TAG.search(info)
    if rm:
        rotation = int(rm.group(1))
    else:
        rm = _ROT_MATRIX.search(info)
        if rm:
            rotation = int(round(float(rm.group(1))))
    if abs(rotation) % 180 == 90:
        width, height = height, width
    am = _AUDIO.search(info)
    if not vm and not am:
        raise MediaError(f"ffmpeg could not read {path}:\n{info[-2000:]}")
    if fps <= 0 or fps > 1000:
        fps = 30.0
    return Probe(
        path=path, duration=duration, fps=fps, width=width, height=height,
        rotation=rotation, has_video=vm is not None, has_audio=am is not None,
        video_codec=vcodec, audio_codec=am.group(1) if am else "",
    )


def aspect_string(width: int, height: int) -> str:
    """Nearest common aspect name (``9:16``, ``16:9`` ...) or the reduced ratio."""
    if width <= 0 or height <= 0:
        return ""
    common = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:5": 4 / 5, "5:4": 5 / 4,
              "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9, "2:3": 2 / 3, "3:2": 3 / 2}
    ratio = width / height
    name, value = min(common.items(), key=lambda kv: abs(math.log(kv[1] / ratio)))
    if abs(math.log(value / ratio)) < 0.02:
        return name
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def fit_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Scale ``(width, height)`` so the long side is ``long_side`` (never upscale); even dims."""
    if width <= 0 or height <= 0:
        raise MediaError("unknown frame size")
    scale = min(1.0, long_side / max(width, height))
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    return w, h


# --------------------------------------------------------------------------- decode


def iter_video(
    path: str | Path,
    fps: Optional[float] = None,
    size: Optional[tuple[int, int]] = None,
    start: float = 0.0,
    duration: Optional[float] = None,
    gray: bool = False,
    chunk_frames: int = 64,
) -> Iterator[np.ndarray]:
    """Yield frames ``(h, w, 3)`` RGB (or ``(h, w)`` gray) ``uint8``.

    ``fps`` resamples to a constant rate; ``size`` is ``(width, height)``.
    """
    path = str(path)
    if size is None:
        p = probe(path)
        size = (p.width, p.height)
    w, h = size
    channels = 1 if gray else 3
    args = [ffmpeg_exe(), "-v", "error", "-nostdin"]
    if start > 0:
        args += ["-ss", f"{start:.4f}"]
    args += ["-i", path]
    if duration is not None:
        args += ["-t", f"{duration:.4f}"]
    filters = []
    if fps:
        filters.append(f"fps={fps:.6g}")
    filters.append(f"scale={w}:{h}:flags=area")
    args += ["-vf", ",".join(filters), "-an", "-sn", "-f", "rawvideo",
             "-pix_fmt", "gray" if gray else "rgb24", "-"]
    frame_bytes = w * h * channels
    stderr = tempfile.TemporaryFile()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=stderr)
    exhausted = False
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes * chunk_frames)
            if not buf:
                exhausted = True
                break
            n = len(buf) // frame_bytes
            arr = np.frombuffer(buf[: n * frame_bytes], np.uint8)
            arr = arr.reshape((n, h, w) if gray else (n, h, w, 3))
            for frame in arr:
                yield frame
    finally:
        if proc.stdout:
            proc.stdout.close()
        if not exhausted and proc.poll() is None:
            proc.kill()  # consumer stopped early; not an error
        code = proc.wait()
        stderr.seek(0)
        err = stderr.read().decode("utf-8", "replace")
        stderr.close()
        if exhausted and code != 0:
            raise MediaError(f"ffmpeg decode failed ({code}): {err[-1500:]}")


def read_video(
    path: str | Path,
    fps: Optional[float] = None,
    size: Optional[tuple[int, int]] = None,
    start: float = 0.0,
    duration: Optional[float] = None,
    gray: bool = False,
) -> np.ndarray:
    """Decode a whole clip to ``(n, h, w, 3)`` RGB (or ``(n, h, w)`` gray) ``uint8``."""
    frames = list(iter_video(path, fps=fps, size=size, start=start, duration=duration, gray=gray))
    if not frames:
        raise MediaError(f"no frames decoded from {path}")
    return np.stack(frames)


def read_frame(path: str | Path, t: float, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Decode the single frame shown at time ``t`` seconds."""
    for frame in iter_video(path, size=size, start=max(0.0, t), duration=None, chunk_frames=1):
        return frame.copy()
    raise MediaError(f"no frame at t={t:.3f}s in {path}")


def read_audio(path: str | Path, sr: int = 22050) -> np.ndarray:
    """Decode the first audio stream to mono ``float32``. Empty array if there is no audio."""
    path = str(path)
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-nostdin", "-i", path, "-vn", "-sn",
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")
        if "does not contain any stream" in err or "matches no streams" in err:
            return np.zeros(0, np.float32)
        raise MediaError(f"ffmpeg audio decode failed: {err[-1500:]}")
    return np.frombuffer(proc.stdout, np.float32).copy()


# --------------------------------------------------------------------------- encode


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> Path:
    """Write mono float audio in ``[-1, 1]`` as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(samples, np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def mux_audio(video_path: str | Path, audio_path: str | Path, out_path: str | Path,
              duration: Optional[float] = None) -> Path:
    """Copy the video stream and add ``audio_path`` as AAC, trimmed to ``duration``."""
    args = [ffmpeg_exe(), "-y", "-v", "error", "-nostdin", "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if duration is not None:
        args += ["-t", f"{duration:.4f}"]
    else:
        args += ["-shortest"]
    args += ["-movflags", "+faststart", str(out_path)]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg mux failed: {proc.stderr.decode('utf-8', 'replace')[-1500:]}")
    return Path(out_path)


class VideoWriter:
    """Pipe RGB ``uint8`` frames into an H.264 MP4.

    >>> with VideoWriter("out.mp4", 1080, 1920, 30, audio_path="song.wav") as vw:
    ...     vw.write(frame)
    """

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        audio_path: str | Path | None = None,
        crf: int = 18,
        preset: str = "veryfast",
    ) -> None:
        if width % 2 or height % 2:
            raise MediaError("width and height must be even for yuv420p")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height, self.fps = width, height, fps
        self.audio_path = Path(audio_path) if audio_path else None
        self.frames_written = 0
        self._tmp_dir = tempfile.mkdtemp(prefix="editast_") if self.audio_path else None
        self._video_target = Path(self._tmp_dir) / "video.mp4" if self._tmp_dir else self.path
        self._stderr = tempfile.TemporaryFile()
        args = [ffmpeg_exe(), "-y", "-v", "error", "-nostdin",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", f"{fps:.6g}", "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", preset,
                "-movflags", "+faststart", str(self._video_target)]
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=self._stderr)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise MediaError(f"expected ({self.height}, {self.width}, 3) uint8, got {frame.shape} {frame.dtype}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames_written += 1

    def close(self) -> Path:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        code = self._proc.wait()
        self._stderr.seek(0)
        err = self._stderr.read().decode("utf-8", "replace")
        self._stderr.close()
        if code != 0:
            raise MediaError(f"ffmpeg encode failed ({code}): {err[-1500:]}")
        if self.audio_path is not None:
            mux_audio(self._video_target, self.audio_path, self.path,
                      duration=self.frames_written / self.fps)
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        return self.path

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            finally:
                self._proc.kill()
                self._proc.wait()
                if not self._stderr.closed:
                    self._stderr.close()
                if self._tmp_dir:
                    shutil.rmtree(self._tmp_dir, ignore_errors=True)
