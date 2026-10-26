"""Download a video from a URL (TikTok, Instagram Reels, YouTube Shorts, ...) with yt-dlp.

yt-dlp is an optional dependency (``pip install editast[fetch]``) and is
imported only when a download is requested. Videos are fetched as MP4 with
audio, preferring at most 1080p (plenty for analysis at 320 px), and merged
with the bundled ffmpeg (:func:`editast.media.ffmpeg_exe`). A file that was
downloaded before is reused instead of fetched again.

YouTube needs a JavaScript runtime to solve its signature challenges, and
yt-dlp enables only Deno by default, so every runtime found on the PATH (Deno,
Node.js, Bun) is enabled; yt-dlp runs them sandboxed. YouTube also sometimes
refuses yt-dlp's default clients ("Sign in to confirm you're not a bot", or
storyboard images only), so a failed YouTube download is retried once with the
embedded-player client, which is usually still served.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

from editast.media import ffmpeg_exe

DOWNLOAD_DIR = Path.home() / ".cache" / "editast" / "downloads"

# Best MP4 video <= 1080p + M4A audio, then progressively looser fallbacks.
FORMAT = (
    "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/"
    "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"
)
_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
YOUTUBE_FALLBACK_CLIENTS = ["web_embedded"]
_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
_JS_RUNTIMES = ("deno", "node", "bun")


class FetchError(RuntimeError):
    """A URL could not be downloaded (or yt-dlp is not installed)."""


class _Silent:
    """yt-dlp logger that prints nothing: failures arrive as exceptions, and a first
    YouTube attempt that the retry recovers from should not print an error."""

    def debug(self, msg: str) -> None:
        pass

    info = warning = error = debug


def is_url(source: Union[str, Path]) -> bool:
    """True for ``http(s)://...``-style sources (local paths, including ``C:\\...``, are not URLs)."""
    return isinstance(source, str) and bool(_URL.match(source.strip()))


def is_youtube(url: str) -> bool:
    """True for youtube.com (any subdomain), youtu.be and youtube-nocookie.com URLs."""
    host = (urlparse(url.strip()).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in _YOUTUBE_HOSTS)


def download(url: str, out_dir: Union[str, Path] = DOWNLOAD_DIR) -> Path:
    """Download ``url`` into ``out_dir`` and return the path of the MP4 file.

    A failed YouTube download is retried with the embedded-player client (see the module docstring).
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise FetchError("downloading URLs needs yt-dlp: pip install yt-dlp (or pip install editast[fetch])") from exc
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(extractor)s_%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _Silent(),
        "ffmpeg_location": ffmpeg_exe(),
    }
    runtimes = {name: {} for name in _JS_RUNTIMES if shutil.which(name)}
    if runtimes:
        options["js_runtimes"] = runtimes
    try:
        return _fetch(yt_dlp, url, options)
    except FetchError as exc:
        if not is_youtube(url):
            raise
        first = exc
    retry = {**options, "extractor_args": {"youtube": {"player_client": YOUTUBE_FALLBACK_CLIENTS}}}
    try:
        return _fetch(yt_dlp, url, retry)
    except FetchError as exc:
        message = f"{first}; retrying with YouTube's embedded-player client failed too: {exc.__cause__ or exc}"
        if not runtimes:
            message += " (yt-dlp needs a JavaScript runtime for YouTube: install Deno or Node.js)"
        raise FetchError(message) from exc


def _fetch(yt_dlp: Any, url: str, options: dict[str, Any]) -> Path:
    """One yt-dlp download with ``options``; any failure becomes a :class:`FetchError`."""
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise FetchError(f"yt-dlp returned no video for {url}")
            if info.get("entries"):  # a playlist / profile page: take the first video
                info = next(e for e in info["entries"] if e)
            path = _downloaded_path(ydl, info)
    except FetchError:
        raise
    except Exception as exc:  # yt-dlp raises DownloadError / ExtractorError and plain OSErrors
        raise FetchError(f"could not download {url}: {exc}") from exc
    if not path.exists():
        raise FetchError(f"yt-dlp finished but {path} does not exist")
    return path


def _downloaded_path(ydl: Any, info: dict[str, Any]) -> Path:
    """Final file of a download (after merging), following yt-dlp's bookkeeping."""
    for item in info.get("requested_downloads") or []:
        if item.get("filepath"):
            return Path(item["filepath"])
    if info.get("filepath"):
        return Path(info["filepath"])
    return Path(ydl.prepare_filename(info)).with_suffix(".mp4")
