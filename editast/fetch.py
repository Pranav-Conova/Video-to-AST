"""Download a video from a URL (TikTok, Instagram Reels, YouTube Shorts, ...) with yt-dlp.

yt-dlp is an optional dependency (``pip install editast[fetch]``) and is
imported only when a download is requested. Videos are fetched as MP4 with
audio, preferring at most 1080p (plenty for analysis at 320 px), and merged
with the bundled ffmpeg (:func:`editast.media.ffmpeg_exe`). A file that was
downloaded before is reused instead of fetched again.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union

from editast.media import ffmpeg_exe

DOWNLOAD_DIR = Path.home() / ".cache" / "editast" / "downloads"

# Best MP4 video <= 1080p + M4A audio, then progressively looser fallbacks.
FORMAT = (
    "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/"
    "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"
)
_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


class FetchError(RuntimeError):
    """A URL could not be downloaded (or yt-dlp is not installed)."""


def is_url(source: Union[str, Path]) -> bool:
    """True for ``http(s)://...``-style sources (local paths, including ``C:\\...``, are not URLs)."""
    return isinstance(source, str) and bool(_URL.match(source.strip()))


def download(url: str, out_dir: Union[str, Path] = DOWNLOAD_DIR) -> Path:
    """Download ``url`` into ``out_dir`` and return the path of the MP4 file."""
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
        "ffmpeg_location": ffmpeg_exe(),
    }
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
