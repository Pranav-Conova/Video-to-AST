"""Turn the rendered frames in out/frames into the README GIF, ../docs/demo.gif.

12 fps (every other frame), one 256-colour palette for the whole clip and no
dithering, so the report's flat colours stay exact, then gifsicle's lossless
optimisation. Lossy GIF compression is left out: it streaks the report's dark
backgrounds.

    npm run frames && python make_gif.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
FPS = 24  # the composition's frame rate, which is the Sintel trailer's
GIF_FPS = 12
OUT = HERE.parent / "docs" / "demo.gif"


def main() -> None:
    frames = sorted((HERE / "out" / "frames").glob("element-*.png"))
    if not frames:
        sys.exit("no frames in out/frames; run `npm run frames` first")
    digits = len(frames[0].stem.split("-")[-1])
    pattern = HERE / "out" / "frames" / f"element-%0{digits}d.png"
    raw = HERE / "out" / "raw.gif"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(FPS), "-i", str(pattern),
        "-vf", f"fps={GIF_FPS},split[a][b];[a]palettegen=stats_mode=full[p];"
               "[b][p]paletteuse=dither=none:diff_mode=rectangle",
        "-loop", "0", str(raw),
    ], check=True)
    gifsicle = HERE / "node_modules" / "gifsicle" / "vendor" / ("gifsicle.exe" if sys.platform == "win32" else "gifsicle")
    OUT.parent.mkdir(exist_ok=True)
    subprocess.run([str(gifsicle), "-O3", str(raw), "-o", str(OUT)], check=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
