"""Turn rendered frames into a README GIF (by default out/frames into ../docs/demo.gif).

12 fps (every other frame), one 256-colour palette for the whole clip and no
dithering, so the report's flat colours stay exact, then gifsicle's lossless
optimisation. Lossy GIF compression is left out: it streaks the report's dark
backgrounds.

    npm run frames && python make_gif.py
    npm run frames:car && python make_gif.py out/car-frames ../docs/car-edit-demo.gif
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
FPS = 24  # the compositions' frame rate (src/Root.tsx)
GIF_FPS = 12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("frames", nargs="?", type=Path, default=HERE / "out" / "frames")
    ap.add_argument("gif", nargs="?", type=Path, default=HERE.parent / "docs" / "demo.gif")
    args = ap.parse_args()
    frames = sorted(args.frames.glob("element-*.png"))
    if not frames:
        sys.exit(f"no frames in {args.frames}; run `npm run frames` first")
    digits = len(frames[0].stem.split("-")[-1])
    pattern = args.frames / f"element-%0{digits}d.png"
    raw = HERE / "out" / "raw.gif"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(FPS), "-i", str(pattern),
        "-vf", f"fps={GIF_FPS},split[a][b];[a]palettegen=stats_mode=full[p];"
               "[b][p]paletteuse=dither=none:diff_mode=rectangle",
        "-loop", "0", str(raw),
    ], check=True)
    gifsicle = HERE / "node_modules" / "gifsicle" / "vendor" / ("gifsicle.exe" if sys.platform == "win32" else "gifsicle")
    out = args.gif.resolve()
    out.parent.mkdir(exist_ok=True)
    subprocess.run([str(gifsicle), "-O3", str(raw), "-o", str(out)], check=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
