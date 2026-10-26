"""Analyse a video and put it, its Edit AST, compact EditScript and HTML report in public/.

The demo composition plays the video next to the report and the EditScript, so it
needs all of them. See README.md for how docs/demo.gif was made from the Dune trailer.

    python prepare.py dune_trailer.mp4 --url https://www.youtube.com/watch?v=n9xhJrPXop4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from editast.fetch import download, is_url
from editast.lang.printer import to_dsl
from editast.pipeline import analyze, signals_for_report
from editast.report.html import write_report

PUBLIC = Path(__file__).resolve().parent / "public"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="video file or URL")
    ap.add_argument("--name", help="file name stem in public/ (default: the file's)")
    ap.add_argument("--url", help="where the video is from, shown in the report header")
    args = ap.parse_args()

    src = download(args.source) if is_url(args.source) else Path(args.source)
    name = args.name or src.stem
    PUBLIC.mkdir(exist_ok=True)
    video = PUBLIC / f"{name}{src.suffix}"
    shutil.copyfile(src, video)

    result = analyze(video, progress=print)
    # The report header shows this instead of a local path.
    result.edit.media.source = args.url or (args.source if is_url(args.source) else video.name)
    result.edit.save(PUBLIC / f"{name}.edit.json")
    (PUBLIC / f"{name}.compact.edit").write_text(to_dsl(result.edit, compact=True), encoding="utf-8")
    write_report(result.edit, PUBLIC / f"{name}.report.html", ctx=result.ctx,
                 signals=signals_for_report(result))
    print(f"wrote {video.name} and its Edit AST, EditScript and HTML report to {PUBLIC}")


if __name__ == "__main__":
    main()
