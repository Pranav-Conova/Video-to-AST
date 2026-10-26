"""Analyse the Sintel trailer and put the video, its Edit AST and the HTML report in public/.

The demo composition plays the video next to the report and the compact
EditScript, so it needs all of them.
Sintel (c) Blender Foundation, CC-BY 3.0, durian.blender.org.

    python prepare.py
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from editast.lang.printer import to_dsl
from editast.pipeline import analyze, signals_for_report
from editast.report.html import write_report

URL = "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4"
CACHE = Path.home() / ".cache" / "editast" / "test-media" / "sintel_trailer-480p.mp4"  # shared with the tests
PUBLIC = Path(__file__).resolve().parent / "public"


def main() -> None:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(CACHE)
    PUBLIC.mkdir(exist_ok=True)
    video = PUBLIC / "sintel_trailer.mp4"
    shutil.copyfile(CACHE, video)

    result = analyze(video, progress=print)
    result.edit.media.source = URL  # the report header shows this instead of a local path
    result.edit.save(PUBLIC / "sintel_trailer.edit.json")
    (PUBLIC / "sintel_trailer.compact.edit").write_text(to_dsl(result.edit, compact=True), encoding="utf-8")
    write_report(result.edit, PUBLIC / "sintel_trailer.report.html", ctx=result.ctx,
                 signals=signals_for_report(result))
    print(f"wrote the video, Edit AST, EditScript and HTML report to {PUBLIC}")


if __name__ == "__main__":
    main()
