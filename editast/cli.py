"""Command-line interface: ``editast <command> ...`` (also ``python -m editast``).

Commands
--------
``analyze SOURCE``      video file or URL -> ``<stem>.edit.json``, ``<stem>.edit`` (EditScript)
                        and ``<stem>.report.html``; prints the tree and a summary.
``show FILE``           print a saved edit (``.json`` or ``.edit``) as tree / dsl / compact / json / summary.
``render TEMPLATE``     replay an edit over your own clips (optionally retimed to new music).
``bench``               synthetic benchmark: generate edits with a known AST, analyse, score.
``compare A B``         structural similarity of two edits, or a full score with ``--truth``.
``schema``              JSON schema of the Edit AST.
``fetch URL``           download a video with yt-dlp.
``batch INPUTS...``     analyse many videos and summarise what their edits have in common.

Optional parts (EditScript, reports, renderer, benchmark, analyzers) are
imported inside the commands that need them, so one broken module only
disables the commands or outputs that depend on it, with a clear message.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from editast import __version__
from editast.ast.nodes import Edit
from editast.context import AnalysisConfig

if TYPE_CHECKING:
    from editast.pipeline import AnalysisResult

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"})
# bench table column -> path into an evaluate.compare() report (F1 / accuracy scores)
BENCH_METRICS: dict[str, tuple[str, ...]] = {
    "overall": ("overall",),
    "cuts": ("cuts", "f1"),
    "trans": ("transitions", "accuracy"),
    "effects": ("effects", "f1"),
    "beats": ("music", "beat_f"),
    "text": ("text", "f1"),
    "b&w": ("look", "bw_accuracy"),
}


class CliError(Exception):
    """A user-facing error: printed without a traceback, exit code 1."""


# =========================================================================== helpers


def _require(module: str, purpose: str) -> ModuleType:
    """Import an optional module or explain which feature is unavailable."""
    try:
        return importlib.import_module(module)
    except Exception as exc:
        raise CliError(f"{purpose} is unavailable: could not import {module} ({type(exc).__name__}: {exc})") from exc


def _stem(path: Path) -> str:
    """``clip.mp4`` -> ``clip``; ``clip.edit.json`` -> ``clip``."""
    name = path.name
    for suffix in (".edit.json", ".truth.json", ".json", ".edit"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or name
    return path.stem


def _load_edit(path_like: str) -> Edit:
    path = Path(path_like)
    if not path.is_file():
        raise CliError(f"no such file: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        try:
            return Edit.from_json(text)
        except ValueError as exc:
            raise CliError(f"{path} is not a valid Edit JSON file: {exc}") from exc
    parser = _require("editast.lang.parser", "reading EditScript (.edit) files")
    return parser.parse_dsl(text)


def _collect_videos(paths: Sequence[str]) -> list[Path]:
    """Video files from file and folder arguments (folders are not searched recursively)."""
    out: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS)
        elif p.is_file():
            out.append(p)
        else:
            raise CliError(f"no such file or folder: {p}")
    return out


def _parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, e.g. 1080x1920, got {value!r}") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return w, h


def _config(args: argparse.Namespace) -> AnalysisConfig:
    cfg = AnalysisConfig(analysis_long_side=args.long_side, target_fps=args.fps, max_duration=args.max_duration,
                         ocr=not args.no_ocr)
    cfg.enabled.update({"text": not args.no_text, "audio": not args.no_audio})
    return cfg


def _metric(report: dict[str, Any], path: tuple[str, ...]) -> Optional[float]:
    value: Any = report
    for key in path:
        if not isinstance(value, dict) or value.get(key) is None:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) else None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")
    return path


class _Ui:
    """Console output: results on stdout, progress / warnings / errors on stderr."""

    def __init__(self, quiet: bool = False) -> None:
        self.out = Console(highlight=False)
        self.err = Console(stderr=True, highlight=False)
        self.quiet = quiet

    def info(self, message: str) -> None:
        if not self.quiet:
            self.err.print(Text(message, style="dim"))

    def warn(self, message: str) -> None:
        self.err.print(Text.assemble(("warning: ", "bold yellow"), message))

    def progress(self, label: str) -> Callable[[str], None]:
        return lambda msg: self.info(f"  {label}: {msg}")


# =========================================================================== commands


def cmd_analyze(args: argparse.Namespace, ui: _Ui) -> int:
    from editast.pipeline import analyze

    result = analyze(args.source, _config(args), progress=None if args.quiet else ui.progress("analyze"))
    stem = _stem(Path(result.ctx.path))
    out_dir = Path(args.output) if args.output else Path("out") / stem
    written = _write_outputs(result, out_dir, stem, ui)
    for warning in result.warnings:
        ui.warn(warning)
    edit = result.edit
    if not args.quiet:
        try:
            tree = importlib.import_module("editast.report.tree")
            tree.print_tree(edit, console=ui.out)
            ui.out.print(Text(tree.summary_text(edit)), soft_wrap=True)
        except Exception as exc:
            ui.warn(f"tree view unavailable (editast.report.tree: {type(exc).__name__}: {exc})")
            ui.out.print(Text(_fallback_summary(edit)), soft_wrap=True)
        timings = ", ".join(f"{k} {v:.1f}s" for k, v in result.timings.items())
        ui.info(f"timings: {timings}")
    for path in written:
        ui.out.print(Text(f"wrote {path}"))
    return 0


def _write_outputs(result: "AnalysisResult", out_dir: Path, stem: str, ui: _Ui) -> list[Path]:
    """``<stem>.edit.json`` always; EditScript and HTML report when those modules work."""
    from editast.pipeline import signals_for_report

    edit = result.edit
    written = [edit.save(out_dir / f"{stem}.edit.json")]
    dsl_path = out_dir / f"{stem}.edit"
    try:
        printer = _require("editast.lang.printer", "EditScript output")
        dsl_path.write_text(printer.to_dsl(edit), encoding="utf-8")
        written.append(dsl_path)
    except Exception as exc:
        ui.warn(f"skipped {dsl_path.name}: {_reason(exc)}")
    html_path = out_dir / f"{stem}.report.html"
    try:
        html = _require("editast.report.html", "the HTML report")
        written.append(Path(html.write_report(edit, html_path, ctx=result.ctx, signals=signals_for_report(result))))
    except Exception as exc:
        ui.warn(f"skipped {html_path.name}: {_reason(exc)}")
    return written


def _reason(exc: Exception) -> str:
    return str(exc) if isinstance(exc, CliError) else f"{type(exc).__name__}: {exc}"


def _fallback_summary(edit: Edit) -> str:
    """Minimal plain-text summary used when editast.report.tree is unavailable."""
    st = edit.stats
    lines = [f"{edit.media.source}: {edit.media.duration:.1f} s, {len(edit.shots())} shots"]
    if edit.music is not None and edit.music.bpm:
        lines.append(f"music: {edit.music.bpm:.1f} BPM, {len(edit.music.drops)} drop(s)")
    if st is not None:
        lines.append(f"pace: {st.cuts_per_minute:.0f} cuts/min, mean shot {st.mean_shot_duration:.2f} s")
    lines.append("sections: " + " > ".join(f"{s.role} ({s.start:.1f}-{s.end:.1f}s)" for s in edit.timeline.sections))
    lines += [f"pattern: {p.description}" for p in edit.patterns]
    return "\n".join(lines)


def cmd_show(args: argparse.Namespace, ui: _Ui) -> int:
    edit = _load_edit(args.file)
    fmt = args.format
    if fmt == "json":
        sys.stdout.write(edit.to_json() + "\n")
    elif fmt in ("dsl", "compact"):
        printer = _require("editast.lang.printer", "EditScript output")
        text = printer.to_dsl(edit, compact=fmt == "compact")
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    elif fmt == "summary":
        tree = _require("editast.report.tree", "the summary view")
        ui.out.print(Text(tree.summary_text(edit)), soft_wrap=True)
    else:
        tree = _require("editast.report.tree", "the tree view")
        tree.print_tree(edit, console=ui.out)
    return 0


def cmd_render(args: argparse.Namespace, ui: _Ui) -> int:
    compiler = _require("editast.render.compiler", "rendering")
    edit = _load_edit(args.template)
    clips = _collect_videos(args.clips)
    if not clips:
        raise CliError("no video clips found in --clips")
    music = Path(args.music) if args.music else None
    if music is not None and not music.is_file():
        raise CliError(f"no such music file: {music}")
    if args.retime:
        if music is None:
            raise CliError("--retime needs --music")
        bpm, offset = _music_grid(music)
        old = edit.music.bpm if edit.music is not None else None
        ui.info(f"retiming {f'{old:.1f}' if old else 'the edit'} -> {bpm:.1f} BPM (first beat {offset:.2f} s)")
        edit = compiler.retime_edit(edit, bpm, offset)
    out = Path(args.output) if args.output else Path("out") / f"{_stem(Path(args.template))}.render.mp4"
    path = compiler.render_edit(edit, clips, out, music=music, size=args.size, fps=args.fps,
                                progress=ui.progress("render"))
    ui.out.print(Text(f"wrote {path}"))
    return 0


def _music_grid(music: Path) -> tuple[float, float]:
    """``(bpm, first beat seconds)`` of a song, via the audio analyzer."""
    from editast.context import context_from_arrays
    from editast.media import read_audio

    audio_mod = _require("editast.analyzers.audio", "--retime (tempo detection)")
    sr = AnalysisConfig().audio_sr
    samples = read_audio(music, sr=sr)
    if samples.size == 0:
        raise CliError(f"{music} has no audio")
    # A one-frame dummy video: the audio analyzer then uses the whole song as its timeline.
    ctx = context_from_arrays(np.zeros((1, 16, 16, 3), np.uint8), fps=30.0, audio=samples, sr=sr, source=str(music))
    music_info = audio_mod.analyze_audio(ctx).music
    if music_info is None or not music_info.bpm:
        raise CliError(f"could not detect a tempo in {music}")
    offset = music_info.beat_offset
    if offset is None:
        offset = music_info.beats[0] if music_info.beats else 0.0
    return float(music_info.bpm), float(offset)


def cmd_bench(args: argparse.Namespace, ui: _Ui) -> int:
    from editast.pipeline import analyze

    generate = _require("editast.bench.generate", "the benchmark")
    evaluate = _require("editast.bench.evaluate", "the benchmark")
    out = Path(args.output)
    ui.info(f"generating {args.n} synthetic case(s) in {out / 'cases'}")
    cases = generate.generate_suite(out / "cases", n=args.n, seed=args.seed)
    config = _config(args)
    rows: list[dict[str, Any]] = []
    for i, (mp4, truth) in enumerate(cases, 1):
        ui.info(f"[{i}/{len(cases)}] analysing {mp4.name}")
        t0 = time.perf_counter()
        result = analyze(mp4, config, progress=None if args.quiet else ui.progress(mp4.stem))
        seconds = time.perf_counter() - t0
        result.edit.save(out / "pred" / f"{mp4.stem}.edit.json")
        report = evaluate.compare(truth, result.edit)
        if not args.quiet:
            ui.out.print(Text(f"\n{mp4.stem}\n{evaluate.format_report(report)}"), soft_wrap=True)
        rows.append({"case": mp4.stem, "video": str(mp4), "seconds": round(seconds, 3),
                     "warnings": result.warnings, "metrics": {k: _metric(report, p) for k, p in BENCH_METRICS.items()},
                     "report": report})
        del result
    mean = {}
    for key in BENCH_METRICS:
        values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        mean[key] = round(float(np.mean(values)), 4) if values else None

    table = Table(title=f"editast bench (n={len(rows)}, seed={args.seed})", box=box.SIMPLE_HEAD,
                  caption="cuts / effects / beats / text: F1; trans / b&w: accuracy")
    table.add_column("case", no_wrap=True)
    for key in BENCH_METRICS:
        table.add_column(key, justify="right")
    table.add_column("time", justify="right")

    def cell(v: Optional[float]) -> str:
        return "-" if v is None else f"{v:.3f}"

    for r in rows:
        table.add_row(r["case"], *(cell(r["metrics"][k]) for k in BENCH_METRICS), f"{r['seconds']:.1f}s")
    table.add_row("mean", *(cell(mean[k]) for k in BENCH_METRICS), "", style="bold")
    ui.out.print(table)
    path = _write_json(out / "bench_results.json", {"n": len(rows), "seed": args.seed, "mean": mean, "cases": rows})
    ui.out.print(Text(f"wrote {path}"))
    return 0


def cmd_compare(args: argparse.Namespace, ui: _Ui) -> int:
    evaluate = _require("editast.bench.evaluate", "comparing edits")
    a, b = _load_edit(args.a), _load_edit(args.b)
    if args.truth:
        ui.out.print(Text(evaluate.format_report(evaluate.compare(a, b))), soft_wrap=True)
    else:
        ui.out.print(Text(f"similarity: {evaluate.similarity(a, b):.3f}"))
    return 0


def cmd_schema(args: argparse.Namespace, ui: _Ui) -> int:
    text = json.dumps(Edit.json_schema(), indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        ui.out.print(Text(f"wrote {path}"))
    else:
        sys.stdout.write(text + "\n")
    return 0


def cmd_fetch(args: argparse.Namespace, ui: _Ui) -> int:
    from editast.fetch import download, is_url

    if not is_url(args.url):
        raise CliError(f"not a URL: {args.url}")
    ui.info(f"downloading {args.url}")
    ui.out.print(Text(str(download(args.url, args.output))))
    return 0


def cmd_batch(args: argparse.Namespace, ui: _Ui) -> int:
    from editast.fetch import is_url
    from editast.pipeline import analyze, summarize_edits

    sources: list[str] = []
    for item in args.inputs:
        sources += [item] if is_url(item) else [str(p) for p in _collect_videos([item])]
    if not sources:
        raise CliError("no videos to analyse")
    out = Path(args.output)
    config = _config(args)
    printer: Optional[ModuleType] = None
    try:
        printer = importlib.import_module("editast.lang.printer")
    except Exception as exc:
        ui.warn(f"EditScript output disabled (editast.lang.printer: {type(exc).__name__}: {exc})")
    edits: list[tuple[str, Edit]] = []
    failed: list[dict[str, str]] = []
    used: set[str] = set()
    for i, src in enumerate(sources, 1):
        ui.info(f"[{i}/{len(sources)}] {src}")
        try:
            result = analyze(src, config, progress=None if args.quiet else ui.progress("analyze"))
        except Exception as exc:
            failed.append({"source": src, "error": f"{type(exc).__name__}: {exc}"})
            ui.warn(f"{src}: {type(exc).__name__}: {exc}")
            continue
        stem = base = _stem(Path(result.ctx.path))
        k = 2
        while stem in used:
            stem, k = f"{base}_{k}", k + 1
        used.add(stem)
        edit = result.edit
        edit.save(out / f"{stem}.edit.json")
        if printer is not None:
            try:
                (out / f"{stem}.edit").write_text(printer.to_dsl(edit), encoding="utf-8")
            except Exception as exc:
                ui.warn(f"{stem}.edit skipped: {type(exc).__name__}: {exc}")
        for warning in result.warnings:
            ui.warn(f"{stem}: {warning}")
        edits.append((src, edit))
        del result  # frames are large; keep only the edit
    summary = summarize_edits(edits)
    summary["failed"] = failed
    path = _write_json(out / "batch_summary.json", summary)

    table = Table(title=f"patterns across {len(edits)} video(s)", box=box.SIMPLE_HEAD)
    table.add_column("pattern")
    table.add_column("videos", justify="right")
    table.add_column("share", justify="right")
    table.add_column("meaning")
    for key, info in summary["patterns"].items():
        table.add_row(escape(key), str(info["videos"]), f"{100 * info['fraction']:.0f}%", escape(info["phrase"]))
    ui.out.print(table)
    for line in summary["common"]:
        ui.out.print(Text(f"- {line}"), soft_wrap=True)
    ui.out.print(Text(f"wrote {path}"))
    return 0 if edits else 1


# =========================================================================== parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="editast",
        description="Reverse-engineer viral short-form video edits into an Abstract Syntax Tree, and render new ones.",
    )
    parser.add_argument("--version", action="version", version=f"editast {__version__}")
    parser.add_argument("--debug", action="store_true", help="show tracebacks and debug logging")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    analysis = argparse.ArgumentParser(add_help=False)
    defaults = AnalysisConfig()
    analysis.add_argument("--no-text", action="store_true", help="skip text overlay detection")
    analysis.add_argument("--no-ocr", action="store_true", help="detect text boxes without reading them")
    analysis.add_argument("--no-audio", action="store_true", help="skip music analysis")
    analysis.add_argument("--long-side", type=int, default=defaults.analysis_long_side, metavar="N",
                          help=f"analysis resolution, long side in px (default {defaults.analysis_long_side})")
    analysis.add_argument("--fps", type=float, default=None, metavar="F", help="analysis frame rate (default: source)")
    analysis.add_argument("--max-duration", type=float, default=defaults.max_duration, metavar="S",
                          help=f"analyse at most S seconds (default {defaults.max_duration:g})")
    analysis.add_argument("--quiet", "-q", action="store_true", help="only print results and warnings")

    p = sub.add_parser("analyze", parents=[analysis], help="analyse a video file or URL into an edit AST")
    p.add_argument("source", help="video file or URL")
    p.add_argument("-o", "--output", metavar="OUT_DIR", help="output folder (default ./out/<stem>)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("show", help="print a saved edit (.json or .edit)")
    p.add_argument("file")
    p.add_argument("--format", "-f", choices=("tree", "dsl", "compact", "json", "summary"), default="tree")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("render", help="render an edit template with your own clips")
    p.add_argument("template", help="edit file (.json or .edit)")
    p.add_argument("--clips", nargs="+", required=True, metavar="PATH", help="video files and/or folders")
    p.add_argument("--music", metavar="FILE", help="audio track to mux into the output")
    p.add_argument("--retime", action="store_true", help="move the edit onto the tempo grid of --music")
    p.add_argument("-o", "--output", metavar="OUT.mp4", help="output video (default ./out/<template>.render.mp4)")
    p.add_argument("--size", type=_parse_size, metavar="WxH", help="output size, e.g. 1080x1920")
    p.add_argument("--fps", type=float, metavar="F", help="output frame rate")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("bench", parents=[analysis], help="score the analyzers on synthetic edits with a known AST")
    p.add_argument("--n", type=int, default=5, help="number of cases (default 5)")
    p.add_argument("--seed", type=int, default=0, help="first seed (default 0)")
    p.add_argument("-o", "--output", default=str(Path("out") / "bench"), metavar="DIR",
                   help="output folder (default ./out/bench)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("compare", help="similarity of two edits, or a score against a truth edit")
    p.add_argument("a", metavar="A")
    p.add_argument("b", metavar="B")
    p.add_argument("--truth", action="store_true", help="treat A as the ground truth and print the full report")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("schema", help="print the Edit JSON schema")
    p.add_argument("-o", "--output", metavar="FILE")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("fetch", help="download a video from a URL (needs yt-dlp)")
    p.add_argument("url")
    p.add_argument("-o", "--output", default=".", metavar="DIR", help="download folder (default .)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("batch", parents=[analysis], help="analyse many videos and summarise their common patterns")
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="video files, folders or URLs")
    p.add_argument("-o", "--output", required=True, metavar="DIR")
    p.set_defaults(func=cmd_batch)
    return parser


def _configure_streams() -> None:
    """Never crash on characters the console encoding cannot show (OCR text, emoji).

    Redirected output (files, pipes) is written as UTF-8: on Windows it would
    otherwise use the legacy code page, which has no arrows or multiplication sign.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of the ``editast`` command; returns the process exit code."""
    args = build_parser().parse_args(argv)
    _configure_streams()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    ui = _Ui(quiet=getattr(args, "quiet", False))
    try:
        return int(args.func(args, ui) or 0)
    except KeyboardInterrupt:
        ui.err.print("interrupted")
        return 130
    except CliError as exc:
        ui.err.print(Text.assemble(("error: ", "bold red"), str(exc)))
        return 1
    except Exception as exc:
        if args.debug:
            raise
        ui.err.print(Text.assemble(("error: ", "bold red"), f"{type(exc).__name__}: {exc}"))
        ui.err.print(Text("(run with --debug for the traceback)", style="dim"))
        return 1
