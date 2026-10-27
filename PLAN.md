# Video Reverse Engineer (`editast`): plan

Reverse-engineer viral short-form edits (TikTok / Reels / Shorts, beat-synced
"velocity" car edits, caption-driven talking heads) into an **Abstract Syntax
Tree** you can read, diff, aggregate across many videos, and *compile back*
into a new video with your own footage and music.

## Pipeline

```
 video / URL ──► ingest ──► AnalysisContext (frames @320px CFR, audio, per-frame features)
                               │
     ┌───────────┬─────────────┼──────────────┬──────────────┬──────────────┐
     ▼           ▼             ▼              ▼              ▼              ▼
   shots      motion        timefx          look           text           audio
 cuts &     zoom/pan/     speed ramps,    grade, B&W,     captions,      BPM, beats,
 transitions rotate/shake  freeze frames   flashes, blur,  OCR, word-by-  downbeats,
                                           invert, RGB     word timing    drops, energy
                                           split, layout
     └───────────┴─────────────┴──────┬───────┴──────────────┴──────────────┘
                                      ▼
                            build (structure synthesis)
               sections (hook/build/drop/outro), beat alignment,
               source-clip grouping, inferred patterns, stats
                                      ▼
                                  Edit AST ──► JSON · EditScript DSL · HTML report · tree view
                                      │
                                      ▼
                     render (compiler): AST + your clips + music ──► new MP4
```

Verification loop: `bench` generates synthetic edits with a **known AST**, renders
them through the compiler, analyzes the result and scores recovered vs. true AST.

## AST (see `editast/ast/nodes.py`)

`Edit → Timeline → Section* → (Shot | Transition)*`, `Shot.effects: Effect*`,
plus `Music`, `TextOverlay*`, `Pattern*`, `EditStats`. Units and conventions are
in the module docstring of `nodes.py` and are binding for all modules.

## Module ownership & contracts

| Module | File(s) | Public API |
|---|---|---|
| foundation (done) | `ast/nodes.py`, `media.py`, `context.py`, `fx.py`, `analyzers/types.py`, `testing/synth.py` | see files |
| shots | `analyzers/shots.py` | `detect_shots(ctx) -> ShotSegmentation` |
| motion | `analyzers/motion.py` | `analyze_motion(ctx, seg) -> MotionAnalysis` |
| time fx | `analyzers/timefx.py` | `analyze_time_effects(ctx, seg, motion) -> EffectsResult` |
| look | `analyzers/look.py` | `analyze_look(ctx, seg) -> LookAnalysis` |
| text | `analyzers/text.py` | `detect_text(ctx, seg) -> TextAnalysis` |
| audio | `analyzers/audio.py` | `analyze_audio(ctx) -> AudioAnalysis` |
| build | `build.py` | `build_edit(ctx, seg, motion, timefx, look, text, audio) -> Edit` (any result may be `None`) |
| pipeline | `pipeline.py`, `fetch.py` | `analyze(source, config=None, progress=None) -> AnalysisResult`; `download(url, out_dir) -> Path` |
| cli | `cli.py`, `__main__.py` | `editast analyze | show | render | bench | compare | schema | fetch | batch` |
| language | `lang/printer.py`, `lang/parser.py` | `to_dsl(edit, compact=False) -> str`; `parse_dsl(text) -> Edit` (lossless round trip) |
| reports | `report/html.py`, `report/tree.py` | `write_report(edit, path, ctx=None, signals=None) -> Path`; `build_tree(edit)`, `print_tree(edit, console=None)`, `summary_text(edit) -> str` |
| render | `render/compiler.py`, `render/sources.py` | `render_edit(edit, sources, out_path, music=None, size=None, fps=None, progress=None) -> Path`; `retime_edit(edit, bpm, offset=0.0) -> Edit`; `VideoClipSource`, `SyntheticSource` |
| bench | `bench/generate.py`, `bench/evaluate.py` | `generate_case(seed, out_dir, ...) -> (Path, Edit)`; `generate_suite(out_dir, n, seed)`; `compare(truth, pred) -> dict`; `similarity(a, b) -> float`; `format_report(report) -> str` |

## Milestones

1. Foundation contract (AST, media I/O, context, fx, synthetic data) — **done**.
2. Parallel build of all modules above, each with its own tests.
3. Integration: full test suite green, CLI end to end, benchmark baseline.
4. Tuning against the benchmark + real-video smoke tests; README.

## Environment

* venv: `C:\Users\presh\.venvs\editast` (kept outside OneDrive to avoid sync churn).
  `& C:\Users\presh\.venvs\editast\Scripts\Activate.ps1` then `editast --help`.
* ffmpeg comes from `imageio-ffmpeg`; OCR from `rapidocr-onnxruntime` (optional);
  URL download from `yt-dlp` (optional).
* Tests: `python -m pytest`.
