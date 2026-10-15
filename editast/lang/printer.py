r"""EditScript printer: render an :class:`~editast.ast.nodes.Edit` as readable source.

EditScript is the AST written as the "source code" of an edit. Full mode is
**lossless**: ``parse_dsl(to_dsl(e)) == Edit.from_json(e.to_json())`` for every
valid Edit (floats are rounded to 4 decimals exactly like ``Edit.to_json``).
Compact mode (``compact=True``) drops noisy or derivable data -- beat lists,
curves, looks / layouts / motion summaries, keyframes, words, frame indices,
confidences and provenance -- and rounds numbers for reading; it still parses.

Example::

    # editast EditScript
    edit "clip.mp4" {
      media 1080x1920 9:16 @30fps duration 15.03s analysis_fps 30 frames 451 audio
      music bpm 140.2 offset 0.12s {
        beats [0.12, 0.55, 0.98]
        drop 6.40s strength 2.1
        section aud1 build 0.00s..6.40s energy 0.42
        energy t0 0.00s dt 0.10s [0.1, 0.4, 0.9]
      }
      section sec1 hook 0.00s..1.71s cps 2.34 {
        shot s1 0.00s..0.85s frames 0..26 beats 2.0 on_beat group A {
          look brightness 0.52 saturation 0.41 dominant #c0a080
          zoom s1.zoom1 0.00s..0.85s 1.0->1.25 ease_out keys [0.00: 1.0, 0.85: 1.25]
        }
        transition t1 flash 0.82s..0.88s s1->s2 color #ffffff on_beat
      }
      text txt1 "WAIT FOR IT" 0.00s..1.71s at center bbox 0.1,0.4,0.8,0.08 anim pop role hook
      pattern p1 cut_every_n_beats(n=2) support 0.83 "Cuts land every 2 beats"
      stats shots 34 mean 0.44s cpm 136.0 { pacing t0 0.00s dt 1.00s [2.0, 3.0] }
      provenance {analyzers: {shots: "1.0"}}
    }

Lexical structure
-----------------
* ``#`` starts a comment to the end of the line -- except ``#rrggbb`` directly
  followed by whitespace, ``, ; ) ] }`` or the end of input, which is a colour.
* Statements end at a newline or ``;``. Newlines inside ``[ ]``, ``( )`` and
  JSON ``{ }`` values are ignored, so long values may wrap.
* ``IDENT``  = ``[A-Za-z_][A-Za-z0-9_.]*`` (ids such as ``s1.zoom1`` are one IDENT).
* ``NUMBER`` = ``[+-]digits[.digits][e[+-]digits]`` optionally followed directly
  by a unit made of letters: ``s`` (seconds), ``fps``, ``x`` (rate / ``WxH``),
  ``deg``, ``Hz``. Units are optional wherever they are accepted.
* ``STRING`` = ``"..."``; any Unicode; escapes ``\\ \" \' \/ \n \r \t \b \f \0
  \xHH \uHHHH \UHHHHHHHH`` (``\u`` surrogate pairs combine).
* ``COLOR``  = ``#rrggbb`` or ``rgb(r, g, b)`` (the latter for any integers).
* Punctuation: ``{ } [ ] ( ) , : = -> .. @``.

Grammar (EBNF; ``clause*`` lists are order-free, each field at most once)
--------------------------------------------------------------------------
::

    script     = "edit" [STRING] clause* "{" edit_stmt* "}"      STRING = media.source
    edit_stmt  = media | music | timeline | section | text | pattern | stats | provenance
    media      = "media" clause*
    music      = "music" clause* ["{" music_stmt* "}"]
    music_stmt = ("beats" | "downbeats" | "onsets") numlist
               | "drop" time clause*
               | "section" [IDENT] LABEL range clause*           (AudioSection)
               | "energy" curve
    timeline   = "timeline" clause*                       (id / confidence only)
    section    = "section" [IDENT] ROLE range clause* ["{" (shot | transition)* "}"]
    shot       = "shot" [IDENT] range clause* ["{" shot_stmt* "}"]
    shot_stmt  = "look" clause* | "layout" clause* | "motion" clause* | effect
    effect     = EFFECT_TYPE [IDENT] range clause*
    transition = "transition" [IDENT] KIND range clause*
    text       = "text" [IDENT] [STRING] range clause* ["{" ("word" STRING range)* "}"]
    pattern    = "pattern" [IDENT] (IDENT | STRING) "(" [key ("="|":") value {"," ...}] ")" clause*
    stats      = "stats" clause* ["{" "pacing" curve "}"]
    provenance = "provenance" object
    curve      = clause* numlist                                  (clauses t0, dt)
    range      = time ".." time            time = NUMBER (unit s optional)
    numlist    = "[" [NUMBER {"," NUMBER} [","]] "]"
    clause     = sugar | IDENT "=" value      (generic escape hatch: any field, JSON value)
    value      = STRING | NUMBER | "true" | "false" | "null" | IDENT (a bare string)
               | "[" [value {"," value} [","]] "]" | object
    object     = "{" [key (":"|"=") value {"," key (":"|"=") value} [","]] "}"
    key        = IDENT | STRING

In a statement head, ``[IDENT]`` is the node id; it is recognised by position
(two words before the range for transitions / sections, an IDENT before ``(``
for patterns). Ids that are not IDENTs are written ``id="..."``.

Sugar clauses per node (``F`` float, ``T`` time, ``I`` int, ``W`` IDENT or STRING)
-----------------------------------------------------------------------------------
=============  ==========================================================================
node           clauses
=============  ==========================================================================
edit           ``conf F``
media          ``WxH`` · ``N:M`` (aspect) · ``aspect STRING`` · ``@Ffps`` (source fps) ·
               ``duration T`` · ``analysis_fps F`` · ``frames I`` · ``audio`` / ``no_audio``
music          ``bpm F`` · ``offset T`` (beat_offset) · ``meter I`` · ``speech`` /
               ``no_speech`` · ``conf F``
audio section  ``energy F`` · ``conf F``            drop: ``strength F``
curve          ``t0 T`` · ``dt T`` · numlist (values)
section        ``cps F`` · ``mean_shot T`` · ``energy F`` · ``conf F``
shot           ``index I`` · ``frames I..I`` · ``beats F`` · ``on_beat`` / ``off_beat`` ·
               ``offset T`` · ``group W`` · ``conf F``
look           ``brightness`` ``contrast`` ``saturation`` ``temperature`` ``tint``
               ``sharpness`` ``vignette`` (each ``F``) · ``dominant COLOR`` · ``bw``
layout         ``letterbox F,F`` (top,bottom) · ``pillarbox F,F`` (left,right) ·
               ``content F,F,F,F`` · ``blur_pad`` · ``split W``
motion         ``camera W`` · ``magnitude F`` · ``zoom F`` · ``pan F,F`` · ``rotation Fdeg``
transition     ``W->W`` (from/to shot; ``none`` = no shot) · ``color COLOR`` ·
               ``on_beat`` / ``off_beat`` · ``offset T`` · ``conf F`` · ``params object``
text           ``at W`` (position) · ``bbox F,F,F,F`` · ``size F`` · ``color COLOR`` ·
               ``outline`` / ``no_outline`` · ``anim W`` · ``role W`` · ``conf F``
pattern        ``support F`` · ``span T..T`` · ``conf F`` · STRING (description)
stats          ``shots I`` · ``transitions I`` · ``mean T`` · ``median T`` · ``min T`` ·
               ``max T`` · ``cpm F`` · ``on_beat_ratio F`` · ``hook T`` · ``first_cut T`` ·
               ``text_coverage F`` · ``effects object`` · ``transition_kinds object``
every effect   bare easing word (``ease_out``, ``punch`` ...) · ``conf F`` ·
               ``keys [T: F, ...]`` (keyframes) · ``params object``
zoom           ``F->F`` (scale) · ``center F,F`` · ``digital`` / ``optical``
pan            ``by F,F`` (dx,dy)
rotate         ``Fdeg->Fdeg`` (angle)
shake          ``amp F`` · ``freq FHz`` · ``rot Fdeg``
speed          bare kind word (``slowmo``, ``ramp_up`` ...) · ``Fx->Fx`` (rate)
flash          ``color COLOR`` · ``intensity F``
blur           ``amount F``          rgb_split: ``offset F``     freeze / invert: --
=============  ==========================================================================

Defaults: an omitted field takes the model default, except three DSL defaults
that match :func:`~editast.ast.nodes.assign_ids`: the edit id is ``edit``, the
timeline id ``timeline`` and the music id ``music``; an omitted shot ``index``
is derived from a canonical shot id (``s3`` -> 2, otherwise 0). Any field may
always be set explicitly with ``name=value`` (e.g. ``id=""``, ``index=7``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from editast.ast.nodes import (
    AudioSection,
    Blur,
    Curve,
    Drop,
    Edit,
    EditStats,
    Flash,
    Freeze,
    Invert,
    Layout,
    Look,
    MediaInfo,
    MotionSummary,
    Music,
    Pan,
    Pattern,
    RGBSplit,
    Rotate,
    Section,
    Shake,
    Shot,
    Speed,
    TextOverlay,
    Timeline,
    Transition,
    Word,
    Zoom,
)

__all__ = ["to_dsl", "Clause", "NodeSpec", "SPECS", "EFFECT_SPECS", "quote", "is_ident", "derived_shot_index"]

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")
_RATIO_RE = re.compile(r"[1-9][0-9]*:[1-9][0-9]*\Z")
_SHOT_ID_RE = re.compile(r"s([1-9][0-9]*)\Z")
_ESCAPE_RE = re.compile('["\\\\\x00-\x1f\x7f-\x9f\u2028\u2029\ufeff\ud800-\udfff]')
_ESCAPES = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}

WIDTH = 100  # soft line width used when wrapping long lists / JSON values

ClauseKind = Literal[
    "time", "num", "int", "word", "str", "color", "vec", "flag", "bare", "arrow", "json",
    "frames", "span", "link", "size", "ratio", "at_fps", "keys", "list", "desc",
]


# --------------------------------------------------------------------------- clause tables


@dataclass(frozen=True)
class Clause:
    """One sugar clause of a node statement.

    ``word`` is the keyword (empty for clauses recognised by their shape, e.g.
    ``1.0->1.2``); ``fields`` the model fields it carries; ``unit`` is printed
    after numbers and accepted (optionally) when parsing, along with ``accept``.
    ``alt`` is the word meaning ``False`` for flags. ``digits`` is the minimum
    number of decimals printed. ``always`` prints the clause even at its default;
    ``compact=False`` drops it in compact mode.
    """

    word: str
    fields: tuple[str, ...]
    kind: ClauseKind
    unit: str = ""
    accept: tuple[str, ...] = ()
    alt: str = ""
    digits: int = 1
    always: bool = False
    compact: bool = True


@dataclass
class NodeSpec:
    """How one node type is written: keyword, sugar clauses, nested fields."""

    model: type[BaseModel]
    keyword: str
    clauses: tuple[Clause, ...] = ()
    children: tuple[str, ...] = ()  # fields written as nested statements
    dsl_defaults: dict[str, Any] = field(default_factory=dict)
    _defaults: dict[str, Any] = field(default_factory=dict, repr=False)

    def default(self, name: str) -> Any:
        """JSON-mode default of ``name`` (DSL overrides first); ``PydanticUndefined`` if required."""
        if name in self.dsl_defaults:
            return self.dsl_defaults[name]
        if name not in self._defaults:
            value = self.model.model_fields[name].get_default(call_default_factory=True)
            if value is not PydanticUndefined:
                value = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
                value = json.loads(json.dumps(value))
            self._defaults[name] = value
        return self._defaults[name]

    def literal_values(self, name: str) -> Optional[tuple[str, ...]]:
        """Allowed values when ``name`` is a (possibly Optional) ``Literal`` field."""
        return literal_values(self.model.model_fields[name].annotation)


def literal_values(annotation: Any) -> Optional[tuple[str, ...]]:
    origin = get_origin(annotation)
    if origin is Literal:
        return tuple(str(a) for a in get_args(annotation))
    if origin is Union:
        for arg in get_args(annotation):
            values = literal_values(arg)
            if values:
                return values
    return None


_CONF = Clause("conf", ("confidence",), "num", compact=False)


def _effect_spec(model: type[BaseModel], main: tuple[Clause, ...], rest: tuple[Clause, ...] = ()) -> NodeSpec:
    keyword = model.model_fields["type"].default
    tail = (
        _CONF,
        Clause("keys", ("keyframes",), "keys", compact=False),
        Clause("params", ("params",), "json"),
    )
    return NodeSpec(model, keyword, main + (Clause("", ("easing",), "bare"),) + rest + tail)


EFFECT_SPECS: dict[str, NodeSpec] = {
    spec.keyword: spec
    for spec in (
        _effect_spec(Zoom, (Clause("", ("scale_from", "scale_to"), "arrow"),),
                     (Clause("center", ("center",), "vec"),
                      Clause("digital", ("digital",), "flag", alt="optical"))),
        _effect_spec(Pan, (Clause("by", ("dx", "dy"), "vec"),)),
        _effect_spec(Rotate, (Clause("", ("angle_from", "angle_to"), "arrow", unit="deg"),)),
        _effect_spec(Shake, (Clause("amp", ("amplitude",), "num"),
                             Clause("freq", ("frequency",), "num", unit="Hz", accept=("hz",)),
                             Clause("rot", ("rotation_amplitude",), "num", unit="deg"))),
        _effect_spec(Speed, (Clause("", ("kind",), "bare"),
                             Clause("", ("rate_from", "rate_to"), "arrow", unit="x"))),
        _effect_spec(Freeze, ()),
        _effect_spec(Flash, (Clause("color", ("color",), "color"),
                             Clause("intensity", ("intensity",), "num"))),
        _effect_spec(Blur, (Clause("amount", ("amount",), "num"),)),
        _effect_spec(Invert, ()),
        _effect_spec(RGBSplit, (Clause("offset", ("offset",), "num"),)),
    )
}

SPECS: dict[str, NodeSpec] = {
    "edit": NodeSpec(Edit, "edit", (_CONF,),
                     children=("media", "music", "timeline", "overlays", "patterns", "stats", "provenance"),
                     dsl_defaults={"id": "edit"}),
    "media": NodeSpec(MediaInfo, "media", (
        Clause("", ("width", "height"), "size", always=True),
        Clause("", ("aspect",), "ratio"),
        Clause("aspect", ("aspect",), "str"),
        Clause("", ("fps",), "at_fps", unit="fps", digits=0, always=True),
        Clause("duration", ("duration",), "time", always=True),
        Clause("analysis_fps", ("analysis_fps",), "num", accept=("fps",), digits=0),
        Clause("frames", ("n_frames",), "int"),
        Clause("audio", ("has_audio",), "flag", alt="no_audio"),
    ), children=("source",)),
    "music": NodeSpec(Music, "music", (
        Clause("bpm", ("bpm",), "num"),
        Clause("offset", ("beat_offset",), "time"),
        Clause("meter", ("meter",), "int"),
        Clause("speech", ("has_speech",), "flag", alt="no_speech"),
        _CONF,
    ), children=("beats", "downbeats", "onsets", "drops", "sections", "energy"),
        dsl_defaults={"id": "music"}),
    "audio_section": NodeSpec(AudioSection, "section", (
        Clause("energy", ("energy",), "num"), _CONF)),
    "drop": NodeSpec(Drop, "drop", (Clause("strength", ("strength",), "num"),)),
    "curve": NodeSpec(Curve, "curve", (
        Clause("t0", ("t0",), "time", always=True),
        Clause("dt", ("dt",), "time", always=True),
        Clause("", ("values",), "list", always=True),
    )),
    "timeline": NodeSpec(Timeline, "timeline", (_CONF,), children=("sections",),
                         dsl_defaults={"id": "timeline"}),
    "section": NodeSpec(Section, "section", (
        Clause("cps", ("cuts_per_second",), "num"),
        Clause("mean_shot", ("mean_shot_duration",), "time"),
        Clause("energy", ("energy",), "num"),
        _CONF,
    ), children=("items",)),
    "shot": NodeSpec(Shot, "shot", (
        Clause("index", ("index",), "int"),
        Clause("frames", ("frame_start", "frame_end"), "frames", compact=False),
        Clause("beats", ("beats",), "num"),
        Clause("on_beat", ("on_beat",), "flag", alt="off_beat"),
        Clause("offset", ("beat_offset",), "time"),
        Clause("group", ("source_group",), "word"),
        _CONF,
    ), children=("look", "layout", "motion", "effects")),
    "look": NodeSpec(Look, "look", tuple(
        Clause(name, (name,), "num")
        for name in ("brightness", "contrast", "saturation", "temperature", "tint", "sharpness", "vignette")
    ) + (Clause("dominant", ("dominant_color",), "color"), Clause("bw", ("bw",), "flag"))),
    "layout": NodeSpec(Layout, "layout", (
        Clause("letterbox", ("letterbox_top", "letterbox_bottom"), "vec"),
        Clause("pillarbox", ("pillarbox_left", "pillarbox_right"), "vec"),
        Clause("content", ("content_box",), "vec"),
        Clause("blur_pad", ("blur_pad",), "flag"),
        Clause("split", ("split",), "word"),
    )),
    "motion": NodeSpec(MotionSummary, "motion", (
        Clause("camera", ("camera",), "word"),
        Clause("magnitude", ("magnitude",), "num"),
        Clause("zoom", ("zoom_ratio",), "num"),
        Clause("pan", ("pan",), "vec"),
        Clause("rotation", ("rotation",), "num", unit="deg"),
    )),
    "transition": NodeSpec(Transition, "transition", (
        Clause("", ("from_shot", "to_shot"), "link"),
        Clause("color", ("color",), "color"),
        Clause("on_beat", ("on_beat",), "flag", alt="off_beat"),
        Clause("offset", ("beat_offset",), "time"),
        _CONF,
        Clause("params", ("params",), "json"),
    )),
    "text": NodeSpec(TextOverlay, "text", (
        Clause("at", ("position",), "word", always=True),
        Clause("bbox", ("bbox",), "vec"),
        Clause("size", ("size",), "num"),
        Clause("color", ("color",), "color"),
        Clause("outline", ("outline",), "flag", alt="no_outline"),
        Clause("anim", ("animation",), "word"),
        Clause("role", ("role",), "word"),
        _CONF,
    ), children=("words",)),
    "word": NodeSpec(Word, "word"),
    "pattern": NodeSpec(Pattern, "pattern", (
        Clause("support", ("support",), "num"),
        Clause("span", ("start", "end"), "span"),
        _CONF,
        Clause("", ("description",), "desc"),
    )),
    "stats": NodeSpec(EditStats, "stats", (
        Clause("shots", ("n_shots",), "int"),
        Clause("transitions", ("n_transitions",), "int"),
        Clause("mean", ("mean_shot_duration",), "time"),
        Clause("median", ("median_shot_duration",), "time"),
        Clause("min", ("min_shot_duration",), "time"),
        Clause("max", ("max_shot_duration",), "time"),
        Clause("cpm", ("cuts_per_minute",), "num"),
        Clause("on_beat_ratio", ("cut_on_beat_ratio",), "num"),
        Clause("hook", ("hook_duration",), "time"),
        Clause("first_cut", ("first_cut_time",), "time"),
        Clause("text_coverage", ("text_coverage",), "num"),
        Clause("effects", ("effect_counts",), "json"),
        Clause("transition_kinds", ("transition_counts",), "json"),
    ), children=("pacing",)),
}

# Fields each statement writes in its head (before the clauses).
HEAD_FIELDS: dict[str, tuple[str, ...]] = {
    "audio_section": ("id", "label", "start", "end"),
    "drop": ("time",),
    "section": ("id", "role", "start", "end"),
    "shot": ("id", "start", "end"),
    "transition": ("id", "kind", "start", "end"),
    "text": ("id", "text", "start", "end"),
    "word": ("text", "start", "end"),
    "pattern": ("id", "rule", "params"),
    "effect": ("id", "start", "end"),
}


# --------------------------------------------------------------------------- formatting helpers


def is_ident(value: Any) -> bool:
    return isinstance(value, str) and IDENT_RE.match(value) is not None


def quote(text: str) -> str:
    """EditScript string literal (non-ASCII printable characters are kept as is)."""

    def esc(m: re.Match[str]) -> str:
        ch = m.group()
        return _ESCAPES.get(ch) or f"\\u{ord(ch):04x}"

    return '"' + _ESCAPE_RE.sub(esc, text) + '"'


def word(value: str) -> str:
    """A bare IDENT when possible, else a string literal."""
    return value if is_ident(value) else quote(value)


def fmt_float(x: float, digits: int = 1) -> str:
    """Shortest fixed-point text with at least ``digits`` decimals that parses back to ``x``."""
    x = float(x)
    if math.isfinite(x) and abs(x) < 1e15:
        for k in range(digits, 18):
            text = f"{x:.{k}f}"
            if float(text) == x:
                return text
    return repr(x)


def derived_shot_index(shot_id: Any) -> int:
    """Index implied by a canonical shot id (``s3`` -> 2); 0 for anything else."""
    m = _SHOT_ID_RE.match(shot_id) if isinstance(shot_id, str) else None
    return int(m.group(1)) - 1 if m else 0


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def literal(value: Any) -> str:
    """One-line JSON-like literal (bare IDENT keys, EditScript strings)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return quote(value)
    if isinstance(value, list):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{word(str(k))}: {literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"cannot write {type(value).__name__} as an EditScript value")


# --------------------------------------------------------------------------- printer


class _Printer:
    def __init__(self, compact: bool) -> None:
        self.compact = compact
        self.lines: list[str] = []
        self.depth = 0

    # ---- low level -----------------------------------------------------------

    def emit(self, text: str) -> None:
        self.lines.append("  " * self.depth + text)

    def blank(self) -> None:
        if self.lines and self.lines[-1] != "":
            self.lines.append("")

    def num(self, x: float, digits: int = 1) -> str:
        if self.compact:
            x = round(float(x), 2 if digits >= 2 else 3)
        return fmt_float(x, digits)

    def time(self, x: float) -> str:
        return self.num(x, 2) + "s"

    def wrap(self, open_: str, items: list[str], close: str) -> str:
        """``open_ item, item ... close`` on one line, or wrapped at :data:`WIDTH`."""
        flat = open_ + ", ".join(items) + close
        indent = "  " * self.depth
        if len(indent) + len(flat) <= WIDTH:
            return flat
        inner = indent + "    "
        rows, row = [], ""
        for item in items:
            piece = item + ","
            if row and len(inner) + len(row) + 1 + len(piece) > WIDTH:
                rows.append(inner + row)
                row = piece
            else:
                row = f"{row} {piece}" if row else piece
        if row:
            rows.append(inner + row)
        return open_ + "\n" + "\n".join(rows) + "\n" + indent + "  " + close

    def value(self, v: Any, extra: int = 0) -> str:
        """JSON-like literal, broken over lines when long (newlines are legal inside brackets)."""
        flat = literal(v)
        indent = "  " * (self.depth + 1 + extra)
        if len(indent) + len(flat) <= WIDTH or not isinstance(v, (list, dict)) or not v:
            return flat
        if isinstance(v, list):
            body = [indent + "  " + self.value(item, extra + 1) for item in v]
            return "[\n" + ",\n".join(body) + "\n" + indent + "]"
        body = [f"{indent}  {word(str(k))}: {self.value(item, extra + 1)}" for k, item in v.items()]
        return "{\n" + ",\n".join(body) + "\n" + indent + "}"

    def color(self, rgb: Any) -> Optional[str]:
        if not (isinstance(rgb, list) and len(rgb) == 3 and all(_is_int(c) for c in rgb)):
            return None
        if all(0 <= c <= 255 for c in rgb):
            return "#{:02x}{:02x}{:02x}".format(*rgb)
        return "rgb({}, {}, {})".format(*rgb)

    # ---- clauses ---------------------------------------------------------------

    def render(self, cl: Clause, vals: list[Any]) -> Optional[str]:
        """Sugar text for ``cl`` with field values ``vals``; ``None`` when not expressible."""
        k = cl.kind
        prefix = cl.word + " " if cl.word else ""
        if k in ("time", "num", "at_fps"):
            if not _is_num(vals[0]):
                return None
            text = self.time(vals[0]) if k == "time" else self.num(vals[0], cl.digits) + cl.unit
            return ("@" if k == "at_fps" else prefix) + text
        if k == "int":
            return prefix + str(vals[0]) if _is_int(vals[0]) else None
        if k in ("word", "str"):
            if not isinstance(vals[0], str):
                return None
            return prefix + (word(vals[0]) if k == "word" else quote(vals[0]))
        if k == "color":
            text = self.color(vals[0])
            return prefix + text if text else None
        if k == "vec":
            nums = vals[0] if len(cl.fields) == 1 else vals
            if not (isinstance(nums, list) and nums and all(_is_num(v) for v in nums)):
                return None
            return prefix + ",".join(self.num(v) for v in nums)
        if k == "flag":
            if vals[0] is True:
                return cl.word
            return cl.alt if vals[0] is False and cl.alt else None
        if k == "bare":
            return vals[0] if isinstance(vals[0], str) and is_ident(vals[0]) else None
        if k == "arrow":
            if not all(_is_num(v) for v in vals):
                return None
            return f"{self.num(vals[0])}{cl.unit}->{self.num(vals[1])}{cl.unit}"
        if k == "json":
            return f"{cl.word} {self.value(vals[0])}"
        if k == "frames":
            return f"{cl.word} {vals[0]}..{vals[1]}" if all(_is_int(v) for v in vals) else None
        if k == "span":
            if not all(_is_num(v) for v in vals):
                return None
            return f"{cl.word} {self.time(vals[0])}..{self.time(vals[1])}"
        if k == "link":
            if not all(v is None or isinstance(v, str) for v in vals):
                return None
            return "->".join("none" if v is None else (v if is_ident(v) and v != "none" else quote(v))
                             for v in vals)
        if k == "size":
            return f"{vals[0]}x{vals[1]}" if all(_is_int(v) and v >= 0 for v in vals) else None
        if k == "ratio":
            return vals[0] if isinstance(vals[0], str) and _RATIO_RE.match(vals[0]) else None
        if k == "keys":
            if not all(isinstance(kf, dict) and _is_num(kf.get("t")) and _is_num(kf.get("v")) for kf in vals[0]):
                return None
            items = [f"{self.num(kf['t'], 2)}: {self.num(kf['v'])}" for kf in vals[0]]
            return self.wrap("keys [", items, "]")
        if k == "list":
            if not all(_is_num(v) for v in vals[0]):
                return None
            return self.wrap("[", [self.num(v) for v in vals[0]], "]")
        if k == "desc":
            return quote(vals[0]) if isinstance(vals[0], str) else None
        raise ValueError(f"unknown clause kind {k!r}")

    def statement(self, spec: NodeSpec, d: dict[str, Any], head: list[str], handled: set[str],
                  extra_defaults: Optional[dict[str, Any]] = None, keyword: Optional[str] = None) -> str:
        """``keyword head... clauses... generic=...`` for node dict ``d``."""
        parts = [keyword or spec.keyword, *head]
        done = set(handled) | set(spec.children) | {"type"}
        extra_defaults = extra_defaults or {}

        def default(name: str) -> Any:
            return extra_defaults[name] if name in extra_defaults else spec.default(name)

        for cl in spec.clauses:
            if any(f in done for f in cl.fields):
                continue
            if self.compact and not cl.compact:
                done.update(cl.fields)
                continue
            vals = [d.get(f) for f in cl.fields]
            if not cl.always and all(v == default(f) for v, f in zip(vals, cl.fields)):
                done.update(cl.fields)
                continue
            text = self.render(cl, vals)
            if text is not None:
                parts.append(text)
                done.update(cl.fields)
        for name, val in d.items():
            if name in done or val == default(name):
                continue
            parts.append(f"{name}={self.value(val)}")
        return " ".join(parts)

    def head_id(self, d: dict[str, Any], head: list[str], handled: set[str]) -> None:
        if is_ident(d.get("id")):
            head.append(d["id"])
            handled.add("id")
        elif d.get("id") == "":
            handled.add("id")

    def range_(self, d: dict[str, Any], head: list[str], handled: set[str]) -> None:
        head.append(f"{self.time(d['start'])}..{self.time(d['end'])}")
        handled.update(("start", "end"))

    def block(self, line: str, body: list[tuple[str, Any]]) -> None:
        """Emit ``line`` and, if ``body`` has items, a ``{ ... }`` block around them."""
        if not body:
            self.emit(line)
            return
        self.emit(line + " {")
        self.depth += 1
        for kind, payload in body:
            getattr(self, "p_" + kind)(payload)
        self.depth -= 1
        self.emit("}")

    # ---- nodes -----------------------------------------------------------------

    def p_edit(self, d: dict[str, Any]) -> None:
        self.emit("# editast EditScript" + (" (compact view: lists, curves, looks and provenance omitted)"
                                            if self.compact else ""))
        head: list[str] = []
        if d["media"]["source"]:
            head.append(quote(d["media"]["source"]))
        self.emit(self.statement(SPECS["edit"], d, head, set()) + " {")
        self.depth += 1
        self.p_media(d["media"])
        if d["music"] is not None:
            self.p_music(d["music"])
        tl = self.statement(SPECS["timeline"], d["timeline"], [], set())
        if tl != "timeline":
            self.emit(tl)
        for sec in d["timeline"]["sections"]:
            self.blank()
            self.p_section(sec)
        if d["overlays"]:
            self.blank()
        for ov in d["overlays"]:
            self.p_text(ov)
        if d["patterns"]:
            self.blank()
        for pat in d["patterns"]:
            self.p_pattern(pat)
        if d["stats"] is not None:
            self.blank()
            self.p_stats(d["stats"])
        if d["provenance"] and not self.compact:
            self.blank()
            self.emit("provenance " + self.value(d["provenance"], -1))
        self.depth -= 1
        self.emit("}")

    def p_media(self, d: dict[str, Any]) -> None:
        self.emit(self.statement(SPECS["media"], d, [], set()))

    def p_music(self, d: dict[str, Any]) -> None:
        body: list[tuple[str, Any]] = []
        lists = [(name, d[name]) for name in ("beats", "downbeats", "onsets") if d[name]]
        if self.compact:
            if lists:
                body.append(("comment", ", ".join(f"{len(v)} {name}" for name, v in lists) + " omitted"))
        else:
            body += [("numlist", item) for item in lists]
        body += [("drop", drop) for drop in d["drops"]]
        body += [("audio_section", sec) for sec in d["sections"]]
        if d["energy"] is not None and not self.compact:
            body.append(("curve", ("energy", d["energy"])))
        self.block(self.statement(SPECS["music"], d, [], set()), body)

    def p_comment(self, text: str) -> None:
        self.emit("# " + text)

    def p_numlist(self, item: tuple[str, list[float]]) -> None:
        name, values = item
        self.emit(self.wrap(name + " [", [self.num(v) for v in values], "]"))

    def p_curve(self, item: tuple[str, dict[str, Any]]) -> None:
        name, d = item
        self.emit(self.statement(SPECS["curve"], d, [], set(), keyword=name))

    def p_drop(self, d: dict[str, Any]) -> None:
        self.emit(self.statement(SPECS["drop"], d, [self.time(d["time"])], {"time"}))

    def p_audio_section(self, d: dict[str, Any]) -> None:
        head: list[str] = []
        handled: set[str] = {"label"}
        self.head_id(d, head, handled)
        head.append(d["label"])
        self.range_(d, head, handled)
        self.emit(self.statement(SPECS["audio_section"], d, head, handled))

    def p_section(self, d: dict[str, Any]) -> None:
        head: list[str] = []
        handled: set[str] = {"role"}
        self.head_id(d, head, handled)
        head.append(d["role"])
        self.range_(d, head, handled)
        body = [(item["type"], item) for item in d["items"]]
        self.block(self.statement(SPECS["section"], d, head, handled), body)

    def p_shot(self, d: dict[str, Any]) -> None:
        head: list[str] = []
        handled: set[str] = set()
        self.head_id(d, head, handled)
        self.range_(d, head, handled)
        line = self.statement(SPECS["shot"], d, head, handled,
                              extra_defaults={"index": derived_shot_index(d["id"])})
        body: list[tuple[str, Any]] = []
        if not self.compact:
            body += [(name, d[name]) for name in ("look", "layout", "motion") if d[name] is not None]
        body += [("effect", eff) for eff in d["effects"]]
        self.block(line, body)

    def p_look(self, d: dict[str, Any]) -> None:
        self.emit(self.statement(SPECS["look"], d, [], set()))

    def p_layout(self, d: dict[str, Any]) -> None:
        self.emit(self.statement(SPECS["layout"], d, [], set()))

    def p_motion(self, d: dict[str, Any]) -> None:
        self.emit(self.statement(SPECS["motion"], d, [], set()))

    def p_effect(self, d: dict[str, Any]) -> None:
        head: list[str] = []
        handled: set[str] = set()
        self.head_id(d, head, handled)
        self.range_(d, head, handled)
        self.emit(self.statement(EFFECT_SPECS[d["type"]], d, head, handled))

    def p_transition(self, d: dict[str, Any]) -> None:
        head: list[str] = []
        handled: set[str] = {"kind"}
        self.head_id(d, head, handled)
        head.append(d["kind"])
        self.range_(d, head, handled)
        self.emit(self.statement(SPECS["transition"], d, head, handled))

    def p_text(self, d: dict[str, Any]) -> None:
        head: list[str] = []
        handled: set[str] = {"text"}
        self.head_id(d, head, handled)
        if d["text"] is not None:
            head.append(quote(d["text"]))
        self.range_(d, head, handled)
        body = [] if self.compact else [("word", w) for w in d["words"]]
        self.block(self.statement(SPECS["text"], d, head, handled), body)

    def p_word(self, d: dict[str, Any]) -> None:
        head = [quote(d["text"]), f"{self.time(d['start'])}..{self.time(d['end'])}"]
        self.emit(self.statement(SPECS["word"], d, head, {"text", "start", "end"}))

    def p_pattern(self, d: dict[str, Any]) -> None:
        head: list[str] = []
        handled: set[str] = {"rule", "params"}
        self.head_id(d, head, handled)
        args = [f"{word(str(k))}={self.value(v)}" for k, v in d["params"].items()]
        head.append(word(d["rule"]) + "(" + ", ".join(args) + ")")
        self.emit(self.statement(SPECS["pattern"], d, head, handled))

    def p_stats(self, d: dict[str, Any]) -> None:
        body = []
        if d["pacing"] is not None and not self.compact:
            body.append(("curve", ("pacing", d["pacing"])))
        self.block(self.statement(SPECS["stats"], d, [], set()), body)


def to_dsl(edit: Edit, compact: bool = False) -> str:
    """Render ``edit`` as EditScript source (see the module docstring for the grammar).

    Full mode is lossless: ``parse_dsl(to_dsl(edit)) == Edit.from_json(edit.to_json())``.
    ``compact=True`` omits beat lists, curves, looks, keyframes, words and provenance.
    """
    data = json.loads(edit.to_json(indent=None))
    printer = _Printer(compact)
    printer.p_edit(data)
    return "\n".join(printer.lines) + "\n"
