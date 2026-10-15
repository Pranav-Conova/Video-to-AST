"""EditScript parser: hand-written tokenizer + recursive-descent parser.

The grammar is documented in :mod:`editast.lang.printer`. Parsing builds the
JSON-mode dictionary of an :class:`~editast.ast.nodes.Edit` (exactly the shape
``Edit.to_json`` produces, minus fields left at their defaults) and validates
it with pydantic at the end, so ``parse_dsl(to_dsl(e)) == Edit.from_json(e.to_json())``.

Sugar clauses are recognised with the clause tables shared with the printer
(:data:`~editast.lang.printer.SPECS`): a clause is found either by its keyword
(``beats 2.0``, ``on_beat``), by a bare enum value (``ease_out``) or by its
shape (``1.0->1.2``, ``s1->s2``, ``1080x1920``, ``9:16``, ``@30fps``, ``[..]``).
``name=value`` sets any other field with a JSON-like literal.

Every error -- lexical, syntactic, or a pydantic validation error mapped back to
the statement that produced the offending node -- raises :class:`EditScriptError`
carrying ``line`` / ``col`` (1-based) and a caret excerpt of the source line.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from pydantic import ValidationError

from editast.ast.nodes import EFFECT_TYPES, Edit
from editast.lang.printer import (
    EFFECT_SPECS,
    SPECS,
    Clause,
    NodeSpec,
    derived_shot_index,
    literal_values,
)

__all__ = ["EditScriptError", "parse_dsl", "tokenize", "Token"]

IDENT, STRING, NUMBER, COLOR, PUNCT, NL, EOF = "ident", "string", "number", "color", "punct", "nl", "eof"

_LEX = re.compile(
    r"""
      (?P<ws>[ \t\r\f\v]+)
    | (?P<nl>[\n;])
    | (?P<color>\#[0-9A-Fa-f]{6})(?=[\s,;)\]}]|\Z)
    | (?P<comment>\#[^\n]*)
    | (?P<number>[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?P<unit>[A-Za-z]*)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_.]*)
    | (?P<punct>->|\.\.|[{}\[\](),:=@])
    | (?P<string>")
    """,
    re.VERBOSE,
)
_STR_CHUNK = re.compile(r'[^"\\\n]+')
_INT_RE = re.compile(r"[+-]?[0-9]+\Z")
_HEX = {"x": 2, "u": 4, "U": 8}
_SIMPLE_ESCAPES = {
    "\\": "\\", '"': '"', "'": "'", "/": "/", "n": "\n", "r": "\r",
    "t": "\t", "b": "\b", "f": "\f", "0": "\0",
}


class EditScriptError(ValueError):
    """A lexical, syntax or validation error at ``line:col`` of EditScript source."""

    def __init__(self, message: str, line: int, col: int, source_line: str = "") -> None:
        self.message, self.line, self.col = message, line, col
        text = f"line {line}:{col}: {message}"
        if source_line.strip():
            text += f"\n    {source_line}\n    {' ' * (col - 1)}^"
        super().__init__(text)


@dataclass(frozen=True, slots=True)
class Token:
    kind: str  # IDENT STRING NUMBER COLOR PUNCT NL EOF
    value: Any  # identifier / decoded string / number text / (r, g, b) / punctuation
    line: int
    col: int
    unit: str = ""  # NUMBER only

    def is_punct(self, text: str) -> bool:
        return self.kind == PUNCT and self.value == text

    def describe(self) -> str:
        if self.kind == NL:
            return "end of line" if self.value == "\n" else "';'"
        if self.kind == EOF:
            return "end of input"
        if self.kind == STRING:
            return f"string {_short(repr(self.value))}"
        if self.kind == NUMBER:
            return f"number {self.value}{self.unit}"
        if self.kind == COLOR:
            return "colour #{:02x}{:02x}{:02x}".format(*self.value)
        return f"'{self.value}'"


def _is_hex(text: str, width: int) -> bool:
    return len(text) == width and all(c in "0123456789abcdefABCDEF" for c in text)


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


# --------------------------------------------------------------------------- tokenizer


def tokenize(text: str) -> list[Token]:
    """Split EditScript ``text`` into tokens (comments and blanks dropped, newlines kept)."""
    lines = text.split("\n")
    toks: list[Token] = []
    pos, line, line_start, n = 0, 1, 0, len(text)
    while pos < n:
        m = _LEX.match(text, pos)
        col = pos - line_start + 1
        if m is None:
            raise EditScriptError(f"unexpected character {text[pos]!r}", line, col, lines[line - 1])
        kind = m.lastgroup
        if kind == "nl":
            if not toks or toks[-1].kind != NL:
                toks.append(Token(NL, m.group(), line, col))
            if m.group() == "\n":
                line, line_start = line + 1, m.end()
        elif kind == "color":
            hexs = m.group("color")[1:]
            toks.append(Token(COLOR, tuple(int(hexs[i:i + 2], 16) for i in (0, 2, 4)), line, col))
        elif kind in ("number", "unit"):
            toks.append(Token(NUMBER, m.group("number"), line, col, m.group("unit")))
        elif kind == "ident":
            toks.append(Token(IDENT, m.group(), line, col))
        elif kind == "punct":
            toks.append(Token(PUNCT, m.group(), line, col))
        elif kind == "string":
            value, end = _read_string(text, pos, line, line_start, lines[line - 1])
            toks.append(Token(STRING, value, line, col))
            pos = end
            continue
        pos = m.end()
    toks.append(Token(EOF, None, line, pos - line_start + 1))
    return toks


def _read_string(text: str, pos: int, line: int, line_start: int, source_line: str) -> tuple[str, int]:
    """Decode the string literal whose opening quote is at ``pos``; returns (value, end)."""
    out: list[str] = []
    i, n = pos + 1, len(text)
    while True:
        m = _STR_CHUNK.match(text, i)
        if m:
            out.append(m.group())
            i = m.end()
        if i >= n or text[i] == "\n":
            raise EditScriptError("unterminated string (strings cannot span lines; use \\n)",
                                  line, pos - line_start + 1, source_line)
        if text[i] == '"':
            return "".join(out), i + 1
        esc = text[i + 1] if i + 1 < n else ""
        if esc in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[esc])
            i += 2
            continue
        width = _HEX.get(esc)
        digits = text[i + 2:i + 2 + width] if width else ""
        if not width or not _is_hex(digits, width):
            what = f"\\{esc}" + (digits if width else "")
            raise EditScriptError(f"invalid escape sequence {what!r}", line, i - line_start + 1, source_line)
        code = int(digits, 16)
        i += 2 + width
        if 0xD800 <= code < 0xDC00 and text.startswith("\\u", i):
            # A high surrogate followed by a low one encodes a single astral code point.
            low = text[i + 2:i + 6]
            if _is_hex(low, 4) and 0xDC00 <= int(low, 16) < 0xE000:
                code = 0x10000 + ((code - 0xD800) << 10) + (int(low, 16) - 0xDC00)
                i += 6
        if code > 0x10FFFF:
            raise EditScriptError(f"code point {code:#x} is out of range", line, i - line_start + 1, source_line)
        out.append(chr(code))


# --------------------------------------------------------------------------- clause index


@dataclass
class _Index:
    """Lookup tables for recognising the sugar clauses of one node type."""

    words: dict[str, Clause]
    flags: dict[str, tuple[Clause, bool]]
    bare: dict[str, Clause]
    shapes: dict[str, Clause]


def _build_index(spec: NodeSpec) -> _Index:
    idx = _Index({}, {}, {}, {})
    for cl in spec.clauses:
        if cl.kind == "flag":
            idx.flags[cl.word] = (cl, True)
            if cl.alt:
                idx.flags[cl.alt] = (cl, False)
        elif cl.kind == "bare":
            for value in spec.literal_values(cl.fields[0]) or ():
                idx.bare[value] = cl
        elif cl.word:
            idx.words[cl.word] = cl
        else:
            idx.shapes[cl.kind] = cl
    return idx


_INDEX: dict[int, _Index] = {id(spec): _build_index(spec) for spec in (*SPECS.values(), *EFFECT_SPECS.values())}


def _suggest(word: str, options: Iterable[str]) -> str:
    close = difflib.get_close_matches(word, list(options), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


# --------------------------------------------------------------------------- parser


class _Node:
    """A node dict under construction plus the fields already set (for duplicate checks)."""

    __slots__ = ("spec", "data", "seen", "tok")

    def __init__(self, spec: NodeSpec, tok: Token, **initial: Any) -> None:
        self.spec, self.tok, self.data, self.seen = spec, tok, dict(initial), set()


class _Parser:
    def __init__(self, text: str) -> None:
        self.lines = text.split("\n")
        self.toks = tokenize(text)
        self.i = 0
        self.where: dict[int, Token] = {}  # id(node dict) -> token of its statement

    # ---- token helpers --------------------------------------------------------

    def peek(self, k: int = 0) -> Token:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Token:
        tok = self.toks[self.i]
        if tok.kind != EOF:
            self.i += 1
        return tok

    def error(self, message: str, tok: Optional[Token] = None) -> EditScriptError:
        tok = tok or self.peek()
        source = self.lines[tok.line - 1] if 0 < tok.line <= len(self.lines) else ""
        return EditScriptError(message, tok.line, tok.col, source)

    def expect_punct(self, text: str, context: str) -> Token:
        tok = self.peek()
        if not tok.is_punct(text):
            raise self.error(f"expected '{text}' {context}, got {tok.describe()}")
        return self.next()

    def skip_nl(self) -> None:
        while self.peek().kind == NL:
            self.next()

    def end_statement(self) -> None:
        tok = self.peek()
        if tok.kind == NL:
            self.next()
        elif not (tok.is_punct("}") or tok.kind == EOF):
            raise self.error(f"expected end of statement, got {tok.describe()}")

    def new(self, spec: NodeSpec, tok: Token, **initial: Any) -> _Node:
        node = _Node(spec, tok, **initial)
        self.where[id(node.data)] = tok
        return node

    def set(self, node: _Node, name: str, value: Any, tok: Token) -> None:
        if name in node.seen:
            raise self.error(f"{node.spec.keyword}: field {name!r} is given twice", tok)
        node.seen.add(name)
        node.data[name] = value

    # ---- scalar readers ---------------------------------------------------------

    def number(self, what: str, units: tuple[str, ...] = ()) -> float:
        tok = self.peek()
        if tok.kind != NUMBER:
            raise self.error(f"expected {what}, got {tok.describe()}")
        if tok.unit and tok.unit.lower() not in {u.lower() for u in units}:
            allowed = " or ".join(repr(u) for u in units) if units else "no unit"
            raise self.error(f"unexpected unit {tok.unit!r} on {what} (allowed: {allowed})")
        self.next()
        return float(tok.value)

    def time(self) -> float:
        return self.number("a time in seconds (e.g. 1.25s)", ("s",))

    def integer(self, what: str) -> int:
        tok = self.peek()
        if tok.kind != NUMBER or tok.unit or not _INT_RE.match(tok.value):
            raise self.error(f"expected {what} (an integer), got {tok.describe()}")
        self.next()
        return int(tok.value)

    def word(self, what: str) -> Token:
        tok = self.peek()
        if tok.kind not in (IDENT, STRING):
            raise self.error(f"expected {what}, got {tok.describe()}")
        return self.next()

    def enum(self, node: _Node, name: str, tok: Token) -> str:
        """Check ``tok`` against the Literal values of field ``name``."""
        allowed = node.spec.literal_values(name)
        value = tok.value
        if allowed and value not in allowed:
            raise self.error(f"unknown {node.spec.keyword} {name} {value!r}{_suggest(value, allowed)}; "
                             f"expected one of: {', '.join(allowed)}", tok)
        return value

    def range_(self, node: _Node) -> None:
        tok = self.peek()
        if tok.kind != NUMBER:
            raise self.error(f"expected a time range like 0.00s..1.20s in {node.spec.keyword} statement, "
                             f"got {tok.describe()}")
        start = self.time()
        self.expect_punct("..", "in time range")
        end = self.time()
        self.set(node, "start", start, tok)
        self.set(node, "end", end, tok)

    def color(self) -> list[int]:
        tok = self.peek()
        if tok.kind == COLOR:
            self.next()
            return list(tok.value)
        if tok.kind == IDENT and tok.value == "rgb" and self.peek(1).is_punct("("):
            self.next()
            self.next()
            out = []
            for k in range(3):
                self.skip_nl()
                out.append(self.integer("a colour component"))
                self.skip_nl()
                self.expect_punct("," if k < 2 else ")", "in rgb(r, g, b)")
            return out
        raise self.error(f"expected a colour (#rrggbb or rgb(r, g, b)), got {tok.describe()}")

    def numlist(self, what: str) -> list[float]:
        self.expect_punct("[", f"to open {what}")
        out: list[float] = []
        while True:
            self.skip_nl()
            if self.peek().is_punct("]"):
                self.next()
                return out
            out.append(self.number(f"a number in {what}", ("s",)))
            self.skip_nl()
            if not self.peek().is_punct("]"):
                self.expect_punct(",", f"between items of {what}")

    # ---- JSON-like values -------------------------------------------------------

    def value(self) -> Any:
        tok = self.peek()
        if tok.kind == STRING:
            self.next()
            return tok.value
        if tok.kind == NUMBER:
            if tok.unit:
                raise self.error(f"units are not allowed in values (got {tok.value}{tok.unit})")
            self.next()
            return int(tok.value) if _INT_RE.match(tok.value) else float(tok.value)
        if tok.kind == IDENT:
            self.next()
            return {"true": True, "false": False, "null": None}.get(tok.value, tok.value)
        if tok.is_punct("["):
            self.next()
            items: list[Any] = []
            while True:
                self.skip_nl()
                if self.peek().is_punct("]"):
                    self.next()
                    return items
                items.append(self.value())
                self.skip_nl()
                if not self.peek().is_punct("]"):
                    self.expect_punct(",", "between list items")
        if tok.is_punct("{"):
            self.next()
            return self.mapping("}")
        raise self.error(f"expected a value (string, number, true/false/null, [list] or {{object}}), "
                         f"got {tok.describe()}")

    def mapping(self, close: str) -> dict[str, Any]:
        """``key: value, ...`` up to and including ``close`` (the opener is consumed)."""
        out: dict[str, Any] = {}
        while True:
            self.skip_nl()
            if self.peek().is_punct(close):
                self.next()
                return out
            key = self.word("a key (identifier or string)")
            sep = self.peek()
            if not (sep.is_punct(":") or sep.is_punct("=")):
                raise self.error(f"expected ':' or '=' after key {key.value!r}, got {sep.describe()}")
            self.next()
            self.skip_nl()
            if key.value in out:
                raise self.error(f"duplicate key {key.value!r}", key)
            out[key.value] = self.value()
            self.skip_nl()
            if not self.peek().is_punct(close):
                self.expect_punct(",", "between entries")

    # ---- clauses ------------------------------------------------------------------

    def clauses(self, node: _Node, block: bool) -> bool:
        """Parse clauses up to the end of the statement; True if a ``{`` block follows."""
        while True:
            tok = self.peek()
            if tok.kind in (NL, EOF) or tok.is_punct("}"):
                return False
            if tok.is_punct("{"):
                if not block:
                    raise self.error(f"'{node.spec.keyword}' statements take no {{ }} block")
                return True
            self.clause(node)

    def clause(self, node: _Node) -> None:
        tok, nxt = self.peek(), self.peek(1)
        idx = _INDEX[id(node.spec)]
        shapes = idx.shapes
        if tok.kind == IDENT and nxt.is_punct("="):
            return self.generic(node)
        if tok.kind == NUMBER and nxt.is_punct("->") and "arrow" in shapes:
            return self.apply(node, shapes["arrow"], tok)
        if tok.kind == NUMBER and tok.unit == "x" and "size" in shapes:
            return self.apply(node, shapes["size"], tok)
        if tok.kind == NUMBER and nxt.is_punct(":") and "ratio" in shapes:
            return self.apply(node, shapes["ratio"], tok)
        if tok.is_punct("@") and "at_fps" in shapes:
            return self.apply(node, shapes["at_fps"], tok)
        if tok.is_punct("[") and "list" in shapes:
            return self.apply(node, shapes["list"], tok)
        if tok.kind in (IDENT, STRING) and nxt.is_punct("->") and "link" in shapes:
            return self.apply(node, shapes["link"], tok)
        if tok.kind == STRING and "desc" in shapes:
            return self.apply(node, shapes["desc"], tok)
        if tok.kind == IDENT:
            if tok.value in idx.words:
                self.next()
                return self.apply(node, idx.words[tok.value], tok)
            if tok.value in idx.flags:
                self.next()
                cl, flag = idx.flags[tok.value]
                return self.set(node, cl.fields[0], flag, tok)
            if tok.value in idx.bare:
                self.next()
                return self.set(node, idx.bare[tok.value].fields[0], tok.value, tok)
            options = [*idx.words, *idx.flags, *idx.bare]
            hint = _suggest(tok.value, options + list(node.spec.model.model_fields))
            listed = ", ".join(options) if options else "none (use name=value)"
            raise self.error(f"unknown clause {tok.value!r} in '{node.spec.keyword}' statement{hint}; "
                             f"clauses: {listed}")
        raise self.error(f"unexpected {tok.describe()} in '{node.spec.keyword}' statement")

    def generic(self, node: _Node) -> None:
        name = self.next()
        self.next()  # '='
        fields = node.spec.model.model_fields
        if name.value == "type" or name.value not in fields or name.value in node.spec.children:
            settable = [f for f in fields if f != "type" and f not in node.spec.children]
            raise self.error(f"'{node.spec.keyword}' has no settable field {name.value!r}"
                             f"{_suggest(name.value, settable)}; fields: {', '.join(settable)}", name)
        self.set(node, name.value, self.value(), name)

    def apply(self, node: _Node, cl: Clause, tok: Token) -> None:
        """Read the value(s) of sugar clause ``cl`` (keyword already consumed) into ``node``."""
        k, f = cl.kind, cl.fields
        units = tuple(u for u in (cl.unit, *cl.accept) if u)
        if k == "time":
            values: list[Any] = [self.time()]
        elif k == "num":
            values = [self.number(f"a number after '{cl.word}'", units)]
        elif k == "int":
            values = [self.integer(f"a value after '{cl.word}'")]
        elif k == "word":
            w = self.word(f"a name after '{cl.word}'")
            values = [self.enum(node, f[0], w)]
        elif k == "str":
            s = self.peek()
            if s.kind != STRING:
                raise self.error(f"expected a string after '{cl.word}', got {s.describe()}")
            values = [self.next().value]
        elif k == "color":
            values = [self.color()]
        elif k == "vec":
            n = len(f) if len(f) > 1 else len(node.spec.default(f[0]))
            nums = [self.number(f"a number after '{cl.word}'")]
            for _ in range(n - 1):
                self.expect_punct(",", f"in '{cl.word}' ({n} comma-separated numbers)")
                nums.append(self.number(f"a number in '{cl.word}'"))
            values = [nums] if len(f) == 1 else nums
        elif k == "arrow":
            a = self.number("a start value", units)
            self.expect_punct("->", "between start and end values")
            values = [a, self.number("an end value after '->'", units)]
        elif k == "json":
            values = [self.value()]
        elif k == "frames":
            a = self.integer("a start frame")
            self.expect_punct("..", "in frame range")
            values = [a, self.integer("an end frame")]
        elif k == "span":
            a = self.time()
            self.expect_punct("..", "in time span")
            values = [a, self.time()]
        elif k == "link":
            ends = []
            for k_ in range(2):
                w = self.word("a shot id or none")
                ends.append(None if (w.kind == IDENT and w.value == "none") else w.value)
                if k_ == 0:
                    self.expect_punct("->", "between shot ids")
            values = ends
        elif k == "size":
            w = self.integer_with_unit("x")
            values = [w, self.integer("a height after 'x'")]
        elif k == "ratio":
            a = self.integer("an aspect numerator")
            self.expect_punct(":", "in aspect ratio")
            values = [f"{a}:{self.integer('an aspect denominator')}"]
        elif k == "at_fps":
            self.next()  # '@'
            values = [self.number("a frame rate after '@'", ("fps",))]
        elif k == "keys":
            values = [self.keyframes()]
        elif k == "list":
            values = [self.numlist("values")]
        elif k == "desc":
            values = [self.next().value]
        else:  # pragma: no cover - guarded by the printer's ClauseKind
            raise self.error(f"internal: unknown clause kind {k!r}", tok)
        for name, value in zip(f, values):
            self.set(node, name, value, tok)

    def integer_with_unit(self, unit: str) -> int:
        tok = self.peek()
        if tok.kind != NUMBER or tok.unit != unit or not _INT_RE.match(tok.value):
            raise self.error(f"expected a size like 1080x1920, got {tok.describe()}")
        self.next()
        return int(tok.value)

    def keyframes(self) -> list[dict[str, float]]:
        self.expect_punct("[", "to open keyframes")
        out: list[dict[str, float]] = []
        while True:
            self.skip_nl()
            if self.peek().is_punct("]"):
                self.next()
                return out
            t = self.time()
            self.expect_punct(":", "between keyframe time and value")
            out.append({"t": t, "v": self.number("a keyframe value")})
            self.skip_nl()
            if not self.peek().is_punct("]"):
                self.expect_punct(",", "between keyframes")

    # ---- statements -------------------------------------------------------------------

    def block(self, opener: Token, handlers: dict[str, Callable[[Token], None]], what: str) -> None:
        self.expect_punct("{", f"to open the {what} block")
        while True:
            self.skip_nl()
            tok = self.peek()
            if tok.is_punct("}"):
                self.next()
                return
            if tok.kind == EOF:
                raise self.error(f"unclosed {what} block opened at line {opener.line}:{opener.col}")
            if tok.kind != IDENT or tok.value not in handlers:
                name = tok.value if tok.kind == IDENT else tok.describe()
                hint = _suggest(tok.value, handlers) if tok.kind == IDENT else ""
                raise self.error(f"unknown statement {name!r} in {what} block{hint}; "
                                 f"expected one of: {', '.join(handlers)}")
            self.next()
            handlers[tok.value](tok)
            self.end_statement()

    def once(self, holder: _Node, name: str, tok: Token) -> None:
        if name in holder.seen:
            raise self.error(f"'{tok.value}' may appear only once here", tok)
        holder.seen.add(name)

    def parse(self) -> Edit:
        self.skip_nl()
        tok = self.peek()
        if not (tok.kind == IDENT and tok.value == "edit"):
            raise self.error(f"an EditScript must start with 'edit', got {tok.describe()}")
        self.next()
        root = self.new(SPECS["edit"], tok, type="edit", id="edit")
        media = self.new(SPECS["media"], tok)
        timeline = self.new(SPECS["timeline"], tok, type="timeline", id="timeline", sections=[])
        root.data.update(media=media.data, timeline=timeline.data, overlays=[], patterns=[])
        if self.peek().kind == STRING:
            media.data["source"] = self.next().value
        if not self.clauses(root, block=True):
            raise self.error("expected '{' to open the edit block")

        def media_stmt(t: Token) -> None:
            self.once(root, "media", t)
            self.where[id(media.data)] = t
            self.clauses(media, block=False)

        def music_stmt(t: Token) -> None:
            self.once(root, "music", t)
            root.data["music"] = self.music(t)

        def timeline_stmt(t: Token) -> None:
            self.once(root, "timeline", t)
            self.where[id(timeline.data)] = t
            self.clauses(timeline, block=False)

        def stats_stmt(t: Token) -> None:
            self.once(root, "stats", t)
            root.data["stats"] = self.stats(t)

        def provenance_stmt(t: Token) -> None:
            self.once(root, "provenance", t)
            if not self.peek().is_punct("{"):
                raise self.error(f"expected '{{' after 'provenance', got {self.peek().describe()}")
            root.data["provenance"] = self.value()

        self.block(tok, {
            "media": media_stmt,
            "music": music_stmt,
            "timeline": timeline_stmt,
            "section": lambda t: timeline.data["sections"].append(self.section(t)),
            "text": lambda t: root.data["overlays"].append(self.text(t)),
            "pattern": lambda t: root.data["patterns"].append(self.pattern(t)),
            "stats": stats_stmt,
            "provenance": provenance_stmt,
        }, "edit")
        self.skip_nl()
        if self.peek().kind != EOF:
            raise self.error(f"unexpected {self.peek().describe()} after the closing '}}' of the edit")
        return self.validate(root.data)

    def music(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["music"], tok, type="music", id="music")
        if not self.clauses(node, block=True):
            return node.data

        def numlist(t: Token) -> None:
            self.once(node, t.value, t)
            node.data[t.value] = self.numlist(t.value)

        def energy(t: Token) -> None:
            self.once(node, "energy", t)
            node.data["energy"] = self.curve(t)

        drops: list[dict[str, Any]] = node.data.setdefault("drops", [])
        sections: list[dict[str, Any]] = node.data.setdefault("sections", [])
        self.block(tok, {
            "beats": numlist, "downbeats": numlist, "onsets": numlist,
            "drop": lambda t: drops.append(self.drop(t)),
            "section": lambda t: sections.append(self.audio_section(t)),
            "energy": energy,
        }, "music")
        return node.data

    def curve(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["curve"], tok)
        self.clauses(node, block=False)
        return node.data

    def drop(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["drop"], tok)
        self.set(node, "time", self.time(), tok)
        self.clauses(node, block=False)
        return node.data

    def kind_head(self, node: _Node, kind_field: str) -> None:
        """``[ID] KIND range`` -- the id is present when two words precede the range."""
        first = self.word(f"a {kind_field} after '{node.spec.keyword}'")
        if self.peek().kind in (IDENT, STRING):
            self.set(node, "id", first.value, first)
            first = self.next()
        self.set(node, kind_field, self.enum(node, kind_field, first), first)
        self.range_(node)

    def audio_section(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["audio_section"], tok, type="audio_section")
        self.kind_head(node, "label")
        self.clauses(node, block=False)
        return node.data

    def section(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["section"], tok, type="section", items=[])
        self.kind_head(node, "role")
        if self.clauses(node, block=True):
            items = node.data["items"]
            self.block(tok, {"shot": lambda t: items.append(self.shot(t)),
                             "transition": lambda t: items.append(self.transition(t))}, "section")
        return node.data

    def opt_id(self, node: _Node) -> None:
        if self.peek().kind in (IDENT, STRING) and self.peek(1).kind == NUMBER:
            tok = self.next()
            self.set(node, "id", tok.value, tok)

    def shot(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["shot"], tok, type="shot", effects=[])
        self.opt_id(node)
        self.range_(node)
        if self.clauses(node, block=True):
            effects = node.data["effects"]

            def attr(t: Token) -> None:
                self.once(node, t.value, t)
                sub = self.new(SPECS[t.value], t)
                self.clauses(sub, block=False)
                node.data[t.value] = sub.data

            def effect(t: Token) -> None:
                sub = self.new(EFFECT_SPECS[t.value], t, type=t.value)
                self.opt_id(sub)
                self.range_(sub)
                self.clauses(sub, block=False)
                effects.append(sub.data)

            handlers: dict[str, Callable[[Token], None]] = {"look": attr, "layout": attr, "motion": attr}
            handlers.update({name: effect for name in EFFECT_TYPES})
            self.block(tok, handlers, "shot")
        if "index" not in node.seen:
            node.data["index"] = derived_shot_index(node.data.get("id", ""))
        return node.data

    def transition(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["transition"], tok, type="transition")
        self.kind_head(node, "kind")
        self.clauses(node, block=False)
        return node.data

    def text(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["text"], tok, type="text", words=[])
        if self.peek().kind == IDENT:
            ident = self.next()
            self.set(node, "id", ident.value, ident)
        if self.peek().kind == STRING:
            s = self.next()
            self.set(node, "text", s.value, s)
        self.range_(node)
        if self.clauses(node, block=True):
            self.block(tok, {"word": lambda t: node.data["words"].append(self.word_stmt(t))}, "text")
        return node.data

    def word_stmt(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["word"], tok)
        s = self.peek()
        if s.kind != STRING:
            raise self.error(f"expected the word's text as a string, got {s.describe()}")
        self.set(node, "text", self.next().value, s)
        self.range_(node)
        self.clauses(node, block=False)
        return node.data

    def pattern(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["pattern"], tok, type="pattern")
        first = self.word("a pattern rule like cut_every_n_beats(n=2)")
        if not self.peek().is_punct("("):
            if first.kind != IDENT:
                raise self.error(f"expected '(' after pattern rule, got {self.peek().describe()}")
            self.set(node, "id", first.value, first)
            first = self.word("a pattern rule after the pattern id")
        self.set(node, "rule", first.value, first)
        self.expect_punct("(", f"after pattern rule {first.value!r} (write {first.value}() for no params)")
        self.set(node, "params", self.mapping(")"), first)
        self.clauses(node, block=False)
        return node.data

    def stats(self, tok: Token) -> dict[str, Any]:
        node = self.new(SPECS["stats"], tok)
        if self.clauses(node, block=True):
            def pacing(t: Token) -> None:
                self.once(node, "pacing", t)
                node.data["pacing"] = self.curve(t)

            self.block(tok, {"pacing": pacing}, "stats")
        return node.data

    # ---- validation -----------------------------------------------------------------------

    def validate(self, data: dict[str, Any]) -> Edit:
        try:
            return Edit.model_validate(data)
        except ValidationError as exc:
            err = exc.errors()[0]
            loc = err.get("loc", ())
            tok = self.locate(data, loc)
            path = ".".join(str(p) for p in loc)
            raise self.error(f"invalid value for {path}: {err.get('msg', 'validation error')}", tok) from None

    def locate(self, data: Any, loc: tuple[Any, ...]) -> Token:
        """Token of the innermost parsed node on the pydantic error path ``loc``."""
        best = self.where.get(id(data), self.toks[0])
        cur = data
        for part in loc:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            elif isinstance(cur, list) and isinstance(part, int) and 0 <= part < len(cur):
                cur = cur[part]
            else:
                continue
            best = self.where.get(id(cur), best)
        return best


def parse_dsl(text: str) -> Edit:
    """Parse EditScript source into an :class:`Edit`.

    Raises :class:`EditScriptError` (with ``line``/``col``) on any lexical,
    syntax or validation error.
    """
    return _Parser(text).parse()
