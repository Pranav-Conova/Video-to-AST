"""Edit AST — typed node definitions for a reverse-engineered video edit.

An edit is modelled as a program:

    Edit
    ├── media        MediaInfo          (container facts: size, fps, duration)
    ├── music        Music              (tempo, beat grid, drops, audio sections)
    ├── timeline     Timeline
    │   └── sections Section*           (hook / build / drop / outro ...)
    │       └── items (Shot | Transition)*   in playback order
    │           Shot.effects  Effect*   (zoom, pan, rotate, shake, speed, freeze,
    │                                    flash, blur, invert, rgb_split)
    ├── overlays     TextOverlay*       (burned-in captions / titles)
    ├── patterns     Pattern*           (inferred editing rules, e.g. "cut every 2 beats")
    └── stats        EditStats          (pacing summary)

Conventions (every producer and consumer of the AST must follow these):

* All times are absolute seconds on the edit timeline (float). A frame ``f`` at
  analysis rate ``media.analysis_fps`` covers ``[f / fps, (f + 1) / fps)``.
* Shots partition the timeline contiguously: ``shot[i].end == shot[i+1].start``.
  ``Shot.frame_start`` / ``frame_end`` are analysis-frame indices, end exclusive.
* A Transition sits between two shots in ``Section.items``. Its span may overlap
  the tail of the previous shot and the head of the next (dissolves, flashes).
  A hard cut has ``start == end == boundary time``. Fade-in at the very start has
  ``from_shot=None``; fade-out at the very end has ``to_shot=None``.
* Effect spans are absolute seconds and lie within their parent shot.
* Spatial quantities are normalised: translations / sizes are fractions of the
  frame width (x) or height (y); bboxes are ``(x, y, w, h)`` in 0..1.
* Zoom ``scale_*``: 1.0 = native framing, >1 = magnified (zoomed in). Analyzers
  only observe relative change, so they report ``scale_from=1.0`` and
  ``scale_to`` = end/start ratio.
* Pan ``dx/dy``: total displacement of image *content* over the span
  (positive = content moves right / down).
* Rotate angles are degrees, positive = counter-clockwise (OpenCV convention).
* Shake ``amplitude``: peak per-axis offset as a fraction of frame width;
  ``frequency`` is the dominant jitter frequency in Hz.
* Speed ``rate``: playback speed relative to real time (0.5 = 2x slow motion).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

# --------------------------------------------------------------------------- enums

Easing = Literal["linear", "ease_in", "ease_out", "ease_in_out", "punch", "step"]
TransitionKind = Literal[
    "cut", "flash", "fade_black", "fade_white", "dissolve",
    "whip", "zoom", "spin", "glitch", "slide", "unknown",
]
SectionRole = Literal["hook", "intro", "build", "drop", "body", "break", "outro"]
SpeedKind = Literal["slowmo", "fast", "ramp_up", "ramp_down", "ramp", "constant"]
TextPosition = Literal["top", "upper", "center", "lower", "bottom"]
TextAnimation = Literal["static", "pop", "word_by_word", "typewriter", "fade", "unknown"]
TextRole = Literal["caption", "title", "hook", "label", "watermark", "unknown"]
AudioLabel = Literal["intro", "build", "drop", "break", "outro", "body", "silence"]
CameraMove = Literal["static", "pan", "zoom", "rotate", "handheld", "complex"]
SplitKind = Literal["none", "vertical", "horizontal"]

RGB = tuple[int, int, int]
BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------- base


class Node(BaseModel):
    """Base of every AST node. ``id`` is assigned by :func:`assign_ids`."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    confidence: float = 1.0

    def iter_children(self) -> Iterator["Node"]:
        """Direct child nodes, in field order."""
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, Node):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Node):
                        yield item

    def walk(self) -> Iterator["Node"]:
        """Pre-order traversal including ``self``."""
        yield self
        for child in self.iter_children():
            yield from child.walk()

    @property
    def node_type(self) -> str:
        return getattr(self, "type", type(self).__name__.lower())


class Timed(Node):
    """A node occupying ``[start, end)`` seconds on the edit timeline."""

    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, start: float, end: float) -> float:
        """Seconds of overlap with ``[start, end)``."""
        return max(0.0, min(self.end, end) - max(self.start, start))

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


class Keyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t: float  # absolute seconds
    v: float


class Curve(BaseModel):
    """A uniformly sampled signal: value[i] is at time ``t0 + i * dt``."""

    model_config = ConfigDict(extra="forbid")
    t0: float = 0.0
    dt: float = 0.1
    values: list[float] = Field(default_factory=list)


# --------------------------------------------------------------------------- effects


class EffectBase(Timed):
    easing: Easing = "linear"
    keyframes: list[Keyframe] = Field(default_factory=list)  # optional sampled curve
    params: dict[str, Any] = Field(default_factory=dict)  # analyzer-specific extras


class Zoom(EffectBase):
    type: Literal["zoom"] = "zoom"
    scale_from: float = 1.0
    scale_to: float = 1.0
    center: tuple[float, float] = (0.5, 0.5)
    digital: Optional[bool] = None  # True = clean post zoom, False = optical/camera move


class Pan(EffectBase):
    type: Literal["pan"] = "pan"
    dx: float = 0.0
    dy: float = 0.0


class Rotate(EffectBase):
    type: Literal["rotate"] = "rotate"
    angle_from: float = 0.0
    angle_to: float = 0.0


class Shake(EffectBase):
    type: Literal["shake"] = "shake"
    amplitude: float = 0.0
    frequency: float = 0.0
    rotation_amplitude: float = 0.0  # degrees


class Speed(EffectBase):
    type: Literal["speed"] = "speed"
    kind: SpeedKind = "constant"
    rate_from: float = 1.0
    rate_to: float = 1.0


class Freeze(EffectBase):
    type: Literal["freeze"] = "freeze"


class Flash(EffectBase):
    type: Literal["flash"] = "flash"
    color: RGB = (255, 255, 255)
    intensity: float = 1.0  # 0..1 blend toward color at peak


class Blur(EffectBase):
    type: Literal["blur"] = "blur"
    amount: float = 0.0  # gaussian sigma as fraction of frame width


class Invert(EffectBase):
    type: Literal["invert"] = "invert"


class RGBSplit(EffectBase):
    type: Literal["rgb_split"] = "rgb_split"
    offset: float = 0.0  # channel offset as fraction of frame width


Effect = Annotated[
    Union[Zoom, Pan, Rotate, Shake, Speed, Freeze, Flash, Blur, Invert, RGBSplit],
    Field(discriminator="type"),
]
EFFECT_TYPES: tuple[str, ...] = (
    "zoom", "pan", "rotate", "shake", "speed", "freeze", "flash", "blur", "invert", "rgb_split",
)


# --------------------------------------------------------------------------- shot attributes


class Look(BaseModel):
    """Measured visual look of a shot (all 0..1 unless noted)."""

    model_config = ConfigDict(extra="forbid")
    brightness: float = 0.5  # mean luma
    contrast: float = 0.2  # luma std
    saturation: float = 0.3  # mean HSV saturation
    temperature: float = 0.0  # (R-B)/(R+B), -1 cool .. +1 warm
    tint: float = 0.0  # G - (R+B)/2, -1..1
    bw: bool = False  # black & white / desaturated grade
    sharpness: float = 0.0  # normalised Laplacian variance (0 = very blurry)
    vignette: float = 0.0  # 0 = none, 1 = strong darkening at the corners
    dominant_color: RGB = (128, 128, 128)


class Layout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    letterbox_top: float = 0.0  # fraction of height covered by a flat bar
    letterbox_bottom: float = 0.0
    pillarbox_left: float = 0.0  # fraction of width
    pillarbox_right: float = 0.0
    blur_pad: bool = False  # content sits on a blurred copy of itself
    content_box: BBox = (0.0, 0.0, 1.0, 1.0)
    split: SplitKind = "none"


class MotionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    magnitude: float = 0.0  # mean per-frame motion, fraction of width per frame
    camera: CameraMove = "static"
    zoom_ratio: float = 1.0  # cumulative scale change over the shot
    pan: tuple[float, float] = (0.0, 0.0)  # cumulative content displacement
    rotation: float = 0.0  # cumulative degrees


# --------------------------------------------------------------------------- timeline nodes


class Shot(Timed):
    type: Literal["shot"] = "shot"
    index: int = 0
    frame_start: int = 0
    frame_end: int = 0  # exclusive
    beats: Optional[float] = None  # duration measured in beats
    on_beat: Optional[bool] = None  # does the shot start on a beat?
    beat_offset: Optional[float] = None  # signed seconds from nearest beat at start
    source_group: Optional[str] = None  # shots that look like the same source clip share a label
    look: Optional[Look] = None
    layout: Optional[Layout] = None
    motion: Optional[MotionSummary] = None
    effects: list[Effect] = Field(default_factory=list)


class Transition(Timed):
    type: Literal["transition"] = "transition"
    kind: TransitionKind = "cut"
    from_shot: Optional[str] = None  # shot id
    to_shot: Optional[str] = None  # shot id
    color: Optional[RGB] = None  # flash / dip colour
    on_beat: Optional[bool] = None
    beat_offset: Optional[float] = None
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def time(self) -> float:
        """Representative boundary time (midpoint of the span)."""
        return 0.5 * (self.start + self.end)


TimelineItem = Annotated[Union[Shot, Transition], Field(discriminator="type")]


class Section(Timed):
    type: Literal["section"] = "section"
    role: SectionRole = "body"
    cuts_per_second: float = 0.0
    mean_shot_duration: float = 0.0
    energy: Optional[float] = None  # mean audio energy 0..1 (relative to the track)
    items: list[TimelineItem] = Field(default_factory=list)

    def shots(self) -> list[Shot]:
        return [i for i in self.items if isinstance(i, Shot)]

    def transitions(self) -> list[Transition]:
        return [i for i in self.items if isinstance(i, Transition)]


class Timeline(Node):
    type: Literal["timeline"] = "timeline"
    sections: list[Section] = Field(default_factory=list)


# --------------------------------------------------------------------------- audio


class Drop(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time: float
    strength: float = 1.0  # relative energy jump


class AudioSection(Timed):
    type: Literal["audio_section"] = "audio_section"
    label: AudioLabel = "body"
    energy: float = 0.0  # mean energy 0..1 relative to the track


class Music(Node):
    type: Literal["music"] = "music"
    bpm: Optional[float] = None
    beat_offset: Optional[float] = None  # time of the first beat
    meter: int = 4
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    onsets: list[float] = Field(default_factory=list)  # strong hits / accents
    drops: list[Drop] = Field(default_factory=list)
    sections: list[AudioSection] = Field(default_factory=list)
    energy: Optional[Curve] = None  # RMS energy, normalised 0..1
    has_speech: Optional[bool] = None


# --------------------------------------------------------------------------- overlays


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    start: float
    end: float


class TextOverlay(Timed):
    type: Literal["text"] = "text"
    text: Optional[str] = None  # OCR result when available
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    position: TextPosition = "center"
    size: float = 0.0  # text line height as fraction of frame height
    color: Optional[RGB] = None
    outline: Optional[bool] = None
    animation: TextAnimation = "static"
    role: TextRole = "unknown"
    words: list[Word] = Field(default_factory=list)


# --------------------------------------------------------------------------- derived


class Pattern(Node):
    """An inferred editing rule, e.g. ``cut_every_n_beats`` with ``params={'n': 2}``."""

    type: Literal["pattern"] = "pattern"
    rule: str
    description: str = ""
    start: Optional[float] = None
    end: Optional[float] = None
    support: float = 0.0  # fraction of eligible instances that follow the rule
    params: dict[str, Any] = Field(default_factory=dict)


class EditStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_shots: int = 0
    n_transitions: int = 0  # non-"cut" transitions
    mean_shot_duration: float = 0.0
    median_shot_duration: float = 0.0
    min_shot_duration: float = 0.0
    max_shot_duration: float = 0.0
    cuts_per_minute: float = 0.0
    cut_on_beat_ratio: Optional[float] = None
    hook_duration: float = 0.0
    first_cut_time: Optional[float] = None
    effect_counts: dict[str, int] = Field(default_factory=dict)
    transition_counts: dict[str, int] = Field(default_factory=dict)
    text_coverage: float = 0.0  # fraction of runtime with on-screen text
    pacing: Optional[Curve] = None  # cuts per second over time


class MediaInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = ""
    duration: float = 0.0
    fps: float = 30.0  # source frame rate
    width: int = 0  # display width (after rotation)
    height: int = 0
    n_frames: int = 0  # analysis frames
    analysis_fps: float = 30.0  # rate that Shot.frame_* refer to
    has_audio: bool = False
    aspect: str = ""  # e.g. "9:16"


class Edit(Node):
    type: Literal["edit"] = "edit"
    schema_version: str = SCHEMA_VERSION
    media: MediaInfo = Field(default_factory=MediaInfo)
    music: Optional[Music] = None
    timeline: Timeline = Field(default_factory=Timeline)
    overlays: list[TextOverlay] = Field(default_factory=list)
    patterns: list[Pattern] = Field(default_factory=list)
    stats: Optional[EditStats] = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    # ---- queries ---------------------------------------------------------------

    def shots(self) -> list[Shot]:
        return [s for sec in self.timeline.sections for s in sec.shots()]

    def transitions(self) -> list[Transition]:
        return [t for sec in self.timeline.sections for t in sec.transitions()]

    def effects(self) -> list[tuple[Shot, EffectBase]]:
        return [(s, e) for s in self.shots() for e in s.effects]

    def find(self, node_id: str) -> Optional[Node]:
        for node in self.walk():
            if node.id == node_id:
                return node
        return None

    def cut_times(self) -> list[float]:
        """Shot boundary times (start of every shot after the first)."""
        return [s.start for s in self.shots()[1:]]

    # ---- serialisation ---------------------------------------------------------

    def to_dict(self, precision: int = 4) -> dict[str, Any]:
        return _round_floats(self.model_dump(mode="json"), precision)

    def to_json(self, indent: int | None = 2, precision: int = 4) -> str:
        return json.dumps(self.to_dict(precision), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Edit":
        return cls.model_validate_json(text)

    def save(self, path: str | Path, precision: int = 4) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(precision=precision), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Edit":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


# --------------------------------------------------------------------------- helpers


def shot_id(index: int) -> str:
    """Canonical id of the shot with 0-based ``index`` (``s1``, ``s2`` ...)."""
    return f"s{index + 1}"


def assign_ids(edit: Edit) -> Edit:
    """Assign deterministic ids to every node (idempotent).

    Shots get ``shot_id(shot.index)``; transitions ``t1..``; sections ``sec1..``;
    effects ``<shot id>.<type><k>``; overlays ``txt1..``; patterns ``p1..``.
    Transition ``from_shot``/``to_shot`` must already hold canonical shot ids.
    """
    edit.id = edit.id or "edit"
    edit.timeline.id = "timeline"
    t_count = 0
    for si, sec in enumerate(edit.timeline.sections, 1):
        sec.id = f"sec{si}"
        for item in sec.items:
            if isinstance(item, Shot):
                item.id = shot_id(item.index)
                counts: dict[str, int] = {}
                for eff in item.effects:
                    counts[eff.type] = counts.get(eff.type, 0) + 1
                    eff.id = f"{item.id}.{eff.type}{counts[eff.type]}"
            else:
                t_count += 1
                item.id = f"t{t_count}"
    if edit.music is not None:
        edit.music.id = "music"
        for ai, sec in enumerate(edit.music.sections, 1):
            sec.id = f"aud{ai}"
    for oi, ov in enumerate(edit.overlays, 1):
        ov.id = f"txt{oi}"
    for pi, pat in enumerate(edit.patterns, 1):
        pat.id = f"p{pi}"
    return edit


class NodeVisitor:
    """Dispatches ``visit_<type>(node)``; falls back to ``generic_visit``.

    Mirrors :class:`ast.NodeVisitor`: override e.g. ``visit_shot`` or
    ``visit_zoom`` and call ``self.generic_visit(node)`` to keep descending.
    """

    def visit(self, node: Node) -> Any:
        method = getattr(self, f"visit_{node.node_type}", self.generic_visit)
        return method(node)

    def generic_visit(self, node: Node) -> None:
        for child in node.iter_children():
            self.visit(child)


def _round_floats(obj: Any, precision: int) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, precision) for v in obj]
    return obj
