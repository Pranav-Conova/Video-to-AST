"""Tests for the EditScript language: lossless printing/parsing, compact mode, errors."""

from __future__ import annotations

import random
import re
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel

from editast.ast import (
    AudioSection, Blur, Curve, Drop, Edit, EditStats, Flash, Freeze, Invert, Keyframe, Layout, Look,
    MediaInfo, MotionSummary, Music, Pan, Pattern, RGBSplit, Rotate, Section, Shake, Shot, Speed,
    TextOverlay, Timeline, Transition, Word, Zoom, assign_ids,
)
from editast.lang.parser import EditScriptError, parse_dsl, tokenize
from editast.lang.printer import to_dsl


def roundtrip_ok(edit: Edit) -> str:
    """Assert the lossless contract and return the printed source."""
    text = to_dsl(edit)
    expected = Edit.from_json(edit.to_json())
    got = parse_dsl(text)
    assert got == expected, f"round trip mismatch\n{text}\n---\n{got.to_json()}\n---\n{expected.to_json()}"
    assert to_dsl(got) == text  # printing is deterministic / idempotent
    return text


# --------------------------------------------------------------------------- a rich hand-built edit

NESTED = {"n": 2, "ratio": 0.5, "flags": [True, False, None], "deep": {"a b": [1, 2.5, {"x": "y\n\"z\""}]},
          "empty": {}, "nil": None, "neg": -3, "big": 12345678901234}


def rich_edit() -> Edit:
    kf = [Keyframe(t=0.0, v=1.0), Keyframe(t=0.25, v=1.12), Keyframe(t=0.5, v=1.3)]
    effects = [
        Zoom(start=0.0, end=0.5, scale_from=1.0, scale_to=1.3, center=(0.4, 0.6), digital=True,
             easing="ease_out", keyframes=kf, params={"method": "flow", "nested": NESTED}, confidence=0.8),
        Pan(start=0.0, end=0.5, dx=0.12, dy=-0.05, easing="ease_in"),
        Rotate(start=0.1, end=0.4, angle_from=-5.0, angle_to=15.5, easing="punch"),
        Shake(start=0.2, end=0.5, amplitude=0.02, frequency=12.5, rotation_amplitude=1.5, easing="step"),
        Speed(start=0.5, end=1.0, kind="ramp_down", rate_from=1.0, rate_to=0.3, easing="ease_in_out",
              keyframes=[Keyframe(t=0.5, v=1.0), Keyframe(t=1.0, v=0.3)]),
        Freeze(start=0.9, end=1.0, confidence=0.6),
        Flash(start=0.5, end=0.55, color=(255, 240, 0), intensity=0.7),
        Blur(start=0.6, end=0.7, amount=0.01, params={"kind": "motion"}),
        Invert(start=0.7, end=0.75),
        RGBSplit(start=0.75, end=0.8, offset=0.004),
        Zoom(start=0.8, end=1.0, scale_to=0.9, digital=False),
    ]
    s0 = Shot(index=0, start=0.0, end=1.0, frame_start=0, frame_end=30, beats=2.0, on_beat=True,
              beat_offset=0.012, source_group="A", confidence=0.9,
              look=Look(brightness=0.61, contrast=0.25, saturation=0.45, temperature=0.12, tint=-0.03,
                        bw=True, sharpness=0.33, vignette=0.4, dominant_color=(200, 100, 50)),
              layout=Layout(letterbox_top=0.12, letterbox_bottom=0.1, pillarbox_left=0.05,
                            pillarbox_right=0.06, blur_pad=True, content_box=(0.05, 0.12, 0.89, 0.78),
                            split="vertical"),
              motion=MotionSummary(magnitude=0.012, camera="handheld", zoom_ratio=1.05, pan=(0.1, -0.02),
                                   rotation=2.5),
              effects=effects)
    s1 = Shot(index=1, start=1.0, end=1.5, frame_start=30, frame_end=45, beats=1.0, on_beat=False,
              beat_offset=-0.03, look=Look(), layout=Layout(), motion=MotionSummary())
    s2 = Shot(index=7, start=1.5, end=2.0, frame_start=45, frame_end=60)  # index != canonical id
    t_in = Transition(kind="fade_black", start=0.0, end=0.2, from_shot=None, to_shot="s1")
    t1 = Transition(kind="flash", start=0.95, end=1.05, from_shot="s1", to_shot="s2", color=(255, 255, 255),
                    on_beat=True, beat_offset=0.004, params={"peak": 0.98, "frames": [29, 30, 31]},
                    confidence=0.7)
    t2 = Transition(kind="cut", start=1.5, end=1.5, from_shot="s2", to_shot="s8", on_beat=False)
    t_out = Transition(kind="fade_white", start=1.8, end=2.0, from_shot="s8", to_shot=None,
                       color=(300, -1, 12))
    sections = [
        Section(role="hook", start=0.0, end=1.0, cuts_per_second=1.0, mean_shot_duration=1.0, energy=0.3,
                items=[t_in, s0, t1], confidence=0.95),
        Section(role="drop", start=1.0, end=2.0, cuts_per_second=2.0, mean_shot_duration=0.5,
                items=[s1, t2, s2, t_out]),
        Section(role="outro", start=2.0, end=2.0),
    ]
    music = Music(
        bpm=128.0, beat_offset=0.05, meter=3, beats=[0.05, 0.52, 0.99, 1.46, 1.93],
        downbeats=[0.05, 1.46], onsets=[0.05, 0.99], drops=[Drop(time=1.0, strength=2.5), Drop(time=1.9)],
        sections=[AudioSection(start=0.0, end=1.0, label="build", energy=0.3),
                  AudioSection(start=1.0, end=2.0, label="drop", energy=0.9, confidence=0.5)],
        energy=Curve(t0=0.0, dt=0.25, values=[0.1, 0.2, 0.4, 0.5, 0.9, 1.0, 0.95, 0.8]),
        has_speech=False, confidence=0.85,
    )
    overlays = [
        TextOverlay(start=0.0, end=1.0, text='WAIT FOR IT "😱"', bbox=(0.1, 0.4, 0.8, 0.08), position="upper",
                    size=0.05, color=(255, 255, 255), outline=True, animation="word_by_word", role="hook",
                    words=[Word(text="WAIT", start=0.0, end=0.3), Word(text="FOR", start=0.3, end=0.6),
                           Word(text='IT "😱"', start=0.6, end=1.0)], confidence=0.75),
        TextOverlay(start=1.0, end=2.0, text=None, outline=False, animation="pop", role="caption"),
        TextOverlay(start=1.2, end=1.4, text="", position="bottom", role="watermark"),
    ]
    patterns = [
        Pattern(rule="cut_every_n_beats", description="Cuts land every 2 beats", start=0.0, end=2.0,
                support=0.83, params={"n": 2, "nested": NESTED}, confidence=0.9),
        Pattern(rule="zoom on drop", description="", start=1.0, support=1.0),
        Pattern(rule="flash_on_downbeat", end=1.5),
    ]
    stats = EditStats(n_shots=3, n_transitions=3, mean_shot_duration=0.6667, median_shot_duration=0.5,
                      min_shot_duration=0.5, max_shot_duration=1.0, cuts_per_minute=60.0, cut_on_beat_ratio=0.5,
                      hook_duration=1.0, first_cut_time=1.0, effect_counts={"zoom": 2, "pan": 1, "rgb_split": 1},
                      transition_counts={"cut": 1, "flash": 1, "fade_black": 1}, text_coverage=0.6,
                      pacing=Curve(t0=0.0, dt=1.0, values=[1.0, 2.0]))
    edit = Edit(
        media=MediaInfo(source="C:\\clips\\velocity edit #1.mp4", duration=2.0, fps=29.97, width=1080,
                        height=1920, n_frames=60, analysis_fps=30.0, has_audio=True, aspect="9:16"),
        music=music, timeline=Timeline(sections=sections), overlays=overlays, patterns=patterns, stats=stats,
        provenance={"editast": "0.1.0", "analyzers": {"shots": {"threshold": 0.35}}, "timings": [0.1, 2.5]},
        confidence=0.66,
    )
    return assign_ids(edit)


def test_rich_edit_roundtrip_covers_every_node_type():
    edit = rich_edit()
    text = roundtrip_ok(edit)
    types_seen = {n.node_type for n in edit.walk()}
    assert {"edit", "timeline", "section", "shot", "transition", "music", "audio_section", "text",
            "pattern", "zoom", "pan", "rotate", "shake", "speed", "freeze", "flash", "blur", "invert",
            "rgb_split"} <= types_seen
    # Human-friendly sugar is used for the common fields.
    for snippet in ("media 1080x1920 9:16 @29.97fps duration 2.00s", "transition t2 flash 0.95s..1.05s s1->s2",
                    "none->s1", "s8->none", "rgb(300, -1, 12)", "1.0->1.3 ease_out", "keys [0.00: 1.0",
                    "on_beat", "off_beat", "group A", "shot s8 1.50s..2.00s", "drop 1.00s strength 2.5",
                    "cut_every_n_beats(n=2", "speed s1.speed1 0.50s..1.00s ramp_down 1.0x->0.3x",
                    "energy t0 0.00s dt 0.25s [0.1, 0.2", "pacing t0", "provenance {"):
        assert snippet in text, snippet


def test_assigned_ids_are_positional_and_dsl_defaults_apply():
    edit = rich_edit()
    text = to_dsl(edit)
    assert text.splitlines()[1].startswith('edit "C:\\\\clips\\\\velocity edit #1.mp4" conf 0.66 {')
    assert "timeline" not in text.split("section", 1)[0]  # canonical timeline id is implicit
    unassigned = Edit.model_validate_json(edit.to_json())
    for node in unassigned.walk():
        node.id = ""
    text2 = roundtrip_ok(unassigned)
    music_line = next(line for line in text2.splitlines() if line.strip().startswith("music "))
    assert 'id="" {' in text2.splitlines()[1] and 'timeline id=""' in text2 and 'id=""' in music_line
    # With non-canonical ids the shot index can no longer be derived, so it is written out.
    assert "shot 1.50s..2.00s index 7 frames 45..60" in text2


def test_strings_with_quotes_newlines_unicode_and_odd_ids():
    weird = 'He said "go"\n\ttab \\ back\x00slash \u2028 sep \x7f del 日本語 🚗💨 #ffffff {[(,;:=)]}'
    edit = Edit(
        media=MediaInfo(source=weird, aspect="1.85:1"),
        timeline=Timeline(sections=[Section(id="sec 1", role="body", items=[
            Shot(id="shot-1", index=4, source_group=weird, start=0.0, end=1.0,
                 effects=[Flash(id="", start=0.0, end=0.1)]),
            Transition(id="t#1", from_shot="none", to_shot="shot-1", kind="glitch"),
        ])]),
        overlays=[TextOverlay(id="none", text=weird, words=[Word(text=weird, start=0.0, end=1.0)])],
        patterns=[Pattern(id="p x", rule=weird, description=weird, params={weird: weird, "": [weird]})],
        provenance={weird: {"ok": weird}},
    )
    text = roundtrip_ok(edit)
    assert "日本語 🚗💨" in text  # unicode is written as is, control characters escaped
    assert "\\u2028" in text and "\\u0000" in text and "\\n" in text
    assert 'aspect "1.85:1"' in text and 'id="sec 1"' in text and '"none"->' in text


def test_string_escapes_in_hand_written_source():
    src = r'''edit { text "a\u00e9\U0001F697\ud83d\ude97\x41\'\/" 0s..1s }'''
    edit = parse_dsl(src)
    assert edit.overlays[0].text == "aé🚗🚗A'/"


# --------------------------------------------------------------------------- randomized edits


_ALPHABET = list("abcXYZ019 _-.#:\"'\\{}[](),;=@\n\t") + ["é", "日", "🚗", "\u2028", "\x00", "\x7f", "Ω"]


class RandomEdits:
    """Random valid Edits generated from the pydantic annotations (covers every field)."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def float_(self) -> float:
        r = self.rng
        return r.choice([
            lambda: r.uniform(-5, 5), lambda: r.uniform(0, 60), lambda: float(r.randint(-3, 300)),
            lambda: 0.0, lambda: -0.0, lambda: r.uniform(-1e-6, 1e-6), lambda: r.uniform(1e14, 1e18),
            lambda: r.choice([0.1, 0.2, 0.3]) * r.randint(1, 9),
        ])()

    def str_(self) -> str:
        r = self.rng
        kind = r.random()
        if kind < 0.3:
            return r.choice(["", "s1", "sec2", "s1.zoom1", "none", "A", "true", "null", "x_y.z"])
        if kind < 0.5:
            return r.choice("abcdefgh") + "".join(r.choice("abc123_.") for _ in range(r.randint(0, 6)))
        return "".join(r.choice(_ALPHABET) for _ in range(r.randint(0, 12)))

    def json_(self, depth: int = 0) -> Any:
        r = self.rng
        options = ["int", "float", "str", "bool", "none"] + (["list", "dict"] if depth < 3 else [])
        kind = r.choice(options)
        if kind == "int":
            return r.randint(-10**6, 10**6)
        if kind == "float":
            return self.float_()
        if kind == "str":
            return self.str_()
        if kind == "bool":
            return r.random() < 0.5
        if kind == "none":
            return None
        if kind == "list":
            return [self.json_(depth + 1) for _ in range(r.randint(0, 4))]
        return {self.str_(): self.json_(depth + 1) for _ in range(r.randint(0, 4))}

    def value(self, ann: Any, depth: int) -> Any:
        r = self.rng
        origin, args = get_origin(ann), get_args(ann)
        if origin is typing.Annotated:
            return self.value(args[0], depth)
        if origin is Literal:
            return r.choice(args)
        if origin in (Union, types.UnionType):
            members = [a for a in args if a is not type(None)]
            if type(None) in args and r.random() < 0.3:
                return None
            return self.value(r.choice(members), depth)
        if origin is tuple:
            return tuple(self.value(a, depth) for a in args)
        if origin is list:
            n = r.randint(0, 4 if depth < 4 else 1)
            return [self.value(args[0], depth + 1) for _ in range(n)]
        if origin is dict:
            if args[1] is int:
                return {self.str_(): r.randint(0, 50) for _ in range(r.randint(0, 3))}
            return {self.str_(): self.json_(1) for _ in range(r.randint(0, 3))}
        if ann is Any:
            return self.json_()
        if ann is bool:
            return r.random() < 0.5
        if ann is int:
            return r.choice([0, r.randint(0, 2000), r.randint(-50, 50)])
        if ann is float:
            return self.float_()
        if ann is str:
            return self.str_()
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return self.model(ann, depth + 1)
        raise TypeError(f"no generator for {ann!r}")

    def model(self, cls: type[BaseModel], depth: int = 0) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, info in cls.model_fields.items():
            required = info.is_required()
            if name == "type" or required or self.rng.random() < 0.75:
                out[name] = self.value(info.annotation, depth)
        return out

    def edit(self) -> Edit:
        return Edit.model_validate(self.model(Edit))


@pytest.mark.parametrize("seed", range(120))
def test_random_edits_roundtrip(seed):
    edit = RandomEdits(seed).edit()
    roundtrip_ok(edit)
    parse_dsl(to_dsl(edit, compact=True))


def test_random_generator_reaches_every_effect_and_field():
    seen_types, seen_fields = set(), set()
    for seed in range(120):
        for node in RandomEdits(seed).edit().walk():
            seen_types.add(node.node_type)
            seen_fields.update(f"{node.node_type}.{k}" for k in node.model_fields_set)
    assert {"zoom", "pan", "rotate", "shake", "speed", "freeze", "flash", "blur", "invert", "rgb_split",
            "shot", "transition", "section", "music", "audio_section", "text", "pattern"} <= seen_types
    assert {"shot.look", "shot.layout", "shot.motion", "zoom.keyframes", "transition.params",
            "music.energy", "text.words", "edit.stats", "edit.provenance"} <= seen_fields


# --------------------------------------------------------------------------- compact mode


def test_compact_mode_omits_noise_and_parses():
    edit = rich_edit()
    compact = to_dsl(edit, compact=True)
    full = to_dsl(edit)
    assert len(compact) < len(full)
    for noisy in ("beats [", "energy t0", "pacing", "look ", "keys [", "provenance {", "frames 0..30", "word \""):
        assert noisy not in compact, noisy
    assert "5 beats, 2 downbeats, 2 onsets omitted" in compact
    back = parse_dsl(compact)
    assert [s.start for s in back.shots()] == pytest.approx([s.start for s in edit.shots()], abs=0.01)
    assert [s.index for s in back.shots()] == [0, 1, 7]
    assert back.music.bpm == pytest.approx(128.0) and back.music.beats == []
    assert [p.rule for p in back.patterns] == [p.rule for p in edit.patterns]
    assert back.overlays[0].text == edit.overlays[0].text


# --------------------------------------------------------------------------- hand-written source

HAND = """
# a hand-written edit; comments, ';' separators and free clause order are fine
edit "demo.mp4" {
  timeline conf 0.9; media 720x1280 @25fps duration 4s audio frames 100  # trailing comment
  music offset 0.1s bpm 120 meter 4 {
    beats [0.1, 0.6,
           1.1, 1.6]
    drop 2s
  }
  section intro 0s..2s {
    shot 0s..1s on_beat { zoom 0s..1s 1->1.5 punch center 0.5, 0.5 }
    transition flash 0.95s..1.05s s1->s2 color #FFFFFF
    shot s2 1s..2s group B { look bw dominant #102030; shake 1s..2s amp 0.02 freq 8hz rot 2deg }
  }
  text "HELLO" 0s..2s at top anim typewriter
  pattern cut_on_beat() support 1.0 "every cut on a beat"
  pattern p9 "speed ramp"(ramp: {from: 1.0, to: 0.25}, tags: [a, "b c"]) span 0s..2s
  stats shots 2 cpm 30 effects {zoom: 1, shake: 1}
}
"""


def test_hand_written_source_parses():
    edit = parse_dsl(HAND)
    assert edit.id == "edit" and edit.timeline.id == "timeline" and edit.music.id == "music"
    assert edit.timeline.confidence == 0.9
    assert edit.media.source == "demo.mp4" and (edit.media.width, edit.media.height) == (720, 1280)
    assert edit.media.fps == 25.0 and edit.media.has_audio and edit.media.n_frames == 100
    assert edit.music.beats == [0.1, 0.6, 1.1, 1.6] and edit.music.drops[0].time == 2.0
    sec = edit.timeline.sections[0]
    shot1, tr, shot2 = sec.items
    assert sec.role == "intro" and shot1.id == "" and shot1.on_beat is True
    zoom = shot1.effects[0]
    assert (zoom.scale_from, zoom.scale_to, zoom.easing) == (1.0, 1.5, "punch")
    assert tr.kind == "flash" and tr.color == (255, 255, 255) and (tr.from_shot, tr.to_shot) == ("s1", "s2")
    assert shot2.index == 1 and shot2.source_group == "B" and shot2.look.bw
    assert shot2.look.dominant_color == (16, 32, 48)
    assert shot2.effects[0].frequency == 8.0 and shot2.effects[0].rotation_amplitude == 2.0
    assert edit.overlays[0].position == "top" and edit.overlays[0].animation == "typewriter"
    p0, p1 = edit.patterns
    assert p0.rule == "cut_on_beat" and p0.description == "every cut on a beat" and p0.params == {}
    assert p1.id == "p9" and p1.rule == "speed ramp" and p1.params["ramp"] == {"from": 1.0, "to": 0.25}
    assert p1.params["tags"] == ["a", "b c"] and (p1.start, p1.end) == (0.0, 2.0)
    assert edit.stats.effect_counts == {"zoom": 1, "shake": 1} and edit.stats.cuts_per_minute == 30.0


def test_tokenizer_positions_and_colour_vs_comment():
    toks = tokenize('edit {\n  x #aabbcc #not a colour\n  y "s"\n}')
    kinds = [(t.kind, t.line, t.col) for t in toks]
    assert ("color", 2, 5) in kinds
    assert ("string", 3, 5) in kinds
    assert all(t.value != "not" for t in toks)


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize("src, line, col, fragment", [
    ("edit {\n  section hook 0s..1s {\n    shot s1 0s..1s bogus 3\n  }\n}", 3, 20, "unknown clause 'bogus'"),
    ("edit {\n  section hook 0s..1s {\n    transition t1 flsh 0s..0s\n  }\n}", 3, 19, "did you mean 'flash'"),
    ("edit {\n  media 10x10\n  media 20x20\n}", 3, 3, "only once"),
    ('edit {\n  text "unterminated 0s..1s\n}', 2, 8, "unterminated string"),
    ("edit {\n  section hook 0s..1s {\n    shot s1 0s..1s\n", 4, 1, "unclosed section block"),
    ("edit {\n  sektion hook 0s..1s\n}", 2, 3, "did you mean 'section'"),
    ("edit {\n  section hook 0s 1s\n}", 2, 19, "expected '..'"),
    ("edit {\n  media 10x10 duration 3fps\n}", 2, 24, "unexpected unit 'fps'"),
    ('edit {\n  text "a\\q" 0s..1s\n}', 2, 10, "invalid escape"),
    ("edit {\n  pattern p1 rule(n=1,, ) support 1\n}", 2, 23, "expected a key"),
    ("edit {\n  media 10x10 frames 2.5\n}", 2, 22, "integer"),
    ("edit {\n  media 10x10 analysis_fps=\"fast\"\n}", 2, 3, "invalid value for media.analysis_fps"),
    ("edit {\n  media 10x10 width=5\n}", 2, 15, "given twice"),
    ("edit {\n  section hook 0s..1s {\n    shot s1 0s..1s on_beat on_beat\n  }\n}", 3, 28, "given twice"),
    ("edit {\n  section hook 0s..1s {\n    shot s1 0s..1s bogus_field=1\n  }\n}", 3, 20, "no settable field"),
    ("edit { media 1x1 } trailing", 1, 20, "after the closing"),
    ("media 1x1", 1, 1, "must start with 'edit'"),
    ("edit {\n  x = 1 $\n}", 2, 9, "unexpected character '$'"),
])
def test_parse_errors_report_line_and_column(src, line, col, fragment):
    with pytest.raises(EditScriptError) as info:
        parse_dsl(src)
    err = info.value
    assert (err.line, err.col) == (line, col), str(err)
    assert fragment in str(err)
    assert re.match(rf"line {line}:{col}: ", str(err))


def test_error_message_shows_source_excerpt():
    with pytest.raises(EditScriptError) as info:
        parse_dsl("edit {\n  section hook 0s..1s cps x\n}")
    text = str(info.value)
    assert "section hook 0s..1s cps x" in text and text.rstrip().endswith("^")
