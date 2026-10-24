"""Score a recovered Edit against a known one, and compare edits structurally.

:func:`compare` (truth vs. prediction) — every category is matched one-to-one
greedily, best pair first:

* **cuts** — shot boundary times within ``max(2 frames, 0.07 s)``;
  precision / recall / F1 and mean absolute offset.
* **transitions** — every shot boundary carries a kind (``cut`` unless a
  Transition says otherwise; fade-in / fade-out are boundaries at 0 and the
  end). Truth boundaries with a non-cut kind are matched to predicted
  boundaries (tolerance widened to half the truth span); the kind accuracy
  is computed over the matched ones (a missed boundary is already a cut
  recall error), with a confusion table that also counts misses, the
  fraction of special transitions detected as non-cut, and false positives
  (predicted non-cut kinds on truth hard cuts or on no boundary).
* **effects** — per effect type, a match needs the same type and a temporal
  IoU >= 0.2 or an overlap of >= 50 % of the shorter span. Parameter errors
  for matched pairs: zoom ratio relative error, pan direction agreement
  (within 45 deg), rotate sign, shake frequency error (Hz), speed slow / fast
  agreement.
* **music** — BPM relative error (plus an octave-tolerant flag: x1, x2, x1/2,
  x3, x1/3 within 4 %), beat F-measure at +-70 ms, drop time error.
* **text** — overlays matched at temporal IoU >= 0.5; mean bbox IoU and OCR
  string similarity of the matches.
* **look** — B&W agreement per truth shot (against the most-overlapping
  predicted shot).
* **overall** — mean of the main scores that are defined for the case: cut F1,
  transition-kind accuracy, effect F1 (micro), beat F-measure, text F1, B&W
  accuracy.

:func:`similarity` (edit vs. edit) — a 0..1 structural similarity of two
viral edits that ignores content: pacing over normalised time, absolute cut
rate, shot lengths in beats, cut-on-beat ratio, effect and transition mixes.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Any, Callable, Optional, Sequence

import numpy as np

from editast.ast.nodes import EFFECT_TYPES, Edit, EffectBase, Shot, TextOverlay, Transition

BEAT_WINDOW = 0.07  # seconds, beat F-measure tolerance
BPM_TOLERANCE = 0.04  # relative
OCTAVE_FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
EFFECT_IOU = 0.2
EFFECT_OVERLAP = 0.5
TEXT_IOU = 0.5
PAN_AGREE_COS = math.cos(math.radians(45.0))


# --------------------------------------------------------------------------- matching primitives


def _prf(tp: int, n_pred: int, n_truth: int) -> dict[str, float]:
    """Precision / recall / F1; an empty truth and prediction is a perfect score."""
    p = tp / n_pred if n_pred else (1.0 if n_truth == 0 else 0.0)
    r = tp / n_truth if n_truth else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _greedy(n_a: int, n_b: int, score: Callable[[int, int], Optional[float]]) -> list[tuple[int, int]]:
    """One-to-one matching: accept pairs by decreasing score (``None`` = not allowed)."""
    pairs = []
    for i in range(n_a):
        for j in range(n_b):
            s = score(i, j)
            if s is not None:
                pairs.append((s, i, j))
    pairs.sort(key=lambda x: -x[0])
    used_a, used_b, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((i, j))
    return out


def match_times(truth: Sequence[float], pred: Sequence[float], tol: float) -> list[tuple[int, int]]:
    """Match event times one-to-one within ``tol`` seconds (closest pairs first)."""
    t = np.asarray(truth, float)
    p = np.asarray(pred, float)
    if t.size == 0 or p.size == 0:
        return []
    d = np.abs(t[:, None] - p[None, :])
    ii, jj = np.nonzero(d <= tol + 1e-9)
    order = np.argsort(d[ii, jj], kind="stable")
    used_t, used_p, out = set(), set(), []
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def _span(start: float, end: float, min_len: float) -> tuple[float, float]:
    return (start, end) if end - start >= min_len else (start, start + min_len)


def _iou(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """(temporal IoU, overlap / shorter duration)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    shorter = min(a[1] - a[0], b[1] - b[0])
    return (inter / union if union > 0 else 0.0), (inter / shorter if shorter > 0 else 0.0)


def _fps(edit: Edit) -> float:
    return edit.media.analysis_fps or 30.0


def _duration(edit: Edit) -> float:
    shots = edit.shots()
    return max(edit.media.duration, shots[-1].end if shots else 0.0)


# --------------------------------------------------------------------------- categories


def _compare_cuts(truth: Edit, pred: Edit, tol: float) -> dict[str, Any]:
    t, p = truth.cut_times(), pred.cut_times()
    m = match_times(t, p, tol)
    out: dict[str, Any] = _prf(len(m), len(p), len(t))
    out.update(n_truth=len(t), n_pred=len(p), matched=len(m), tolerance=tol,
               mean_offset=float(np.mean([abs(t[i] - p[j]) for i, j in m])) if m else 0.0)
    return out


def boundaries(edit: Edit) -> list[tuple[float, str, float]]:
    """``(time, kind, half_span)`` for every shot boundary, plus fade-in / fade-out."""
    shots = edit.shots()
    by_id = {s.id: s for s in shots if s.id}
    entries: dict[float, tuple[str, float]] = {round(t, 6): ("cut", 0.0) for t in edit.cut_times()}
    end = _duration(edit)
    for tr in edit.transitions():
        if tr.kind == "cut":
            continue
        t = _transition_time(tr, by_id, end)
        fade = (tr.from_shot is None) != (tr.to_shot is None)
        key = min(entries, key=lambda k: abs(k - t)) if entries and not fade else None
        if key is not None and abs(key - t) <= 0.5 * tr.duration + 1e-3:
            entries[key] = (tr.kind, 0.5 * tr.duration)
        else:
            entries[round(t, 6)] = (tr.kind, 0.5 * tr.duration)
    return sorted((t, k, h) for t, (k, h) in entries.items())


def _transition_time(tr: Transition, by_id: dict[str, Shot], end: float) -> float:
    if tr.from_shot is None and tr.to_shot is not None:
        return 0.0  # fade-in
    if tr.to_shot is None and tr.from_shot is not None:
        return end  # fade-out
    if tr.to_shot in by_id:
        return by_id[tr.to_shot].start
    if tr.from_shot in by_id:
        return by_id[tr.from_shot].end
    return tr.time


def _compare_transitions(truth: Edit, pred: Edit, tol: float) -> dict[str, Any]:
    tb, pb = boundaries(truth), boundaries(pred)

    def score(i: int, j: int) -> Optional[float]:
        d = abs(tb[i][0] - pb[j][0])
        return -d if d <= max(tol, tb[i][2]) + 1e-9 else None

    pairs = dict(_greedy(len(tb), len(pb), score))
    confusion: dict[str, dict[str, int]] = {}
    n_special = matched = correct = detected = 0
    for i, (_, kind, _) in enumerate(tb):
        if kind == "cut":
            continue
        n_special += 1
        got = pb[pairs[i]][1] if i in pairs else "missed"
        confusion.setdefault(kind, {})
        confusion[kind][got] = confusion[kind].get(got, 0) + 1
        matched += i in pairs
        correct += got == kind
        detected += got not in ("cut", "missed")
    # A predicted special transition on a truth hard cut (or on no truth boundary) is a false positive.
    special_pairs = {pairs[i] for i, (_, kind, _) in enumerate(tb) if kind != "cut" and i in pairs}
    false_special = sum(1 for j, (_, kind, _) in enumerate(pb) if kind != "cut" and j not in special_pairs)
    return {
        "n_truth": n_special,
        "matched": matched,
        "accuracy": correct / matched if matched else None,  # kind accuracy on matched boundaries
        "detected": detected / n_special if n_special else None,  # truth specials predicted as non-cut
        "false_positives": false_special,
        "confusion": confusion,
    }


def _speed_class(eff: EffectBase) -> str:
    rate = 0.5 * (getattr(eff, "rate_from", 1.0) + getattr(eff, "rate_to", 1.0))
    return "slow" if rate < 0.98 else ("fast" if rate > 1.02 else "normal")


def _param_errors(kind: str, pairs: list[tuple[EffectBase, EffectBase]]) -> dict[str, float]:
    if not pairs:
        return {}
    if kind == "zoom":
        errs = []
        for t, p in pairs:
            rt = t.scale_to / t.scale_from if t.scale_from else t.scale_to  # type: ignore[attr-defined]
            rp = p.scale_to / p.scale_from if p.scale_from else p.scale_to  # type: ignore[attr-defined]
            errs.append(abs(rp - rt) / max(rt, 1e-9))
        return {"ratio_rel_error": float(np.mean(errs))}
    if kind == "pan":
        agree = []
        for t, p in pairs:
            a = np.array([t.dx, t.dy])  # type: ignore[attr-defined]
            b = np.array([p.dx, p.dy])  # type: ignore[attr-defined]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            agree.append(bool(na > 0 and nb > 0 and a @ b / (na * nb) >= PAN_AGREE_COS))
        return {"direction_agreement": float(np.mean(agree))}
    if kind == "rotate":
        signs = [np.sign(t.angle_to - t.angle_from) == np.sign(p.angle_to - p.angle_from)  # type: ignore[attr-defined]
                 for t, p in pairs]
        return {"sign_agreement": float(np.mean(signs))}
    if kind == "shake":
        freq = [abs(t.frequency - p.frequency) for t, p in pairs]  # type: ignore[attr-defined]
        amp = [abs(p.amplitude - t.amplitude) / max(t.amplitude, 1e-9) for t, p in pairs]  # type: ignore[attr-defined]
        return {"frequency_error_hz": float(np.mean(freq)), "amplitude_rel_error": float(np.mean(amp))}
    if kind == "speed":
        return {"slow_fast_agreement": float(np.mean([_speed_class(t) == _speed_class(p) for t, p in pairs]))}
    return {}


def _compare_effects(truth: Edit, pred: Edit) -> dict[str, Any]:
    min_len = 1.0 / _fps(truth)
    te = [e for _, e in truth.effects()]
    pe = [e for _, e in pred.effects()]
    by_type: dict[str, Any] = {}
    tp_all = 0
    for kind in EFFECT_TYPES:
        t = [e for e in te if e.type == kind]
        p = [e for e in pe if e.type == kind]
        if not t and not p:
            continue
        ts = [_span(e.start, e.end, min_len) for e in t]
        ps = [_span(e.start, e.end, min_len) for e in p]

        def score(i: int, j: int) -> Optional[float]:
            iou, cover = _iou(ts[i], ps[j])
            return iou + 1e-3 * cover if (iou >= EFFECT_IOU or cover >= EFFECT_OVERLAP) else None

        m = _greedy(len(t), len(p), score)
        tp_all += len(m)
        entry: dict[str, Any] = _prf(len(m), len(p), len(t))
        entry.update(n_truth=len(t), n_pred=len(p), matched=len(m),
                     mean_iou=float(np.mean([_iou(ts[i], ps[j])[0] for i, j in m])) if m else 0.0)
        entry.update(_param_errors(kind, [(t[i], p[j]) for i, j in m]))
        by_type[kind] = entry
    out: dict[str, Any] = _prf(tp_all, len(pe), len(te))
    f1s = [v["f1"] for v in by_type.values()]
    out.update(n_truth=len(te), n_pred=len(pe), f1_macro=float(np.mean(f1s)) if f1s else 1.0, by_type=by_type)
    return out


def _compare_music(truth: Edit, pred: Edit) -> Optional[dict[str, Any]]:
    tm, pm = truth.music, pred.music
    if tm is None:
        return None
    out: dict[str, Any] = {"truth_bpm": tm.bpm, "pred_bpm": pm.bpm if pm else None}
    if tm.bpm and pm is not None and pm.bpm:
        rel = abs(pm.bpm - tm.bpm) / tm.bpm
        out.update(bpm_rel_error=rel, bpm_ok=rel <= BPM_TOLERANCE,
                   bpm_ok_octave=any(abs(pm.bpm * f - tm.bpm) / tm.bpm <= BPM_TOLERANCE for f in OCTAVE_FACTORS))
    else:
        out.update(bpm_rel_error=None, bpm_ok=False, bpm_ok_octave=False)
    pred_beats = pm.beats if pm else []
    m = match_times(tm.beats, pred_beats, BEAT_WINDOW)
    beat = _prf(len(m), len(pred_beats), len(tm.beats))
    out.update(beat_precision=beat["precision"], beat_recall=beat["recall"], beat_f=beat["f1"])
    if tm.drops:
        pred_drops = [d.time for d in pm.drops] if pm else []
        t0 = tm.drops[0].time
        out["drop_error"] = float(min(abs(t0 - d) for d in pred_drops)) if pred_drops else None
    else:
        out["drop_error"] = None
    return out


def _normalise_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").upper()).strip()


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _compare_text(truth: Edit, pred: Edit) -> dict[str, Any]:
    min_len = 1.0 / _fps(truth)
    t: list[TextOverlay] = truth.overlays
    p: list[TextOverlay] = pred.overlays
    ts = [_span(o.start, o.end, min_len) for o in t]
    ps = [_span(o.start, o.end, min_len) for o in p]

    def score(i: int, j: int) -> Optional[float]:
        iou = _iou(ts[i], ps[j])[0]
        return iou if iou >= TEXT_IOU else None

    m = _greedy(len(t), len(p), score)
    out: dict[str, Any] = _prf(len(m), len(p), len(t))
    with_text = [(t[i], p[j]) for i, j in m if t[i].text]
    out.update(
        n_truth=len(t), n_pred=len(p), matched=len(m),
        bbox_iou=float(np.mean([_bbox_iou(t[i].bbox, p[j].bbox) for i, j in m])) if m else 0.0,
        ocr_similarity=float(np.mean([difflib.SequenceMatcher(None, _normalise_text(a.text),
                                                               _normalise_text(b.text)).ratio()
                                      for a, b in with_text])) if with_text else None,
    )
    return out


def _compare_look(truth: Edit, pred: Edit) -> dict[str, Any]:
    tshots, pshots = truth.shots(), pred.shots()
    agree, hits, n_bw = [], 0, 0
    for s in tshots:
        best = max(pshots, key=lambda q: q.overlap(s.start, s.end), default=None)
        t_bw = bool(s.look and s.look.bw)
        p_bw = bool(best is not None and best.overlap(s.start, s.end) > 0 and best.look and best.look.bw)
        agree.append(t_bw == p_bw)
        n_bw += t_bw
        hits += t_bw and p_bw
    return {"bw_accuracy": float(np.mean(agree)) if agree else 1.0, "n_bw": n_bw,
            "bw_recall": hits / n_bw if n_bw else None}


# --------------------------------------------------------------------------- public API


def compare(truth: Edit, pred: Edit) -> dict[str, Any]:
    """Score ``pred`` against ``truth`` (see the module docstring for the metrics)."""
    tol = max(2.0 / _fps(truth), 0.07)
    report: dict[str, Any] = {
        "cuts": _compare_cuts(truth, pred, tol),
        "transitions": _compare_transitions(truth, pred, tol),
        "effects": _compare_effects(truth, pred),
        "music": _compare_music(truth, pred),
        "text": _compare_text(truth, pred),
        "look": _compare_look(truth, pred),
    }
    main = [report["cuts"]["f1"], report["transitions"]["accuracy"], report["effects"]["f1"],
            report["music"]["beat_f"] if report["music"] else None,
            report["text"]["f1"] if (truth.overlays or pred.overlays) else None,
            report["look"]["bw_accuracy"]]
    scores = [s for s in main if s is not None]
    report["overall"] = float(np.mean(scores)) if scores else 0.0
    return report


def _fmt(v: Any, width: int = 6) -> str:
    """Right-aligned table cell (``width=0``: unpadded)."""
    if v is None:
        return "-".rjust(width)
    if isinstance(v, bool):
        return ("yes" if v else "no").rjust(width)
    if isinstance(v, float):
        return f"{v:.3f}".rjust(width)
    return str(v).rjust(width)


def format_report(report: dict[str, Any]) -> str:
    """Compact aligned text table of a :func:`compare` report."""
    rows: list[tuple[str, Any, Any, Any, str]] = []
    c = report["cuts"]
    rows.append(("cuts", c["precision"], c["recall"], c["f1"],
                 f"{c['matched']}/{c['n_truth']} truth, {c['n_pred']} pred,"
                 f" mean offset {c['mean_offset'] * 1000:.0f} ms"))
    tr = report["transitions"]
    conf = "; ".join(f"{k}->" + ",".join(f"{pk}:{n}" for pk, n in v.items()) for k, v in tr["confusion"].items())
    rows.append(("transition kind", None, None, tr["accuracy"],
                 f"{tr['matched']}/{tr['n_truth']} matched, detected {_fmt(tr['detected'], 0)},"
                 f" fp {tr['false_positives']}" + (f"  [{conf}]" if conf else "")))
    ef = report["effects"]
    rows.append(("effects (all)", ef["precision"], ef["recall"], ef["f1"],
                 f"{ef['n_truth']} truth, {ef['n_pred']} pred, macro F1 {ef['f1_macro']:.3f}"))
    for kind, e in ef["by_type"].items():
        extras = [f"{e['matched']}/{e['n_truth']}"]
        extras += [f"{k}={_fmt(v, 0)}" for k, v in e.items()
                   if k not in ("precision", "recall", "f1", "n_truth", "n_pred", "matched")]
        rows.append((f"  {kind}", e["precision"], e["recall"], e["f1"], " ".join(extras)))
    mu = report.get("music")
    if mu:
        pred_bpm = f"{mu['pred_bpm']:.1f}" if mu["pred_bpm"] else "-"
        truth_bpm = f"{mu['truth_bpm']:.1f}" if mu["truth_bpm"] else "-"
        bpm = (f"bpm {pred_bpm} vs {truth_bpm} (err {_fmt(mu['bpm_rel_error'], 0)}, ok {_fmt(mu['bpm_ok'], 0)},"
               f" octave ok {_fmt(mu['bpm_ok_octave'], 0)}) drop err {_fmt(mu['drop_error'], 0)} s")
        rows.append(("beats", mu["beat_precision"], mu["beat_recall"], mu["beat_f"], bpm))
    tx = report["text"]
    rows.append(("text", tx["precision"], tx["recall"], tx["f1"],
                 f"{tx['matched']}/{tx['n_truth']} bbox IoU {tx['bbox_iou']:.3f} OCR {_fmt(tx['ocr_similarity'], 0)}"))
    lk = report["look"]
    rows.append(("look b&w", None, None, lk["bw_accuracy"],
                 f"{lk['n_bw']} b&w shots, recall {_fmt(lk['bw_recall'], 0)}"))
    name_w = max(len(r[0]) for r in rows) + 2
    lines = [f"{'metric':<{name_w}}{'P':>7}{'R':>7}{'F1':>7}  details", "-" * (name_w + 21 + 30)]
    for name, p, r, f, extra in rows:
        lines.append(f"{name:<{name_w}}{_fmt(p, 7)}{_fmt(r, 7)}{_fmt(f, 7)}  {extra}")
    lines.append(f"{'overall':<{name_w}}{'':>14}{_fmt(report['overall'], 7)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- similarity

_LENGTH_BINS = (0.0, 0.75, 1.5, 3.0, 6.0, math.inf)  # shot length in beats
_PACING_BINS = 10
_TRANSITION_KINDS = ("cut", "flash", "fade_black", "fade_white", "dissolve", "whip", "zoom", "spin", "glitch",
                     "slide", "unknown")
_WEIGHTS = {"pacing_shape": 0.15, "pacing_rate": 0.2, "shot_lengths": 0.2, "on_beat": 0.1, "effects": 0.2,
            "transitions": 0.15}


def _distribution(counts: np.ndarray) -> Optional[np.ndarray]:
    total = float(counts.sum())
    return counts / total if total > 0 else None


def _intersection(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    return float(np.minimum(a, b).sum())


def _ratio(a: float, b: float) -> float:
    if a <= 0 and b <= 0:
        return 1.0
    return min(a, b) / max(a, b)


def edit_signature(edit: Edit) -> dict[str, Any]:
    """Content-free structural features of an edit (inputs of :func:`similarity`)."""
    shots = edit.shots()
    total = _duration(edit)
    music = edit.music
    period = 60.0 / music.bpm if music is not None and music.bpm else None
    cuts = np.asarray(edit.cut_times(), float)
    pacing = np.histogram(cuts / total if total > 0 else cuts, bins=_PACING_BINS, range=(0.0, 1.0))[0].astype(float)
    pacing = np.convolve(pacing, [0.25, 0.5, 0.25], mode="same")  # tolerate small shifts
    lengths = np.array([s.duration / (period or 0.5) for s in shots])  # 120 BPM grid when unknown
    on_beat: Optional[float] = None
    if music is not None and music.beats and cuts.size:
        beats = np.asarray(music.beats, float)
        on_beat = float(np.mean(np.min(np.abs(cuts[:, None] - beats[None, :]), axis=1) <= BEAT_WINDOW))
    effects = np.array([sum(1 for _, e in edit.effects() if e.type == k) for k in EFFECT_TYPES], float)
    kinds = [k for _, k, _ in boundaries(edit)]
    transitions = np.array([kinds.count(k) for k in _TRANSITION_KINDS], float)
    return {
        "pacing_shape": _distribution(pacing),
        "cuts_per_second": cuts.size / total if total > 0 else 0.0,
        "shot_lengths": _distribution(np.histogram(lengths, bins=_LENGTH_BINS)[0].astype(float)),
        "on_beat": on_beat,
        "effect_mix": _distribution(effects),
        "effects_per_second": float(effects.sum()) / total if total > 0 else 0.0,
        "transition_mix": _distribution(transitions),
    }


def similarity(a: Edit, b: Edit) -> float:
    """Structural similarity of two edits in ``[0, 1]`` (1 = same editing grammar).

    A weighted mean of: pacing curve shape (cuts over normalised time), cut
    rate ratio, shot-length-in-beats histogram, cut-on-beat ratio, effect mix
    and density, and transition mix. Components undefined for either edit
    (e.g. on-beat ratio without music) are left out.
    """
    sa, sb = edit_signature(a), edit_signature(b)
    parts = {
        "pacing_shape": _intersection(sa["pacing_shape"], sb["pacing_shape"]),
        "pacing_rate": _ratio(sa["cuts_per_second"], sb["cuts_per_second"]),
        "shot_lengths": _intersection(sa["shot_lengths"], sb["shot_lengths"]),
        "effects": 0.5 * _intersection(sa["effect_mix"], sb["effect_mix"])
        + 0.5 * _ratio(sa["effects_per_second"], sb["effects_per_second"]),
        "transitions": _intersection(sa["transition_mix"], sb["transition_mix"]),
    }
    if sa["on_beat"] is not None and sb["on_beat"] is not None:
        parts["on_beat"] = 1.0 - abs(sa["on_beat"] - sb["on_beat"])
    weight = sum(_WEIGHTS[k] for k in parts)
    value = sum(_WEIGHTS[k] * v for k, v in parts.items()) / weight
    return round(float(np.clip(value, 0.0, 1.0)), 6)
