"""Pre-registered confirmatory evaluation (runs once, on the imported Via B corpus, after the freeze).

Inputs: the imported reference (``importer.load_reference``), one predictions file per system, optionally the S6
gate admission file. Nothing here tunes anything: thresholds, units and the family of comparisons come from
``config.CONFIRMATORY`` and the pre-registration. Absent values are reported as unavailable, never as 0.

Predictions file (JSON), one per system::

    {"system": "vc", "version": "<commit>", "supported_classes": ["filler_word", ...],
     "recordings": {"VBT_S01_PRES1": {"status": "ok",
        "events": [{"class": "filler_word", "start_s": 1.2, "end_s": 1.5, "score": 0.8, "decision": "accept"}]}}}

A recording missing from the file, or with ``status != "ok"``, counts as a system failure: zero detections in the
primary analysis (worst case) and is excluded in a sensitivity analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mrcd.annotation_eval import metrics as M
from mrcd.annotation_eval.agreement import segment_positive, segments
from mrcd.annotation_eval.config import (
    ABLATION_SYSTEMS,
    BASELINE_SYSTEM,
    CONFIG_VERSION,
    CONFIRMATORY as CFG,
    config_hash,
)
from mrcd.annotation_eval.freeze import FreezeError, require_frozen
from mrcd.annotation_eval.importer import load_reference
from mrcd.annotation_eval.schema import ALL_CLASSES, CORE_CLASSES


# ------------------------------------------------------------------------------------------ predictions
def load_predictions(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "system" not in d or "recordings" not in d:
        raise ValueError(f"{path}: predictions need 'system' and 'recordings'")
    d.setdefault("supported_classes", list(ALL_CLASSES))
    return d


def _hyp_events(pred: dict, rec: str, cls: str) -> tuple[list[dict], bool]:
    r = pred["recordings"].get(rec)
    if r is None or r.get("status") != "ok":
        return [], False
    return [e for e in r.get("events", []) if e.get("class") == cls], True


def _ignored(iv, ignore: list[dict], cls: str) -> bool:
    return any(
        (g["classes"] == "all" or cls in g["classes"])
        and M.overlap(iv, (g["start_s"], g["end_s"])) > 0
        for g in ignore
    )


# ------------------------------------------------------------------------------------ per-recording counts
def event_vector(
    ref: dict, rec: str, pred: dict, cls: str, thr: float, any_overlap: bool = False
):
    """[tp, fp, fn] plus per-hypothesis status list, for one recording and class."""
    r = ref["recordings"][rec]
    ref_iv = r["events"][cls]
    hyp, ok = _hyp_events(pred, rec, cls)
    hyp_iv = [(e["start_s"], e["end_s"]) for e in hyp]
    pairs = M.match_events(ref_iv, hyp_iv, thr, any_overlap=any_overlap)
    matched = {j for _, j, _ in pairs}
    status = []
    for j, iv in enumerate(hyp_iv):
        if j in matched:
            status.append("tp")
        elif _ignored(iv, r["ignore"], cls):
            status.append("ign")
        else:
            status.append("fp")
    tp, fp = len(pairs), status.count("fp")
    return np.array([tp, fp, len(ref_iv) - tp], float), hyp, status, ok


def segment_vector(
    ref: dict,
    rec: str,
    pred: dict,
    cls: str,
    only: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    r = ref["recordings"][rec]
    hyp, _ = _hyp_events(pred, rec, cls)
    hiv = [(e["start_s"], e["end_s"]) for e in hyp]
    ign = [
        (g["start_s"], g["end_s"])
        for g in r["ignore"]
        if g["classes"] == "all" or cls in g["classes"]
    ]
    tp = fp = fn = tn = 0
    for seg in only if only is not None else segments(r["duration_s"]):
        if M.fraction_inside(seg, ign) > CFG["segment_ignore_fraction"]:
            continue
        a, b = segment_positive(seg, r["events"][cls]), segment_positive(seg, hiv)
        tp, fp, fn, tn = (
            tp + (a and b),
            fp + (b and not a),
            fn + (a and not b),
            tn + (not a and not b),
        )
    return np.array([tp, fp, fn, tn], float)


def f1_of(v: np.ndarray) -> float | None:
    return M.prf(v[0], v[1], v[2])[2]


def prec_of(v: np.ndarray) -> float | None:
    return M.prf(v[0], v[1], v[2])[0]


def rec_of(v: np.ndarray) -> float | None:
    return M.prf(v[0], v[1], v[2])[1]


def _scope(ref: dict, cls: str) -> list[str]:
    return [
        k
        for k, r in ref["recordings"].items()
        if r["coverage"].get(cls) in ("pos", "neg")
    ]


def _by_speaker(ref: dict, recs: list[str], fn) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for rec in recs:
        v = fn(rec)
        sp = ref["recordings"][rec]["speaker"]
        out[sp] = out.get(sp, 0) + v
    return out


def _ci(counts: dict, stat) -> dict:
    return M.speaker_bootstrap(
        counts, stat, B=CFG["bootstrap_B"], seed=CFG["seed"], level=CFG["ci_level"]
    )


def _loo_range(counts: dict, stat) -> dict:
    """Leave-one-speaker-out range of the statistic (influence of any single speaker)."""
    if len(counts) < 3:
        return {"min": None, "max": None}
    vals = []
    for k in counts:
        v = stat(sum(x for kk, x in counts.items() if kk != k))
        if v is not None:
            vals.append(v)
    return {"min": min(vals), "max": max(vals)} if vals else {"min": None, "max": None}


# ------------------------------------------------------------------------------------------- adequacy
def class_adequacy(ref: dict, cls: str) -> dict:
    scope = _scope(ref, cls)
    pos = sum(len(ref["recordings"][r]["events"][cls]) for r in scope)
    spk = {ref["recordings"][r]["speaker"] for r in scope}
    pos_spk = {
        ref["recordings"][r]["speaker"]
        for r in scope
        if ref["recordings"][r]["events"][cls]
    }
    gate = ref["manifest"]["classes"][cls].get("agreement_gate_passed")
    reasons = []
    if cls not in CORE_CLASSES:
        reasons.append("optional class (never in the confirmatory family)")
    if not scope:
        reasons.append("coverage not declared complete in any recording")
    if pos < CFG["min_positives_per_class"]:
        reasons.append(f"positives {pos} < {CFG['min_positives_per_class']}")
    if len(spk) < CFG["min_speakers"]:
        reasons.append(f"speakers {len(spk)} < {CFG['min_speakers']}")
    if len(pos_spk) < CFG["min_positive_speakers"]:
        reasons.append(
            f"speakers with positives {len(pos_spk)} < {CFG['min_positive_speakers']}"
        )
    if gate is not True:
        reasons.append(
            "annotation agreement gate not passed"
            if gate is False
            else "agreement gate not computed"
        )
    return {
        "positives": pos,
        "speakers": len(spk),
        "speakers_with_positives": len(pos_spk),
        "agreement_gate_passed": gate,
        "confirmatory": not reasons,
        "reasons_not_confirmatory": reasons,
    }


# ---------------------------------------------------------------------------------------------- core
def _system_class_block(ref: dict, pred: dict, cls: str) -> dict:
    scope = _scope(ref, cls)
    block: dict[str, Any] = {}
    prim = lambda rec: event_vector(ref, rec, pred, cls, CFG["primary_iou"])[0]  # noqa: E731
    cnt = _by_speaker(ref, scope, prim)
    block["event"] = {
        "iou": CFG["primary_iou"],
        "totals_tp_fp_fn": [int(x) for x in sum(cnt.values())] if cnt else None,
        "f1": _ci(cnt, f1_of),
        "precision": _ci(cnt, prec_of),
        "recall": _ci(cnt, rec_of),
        "leave_one_speaker_out_f1": _loo_range(cnt, f1_of),
    }
    sens = {}
    for thr in CFG["sensitivity_iou"]:
        c = _by_speaker(
            ref, scope, lambda rec, t=thr: event_vector(ref, rec, pred, cls, t)[0]
        )
        sens[f"iou_{thr}"] = _ci(c, f1_of)
    if CFG["sensitivity_any_overlap"]:
        c = _by_speaker(
            ref, scope, lambda rec: event_vector(ref, rec, pred, cls, 0.0, True)[0]
        )
        sens["any_overlap"] = _ci(c, f1_of)
    ok_recs = [r for r in scope if _hyp_events(pred, r, cls)[1]]
    c = _by_speaker(ref, ok_recs, prim)
    sens["excluding_failed_recordings"] = {
        **_ci(c, f1_of),
        "n_recordings": len(ok_recs),
        "n_failed": len(scope) - len(ok_recs),
    }
    block["sensitivity_f1"] = sens
    seg = _by_speaker(ref, scope, lambda rec: segment_vector(ref, rec, pred, cls))
    block["segment_presence"] = {
        "segment_s": CFG["segment_s"],
        "f1": _ci(seg, f1_of),
        "precision": _ci(seg, prec_of),
        "recall": _ci(seg, rec_of),
    }
    block["coverage_risk"] = _coverage_risk(ref, pred, cls, scope)
    return block


def _coverage_risk(ref: dict, pred: dict, cls: str, scope: list[str]) -> dict:
    """Coverage = accepted / predicted; risk = false positives among accepted. Needs 'decision' on events."""
    has_decision = any(
        "decision" in e for rec in scope for e in _hyp_events(pred, rec, cls)[0]
    )
    n_pred = sum(len(_hyp_events(pred, rec, cls)[0]) for rec in scope)
    if n_pred == 0:
        return {"available": False, "reason": "system produced no events of this class"}
    if not has_decision:
        return {
            "available": False,
            "reason": "predictions carry no 'decision' field (no abstention policy)",
        }

    def vec(rec):
        _, hyp, status, _ = event_vector(ref, rec, pred, cls, CFG["primary_iou"])
        n = acc = acc_fp = 0
        for e, s in zip(hyp, status):
            if s == "ign":
                continue
            n += 1
            if e.get("decision", "accept") == "accept":
                acc += 1
                acc_fp += s == "fp"
        return np.array([n, acc, acc_fp], float)

    cnt = _by_speaker(ref, scope, vec)
    cov = _ci(cnt, lambda v: None if v[0] == 0 else v[1] / v[0])
    risk = _ci(cnt, lambda v: None if v[1] == 0 else v[2] / v[1])
    ref_cr = CFG["reference_coverage_risk"]
    return {
        "available": True,
        "coverage": cov,
        "risk": risk,
        "reference_only": ref_cr,
        "meets_reference": None
        if cov["lo"] is None or risk["hi"] is None
        else bool(cov["lo"] >= ref_cr["coverage"] and risk["hi"] <= ref_cr["risk"]),
    }


def _paired(ref: dict, pa: dict, pb: dict, cls: str) -> dict:
    scope = _scope(ref, cls)
    fa = lambda rec: event_vector(ref, rec, pa, cls, CFG["primary_iou"])[0]  # noqa: E731
    fb = lambda rec: event_vector(ref, rec, pb, cls, CFG["primary_iou"])[0]  # noqa: E731
    ca, cb = _by_speaker(ref, scope, fa), _by_speaker(ref, scope, fb)
    return M.paired_speaker_bootstrap(
        ca, cb, f1_of, B=CFG["bootstrap_B"], seed=CFG["seed"], level=CFG["ci_level"]
    )


def evaluate(
    corpus_dir: Path,
    predictions: dict[str, Path],
    *,
    freeze_file: Path | None = None,
    freeze_root: Path | None = None,
    allow_unfrozen: bool = False,
    gate_file: Path | None = None,
    classes: tuple[str, ...] = ALL_CLASSES,
) -> dict:
    """Run the pre-registered analysis. Refuses to run (FreezeError) unless the freeze record is intact, unless
    ``allow_unfrozen`` is set, in which case the report is stamped NOT CONFIRMATORY."""
    frozen = None
    if not allow_unfrozen:
        if freeze_file is None:
            raise FreezeError(
                "a freeze record is required for the confirmatory evaluation"
            )
        frozen = require_frozen(freeze_file, **({} if freeze_root is None else {"root": freeze_root}))
    ref = load_reference(corpus_dir)
    preds = {name: load_predictions(p) for name, p in predictions.items()}
    for name, p in preds.items():
        if p["system"] != name:
            raise ValueError(
                f"predictions file for {name!r} declares system {p['system']!r}"
            )
    out: dict[str, Any] = {
        "status": "CONFIRMATORY"
        if frozen
        else "NOT CONFIRMATORY (exploratory run without a valid freeze)",
        "config_version": CONFIG_VERSION,
        "config_hash": config_hash(),
        "freeze": frozen,
        "corpus": {
            k: ref["manifest"][k]
            for k in ("n_recordings", "n_speakers", "duration_s", "examples_sha256")
        },
        "systems": {
            n: {
                "version": p.get("version"),
                "supported_classes": p["supported_classes"],
            }
            for n, p in preds.items()
        },
        "classes": {},
    }
    family: list[tuple[str, str]] = []
    pvals: list[float | None] = []
    for cls in classes:
        adeq = class_adequacy(ref, cls)
        row: dict[str, Any] = {
            "adequacy": adeq,
            "systems": {},
            "paired_vs_baseline": {},
        }
        for name, p in preds.items():
            if cls not in p["supported_classes"]:
                row["systems"][name] = {
                    "available": False,
                    "reason": "class not supported by the system",
                }
                continue
            row["systems"][name] = _system_class_block(ref, p, cls)
        base = preds.get(BASELINE_SYSTEM)
        if base is not None and cls in base["supported_classes"]:
            for name, p in preds.items():
                if name == BASELINE_SYSTEM or cls not in p["supported_classes"]:
                    continue
                res = _paired(ref, p, base, cls)
                row["paired_vs_baseline"][name] = res
                if adeq["confirmatory"]:
                    family.append((cls, name))
                    pvals.append(res["p"])
        out["classes"][cls] = row
    adj = M.holm(pvals)
    out["family"] = {
        "size": sum(p is not None for p in pvals),
        "description": "F1(event, IoU>=0.5) of each non-baseline system vs the lexical rule, per confirmatory class; Holm",
        "alpha": CFG["alpha"],
    }
    for (cls, name), a in zip(family, adj):
        cell = out["classes"][cls]["paired_vs_baseline"][name]
        cell["p_holm"] = a
        cell["significant_holm"] = None if a is None else bool(a < CFG["alpha"])
    if gate_file is not None and all(s in preds for s in ABLATION_SYSTEMS):
        out["ablation_at_vs_atv"] = ablation(ref, preds, Path(gate_file), classes)
    elif gate_file is not None:
        out["ablation_at_vs_atv"] = {
            "available": False,
            "reason": f"need predictions of {ABLATION_SYSTEMS}",
        }
    return out


def ablation(ref: dict, preds: dict, gate_file: Path, classes) -> dict:
    """A+T vs A+T+V on the units admitted by the frozen S6 gate (segment presence F1, paired by speaker)."""
    g = json.loads(Path(gate_file).read_text(encoding="utf-8"))
    admitted: dict[str, list[tuple[float, float]]] = {}
    total_segments = admitted_segments = 0
    for rec, r in ref["recordings"].items():
        grid = segments(r["duration_s"])
        total_segments += len(grid)
        sel = [tuple(x) for x in g.get("admitted", {}).get(rec, [])]
        grid_set = {(round(a, 6), round(b, 6)) for a, b in grid}
        bad = [x for x in sel if (round(x[0], 6), round(x[1], 6)) not in grid_set]
        if bad:
            raise ValueError(
                f"{rec}: admitted segments not on the evaluation grid: {bad[:3]}"
            )
        admitted[rec] = sel
        admitted_segments += len(sel)
    frac = admitted_segments / total_segments if total_segments else None
    out: dict[str, Any] = {
        "gate_config_hash": g.get("gate_config_hash"),
        "admitted_fraction": frac,
        "meets_min_admitted_fraction": None
        if frac is None
        else bool(frac >= CFG["min_gate_admitted_fraction"]),
        "classes": {},
    }
    pv, names = [], []
    for cls in classes:
        if cls not in CORE_CLASSES:
            continue
        scope = [r for r in _scope(ref, cls) if admitted.get(r)]
        ca = _by_speaker(
            ref,
            scope,
            lambda rec: segment_vector(ref, rec, preds["at"], cls, admitted[rec]),
        )
        cb = _by_speaker(
            ref,
            scope,
            lambda rec: segment_vector(ref, rec, preds["atv"], cls, admitted[rec]),
        )
        pos = sum(v[0] + v[2] for v in ca.values())
        pos_spk = sum(1 for v in ca.values() if v[0] + v[2] > 0)
        adequate = bool(
            pos >= CFG["min_positives_per_class"]
            and pos_spk >= CFG["min_positive_speakers"]
        )
        res = M.paired_speaker_bootstrap(
            cb, ca, f1_of, B=CFG["bootstrap_B"], seed=CFG["seed"], level=CFG["ci_level"]
        )
        cell = {
            "admitted_positive_segments": int(pos),
            "speakers_with_admitted_positives": pos_spk,
            "adequate_sample": adequate,
            "f1_at": _ci(ca, f1_of),
            "f1_atv": _ci(cb, f1_of),
            "diff_atv_minus_at": res,
        }
        if adequate and res["p"] is not None:
            pv.append(res["p"])
            names.append(cls)
        out["classes"][cls] = cell
    for cls, a in zip(names, M.holm(pv)):
        out["classes"][cls]["p_holm"] = a
    margin = CFG["equivalence_margin_f1"]
    for cls, cell in out["classes"].items():
        d = cell["diff_atv_minus_at"]
        if not cell["adequate_sample"] or d["lo"] is None:
            cell["verdict"] = "no concluyente (inadequate admitted sample)"
        elif d["lo"] > 0 and cell.get("p_holm", 1) < CFG["alpha"]:
            cell["verdict"] = "aporte medido"
        elif d["hi"] < 0:
            cell["verdict"] = "empeora"
        elif d["lo"] >= -margin and d["hi"] <= margin:
            cell["verdict"] = f"sin aporte (CI inside +-{margin})"
        else:
            cell["verdict"] = "no concluyente"
    return out


def render_markdown(res: dict) -> str:
    def f(x, nd=3):
        return "n/d" if x is None else f"{x:.{nd}f}"

    def ci(b):
        return f"{f(b['point'])} [{f(b['lo'])}, {f(b['hi'])}]"

    lines = [f"# Confirmatory evaluation report: {res['status']}", ""]
    c = res["corpus"]
    lines.append(
        f"Corpus: {c['n_recordings']} recordings, {c['n_speakers']} speakers, {c['duration_s']} s; config {res['config_hash'][:12]}."
    )
    lines.append("")
    for cls, row in res["classes"].items():
        a = row["adequacy"]
        tag = (
            "CONFIRMATORY"
            if a["confirmatory"]
            else "EXPLORATORY: " + "; ".join(a["reasons_not_confirmatory"])
        )
        lines += [
            f"## {cls} ({a['positives']} positives, {a['speakers_with_positives']} speakers): {tag}",
            "",
        ]
        lines.append("| system | event F1 (IoU 0.5) | P | R | segment F1 |")
        lines.append("|---|---|---|---|---|")
        for name, s in row["systems"].items():
            if s.get("available") is False:
                lines.append(f"| {name} | n/a: {s['reason']} | | | |")
                continue
            e, sg = s["event"], s["segment_presence"]
            lines.append(
                f"| {name} | {ci(e['f1'])} | {ci(e['precision'])} | {ci(e['recall'])} | {ci(sg['f1'])} |"
            )
        for name, p in row["paired_vs_baseline"].items():
            lines.append(
                f"\n{name} - {BASELINE_SYSTEM}: dF1 {f(p['diff'])} [{f(p['lo'])}, {f(p['hi'])}], p {f(p['p'])}, Holm {f(p.get('p_holm'))}"
            )
        lines.append("")
    if "ablation_at_vs_atv" in res:
        lines += [
            "## Ablation A+T vs A+T+V (S6 gate)",
            "",
            json.dumps(res["ablation_at_vs_atv"], indent=1, default=str),
        ]
    return "\n".join(lines) + "\n"
