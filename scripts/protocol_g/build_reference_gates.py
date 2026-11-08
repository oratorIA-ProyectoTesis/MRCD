#!/usr/bin/env python
"""Reference quality gates QG-1..QG-5 of protocol G (section 7) -> REFERENCE_GATES_G.json.

Reads ONLY committed annotation files (pass 1, pass 2, optional rater-2 file; ORIGINAL item ids, i.e. already unblinded)
and imports the kit v1.1 functions unchanged: ``agreement.compute`` / ``compute_markers`` / ``match_events`` /
``cohen_kappa`` / ``krippendorff_alpha_nominal``. ``subset_below_minimum`` and ``function_kappa_descriptive_only`` are
copied verbatim from those outputs. The output carries no timestamp, so re-running on the same files reproduces it
byte for byte (``run`` regenerates and compares). No system output exists at this point.

Usage:
  python -m scripts.protocol_g.build_reference_gates --pass1 P1.csv --pass2 P2.csv [--rater2 R2.csv] --out REFERENCE_GATES_G.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.annotation import agreement as AG
from scripts.protocol_g import common as C

SPEC = {
    "QG-1": "per-class item-level kappa (items common to both passes; label = >= 1 event of the class in any state) >= %.2f and Dice (per_class verbatim) >= %.2f, assessable iff >= %d events in each pass"
    % (C.QG1_KAPPA, C.QG1_DICE, C.QG1_MIN_EVENTS),
    "QG-2": "nominal kappa on the 3-way state of REP-REP pairs matched per item (match_events, IoU >= 0.10) and on marker_function of matched marker spans (compute_markers) >= %.2f"
    % C.QG2_KAPPA,
    "QG-3": "rater 2 vs pass 1 AND vs pass 2: item-level presence kappa >= %.2f per class, assessable iff not subset_below_minimum and the pass has >= %d events of the class on the subset items; undefined kappa when assessable = failed; markers via compute_markers (subset)"
    % (C.QG3_KAPPA, C.QG3_MIN_EVENTS),
    "QG-4": "boundary_agreement and mean_iou_matched, descriptive, no gate",
    "QG-5": "function_kappa_descriptive_only of compute_markers (intra) must be false",
}


# ------------------------------------------------------------------------------------------------ helpers
def _presence(A: dict, B: dict, cls: str) -> tuple[list[bool], list[bool]]:
    common = sorted(set(A) & set(B))
    return (
        [any(e["class"] == cls for e in A[i]) for i in common],
        [any(e["class"] == cls for e in B[i]) for i in common],
    )


def presence_stats(A: dict, B: dict, cls: str) -> dict:
    x, y = _presence(A, B, cls)
    return {
        "kappa": AG.cohen_kappa(x, y),
        "alpha": AG.krippendorff_alpha_nominal(x, y),
        "n_items": len(x),
    }


def rep_state_kappa(A: dict, B: dict, min_iou: float = 0.1) -> dict:
    """kappa on ``state`` of matched REP events (REP events matched with REP events only)."""
    sa, sb = [], []
    for item in sorted(set(A) & set(B)):
        ea = [e for e in A[item] if e["class"] == "repetition"]
        eb = [e for e in B[item] if e["class"] == "repetition"]
        pairs, _ua, _ub = AG.match_events(ea, eb, min_iou)
        for x, y, _v in pairs:
            sa.append(x["state"])
            sb.append(y["state"])
    return {
        "n_matched": len(sa),
        "kappa": AG.cohen_kappa(sa, sb),
        "alpha": AG.krippendorff_alpha_nominal(sa, sb),
    }


def boundary_by_class(
    A: dict, B: dict, min_iou: float = 0.1, tol: float = 0.25
) -> dict:
    out = {}
    for cls in C.GATE_CLASSES:
        ious, within = [], 0
        for item in sorted(set(A) & set(B)):
            ea = [e for e in A[item] if e["class"] == cls]
            eb = [e for e in B[item] if e["class"] == cls]
            pairs, _ua, _ub = AG.match_events(ea, eb, min_iou)
            for x, y, v in pairs:
                ious.append(v)
                within += (
                    round(abs(x["start"] - y["start"]), 3) <= tol
                    and round(abs(x["end"] - y["end"]), 3) <= tol
                )
        out[cls] = {
            "n_matched": len(ious),
            "mean_iou_matched": (sum(ious) / len(ious)) if ious else None,
            "boundary_agreement": (within / len(ious)) if ious else None,
        }
    return out


def _count(rep: dict, cls: str, key: str) -> int:
    return int((rep["per_class"].get(cls) or {}).get(key, 0))


# ------------------------------------------------------------------------------------------------ gates
def qg1(intra: dict, A: dict, B: dict) -> dict:
    res = {}
    for cls in C.GATE_CLASSES:
        n1, n2 = _count(intra, cls, "n_a"), _count(intra, cls, "n_b")
        ps = presence_stats(A, B, cls)
        dice = (intra["per_class"].get(cls) or {}).get("dice")  # verbatim
        rec = {
            "n_events_pass1": n1,
            "n_events_pass2": n2,
            "kappa": ps["kappa"],
            "alpha": ps["alpha"],
            "dice": dice,
        }
        if min(n1, n2) < C.QG1_MIN_EVENTS:
            rec["status"] = "not_assessable"
            rec["reason"] = (
                f"fewer than {C.QG1_MIN_EVENTS} events of the class in a pass"
            )
        elif ps["kappa"] is None or dice is None:
            rec["status"] = "fail"
            rec["reason"] = (
                "undefined statistic on the intra comparison (not_available) counts as a failed gate"
            )
        elif ps["kappa"] >= C.QG1_KAPPA and dice >= C.QG1_DICE:
            rec["status"] = "pass"
        else:
            rec["status"] = "fail"
            rec["reason"] = "below threshold"
        res[cls] = rec
    return res


def qg2_rep(A: dict, B: dict) -> dict:
    r = rep_state_kappa(A, B)
    if r["kappa"] is None:
        return {
            **r,
            "status": "fail",
            "reason": "kappa undefined on the intra comparison (not_available) counts as failed",
        }
    return {**r, "status": "pass" if r["kappa"] >= C.QG2_KAPPA else "fail"}


def markers_block(intra_m: dict) -> dict:
    """QG-5 first; QG-2 for markers is read only if QG-5 passes (draft section 7)."""
    q5_desc = intra_m["function_kappa_descriptive_only"]  # verbatim
    q5 = {
        "function_kappa_descriptive_only": q5_desc,
        "detection_f1": intra_m["detection_f1"],
        "status": "fail" if q5_desc else "pass",
    }
    if q5_desc:
        q2 = {
            "status": "not_read",
            "reason": "QG-5 failed on the intra comparison",
            "kappa": intra_m["kappa"],
        }
    elif intra_m["kappa"] is None:
        q2 = {
            "status": "fail",
            "kappa": None,
            "alpha": intra_m["alpha"],
            "reason": intra_m["not_available_reason"] or "kappa undefined",
            "n_matched_spans": intra_m["n_matched_spans"],
        }
    else:
        q2 = {
            "status": "pass" if intra_m["kappa"] >= C.QG2_KAPPA else "fail",
            "kappa": intra_m["kappa"],
            "alpha": intra_m["alpha"],
            "n_matched_spans": intra_m["n_matched_spans"],
        }
    return {"qg5": q5, "qg2": q2}


def _na_reason_subset(rep: dict, cls: str | None) -> list[str]:
    reasons = []
    if rep["subset_below_minimum"]:  # verbatim
        reasons.append(
            f"subset_below_minimum (n_items_common = {rep['n_items_common']})"
        )
    return reasons


def _combine(per: dict[str, tuple[str, str | None]]) -> tuple[str, list[str]]:
    """Per-comparison outcomes (status, reason) -> gate status. Both comparisons (vs pass 1, vs pass 2) must pass.
    Any assessable comparison that fails -> fail; otherwise any not-assessable comparison -> not_assessable."""
    if any(st == "fail" for st, _ in per.values()):
        return "fail", [f"{n}: {r}" for n, (st, r) in per.items() if st == "fail"]
    if any(st == "not_assessable" for st, _ in per.values()):
        return "not_assessable", [
            f"{n}: {r}" for n, (st, r) in per.items() if st == "not_assessable"
        ]
    return "pass", []


def qg3_class(subs: dict, A_r2: dict | None, passes: dict, cls: str) -> dict:
    """``subs``: {'pass1': compute(pass1, r2, subset), 'pass2': ...}; ``passes``: {'pass1': A, 'pass2': B} dicts.

    Per comparison: not assessable iff ``subset_below_minimum`` (verbatim) or the PASS has fewer than 10 events of the
    class on the common subset items (``per_class[c].n_a`` of ``compute(pass, rater2, subset)``). Rater 2's own count is
    reported but never makes the gate not assessable: a rater 2 who hears fewer events is a disagreement (it can fail
    the gate), not a lack of check. Assessable with an undefined kappa (both labels constant on the same value) counts
    as FAILED, the same rule as an undefined statistic on the intra comparison."""
    if subs is None:
        return {"status": "not_assessable", "reasons": ["no rater-2 file"]}
    rec = {"vs": {}}
    per = {}
    for name, sub in subs.items():
        n_a, n_b = _count(sub, cls, "n_a"), _count(sub, cls, "n_b")
        ps = presence_stats(passes[name], A_r2, cls)
        rec["vs"][name] = {
            "subset_below_minimum": sub["subset_below_minimum"],
            "n_items_common": sub["n_items_common"],
            "n_events_pass": n_a,
            "n_events_rater2": n_b,
            "kappa": ps["kappa"],
            "alpha": ps["alpha"],
        }
        if sub["subset_below_minimum"]:
            per[name] = ("not_assessable", "subset_below_minimum")
        elif n_a < C.QG3_MIN_EVENTS:
            per[name] = (
                "not_assessable",
                f"fewer than {C.QG3_MIN_EVENTS} events of the class in the pass on the subset items",
            )
        elif ps["kappa"] is None:
            per[name] = ("fail", "kappa undefined in the subset (not_available) counts as failed")
        elif ps["kappa"] < C.QG3_KAPPA:
            per[name] = ("fail", "kappa below threshold")
        else:
            per[name] = ("pass", None)
    status, reasons = _combine(per)
    return {**rec, "status": status, **({"reasons": reasons} if reasons else {})}


def qg3_markers(msubs: dict | None) -> dict:
    """Per comparison: not assessable iff ``subset_below_minimum`` or kappa is None WITH ``not_available_reason``
    (fewer than MIN_MARKER_SPANS matched spans); failed iff ``function_kappa_descriptive_only`` (verbatim), kappa
    undefined with enough matched spans (constant functions, as the intra rule), or kappa < 0.40."""
    if msubs is None:
        return {"status": "not_assessable", "reasons": ["no rater-2 file"]}
    rec = {"vs": {}}
    per = {}
    for name, m in msubs.items():
        rec["vs"][name] = {
            "subset_below_minimum": m["subset_below_minimum"],  # verbatim
            "function_kappa_descriptive_only": m[
                "function_kappa_descriptive_only"
            ],  # verbatim
            "kappa": m["kappa"],
            "alpha": m["alpha"],
            "n_matched_spans": m["n_matched_spans"],
            "not_available_reason": m["not_available_reason"],
        }
        if m["subset_below_minimum"]:
            per[name] = ("not_assessable", "subset_below_minimum")
        elif m["kappa"] is None and m["not_available_reason"]:
            per[name] = ("not_assessable", m["not_available_reason"])
        elif m["function_kappa_descriptive_only"]:
            per[name] = ("fail", "function_kappa_descriptive_only is true")
        elif m["kappa"] is None:
            per[name] = ("fail", "kappa undefined in the subset (not_available) counts as failed")
        elif m["kappa"] < C.QG3_KAPPA:
            per[name] = ("fail", "kappa below threshold")
        else:
            per[name] = ("pass", None)
    status, reasons = _combine(per)
    return {**rec, "status": status, **({"reasons": reasons} if reasons else {})}


def reference_status(g1: dict, g3: dict) -> tuple[str, list[str]]:
    """Per class: descriptive (QG-1 not pass, or QG-3 failed) > conditional (QG-3 not assessable) > ok."""
    notes = []
    if g1["status"] != "pass":
        notes.append(
            f"QG-1 {g1['status']}: class descriptive ({g1.get('reason', '')})".strip()
        )
        return "descriptive", notes
    if g3["status"] == "fail":
        notes.append("QG-3 failed: class descriptive")
        return "descriptive", notes
    if g3["status"] == "not_assessable":
        notes.append(
            "inter-rater gate not assessable: " + "; ".join(g3.get("reasons", []))
        )
        return "conditional", notes
    return "ok", notes


def marker_status(mb: dict, g3m: dict) -> tuple[str, list[str]]:
    notes = []
    if mb["qg5"]["status"] != "pass":
        return "not_evaluable", [
            "QG-5: function_kappa_descriptive_only is true on the intra comparison"
        ]
    if mb["qg2"]["status"] != "pass":
        return "not_evaluable", ["QG-2 (markers) " + mb["qg2"]["status"]]
    if g3m["status"] == "fail":
        return "descriptive", [
            "QG-3 (markers) failed: PG3 descriptive, Holm member not evaluable"
        ]
    if g3m["status"] == "not_assessable":
        return "conditional", [
            "inter-rater gate not assessable: " + "; ".join(g3m.get("reasons", []))
        ]
    return "ok", notes


def build_gates(
    pass1: Path, pass2: Path, rater2: Path | None = None, *, require_hashed: bool = True
) -> dict:
    files = {
        "pass1": Path(pass1),
        "pass2": Path(pass2),
        **({"rater2": Path(rater2)} if rater2 else {}),
    }
    if require_hashed:
        for p in files.values():
            C.check_sidecar(p)
    A, B = AG.read_annotations(files["pass1"]), AG.read_annotations(files["pass2"])
    intra = AG.compute(files["pass1"], files["pass2"], mode="intra")
    intra_m = AG.compute_markers(files["pass1"], files["pass2"], mode="intra")
    g1 = qg1(intra, A, B)
    rep2 = qg2_rep(A, B)
    mb = markers_block(intra_m)

    subs = msubs = R = None
    if rater2:
        R = AG.read_annotations(files["rater2"])
        subs = {
            n: AG.compute(files[n], files["rater2"], mode="subset")
            for n in ("pass1", "pass2")
        }
        msubs = {
            n: AG.compute_markers(files[n], files["rater2"], mode="subset")
            for n in ("pass1", "pass2")
        }
    passes = {"pass1": A, "pass2": B}
    classes = {}
    for cls in C.GATE_CLASSES:
        g3 = qg3_class(subs, R, passes, cls)
        status, notes = reference_status(g1[cls], g3)
        classes[cls] = {
            "qg1": g1[cls],
            "qg3": g3,
            "reference_status": status,
            "notes": notes,
        }
    # QG-2 for REP gates the convention contrast (PGS4/PGS5) only
    g3m = qg3_markers(msubs)
    m_status, m_notes = marker_status(mb, g3m)
    return {
        "spec": SPEC,
        "inputs": {
            k: {"name": v.name, "sha256": C.H.sha256_file(v)} for k, v in files.items()
        },
        "classes": classes,
        "qg2_repetition_state": rep2,
        "rep_state_gate_ok": rep2["status"] == "pass",
        "markers": {
            **mb,
            "qg3": g3m,
            "pg3_reference_status": m_status,
            "notes": m_notes,
        },
        "qg4_boundaries": {
            "overall": {
                "boundary_agreement": intra["boundary_agreement"],
                "mean_iou_matched": intra["mean_iou_matched"],
            },
            "by_class": boundary_by_class(A, B),
        },
        "agreement": {
            "intra": intra,
            "intra_markers": intra_m,
            "subset": subs,
            "subset_markers": msubs,
        },
        "rater2_available": rater2 is not None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pass1", type=Path, required=True)
    ap.add_argument("--pass2", type=Path, required=True)
    ap.add_argument("--rater2", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--no-hash-check",
        action="store_true",
        help="development only: skip the .sha256 sidecar check",
    )
    a = ap.parse_args(argv)
    g = build_gates(a.pass1, a.pass2, a.rater2, require_hashed=not a.no_hash_check)
    a.out.write_text(C.dumps(g) + "\n", "utf8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
