#!/usr/bin/env python
"""Secondary arm cw2 of the confirmatory run (annex: results/confirmatoria_20261107/PROTOCOL_ANNEX_20261107_SECONDARY_CW2.md).

Sub-commands
------------
verify    check the annex sha256, the annex manifest and everything ``run_confirmatoria.py verify`` checks; runs nothing
run       the secondary arm: CrisperWhisper 2.0 (profile ``cw2``) words + the SAME lexical rule as the primary ``rule``.
          Refuses unless the primary result is complete, the annex hash matches and the frozen system/protocol hold.
          Writes CONFIRMATORY_RESULT_SECONDARY_CW2.json (aggregates only). On any failure it writes a ``not_available``
          record with the reason (never 0, no retry).

Design constraint: ``run_confirmatoria.py`` is frozen by hash in PROTOCOL.sha256.json, so it is NOT edited. This file
imports it and reuses its functions unchanged (reference, statistics, caching, rule, mapping). Nothing here writes to
the primary outputs; it only READS ``CONFIRMATORY_RESULT.json``, ``RUN.lock.json`` and ``local/nocando_unit_predictions.csv``.
Nothing here is fitted, selected or thresholded.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "run_confirmatoria", _HERE / "run_confirmatoria.py"
)
H = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_confirmatoria", H)
_spec.loader.exec_module(
    H
)  # sets OMP_NUM_THREADS=4 / HF_HUB_OFFLINE=1 exactly as the primary does

SELF = Path(__file__).resolve()
ANNEX_MD = H.OUT_DIR / "PROTOCOL_ANNEX_20261107_SECONDARY_CW2.md"
ANNEX_FREEZE = H.OUT_DIR / "PROTOCOL_ANNEX_SECONDARY_CW2.sha256.json"
RESULT_SECONDARY = "CONFIRMATORY_RESULT_SECONDARY_CW2.json"
LOCK_SECONDARY = "RUN_SECONDARY_CW2.lock.json"
CONCLUSIONS_NAME = "PRIMARY_CONCLUSIONS.md"
PRED_CSV = Path("local") / "nocando_unit_predictions.csv"

# ---- the cw2 system, pinned (the annex repeats these values; a test checks they agree)
CW2_PROFILE = "cw2"
CW2_MODEL_ID = "nyralabs/CrisperWhisper2.0_large"
CW2_MODEL_REVISION = "f4334f6e8193f2691212d49b20fa12d370e13896"
#: sha256 over the sorted lines ``<file name>\t<file sha256>\n`` of the 14 files of the pinned snapshot
CW2_WEIGHTS_MANIFEST_SHA256 = "b22cf7a5208e73405514f32b9e7686019364771d51a8a8683f105596d88e7ac9"
CW2_DEVICE = "cuda"
CW2_COMPUTE_TYPE = "float16"
#: the classifier output of this arm is discarded; the cheapest valid decision profile is used (words only are scored)
CW2_DECISION_PROFILE_UNUSED = "v0"

FAMILY = [
    "S1_cw2_rule_vs_default_repetition_f1",
    "S2_cw2_rule_vs_default_filler_recall",
]
ALPHA = 0.05


# ------------------------------------------------------------------------------------------------ pinning
def snapshot_manifest(path: Path) -> dict:
    """sha256 of every file of a model snapshot directory (symlinks resolved) and one combined digest."""
    files = {}
    for p in sorted(Path(path).iterdir()):
        if p.is_file():
            files[p.name] = H.sha256_file(p.resolve())
    digest = H.sha256_bytes(
        "".join(f"{n}\t{h}\n" for n, h in sorted(files.items())).encode()
    )
    return {"files": files, "combined_sha256": digest}


def local_snapshot_dir(
    model_id: str = CW2_MODEL_ID, revision: str = CW2_MODEL_REVISION
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, revision=revision, local_files_only=True))


# ------------------------------------------------------------------------------------- annex / freeze checks
def check_annex() -> dict:
    """The annex hash must match the frozen value; the secondary harness must match too. No-op on the primary files."""
    if not ANNEX_FREEZE.is_file():
        raise SystemExit(
            "PROTOCOL_ANNEX_SECONDARY_CW2.sha256.json missing: the secondary arm is not frozen"
        )
    fz = json.loads(ANNEX_FREEZE.read_text("utf8"))
    if not ANNEX_MD.is_file() or H.sha256_text_file(ANNEX_MD) != fz["annex_sha256"]:
        raise SystemExit(
            "the annex does not match its frozen sha256: the secondary arm refuses to run"
        )
    for rel, want in fz.get("files", {}).items():
        p = H.REPO / rel
        if not p.is_file() or H.sha256_text_file(p) != want:
            raise SystemExit(f"{rel} does not match the annex manifest")
    return fz


def verify_secondary(a, *, check_data: bool = True) -> dict:
    state = H.verify_all(
        a, check_data=check_data
    )  # protocol, primary harness, system tree, data, committed state
    fz = check_annex()
    for p in (ANNEX_MD, ANNEX_FREEZE, SELF):
        if not H.committed_clean(p):
            raise SystemExit(
                f"{p.name} must be committed and unmodified before the secondary run"
            )
    return {**state, "annex_freeze": fz}


def primary_complete(out: Path) -> dict:
    """The primary result must exist, be valid JSON, match the hash in a ``completed`` lock and have the unit CSV."""
    res, lock = out / H.RESULT_NAME, out / H.LOCK_NAME
    if not res.is_file() or not lock.is_file() or not (out / PRED_CSV).is_file():
        raise SystemExit(
            "the primary result is not complete: the secondary arm refuses to run before it"
        )
    lk = json.loads(lock.read_text("utf8"))
    if lk.get("status") != "completed" or lk.get("result_sha256") != H.sha256_file(res):
        raise SystemExit(
            "RUN.lock.json is not 'completed' for this primary result: the secondary arm refuses"
        )
    json.loads(res.read_text("utf8"))
    return lk


def acquire_secondary_lock(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (out / RESULT_SECONDARY).exists() or (out / LOCK_SECONDARY).exists():
        raise SystemExit(
            "the secondary arm was already started: it runs once, with no retries"
        )
    fd = os.open(out / LOCK_SECONDARY, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf8") as f:
        json.dump({**payload, "status": "started"}, f, indent=1)


def read_primary_predictions(out: Path) -> dict:
    """Primary unit predictions (ids and 0/1 only) keyed by unit_id."""
    rows = {}
    with (out / PRED_CSV).open(newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            rows[r["unit_id"]] = {
                k: int(v)
                for k, v in r.items()
                if k not in ("unit_id", "recording", "speaker")
            }
    return rows


# ------------------------------------------------------------------------------------------------ the arm
def not_available(reason: str, **extra) -> dict:
    return {"status": "not_available", "reason": reason, **extra}


def score_cw2(cell, analyzer, options_cls, progress=print) -> dict:
    """cw2 words + the primary rule on every recording. Any recording error makes the whole arm not available."""
    by_rec: dict[str, list] = {}
    for i, u in enumerate(cell.units):
        by_rec.setdefault(u["recording"], []).append((i, u))
    pred = {c: np.zeros(len(cell.units), int) for c in ("repetition", "filler_word")}
    dropped = 0
    for rid in sorted(cell.recordings):
        units = by_rec.get(rid, [])
        progress(f"[cw2 {cell.name}] {rid} ({len(units)} units)")
        r = H.run_recording(
            analyzer,
            options_cls,
            cell.recordings[rid]["audio"](),
            [CW2_DECISION_PROFILE_UNUSED],
        )
        if r["errors"] or r["words"] is None:
            return not_available(
                f"analysis failed on {rid}: {r['errors'] or 'no ASR output'}"
            )
        for c, lst in H.rule_events(r["words"]).items():
            hit, dr = H.assign_to_units(units, lst)
            dropped += dr
            for i in hit:
                pred[c][i] = 1
    return {"status": "ok", "pred": pred, "dropped_events": dropped}


def _gate(y, rec, spk) -> dict:
    pos = y == 1
    g = {
        "positives": int(pos.sum()),
        "recordings_with_positives": int(len(set(rec[pos]))),
        "speakers_with_positives": int(len(set(spk[pos]))),
    }
    g["evaluable"] = bool(
        g["positives"] >= H.GATE_MIN_POS
        and g["recordings_with_positives"] >= H.GATE_MIN_RECORDINGS
        and g["speakers_with_positives"] >= H.GATE_MIN_SPEAKERS
    )
    return g


def _paired(c1, c0, fn):
    return H.paired_delta(c1, c0, fn)


def evaluate_secondary(cell, cw2_pred: dict, primary: dict) -> dict:
    """S1, S2 (Holm within {S1, S2}) and the descriptive cw2+rule vs p0 rule row. Same statistics as the primary."""
    u = cell.units
    ids = [x["unit_id"] for x in u]
    if set(ids) != set(primary):
        raise SystemExit(
            "the primary prediction file does not cover exactly the units of the rebuilt cell"
        )
    rec = np.array([x["recording"] for x in u])
    spk = np.array([x["speaker"] for x in u])
    y_rep = np.array([x["rep_literal"] for x in u])
    y_fill = np.array([int(x["n_fillers"] > 0) for x in u])
    col = lambda k: np.array([primary[i][k] for i in ids])  # noqa: E731
    if not (
        np.array_equal(col("y_rep"), y_rep) and np.array_equal(col("y_fill"), y_fill)
    ):
        raise SystemExit(
            "the primary reference columns differ from the rebuilt reference"
        )
    d_rep, r_rep, d_fill, r_fill = (
        col("default_rep"),
        col("rule_rep"),
        col("default_fill"),
        col("rule_fill"),
    )
    c_rep, c_fill = cw2_pred["repetition"], cw2_pred["filler_word"]

    out: dict = {"classes": {}}
    tests, plist = {}, []
    # ---- repetition: F1 over all units
    g = _gate(y_rep, rec, spk)
    rep = {"gate": g, "metrics": {}}
    if g["evaluable"]:
        C = {
            k: H.group_counts(y_rep, v, rec)
            for k, v in (("cw2_rule", c_rep), ("default", d_rep), ("p0_rule", r_rep))
        }
        for k, c in C.items():
            t = c.sum(0)
            rep["metrics"][k] = {
                "tp": int(t[0]),
                "fp": int(t[1]),
                "fn": int(t[2]),
                "precision": H.point(c, H.precision_of),
                "recall": H.point(c, H.recall_of),
                "f1": H.point(c, H.f1_of),
                "f1_ci95": H.boot_ci(c, H.f1_of),
            }
        tests[FAMILY[0]] = {
            **_paired(C["cw2_rule"], C["default"], H.f1_of),
            "sides": "two-sided",
        }
        rep["descriptive_cw2_rule_vs_p0_rule"] = {
            k: v
            for k, v in _paired(
                C["cw2_rule"], C["p0_rule"], H.f1_of
            ).items()
            if k in ("delta", "ci", "n_boot")
        }
    out["classes"]["repetition"] = rep
    # ---- filler: recall over reference-positive units
    g = _gate(y_fill, rec, spk)
    fil = {
        "gate": g,
        "metrics": {},
        "note": "recall on transcript-marked non-lexical filled pauses; no precision, no F1",
    }
    if g["evaluable"]:
        pos = y_fill == 1
        C = {
            k: H.group_counts(y_fill[pos], v[pos], rec[pos])
            for k, v in (("cw2_rule", c_fill), ("default", d_fill), ("p0_rule", r_fill))
        }
        for k, c in C.items():
            t = c.sum(0)
            fil["metrics"][k] = {
                "tp": int(t[0]),
                "fn": int(t[2]),
                "recall": H.point(c, H.recall_of),
                "recall_ci95": H.boot_ci(c, H.recall_of),
            }
        tests[FAMILY[1]] = {
            **_paired(C["cw2_rule"], C["default"], H.recall_of),
            "sides": "two-sided",
        }
        fil["descriptive_cw2_rule_vs_p0_rule"] = {
            k: v
            for k, v in _paired(C["cw2_rule"], C["p0_rule"], H.recall_of).items()
            if k in ("delta", "ci", "n_boot")
        }
    out["classes"]["filler_word"] = fil
    # ---- Holm within {S1, S2} only (size 2, fixed: a test that cannot be computed still counts)
    for n in FAMILY:
        t = tests.get(n)
        plist.append(float("nan") if t is None else t["p_two"])
    adj = H.holm(plist)
    out["secondary_tests"] = {}
    for n, p, a in zip(FAMILY, plist, adj):
        t = tests.get(n)
        if t is None:
            out["secondary_tests"][n] = {
                "status": "NOT EVALUABLE (gate failed)",
                "p_raw": None,
                "p_holm": None,
                "verdict": "not evaluable",
            }
            continue
        ci_ok = t["ci"][0] > 0 or t["ci"][1] < 0
        t.pop("p_one_gt", None)
        t.update(
            p_raw=p,
            p_holm=a,
            direction="cw2_better"
            if t["delta"] > 0
            else "cw2_worse"
            if t["delta"] < 0
            else "none",
            verdict="supported" if (a < ALPHA and ci_ok) else "not demonstrated",
        )
        out["secondary_tests"][n] = t
    out["holm_family"] = {
        "size": len(FAMILY),
        "members": list(FAMILY),
        "p_raw": plist,
        "p_holm": adj,
        "alpha": ALPHA,
    }
    out["dropped_events_no_unit"] = cw2_pred["dropped_events"]
    return out


def environment_record(snapshot: Path | None) -> dict:
    env = {
        "versions": H.env_versions(),
        "device_requested": CW2_DEVICE,
        "compute_type_expected": CW2_COMPUTE_TYPE,
    }
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        env["torch"] = torch.__version__
        env["cuda_runtime"] = getattr(torch.version, "cuda", None)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        env["cuda_available"], env["gpu_name"], env["torch_error"] = (
            False,
            None,
            f"{type(exc).__name__}: {exc}",
        )
    env["snapshot_dir_name"] = None if snapshot is None else snapshot.name
    return env


def write_result(out: Path, final: dict) -> None:
    (out / RESULT_SECONDARY).write_text(
        json.dumps(final, indent=1, default=float), "utf8"
    )
    lock = json.loads((out / LOCK_SECONDARY).read_text("utf8"))
    lock.update(
        {"status": "completed", "result_sha256": H.sha256_file(out / RESULT_SECONDARY)}
    )
    (out / LOCK_SECONDARY).write_text(json.dumps(lock, indent=1), "utf8")


def run_secondary(
    cell,
    out: Path,
    make_analyzer,
    options_cls,
    header: dict,
    *,
    preflight,
    progress=print,
) -> dict:
    """Core of the arm, injectable for tests. ``preflight`` -> (None | reason, env dict); a reason means not available."""
    reason, env = preflight()
    final = {
        "label": "SECONDARY ARM cw2 (annex 2026-11-07); reported separately; never replaces or alters the primary result",
        **header,
        "environment": env,
        "finished_utc": None,
    }
    if reason is None:
        try:
            analyzer = make_analyzer()
            res = score_cw2(cell, analyzer, options_cls, progress)
            if res["status"] == "ok":
                final.update(
                    status="ok",
                    **evaluate_secondary(
                        cell,
                        {**res["pred"], "dropped_events": res["dropped_events"]},
                        read_primary_predictions(out),
                    ),
                )
            else:
                final.update(res)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - any crash is "not available" with its reason; no retry
            final.update(not_available(f"{type(exc).__name__}: {exc}"))
    else:
        final.update(not_available(reason))
    final["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_result(out, final)
    return final


def _preflight_real():
    snap = None
    try:
        import torch

        if not torch.cuda.is_available():
            return "CUDA is not available on this machine", environment_record(None)
    except Exception as exc:  # noqa: BLE001
        return f"torch import failed: {type(exc).__name__}: {exc}", environment_record(
            None
        )
    try:
        snap = local_snapshot_dir()
        m = snapshot_manifest(snap)
    except Exception as exc:  # noqa: BLE001
        return (
            f"pinned cw2 snapshot not available locally: {type(exc).__name__}: {exc}",
            environment_record(snap),
        )
    env = environment_record(snap)
    env["weights_manifest_sha256_at_run"] = m["combined_sha256"]
    if m["combined_sha256"] != CW2_WEIGHTS_MANIFEST_SHA256:
        return "local cw2 weights do not match the pinned manifest hash", env
    return None, env


def cmd_verify(a) -> int:
    verify_secondary(a)
    print(
        "OK: primary frozen values, annex sha256 and annex manifest match; committed state clean"
    )
    return 0


def cmd_run(a) -> int:
    state = verify_secondary(a)
    out = H.OUT_DIR
    primary_complete(out)
    concl = out / CONCLUSIONS_NAME
    if not concl.is_file() or concl.stat().st_size == 0 or not H.committed_clean(concl):
        raise SystemExit(f"{CONCLUSIONS_NAME} must be written and committed BEFORE the secondary arm runs (annex section 6)")
    acquire_secondary_lock(
        out,
        {
            "system_commit": state["system_freeze"]["system_commit"],
            "annex_sha256": state["annex_freeze"]["annex_sha256"],
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": sys.argv[1:],
        },
    )
    from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer

    cell = H.build_nocando_cell(H.corpora_root(a))
    header = {
        "system_commit": state["system_freeze"]["system_commit"],
        "git_head_at_run": state["head"],
        "annex_sha256": state["annex_freeze"]["annex_sha256"],
        "primary_conclusions_sha256": H.sha256_text_file(concl),
        "protocol_sha256": state["protocol_freeze"]["protocol_sha256"],
        "primary_harness_sha256": state["protocol_freeze"]["harness_sha256"],
        "secondary_harness_sha256": H.sha256_text_file(SELF),
        "system": {
            "asr_profile": CW2_PROFILE,
            "model_id": CW2_MODEL_ID,
            "model_revision": CW2_MODEL_REVISION,
            "weights_manifest_sha256_pinned": CW2_WEIGHTS_MANIFEST_SHA256,
            "rule": "same as primary 'rule'",
            "licence": "CrisperWhisper weights and outputs: non-commercial research only; no transcript is stored",
        },
    }
    final = run_secondary(
        cell,
        out,
        lambda: MrcdAnalyzer(max_duration_s=3600.0, device=CW2_DEVICE),
        partial(AnalysisOptions, asr_profile=CW2_PROFILE),
        header,
        preflight=_preflight_real,
    )
    print("done:", out / RESULT_SECONDARY, "status:", final["status"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("verify", "run"):
        p = sub.add_parser(n)
        p.add_argument(
            "--corpora-root",
            default=None,
            help="root of the corpus repository (or $MRCD_CORPORA_ROOT)",
        )
        p.add_argument(
            "--system-commit",
            required=True,
            help="commit sha of the frozen system (>= 12 hex chars)",
        )
    a = ap.parse_args(argv)
    return {"verify": cmd_verify, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
