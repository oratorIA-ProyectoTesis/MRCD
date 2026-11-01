"""Re-run the comparisons affected by the engine changes, from the FROZEN caches (never runs Whisper).

(a) AMERESCO-32 pilot and AMERESCO-98: V0 and Vc (prolongation) with and without the terminal window,
    (i) from the frozen feature pickles and (ii) end to end through DisfluencyEngine (cached ASR words).
(b) DIHANA fold 4 (tune): V0 and Vc predictions from the new code equal the previous code unit by unit.
(c) DIHANA fold 5 (test): REGRESSION ONLY. The new code must reproduce the already frozen figures
    (V0 0.506/0.540, Vc 0.577/0.631 for filler/prolongation). Nothing here selects a threshold, option or
    default: fold 5 is read, never used to decide.

usage: python rerun_comparisons.py [--out DIR] [--engine-limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "improve"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from common import OUT as MEJ, CLASSES3, CLASSES4, confusion  # noqa: E402
from evalkit import LABELS, pred_argmax, unit_table  # noqa: E402

from mrcd import adaptation as A  # noqa: E402
from mrcd.engine import DisfluencyEngine  # noqa: E402
from mrcd.extractors.linguistic import Word  # noqa: E402
from mrcd.inference import load_ckpt  # noqa: E402
from mrcd.reproducibility import file_sha256  # noqa: E402

CKPT = REPO / "models" / "ckpt_audio_text.pt"


def fmt(c):
    return f"{c['tp']}/{c['fp']}/{c['fn']}/{c['tn']}  P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}"


def off_grid(rec):
    """True for the window the pilot recipe adds at b-3 when it is not on the a-grid."""
    a = rec["target"][0]
    k = (np.asarray(rec["starts"], float) - a) / 0.5
    return np.abs(k - np.round(k)) > 1e-6


def ameresco_cached(head):
    rows = unit_table("ameresco", "zero")
    thr = head.thresholds["prolongation"]["threshold"]
    pi = LABELS.index("prolongation")
    ci = head.classes.index("prolongation")
    out = {}
    for variant in ("with_terminal", "without_terminal"):
        pred = {"V0": {}, "Vc": {}}
        for r in rows:
            rec = r["rec"]
            keep = (
                np.ones(len(rec["starts"]), bool)
                if variant == "with_terminal"
                else ~off_grid(rec)
            )
            probs, pooled = rec["probs"][keep], rec["pooled"][keep]
            pred["V0"][r["unit_id"]] = int((probs.argmax(1) == pi).any())
            pred["Vc"][r["unit_id"]] = int(
                head.window_scores(pooled)[:, ci].max() >= thr
            )
        out[variant] = {}
        for part, sel in (
            ("pilot32", lambda r: r["split"] == "pilot32"),
            ("all98", lambda r: True),
            ("new66", lambda r: r["split"] == "new66"),
        ):
            rs = [r for r in rows if sel(r)]
            y = [r["ref"]["prolongation"] for r in rs]
            out[variant][part] = {
                v: confusion(y, [pred[v][r["unit_id"]] for r in rs])
                for v in ("V0", "Vc")
            }
        out[variant]["_pred"] = pred
    return rows, out


class CachedASR:
    compute_type, device, model_size = "int8", "cpu", "cache"

    def __init__(self, words):
        self.words = words

    def transcribe_chunked(self, audio, chunk_s=8.0):
        return self.words

    def transcribe(self, audio):
        return self.words


def ameresco_engine(rows, limit):
    """End to end through DisfluencyEngine (bs=1), V0 and Vc, one pass with the terminal window."""
    net, norm, labels, prov = load_ckpt(CKPT, "cpu")
    head = A.load_vc(file_sha256(CKPT))
    thr = head.thresholds["prolongation"]["threshold"]
    ci = head.classes.index("prolongation")
    man = {
        m["unit_id"]: m
        for m in json.loads(
            (REPO / "results/baselines_20261030/ameresco/manifest.json").read_text(
                "utf8"
            )
        )
    }
    res = {}
    for k, r in enumerate(rows[:limit] if limit else rows):
        m = man[r["unit_id"]]
        audio, _ = sf.read(m["wav"], dtype="float32")
        words = [
            Word(**w)
            for w in json.loads(
                (MEJ / "asr_cache" / "ameresco" / f"{r['unit_id']}.json").read_text(
                    "utf8"
                )
            )["words"]
        ]
        eng = DisfluencyEngine(
            net, norm, list(labels), CachedASR(words), prov, head=head
        )
        eng._device = "cpu"
        a, b = m["target_start_s"], m["target_end_s"]
        res_ = eng.analyze(audio, keep_windows=True, terminal_window=True, span=(a, b))
        entry = {}
        for variant, keep in (
            ("with_terminal", lambda w: True),
            ("without_terminal", lambda w: not w.terminal),
        ):
            ws = [w for w in res_.windows if keep(w)]
            entry[variant] = {
                "V0": int(any(w.label == "prolongation" for w in ws)),
                "Vc": int(max(w.head_scores["scores"]["prolongation"] for w in ws) >= thr),
            }
        # explicit run without the terminal window must agree with the filtered one
        if k < 3:
            r0 = eng.analyze(audio, terminal_window=False, span=(a, b))
            assert r0.head["classes"]["prolongation"]["present"] == bool(
                entry["without_terminal"]["Vc"]
            ), r["unit_id"]
        res[r["unit_id"]] = entry
        if k % 10 == 0:
            print("engine", k, r["unit_id"], flush=True)
    return res


def dihana(head, fold_split):
    """Per-class predictions for V0 and Vc with the NEW code, plus the previous code path for Vc."""
    rows = unit_table("dihana", "zero", {fold_split})
    ck, _ = A.load_adapted(
        MEJ / "checkpoints" / "ckpt_head_dihana_f123_zero.pt"
    )  # previous code path
    thr_old = json.loads((MEJ / "frozen_head_zero.json").read_text())["thresholds"]
    out = {"n": len(rows), "classes": {}}
    same_head = True
    for c in CLASSES4:
        out["classes"][c] = {}
    preds = {
        "V0": {c: [] for c in CLASSES4},
        "Vc_new": {c: [] for c in CLASSES3},
        "Vc_old": {c: [] for c in CLASSES3},
    }
    for r in rows:
        for c in CLASSES4:
            preds["V0"][c].append(pred_argmax(r, c))
        dec = head.unit_decision(head.window_scores(r["rec"]["pooled"]))
        old = A.head_scores(
            ck["state"],
            r["rec"]["pooled"],
            len(ck["classes"]),
            ck["head_kind"],
            ck["hidden"],
        )
        for i, c in enumerate(CLASSES3):
            preds["Vc_new"][c].append(int(dec[c]["present"]))
            preds["Vc_old"][c].append(int(old[:, i].max() >= thr_old[c]["threshold"]))
    for c in CLASSES4:
        y = [r["ref"][c] for r in rows]
        out["classes"][c]["V0"] = confusion(y, preds["V0"][c])
        if c in CLASSES3:
            out["classes"][c]["Vc"] = confusion(y, preds["Vc_new"][c])
            same_head &= preds["Vc_new"][c] == preds["Vc_old"][c]
    out["vc_new_equals_previous_code_unit_by_unit"] = bool(same_head)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "paridad_live_20261030"))
    ap.add_argument("--engine-limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    head = A.load_vc(file_sha256(CKPT))
    result = {"vc_sha256": head.sha256, "base_sha256": head.base_sha256}
    rows, cached = ameresco_cached(head)
    result["ameresco_cached"] = {
        v: {p: d for p, d in cached[v].items() if p != "_pred"} for v in cached
    }
    eng = ameresco_engine(rows, a.engine_limit)
    result["ameresco_engine"] = {}
    for variant in ("with_terminal", "without_terminal"):
        result["ameresco_engine"][variant] = {}
        for part, sel in (
            ("pilot32", lambda r: r["split"] == "pilot32"),
            ("all98", lambda r: True),
        ):
            rs = [r for r in rows if sel(r) and r["unit_id"] in eng]
            y = [r["ref"]["prolongation"] for r in rs]
            result["ameresco_engine"][variant][part] = {
                v: confusion(y, [eng[r["unit_id"]][variant][v] for r in rs])
                for v in ("V0", "Vc")
            }
    disc = {
        v: [
            r["unit_id"]
            for r in rows
            if r["unit_id"] in eng
            and eng[r["unit_id"]][v_][c] != cached[v_]["_pred"][c][r["unit_id"]]
        ]
        for v_ in ("with_terminal", "without_terminal")
        for c in ("V0", "Vc")
        for v in [f"{v_}/{c}"]
    }
    result["engine_vs_cached_discordant_units"] = disc
    result["dihana_fold4_tune"] = dihana(head, "tune")
    result["dihana_fold5_test_regression_only"] = dihana(head, "test")
    frozen = json.loads((MEJ / "final" / "FINAL_metrics.json").read_text())["tables"]
    reg = {}
    for t in frozen:
        if t["corpus"] == "dihana" and t["version"] in ("V0", "Vc"):
            got = result["dihana_fold5_test_regression_only"]["classes"][
                t["class"]
            ].get(t["version"])
            if got:
                reg[f"{t['class']}/{t['version']}"] = all(
                    got[k] == t[k] for k in ("tp", "fp", "fn", "tn")
                )
        if t["corpus"] == "ameresco" and t["version"] in ("V0", "Vc"):
            got = result["ameresco_cached"]["with_terminal"]["all98"][t["version"]]
            reg[f"ameresco98/{t['version']}"] = all(
                got[k] == t[k] for k in ("tp", "fp", "fn", "tn")
            )
    result["fold5_and_ameresco_match_frozen_final_metrics"] = reg
    result["elapsed_s"] = time.time() - t0
    (out / "rerun_comparisons.json").write_text(
        json.dumps(result, indent=1, default=float), "utf8"
    )
    print("\nAMERESCO (frozen features)")
    for v, d in result["ameresco_cached"].items():
        for p, dd in d.items():
            for k, c in dd.items():
                print(f"  {v:17s} {p:7s} {k}: {fmt(c)}")
    print("AMERESCO (Engine end to end)")
    for v, d in result["ameresco_engine"].items():
        for p, dd in d.items():
            for k, c in dd.items():
                print(f"  {v:17s} {p:7s} {k}: {fmt(c)}")
    print("discordant engine-vs-cached:", {k: len(v) for k, v in disc.items()})
    for tag in ("dihana_fold4_tune", "dihana_fold5_test_regression_only"):
        d = result[tag]
        print(
            tag,
            "n",
            d["n"],
            "Vc new == previous code:",
            d["vc_new_equals_previous_code_unit_by_unit"],
        )
        for c, v in d["classes"].items():
            print("  ", c, {k: fmt(x) for k, x in v.items()})
    print("matches frozen:", reg)


if __name__ == "__main__":
    main()
