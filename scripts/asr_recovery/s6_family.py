"""Pre-declared family (protocol 6): best variant of each O vs the ctx_LR(B / P0) control, per task; Holm over 6 tests."""

import _pin  # noqa: F401  (MUST be first)
import json

from mrcd.asr_recovery.common import OUT, TASKS
from mrcd.context_eval.metrics import holm

CAND = {
    "O1": ("P1", "P2"),
    "O2": ("O2_S", "O2_C+S"),
    "O3": ("O3_C+AC_asr", "O3_C+AC_vad"),
}


def main():
    o1 = json.loads((OUT / "run_o1" / "results.json").read_text("utf8"))
    o23 = json.loads((OUT / "run_o2_o3" / "results.json").read_text("utf8"))
    fam = []
    for task in TASKS:
        for o, cands in CAND.items():
            best = None
            for c in cands:
                if o == "O1":
                    m = o1["tasks"][task]["variants"][c]["models"]["ctx_LR"]
                    f1, pr = m["w_f1"], m["paired_f1w_vs_P0"]
                else:
                    m = o23["tasks"][task]["models"][c]
                    f1, pr = m["w_f1"], m["paired_vs_ctx_LR_B"]
                if best is None or f1 > best["f1"]:
                    best = {"task": task, "O": o, "variant": c, "f1": f1, **pr}
            fam.append(best)
    adj = holm([f["p_boot"] for f in fam])
    for f, a in zip(fam, adj):
        f["p_holm"] = a
        f["verdict"] = (
            "supported after Holm"
            if a == a and a < 0.05 and (f["ci"][0] > 0 or f["ci"][1] < 0)
            else "not demonstrated"
        )
    (OUT / "family.json").write_text(json.dumps(fam, indent=1, default=float), "utf8")
    for f in fam:
        print(f"{f['task']:11s} {f['O']} {f['variant']:12s} F1={f['f1']:.3f} d={f['delta']:+.3f} ci=[{f['ci'][0]:+.3f},{f['ci'][1]:+.3f}] p={f['p_boot']:.4f} holm={f['p_holm']:.4f} {f['verdict']}")


if __name__ == "__main__":
    main()
