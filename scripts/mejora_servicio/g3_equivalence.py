"""G3: incremental == offline with decision profile o2026_11b, real Whisper, full HTTP stack in process.

    .venv\\Scripts\\python.exe -m scripts.mejora_servicio.g3_equivalence [--skip-long]

Reuses the S8 rig (``scripts/s8_incremental/equivalence_real.py``): device cpu, p0 and p1 engines preloaded, warm-up,
offline reference on the same engine worker. Cases fixed by PROTOCOL.md 1.6 (G3 ii-iii). Output: pass/fail, digests,
event counts and timings only (no audio, no text).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.s8_incremental import equivalence_real as E  # noqa: E402
from mrcd.service.incremental.session import IncrementalConfig  # noqa: E402
from tests.incremental_rig import split, strip_volatile  # noqa: E402

OUT = REPO / "results" / "mejora_servicio_20261102" / "g3"


def kinds(doc) -> dict:
    out: dict = {}
    for e in (doc or {}).get("events", []):
        out[e["score"]["kind"]] = out.get(e["score"]["kind"], 0) + 1
    return out


def case(rig, name, pcm, opts_q, opts, batches=None, **drive):
    off = rig.offline(pcm, **opts)
    st = rig.run(pcm, batches or split(pcm, 2560), opts_q, **drive)
    res = st["result"]
    ok = st["status"] == off["status"] == "completed" and strip_volatile(
        res
    ) == strip_volatile(off)
    k_off, k_inc = kinds(off), kinds(res)
    ok = (
        ok
        and k_off == k_inc
        and k_off.get("unit_pipeline_model_score", 0)
        + k_off.get("lexical_rule_indicator", 0)
        + k_off.get("text_route_logistic_score", 0)
        > 0
    )
    cache = getattr(rig.registry.get(st["sid"]), "cache_stats", None)
    E.record(
        name,
        ok,
        seconds=round(len(pcm) / 2 / E.SR, 3),
        offline_sha=E.sha(off),
        incremental_sha=E.sha(res),
        kinds=k_off,
        cache=cache,
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--only-long", action="store_true")
    ap.add_argument("--tag", default="g3_equivalence")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    E.OUT_FILE = OUT / f"{args.tag}.json"
    c60, a24, syn = E.pcm_of(E.C60), E.pcm_of(E.A24), E.pcm_of(E.SYN)
    rig = E.RealRig(
        IncrementalConfig(max_batch_bytes=64 << 20, max_sessions=6),
        ["p0", "p1_disfluent_prompt"],
    )
    q = "?decision_profile=o2026_11b&include_text=true&decision=true"
    o = dict(decision_profile="o2026_11b", include_text=True, decision=True)
    if not args.only_long:
        cuts = {
            "c01_13.0s": E.cut(c60, 13.0),
            "c02_13.6s_ameresco": a24,
            "c04_30.0s": E.cut(c60, 30.0),
            "c06_37.0s": E.cut(c60, 37.0),
            "c07_45.466s_synthetic": syn,
            "c08_60.0s": c60,
        }
        for name, pcm in cuts.items():
            case(rig, f"{name}|o2026_11b", pcm, q, o)
        b2 = split(syn, 32000)
        order = list(range(len(b2)))
        random.Random(5).shuffle(order)
        case(
            rig,
            "c11_partition_random|o2026_11b",
            syn,
            q,
            o,
            split(syn, rng_seed=20261101),
        )
        case(rig, "c12_out_of_order|o2026_11b", syn, q, o, b2, order=order)
        case(rig, "c13_duplicates|o2026_11b", syn, q, o, b2, dup=True)
    if not args.skip_long:
        long_pcm = (c60 * 10)[: int(599.9 * E.SR) * 2]
        case(
            rig,
            "c08_599.9s|o2026_11b",
            long_pcm,
            "?decision_profile=o2026_11b&include_text=true",
            dict(decision_profile="o2026_11b", include_text=True),
            split(long_pcm, 32000),
        )
    ok = all(r["pass"] for r in E.RESULTS)
    print(
        json.dumps(
            {
                "passed": sum(r["pass"] for r in E.RESULTS),
                "total": len(E.RESULTS),
                "all_pass": ok,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
