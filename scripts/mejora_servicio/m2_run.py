"""M2 measurement: the service document of each SHORT DIHANA turn (folds 1-4, 2 s <= duration < 13 s), analysed alone.

    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\m2_run.py --profile o2026_11 --part 0 --nparts 2

The analyzer runs with ``min_duration_s=2.0`` (the opt-in short-audio path: no rejection; audio shorter than one 3 s
window is zero-padded at the end for the window features only; everything else is the unchanged service code). Options =
service defaults except ``decision_profile`` (the profile that is the service default after the M1 decision; for
``o2026_11b`` the repetition route is the one M1 selected). Resumable: one JSON line per turn.
"""

import common as K  # noqa: F401  (pins OMP before numpy)

import argparse
import json
import time
import wave

from mrcd.service import decision_profile as DP
from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer
from mrcd.service.contract import validate_document

import freeze

M2 = K.OUT / "m2"
LO_S, HI_S = 2.0, 13.0


def short_units() -> list[dict]:
    pool = K.RC.load_pool()  # folds 1-4 only (asserted)
    out = []
    for u in pool:
        with wave.open(str(K.WAV_DIR / f"{u['unit_id']}.wav")) as w:
            d = w.getnframes() / w.getframerate()
        if LO_S <= d < HI_S:
            out.append({"unit_id": u["unit_id"], "speaker": u["speaker"], "dur_s": d})
    return sorted(out, key=lambda r: r["unit_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=[DP.PROFILE_O, DP.PROFILE_OB])
    ap.add_argument("--route", default=None, choices=[None, *DP.REPETITION_ROUTES])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    freeze.verify()
    if args.profile == DP.PROFILE_OB:
        if not args.route:
            raise SystemExit(
                "--route is required for o2026_11b (the route M1 selected)"
            )
        DP.OB_REPETITION_ROUTE = args.route
    units = short_units()
    tag = args.profile + (f"_{args.route}" if args.route else "")
    dest = M2 / "runs" / tag
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / f"part{args.part}of{args.nparts}.jsonl"
    done = set()
    if out_file.exists():
        done = {
            json.loads(line)["unit_id"]
            for line in out_file.read_text("utf8").splitlines()
            if line.strip()
        }
    an = MrcdAnalyzer(device="cpu", min_duration_s=LO_S)
    opts = AnalysisOptions(decision_profile=args.profile)
    an.preload(opts)
    todo = [
        u
        for i, u in enumerate(units)
        if i % args.nparts == args.part and u["unit_id"] not in done
    ]
    if args.limit:
        todo = todo[: args.limit]
    with out_file.open("a", encoding="utf8") as fh:
        for i, u in enumerate(todo):
            data = (K.WAV_DIR / f"{u['unit_id']}.wav").read_bytes()
            t0 = time.perf_counter()
            doc = an.analyze_bytes(data, opts)
            dt = time.perf_counter() - t0
            c = K.compact_doc(doc)
            c.update(
                unit_id=u["unit_id"],
                dur_s=u["dur_s"],
                wall_s=round(dt, 3),
                valid_errors=validate_document(doc),
            )
            fh.write(json.dumps(c) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(
                    f"[{args.part}] {i + 1}/{len(todo)} {u['unit_id']} {u['dur_s']:.1f}s {dt:.1f}s {c['status']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
