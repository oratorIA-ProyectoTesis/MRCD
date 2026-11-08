"""Plan C sample (results/plan_c_20261108/PROTOCOL_EXPLORATORY_C.md section 2). Writes cache/sample.json (git-ignored).

usage: python scripts/plan_c/select_sample.py     (repo .venv, cwd = repo root, PYTHONPATH = repo root)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "verbatim_ft"))
import dihana_io as dio  # noqa: E402

OUT = REPO / "results" / "plan_c_20261108"
SEED = "20261108"
QUOTA = {"R": 24, "F": 12, "N": 12}
MAX_DUR_S = 30.0


def key(uid: str) -> str:
    return hashlib.sha256(f"{SEED}:{uid}".encode()).hexdigest()


def stratum(s: dict, n_fill: int) -> str | None:
    if s["rep_literal"] == 1:
        return "R"
    if s["rep_literal"] == 0 and s["revision"] == 0 and n_fill >= 1:
        return "F"
    if s["kind"] == "neg" and n_fill == 0:
        return "N"
    return None


def main() -> None:
    turns = dio.load_turns()
    frame = dio.load_frame()
    cand = {k: [] for k in QUOTA}
    for s in frame:
        uid = s["unit_id"]
        if dio.duration_s(uid) > MAX_DUR_S:
            continue
        st = stratum(s, dio.n_nonlex_fillers(turns[uid]))
        if st:
            cand[st].append(s)
    used, picked = set(), []
    for st in ("R", "F", "N"):
        n = 0
        for s in sorted(cand[st], key=lambda s: key(s["unit_id"])):
            if n == QUOTA[st]:
                break
            if s["speaker"] in used:
                continue
            used.add(s["speaker"])
            picked.append(
                {
                    "unit_id": s["unit_id"],
                    "speaker": s["speaker"],
                    "fold": int(s["fold"]),
                    "stratum": st,
                }
            )
            n += 1
        if n != QUOTA[st]:
            raise SystemExit(f"stratum {st}: only {n} units")
    dio.assert_folds(p["fold"] for p in picked)
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    (OUT / "cache" / "sample.json").write_text(
        json.dumps({"seed": SEED, "units": picked}, indent=1), "utf8"
    )
    dur = sum(dio.duration_s(p["unit_id"]) for p in picked)
    print(
        json.dumps(
            {
                "n": len(picked),
                "speakers": len(used),
                "audio_s": round(dur, 1),
                "candidates": {k: len(v) for k, v in cand.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
