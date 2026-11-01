"""System Usability Scale (SUS) scoring for the OE4 usability kit.

STATUS: NOT RUN. This script computes scores from a CSV of questionnaire answers. No SUS data exists yet;
running it on the empty template reports "not available" and never a score of 0.

Scoring (Brooke, 1996; the formula is the standard one and does not depend on the Spanish wording):
  * Items are answered on a 1-5 scale (1 = strongly disagree, 5 = strongly agree).
  * Odd items (1, 3, 5, 7, 9) are positively worded: contribution = answer - 1.
  * Even items (2, 4, 6, 8, 10) are negatively worded: contribution = 5 - answer.
  * SUS = 2.5 * (sum of the ten contributions), a number between 0 and 100.
  * A SUS value is NOT a percentage and is not a calibrated probability.

Usage:
  python sus_score.py responses.csv [--threshold 70] [--min-users 10] [--seed 0] [--json]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

N_ITEMS = 10
ITEM_COLUMNS = [f"item_{i}" for i in range(1, N_ITEMS + 1)]


def sus_score(answers: list[int]) -> float:
    """Return the SUS score (0-100) for ten answers on a 1-5 scale.

    Raises ValueError if the number of answers is not ten or any answer is not an integer in 1..5.
    """
    if len(answers) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} answers, got {len(answers)}")
    total = 0
    for idx, a in enumerate(answers, start=1):
        if isinstance(a, bool) or not isinstance(a, int) or not 1 <= a <= 5:
            raise ValueError(
                f"item {idx}: answer must be an integer from 1 to 5, got {a!r}"
            )
        total += (a - 1) if idx % 2 == 1 else (5 - a)
    return 2.5 * total


def _parse_row(row: dict[str, str]) -> tuple[list[int] | None, str]:
    """Return (answers, reason). answers is None when the row is not scorable; reason says why."""
    values = []
    for col in ITEM_COLUMNS:
        raw = (row.get(col) or "").strip()
        if raw == "":
            return None, f"missing {col}"
        try:
            values.append(int(raw))
        except ValueError:
            return None, f"non-integer {col}: {raw!r}"
    try:
        sus_score(values)
    except ValueError as exc:
        return None, str(exc)
    return values, ""


def bootstrap_ci(
    scores: list[float], seed: int = 0, n_boot: int = 5000
) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI of the mean. None when fewer than 2 scores."""
    if len(scores) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(scores, k=len(scores))) for _ in range(n_boot)
    )
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def summarize(
    path: Path, threshold: float = 70.0, min_users: int = 10, seed: int = 0
) -> dict:
    """Score every row of the CSV. Rows that cannot be scored are listed with a reason and never counted as 0."""
    scored: dict[str, float] = {}
    excluded: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [
            c
            for c in ["participant_id", *ITEM_COLUMNS]
            if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        for n, row in enumerate(reader, start=2):
            pid = (row.get("participant_id") or "").strip() or f"row{n}"
            if pid in scored or pid in excluded:
                excluded[f"{pid}#dup@row{n}"] = "duplicate participant_id"
                continue
            answers, reason = _parse_row(row)
            if answers is None:
                if all((row.get(c) or "").strip() == "" for c in ITEM_COLUMNS):
                    continue  # empty template row: nothing was answered, nothing to report
                excluded[pid] = reason
            else:
                scored[pid] = sus_score(answers)
    values = list(scored.values())
    n = len(values)
    out: dict = {
        "status": "NOT_RUN" if n == 0 else "COMPUTED",
        "n_scored": n,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "per_participant": scored,
        "threshold": threshold,
        "min_users": min_users,
    }
    if n == 0:
        out.update(
            mean=None,
            median=None,
            sd=None,
            ci95=None,
            meets_threshold=None,
            note="not available: no complete responses (this is not a score of 0)",
        )
        return out
    out["mean"] = statistics.fmean(values)
    out["median"] = statistics.median(values)
    out["sd"] = statistics.stdev(values) if n > 1 else None
    out["ci95"] = bootstrap_ci(values, seed)
    if n < min_users:
        out["meets_threshold"] = None
        out["note"] = (
            f"n={n} < {min_users}: the OE4 usability indicator cannot be evaluated; the numbers are descriptive only"
        )
    else:
        out["meets_threshold"] = out["mean"] >= threshold
        out["note"] = (
            "descriptive mean compared with the declared threshold; the CI is a bootstrap over participants"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv", type=Path)
    p.add_argument("--threshold", type=float, default=70.0)
    p.add_argument("--min-users", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = p.parse_args(argv)
    res = summarize(args.csv, args.threshold, args.min_users, args.seed)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(
            f"status: {res['status']}  scored: {res['n_scored']}  excluded: {res['n_excluded']}"
        )
        if res["n_scored"]:
            ci = res["ci95"]
            print(
                f"mean {res['mean']:.1f}  median {res['median']:.1f}  sd "
                f"{'n/a' if res['sd'] is None else format(res['sd'], '.1f')}  "
                f"CI95 {'n/a' if ci is None else f'[{ci[0]:.1f}, {ci[1]:.1f}]'}"
            )
        print(res["note"])
        for pid, reason in res["excluded"].items():
            print(f"  excluded {pid}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
