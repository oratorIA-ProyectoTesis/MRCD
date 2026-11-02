"""``python -m mrcd.recording_check <command>``: technical verification of own recordings (no content stored)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m mrcd.recording_check", description=__doc__
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(
        "verify",
        help="verify recordings; writes one JSON report per file (numbers only)",
    )
    s.add_argument("files", nargs="+", type=Path)
    s.add_argument("--out-dir", type=Path, help="directory for <id>.report.json")
    s.add_argument(
        "--no-face",
        action="store_true",
        help="skip the S6 face gate (about 11 ms per frame)",
    )
    s.add_argument("--face-model")

    s = sub.add_parser(
        "report",
        help="aggregate per-recording reports (+ optional engine run summaries)",
    )
    s.add_argument(
        "--reports", type=Path, required=True, help="directory with *.report.json"
    )
    s.add_argument("--runs", type=Path, help="JSON list of engine run summaries")
    s.add_argument("--out", type=Path, required=True, help="output directory")

    s = sub.add_parser(
        "freeze",
        help="freeze the technical pre-registration and its code (before recording)",
    )
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("verify-freeze")
    s.add_argument("freeze", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "verify":
        from mrcd.recording_check.verify_recording import verify_file

        bad = 0
        for f in a.files:
            rep = verify_file(f, face_model=a.face_model, run_face=not a.no_face)
            print(f"{rep['file']}: {'OK' if rep['all_checks_passed'] else 'REVISAR'}")
            for k, v in rep["verdicts"].items():
                print(f"  {k:22s} {'ok' if v else ('?' if v is None else 'FALLA')}")
            if a.out_dir:
                a.out_dir.mkdir(parents=True, exist_ok=True)
                (a.out_dir / (f.stem + ".report.json")).write_text(
                    json.dumps(rep, indent=2, default=float), encoding="utf-8"
                )
            bad += not rep["all_checks_passed"]
        return 1 if bad else 0
    if a.cmd == "report":
        from mrcd.recording_check.technical import render_markdown, summarize

        reports = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(a.reports.glob("*.report.json"))
        ]
        runs = json.loads(a.runs.read_text(encoding="utf-8")) if a.runs else []
        res = summarize(reports, runs)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "technical_summary.json").write_text(
            json.dumps(res, indent=2, default=float), encoding="utf-8"
        )
        (a.out / "technical_summary.md").write_text(
            render_markdown(res), encoding="utf-8"
        )
        print(render_markdown(res))
        return 0
    if a.cmd == "freeze":
        from mrcd.recording_check.freeze import freeze

        rec = freeze(a.out, tag="preregistro-tecnico")
        print(
            json.dumps(
                {k: rec[k] for k in ("frozen_at_utc", "config_hash", "git_head")},
                indent=2,
            )
        )
        return 0
    if a.cmd == "verify-freeze":
        from mrcd.recording_check.freeze import verify_freeze

        bad = verify_freeze(a.freeze)
        print("INTACTO" if not bad else "\n".join(bad))
        return 1 if bad else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
