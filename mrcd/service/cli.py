"""Command line: ``python -m mrcd.service {analyze,serve,validate,schema}``.

Run with ``OMP_NUM_THREADS=4`` already in the environment (PowerShell: ``$env:OMP_NUM_THREADS=4``): the ASR
text depends on it and it must be set before numpy is imported, which a library cannot do for you.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mrcd.service.analyzer import DEFAULT_CHECKPOINT, AnalysisOptions, MrcdAnalyzer
from mrcd.service.contract import SCHEMA, validate_document


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--ckpt",
        default=str(DEFAULT_CHECKPOINT),
        help="engine checkpoint (default: models/ckpt_audio_text.pt)",
    )
    p.add_argument(
        "--whisper",
        default="small",
        help="Whisper model size; must already be in the local cache",
    )
    p.add_argument("--max-duration-s", type=float, default=600.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mrcd.service", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse one WAV file and print the v1 JSON")
    a.add_argument("audio")
    a.add_argument("--out", help="write the JSON here instead of stdout")
    a.add_argument(
        "--asr-profile",
        default="p0",
        help="p0 (default, validated default) or p1_disfluent_prompt (EXPERIMENTAL)",
    )
    a.add_argument(
        "--head",
        default=None,
        choices=["vc"],
        help="EXPERIMENTAL opt-in head; default none (V0)",
    )
    a.add_argument(
        "--include-text",
        action="store_true",
        help="include transcript / event text (off by default: privacy)",
    )
    _add_engine_args(a)

    s = sub.add_parser("serve", help="local development HTTP server (loopback only)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="allowed browser origin, e.g. http://localhost:5173",
    )
    s.add_argument("--allow-non-loopback", action="store_true")
    _add_engine_args(s)

    v = sub.add_parser("validate", help="validate a JSON file against the v1 contract")
    v.add_argument("json_file")

    sub.add_parser("schema", help="print the v1 JSON Schema")

    args = parser.parse_args(argv)
    if args.cmd == "schema":
        print(json.dumps(SCHEMA, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "validate":
        errors = validate_document(
            json.loads(Path(args.json_file).read_text(encoding="utf-8"))
        )
        for e in errors:
            print(e, file=sys.stderr)
        print("valid" if not errors else f"{len(errors)} violation(s)")
        return 0 if not errors else 1

    analyzer = MrcdAnalyzer(
        args.ckpt, whisper_size=args.whisper, max_duration_s=args.max_duration_s
    )
    if args.cmd == "serve":
        from mrcd.service.server import serve

        print(
            f"MRCD service (development) on http://{args.host}:{args.port}  - no auth, loopback only",
            file=sys.stderr,
        )
        serve(
            analyzer,
            args.host,
            args.port,
            tuple(args.cors_origin),
            args.allow_non_loopback,
        )
        return 0

    doc = analyzer.analyze_file(
        args.audio,
        AnalysisOptions(
            asr_profile=args.asr_profile, head=args.head, include_text=args.include_text
        ),
    )
    errors = validate_document(doc)
    if errors:  # a contract violation is a service bug: never emit it silently
        print(
            "INTERNAL: response violates the v1 contract:\n" + "\n".join(errors),
            file=sys.stderr,
        )
        return 3
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if doc["status"] == "completed" else 2
