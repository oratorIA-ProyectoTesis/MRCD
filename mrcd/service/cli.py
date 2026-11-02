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
from mrcd.service.decision_profile import DECISION_PROFILES, DEFAULT_DECISION_PROFILE
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
    p.add_argument(
        "--device",
        default=None,
        choices=["cpu"],
        help="pin the ASR device (default: auto, which falls back to CPU int8); required by --incremental",
    )


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
    a.add_argument(
        "--include-out-of-scope",
        action="store_true",
        help="also emit block / rhetorical_pause / neutral_pause / revision, marked out_of_scope_unvalidated "
        "and never counted in totals (default: hidden)",
    )
    a.add_argument(
        "--decision",
        action="store_true",
        help="EXPERIMENTAL: add a three-state decision (present / absent / uncertain) per event",
    )
    a.add_argument(
        "--text-route",
        default=None,
        choices=["rep_ctx_p1"],
        help="EXPERIMENTAL: text route for repetition (best with --asr-profile p1_disfluent_prompt)",
    )
    a.add_argument(
        "--decision-profile",
        default=DEFAULT_DECISION_PROFILE,
        choices=list(DECISION_PROFILES),
        help="o2026_11 (service default: O-campaign pipelines for filler_word / prolongation) or v0 (historical engine decision)",
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
    s.add_argument(
        "--no-docs",
        action="store_true",
        help="disable Swagger UI (/docs); docs are always off with --allow-non-loopback",
    )
    s.add_argument(
        "--incremental",
        action="store_true",
        help="enable /v1/sessions (B incremental). Requires --device cpu, --preload and --warmup",
    )
    s.add_argument(
        "--preload",
        default="",
        help="comma-separated profiles (or profile:head) to load at start, e.g. p0 or p0,p1_disfluent_prompt",
    )
    s.add_argument("--warmup", action="store_true", help="run one synthetic analysis per preloaded profile before serving")
    s.add_argument("--max-sessions", type=int, default=4)
    s.add_argument("--session-idle-timeout-s", type=float, default=120.0)
    s.add_argument("--max-session-seconds", type=float, default=600.0)
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
        args.ckpt,
        whisper_size=args.whisper,
        device=args.device,
        max_duration_s=args.max_duration_s,
    )
    if args.cmd == "serve":
        from mrcd.service.incremental.session import IncrementalConfig
        from mrcd.service.server import serve

        preload = []
        for item in [x.strip() for x in args.preload.split(",") if x.strip()]:
            prof, _, head = item.partition(":")
            preload.append(AnalysisOptions(asr_profile=prof, head=head or None))
        if args.incremental and not (args.device == "cpu" and preload and args.warmup):
            parser.error("--incremental requires --device cpu, --preload <profiles> and --warmup (deterministic process state)")
        incremental = (
            IncrementalConfig(
                max_sessions=args.max_sessions,
                idle_timeout_s=args.session_idle_timeout_s,
                max_session_seconds=args.max_session_seconds,
            )
            if args.incremental
            else None
        )

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
            enable_docs=not (args.no_docs or args.allow_non_loopback),
            incremental=incremental,
            preload=tuple(preload),
            warmup=args.warmup,
        )
        return 0

    doc = analyzer.analyze_file(
        args.audio,
        AnalysisOptions(
            asr_profile=args.asr_profile,
            head=args.head,
            include_text=args.include_text,
            output_scope="all" if args.include_out_of_scope else "evaluated",
            decision=args.decision,
            text_route=args.text_route,
            decision_profile=args.decision_profile,
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
