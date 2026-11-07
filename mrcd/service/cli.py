"""Command line: ``mrcd {serve,analyze,download,sample,validate,schema}`` (same as ``python -m mrcd.service``).

The ``mrcd`` console command sets ``OMP_NUM_THREADS=4`` when it is missing, before numpy is imported (see
``mrcd_launcher``). With ``python -m mrcd.service`` put it in the environment yourself (PowerShell:
``$env:OMP_NUM_THREADS=4``): the ASR text depends on it and a library cannot set it in time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer
from mrcd.service.decision_profile import DECISION_PROFILES, DEFAULT_DECISION_PROFILE
from mrcd.service.contract import SCHEMA, validate_document


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--ckpt",
        default=None,
        help="engine checkpoint (default: $MRCD_CHECKPOINT, else models/ckpt_audio_text.pt in a clone, sha256-verified; "
        "the checkpoint is not distributed, see models/README.md)",
    )
    p.add_argument(
        "--whisper",
        default="small",
        help="Whisper model size; downloaded once on first use (~486 MB for small) unless HF_HUB_OFFLINE=1",
    )
    p.add_argument("--max-duration-s", type=float, default=600.0)
    p.add_argument(
        "--device",
        default=None,
        choices=["cpu"],
        help="pin the ASR device (default: auto, which falls back to CPU int8); required by --incremental",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrcd", description=__doc__)
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
        "--rate-limit",
        type=float,
        default=None,
        metavar="REQ_PER_S",
        help="per-client sustained requests/s (token bucket, 429 + Retry-After). Default: 10 when binding "
        "non-loopback, off on loopback. The bearer token is read from MRCD_SERVICE_TOKEN (required off loopback)",
    )
    s.add_argument(
        "--trusted-proxy",
        action="append",
        default=[],
        metavar="ADDR",
        help="address of the reverse proxy (e.g. 127.0.0.1) whose X-Forwarded-For is used as the rate-limit client key",
    )
    s.add_argument("--rate-burst", type=int, default=30, help="per-client burst size for --rate-limit")
    s.add_argument(
        "--no-docs",
        action="store_true",
        help="disable the product page (/) and Swagger UI (/docs); both are off with --allow-non-loopback unless --docs",
    )
    s.add_argument(
        "--docs",
        action="store_true",
        help="keep the product page and /docs on even with --allow-non-loopback (for a container whose port is "
        "published on the host loopback only, e.g. -p 127.0.0.1:8765:8765)",
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
    s.add_argument(
        "--warmup",
        action="store_true",
        help="run one synthetic analysis per preloaded profile before serving",
    )
    s.add_argument("--max-sessions", type=int, default=4)
    s.add_argument("--session-idle-timeout-s", type=float, default=120.0)
    s.add_argument("--max-session-seconds", type=float, default=600.0)
    _add_engine_args(s)

    v = sub.add_parser("validate", help="validate a JSON file against the v1 contract")
    v.add_argument("json_file")

    sub.add_parser("schema", help="print the v1 JSON Schema")

    d = sub.add_parser(
        "download",
        help="fetch the Whisper model into the local cache now (first-run download)",
    )
    d.add_argument("--whisper", default="small")

    m = sub.add_parser(
        "sample",
        help="write the bundled synthetic Spanish sample WAV (16 kHz mono PCM16)",
    )
    m.add_argument("out", nargs="?", default="sample_es.wav")

    args = parser.parse_args(argv)
    if args.cmd == "sample":
        from mrcd.service.web import SAMPLE_WAV

        Path(args.out).write_bytes(SAMPLE_WAV.read_bytes())
        print(args.out)
        return 0
    if args.cmd == "download":
        return 0 if _ensure_whisper(args.whisper) else 2
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

    preload = []
    if args.cmd == "serve":
        for item in [x.strip() for x in args.preload.split(",") if x.strip()]:
            prof, _, head = item.partition(":")
            preload.append(AnalysisOptions(asr_profile=prof, head=head or None))
        if args.incremental and not (args.device == "cpu" and preload and args.warmup):
            parser.error(
                "--incremental requires --device cpu, --preload <profiles> and --warmup (deterministic process state)"
            )

    from mrcd.weights import CheckpointError, resolve_checkpoint

    try:
        checkpoint = resolve_checkpoint(args.ckpt)
    except CheckpointError as exc:
        print(f"[mrcd] {exc}", file=sys.stderr)
        return 2
    if not _ensure_whisper(args.whisper):
        return 2
    analyzer = MrcdAnalyzer(
        checkpoint,
        whisper_size=args.whisper,
        device=args.device,
        max_duration_s=args.max_duration_s,
    )
    if args.cmd == "serve":
        from mrcd.service.incremental.session import IncrementalConfig
        from mrcd.service.security import RateLimitConfig
        from mrcd.service.server import serve

        incremental = (
            IncrementalConfig(
                max_sessions=args.max_sessions,
                idle_timeout_s=args.session_idle_timeout_s,
                max_session_seconds=args.max_session_seconds,
            )
            if args.incremental
            else None
        )

        enable_docs = not args.no_docs and (args.docs or not args.allow_non_loopback)
        shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        print(
            f"MRCD service (development) on http://{args.host}:{args.port}  "
            + (", bearer token required (MRCD_SERVICE_TOKEN)" if args.allow_non_loopback else ", loopback only, no token unless MRCD_SERVICE_TOKEN is set")
            + (
                f"\n  page: http://{shown}:{args.port}/   API docs: http://{shown}:{args.port}/docs"
                if enable_docs
                else ""
            )
            + (
                "\n  preloading and warming up the engine; requests are served once this finishes"
                if preload
                else ""
            ),
            file=sys.stderr,
            flush=True,
        )
        serve(
            analyzer,
            args.host,
            args.port,
            tuple(args.cors_origin),
            args.allow_non_loopback,
            enable_docs=enable_docs,
            incremental=incremental,
            preload=tuple(preload),
            warmup=args.warmup,
            device=args.device,
            trusted_proxies=tuple(args.trusted_proxy),
            rate_limit=(
                RateLimitConfig(rate_per_s=args.rate_limit, burst=args.rate_burst)
                if args.rate_limit
                else (RateLimitConfig(burst=args.rate_burst) if args.rate_burst != 30 else None)
            ),
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


def _ensure_whisper(size: str) -> bool:
    from mrcd.service.whisper_cache import WhisperModelMissing, ensure_whisper_model

    try:
        ensure_whisper_model(size)
    except WhisperModelMissing as exc:
        print(f"[mrcd] {exc}", file=sys.stderr)
        return False
    return True
