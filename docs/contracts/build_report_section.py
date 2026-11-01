"""Reference mapping: mrcd.analysis/v1 document -> oratoria.mrcd-report/v1 section (the report-side record).

This is the executable form of section 8 of APP_INTEGRATION_CONTRACT.md. The app-side implementer may port it; the
rules are normative, the code is a reference.

Run (repo root):
  .venv/Scripts/python.exe docs/contracts/build_report_section.py \
      docs/contracts/examples/service_analysis_v1.json docs/contracts/examples/app_report_section_example.json

The session id and the first-chunk epoch time are PLACEHOLDERS (no app session produced this example); everything
else comes from the real service output. Exit 1 if the produced record does not validate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from mrcd.service.validate import validate  # noqa: E402

SCHEMA = json.loads(
    (HERE / "schemas" / "app_integration.v1.schema.json").read_text(encoding="utf-8")
)


def build_report_section(
    analysis: dict,
    *,
    session_id: str,
    segment_index: int | None,
    audio_start_session_ms: int,
    job_id: str | None,
) -> dict:
    """Mechanical, loss-aware mapping. Never invents a value: a missing measure stays unavailable with its reason."""
    if analysis.get("schema_version") != "mrcd.analysis/v1":
        raise ValueError("only mrcd.analysis/v1 is understood")
    failed = analysis["status"] != "completed"
    prov = analysis.get("provenance") or {}
    asr = prov.get("asr") or {}
    req = analysis["request"]
    duration = req["audio_duration_s"]
    events = []
    for e in analysis["events"]:
        events.append(
            {
                "id": e["id"],
                "class": e["class"],
                "start_session_ms": audio_start_session_ms + round(e["start_s"] * 1000),
                "end_session_ms": audio_start_session_ms + round(e["end_s"] * 1000),
                "interval_semantics": e["interval_semantics"],
                "score": dict(
                    e["score"]
                ),  # copied verbatim: calibrated false, label "uncalibrated"
                "experimental": bool(e["experimental"]),
            }
        )
    summary = analysis["summary"]
    return {
        "schema_version": "oratoria.mrcd-report/v1",
        "session_id": session_id,
        "segment_index": segment_index,
        "status": "failed" if failed else "completed",
        "unavailable_reason": (analysis["error"] or {}).get("message")
        if failed
        else None,
        "source": {
            "contract": "mrcd.analysis/v1",
            "job_id": job_id,
            "mrcd_version": (prov.get("service") or {}).get("mrcd_version"),
            "asr_profile": asr.get("profile") or req.get("asr_profile"),
            "experimental": bool(analysis["experimental"]["active"]),
            "audio_sha256": req.get("audio_sha256"),
        },
        "audio": {"start_session_ms": audio_start_session_ms, "duration_s": duration},
        "events": events,
        "summary": None
        if summary is None
        else {
            "n_events": summary["n_events"],
            "events_per_minute": summary["events_per_minute"],
        },
        "modalities": analysis["modalities"],
        "warnings": analysis["warnings"],
    }


def main(argv: list[str]) -> int:
    src, dst = Path(argv[1]), Path(argv[2])
    analysis = json.loads(src.read_text(encoding="utf-8"))
    section = build_report_section(
        analysis,
        session_id="EXAMPLE-PLACEHOLDER-NOT-A-REAL-SESSION",
        segment_index=0,
        audio_start_session_ms=0,
        job_id=None,
    )
    errs = validate(section, SCHEMA)
    if errs:
        print("INVALID:", errs[:5])
        return 1
    dst.write_text(
        json.dumps(section, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
