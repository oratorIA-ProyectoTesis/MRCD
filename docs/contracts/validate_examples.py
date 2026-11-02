"""Validate the real fixtures in docs/contracts/examples/ against the CURRENT-state schemas, plus semantic checks.

Run (repo root):  .venv/Scripts/python.exe docs/contracts/validate_examples.py
Uses the in-repo subset validator mrcd.service.validate (jsonschema is not installed). Exit 1 on any failure.
The semantic checks assert facts that the contracts document (not model quality).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from mrcd.service.validate import validate  # noqa: E402

EX, SC = HERE / "examples", HERE / "schemas"
fails: list[str] = []


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def check(name, cond, detail=""):
    print(
        ("ok   " if cond else "FAIL ")
        + name
        + (f" {detail}" if detail and not cond else "")
    )
    if not cond:
        fails.append(name)


res_schema = load(SC / "engine_result.current.schema.json")
win_schema = load(SC / "engine_windows.current.schema.json")
row_schema = load(SC / "live_row.current.schema.json")

for name in ("batch_p0.json", "batch_p0_vc.json", "batch_p1.json"):
    d = load(EX / name)
    errs = validate(d, res_schema)
    check(f"{name} matches engine_result.current", not errs, errs[:3])
    ev = d["events"]
    check(
        f"{name} events sorted and non-overlapping",
        all(a["end"] <= b["start"] + 1e-9 for a, b in zip(ev, ev[1:])),
    )
    check(
        f"{name} every event has 0 <= start < end <= duration + stride",
        all(0 <= e["start"] < e["end"] <= d["duration"] + 0.5 for e in ev),
    )
    check(f"{name} no event labelled fluent", all(e["label"] != "fluent" for e in ev))
    check(
        f"{name} provenance warning present",
        isinstance(d["warning"], str) and "PRUEBA DE HUMO" in d["warning"],
    )
    bad = {"probability", "calibrated", "uncertainty", "uncertain"}
    check(
        f"{name} no probability/uncertainty keys at event level",
        all(not (bad & set(e)) for e in ev),
    )

p0, p1, vc = (
    load(EX / n) for n in ("batch_p0.json", "batch_p1.json", "batch_p0_vc.json")
)
check(
    "p0 events carry no asr_profile mark",
    all("asr_profile" not in e for e in p0["events"]),
)
check(
    "p1 events carry asr_profile + asr_experimental",
    all(
        e.get("asr_profile") == "p1_disfluent_prompt"
        and e.get("asr_experimental") is True
        for e in p1["events"]
    ),
)
check("p1 warning appended with EXPERIMENTAL note", "EXPERIMENTAL" in p1["warning"])
check(
    "vc head block present and not calibrated",
    vc["head"]["scores_calibrated"] is False
    and all(c["calibrated"] is False for c in vc["head"]["classes"].values()),
)
check("p0 has no head block", "head" not in p0)


def key(e):
    return (e["label"], e["start"], e["end"], e["confidence"])


check(
    "same events with and without head (V0 untouched)",
    [key(e) for e in p0["events"]] == [key(e) for e in vc["events"]],
)

wins = load(EX / "batch_p0_windows.json")
errs = validate(wins, win_schema)
check("batch_p0_windows matches engine_windows.current", not errs, errs[:3])
check(
    "window probabilities sum to 1 (softmax; NOT a calibration claim)",
    all(abs(sum(w["probabilities"].values()) - 1) < 1e-4 for w in wins),
)
check(
    "window confidence == max probability == probability of its label",
    all(
        abs(w["confidence"] - max(w["probabilities"].values())) < 1e-9
        and w["probabilities"][w["label"]] == w["confidence"]
        for w in wins
    ),
)
check(
    "grid: window starts every 0.5 s",
    all(abs((b["start"] - a["start"]) - 0.5) < 1e-6 for a, b in zip(wins, wins[1:])),
)
for e in vc["events"]:
    run = [
        w
        for w in wins
        if w["label"] == e["label"]
        and e["start"] - 1e-9 <= w["start"] < e["end"] - 1e-9
    ]
    check(
        f"event {e['label']}@{e['start']}: confidence == max window confidence of its run",
        bool(run) and abs(max(w["confidence"] for w in run) - e["confidence"]) < 1e-9,
    )
    check(
        f"event {e['label']}@{e['start']}: end == last window start + 0.5",
        bool(run)
        and abs(e["end"] - (run[-1]["start"] + 0.5)) < 1e-9
        and run[0]["start"] == e["start"],
    )

for cfg in ("v1", "v2", "v3"):
    rows = load(EX / f"live_rows_{cfg}.json")["rows"]
    errs = [er for r in rows for er in validate(r, row_schema)]
    check(f"live_rows_{cfg} rows match live_row.current", not errs, errs[:3])
    check(
        f"live_rows_{cfg} confianza == max of todas_las_probabilidades",
        all(
            abs(r["confianza"] - max(r["todas_las_probabilidades"].values())) < 1e-12
            for r in rows
        ),
    )
    if cfg == "v1":
        extra = {"status", "revises", "terminal", "emitted_at_s", "f0_center_st"}
        check(
            "v1 rows have no status/revises/terminal/emitted_at_s/f0_center_st",
            all(not (extra & set(r)) for r in rows),
        )
    else:
        prov = {r["timestamp"] for r in rows if r["status"] == "provisional"}
        rev = [r for r in rows if "revises" in r]
        check(
            f"{cfg} every revises equals its own timestamp and points to a provisional row",
            all(r["revises"] in prov and r["revises"] == r["timestamp"] for r in rev),
        )
        last = {}
        for r in rows:
            last[r["timestamp"]] = r
        check(
            f"{cfg} last row per timestamp covers the whole 0.5 s grid",
            sorted(last) == [i * 0.5 for i in range(len(last))],
        )
        check(
            f"{cfg} the last row of every revised timestamp is final",
            all(last[t]["status"] == "final" for t in prov),
        )

a, b = load(EX / "live_rows_v2.json")["rows"], load(EX / "live_rows_v3.json")["rows"]
check(
    "v2 and v3 rows identical on this 20 s excerpt (says nothing about long sessions)",
    a == b,
)

from mrcd.service.contract import validate_document  # noqa: E402

app_schema = load(SC / "app_integration.v1.schema.json")
for name in (
    "app_report_section_example.json",
    "app_report_section_failed_example.json",
    "app_session_binding_example.json",
):
    errs = validate(load(EX / name), app_schema)
    check(f"{name} matches app_integration.v1", not errs, errs[:3])
for name in (
    "service_analysis_v1.json",
    "service_error_audio_too_short.json",
    "service_analysis_v1_2_default_scope.json",
    "service_analysis_v1_2_p1_route_decision_all.json",
    "service_analysis_v1_4_default_profile.json",
    "service_analysis_v1_4_v0.json",
):
    errs = validate_document(load(EX / name))
    check(f"{name} matches mrcd.analysis/v1 (service contract)", not errs, errs[:3])
sec = load(EX / "app_report_section_example.json")
check(
    "report section never turns an unavailable measure into 0",
    all(
        not (m["available"] is False and m["value"] == 0)
        for m in [sec["audio"]["duration_s"]]
        + ([sec["summary"]["events_per_minute"]] if sec["summary"] else [])
    ),
)
check(
    "report section keeps video unavailable with a reason",
    sec["modalities"]["kinesic_video"]["available"] is False
    and bool(sec["modalities"]["kinesic_video"]["reason"]),
)
check(
    "every report event score is labelled uncalibrated",
    all(
        e["score"]["calibrated"] is False and e["score"]["label"] == "uncalibrated"
        for e in sec["events"]
    ),
)
svc = load(EX / "service_analysis_v1.json")
check(
    "service events match engine events (same labels, starts, ends)",
    [(e["class"], e["start_s"], e["end_s"]) for e in svc["events"]]
    == [(e["label"], e["start"], e["end"]) for e in p0["events"]],
)

print("\n" + (f"FAILED: {len(fails)}" if fails else "all fixture checks passed"))
sys.exit(1 if fails else 0)
