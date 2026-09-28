"""Versioned exports: CSV for analysis, Markdown run report, ELAN EAF for external review.
The EAF tier is `human_disfluency`, readable by core.dataset.read_human_events."""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET

from core.evaluation import PAUSES

CSV_FIELDS = (
    "event_id",
    "recording_id",
    "speaker_id",
    "segment_id",
    "start_ms",
    "end_ms",
    "label",
    "decision",
    "text",
    "score",
    "score_kind",
    "source_kind",
)


def events_csv(events: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({**e, "text": e.get("text") or e.get("verbatim_text")} for e in events)
    return buf.getvalue()


def run_report(run: dict) -> str:
    """Human-readable run report; counts keep pauses apart from disfluencies."""
    d, res = run["data"], run["data"].get("result", {})
    events = [e for e in res.get("events", []) if e.get("decision", "event") == "event"]
    lines = [
        f"# Ejecución {run['id']} — {d['system']}",
        "",
        f"- Estado: **{run['status']}** · grabación `{run['parent']}` · configuración `{d.get('config', {})}`",
        f"- Versión: `{res.get('model_version', {})}`",
        f"- Advertencia: {res.get('warning') or 'ninguna'}",
        f"- Disfluencias: **{sum(e['label'] not in PAUSES for e in events)}** · pausas (no disfluencia): "
        f"**{sum(e['label'] in PAUSES for e in events)}** · inciertos: "
        f"**{sum(e.get('decision') == 'uncertain' for e in res.get('events', []))}**",
        f"- Tiempos por etapa: `{res.get('timings', {})}` · RTF: {res.get('rtf')}",
        f"- Segmentos: `{res.get('segments', {})}`",
        "",
        "Una detección no es un diagnóstico ni una medida de calidad comunicativa.",
        "",
        "| inicio (s) | fin (s) | clase | decisión | texto |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {e['start_ms'] / 1000:.2f} | {e['end_ms'] / 1000:.2f} | {e['label']} | {e.get('decision', 'event')} "
        f"| {(e.get('text') or '').replace('|', '/')} |"
        for e in res.get("events", [])
    ]
    return "\n".join(lines) + "\n"


def eaf(events: list[dict], media_url: str, author: str) -> bytes:
    root = ET.Element("ANNOTATION_DOCUMENT", AUTHOR=author, FORMAT="3.0", VERSION="3.0")
    header = ET.SubElement(root, "HEADER", MEDIA_FILE="", TIME_UNITS="milliseconds")
    ET.SubElement(header, "MEDIA_DESCRIPTOR", MEDIA_URL=media_url, MIME_TYPE="audio/x-wav")
    order = ET.SubElement(root, "TIME_ORDER")
    tier = ET.Element("TIER", TIER_ID="human_disfluency", LINGUISTIC_TYPE_REF="default-lt")
    for i, e in enumerate(e for e in events if e.get("decision", "event") == "event"):
        for side in ("start_ms", "end_ms"):
            ET.SubElement(order, "TIME_SLOT", TIME_SLOT_ID=f"ts{i}{side[0]}", TIME_VALUE=str(e[side]))
        ann = ET.SubElement(
            ET.SubElement(tier, "ANNOTATION"),
            "ALIGNABLE_ANNOTATION",
            ANNOTATION_ID=e.get("event_id") or f"a{i}",
            TIME_SLOT_REF1=f"ts{i}s",
            TIME_SLOT_REF2=f"ts{i}e",
        )
        ET.SubElement(ann, "ANNOTATION_VALUE").text = e["label"]
    root.append(tier)
    ET.SubElement(root, "LINGUISTIC_TYPE", LINGUISTIC_TYPE_ID="default-lt", TIME_ALIGNABLE="true")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
