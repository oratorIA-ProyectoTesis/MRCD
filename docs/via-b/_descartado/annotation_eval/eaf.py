"""Build, read and validate the Via B ELAN (EAF 3.0) files.

One file per recording and annotator (``A1``, ``A2``) plus one adjudicated file (``ADJ``). Annotators never receive
a file that contains another annotator's tiers (independence) or any system output.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from mrcd.annotation_eval.schema import (
    ALL_CLASSES,
    ANNOTATOR_RE,
    COVERAGE_VALUES,
    GUIDE_PROPERTY,
    PROPERTY_PREFIX_COVERAGE,
    RECORDING_RE,
    TIER_CLASSES,
    TIER_KEYS,
    TIER_RE,
    UNCERTAIN_CLASS,
    VOCAB,
    tier_id,
)

TOL_MS = (
    2  # tolerance when comparing an annotation end with the declared audio duration
)
WARN_MIN_MS = 80
WARN_MAX_MS = 8000
WARN_MAX_PAUSE_MS = 15000
CV_DESCRIPTIONS = {
    "filler_word": "Pausa llena o muletilla funcionando como relleno (ver guia, seccion 3)",
    "prolongation": "Alargamiento observable de un sonido o palabra (no se anota la funcion)",
    "repetition": "Repeticion literal de sonido, silaba, palabra o frase",
    "revision": "Reformulacion con reparandum y correccion (capa opcional)",
    "block": "Bloqueo audible o visible, definicion conservadora (capa opcional)",
    "pause_function": "Funcion de una pausa silenciosa (capa opcional): retorica o neutra",
    "excluir": "Region fuera de evaluacion (claqueta, otra voz, ruido, fuera de tarea)",
}


@dataclass(frozen=True)
class Ev:
    tier: str  # tier key (filler_word, ..., pause_function, excluir)
    annotator: str
    value: str
    start_ms: int
    end_ms: int

    @property
    def cls(self) -> str | None:
        return VOCAB[self.tier].get(self.value)

    @property
    def uncertain(self) -> bool:
        return self.value in UNCERTAIN_CLASS


@dataclass
class ParsedEAF:
    path: str
    properties: dict[str, str] = field(default_factory=dict)
    events: list[Ev] = field(default_factory=list)
    tier_ids: list[str] = field(default_factory=list)
    cv_by_tier: dict[str, set[str]] = field(default_factory=dict)
    structural_errors: list[str] = field(default_factory=list)

    @property
    def annotators(self) -> set[str]:
        return {m.group("ann") for t in self.tier_ids if (m := TIER_RE.match(t))}

    @property
    def duration_ms(self) -> int | None:
        v = self.properties.get("audio_duration_ms")
        return int(v) if v and v.isdigit() else None

    def coverage(self, cls: str) -> str | None:
        return self.properties.get(PROPERTY_PREFIX_COVERAGE + cls)


# --------------------------------------------------------------------------------------------- build
def _indent(root: ET.Element) -> None:
    ET.indent(root, space="    ")


def build_eaf(
    recording_id: str,
    annotators: Sequence[str],
    media_name: str,
    duration_ms: int,
    *,
    events: Iterable[Ev] = (),
    coverage: dict[str, str] | None = None,
    properties: dict[str, str] | None = None,
    guide_version: str = "0.1-borrador",
    mime: str = "audio/x-wav",
) -> ET.ElementTree:
    """Create an EAF. ``annotators`` lists which annotator tier sets to include (['A1'] for an annotator file)."""
    for a in annotators:
        if not ANNOTATOR_RE.match(a):
            raise ValueError(f"invalid annotator {a!r}")
    events = list(events)
    slots: dict[int, str] = {}
    for e in events:
        for t in (e.start_ms, e.end_ms):
            slots.setdefault(t, f"ts{len(slots) + 1}")
    root = ET.Element(
        "ANNOTATION_DOCUMENT",
        {
            "AUTHOR": "",
            "DATE": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "FORMAT": "3.0",
            "VERSION": "3.0",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan/EAFv3.0.xsd",
        },
    )
    header = ET.SubElement(
        root, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"}
    )
    ET.SubElement(
        header,
        "MEDIA_DESCRIPTOR",
        {
            "MEDIA_URL": f"file:///{media_name}",
            "MIME_TYPE": mime,
            "RELATIVE_MEDIA_URL": f"./{media_name}",
        },
    )
    props = {
        "recording_id": recording_id,
        "audio_duration_ms": str(int(duration_ms)),
        GUIDE_PROPERTY: guide_version,
        "annotator": ",".join(annotators),
    }
    if (m := RECORDING_RE.match(recording_id)) is not None:
        props["speaker_id"] = f"VB{m.group('phase')}_S{m.group('spk')}"
        props["phase"] = "practice" if m.group("phase") == "P" else "test"
    for c in ALL_CLASSES:
        props[PROPERTY_PREFIX_COVERAGE + c] = (coverage or {}).get(c, "not_annotated")
    props.update(properties or {})
    for k, v in props.items():
        ET.SubElement(header, "PROPERTY", {"NAME": k}).text = v
    ET.SubElement(header, "PROPERTY", {"NAME": "lastUsedAnnotationId"}).text = str(
        len(events)
    )

    order = ET.SubElement(root, "TIME_ORDER")
    for t, sid in sorted(slots.items()):
        ET.SubElement(order, "TIME_SLOT", {"TIME_SLOT_ID": sid, "TIME_VALUE": str(t)})

    ann_n = 0
    for a in annotators:
        for key in TIER_KEYS:
            tier = ET.SubElement(
                root,
                "TIER",
                {
                    "ANNOTATOR": a,
                    "LINGUISTIC_TYPE_REF": f"lt_{key}",
                    "TIER_ID": tier_id(a, key),
                },
            )
            for e in sorted(
                (x for x in events if x.annotator == a and x.tier == key),
                key=lambda x: x.start_ms,
            ):
                ann_n += 1
                node = ET.SubElement(
                    ET.SubElement(tier, "ANNOTATION"), "ALIGNABLE_ANNOTATION"
                )
                node.set("ANNOTATION_ID", f"a{ann_n}")
                node.set("TIME_SLOT_REF1", slots[e.start_ms])
                node.set("TIME_SLOT_REF2", slots[e.end_ms])
                ET.SubElement(node, "ANNOTATION_VALUE").text = e.value
    for key in TIER_KEYS:
        ET.SubElement(
            root,
            "LINGUISTIC_TYPE",
            {
                "GRAPHIC_REFERENCES": "false",
                "LINGUISTIC_TYPE_ID": f"lt_{key}",
                "TIME_ALIGNABLE": "true",
                "CONTROLLED_VOCABULARY_REF": f"cv_{key}",
            },
        )
    ET.SubElement(root, "LOCALE", {"LANGUAGE_CODE": "es", "COUNTRY_CODE": "PE"})
    ET.SubElement(
        root,
        "LANGUAGE",
        {
            "LANG_ID": "spa",
            "LANG_LABEL": "Spanish",
            "LANG_DEF": "http://cdb.iso.org/lg/CDB-00130975-001",
        },
    )
    for stereotype in (
        "Time_Subdivision",
        "Symbolic_Subdivision",
        "Symbolic_Association",
        "Included_In",
    ):
        ET.SubElement(
            root, "CONSTRAINT", {"DESCRIPTION": stereotype, "STEREOTYPE": stereotype}
        )
    n_cve = 0
    for key in TIER_KEYS:
        cv = ET.SubElement(root, "CONTROLLED_VOCABULARY", {"CV_ID": f"cv_{key}"})
        ET.SubElement(cv, "DESCRIPTION", {"LANG_REF": "spa"}).text = CV_DESCRIPTIONS[
            key
        ]
        for value in VOCAB[key]:
            n_cve += 1
            entry = ET.SubElement(cv, "CV_ENTRY_ML", {"CVE_ID": f"cveid{n_cve}"})
            ET.SubElement(
                entry, "CVE_VALUE", {"DESCRIPTION": "", "LANG_REF": "spa"}
            ).text = value
    _indent(root)
    return ET.ElementTree(root)


def write_eaf(tree: ET.ElementTree, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------------------------- parse
def parse_eaf(path: str | Path) -> ParsedEAF:
    out = ParsedEAF(path=str(path))
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        out.structural_errors.append(f"xml_parse_error: {e}")
        return out
    for p in root.iter("PROPERTY"):
        out.properties[p.get("NAME", "")] = (p.text or "").strip()
    slots = {s.get("TIME_SLOT_ID"): s.get("TIME_VALUE") for s in root.iter("TIME_SLOT")}
    cvs = {
        cv.get("CV_ID"): {(v.text or "").strip() for v in cv.iter("CVE_VALUE")}
        for cv in root.iter("CONTROLLED_VOCABULARY")
    }
    lt_cv = {
        lt.get("LINGUISTIC_TYPE_ID"): lt.get("CONTROLLED_VOCABULARY_REF")
        for lt in root.iter("LINGUISTIC_TYPE")
    }
    for t in root.iter("TIER"):
        tid = t.get("TIER_ID", "")
        out.tier_ids.append(tid)
        m = TIER_RE.match(tid)
        if not m or m.group("key") not in TIER_KEYS:
            out.structural_errors.append(f"unknown_tier: {tid!r}")
            continue
        key, ann = m.group("key"), m.group("ann")
        out.cv_by_tier[tid] = cvs.get(lt_cv.get(t.get("LINGUISTIC_TYPE_REF")), set())
        if list(t.iter("REF_ANNOTATION")):
            out.structural_errors.append(
                f"ref_annotation_in_tier: {tid!r} (time-aligned intervals required)"
            )
        for a in t.iter("ALIGNABLE_ANNOTATION"):
            v1, v2 = (
                slots.get(a.get("TIME_SLOT_REF1")),
                slots.get(a.get("TIME_SLOT_REF2")),
            )
            val = (a.findtext("ANNOTATION_VALUE") or "").strip()
            if (
                v1 is None
                or v2 is None
                or not (v1.lstrip("-").isdigit() and v2.lstrip("-").isdigit())
            ):
                out.structural_errors.append(
                    f"unaligned_annotation: {tid!r} {a.get('ANNOTATION_ID')!r}"
                )
                continue
            out.events.append(Ev(key, ann, val, int(v1), int(v2)))
    return out


# ----------------------------------------------------------------------------------------- validation
@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.detail}"


@dataclass
class ValidationReport:
    path: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, detail: str) -> None:
        self.issues.append(Issue(severity, code, detail))


def validate_eaf(
    path: str | Path,
    *,
    expected_annotator: str | None = None,
    audio_duration_ms: int | None = None,
    require_complete: Iterable[str] = (),
    parsed: ParsedEAF | None = None,
) -> ValidationReport:
    """Check tiers, vocabularies, overlaps, boundaries inside the audio and the declared coverage.

    ``require_complete``: MRCD classes whose coverage must be declared ``complete`` (e.g. the core classes of a
    confirmatory file). ``audio_duration_ms`` overrides the duration stored in the header; one of the two is
    required (an interval can only be checked against a known duration).
    """
    p = parsed or parse_eaf(path)
    rep = ValidationReport(path=str(path))
    for s in p.structural_errors:
        rep.add("error", s.split(":")[0], s)
    if p.structural_errors and not p.events and not p.tier_ids:
        return rep

    rec = p.properties.get("recording_id", "")
    if not RECORDING_RE.match(rec):
        rep.add(
            "error",
            "bad_recording_id",
            f"recording_id {rec!r} does not match VB<P|T>_S<nn>_<PRES|ESP><k>",
        )
    if not p.properties.get(GUIDE_PROPERTY):
        rep.add(
            "error", "missing_guide_version", "header property guide_version is empty"
        )

    anns = p.annotators
    if expected_annotator is not None:
        foreign = anns - {expected_annotator}
        if foreign:
            rep.add(
                "error",
                "foreign_annotator_tier",
                f"file of {expected_annotator} contains tiers of {sorted(foreign)}",
            )
        if expected_annotator not in anns:
            rep.add(
                "error", "missing_annotator_tiers", f"no tiers for {expected_annotator}"
            )
    for a in sorted(anns):
        for key in TIER_KEYS:
            tid = f"{a}_{key}"
            if tid not in p.tier_ids:
                rep.add("error", "missing_tier", tid)
            elif p.cv_by_tier.get(tid) != set(VOCAB[key]):
                rep.add(
                    "error",
                    "vocabulary_mismatch",
                    f"{tid}: expected {sorted(VOCAB[key])}, found {sorted(p.cv_by_tier.get(tid, set()))}",
                )

    dur = audio_duration_ms if audio_duration_ms is not None else p.duration_ms
    if dur is None:
        rep.add(
            "error",
            "unknown_audio_duration",
            "no audio_duration_ms in the header and none supplied",
        )
    elif (
        audio_duration_ms is not None
        and p.duration_ms is not None
        and abs(p.duration_ms - audio_duration_ms) > TOL_MS
    ):
        rep.add(
            "error",
            "duration_mismatch",
            f"header {p.duration_ms} ms vs audio {audio_duration_ms} ms",
        )

    by_tier: dict[tuple[str, str], list[Ev]] = {}
    for e in p.events:
        by_tier.setdefault((e.annotator, e.tier), []).append(e)
        if e.value not in VOCAB[e.tier]:
            rep.add(
                "error",
                "value_not_in_vocabulary",
                f"{e.annotator}_{e.tier}: {e.value!r} at {e.start_ms} ms",
            )
        if e.start_ms < 0 or e.end_ms <= e.start_ms:
            rep.add(
                "error",
                "bad_interval",
                f"{e.annotator}_{e.tier}: [{e.start_ms}, {e.end_ms}] ms",
            )
            continue
        if dur is not None and e.end_ms > dur + TOL_MS:
            rep.add(
                "error",
                "outside_audio",
                f"{e.annotator}_{e.tier}: end {e.end_ms} ms > duration {dur} ms",
            )
        length = e.end_ms - e.start_ms
        limit = (
            WARN_MAX_PAUSE_MS
            if e.tier in ("pause_function", "excluir")
            else WARN_MAX_MS
        )
        if e.tier != "excluir" and length < WARN_MIN_MS:
            rep.add(
                "warning",
                "very_short_event",
                f"{e.annotator}_{e.tier}: {length} ms at {e.start_ms} ms",
            )
        if length > limit:
            rep.add(
                "warning",
                "very_long_event",
                f"{e.annotator}_{e.tier}: {length} ms at {e.start_ms} ms",
            )
    for (a, key), evs in by_tier.items():
        evs = sorted(evs, key=lambda x: (x.start_ms, x.end_ms))
        for x, y in zip(evs, evs[1:]):
            if y.start_ms < x.end_ms:
                rep.add(
                    "error",
                    "overlap_in_tier",
                    f"{a}_{key}: [{x.start_ms},{x.end_ms}] and [{y.start_ms},{y.end_ms}]",
                )

    for a in sorted(anns):
        for key, classes in TIER_CLASSES.items():
            has = bool(by_tier.get((a, key)))
            for c in classes:
                cov = p.coverage(c)
                if cov not in COVERAGE_VALUES:
                    rep.add("error", "bad_coverage_value", f"coverage.{c}={cov!r}")
                elif has and cov != "complete":
                    rep.add(
                        "error",
                        "annotations_without_complete_coverage",
                        f"{a}_{key} has events but coverage.{c}={cov}",
                    )
    for c in require_complete:
        if p.coverage(c) != "complete":
            rep.add(
                "error",
                "coverage_not_complete",
                f"coverage.{c} must be 'complete' (is {p.coverage(c)!r})",
            )
    return rep


def declare_coverage(
    path: str | Path, complete: Iterable[str], *, not_annotated: Iterable[str] = ()
) -> None:
    """Rewrite the header coverage properties of a closed EAF (the file must not be open in ELAN)."""
    tree = ET.parse(path)
    header = tree.getroot().find("HEADER")
    if header is None:
        raise ValueError("EAF without HEADER")
    wanted = {c: "complete" for c in complete} | {
        c: "not_annotated" for c in not_annotated
    }
    bad = set(wanted) - set(ALL_CLASSES)
    if bad:
        raise ValueError(f"unknown classes {sorted(bad)}")
    seen = set()
    for prop in header.iter("PROPERTY"):
        name = prop.get("NAME", "")
        if (
            name.startswith(PROPERTY_PREFIX_COVERAGE)
            and name[len(PROPERTY_PREFIX_COVERAGE) :] in wanted
        ):
            c = name[len(PROPERTY_PREFIX_COVERAGE) :]
            prop.text = wanted[c]
            seen.add(c)
    for c in set(wanted) - seen:
        ET.SubElement(
            header, "PROPERTY", {"NAME": PROPERTY_PREFIX_COVERAGE + c}
        ).text = wanted[c]
    tree.write(path, encoding="utf-8", xml_declaration=True)
