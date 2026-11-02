"""Import adjudicated Via B EAF files into the ``mrcd.corpora`` Example schema (provenance human).

Only human files are read here. Coverage is declared per class and per recording in the adjudicated header; a class
that is not declared ``complete`` stays ``unknown`` (never a negative). Uncertain events and excluded regions are kept
as ignore regions, not as positives or negatives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from mrcd.annotation_eval.config import CONFIG_VERSION
from mrcd.annotation_eval.eaf import parse_eaf, validate_eaf
from mrcd.annotation_eval.schema import (
    ALL_CLASSES,
    CORE_CLASSES,
    GUIDE_PROPERTY,
    UNCERTAIN_CLASS,
    parse_recording_id,
)
from mrcd.constants import TAXONOMY
from mrcd.corpora.base import (
    CANONICAL_KINDS,
    Annotation,
    CorpusFormatError,
    Example,
    SourceFile,
    sha256_file,
    wav_info,
)

CORPUS = "via_b"
KIND = CANONICAL_KINDS["viabelantag"]
#: ELAN value -> canonical source label accepted by the corpora allow-list.
_LABEL = {
    "filler_word": "filler_word",
    "prolongation": "prolongation",
    "repetition": "repetition",
    "block": "block",
    "revision": "revision",
    "retorica": "retorica",
    "neutra": "neutra",
}


class ImportRefused(ValueError):
    """The adjudicated file or its audio does not satisfy the import contract."""


def example_from_eaf(
    eaf_path: Path, wav_path: Path, root: Path, *, annotator: str = "ADJ"
) -> Example:
    parsed = parse_eaf(eaf_path)
    rec = parsed.properties.get("recording_id", "")
    info = parse_recording_id(rec)
    dur_s = wav_info(wav_path, rate=16000, channels=1)
    rep = validate_eaf(
        eaf_path,
        expected_annotator=None,
        audio_duration_ms=int(round(dur_s * 1000)),
        parsed=parsed,
    )
    if annotator not in parsed.annotators:
        raise ImportRefused(f"{rec}: no {annotator} tiers in {eaf_path}")
    if not rep.ok:
        raise ImportRefused(
            f"{rec}: invalid EAF: " + "; ".join(str(i) for i in rep.errors[:5])
        )
    events = [e for e in parsed.events if e.annotator == annotator]
    anns: list[Annotation] = []
    ignore: list[dict] = []
    guide = parsed.properties.get(GUIDE_PROPERTY, "")
    for e in sorted(events, key=lambda x: (x.start_ms, x.end_ms)):
        s, t = e.start_ms / 1000, e.end_ms / 1000
        if e.tier == "excluir":
            ignore.append(
                {"start_s": s, "end_s": t, "classes": "all", "reason": e.value}
            )
        elif e.uncertain:
            ignore.append(
                {
                    "start_s": s,
                    "end_s": t,
                    "classes": list(UNCERTAIN_CLASS[e.value]),
                    "reason": e.value,
                }
            )
            anns.append(
                Annotation(
                    KIND,
                    e.value,
                    "human",
                    s,
                    t,
                    locator={"tier": f"{annotator}_{e.tier}"},
                )
            )
        else:
            anns.append(
                Annotation(
                    KIND,
                    _LABEL[e.value],
                    "human",
                    s,
                    t,
                    locator={"tier": f"{annotator}_{e.tier}"},
                    proposed_class=e.cls,
                    mapping_strength="exact",
                    justification=f"Adjudicated human annotation, guide {guide}",
                )
            )
    mapped = {a.proposed_class for a in anns if a.proposed_class}
    coverage, basis = {}, {}
    for c in TAXONOMY:
        if parsed.coverage(c) == "complete":
            coverage[c] = "pos" if c in mapped else "neg"
            if c not in mapped:
                basis[c] = (
                    f"complete coverage declared for this recording (double independent annotation + adjudication, "
                    f"guide {guide}); uncertain events and excluded regions are ignore regions"
                )
        else:
            coverage[c] = "unknown"
    rel = lambda p: p.resolve().relative_to(root.resolve()).as_posix()  # noqa: E731
    return Example(
        corpus=CORPUS,
        recording_id=rec,
        speaker_id=info["speaker"],
        group_id=info["speaker"],
        source_files=(
            SourceFile(rel(wav_path), sha256_file(wav_path)),
            SourceFile(rel(eaf_path), sha256_file(eaf_path)),
        ),
        original_text="",
        original_annotations=tuple(anns),
        temporal_unit="seconds",
        start_s=0.0,
        end_s=dur_s,
        source_label=None,
        proposed_mrcd_mapping=None,
        mapping_justification="",
        coverage=coverage,
        negative_basis=basis,
        provenance="human",
        input_view="",
        audio_duration_s=dur_s,
        extra={
            "phase": "practice" if info["phase"] == "P" else "test",
            "task": info["task"],
            "guide_version": guide,
            "ignore_regions": ignore,
        },
    )


def import_corpus(
    adj_dir: Path,
    media_dir: Path,
    out_dir: Path,
    *,
    root: Path | None = None,
    agreement: dict | None = None,
    include_practice: bool = False,
) -> dict:
    """Write ``examples.jsonl`` and ``manifest.json`` into ``out_dir``. Practice recordings are skipped by default
    (they calibrate the guide and never enter the test)."""
    root = root or adj_dir.parent
    examples, skipped = [], []
    for eaf in sorted(adj_dir.glob("*.eaf")):
        wav = media_dir / (eaf.stem + ".wav")
        if not wav.is_file():
            raise ImportRefused(f"missing audio for {eaf.name}: {wav}")
        ex = example_from_eaf(eaf, wav, root)
        if ex.extra["phase"] == "practice" and not include_practice:
            skipped.append(ex.recording_id)
            continue
        examples.append(ex)
    if not examples:
        raise ImportRefused("no test recordings to import")
    ids = [e.recording_id for e in examples]
    if len(set(ids)) != len(ids):
        raise ImportRefused("duplicate recording ids")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(e.to_dict(), sort_keys=True, ensure_ascii=False) for e in examples
    ]
    jl = out_dir / "examples.jsonl"
    jl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    per_class = {}
    for c in ALL_CLASSES:
        pos = sum(
            1 for e in examples for a in e.original_annotations if a.proposed_class == c
        )
        recs_complete = sum(1 for e in examples if e.coverage[c] != "unknown")
        gate = None
        if agreement is not None:
            gate = agreement.get("classes", {}).get(c, {}).get("passes_gate")
        per_class[c] = {
            "positives": pos,
            "speakers_with_positives": len(
                {
                    e.group_id
                    for e in examples
                    for a in e.original_annotations
                    if a.proposed_class == c
                }
            ),
            "recordings_with_complete_coverage": recs_complete,
            "recordings": len(examples),
            "agreement_gate_passed": gate,
            "core": c in CORE_CLASSES,
        }
    manifest = {
        "corpus": CORPUS,
        "importer_config": CONFIG_VERSION,
        "n_recordings": len(examples),
        "n_speakers": len({e.group_id for e in examples}),
        "duration_s": round(sum(e.audio_duration_s for e in examples), 3),
        "skipped_practice": skipped,
        "examples_sha256": sha256_file(jl),
        "classes": per_class,
        "human_annotation": {
            e.recording_id: e.manifest_human_annotation() for e in examples
        },
        "note": "Coverage 'unknown' is never a negative. human_annotation.coverage is 'complete' only if all seven classes were annotated.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_reference(corpus_dir: Path) -> dict:
    """Reference structure used by the evaluation: recording -> events per class, ignore regions, coverage."""
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    jl = corpus_dir / "examples.jsonl"
    from mrcd.corpora.base import sha256_file as _sha

    if _sha(jl) != manifest["examples_sha256"]:
        raise CorpusFormatError(
            "examples.jsonl does not match the manifest hash (file modified after import)"
        )
    recs = {}
    for line in jl.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        ev: dict[str, list[tuple[float, float]]] = {c: [] for c in ALL_CLASSES}
        for a in d["original_annotations"]:
            if a["proposed_class"]:
                ev[a["proposed_class"]].append((a["start_s"], a["end_s"]))
        recs[d["recording_id"]] = {
            "speaker": d["group_id"],
            "duration_s": d["audio_duration_s"],
            "events": {c: sorted(v) for c, v in ev.items()},
            "coverage": d["coverage"],
            "ignore": d["extra"]["ignore_regions"],
        }
    return {"manifest": manifest, "recordings": recs}


def classes_in_corpus(manifest: dict) -> Iterable[str]:
    return [
        c
        for c, v in manifest["classes"].items()
        if v["recordings_with_complete_coverage"] > 0
    ]
