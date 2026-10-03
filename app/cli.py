"""Administration commands (data manager role).

python -m app.cli add-user ana annotator
python -m app.cli import-pilot C:/zArnes-tesis/gpt/piloto-daniel --speaker spk_daniel
python -m app.cli make-tasks REC --annotators usr_a,usr_b --split pilot --segments s001,s012,s024
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np

from app import data_dir, media, media_dir, open_store
from app.ops import create_campaign, create_tasks, new_user, pair_cases
from app.worker import prepare_recording
from core import evaluation

IMPORTER_VERSION = "pilot-import-v1"
TEXT_SCOPE = {"mrcd": "window_context", "gpt": "model_proposal", "rules": None}


def add_user(store, a):
    u, token = new_user(store, a.name, a.role)
    print(f"{u['id']}  token (se muestra una sola vez): {token}")


def add_project(store, a):
    store.create(
        "project",
        {
            "purpose": a.purpose,
            "language": a.language,
            "guideline_version": a.guide,
            "external_processing": a.external_processing,
            "splits": {},
        },
        id=a.id,
        status="active",
    )
    print(a.id)


def _pilot_events(system: str, raw: dict, run_id: str) -> list[dict]:
    out = []
    for i, e in enumerate(raw["events"]):
        out.append(
            {
                "event_id": f"{run_id}-e{i:04d}",
                "label": e["label"],
                "decision": e.get("decision", "event"),
                "start_ms": round(e["start"] * 1000),
                "end_ms": round(e["end"] * 1000),
                "segment_id": e.get("segment_id"),
                "text": e.get("text"),
                "text_scope": TEXT_SCOPE[system],
                "verbatim_text": None,
                "evidence": e.get("evidence"),
                "score": e.get("score_uncalibrated"),
                "score_kind": "uncalibrated" if "score_uncalibrated" in e else None,
                "source_kind": "model_prediction",
            }
        )
    return out


def import_pilot(store, a):
    """Import the Daniel pilot without touching the source files; counts are printed
    so they can be checked against the audit (57 MRCD events, 30 legitimate pauses)."""
    root = Path(a.dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    wav = root / "audio" / f"{manifest['recording_id']}-16k.wav"
    if media.sha256(wav) != manifest["audio_sha256"]:
        raise SystemExit("El audio no coincide con el manifiesto; no se importa nada.")
    sha = manifest["audio_sha256"]
    rec = next((r for r in store.find("recording") if r["data"]["sha256"] == sha), None)
    if rec is None:
        target = media_dir({"data": {"sha256": sha}})
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wav, target / "original.wav")
        rec = store.create(
            "recording",
            {
                "sha256": sha,
                "original_name": "original.wav",
                "filename": wav.name,
                "speaker_id": a.speaker,
                "project": a.project,
                "example": True,
                "provenance": {
                    "importer": IMPORTER_VERSION,
                    "source_original_sha256": manifest["original_sha256"],
                    "manifest": str(root / "manifest.json"),
                },
            },
            owner=a.owner,
            status="queued",
        )
        prepare_recording(store, rec)
        rec = store.get(rec["id"])
        segs = [
            {
                "id": s["id"],
                "pilot": s["pilot"],
                **{f"{k}_ms": round(s[k] * 1000) for k in ("core_start", "core_end", "audio_start", "audio_end")},
            }
            for s in manifest["segments"]
        ]
        rec = store.update(rec["id"], {**rec["data"], "segments": segs})
    pilot = [s["id"] for s in manifest["segments"] if s["pilot"]]
    for system in ("gpt", "mrcd", "rules"):
        path = next(
            (
                p
                for p in (root / "resultados" / "piloto" / f"{system}{suffix}.json" for suffix in ("", "_partial"))
                if p.exists()
            ),
            None,
        )
        if path is None:
            print(f"{system}: sin resultados")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        identity = f"import:{media.sha256(path)}"
        run = next((r for r in store.find("run", parent=rec["id"]) if r["data"]["identity"] == identity), None)
        if run is None:
            done = {r["segment_id"] for r in raw.get("records", [])}
            segments = {s: "succeeded" if s in done else "not_run" for s in pilot}
            status = (
                "succeeded"
                if all(v == "succeeded" for v in segments.values()) and not raw.get("incomplete")
                else "partial"
            )
            run = store.create(
                "run",
                {
                    "system": system,
                    "identity": identity,
                    "comparison": "cmp_pilot_daniel",
                    "imported_from": str(path),
                    "importer": IMPORTER_VERSION,
                    "config": {
                        k: raw.get(k)
                        for k in ("asr_chunk_s", "device", "input_scope", "model", "checkpoint")
                        if k in raw
                    },
                },
                parent=rec["id"],
                owner=a.owner,
                status=status,
            )
            per_segment = {r["segment_id"]: r.get("seconds") for r in raw.get("records", [])}
            result = {
                "events": _pilot_events(system, raw, run["id"]),
                "segments": segments,
                "legitimate": [
                    {**e, "start_ms": round(e["start"] * 1000), "end_ms": round(e["end"] * 1000)}
                    for e in raw.get("legitimate_uses", [])
                ],
                "records": raw.get("records", []),
                "incomplete": bool(raw.get("incomplete")),
                "timings": {
                    "per_segment_s": per_segment,
                    "total_s": sum(v or 0 for v in per_segment.values()),
                    "initialization_s": raw.get("initialization_seconds"),
                },
                "model_version": {k: raw.get(k) for k in ("model", "prompt_sha256", "checkpoint_sha256") if k in raw},
                "warning": next(
                    (r.get("warning") for r in raw.get("records", []) if r.get("warning")), raw.get("timestamp_status")
                ),
            }
            run = store.update(run["id"], {**run["data"], "result": result})
        ev = [e for e in run["data"]["result"]["events"] if e["decision"] == "event"]
        pauses = sum(e["label"] in evaluation.PAUSES for e in ev)
        print(
            f"{system}: {run['status']} {run['data']['result']['segments']} eventos={len(ev)} "
            f"disfluencias={len(ev) - pauses} pausas={pauses} "
            f"usos_legítimos={len(run['data']['result']['legitimate'])} "
            f"segundos={run['data']['result']['timings']['total_s']:.2f} "
            f"init={run['data']['result']['timings']['initialization_s']}"
        )
    print(f"grabación {rec['id']}")


def make_tasks(store, a):
    try:
        tasks = create_tasks(
            store,
            a.recording,
            a.annotators.split(","),
            mode=a.mode,
            split=a.split,
            segments=a.segments.split(",") if a.segments else None,
            run_id=a.run,
            region_s=a.region_s,
            context_s=a.context_s,
            guide=a.guide,
            author="cli",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(f"{len(tasks)} tareas")


def make_cases(store, a):
    created, skipped = pair_cases(store, a.adjudicator, "cli")
    print(chr(10).join(created + [f"omitido {x}" for x in skipped]) or "sin regiones con dos anotaciones finalizadas")


def assign_splits(store, a):
    """Speaker-grouped split fixed before training; pilot speakers never go to test."""
    project = store.get(a.project, "project")
    speakers = sorted(
        {
            r["data"].get("speaker_id")
            for r in store.find("recording")
            if r["data"].get("speaker_id") and r["data"].get("project") == a.project
        }
    )
    excluded = set(a.exclude_test.split(",")) if a.exclude_test else set()
    rng = np.random.default_rng(a.seed)
    order = [speakers[i] for i in rng.permutation(len(speakers))]
    test = [s for s in order if s not in excluded][: a.test]
    rest = [s for s in order if s not in test]
    splits = {**{s: "test" for s in test}, **{s: "dev" for s in rest[: a.dev]}, **{s: "train" for s in rest[a.dev :]}}
    if project["data"].get("splits") and not a.force:
        raise SystemExit("La partición ya existe; cambiarla después de entrenar filtra información. Usa --force.")
    store.update(project["id"], {**project["data"], "splits": splits, "split_seed": a.seed}, author="cli")
    print(json.dumps(splits, indent=1))


def queue_dev_cases(store, a):
    """Improvement cycle step 3: assisted training cases from dev disagreements."""
    run_a, run_b = store.get(a.run_a, "run"), store.get(a.run_b, "run")
    rec = store.get(run_a["parent"])
    split = (
        (store.get(rec["data"].get("project") or "", "project") or {"data": {}})["data"]
        .get("splits", {})
        .get(rec["data"].get("speaker_id"), a.split)
    )
    ev_a, ev_b = run_a["data"]["result"]["events"], run_b["data"]["result"]["events"]
    matched = {i for i, _ in evaluation.match(ev_a, ev_b, evaluation.by_iou(0.5))}
    cases = [{**e, "kind": "disagreement"} for i, e in enumerate(ev_a) if i not in matched]
    cases += [{**e, "kind": "uncertain"} for e in ev_a + ev_b if e.get("decision") == "uncertain"]
    duration = rec["data"]["media"]["analysis"]["duration_ms"]
    rng = np.random.default_rng(a.seed)
    cases += [
        {"start_ms": int(t), "end_ms": int(t) + 1000, "kind": "random"}
        for t in rng.integers(0, max(1, duration - 1000), a.n)
    ]
    for c in evaluation.select_for_review(
        [{**c, "split": split, "speaker_id": rec["data"].get("speaker_id")} for c in cases], a.n
    ):
        start = max(0, (c["start_ms"] + c["end_ms"]) // 2 - a.region_s * 500)
        store.create(
            "task",
            {
                "recording_id": rec["id"],
                "start_ms": start,
                "end_ms": min(duration, start + a.region_s * 1000),
                "context_ms": 5000,
                "mode": "assisted",
                "split": split,
                "assignee": None,
                "run_id": run_a["id"],
                "speaker_id": rec["data"].get("speaker_id"),
                "guideline_version": a.guide,
                "selection": c["selection"],
            },
            parent=rec["id"],
            status="pool",
        )
    print("casos en cola de revisión asistida (solo desarrollo/entrenamiento)")


def alias_speaker(store, a):
    """Same known participant under two IDs (e.g. two sessions) -> one identity for leakage checks."""
    project = store.get(a.project, "project")
    aliases = {**project["data"].get("speaker_aliases", {}), a.alias: a.canonical}
    store.update(project["id"], {**project["data"], "speaker_aliases": aliases}, author="cli")
    print(f"{a.alias} -> {a.canonical}")


def add_recording(store, a):
    """Register an audio/video file and prepare it now (no worker needed)."""
    src = Path(a.path)
    sha = media.sha256(src)
    existing = next((r for r in store.find("recording") if r["data"]["sha256"] == sha), None)
    if existing:
        print(f"ya existe: {existing['id']}")
        return
    target = media_dir({"data": {"sha256": sha}})
    target.mkdir(parents=True, exist_ok=True)
    name = "original" + src.suffix.lower()
    shutil.copyfile(src, target / name)
    provenance = {"source": src.name}
    if a.align_to:
        provenance["alignment"] = {
            "recording_id": a.align_to,
            "offset_ms": a.align_offset_ms,
            "meaning": "evento en t de esta grabación = t + offset_ms en la otra",
        }
    rec = store.create(
        "recording",
        {
            "sha256": sha,
            "original_name": name,
            "filename": a.label or src.name,
            "speaker_id": a.speaker,
            "project": a.project,
            "provenance": provenance,
        },
        status="queued",
    )
    prepare_recording(store, rec)
    rec = store.get(rec["id"])
    print(f"{rec['id']} {rec['status']} {rec['data'].get('error', '')}")


def make_campaign(store, a):
    """Blind campaign over the chosen 30 s segments, split in regions of --region-s seconds."""
    rec = store.get(a.recording, "recording")
    segs = [s for s in rec["data"]["segments"] if s["id"] in a.segments.split(",")]
    regions = [
        (start, min(start + a.region_s * 1000, s["core_end_ms"]))
        for s in segs
        for start in range(s["core_start_ms"], s["core_end_ms"], a.region_s * 1000)
    ]
    try:
        camp = create_campaign(
            store, a.slug, a.recording, regions, title=a.title, description=a.description, max_participants=a.max
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if camp["data"]["video"]:  # cut every clip now so reviewers never wait for ffmpeg
        orig = media_dir(rec) / rec["data"]["original_name"]
        dur = rec["data"]["media"]["analysis"]["duration_ms"]
        for start, end in regions:
            media.video_clip(orig, media_dir(rec), max(0, start - 5000), min(dur, end + 5000))
    print(
        f"campaña {a.slug}: {len(regions)} regiones, {sum(e - s for s, e in regions) / 1000:.0f} s por persona, "
        f"video={camp['data']['video']}. Enlace: <URL>/#/c/{a.slug}"
    )


def promote(store, a):
    model = store.get(a.model, "model")
    store.update(model["id"], {**model["data"], "promotion": {"reason": a.reason}}, status="promoted", author="cli")
    print(f"{model['id']} promovido; los checkpoints anteriores se conservan")


def backup(store, a):
    """Copy DB and media, then verify the copy by reopening it and re-hashing media."""
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    db = data_dir() / "app.db"
    with sqlite3.connect(db) as src, sqlite3.connect(dest / "app.db") as out:
        src.backup(out)
    shutil.copytree(data_dir() / "media", dest / "media", dirs_exist_ok=True)
    with sqlite3.connect(dest / "app.db") as check:
        copied = check.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    originals = [r for r in store.find("recording")]
    bad = [
        r["id"]
        for r in originals
        if media.sha256(dest / "media" / r["data"]["sha256"] / r["data"]["original_name"]) != r["data"]["sha256"]
    ]
    print(f"documentos={copied} grabaciones={len(originals)} hashes_invalidos={bad}")
    if bad:
        raise SystemExit(1)


def quality_report(store, a):
    """Interaction targets (p95) and review effort measured from client telemetry."""
    anns = [x for x in store.find("annotation") if x["data"].get("telemetry")]
    report = {}
    for key, target in (("open_ms", 2000), ("save_ms", 1000), ("play_ms", 150)):
        values = [v for x in anns for v in x["data"]["telemetry"].get(key, [])]
        report[key] = {"n": len(values), "p95": float(np.percentile(values, 95)) if values else None, "target": target}
    done = [x for x in store.find("annotation", status="submitted")]
    audio_ms = sum(store.get(x["parent"])["data"]["end_ms"] - store.get(x["parent"])["data"]["start_ms"] for x in done)
    report["effort_active_min_per_audio_min"] = (
        sum(x["data"]["active_ms"] for x in done) / audio_ms if audio_ms else None
    )
    report["agents"] = sorted({x["data"]["telemetry"].get("agent") for x in anns} - {None})
    scores = [x["data"]["score"] for x in store.find("sus")]
    report["sus"] = {"n": len(scores), "mean": sum(scores) / len(scores) if scores else None, "target": 70}
    report["feedback_pending"] = len(store.find("feedback", status="pending"))
    print(json.dumps(report, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add-user")
    p.add_argument("name")
    p.add_argument("role", choices=["admin", "annotator", "adjudicator", "researcher", "user"])
    p = sub.add_parser("add-project")
    p.add_argument("id")
    p.add_argument("--purpose", required=True)
    p.add_argument("--language", default="es")
    p.add_argument("--guide", default="guia-anotacion-v1")
    p.add_argument("--external-processing", action="store_true")
    p = sub.add_parser("import-pilot")
    p.add_argument("dir")
    p.add_argument("--speaker", default="spk_daniel")
    p.add_argument("--project")
    p.add_argument("--owner")
    p = sub.add_parser("make-tasks")
    p.add_argument("recording")
    p.add_argument("--annotators", required=True)
    p.add_argument("--mode", choices=["blind", "assisted"], default="blind")
    p.add_argument("--split", choices=["pilot", "train", "dev", "test"])
    p.add_argument("--segments")
    p.add_argument("--run")
    p.add_argument("--region-s", type=int, default=15)
    p.add_argument("--context-s", type=int, default=5)
    p.add_argument("--guide", default="guia-anotacion-v1")
    p = sub.add_parser("make-cases")
    p.add_argument("--adjudicator", required=True)
    p = sub.add_parser("assign-splits")
    p.add_argument("project")
    p.add_argument("--dev", type=int, default=6)
    p.add_argument("--test", type=int, default=6)
    p.add_argument("--exclude-test", help="hablantes del piloto de guía, separados por coma")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("queue-dev-cases")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--split", default="dev")
    p.add_argument("--region-s", type=int, default=15)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--guide", default="guia-anotacion-v1")
    p = sub.add_parser("alias-speaker")
    p.add_argument("project")
    p.add_argument("alias")
    p.add_argument("canonical")
    p = sub.add_parser("add-recording")
    p.add_argument("path")
    p.add_argument("--speaker")
    p.add_argument("--project")
    p.add_argument("--label", help="nombre visible")
    p.add_argument("--align-to", help="otra grabación del mismo audio")
    p.add_argument("--align-offset-ms", type=float, default=0.0)
    p = sub.add_parser("make-campaign")
    p.add_argument("slug")
    p.add_argument("--recording", required=True)
    p.add_argument("--segments", required=True, help="p. ej. s001,s012,s024")
    p.add_argument("--region-s", type=int, default=15)
    p.add_argument("--title", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--max", type=int, default=30)
    p = sub.add_parser("promote")
    p.add_argument("model")
    p.add_argument("--reason", required=True)
    sub.add_parser("quality-report")
    p = sub.add_parser("backup")
    p.add_argument("dest")
    a = ap.parse_args(argv)
    globals()[a.cmd.replace("-", "_")](open_store(), a)


if __name__ == "__main__":
    main()
