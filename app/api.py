"""HTTP API. Heavy work runs in app.worker; this process never waits for inference.

    uvicorn app.api:app --port 8000     (web client at http://127.0.0.1:8000/)

Authorization is server-side by role and assignment. Blind review endpoints never
return predictions, scores, model names or selection reasons.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from app import ROOT, exports, media, media_dir, open_store
from app.ops import ROLES, create_tasks, new_user, open_case, pair_cases
from app.store import Conflict, Store
from core import evaluation
from core.constants import TAXONOMY
from core.engine import ASR_DEFAULTS

TERMINAL = {"succeeded", "partial", "failed", "cancelled"}
TAXONOMY_VERSION = "mrcd-taxonomy-v1"
GUIDES = ROOT / "docs"


# ------------------------------------------------------------------ contracts
class Event(BaseModel):
    event_id: str | None = None
    start_ms: int = Field(ge=0)
    end_ms: int
    label: str | None = None
    decision: Literal["event", "uncertain", "not_evaluable"] = "event"
    uncertainty: Literal["class", "boundaries", "audio"] | None = None
    verbatim_text: str | None = None
    context_text: str | None = None
    speaker_id: str | None = None
    parent_event_id: str | None = None
    source_kind: str = "human_independent"
    suggestion_id: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms debe ser mayor que start_ms")
        if self.label is not None and self.label not in TAXONOMY:
            raise ValueError(f"etiqueta fuera de {TAXONOMY_VERSION}")
        if self.decision == "event" and self.label is None:
            raise ValueError("un evento necesita etiqueta; usa decision=uncertain si no hay certeza")
        return self


class Contextual(BaseModel):
    """Legitimate/contextual use of an expression, kept apart from disfluent events."""

    start_ms: int = Field(ge=0)
    end_ms: int
    expression: str
    context_text: str | None = None
    is_disfluent: bool | None = None
    function: str | None = None


class AnnotationBody(BaseModel):
    events: list[Event] = []
    contextual: list[Contextual] = []
    coverage: list[tuple[int, int]] = []
    actions: list[dict] = []
    active_ms: int = 0
    asr_used: bool = False
    telemetry: dict = {}
    client_op: str | None = None


class RunRequest(BaseModel):
    recording_id: str
    systems: list[Literal["mrcd", "rules", "gpt"]] = ["mrcd"]
    config: dict = {}


class TaskRequest(BaseModel):
    recording_id: str
    start_ms: int = Field(ge=0)
    end_ms: int
    context_ms: int = 5000
    mode: Literal["blind", "assisted"] = "blind"
    split: Literal["pilot", "train", "dev", "test"] = "dev"
    assignee: str | None = None
    run_id: str | None = None
    guideline_version: str = "guia-anotacion-v1"


class CaseRequest(BaseModel):
    task_a: str
    task_b: str
    adjudicator: str


class AdjudicationRequest(BaseModel):
    case_id: str
    events: list[Event]
    contextual: list[Contextual] = []
    reasons: dict[str, str] = {}
    based_on: dict[str, int]


class SnapshotRequest(BaseModel):
    name: str
    include: Literal["adjudicated", "submitted"] = "adjudicated"
    splits: list[str] = ["train", "dev", "test"]


class EvaluationRequest(BaseModel):
    snapshot_id: str
    systems: dict[str, list[str]]
    protocol: dict = {}
    bootstrap: int = 1000


class FeedbackRequest(BaseModel):
    run_id: str
    kind: Literal["ok", "wrong", "missing"]
    event_id: str | None = None
    time_ms: int | None = None
    note: str | None = None


class SusRequest(BaseModel):
    answers: list[int]
    context: str = "playground"


class UserRequest(BaseModel):
    name: str
    role: Literal[ROLES]


class TaskBatchRequest(BaseModel):
    recording_id: str
    annotators: list[str]
    mode: Literal["blind", "assisted"] = "blind"
    split: Literal["pilot", "train", "dev", "test"] | None = None
    segments: list[str] | None = None
    run_id: str | None = None
    region_s: int = 15


class PairRequest(BaseModel):
    adjudicator: str


# ----------------------------------------------------------------- helpers
def need(user: dict, *roles: str) -> None:
    if user["data"]["role"] not in (*roles, "admin"):
        raise HTTPException(403, "rol no autorizado para esta acción")


def found(doc: dict | None, what: str = "recurso") -> dict:
    if doc is None:
        raise HTTPException(404, f"{what} no encontrado")
    return doc


def can_see_recording(user: dict, rec: dict) -> None:
    if rec["owner"] != user["id"] and not rec["data"].get("example"):
        need(user, "researcher")


def covered(intervals: list, start: int, end: int) -> list[tuple[int, int]]:
    """Gaps of [start, end) not covered by the given intervals."""
    gaps, cursor = [], start
    for a, b in sorted(map(tuple, intervals)):
        if a > cursor:
            gaps.append((cursor, min(a, end)))
        cursor = max(cursor, b)
        if cursor >= end:
            break
    return gaps + ([(cursor, end)] if cursor < end else [])


def create_app(store: Store | None = None) -> FastAPI:
    store = store or open_store()
    app = FastAPI(title="MRCD")
    api = APIRouter(prefix="/api")

    def user(request: Request) -> dict:
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or request.query_params.get(
            "token", ""
        )
        digest = hashlib.sha256(token.encode()).hexdigest()
        match = [u for u in store.find("user", status="active") if token and u["data"]["token_sha256"] == digest]
        if not match:
            raise HTTPException(401, "token inválido")
        return match[0]

    def own_task(task_id: str, u: dict) -> dict:
        task = found(store.get(task_id, "task"), "tarea")
        if task["owner"] != u["id"]:
            need(u)
        return task

    def annotation_of(task: dict) -> dict:
        ann = store.get(f"ann_{task['id']}")
        return ann or store.create(
            "annotation",
            AnnotationBody().model_dump(),
            id=f"ann_{task['id']}",
            parent=task["id"],
            owner=task["owner"],
            status="in_progress",
        )

    def clip(rec: dict, start_ms: int, end_ms: int) -> Response:
        duration = rec["data"]["media"]["analysis"]["duration_ms"]
        start_ms, end_ms = max(0, start_ms), min(end_ms, duration)
        if end_ms <= start_ms:
            raise HTTPException(422, "intervalo vacío o fuera del audio")
        return Response(
            media.clip_wav(media_dir(rec) / "analysis.wav", start_ms, end_ms),
            media_type="audio/wav",
            headers={"X-Media-Offset-Ms": str(start_ms), "Cache-Control": "private"},
        )

    # ------------------------------------------------------------ identity
    @api.get("/me")
    def me(u=Depends(user)):
        return {"id": u["id"], "name": u["data"]["name"], "role": u["data"]["role"]}

    @api.get("/guideline/{version}")
    def guideline(version: str, u=Depends(user)):
        path = GUIDES / f"{version}.md"
        if not path.resolve().is_relative_to(GUIDES.resolve()) or not path.exists():
            raise HTTPException(404, "guía no encontrada")
        return {"version": version, "markdown": path.read_text(encoding="utf-8")}

    # ---------------------------------------------------------- recordings
    @api.post("/recordings", status_code=202)
    async def upload(
        request: Request,
        filename: str,
        speaker_id: str | None = None,
        session_id: str | None = None,
        project: str | None = None,
        task: str | None = None,
        capture: str | None = None,
        meta: str | None = None,
        u=Depends(user),
    ):
        need(u, "user", "researcher")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            async for chunk in request.stream():
                tmp.write(chunk)
        sha = media.sha256(Path(tmp.name))
        mine = [r for r in store.find("recording", owner=u["id"]) if r["data"]["sha256"] == sha]
        if mine:
            Path(tmp.name).unlink()
            return {"id": mine[0]["id"], "status": mine[0]["status"], "duplicate": True}
        target = media_dir({"data": {"sha256": sha}})
        target.mkdir(parents=True, exist_ok=True)
        name = "original" + Path(filename).suffix.lower()
        shutil.move(tmp.name, target / name)
        rec = store.create(
            "recording",
            {
                "sha256": sha,
                "original_name": name,
                "filename": filename,
                "speaker_id": speaker_id,
                "session_id": session_id,
                "project": project,
                "task": task,
                "capture": json.loads(capture) if capture else None,
                "meta": json.loads(meta) if meta else None,  # variety, device, environment
            },
            owner=u["id"],
            status="queued",
            author=u["id"],
        )
        return {"id": rec["id"], "status": rec["status"]}

    @api.get("/recordings")
    def recordings(u=Depends(user)):
        docs = (
            store.find("recording")
            if u["data"]["role"] in ("researcher", "admin")
            else [r for r in store.find("recording") if r["owner"] == u["id"] or r["data"].get("example")]
        )
        return [
            {
                "id": r["id"],
                "status": r["status"],
                "filename": r["data"].get("filename"),
                "example": r["data"].get("example", False),
                "created": r["created"],
                "duration_ms": r["data"].get("media", {}).get("analysis", {}).get("duration_ms"),
                "speaker_id": r["data"].get("speaker_id"),
                "segments": [{"id": g["id"], "pilot": g.get("pilot", False)} for g in r["data"].get("segments", [])],
            }
            for r in docs
        ]

    @api.get("/recordings/{rid}")
    def recording(rid: str, u=Depends(user)):
        rec = found(store.get(rid, "recording"), "grabación")
        can_see_recording(u, rec)
        return rec

    @api.get("/recordings/{rid}/media")
    def recording_media(rid: str, variant: Literal["analysis", "original"] = "analysis", u=Depends(user)):
        rec = found(store.get(rid, "recording"), "grabación")
        can_see_recording(u, rec)
        path = media_dir(rec) / ("analysis.wav" if variant == "analysis" else rec["data"]["original_name"])
        return FileResponse(path, headers={"Cache-Control": "private"})  # honours HTTP Range

    @api.get("/recordings/{rid}/peaks")
    def peaks(rid: str, pps: int = 100, u=Depends(user)):
        if pps not in media.PEAK_LEVELS:
            raise HTTPException(422, f"pps debe ser uno de {media.PEAK_LEVELS}")
        rec = found(store.get(rid, "recording"), "grabación")
        can_see_recording(u, rec)
        import numpy as np

        values = np.load(media_dir(rec) / f"peaks_{pps}.npy").astype(float).round(3).tolist()
        return {"pps": pps, "duration_ms": rec["data"]["media"]["analysis"]["duration_ms"], "peaks": [values]}

    @api.get("/recordings/{rid}/clip")
    def recording_clip(rid: str, start_ms: int, end_ms: int, u=Depends(user)):
        rec = found(store.get(rid, "recording"), "grabación")
        can_see_recording(u, rec)
        return clip(rec, start_ms, end_ms)

    # ------------------------------------------------------ analysis runs
    @api.post("/analysis-runs", status_code=202)
    def create_runs(body: RunRequest, u=Depends(user)):
        need(u, "user", "researcher")
        rec = found(store.get(body.recording_id, "recording"), "grabación")
        can_see_recording(u, rec)
        if set(body.systems) - {"mrcd"}:
            need(u, "researcher")
        comparison = f"cmp_{random.getrandbits(48):012x}"
        runs = []
        config = {**ASR_DEFAULTS, **body.config}
        for system in body.systems:
            identity = hashlib.sha256(
                json.dumps([rec["data"]["sha256"], system, config], sort_keys=True).encode()
            ).hexdigest()
            same = [
                r
                for r in store.find("run", parent=rec["id"])
                if r["data"]["identity"] == identity and r["status"] not in ("failed", "cancelled")
            ]
            runs.append(
                same[0]
                if same
                else store.create(
                    "run",
                    {"system": system, "config": config, "identity": identity, "comparison": comparison},
                    parent=rec["id"],
                    owner=u["id"],
                    status="queued",
                )
            )
        return {
            "comparison_id": comparison,
            "runs": [{"id": r["id"], "system": r["data"]["system"], "status": r["status"]} for r in runs],
        }

    def readable_run(rid: str, u: dict) -> dict:
        need(u, "user", "researcher")
        run = found(store.get(rid, "run"), "ejecución")
        can_see_recording(u, store.get(run["parent"]))
        return run

    @api.get("/analysis-runs")
    def list_runs(recording_id: str, u=Depends(user)):
        return [readable_run(r["id"], u) for r in store.find("run", parent=recording_id)]

    @api.get("/analysis-runs/{rid}")
    def get_run(rid: str, u=Depends(user)):
        return readable_run(rid, u)

    @api.get("/analysis-runs/{rid}/stream")
    async def stream(rid: str, u=Depends(user)):
        readable_run(rid, u)

        async def events():
            last = None
            while True:
                run = store.get(rid)
                state = {"status": run["status"], "stage": run["data"].get("stage"), "rev": run["rev"]}
                if state != last:
                    yield f"data: {json.dumps(state)}\n\n"
                    last = state
                if run["status"] in TERMINAL:
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(events(), media_type="text/event-stream")

    @api.post("/analysis-runs/{rid}/cancel")
    def cancel(rid: str, u=Depends(user)):
        for _ in range(5):
            run = readable_run(rid, u)
            if run["status"] in TERMINAL | {"cancel_requested"}:
                return {"status": run["status"]}
            try:  # the status depends on the revision read: a claimed run is only asked to stop
                new = store.update(
                    rid,
                    {**run["data"], "cancel": {"by": u["id"]}},
                    expected_rev=run["rev"],
                    author=u["id"],
                    status="cancelled" if run["status"] == "queued" else "cancel_requested",
                )
                return {"status": new["status"]}
            except Conflict:
                continue
        raise HTTPException(409, "la ejecución cambió de estado; reintenta")

    @api.get("/experiments/comparisons/{cid}")
    def comparison(cid: str, u=Depends(user)):
        need(u, "researcher")
        runs = [r for r in store.find("run") if r["data"].get("comparison") == cid]
        if not runs:
            raise HTTPException(404, "comparación no encontrada")
        done = [r for r in runs if r["status"] in ("succeeded", "partial")]
        segments = store.get(runs[0]["parent"])["data"]["segments"]

        def common(a: dict, b: dict) -> dict:  # compare only segments both systems completed
            ok = [
                s for s in segments if all(r["data"]["result"]["segments"].get(s["id"]) == "succeeded" for r in (a, b))
            ]
            return {None: [(s["core_start_ms"], s["core_end_ms"]) for s in ok]}

        pairs = {
            f"{a['data']['system']}~{b['data']['system']}": {
                c: v["f1"]
                for c, v in evaluation.evaluate(
                    a["data"]["result"]["events"], b["data"]["result"]["events"], coverage=common(a, b)
                )["per_class"].items()
            }
            for i, a in enumerate(done)
            for b in done[i + 1 :]
        }
        reference = [
            a
            for a in store.find("adjudication")
            if a["data"]["recording_id"] == runs[0]["parent"] and a["data"]["split"] != "test"
        ]
        return {"runs": runs, "symmetric_agreement": pairs, "reference": reference}

    # ------------------------------------------------------------ review
    @api.post("/review/tasks")
    def create_task(body: TaskRequest, u=Depends(user)):
        need(u, "researcher")
        if body.mode == "assisted" and body.split == "test":
            raise HTTPException(400, "el conjunto de prueba solo admite anotación ciega")
        if body.end_ms <= body.start_ms:
            raise HTTPException(422, "intervalo vacío")
        rec = found(store.get(body.recording_id, "recording"), "grabación")
        if rec["status"] != "ready":
            raise HTTPException(409, "la grabación aún no está preparada")
        task = store.create(
            "task",
            {**body.model_dump(), "speaker_id": rec["data"].get("speaker_id")},
            parent=rec["id"],
            owner=body.assignee,
            status="assigned" if body.assignee else "pool",
            author=u["id"],
        )
        return {"id": task["id"], "status": task["status"]}

    def public_task(task: dict) -> dict:
        d = task["data"]
        return {
            "id": task["id"],
            "status": task["status"],
            "mode": d["mode"],
            "start_ms": d["start_ms"],
            "end_ms": d["end_ms"],
            "context_ms": d["context_ms"],
            "guideline_version": d["guideline_version"],
        }

    @api.get("/review/tasks")
    def inbox(u=Depends(user)):
        tasks = [public_task(t) for t in store.find("task", owner=u["id"])]
        cases = [{"id": c["id"], "status": c["status"]} for c in store.find("case", owner=u["id"])]
        pending = sum(t["end_ms"] - t["start_ms"] for t in tasks if t["status"] != "submitted")
        return {"tasks": tasks, "cases": cases, "pending_minutes": round(pending / 60000, 2)}

    @api.get("/review/tasks/next")
    def next_task(u=Depends(user)):
        need(u, "annotator")
        mine = [t for t in store.find("task", owner=u["id"]) if t["status"] in ("assigned", "in_progress")]
        if not mine:  # development pool: assisted cases sent from the product or comparison view
            for t in store.find("task", status="pool"):
                try:
                    mine = [
                        store.update(t["id"], expected_rev=t["rev"], owner=u["id"], status="assigned", author=u["id"])
                    ]
                    break
                except Conflict:
                    continue
        return public_task(mine[0]) if mine else None

    @api.get("/review/tasks/{tid}")
    def get_task(tid: str, u=Depends(user)):
        task = own_task(tid, u)
        d = task["data"]
        suggestions = []
        if d["mode"] == "assisted" and d.get("run_id"):
            run = store.get(d["run_id"], "run") or {"data": {}}
            suggestions = [
                {k: e.get(k) for k in ("event_id", "start_ms", "end_ms", "label", "text", "decision")}
                for e in run["data"].get("result", {}).get("events", [])
                if e["end_ms"] > d["start_ms"] - d["context_ms"] and e["start_ms"] < d["end_ms"] + d["context_ms"]
            ]
        return {
            "task": public_task(task),
            "annotation": annotation_of(task),
            "suggestions": suggestions,
            "taxonomy": list(TAXONOMY),
            "taxonomy_version": TAXONOMY_VERSION,
        }

    @api.get("/review/tasks/{tid}/audio")
    def task_audio(tid: str, extra_ms: int = 0, u=Depends(user)):
        d = own_task(tid, u)["data"]
        ctx = d["context_ms"] + min(max(extra_ms, 0), 30_000)  # widen context without moving the region
        return clip(store.get(d["recording_id"]), d["start_ms"] - ctx, d["end_ms"] + ctx)

    @api.get("/annotations/{aid}/history")
    def annotation_history(aid: str, u=Depends(user)):
        ann = found(store.get(aid, "annotation"), "anotación")
        own_task(ann["parent"], u)
        return [
            {
                "rev": h["rev"],
                "created": h["created"],
                "status": h["status"],
                "author": h["author"],
                "n_events": len(h["data"].get("events", [])),
            }
            for h in store.history(aid)
        ]

    @api.get("/review/tasks/{tid}/asr")
    def task_asr(tid: str, u=Depends(user)):
        """ASR words only (no disfluency marks), offered after a first listen; the
        client records its use in the annotation (`asr_used`)."""
        d = own_task(tid, u)["data"]
        lo, hi = d["start_ms"] - d["context_ms"], d["end_ms"] + d["context_ms"]
        runs = [r for r in store.find("run", parent=d["recording_id"]) if r["data"].get("result", {}).get("words")]
        words = runs[-1]["data"]["result"]["words"] if runs else []
        return [
            {"text": w["text"], "start_ms": round(w["start"] * 1000), "end_ms": round(w["end"] * 1000)}
            for w in words
            if lo <= w["start"] * 1000 < hi
        ]

    @api.patch("/annotations/{aid}")
    def patch_annotation(aid: str, body: AnnotationBody, if_match: int = Header(...), u=Depends(user)):
        ann = found(store.get(aid, "annotation"), "anotación")
        task = own_task(ann["parent"], u)
        if ann["status"] == "submitted":
            raise HTTPException(409, "anotación finalizada; reabrir crea otra revisión")
        if body.client_op and body.client_op == ann["data"].get("client_op"):
            return ann  # idempotent retry of an already applied edit
        if task["data"]["mode"] == "assisted":
            for e in body.events:
                if e.suggestion_id:
                    e.source_kind = "human_verified_model_suggestion"
        try:
            new = store.update(aid, body.model_dump(), expected_rev=if_match, author=u["id"])
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from None
        if task["status"] == "assigned":
            store.update(task["id"], status="in_progress")
        return new

    @api.post("/review/tasks/{tid}/submit")
    def submit(tid: str, u=Depends(user)):
        task = own_task(tid, u)
        ann, d = annotation_of(task), task["data"]
        lo, hi = d["start_ms"] - d["context_ms"], d["end_ms"] + d["context_ms"]
        body = ann["data"]
        outside = [e for e in body["events"] + body["contextual"] if e["start_ms"] < max(0, lo) or e["end_ms"] > hi]
        gaps = covered(
            body["coverage"] + [(e["start_ms"], e["end_ms"]) for e in body["events"]], d["start_ms"], d["end_ms"]
        )
        if outside or gaps:
            raise HTTPException(422, {"outside_context": outside, "unreviewed_gaps_ms": gaps})
        store.update(ann["id"], status="submitted", author=u["id"])
        store.update(tid, status="submitted", author=u["id"])
        return {"status": "submitted"}

    @api.post("/review/tasks/{tid}/reopen")
    def reopen(tid: str, u=Depends(user)):
        need(u)
        found(store.get(tid, "task"), "tarea")
        store.update(f"ann_{tid}", status="in_progress", author=u["id"])
        store.update(tid, status="in_progress", author=u["id"])
        return {"status": "in_progress"}

    # -------------------------------------------------------- adjudication
    @api.post("/adjudication-cases")
    def create_case(body: CaseRequest, u=Depends(user)):
        need(u)
        try:
            return {"id": open_case(store, body.task_a, body.task_b, body.adjudicator, u["id"])["id"]}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    def own_case(cid: str, u: dict) -> dict:
        case = found(store.get(cid, "case"), "caso")
        if case["owner"] != u["id"]:
            need(u)
        return case

    @api.get("/review/adjudication/{cid}")
    def get_case(cid: str, u=Depends(user)):
        case = own_case(cid, u)
        tracks = {}
        for side, tid in zip("AB", case["data"]["tasks"]):
            ann = store.get(f"ann_{tid}")
            tracks[side] = {"rev": ann["rev"], "events": ann["data"]["events"], "contextual": ann["data"]["contextual"]}
        return {
            "case": {
                "id": cid,
                "status": case["status"],
                **{k: case["data"][k] for k in ("start_ms", "end_ms", "context_ms", "guideline_version")},
            },
            "tracks": tracks,
            "taxonomy": list(TAXONOMY),
        }

    @api.get("/review/adjudication/{cid}/audio")
    def case_audio(cid: str, extra_ms: int = 0, u=Depends(user)):
        d = own_case(cid, u)["data"]
        ctx = d["context_ms"] + min(max(extra_ms, 0), 30_000)
        return clip(store.get(d["recording_id"]), d["start_ms"] - ctx, d["end_ms"] + ctx)

    @api.post("/adjudications")
    def adjudicate(body: AdjudicationRequest, u=Depends(user)):
        case = own_case(body.case_id, u)
        current = {side: store.get(f"ann_{tid}")["rev"] for side, tid in zip("AB", case["data"]["tasks"])}
        if current != body.based_on:
            raise HTTPException(409, "las anotaciones A/B cambiaron; recarga el caso")
        adj = store.create(
            "adjudication",
            {
                **{
                    k: case["data"][k]
                    for k in ("recording_id", "start_ms", "end_ms", "split", "speaker_id", "guideline_version", "mode")
                },
                "case_id": case["id"],
                "events": [e.model_dump() for e in body.events],
                "contextual": [c.model_dump() for c in body.contextual],
                "reasons": body.reasons,
                "based_on": {
                    side: {"task": tid, "rev": current[side]} for side, tid in zip("AB", case["data"]["tasks"])
                },
            },
            parent=case["id"],
            owner=u["id"],
            status="final",
            author=u["id"],
        )
        store.update(case["id"], status="closed", author=u["id"])
        return {"id": adj["id"]}

    # ---------------------------------------------------- data snapshots
    @api.post("/dataset-snapshots")
    def snapshot(body: SnapshotRequest, u=Depends(user)):
        need(u)
        if body.include == "adjudicated":
            items = [
                a["data"] | {"source": a["id"], "source_kind": "human_adjudicated"}
                for a in store.find("adjudication", status="final")
            ]
        else:  # single submitted annotations, e.g. assisted training material
            items = []
            for t in store.find("task", status="submitted"):
                ann = store.get(f"ann_{t['id']}")
                items.append(
                    {
                        **{
                            k: t["data"][k]
                            for k in (
                                "recording_id",
                                "start_ms",
                                "end_ms",
                                "split",
                                "guideline_version",
                                "mode",
                                "speaker_id",
                            )
                        },
                        "events": ann["data"]["events"],
                        "contextual": ann["data"]["contextual"],
                        "source": f"{ann['id']}@{ann['rev']}",
                        "source_kind": "human_single_annotation",
                    }
                )
        items = [i for i in items if i["split"] in body.splits]
        splits_of: dict[str, set] = {}
        for i in items:
            rec = store.get(i["recording_id"])
            project = store.get(rec["data"].get("project") or "", "project") or {"data": {}}
            speaker = project["data"].get("speaker_aliases", {}).get(i["speaker_id"], i["speaker_id"])
            i["media_sha256"] = rec["data"]["media"]["analysis"]["sha256"]
            for key in (f"speaker:{speaker}", f"media:{i['media_sha256']}"):
                splits_of.setdefault(key, set()).add(i["split"])
        leaks = {k: sorted(v) for k, v in splits_of.items() if len(v) > 1 and not k.endswith(":None")}
        if leaks:
            raise HTTPException(409, {"leakage": leaks})
        missing_speaker = [i["recording_id"] for i in items if not i["speaker_id"]]
        if "test" in body.splits and missing_speaker:
            raise HTTPException(422, {"speaker_id_required": sorted(set(missing_speaker))})
        content = {
            "name": body.name,
            "include": body.include,
            "taxonomy_version": TAXONOMY_VERSION,
            "items": sorted(items, key=lambda i: (i["recording_id"], i["start_ms"])),
        }
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        existing = store.get(f"snp_{digest[:16]}")
        return existing or store.create(
            "snapshot",
            {**content, "sha256": digest},
            id=f"snp_{digest[:16]}",
            owner=u["id"],
            status="frozen",
            author=u["id"],
        )

    @api.post("/evaluation-runs")
    def evaluate(body: EvaluationRequest, u=Depends(user)):
        need(u, "researcher")
        snap = found(store.get(body.snapshot_id, "snapshot"), "snapshot")
        items = snap["data"]["items"]
        coverage: dict = {}
        ref = []
        for i in items:
            coverage.setdefault(i["recording_id"], []).append((i["start_ms"], i["end_ms"]))
            ref += [{**e, "recording_id": i["recording_id"], "speaker_id": i["speaker_id"]} for e in i["events"]]
        legit = [
            {**c, "recording_id": i["recording_id"]}
            for i in items
            for c in i["contextual"]
            if c.get("is_disfluent") is False
        ]
        evaluable = sum(b - a for spans in coverage.values() for a, b in spans)
        protocol = {"iou_threshold": 0.5, "sensitivity": (0.3, 0.7), "tolerance_ms": 100, **body.protocol}
        hyps, results, missing = {}, {}, {}
        for name, run_ids in body.systems.items():
            runs = [found(store.get(r, "run"), "ejecución") for r in run_ids]
            speaker = {r["id"]: store.get(r["parent"])["data"].get("speaker_id") for r in runs}
            hyps[name] = [
                {**e, "recording_id": r["parent"], "speaker_id": speaker[r["id"]]}
                for r in runs
                for e in r["data"].get("result", {}).get("events", [])
                if evaluation.in_coverage({**e, "recording_id": r["parent"]}, coverage)
            ]
            missing[name] = sorted(
                set(coverage) - {r["parent"] for r in runs if r["status"] in ("succeeded", "partial")}
            )
            results[name] = evaluation.evaluate(
                ref, hyps[name], coverage=coverage, evaluable_ms=evaluable, legitimate=legit, **protocol
            )

        def f1(r, *h):
            return evaluation.evaluate(r, h[0], coverage=None, **protocol)["micro"]["disfluency"]["f1"]

        cis = {n: evaluation.bootstrap_ci(ref, [hyps[n]], f1, n=body.bootstrap) for n in hyps}
        names = list(hyps)
        if len(names) == 2:
            cis[f"{names[1]}-{names[0]}"] = evaluation.bootstrap_ci(
                ref,
                [hyps[names[1]], hyps[names[0]]],
                lambda r, a, b: None if f1(r, a) is None or f1(r, b) is None else f1(r, a) - f1(r, b),
                n=body.bootstrap,
            )
        prior = [e for e in store.find("evaluation") if e["data"]["snapshot_id"] == snap["id"]]
        doc = store.create(
            "evaluation",
            {
                "snapshot_id": snap["id"],
                "snapshot_sha256": snap["data"]["sha256"],
                "systems": body.systems,
                "protocol": protocol,
                "results": results,
                "missing_runs": missing,
                "bootstrap_ci": cis,
                "prior_evaluations_on_snapshot": len(prior),
            },
            owner=u["id"],
            status="final",
            author=u["id"],
        )
        return doc

    # ------------------------------------------------ product feedback
    @api.post("/feedback")
    def feedback(body: FeedbackRequest, u=Depends(user)):
        run = readable_run(body.run_id, u)
        doc = store.create(
            "feedback",
            {**body.model_dump(), "recording_id": run["parent"]},
            parent=run["id"],
            owner=u["id"],
            status="pending",
            author=u["id"],
        )
        return {"id": doc["id"], "status": doc["status"]}

    @api.post("/sus")
    def sus(body: SusRequest, u=Depends(user)):
        try:
            score = evaluation.sus_score(body.answers)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        store.create("sus", {**body.model_dump(), "score": score, "role": u["data"]["role"]}, owner=u["id"])
        return {"score": score}

    @api.get("/sus/summary")
    def sus_summary(u=Depends(user)):
        need(u, "researcher")
        scores = [d["data"]["score"] for d in store.find("sus")]
        mean = sum(scores) / len(scores) if scores else None
        sd = (sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)) ** 0.5 if len(scores) > 1 else None
        return {"n": len(scores), "mean": mean, "sd": sd, "target": 70, "min_users": 10}

    # --------------------------------------------------------- exports
    @api.get("/exports/{doc_id}")
    def export(doc_id: str, format: Literal["json", "csv", "md", "eaf"] = "json", u=Depends(user)):
        doc = found(store.get(doc_id), "documento")
        if doc["kind"] == "run":
            readable_run(doc_id, u)
            events = doc["data"].get("result", {}).get("events", [])
        elif doc["kind"] in ("snapshot", "evaluation"):
            need(u, "researcher")
            events = [
                {**e, "recording_id": i["recording_id"], "speaker_id": i["speaker_id"]}
                for i in doc["data"].get("items", [])
                for e in i["events"]
            ]
        else:
            raise HTTPException(400, "exportación no disponible para este tipo")
        name = f"{doc_id}.{format}"
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        if format == "csv":
            return Response(exports.events_csv(events), media_type="text/csv", headers=headers)
        if format == "md":
            if doc["kind"] != "run":
                raise HTTPException(400, "el reporte legible es por ejecución")
            return Response(exports.run_report(doc), media_type="text/markdown", headers=headers)
        if format == "eaf":
            return Response(
                exports.eaf(events, f"{doc.get('parent') or doc_id}.wav", u["data"]["name"]),
                media_type="application/xml",
                headers=headers,
            )
        return Response(
            json.dumps({**doc, "export_version": "mrcd-export-v1"}, ensure_ascii=False, indent=1),
            media_type="application/json",
            headers=headers,
        )

    # ------------------------------------------------ administration (data manager)
    def fail(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @api.get("/admin/overview")
    def overview(u=Depends(user)):
        need(u)
        names = {x["id"]: x["data"]["name"] for x in store.find("user")}
        return {
            "users": [
                {"id": x["id"], "name": x["data"]["name"], "role": x["data"]["role"], "status": x["status"]}
                for x in store.find("user")
            ],
            "tasks": [
                {
                    **public_task(t),
                    "owner": names.get(t["owner"]),
                    "recording_id": t["data"]["recording_id"],
                    "split": t["data"]["split"],
                }
                for t in store.find("task")
            ],
            "cases": [
                {
                    "id": c["id"],
                    "status": c["status"],
                    "adjudicator": names.get(c["owner"]),
                    "recording_id": c["data"]["recording_id"],
                    "start_ms": c["data"]["start_ms"],
                    "end_ms": c["data"]["end_ms"],
                }
                for c in store.find("case")
            ],
        }

    @api.post("/admin/users")
    def add_user(body: UserRequest, u=Depends(user)):
        need(u)
        doc, token = fail(new_user, store, body.name, body.role)
        return {"id": doc["id"], "token": token}

    @api.post("/admin/tasks")
    def add_tasks(body: TaskBatchRequest, u=Depends(user)):
        need(u)
        return {
            "created": len(
                fail(
                    create_tasks,
                    store,
                    body.recording_id,
                    body.annotators,
                    mode=body.mode,
                    split=body.split,
                    segments=body.segments,
                    run_id=body.run_id,
                    region_s=body.region_s,
                    author=u["id"],
                )
            )
        }

    @api.post("/admin/cases")
    def add_cases(body: PairRequest, u=Depends(user)):
        need(u)
        created, skipped = pair_cases(store, body.adjudicator, u["id"])
        return {"created": created, "skipped": skipped}

    # ------------------------------------------------------ experiments
    @api.get("/experiments")
    def experiments(u=Depends(user)):
        need(u, "researcher")
        files = {r["id"]: r["data"].get("filename") for r in store.find("recording")}
        return {
            "runs": [
                {
                    "id": r["id"],
                    "system": r["data"]["system"],
                    "status": r["status"],
                    "recording_id": r["parent"],
                    "filename": files.get(r["parent"]),
                    "comparison": r["data"].get("comparison"),
                    "created": r["created"],
                }
                for r in store.find("run")
            ],
            "snapshots": [
                {
                    "id": x["id"],
                    "name": x["data"]["name"],
                    "items": len(x["data"]["items"]),
                    "splits": sorted({i["split"] for i in x["data"]["items"]}),
                    "created": x["created"],
                }
                for x in store.find("snapshot")
            ],
            "evaluations": [
                {
                    "id": x["id"],
                    "snapshot_id": x["data"]["snapshot_id"],
                    "created": x["created"],
                    "systems": list(x["data"]["systems"]),
                    "results": x["data"]["results"],
                    "bootstrap_ci": x["data"]["bootstrap_ci"],
                    "prior": x["data"]["prior_evaluations_on_snapshot"],
                }
                for x in store.find("evaluation")
            ],
        }

    app.include_router(api)

    # Cache-busting: browsers cache /app.js and /app.css aggressively across deploys.
    # The index route stamps a version query string derived from the files' content,
    # so a new deploy is fetched automatically instead of needing a hard refresh.
    static_dir = Path(__file__).parent / "static"
    index_html = (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/", include_in_schema=False)
    def index():
        version = hashlib.sha256(
            b"".join((static_dir / name).read_bytes() for name in ("app.js", "app.css"))
        ).hexdigest()[:10]
        html = index_html.replace("/app.js", f"/app.js?v={version}").replace("/app.css", f"/app.css?v={version}")
        return Response(html, media_type="text/html")

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


def __getattr__(name):  # `uvicorn app.api:app` builds the app lazily, tests call create_app()
    if name == "app":
        return create_app()
    raise AttributeError(name)
