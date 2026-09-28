"""Data-manager operations shared by the web administration view and the CLI.
They raise ValueError with a user-facing message; callers turn it into 422 or exit."""

from __future__ import annotations

import hashlib
import random
import secrets

from app.store import Store

ROLES = ("admin", "annotator", "adjudicator", "researcher", "user")


def new_user(store: Store, name: str, role: str) -> tuple[dict, str]:
    if role not in ROLES or not name.strip():
        raise ValueError("nombre vacío o rol desconocido")
    token = secrets.token_urlsafe(24)
    doc = store.create(
        "user",
        {"name": name.strip(), "role": role, "token_sha256": hashlib.sha256(token.encode()).hexdigest()},
        status="active",
    )
    return doc, token


def create_tasks(
    store: Store,
    recording_id: str,
    annotators: list[str],
    *,
    mode: str = "blind",
    split: str | None = None,
    segments: list[str] | None = None,
    run_id: str | None = None,
    region_s: int = 15,
    context_s: int = 5,
    guide: str = "guia-anotacion-v1",
    author: str | None = None,
) -> list[dict]:
    """One task per annotator per region (10–20 s) of the selected analysis segments."""
    rec = store.get(recording_id, "recording")
    if rec is None or rec["status"] != "ready":
        raise ValueError("la grabación no existe o aún no está preparada")
    if not annotators or any(
        (u := store.get(a, "user")) is None or u["data"]["role"] not in ("annotator", "admin") for a in annotators
    ):
        raise ValueError("elige personas con rol de anotación")
    project = store.get(rec["data"].get("project") or "", "project") or {"data": {}}
    split = split or project["data"].get("splits", {}).get(rec["data"].get("speaker_id"))
    if not split:
        raise ValueError("la grabación no tiene partición: elige una o asígnala al proyecto")
    if split == "test" and mode == "assisted":
        raise ValueError("el conjunto de prueba solo admite anotación ciega")
    if not 5 <= region_s <= 30:
        raise ValueError("la región debe durar entre 5 y 30 s")
    segs = [s for s in rec["data"]["segments"] if not segments or s["id"] in segments]
    return [
        store.create(
            "task",
            {
                "recording_id": rec["id"],
                "start_ms": start,
                "end_ms": min(start + region_s * 1000, seg["core_end_ms"]),
                "context_ms": context_s * 1000,
                "mode": mode,
                "split": split,
                "assignee": who,
                "run_id": run_id,
                "speaker_id": rec["data"].get("speaker_id"),
                "guideline_version": guide,
            },
            parent=rec["id"],
            owner=who,
            status="assigned",
            author=author,
        )
        for seg in segs
        for start in range(seg["core_start_ms"], seg["core_end_ms"], region_s * 1000)
        for who in annotators
    ]


def open_case(store: Store, task_a: str, task_b: str, adjudicator: str, author: str) -> dict:
    """Two submitted independent annotations of one region -> adjudication case (A/B order shuffled)."""
    a, b = store.get(task_a, "task"), store.get(task_b, "task")
    if a is None or b is None:
        raise ValueError("tarea no encontrada")
    region = {(t["data"]["recording_id"], t["data"]["start_ms"], t["data"]["end_ms"]) for t in (a, b)}
    if len(region) != 1 or {a["status"], b["status"]} != {"submitted"} or a["owner"] == b["owner"]:
        raise ValueError("se requieren dos tareas finalizadas, de personas distintas, sobre la misma región")
    if adjudicator in (a["owner"], b["owner"]):
        raise ValueError("el adjudicador debe ser distinto de los anotadores")
    order = [a["id"], b["id"]]
    random.shuffle(order)
    d = a["data"]
    return store.create(
        "case",
        {
            "tasks": order,
            **{
                k: d.get(k)
                for k in (
                    "recording_id",
                    "start_ms",
                    "end_ms",
                    "context_ms",
                    "split",
                    "speaker_id",
                    "guideline_version",
                )
            },
            "mode": "assisted" if "assisted" in (d["mode"], b["data"]["mode"]) else "blind",
        },
        parent=d["recording_id"],
        owner=adjudicator,
        status="open",
        author=author,
    )


def pair_cases(store: Store, adjudicator: str, author: str) -> tuple[list[str], list[str]]:
    """Every region with two finished annotations by different people and no case yet."""
    used = {t for c in store.find("case") for t in c["data"]["tasks"]}
    groups: dict = {}
    for t in store.find("task", status="submitted"):
        if t["id"] not in used:
            groups.setdefault((t["data"]["recording_id"], t["data"]["start_ms"], t["data"]["end_ms"]), {}).setdefault(
                t["owner"], t
            )
    created, skipped = [], []
    for owners in groups.values():
        if len(owners) < 2:
            continue
        first, second = list(owners.values())[:2]
        try:
            created.append(open_case(store, first["id"], second["id"], adjudicator, author)["id"])
        except ValueError as exc:
            skipped.append(f"{first['id']}/{second['id']}: {exc}")
    return created, skipped
