"""Data-manager operations shared by the web administration view and the CLI.
They raise ValueError with a user-facing message; callers turn it into 422 or exit."""

from __future__ import annotations

import hashlib
import random
import re
import secrets

from app.store import Store

ROLES = ("admin", "annotator", "adjudicator", "researcher", "user")


def new_user(store: Store, name: str, role: str, token: str | None = None) -> tuple[dict, str]:
    if role not in ROLES or not name.strip():
        raise ValueError("nombre vacío o rol desconocido")
    token = token or secrets.token_urlsafe(24)
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


# --------------------------------------------------------------- campaigns
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{2,39}$")
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")


def create_campaign(
    store: Store,
    slug: str,
    recording_id: str,
    regions: list[tuple[int, int]],
    *,
    title: str,
    description: str = "",
    context_ms: int = 5000,
    max_participants: int = 30,
    guide: str = "guia-anotacion-v1",
) -> dict:
    """Same predefined regions for every participant, blind, so their annotations are independent."""
    rec = store.get(recording_id, "recording")
    if not SLUG.match(slug) or store.get(slug) is not None:
        raise ValueError("identificador inválido o ya usado (minúsculas, números y guiones)")
    if rec is None or rec["status"] != "ready" or not regions:
        raise ValueError("la grabación no existe, no está preparada o no hay regiones")
    video = bool(rec["data"]["media"]["original"].get("has_video"))
    return store.create(
        "campaign",
        {
            "title": title,
            "description": description,
            "recording_id": recording_id,
            "regions": [list(r) for r in regions],
            "context_ms": context_ms,
            "mode": "blind",
            "split": "pilot",
            "guideline_version": guide,
            "max_participants": max_participants,
            "video": video,
        },
        id=slug,
        status="open",
    )


def join_campaign(store: Store, slug: str, name: str, email: str) -> tuple[dict, str]:
    """Name + email -> annotator account and one blind task per region. Joining again with
    the same email resumes the same tasks with a new token (older sessions stop working)."""
    camp = store.get(slug, "campaign")
    if camp is None or camp["status"] != "open":
        raise LookupError("campaña no disponible")
    name, email = name.strip(), email.strip().lower()
    if not 2 <= len(name) <= 80 or not EMAIL.match(email):
        raise ValueError("escribe tu nombre y un correo válido")
    people = [u for u in store.find("user", parent=slug)]
    user = next((u for u in people if u["data"].get("email") == email), None)
    token = secrets.token_urlsafe(24)
    digest = hashlib.sha256(token.encode()).hexdigest()
    if user is None:
        if len(people) >= camp["data"]["max_participants"]:
            raise ValueError("la campaña ya tiene todos sus participantes")
        user = store.create(
            "user",
            {"name": name, "role": "annotator", "email": email, "token_sha256": digest, "campaign": slug},
            parent=slug,
            status="active",
        )
    else:
        user = store.update(user["id"], {**user["data"], "name": name, "token_sha256": digest})
    d, rec = camp["data"], store.get(camp["data"]["recording_id"])
    have = {(t["data"]["start_ms"], t["data"]["end_ms"]) for t in store.find("task", owner=user["id"])}
    for start, end in d["regions"]:
        if (start, end) not in have:
            store.create(
                "task",
                {
                    "recording_id": d["recording_id"],
                    "start_ms": start,
                    "end_ms": end,
                    "context_ms": d["context_ms"],
                    "mode": "blind",
                    "split": d["split"],
                    "assignee": user["id"],
                    "run_id": None,
                    "campaign": slug,
                    "speaker_id": rec["data"].get("speaker_id"),
                    "guideline_version": d["guideline_version"],
                },
                parent=d["recording_id"],
                owner=user["id"],
                status="assigned",
            )
    return user, token


def campaign_progress(store: Store, slug: str) -> dict:
    """Per-participant progress and pairwise agreement on the regions both finished."""
    from core.evaluation import agreement

    camp = store.get(slug, "campaign")
    people = store.find("user", parent=slug)
    done: dict[str, dict] = {}
    rows = []
    for u in people:
        tasks = store.find("task", owner=u["id"])
        finished = [t for t in tasks if t["status"] == "submitted"]
        anns = {f"{t['data']['start_ms']}": (t["data"], store.get(f"ann_{t['id']}")) for t in finished}
        done[u["id"]] = {  # only marks touching the region, as the submit rule requires
            k: [
                e
                for e in a["data"]["events"]
                if e["decision"] == "event" and e["end_ms"] > td["start_ms"] and e["start_ms"] < td["end_ms"]
            ]
            for k, (td, a) in anns.items()
        }
        rows.append(
            {
                "name": u["data"]["name"],
                "email": u["data"]["email"],
                "submitted": len(finished),
                "total": len(tasks),
                "active_min": round(sum(a["data"].get("active_ms", 0) for _, a in anns.values()) / 60000, 1),
            }
        )
    pairs = []
    for i, a in enumerate(people):
        for b in people[i + 1 :]:
            common = sorted(set(done[a["id"]]) & set(done[b["id"]]))
            if common:
                ev = lambda uid: [{**e, "recording_id": k} for k in common for e in done[uid][k]]  # noqa: E731
                g = agreement(ev(a["id"]), ev(b["id"]))
                pairs.append(
                    {
                        "a": a["data"]["name"],
                        "b": b["data"]["name"],
                        "regions": len(common),
                        "existence": g["existence_agreement"],
                        "kappa": g["class_kappa"],
                        "raw": g["class_raw_agreement"],
                    }
                )
    return {
        "campaign": {
            "slug": slug,
            **{k: camp["data"][k] for k in ("title", "regions", "max_participants")},
            "status": camp["status"],
        },
        "participants": rows,
        "agreement": pairs,
    }
