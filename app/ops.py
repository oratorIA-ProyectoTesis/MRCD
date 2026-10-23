"""Data-manager operations shared by the web administration view and the CLI.
They raise ValueError with a user-facing message; callers turn it into 422 or exit."""

from __future__ import annotations

import hashlib
import random
import re
import secrets

from app.store import Conflict, Store, now
from core.constants import TAXONOMY_VERSION

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


def require_generic_annotator(store: Store, user_id: str) -> None:
    """Campaign-only accounts must receive work through their campaign, not generic assignment."""
    person = store.get(user_id, "user")
    if not person or person["status"] != "active" or person["data"].get("role") not in ("annotator", "admin"):
        raise ValueError("Choose an active annotator or administrator.")
    if person["data"].get("campaign") or store.get(person["parent"] or "", "campaign"):
        raise ValueError("Campaign participants cannot receive generic tasks; use their authorized campaign extension.")


def require_task_source(store: Store, run_id: str | None, recording_id: str):
    if run_id:
        from app.human_review import task_run

        run = store.get(run_id, "run")
        if not run or run["parent"] != recording_id:
            raise ValueError("task source run must belong to its recording")
        try:
            task_run(store, {"data":dict(run_id=run_id,recording_id=recording_id)})
        except ValueError as exc:
            raise ValueError(f"task proposal adapter rejected result: {exc}") from exc


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
    if mode == "assisted":
        require_task_source(store, run_id, recording_id)
    if not annotators:
        raise ValueError("elige personas con rol de anotación")
    for user_id in annotators:  # validate the whole batch before creating any tasks
        require_generic_annotator(store, user_id)
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


def require_global_adjudicator(store: Store, user_id: str) -> None:
    person = store.get(user_id, "user")
    if not person or person["status"] != "active" or person["data"].get("role") not in ("adjudicator", "admin"):
        raise ValueError("Choose an active adjudicator or administrator.")
    if person["data"].get("campaign") or store.get(person["parent"] or "", "campaign"):
        raise ValueError("Campaign-only accounts cannot adjudicate cases.")


def open_case(store: Store, task_a: str, task_b: str, adjudicator: str, author: str) -> dict:
    """Two submitted independent annotations of one region -> adjudication case (A/B order shuffled)."""
    require_global_adjudicator(store, adjudicator)
    a, b = store.get(task_a, "task"), store.get(task_b, "task")
    if a is None or b is None:
        raise ValueError("tarea no encontrada")
    region = {(t["data"]["recording_id"], t["data"]["start_ms"], t["data"]["end_ms"]) for t in (a, b)}
    if len(region) != 1 or {a["status"], b["status"]} != {"submitted"} or a["owner"] == b["owner"]:
        raise ValueError("se requieren dos tareas finalizadas, de personas distintas, sobre la misma región")
    if adjudicator in (a["owner"], b["owner"]):
        raise ValueError("el adjudicador debe ser distinto de los anotadores")
    for key in ("split", "speaker_id", "guideline_version", "context_ms"):
        if a["data"].get(key) != b["data"].get(key):
            raise ValueError(f"{key}: independent task metadata must agree")
    rec = store.get(a["data"]["recording_id"], "recording")
    if not rec or rec["status"] != "ready":
        raise ValueError("recording_id: a ready recording is required")
    order = [a["id"], b["id"]]
    random.shuffle(order)
    d = a["data"]
    return store.create(
        "case",
        {
            "tasks": order,
            "taxonomy_version": TAXONOMY_VERSION,
            "media_sha256": rec["data"].get("media", {}).get("analysis", {}).get("sha256"),
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
            "source_modes": {side: store.get(tid, "task")["data"]["mode"] for side, tid in zip("AB", order)},
        },
        parent=d["recording_id"],
        owner=adjudicator,
        status="open",
        author=author,
    )


def pair_cases(store: Store, adjudicator: str, author: str) -> tuple[list[str], list[str]]:
    """Every region with two finished annotations by different people and no case yet."""
    require_global_adjudicator(store, adjudicator)  # reject before creating or consuming any pair
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
CONSENT_VERSION = "mrcd-campaign-consent-v1"
CONSENT_TEXT = (
    "I agree that my name, email, and annotations will be stored for the MRCD thesis research. "
    "I agree not to record, download, or share the supplied content."
)


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


def join_campaign(store: Store, slug: str, name: str, email: str, *, consent_version: str) -> tuple[dict, str]:
    """New consented registrations only; an email is never proof of token possession."""
    name, email = name.strip(), email.strip().lower()
    if not 2 <= len(name) <= 80 or not EMAIL.match(email):
        raise ValueError("escribe tu nombre y un correo válido")
    if consent_version != CONSENT_VERSION:
        raise ValueError("Reload the campaign and accept its current consent statement.")
    with store.serialized(slug) as tx:
        camp = tx.get(slug, "campaign")
        if camp is None or camp["status"] != "open":
            raise LookupError("campaña no disponible")
        people = tx.find("user", parent=slug)
        if any(u["data"].get("email", "").strip().lower() == email for u in people):
            raise Conflict("Unable to register. Return with your personal token, or contact the administrator.")
        if len(people) >= camp["data"]["max_participants"]:
            raise ValueError("la campaña ya tiene todos sus participantes")
        token = secrets.token_urlsafe(24)
        user = tx.create(
            "user",
            {"name": name, "role": "annotator", "email": email,
             "token_sha256": hashlib.sha256(token.encode()).hexdigest(), "campaign": slug,
             "consent": {"version": consent_version, "accepted_at": now(), "campaign": slug,
                         "text_sha256": hashlib.sha256(CONSENT_TEXT.encode()).hexdigest()}},
            parent=slug,
            status="active",
        )
        campaign_tasks(tx, camp, user)
        return user, token


def resume_campaign(store: Store, slug: str, token: str) -> dict:
    """Validate possession inside the same lock as resets; do not rotate or backfill consent."""
    with store.serialized(slug) as tx:
        camp = tx.get(slug, "campaign")
        if camp is None:
            raise LookupError("campaña no disponible")
        digest = hashlib.sha256(token.encode()).hexdigest()
        person = next((p for p in tx.find("user", status="active")
                       if token and secrets.compare_digest(p["data"].get("token_sha256", ""), digest)), None)
        if person is None:
            raise PermissionError("Invalid or inactive access token.")
        if person["data"].get("role") != "annotator" or person["parent"] != slug or person["data"].get("campaign") != slug:
            raise Conflict("This token does not belong to a participant in this campaign.")
        if camp["status"] == "open":
            campaign_tasks(tx, camp, person)
        return person


def reset_campaign_access(store: Store, slug: str, participant_id: str, actor: dict, *, reason: str) -> tuple[dict, str]:
    """Administrator-verified recovery, never email-based recovery or account reactivation."""
    if not 10 <= len(reason.strip()) <= 500:
        raise ValueError("Record a manual identity-check reason (10–500 characters; no secrets).")
    with store.serialized(slug) as tx:
        admin = tx.get(actor["id"], "user")
        if not admin or admin["status"] != "active" or admin["data"].get("role") != "admin":
            raise PermissionError("Only an active administrator can reset access.")
        person, camp = tx.get(participant_id, "user"), tx.get(slug, "campaign")
        if not camp or not person or person["status"] != "active" or person["parent"] != slug or (
            person["data"].get("campaign") != slug or person["data"].get("role") != "annotator"
        ):
            raise ValueError("Choose an active participant belonging to this campaign.")
        token = secrets.token_urlsafe(24)
        person = tx.update(person["id"], {
            **person["data"], "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "access_reset": {"by": admin["id"], "at": now(), "reason": reason.strip(), "identity_check": "manual"},
        }, author=admin["id"])
        return person, token


def extend_campaign(store: Store, slug: str, regions: list[tuple[int, int]]) -> dict:
    """Add regions to an open campaign (existing ones are kept) and assign them to everyone already in it."""
    with store.serialized(slug) as tx:
        camp = tx.get(slug, "campaign")
        if camp is None or camp["status"] != "open":
            raise ValueError("open campaign not found")
        old = [tuple(r) for r in camp["data"]["regions"]]
        camp = tx.update(slug, {**camp["data"], "regions": [list(r) for r in sorted(set(old) | set(map(tuple, regions)))]})
        for user in tx.find("user", parent=slug, status="active"):
            campaign_tasks(tx, camp, user)
        return camp


def campaign_tasks(store: Store, camp: dict, user: dict) -> None:
    """One blind task per campaign region the person does not have yet."""
    d, rec = camp["data"], store.get(camp["data"]["recording_id"])
    have = {(t["data"]["start_ms"], t["data"]["end_ms"]) for t in store.find("task", owner=user["id"])
            if t["data"].get("campaign") == camp["id"]}
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
                    "campaign": camp["id"],
                    "speaker_id": rec["data"].get("speaker_id"),
                    "guideline_version": d["guideline_version"],
                },
                id="tas_" + hashlib.sha256(f"{camp['id']}:{user['id']}:{start}:{end}".encode()).hexdigest()[:24],
                parent=d["recording_id"],
                owner=user["id"],
                status="assigned",
            )
            have.add((start, end))


def campaign_progress(store: Store, slug: str) -> dict:
    """Per-participant progress and pairwise agreement on the regions both finished."""
    from core.evaluation import agreement

    camp = store.get(slug, "campaign")
    people = store.find("user", parent=slug)
    done: dict[str, dict] = {}
    rows = []
    for u in people:
        tasks = [t for t in store.find("task", owner=u["id"])
                 if t["data"].get("campaign") == slug and t["data"].get("recording_id") == camp["data"]["recording_id"]]
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
                "id": u["id"],
                "name": u["data"]["name"],
                "email": u["data"]["email"],
                "status": u["status"],
                "consent": u["data"].get("consent"),  # None means historical evidence is unavailable
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
