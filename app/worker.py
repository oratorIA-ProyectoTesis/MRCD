"""Job runner outside the HTTP process: media preparation and analysis runs.

    python -m app.worker

One heavy analysis at a time per process; models load once per process. The
database is the queue, so a crash leaves recoverable state instead of lost jobs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app import media, media_dir, open_store
from app.store import Store

MAX_ATTEMPTS = 3
STALE_S = 900


class Cancelled(Exception):
    pass


@dataclass
class Job:
    run_id: str
    recording: dict
    project: dict
    config: dict
    audio: np.ndarray
    media_dir: Path
    stage: callable


def prepare_recording(store: Store, rec: dict) -> None:
    try:
        d = media_dir(rec)
        info = media.derive(d / rec["data"]["original_name"], d)
        store.update(
            rec["id"],
            {**rec["data"], "media": info, "segments": media.segments(info["analysis"]["duration_ms"])},
            status="ready",
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a failed preparation
        store.update(rec["id"], {**rec["data"], "error": str(exc)[:500]}, status="failed")


def execute(store: Store, run: dict, systems: dict) -> None:
    rec = store.get(run["parent"])
    if rec["status"] in ("queued", "preparing"):
        store.update(run["id"], status="queued")
        return
    current = {"stage": "load_audio"}

    def stage(name: str) -> None:
        doc = store.get(run["id"])
        if doc["status"] in ("cancel_requested", "cancelled"):
            raise Cancelled
        current["stage"] = name
        stages = doc["data"].get("stages", []) + [{"stage": name, "at": time.time()}]
        store.update(run["id"], {**doc["data"], "stage": name, "stages": stages, "heartbeat": time.time()})

    try:
        if rec["status"] != "ready":
            raise ValueError(f"grabación no disponible: {rec['data'].get('error', rec['status'])}")
        stage("load_audio")
        project = store.get(rec["data"].get("project") or "", "project") or {"data": {}}
        job = Job(
            run["id"],
            rec,
            project["data"],
            run["data"].get("config", {}),
            media.load(media_dir(rec) / "analysis.wav"),
            media_dir(rec),
            stage,
        )
        result = systems[run["data"]["system"]](job)
        stage("save")
        ok = [s == "succeeded" for s in result["segments"].values()]
        status = "succeeded" if ok and all(ok) else "partial" if any(ok) else "failed"
        doc = store.get(run["id"])
        store.update(
            run["id"], {**doc["data"], "result": result, "stage": "done", "finished": time.time()}, status=status
        )
        model = result.get("model_version", {})
        if model.get("checkpoint_sha256") and not store.get(f"mdl_{model['checkpoint_sha256'][:16]}"):
            store.create("model", model, id=f"mdl_{model['checkpoint_sha256'][:16]}", status="candidate")
    except Cancelled:
        store.update(run["id"], {**store.get(run["id"])["data"], "stage": current["stage"]}, status="cancelled")
    except Exception as exc:  # noqa: BLE001 - recorded with its stage, not retried blindly
        doc = store.get(run["id"])
        store.update(
            run["id"],
            {
                **doc["data"],
                "error": {"stage": current["stage"], "type": type(exc).__name__, "message": str(exc)[:500]},
            },
            status="failed",
        )


def recover(store: Store) -> None:
    """Requeue work abandoned by a dead worker, with a bounded number of attempts."""
    for rec in store.find("recording", status="preparing"):
        store.update(rec["id"], status="queued")
    for status in ("running", "cancel_requested"):
        for run in store.find("run", status=status):
            if time.time() - run["data"].get("heartbeat", 0) < STALE_S:
                continue
            attempts = run["data"].get("attempts", 0) + 1
            store.update(
                run["id"],
                {**run["data"], "attempts": attempts},
                status="cancelled"
                if status == "cancel_requested"
                else "queued"
                if attempts < MAX_ATTEMPTS
                else "failed",
            )


def work_once(store: Store, systems: dict) -> bool:
    if rec := store.claim("recording", "queued", "preparing"):
        prepare_recording(store, rec)
        return True
    if run := store.claim("run", "queued", "running"):
        try:
            execute(store, run, systems)
        except Exception as exc:  # noqa: BLE001 - one broken job must not stop the queue
            print(f"[worker] {run['id']}: {type(exc).__name__}: {exc}", flush=True)
        return True
    return False


def main() -> None:
    from app.systems import SYSTEMS

    store = open_store()
    recover(store)
    print("[worker] listo", flush=True)
    while True:
        if not work_once(store, SYSTEMS):
            time.sleep(1.0)


if __name__ == "__main__":
    main()
