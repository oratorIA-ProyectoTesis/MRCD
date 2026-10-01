"""One command to run MRCD: API and worker together, admin created on first start.

    python -m app                      # http://127.0.0.1:8000
    python -m app --host 0.0.0.0 --port 8000 --no-worker

MRCD_ADMIN_TOKEN fixes the first admin's token (useful in Docker); otherwise one is
generated and printed once. The worker runs in its own process so long analyses do
not block HTTP requests.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os

import uvicorn

from app import open_store
from app.ops import new_user
from app.worker import main as worker_main


def bootstrap_admin(store) -> None:
    if store.find("user", status="active"):
        return
    _, token = new_user(store, "admin", "admin", os.environ.get("MRCD_ADMIN_TOKEN") or None)
    shown = "(el definido en MRCD_ADMIN_TOKEN)" if os.environ.get("MRCD_ADMIN_TOKEN") else token
    print(
        f"\n[mrcd] Primera ejecución: se creó la cuenta «admin».\n[mrcd] Token de acceso: {shown}\n"
        "[mrcd] Guárdalo: no se vuelve a mostrar. Más cuentas en la web, Administración > Personas.\n",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m app", description="Levanta la API y el worker de MRCD.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-worker", action="store_true", help="solo la API (el worker corre aparte)")
    a = ap.parse_args()
    store = open_store()
    bootstrap_admin(store)
    if not a.no_worker:  # spawn: a clean process, never a fork of the API's DB connections
        multiprocessing.get_context("spawn").Process(target=worker_main, daemon=True, name="mrcd-worker").start()
    from app.api import create_app

    print(f"[mrcd] Abre http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}", flush=True)
    uvicorn.run(create_app(store), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
