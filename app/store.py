"""One-table document store with revisions and an append-only history.

Every entity (recording, run, task, annotation, case, adjudication, snapshot,
evaluation, model, feedback, user, project) is a JSON document. SQL is portable:
SQLite by default, PostgreSQL through DATABASE_URL.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

IMMUTABLE = {"snapshot", "evaluation"}
COLUMNS = ("parent", "owner", "status")


class Conflict(Exception):
    """Expected revision differs, or the document is immutable/locked."""


class _Retry(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, url: str):
        sqlite = url.startswith("sqlite")
        self.engine = create_engine(url, connect_args={"timeout": 30} if sqlite else {})
        with self.engine.begin() as c:
            if sqlite:  # API and worker processes share the file
                c.execute(text("PRAGMA journal_mode=WAL"))
            c.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS docs (id VARCHAR(80) PRIMARY KEY, kind VARCHAR(32) NOT NULL,"
                    " parent VARCHAR(80), owner VARCHAR(80), status VARCHAR(32), rev INTEGER NOT NULL,"
                    " data TEXT NOT NULL, created VARCHAR(40), updated VARCHAR(40))"
                )
            )
            c.execute(text("CREATE INDEX IF NOT EXISTS docs_kind ON docs (kind, parent, status)"))
            c.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS history (id VARCHAR(80), rev INTEGER, author VARCHAR(80),"
                    " status VARCHAR(32), data TEXT, created VARCHAR(40), PRIMARY KEY (id, rev))"
                )
            )

    @staticmethod
    def _doc(row) -> dict | None:
        return None if row is None else {**row._mapping, "data": json.loads(row.data)}

    def create(self, kind: str, data: dict, *, id: str | None = None, author: str | None = None, **cols) -> dict:
        doc = {
            "id": id or f"{kind[:3]}_{uuid.uuid4().hex[:12]}",
            "kind": kind,
            "rev": 1,
            "created": now(),
            "updated": now(),
            **{k: cols.get(k) for k in COLUMNS},
        }
        with self.engine.begin() as c:
            c.execute(
                text("INSERT INTO docs VALUES (:id, :kind, :parent, :owner, :status, :rev, :data, :created, :updated)"),
                {**doc, "data": json.dumps(data)},
            )
            if author:
                self._log(c, doc["id"], 1, author, doc["status"], data)
        return {**doc, "data": data}

    def get(self, id: str, kind: str | None = None) -> dict | None:
        with self.engine.connect() as c:
            doc = self._doc(c.execute(text("SELECT * FROM docs WHERE id = :id"), {"id": id}).first())
        return doc if doc and (kind is None or doc["kind"] == kind) else None

    def find(self, kind: str, **filters) -> list[dict]:
        where = "".join(f" AND {k} = :{k}" for k in filters if k in COLUMNS)
        with self.engine.connect() as c:
            rows = c.execute(
                text(f"SELECT * FROM docs WHERE kind = :kind{where} ORDER BY created"), {"kind": kind, **filters}
            )
            return [self._doc(r) for r in rows]

    def update(
        self, id: str, data: dict | None = None, *, expected_rev: int | None = None, author: str | None = None, **cols
    ) -> dict:
        """Optimistic concurrency: with `expected_rev` the write succeeds only on that revision;
        without it, a lost race re-reads the row and retries (columns not passed keep the fresh value)."""
        for _ in range(1 if expected_rev is not None else 5):
            try:
                return self._update(id, data, expected_rev, author, cols)
            except _Retry:
                continue
        raise Conflict("escritura concurrente")

    def _update(self, id, data, expected_rev, author, cols) -> dict:
        with self.engine.begin() as c:
            doc = self._doc(c.execute(text("SELECT * FROM docs WHERE id = :id"), {"id": id}).first())
            if doc is None:
                raise KeyError(id)
            if doc["kind"] in IMMUTABLE:
                raise Conflict(f"{doc['kind']} es inmutable")
            if expected_rev is not None and expected_rev != doc["rev"]:
                raise Conflict(f"revisión esperada {expected_rev}, actual {doc['rev']}")
            new = {
                **doc,
                **{k: v for k, v in cols.items() if k in COLUMNS},
                "rev": doc["rev"] + 1,
                "updated": now(),
                "data": doc["data"] if data is None else data,
            }
            hit = c.execute(
                text(
                    "UPDATE docs SET parent = :parent, owner = :owner, status = :status, rev = :rev,"
                    " data = :data, updated = :updated WHERE id = :id AND rev = :old"
                ),
                {**new, "data": json.dumps(new["data"]), "old": doc["rev"]},
            ).rowcount
            if not hit:
                raise _Retry
            if author:
                self._log(c, id, new["rev"], author, new["status"], new["data"])
        return new

    def claim(self, kind: str, status: str, to: str) -> dict | None:
        """Atomically move the oldest document in `status` to `to` (worker queue)."""
        for doc in self.find(kind, status=status):
            with self.engine.begin() as c:
                if c.execute(
                    text(
                        "UPDATE docs SET status = :to, rev = rev + 1, updated = :t WHERE id = :id AND status = :status"
                    ),
                    {"to": to, "t": now(), "id": doc["id"], "status": status},
                ).rowcount:
                    return self.get(doc["id"])
        return None

    def history(self, id: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(text("SELECT * FROM history WHERE id = :id ORDER BY rev"), {"id": id})
            return [{**r._mapping, "data": json.loads(r.data)} for r in rows]

    @staticmethod
    def _log(c, id, rev, author, status, data):
        c.execute(
            text("INSERT INTO history VALUES (:id, :rev, :author, :status, :data, :created)"),
            {"id": id, "rev": rev, "author": author, "status": status, "data": json.dumps(data), "created": now()},
        )
