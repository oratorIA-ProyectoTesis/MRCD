"""Frozen human snapshot -> training/evaluation windows (M4, dataset contract 1.2.0).

Only windows fully inside explicitly reviewed regions are kept, so empty audio that
nobody reviewed never becomes a negative. Windows touching uncertain/not-evaluable
spans are dropped instead of being labelled. Audio-only: kin is zero with has_video=False.

    python scripts/snapshot_windows.py snp_xxx --split train --out data/windows_human_train.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import media, media_dir, open_store  # noqa: E402
from app.systems import prepared  # noqa: E402
from core.constants import KINESIC_DIM, KINESIC_STEPS, LABELS, WINDOW_S  # noqa: E402
from core.contracts import FEATURE_SCHEMA_VERSION, validate_features  # noqa: E402
from core.dataset import labels_from_events, window_starts  # noqa: E402
from core.engine import window_tensors  # noqa: E402


def recording_windows(rec: dict, items: list[dict], whisper: str) -> dict | None:
    audio = media.load(media_dir(rec) / "analysis.wav")
    prep = prepared(rec, audio, {"whisper_size": whisper})
    starts = window_starts(prep.duration_s)
    inside = np.array(
        [any(i["start_ms"] <= s * 1000 and (s + WINDOW_S) * 1000 <= i["end_ms"] for i in items) for s in starts], bool
    )
    unknown = [e for i in items for e in i["events"] if e["decision"] != "event"]
    clean = np.array(
        [not any(e["start_ms"] < (s + WINDOW_S) * 1000 and s * 1000 < e["end_ms"] for e in unknown) for s in starts],
        bool,
    )
    starts = starts[inside & clean]
    if not len(starts):
        return None
    events = [
        {"category": e["label"], "start_ms": e["start_ms"], "end_ms": e["end_ms"]}
        for i in items
        for e in i["events"]
        if e["decision"] == "event"
    ]
    y, multi = labels_from_events(events, starts)
    n = len(starts)
    return {
        **window_tensors(prep.frames, prep.words, starts),
        "y": y,
        "multi": multi,
        "kin": np.zeros((n, KINESIC_STEPS, KINESIC_DIM), np.float32),
        "kmask": np.zeros((n, KINESIC_STEPS), bool),
        "has_video": np.zeros(n, bool),
        "start_ms": (starts * 1000).astype(np.int64),
        "group": np.array([items[0]["speaker_id"]] * n),
        "rec": np.array([rec["id"]] * n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test", "pilot"])
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    store = open_store()
    snap = store.get(a.snapshot, "snapshot")
    by_rec: dict = {}
    for item in snap["data"]["items"]:
        if item["split"] == a.split:
            by_rec.setdefault(item["recording_id"], []).append(item)
    parts = [p for rid, items in by_rec.items() if (p := recording_windows(store.get(rid), items, a.whisper))]
    if not parts:
        raise SystemExit("Sin ventanas dentro de regiones revisadas.")
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    payload = dict(
        data,
        label_source=np.array("human"),
        labels=np.array(LABELS),
        feature_schema=np.array(FEATURE_SCHEMA_VERSION),
        snapshot=np.array(snap["id"]),
        snapshot_sha256=np.array(snap["data"]["sha256"]),
    )
    validate_features(payload, require_labels=True)
    np.savez_compressed(a.out, **payload)
    print(f"{a.out}: {len(data['y'])} ventanas, {len(set(data['group']))} hablantes, snapshot {snap['id']}")


if __name__ == "__main__":
    main()
