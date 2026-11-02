"""In-memory per-session cache of decoded ASR chunks. Never persisted; dropped with the session."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace

import numpy as np


def segment_key(namespace: str, start_sample: int, seg: np.ndarray) -> tuple:
    """Identity of one decode call: ASR configuration, position, length and the exact float32 content."""
    seg = np.ascontiguousarray(seg, dtype=np.float32)
    return (namespace, int(start_sample), int(len(seg)), hashlib.sha256(seg.tobytes()).hexdigest())


class ChunkCache:
    def __init__(self) -> None:
        self._items: dict[tuple, tuple[list, list]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: tuple) -> bool:
        with self._lock:
            return key in self._items

    def put(self, key: tuple, words: list, events: list) -> None:
        with self._lock:
            self._items[key] = ([replace(w) for w in words], [dict(e) for e in events])

    def get(self, key: tuple) -> tuple[list, list] | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            self.hits += 1
            return [replace(w) for w in item[0]], [dict(e) for e in item[1]]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
