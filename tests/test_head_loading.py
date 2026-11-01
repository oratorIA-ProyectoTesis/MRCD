"""load_adapted must never execute a pickle and must hash the bytes it deserializes."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from mrcd import adaptation as A

REPO = Path(__file__).resolve().parents[1]
VC = REPO / "results" / "mejoras_20261030" / "checkpoints" / "ckpt_head_dihana_f123_zero.pt"


class Evil:
    marker: Path

    def __reduce__(self):
        return (Path(Evil.marker).write_text, ("executed",))


@pytest.mark.parametrize("with_hash", [False, True])
def test_malicious_pickle_is_not_executed(tmp_path, with_hash):
    Evil.marker = tmp_path / "pwned.txt"
    bad = tmp_path / "evil.pt"
    with open(bad, "wb") as f:
        pickle.dump(Evil(), f)
    digest = A.sha256_file(bad)
    with pytest.raises(Exception):
        A.load_adapted(bad, expected_sha256=digest if with_hash else None)
    assert not Evil.marker.exists()


def test_load_reads_the_file_once(tmp_path, monkeypatch):
    """The hashed bytes are the deserialized bytes: no second open of the path."""
    ck = {"format": "mrcd-adapted-head-v1", "base_ckpt_path": str(tmp_path / "none.pt"), "base_ckpt_sha256": "0", "classes": ["a"]}
    p = tmp_path / "h.pt"
    torch.save(ck, p)
    digest = A.sha256_file(p)
    opened = []
    real = torch.load

    def spy(f, *a, **k):
        opened.append(type(f).__name__)
        return real(f, *a, **k)

    monkeypatch.setattr(torch, "load", spy)
    A.load_adapted(p, expected_sha256=digest)
    assert opened == ["BytesIO"]


@pytest.mark.skipif(not VC.exists(), reason="Vc checkpoint not present")
def test_real_vc_checkpoint_loads_with_weights_only():
    ck, digest = A.load_adapted(VC, expected_sha256=A.VC_SHA256)
    assert digest == A.VC_SHA256 and ck["classes"] == ["filler_word", "prolongation", "repetition"]
