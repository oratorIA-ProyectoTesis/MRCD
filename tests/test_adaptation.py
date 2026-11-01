import numpy as np
import pytest
import torch

from mrcd import adaptation as A
from mrcd.models.fusion import CrossModalFusion


def _tiny_batch(n=5):
    rng = np.random.default_rng(0)
    return {
        "ac": rng.standard_normal((n, 300, 5)).astype(np.float32),
        "pros": rng.standard_normal((n, 9)).astype(np.float32),
        "ling": rng.standard_normal((n, 7, 12)).astype(np.float32),
        "lpos": rng.integers(0, 100, (n, 7)).astype(np.int64),
        "lmask": np.ones((n, 7), bool),
    }


def test_pooled_features_reproduce_the_head():
    torch.manual_seed(0)
    net = CrossModalFusion(
        8, ac_dim=5, ling_dim=12, d=64, mode="audio_text", pros_dim=9
    ).eval()
    d = _tiny_batch()
    probs, pooled = A.predict_pooled(net, {}, d, "cpu", bs=2)
    assert probs.shape == (5, 8) and pooled.shape == (5, 128)
    again = torch.softmax(net.head(torch.as_tensor(pooled)), -1).detach().numpy()
    assert np.allclose(again, probs, atol=1e-6)


def test_unit_max_matches_numpy():
    z = torch.tensor([[1.0, -1.0], [3.0, 0.0], [2.0, 5.0]])
    idx = torch.tensor([0, 0, 1])
    out = A.unit_max(z, idx, 2)
    assert out.tolist() == [[3.0, 0.0], [2.0, 5.0]]


def test_fit_head_learns_a_separable_signal_with_mil():
    rng = np.random.default_rng(1)
    U, per = 120, 3
    Y = (rng.random((U, 1)) < 0.4).astype(np.float32)
    X = rng.standard_normal((U * per, 6)).astype(np.float32)
    uidx = np.repeat(np.arange(U), per)
    for u in range(U):  # the signal sits in ONE window of positive turns
        if Y[u, 0]:
            X[u * per + 1, 0] += 4.0
    snaps = A.fit_head(
        X, uidx, Y, kind="linear", wd=1e-3, lr=5e-2, epochs=150, snapshot_every=50
    )
    s = A.head_scores(snaps[150], X, 1, "linear")
    unit_score = np.array([s[uidx == u, 0].max() for u in range(U)])
    assert A.average_precision(Y[:, 0], unit_score) > 0.9
    assert sorted(snaps) == [50, 100, 150]


def test_average_precision_basic():
    assert A.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(
        (1 + 2 / 3) / 2
    )
    assert np.isnan(A.average_precision([0, 0], [0.1, 0.2]))


def test_save_refuses_overwrite_and_load_checks_base_hash(tmp_path):
    base = tmp_path / "base.pt"
    base.write_bytes(b"base-v1")
    state = A.Head(4, 2).state_dict()
    out = tmp_path / "new.pt"
    h = A.save_adapted(
        out,
        base_ckpt=base,
        classes=["a", "b"],
        state=state,
        kind="linear",
        hidden=32,
        in_dim=4,
        meta={},
    )
    assert len(h) == 64
    with pytest.raises(FileExistsError):
        A.save_adapted(
            out,
            base_ckpt=base,
            classes=["a", "b"],
            state=state,
            kind="linear",
            hidden=32,
            in_dim=4,
            meta={},
        )
    ck, h2 = A.load_adapted(out)
    assert h2 == h and ck["classes"] == ["a", "b"]
    base.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        A.load_adapted(out)
