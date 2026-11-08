import csv
import hashlib

import pytest

from scripts.annotation import unblind as ub
from scripts.annotation.blind_session import inside_repo, make_blinded

ROWS = [
    {
        "item_id": f"it_{k:04d}",
        "file": f"speaker{k % 3}_es_{k % 3 + 1}.mp3",
        "sha256": "ab" * 32,
        "start_s": f"{10 + k}.5",
        "end_s": f"{18 + k}.5",
        "speaker_if_in_name": f"speaker{k % 3}",
    }
    for k in range(30)
]


def test_blinded_list_leaks_nothing():
    key, listing = make_blinded(ROWS, seed=1)
    blob = " ".join(str(v) for r in listing for v in r.values())
    for r in ROWS:
        assert r["item_id"] not in blob and r["file"] not in blob
        assert r["speaker_if_in_name"] not in blob and r["start_s"] not in blob
    assert sorted(k["orig_item_id"] for k in key) == sorted(r["item_id"] for r in ROWS)
    assert len({k["item_id"] for k in key}) == len(ROWS)
    assert set(listing[0]) == {"position", "blind_id"}


def test_passes_differ_and_are_reproducible():
    k1, _ = make_blinded(ROWS, seed=1)
    k1b, _ = make_blinded(ROWS, seed=1)
    k2, _ = make_blinded(ROWS, seed=2)
    assert k1 == k1b
    assert {k["item_id"] for k in k1}.isdisjoint({k["item_id"] for k in k2})
    assert [k["orig_item_id"] for k in k1] != [k["orig_item_id"] for k in k2]


def test_key_inside_repo_is_detected():
    from scripts.annotation.blind_session import REPO

    assert inside_repo(REPO / "docs" / "key.csv")
    assert not inside_repo(REPO.parent / "outside_key.csv")


def test_unblind_requires_hash_and_matches(tmp_path):
    key, _ = make_blinded(ROWS, seed=3)
    ann = tmp_path / "p1.csv"
    with ann.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "annotator", "class", "state", "start_s", "end_s", "note"])
        w.writerow([key[0]["item_id"], "author", "none", "none", "", "", ""])
    with pytest.raises(SystemExit):
        ub.check_hash(ann)  # no sidecar yet
    side = tmp_path / "p1.csv.sha256"
    side.write_text(hashlib.sha256(ann.read_bytes()).hexdigest() + "  p1.csv\n")
    ub.check_hash(ann)  # ok
    ann.write_text(ann.read_text() + "tampered\n")
    with pytest.raises(SystemExit):
        ub.check_hash(ann)
    rows = ub.unblind([{"item_id": key[0]["item_id"], "annotator": "author"}], key)
    assert rows[0]["item_id"] == key[0]["orig_item_id"]
    with pytest.raises(SystemExit):
        ub.unblind([{"item_id": "zzz", "annotator": "x"}], key)


def test_prev_key_reuse_detected():
    from scripts.annotation.blind_session import check_fresh

    k1, _ = make_blinded(ROWS, seed=1)
    k1b, _ = make_blinded(ROWS, seed=1)
    k2, _ = make_blinded(ROWS, seed=2)
    assert check_fresh(k1b, k1) is not None
    assert check_fresh(k2, k1) is None
    assert "sha256" in k1[0]


def test_draw_subset_seeded_and_bounded():
    from scripts.annotation.draw_subset import draw

    ids = [r["item_id"] for r in ROWS]
    assert draw(ids, 10, 7) == draw(ids, 10, 7)
    assert draw(ids, 10, 7) != draw(ids, 10, 8)
    with pytest.raises(ValueError):
        draw(ids, 99, 1)
