"""Group assignment, split crossings, duplicates and archive safety."""

import zipfile

import pytest

from mrcd.corpora.base import UnsafeArchiveError
from mrcd.corpora.eslora import read_zip_members, safe_member_name
from mrcd.corpora.groups import (
    AmbiguousGroupError,
    PartitionLeakError,
    UnresolvedGroupError,
    assert_no_group_crossing,
    find_duplicate_sources,
    find_group_crossings,
    group_for_name,
    load_sample_parents,
)


@pytest.fixture
def parents(tmp_path):
    p = tmp_path / "xcorr.csv"
    p.write_text(
        "sample,dialog,max_ncc,offset_s,dur_s\n"
        "utt74,d01,0.2,1,1\nutt74,d12,0.99,284.2,439\nutt74,d15,0.29,5,1\n"
        "utt1,d19,0.96,0.4,610\nutt1,d06,0.39,0.4,1\n"
        "utt9,d01,0.95,1,1\nutt9,d02,0.93,1,1\n"  # ambiguous: runner-up too close
        "utt8,d01,0.40,1,1\nutt8,d02,0.30,1,1\n",  # too weak
        encoding="utf-8",
    )
    return load_sample_parents(p)


def test_sample_parents_come_from_the_csv_and_ambiguous_ones_are_dropped(parents):
    assert parents == {"utt74": "d12", "utt1": "d19"}


@pytest.mark.parametrize(
    "name,group",
    [
        ("d12.wav", "d12"),
        ("d12.avi", "d12"),
        ("d12.txt", "d12"),
        ("d12dimedamsl.xls", "d12"),
        ("d19_cargosabonos_web.xls", "d19"),
        ("d12_16k_cache.npz", "d12"),
        ("d12_seg003.wav", "d12"),
        ("D12.WAV", "d12"),
        ("utt74.wav", "d12"),
        ("utt74_pal.phn", "d12"),
        ("utt74_rep_tipo.phn", "d12"),
        ("utt1.avi", "d19"),
        ("alofonosutt1.phn", "d19"),
        ("DIME/muestras/utt1.wav", "d19"),
        ("SCOM_C_04_head.txt", "SCOM_C_04"),
        ("SCOM_H11_047_content.txt", "SCOM_H11_047"),
        ("SCOM_C_04_chunk_2.wav", "SCOM_C_04"),
    ],
)
def test_derived_files_inherit_their_parent_group(parents, name, group):
    assert group_for_name(name, parents) == group


def test_unresolvable_names_raise(parents):
    with pytest.raises(UnresolvedGroupError):
        group_for_name("utt99.wav", parents)  # parent not established by the audit
    with pytest.raises(UnresolvedGroupError):
        group_for_name("notes.txt", parents)
    with pytest.raises(UnresolvedGroupError):
        group_for_name("utt74.wav")  # no correspondence table at all


def test_contradictory_name_is_ambiguous(parents):
    with pytest.raises(AmbiguousGroupError):
        group_for_name("d19_utt74.wav", parents)


def test_utt74_and_d12_in_different_splits_is_a_leak(parents):
    files = ["d12.wav", "utt74.wav", "d19.wav", "utt1.avi", "d01.wav"]
    groups = {f: group_for_name(f, parents) for f in files}
    bad = {
        "d12.wav": "train",
        "utt74.wav": "test",
        "d19.wav": "train",
        "utt1.avi": "test",
        "d01.wav": "train",
    }
    crossings = find_group_crossings(bad, groups)
    assert crossings == {"d12": {"train", "test"}, "d19": {"train", "test"}}
    with pytest.raises(PartitionLeakError):
        assert_no_group_crossing(bad, groups)
    good = {
        "d12.wav": "test",
        "utt74.wav": "test",
        "d19.wav": "train",
        "utt1.avi": "train",
        "d01.wav": "train",
    }
    assert_no_group_crossing(good, groups)


def test_avi_wav_derivatives_must_share_the_split(parents):
    names = ["d05.avi", "d05.wav", "d05_cache.npz", "d05_seg1.wav"]
    groups = {n: group_for_name(n, parents) for n in names}
    split = {n: "dev" for n in names}
    split["d05_seg1.wav"] = "test"
    with pytest.raises(PartitionLeakError):
        assert_no_group_crossing(split, groups)


def test_item_without_group_is_an_error():
    with pytest.raises(UnresolvedGroupError):
        find_group_crossings({"x": "train"}, {})


def test_duplicate_sources_are_reported():
    dups = find_duplicate_sources(
        [("a", "g1", "h1"), ("b", "g2", "h1"), ("c", "g3", "h3"), ("a", "g1", "h1")]
    )
    assert dups == [("a", "b")]
    assert find_duplicate_sources([("a", "g", "h"), ("a", "g", "h")]) == []


# ------------------------------------------------------------------ archive safety
@pytest.mark.parametrize(
    "name",
    [
        "../evil.txt",
        "/abs.txt",
        "C:/x.txt",
        "a/../../b.txt",
        "a\\b.txt",
        "",
        "ok/../..",
        "C:evil.txt",
    ],
)
def test_unsafe_member_names_rejected(name):
    with pytest.raises(UnsafeArchiveError):
        safe_member_name(name)


def test_safe_names_accepted():
    for n in ("SCOM_C_01_head.txt", "dir/sub/file.txt", "AUTHORS.txt"):
        assert safe_member_name(n) == n


def test_zip_with_traversal_is_refused_and_nothing_is_written(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("good.txt", "x")
        f.writestr("../escaped.txt", "pwned")
    with pytest.raises(UnsafeArchiveError):
        read_zip_members(z)
    assert not (tmp_path.parent / "escaped.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.zip"]


def test_zip_member_size_cap(tmp_path, monkeypatch):
    import mrcd.corpora.eslora as es

    z = tmp_path / "big.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("a.txt", "x" * 100)
    monkeypatch.setattr(es, "MAX_MEMBER_BYTES", 10)
    with pytest.raises(UnsafeArchiveError):
        es.read_zip_members(z)
