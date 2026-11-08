import pytest

from scripts.annotation.labels_to_csv import convert


def setup(tmp_path, labels: dict[str, str]):
    clips, lab = tmp_path / "clips", tmp_path / "labels"
    clips.mkdir(parents=True)
    lab.mkdir(parents=True)
    for cid, text in labels.items():
        (clips / f"{cid}.wav").write_bytes(b"")
        (lab / f"{cid}.txt").write_bytes(text.encode("utf-8"))
    return clips, lab


def test_bom_and_frequency_lines_are_handled(tmp_path):
    txt = "\ufeff1.0\t1.5\tFP d\n\\t100.0\t200.0\n2.0\t3.0\tREP i borde\n"
    clips, lab = setup(tmp_path, {"c1": txt})
    rows, orphans = convert(clips, lab, "me")
    assert [r[2] for r in rows] == ["filled_pause", "repetition"]
    assert rows[1][3] == "uncertain" and rows[1][6] == "borde" and not orphans


def test_empty_file_means_none(tmp_path):
    clips, lab = setup(tmp_path, {"c1": ""})
    rows, _ = convert(clips, lab, "me")
    assert rows == [["c1", "me", "none", "none", "", "", ""]]


def test_short_line_and_empty_text_report_clip_and_line(tmp_path):
    for bad in ("1.0\t2.0\n", "1.0\t2.0\t \n", "1.0\n"):
        clips, lab = setup(tmp_path / str(abs(hash(bad))), {"cx": "1.0\t1.2\tFP d\n" + bad})
        with pytest.raises(ValueError, match=r"cx line 2"):
            convert(clips, lab, "me")


def test_point_label_rejected(tmp_path):
    clips, lab = setup(tmp_path, {"c1": "1.0\t1.0\tFP d\n"})
    with pytest.raises(ValueError, match=r"c1 line 1.*end <= start"):
        convert(clips, lab, "me")


def test_orphan_label_file_warned_and_missing_file_error(tmp_path):
    clips, lab = setup(tmp_path, {"c1": ""})
    (lab / "ghost.txt").write_text("")
    _, orphans = convert(clips, lab, "me")
    assert orphans == ["ghost"]
    (clips / "c2.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="missing label file for c2"):
        convert(clips, lab, "me")


def test_bad_state_names_clip(tmp_path):
    clips, lab = setup(tmp_path, {"c1": "1.0\t2.0\tREP x\n"})
    with pytest.raises(ValueError, match="c1"):
        convert(clips, lab, "me")
