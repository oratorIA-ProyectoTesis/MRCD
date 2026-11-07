"""CHM150 importer and transcript-derived reference on SYNTHETIC fixtures (no CHM150 text or audio)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mrcd.context_eval.survival import turn_survives
from mrcd.context_eval.textfeat import rule_rep
from mrcd.context_eval.transcript_reference import (
    lexical_tokens,
    literal_repetition_marks,
    utterance_reference,
)
from mrcd.corpora.base import CLASSES, CorpusFormatError, LeakError, UnsafeArchiveError
from mrcd.corpora.chm150 import (
    import_chm150,
    parse_paths,
    parse_transcription,
    speaker_of,
)

# Invented utterances and ids (format only); none is taken from the corpus.
LINES = {
    "F01ENE0900_0000": "la casa la casa tiene ventanas",
    "F01ENE0900_0001": "un perro grande",
    "M02FEB1000_0000": "mm el tren llega tarde",
}


def _make_corpus(tmp: Path, lines=LINES, paths_extra=(), rate=16000) -> Path:
    root = tmp / "CHM150"
    (root / "data" / "transcriptions").mkdir(parents=True)
    (root / "docs").mkdir()
    paths = []
    for uid in lines:
        spk = uid.split("_")[0]
        date = spk[1:6] + "2020"
        rel = f"/speech/{date}/{spk}/{uid}.flac"
        p = root / "data" / rel.lstrip("/")
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, np.zeros(int(rate * 0.5), np.float32), rate, format="FLAC")
        paths.append(rel)
    (root / "data" / "transcriptions" / "CHM150.transcription").write_text(
        "".join(f"{t} {u}\n" for u, t in lines.items()), encoding="utf-8"
    )
    (root / "docs" / "CHM150.paths").write_text(
        "\n".join([*paths, *paths_extra]) + "\n", encoding="utf-8"
    )
    return root


def test_import_groups_by_speaker_and_keeps_every_class_unknown(tmp_path):
    ex = import_chm150(_make_corpus(tmp_path))
    assert [e.recording_id for e in ex] == list(LINES)
    assert [e.group_id for e in ex] == ["F01ENE0900", "F01ENE0900", "M02FEB1000"]
    e = ex[0]
    assert e.corpus == "chm150" and e.speaker_id == e.group_id
    assert e.coverage == {c: "unknown" for c in CLASSES}
    assert e.original_annotations == () and e.negative_basis == {}
    assert e.input_view == LINES["F01ENE0900_0000"]
    assert e.audio_duration_s == pytest.approx(0.5)
    assert e.extra["sex"] == "F"
    assert len(e.source_files) == 2 and all(len(s.sha256) == 64 for s in e.source_files)
    assert e.manifest_human_annotation()["coverage"] == "partial"


def test_transcription_parser_fails_loud():
    with pytest.raises(CorpusFormatError):
        parse_transcription("hola mundo NOT_AN_ID\n")
    with pytest.raises(CorpusFormatError):
        parse_transcription("F01ENE0900_0000\n")
    with pytest.raises(CorpusFormatError):
        parse_transcription("uno F01ENE0900_0000\ndos F01ENE0900_0000\n")
    with pytest.raises(CorpusFormatError):
        speaker_of("X01ENE0900_0000")


def test_paths_are_validated(tmp_path):
    with pytest.raises(UnsafeArchiveError):
        parse_paths("/speech/../../evil/F01ENE0900_0000.flac\n")
    with pytest.raises(UnsafeArchiveError):
        parse_paths("C:/speech/01ENE2020/F01ENE0900/F01ENE0900_0000.flac\n")
    with pytest.raises(CorpusFormatError):  # file filed under another speaker's folder
        parse_paths("/speech/01ENE2020/F01ENE0901/F01ENE0900_0000.flac\n")


def test_audio_without_transcription_is_refused(tmp_path):
    root = _make_corpus(
        tmp_path, paths_extra=["/speech/03MAR2020/F03MAR1100/F03MAR1100_0000.flac"]
    )
    with pytest.raises(CorpusFormatError):
        import_chm150(root)


def test_wrong_sample_rate_is_refused(tmp_path):
    with pytest.raises(CorpusFormatError):
        import_chm150(_make_corpus(tmp_path, rate=8000))


def test_tag_in_transcription_is_a_leak(tmp_path):
    with pytest.raises(LeakError):
        import_chm150(
            _make_corpus(tmp_path, lines={"F01ENE0900_0000": "un <ruido> gato"})
        )


# ----------------------------------------------------------------------------- reference
@pytest.mark.parametrize(
    "text, expected",
    [
        ("el gato el gato duerme", 1),  # n = 2
        ("yo yo quiero pan", 1),  # n = 1
        ("una mesa azul", 0),
        ("el elefante corre", 0),  # prefix, not an exact repeat
        ("pan em pan", 1),  # filled pause removed first, as in the DIHANA text view
        ("em em una silla", 0),  # repeated filled pause is not a lexical repetition
        ("este este carro", 1),  # 'este' is lexical: never removed
        ("a b c d a b c d", 1),  # n = 4
        (
            "a b c d f a b c d f",
            0,
        ),  # n = 5 is outside the 1-4 range
    ],
)
def test_rep_literal(text, expected):
    assert utterance_reference(text)["rep_literal"] == expected


def test_reference_agrees_with_the_lexical_rule_on_filler_free_text():
    for t in ("el gato el gato duerme", "una mesa azul", "no no no", "sol luna sol"):
        assert utterance_reference(t)["rep_literal"] == rule_rep(t)


def test_fillers_counted_and_conjunction_e_kept():
    toks, fill = lexical_tokens("e vimos casas e islas mm este")
    assert fill == ["e", "mm"]
    assert toks == ["vimos", "casas", "e", "islas", "este"]
    r = utterance_reference("e vimos casas e islas mm este")
    assert r["fillers"] == {"e": 1, "mm": 1} and r["n_este"] == 1


def test_marks_feed_the_dihana_survival_heuristic_unchanged():
    ref = utterance_reference("vi un barco un barco rojo")
    assert ref["rep_literal"] == 1
    assert any(m["span"] == "un barco un barco" for m in ref["rep_marks"])
    assert turn_survives("repetition", ref["rep_marks"], "Vi un barco, un barco rojo.")
    assert not turn_survives("repetition", ref["rep_marks"], "Vi un barco rojo.")


def test_marks_shape():
    m = literal_repetition_marks(["x", "y", "x", "y"])
    assert [(k["n"], k["start_tok"]) for k in m] == [(2, 0)]
    assert m[0]["mt"] == m[0]["ct"] == ["x", "y"]
