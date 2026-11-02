"""Input views are separated from answers; nothing target-like reaches model features."""

import dataclasses
import xml.etree.ElementTree as ET

import pytest

from mrcd.corpora.base import (
    CLASSES,
    LeakError,
    SourceFile,
    assert_features_clean,
)
from mrcd.corpora.dihana import dihana_input_view, import_dihana_turns
from mrcd.corpora.dime import dime_input_view, import_dime_dialog
from mrcd.corpora.eslora import convert_turn, import_eslora_recording
from tests.test_corpora_base import ex, make_wav

DIME_TXT = (
    "Dialogue: s6-t1-g1\nNumber of utterances files: 3\nLength of dialogue: 2.0\n"
    "Estimated number of turns: 2\n\n"
    "utt1  : s:  mm claro <ruido> <sil> sobre la mesa\n"
    "utt2  : u:  <no-vocal> bue-bueno + okey +\n"
    "utt3  :     ne[ce]sito una una lampara <sil> una lampara ?\n"
)


@pytest.fixture
def dime_dir(tmp_path):
    (tmp_path / "transcripciones_y_actos").mkdir()
    (tmp_path / "audio_wav_16k_mono").mkdir()
    (tmp_path / "transcripciones_y_actos" / "d12.txt").write_bytes(
        DIME_TXT.encode("cp1252")
    )
    make_wav(tmp_path / "audio_wav_16k_mono" / "d12.wav", 2.0)
    return tmp_path


# ------------------------------------------------------------------ DIME
def test_dime_view_keeps_speech_and_drops_answers():
    assert (
        dime_input_view("mm claro <ruido> <sil> sobre la mesa")
        == "mm claro sobre la mesa"
    )
    assert dime_input_view("<no-vocal> bue-bueno + okey +") == "bue bueno okey"
    assert (
        dime_input_view("ne[ce]sito una una lampara <sil> una lampara ?")
        == "necesito una una lampara una lampara ?"
    )


def test_dime_view_does_not_reconstruct_missing_disfluencies():
    # no <sil>, no lengthening mark in the TXT: the view cannot invent them
    assert (
        dime_input_view("y luego el una lampara el casi")
        == "y luego el una lampara el casi"
    )


def test_dime_dialog_is_unlabeled_and_separated(dime_dir):
    e = import_dime_dialog("d12", dime_dir, dime_dir)
    assert set(e.coverage.values()) == {"unknown"}
    assert e.group_id == "d12" and e.speaker_id is None
    assert e.input_view.splitlines() == [
        "mm claro sobre la mesa",
        "bue bueno okey",
        "necesito una una lampara una lampara ?",
    ]
    # answers live only in the separate structure
    labels = {a.source_label for a in e.original_annotations}
    assert {"<sil>", "<ruido>", "<no-vocal>", "hyphen_fragment"} <= labels
    assert all(a.proposed_class is None for a in e.original_annotations)
    assert e.model_features() == {"input_text": e.input_view}
    assert e.manifest_human_annotation()["coverage"] == "partial"


def test_dime_dialog_rejects_unsafe_id_and_wrong_audio(dime_dir):
    with pytest.raises(Exception):
        import_dime_dialog("../d12", dime_dir, dime_dir)
    make_wav(dime_dir / "audio_wav_16k_mono" / "d12.wav", 1.0, rate=8000)
    with pytest.raises(Exception):
        import_dime_dialog("d12", dime_dir, dime_dir)


# ------------------------------------------------------------------ ESLORA
def test_eslora_view_strips_tags_keeps_spoken_words():
    raw = (
        "y además <palabra_cortada>ab</palabra_cortada> <palabra_cortada>a</palabra_cortada> "
        "mayoría <alargamiento>abrumadoras</alargamiento> <pausa/> eeh <pausa_larga/> sí "
        '<ruido tipo="chasquido boca"/> <ficticio>Dan</ficticio>'
    )
    view, anns = convert_turn("hab1", raw, 0)
    assert view == "y además ab a mayoría abrumadoras eeh sí Dan"
    assert [a.source_label for a in anns] == [
        "palabra_cortada",
        "palabra_cortada",
        "alargamiento",
        "pausa",
        "pausa_larga",
    ]
    pro = [a for a in anns if a.proposed_class]
    assert len(pro) == 1 and pro[0].proposed_class == "prolongation"
    assert pro[0].mapping_strength == "approximate" and pro[0].text == "abrumadoras"
    assert all(a.start_s is None for a in anns)  # untimed: no invented times


def test_eslora_fragment_and_pauses_not_mapped():
    _, anns = convert_turn(
        "hab1", "el <palabra_cortada>qu</palabra_cortada> que <pausa/> que", 0
    )
    assert all(a.proposed_class is None for a in anns)


@pytest.mark.parametrize(
    "bad",
    [
        "a <alargamiento>b",
        "a </alargamiento>",
        "a <foo>x</foo>",
        "<alargamiento><alargamiento>x</alargamiento></alargamiento>",
    ],
)
def test_eslora_broken_or_unknown_markup_is_an_error(bad):
    with pytest.raises(Exception):
        convert_turn("hab1", bad, 0)


def _eslora_example(
    content="hab1\tuna <alargamiento>casa</alargamiento> <pausa/> bonita\n",
):
    head = "Nombre fichero de audio\tSCOM_C_99\nhab1\tHablante: SCOM_C_99_hab1\n"
    return import_eslora_recording(
        "SCOM_C_99", head.encode(), content.encode(), "z.zip", "b" * 64
    )


def test_eslora_example_coverage_speaker_and_group():
    e = _eslora_example()
    assert e.temporal_unit == "untimed" and e.group_id == "SCOM_C_99"
    assert e.speaker_id is None  # per-recording ids are not identities
    assert e.coverage["prolongation"] == "pos"
    assert all(e.coverage[c] == "unknown" for c in CLASSES if c != "prolongation")
    assert e.input_view == "una casa bonita"


def test_eslora_without_any_tag_is_all_unknown():
    e = _eslora_example("hab1\tuna casa bonita\n")
    assert set(e.coverage.values()) == {"unknown"}  # absence of a tag is not a negative


def test_eslora_head_must_name_the_recording():
    with pytest.raises(Exception):
        import_eslora_recording(
            "SCOM_C_99",
            b"Nombre fichero de audio\tSCOM_C_01\n",
            b"hab1\tx\n",
            "z.zip",
            "b" * 64,
        )


# ------------------------------------------------------------------ DIHANA
XML = """<?xml version="1.0" encoding="UTF-8"?>
<anotacion><hablantes><hablante id="usuario" nombre="B101"/><hablante id="sistema" nombre="S"/></hablantes>
<turno id="1" hablante="usuario" fichero_audio_ss="x.ul"><t> <f tipo="e"/> <d tipo="abrir"> hola </d> qu<a>e</a> quiero <r tipo="repetición"><m> el </m><c> el </c></r> horario <l tipo="cortada" palabra="gracias"> gracia </l> </t></turno>
<turno id="2" hablante="usuario" fichero_audio_ss="y.ul"><t> No . </t></turno>
<turno id="3" hablante="sistema" fichero_audio_ss="z.ul"><t> sistema </t></turno>
</anotacion>"""


def test_dihana_view_has_no_tags_attributes_or_unwritten_fillers(tmp_path):
    t = ET.fromstring("<t>" + XML.split("<t>")[1].split("</t>")[0] + "</t>")
    v = dihana_input_view(t)
    assert v == "hola que quiero el el horario gracia"
    assert "gracias" not in v.split()  # attribute with the intended word stays hidden
    assert "eh" not in v.split()  # <f/> is empty: the filler is not reconstructed


def test_dihana_turns_labels_and_negatives(tmp_path):
    p = tmp_path / "B101_A.xml"
    p.write_text(XML, encoding="utf-8")
    t1, t2 = import_dihana_turns(p, tmp_path, {"B101": 2})
    assert len(import_dihana_turns(p, tmp_path)) == 2  # system turn excluded
    assert t1.speaker_id == t1.group_id == "B101" and t1.extra["fold"] == 2
    assert [
        t1.coverage[c]
        for c in ("filler_word", "prolongation", "repetition", "revision")
    ] == ["pos", "pos", "pos", "neg"]
    assert (
        t2.coverage["filler_word"] == "neg"
    )  # documented by the manual (adapter convention)
    assert all(
        t2.coverage[c] == "unknown"
        for c in ("block", "rhetorical_pause", "neutral_pause")
    )
    assert t2.input_view == "No ."


# ------------------------------------------------------------------ leakage guards
@pytest.mark.parametrize(
    "corpus,leaky",
    [
        ("dime", "por el <sil> momento"),
        ("dime", "pue-puede"),
        ("dime", "okey + si"),
        ("dime", "ne[ce]sito"),
        ("dime", "s: hola"),
        ("eslora", "casa <alargamiento>x</alargamiento>"),
        ("eslora", "<pausa/> hola"),
        ("eslora", "hab1 hola"),
        ("dihana", '<f tipo="e"/> hola'),
    ],
)
def test_example_construction_fails_when_a_response_token_is_in_input_view(
    corpus, leaky
):
    with pytest.raises(LeakError):
        ex(corpus=corpus, input_view=leaky)


def test_clean_example_passes():
    ex(input_view="una casa bonita")


@pytest.mark.parametrize(
    "key",
    [
        "label",
        "target_class",
        "source_label",
        "coverage",
        "filename",
        "recording_id",
        "group",
        "events",
        "split",
    ],
)
def test_target_like_columns_are_not_features(key):
    with pytest.raises(LeakError):
        assert_features_clean({key: "x"}, ex())


def test_file_and_group_names_cannot_leak_into_text_features():
    e = ex(
        recording_id="d12",
        group_id="d12",
        source_files=(SourceFile("DIME/audio/d12.wav", "a" * 64),),
    )
    with pytest.raises(LeakError):
        assert_features_clean({"input_text": "hola d12 adios"}, e)
    with pytest.raises(LeakError):
        assert_features_clean({"input_text": "ver d12.wav"}, e)
    assert_features_clean({"input_text": "hola adios"}, e)


def test_model_features_expose_only_input_text():
    e = ex(input_view="hola")
    assert set(e.model_features()) == {"input_text"}
    mutated = dataclasses.replace(e, recording_id="hola")
    with pytest.raises(LeakError):
        mutated.model_features()
