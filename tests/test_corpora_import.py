"""utt74 repair import (synthetic replica + real files when present) and corpus-level checks."""

import os
from pathlib import Path

import pytest

from mrcd.corpora.base import CLASSES, InvalidBoundsError
from mrcd.corpora.dime import import_dime_sample, import_utt74, list_dialog_ids
from mrcd.corpora.groups import UnresolvedGroupError
from tests.test_corpora_base import make_wav

HDR = "MillisecondsPerFrame: 1.0\nEND OF HEADER\n"
PAL = """0.000000 194.355194 .bn
194.355194 288.011871 y
288.011871 537.997803 ahora
537.997803 641.512939 un
641.512939 909.103455 .sil
909.103455 1042.898926 la
1042.898926 1680.032104 estufa
1680.032104 1812.573486 un
1812.573486 3338.9375 poco
"""
EST = "537.997803 641.512939 reparandum\n909.103455 1812.573486 alteracion\n"
TIPO = '0.000000 3338.937500 " modification repair"\n'


@pytest.fixture
def muestras(tmp_path):
    m = tmp_path / "muestras"
    m.mkdir()
    make_wav(m / "utt74.wav", 3.3389375)
    (m / "utt74_pal.phn").write_text(HDR + PAL, encoding="latin-1")
    (m / "utt74_rep_est.phn").write_text(HDR + EST, encoding="latin-1")
    (m / "utt74_rep_tipo.phn").write_text(HDR + TIPO, encoding="latin-1")
    return m


PARENTS = {"utt74": "d12"}


def test_utt74_replica_maps_to_revision_with_second_units(muestras, tmp_path):
    e = import_utt74(muestras, tmp_path, PARENTS)
    ev = [a for a in e.original_annotations if a.proposed_class]
    assert len(ev) == 1
    r = ev[0]
    # reparandum start .. alteration end, ms converted to s (independent literals)
    assert (r.start_s, r.end_s) == (0.537998, 1.812573)
    assert r.proposed_class == "revision" and r.source_label == "modification repair"
    assert r.locator["reparandum"] == [0.537998, 0.641513]
    assert r.locator["alteration"] == [0.909103, 1.812573]
    assert r.locator["reparandum_words"] == ["un"]
    assert r.locator["alteration_words"] == ["la", "estufa", "un"]
    sil = r.locator["editing_silence_s"]
    assert len(sil) == 1 and round(sil[0][1] - sil[0][0], 3) == 0.268
    assert e.audio_duration_s == 3.338938 and e.end_s == e.audio_duration_s


def test_utt74_replica_coverage_group_and_contract(muestras, tmp_path):
    e = import_utt74(muestras, tmp_path, PARENTS)
    assert e.coverage["revision"] == "pos"
    assert all(e.coverage[c] == "unknown" for c in CLASSES if c != "revision")
    assert e.group_id == "d12"
    assert e.manifest_human_annotation()["coverage"] == "partial"
    assert e.events_as_dataset_dicts() == [
        dict(
            category="revision",
            start_ms=538,
            end_ms=1813,
            confidence=1.0,
            source="human",
        )
    ]
    # the repetition reading is only secondary: it must not be emitted
    assert not any(a.proposed_class == "repetition" for a in e.original_annotations)
    # the silence is kept but is not a pause class
    assert any(
        a.source_label == "sil" and a.proposed_class is None
        for a in e.original_annotations
    )
    assert e.input_view == "y ahora un la estufa un poco"


def test_utt74_requires_audited_parent(muestras, tmp_path):
    with pytest.raises(UnresolvedGroupError):
        import_utt74(muestras, tmp_path, {})


def test_utt74_rejects_alteration_beyond_audio(muestras, tmp_path):
    (muestras / "utt74_rep_est.phn").write_text(
        HDR + "537.99 641.5 reparandum\n909.1 4000.0 alteracion\n", encoding="latin-1"
    )
    with pytest.raises(InvalidBoundsError):
        import_utt74(muestras, tmp_path, PARENTS)


def test_utt74_rejects_reparandum_after_alteration(muestras, tmp_path):
    (muestras / "utt74_rep_est.phn").write_text(
        HDR + "1000.0 1100.0 reparandum\n200.0 800.0 alteracion\n", encoding="latin-1"
    )
    with pytest.raises(InvalidBoundsError):
        import_utt74(muestras, tmp_path, PARENTS)


def test_unaudited_repair_type_is_not_mapped(muestras, tmp_path):
    (muestras / "utt74_rep_tipo.phn").write_text(
        HDR + '0.0 3338.9375 " fresh start"\n', encoding="latin-1"
    )
    e = import_utt74(muestras, tmp_path, PARENTS)
    assert e.proposed_mrcd_mapping is None and set(e.coverage.values()) == {"unknown"}


def test_word_tier_beyond_audio_rejected(muestras, tmp_path):
    make_wav(muestras / "utt74.wav", 1.0)
    with pytest.raises(InvalidBoundsError):
        import_utt74(muestras, tmp_path, PARENTS)


def test_sample_id_is_validated(muestras, tmp_path):
    with pytest.raises(Exception):
        import_dime_sample(
            "../utt74", muestras, tmp_path, PARENTS, word_tier="utt74_pal.phn"
        )


def test_dialog_listing_ignores_incomplete_pairs(tmp_path):
    (tmp_path / "transcripciones_y_actos").mkdir()
    (tmp_path / "audio_wav_16k_mono").mkdir()
    for n in ("d01", "d02"):
        (tmp_path / "transcripciones_y_actos" / f"{n}.txt").write_text("x")
    make_wav(tmp_path / "audio_wav_16k_mono" / "d01.wav", 0.1)
    assert list_dialog_ids(tmp_path) == ["d01"]


# ----------------------------------------------------- real files (skipped if absent)
REPO = Path(__file__).resolve().parents[1]
MATERIAL = REPO / "material" / "MRCD_Material_4_Corpus_20261030"
XCORR = REPO / "results/ejecucion_material4_20261030/audit/xcorr_samples.csv"
real = pytest.mark.skipif(
    not (MATERIAL / "DIME").is_dir() or not XCORR.is_file(),
    reason="material/ not available",
)


@real
def test_real_utt74_matches_audit():
    from mrcd.corpora.groups import load_sample_parents

    parents = load_sample_parents(XCORR)
    assert parents == {"utt1": "d19", "utt74": "d12"}
    e = import_utt74(MATERIAL / "DIME" / "muestras", MATERIAL, parents)
    r = [a for a in e.original_annotations if a.proposed_class][0]
    assert (r.start_s, r.end_s) == (0.537998, 1.812573)
    assert e.input_view == "y ahora un la estufa un poco hacia el fregadero"
    assert e.group_id == "d12"


@real
def test_real_dime_dialogs_are_unlabeled_and_match_audit_counts():
    from mrcd.corpora.dime import import_dime_dialog

    dime = MATERIAL / "DIME"
    ids = list_dialog_ids(dime)
    assert len(ids) == 20
    ex = [import_dime_dialog(i, dime, MATERIAL) for i in ids]
    assert sum(e.extra["utterances_parsed"] for e in ex) == 3283
    assert round(sum(e.audio_duration_s for e in ex) / 60, 3) == 241.618
    assert all(set(e.coverage.values()) == {"unknown"} for e in ex)
    for e in ex:
        e.model_features()


@real
def test_real_eslora_counts_match_audit():
    from mrcd.corpora.eslora import eslora_stats, import_eslora_zip

    ex = import_eslora_zip(MATERIAL / "ESLORA" / "corpus_publico.zip", MATERIAL)
    st = eslora_stats(ex)
    assert st["recordings"] == 130
    assert st["tag_counts"]["alargamiento"] == 32812
    assert st["tag_counts"]["palabra_cortada"] == 7689
    assert st["tag_counts"]["pausa"] == 87972
    for e in ex[:10]:
        e.model_features()
        assert not any(
            a.proposed_class not in (None, "prolongation")
            for a in e.original_annotations
        )


DIHANA = Path(
    (os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>") + r"\01_DIHANA\extraido\dihana")
)
TURNS = REPO / "results/baselines_20261030/dihana/turns.csv"


@pytest.mark.skipif(
    not DIHANA.is_dir() or not TURNS.is_file(), reason="DIHANA not available"
)
def test_dihana_parity_with_baseline_adapter():
    import csv
    from mrcd.corpora.dihana import import_dihana_turns

    rows = {r["turn_id"]: r for r in csv.DictReader(TURNS.open(encoding="utf-8-sig"))}
    xmls = sorted((DIHANA / "data").glob("*/*/*.xml"))[:60]
    n = 0
    for x in xmls:
        for e in import_dihana_turns(x, DIHANA):
            r = rows[e.recording_id]
            assert e.input_view == r["ref_text_marks_stripped"]
            for cls, col in (
                ("filler_word", "filler_present"),
                ("prolongation", "lengthening_present"),
                ("repetition", "repetition_present"),
                ("revision", "reformulation_present"),
            ):
                assert (e.coverage[cls] == "pos") == bool(int(r[col]))
            n += 1
    assert n > 100


# ---- revision 6: every DIHANA tag the importer maps must be in the allow-list (no corpus needed) ----
DIHANA_FIXTURE = """<?xml version="1.0" encoding="ISO-8859-1"?>
<dialogo><hablantes><hablante id="usuario" nombre="spk_x"/></hablantes>
<turno id="1" hablante="usuario"><t>quiero <f tipo="a"/> <f tipo="e"/> <f tipo="m"/> <f tipo="sin_identificar"/> un <a>billete</a> <r tipo="repeticion">el el</r> <r tipo="sustitucion">a</r> <r tipo="insercion">b</r> <r tipo="borrado">c</r> <l palabra="horario">horar</l></t></turno>
</dialogo>
"""


def _dihana_fixture(tmp_path):
    root = tmp_path / "dihana"
    (root / "data" / "a" / "b").mkdir(parents=True)
    x = root / "data" / "a" / "b" / "fx.xml"
    x.write_bytes(DIHANA_FIXTURE.encode("latin-1"))
    return root, x


def test_dihana_fixture_maps_every_class_including_r_borrado(tmp_path):
    from mrcd.corpora.dihana import import_dihana_turns

    root, x = _dihana_fixture(tmp_path)
    (e,) = import_dihana_turns(x, root)
    labels = {a.source_label: a.proposed_class for a in e.original_annotations}
    assert labels["r:borrado"] == "revision"
    assert labels["r:sustitucion"] == labels["r:insercion"] == "revision"
    assert labels["r:repeticion"] == "repetition" and labels["a"] == "prolongation"
    assert {labels[k] for k in ("f:a", "f:e", "f:m", "f:sin_identificar")} == {"filler_word"}
    assert labels["l"] is None


def test_removing_r_borrado_from_the_allow_list_breaks_the_dihana_import(tmp_path, monkeypatch):
    import mrcd.corpora.base as base
    from mrcd.corpora.dihana import import_dihana_turns

    root, x = _dihana_fixture(tmp_path)
    reduced = {k: dict(v) for k, v in base.ALLOWED_MAPPINGS.items()}
    del reduced["dihanatag"]["rborrado"]
    monkeypatch.setattr(base, "ALLOWED_MAPPINGS", reduced)
    with pytest.raises(base.ForbiddenMappingError):
        import_dihana_turns(x, root)
