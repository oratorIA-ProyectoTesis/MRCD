"""Schema, units, bounds, coverage and mapping rules of the corpus importers."""

import dataclasses
import wave
from pathlib import Path

import numpy as np
import pytest

from mrcd.corpora.base import (
    CLASSES,
    Annotation,
    CorpusFormatError,
    Example,
    ForbiddenMappingError,
    InvalidBoundsError,
    SourceFile,
    ms_to_s,
    wav_info,
)
from mrcd.corpora.dime import XlsNotLabelsError, parse_phn, refuse_xls_as_labels
from mrcd.dataset import build_recording_windows

SHA = "a" * 64


def make_wav(path: Path, seconds: float, rate: int = 16000, channels: int = 1) -> Path:
    n = int(round(seconds * rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(n * channels, np.int16).tobytes())
    return path


def ex(annotations=(), coverage=None, **kw):
    cov = {c: "unknown" for c in CLASSES}
    cov.update(coverage or {})
    base = dict(
        corpus="dime",
        recording_id="r1",
        speaker_id=None,
        group_id="g1",
        source_files=(SourceFile("a.wav", SHA),),
        original_text="",
        original_annotations=tuple(annotations),
        temporal_unit="seconds",
        start_s=0.0,
        end_s=2.0,
        source_label=None,
        proposed_mrcd_mapping=None,
        mapping_justification="",
        coverage=cov,
        negative_basis={},
        provenance="human",
        input_view="hola",
        audio_duration_s=2.0,
    )
    base.update(kw)
    return Example(**base)


_LEGIT = {  # class -> (source_kind, source_label) pairs of the positive allow-list
    "revision": ("dime_phn_repair", "modification repair"),
    "prolongation": ("eslora_tag", "alargamiento"),
    "repetition": ("dihana_tag", "r:repeticion"),
}


def ann(s, e, cls="revision", label=None):
    kind, lab = _LEGIT[cls]
    return Annotation(
        kind,
        lab,
        "human",
        start_s=s,
        end_s=e,
        proposed_class=cls,
        mapping_strength="exact",
        justification="why",
    )


# ------------------------------------------------------------------ units
def test_ms_to_s_independent_values():
    assert ms_to_s(537.997803) == 0.537998
    assert ms_to_s(3338.9375) == 3.338938
    assert ms_to_s(1500) == 1.5
    assert ms_to_s(100, ms_per_frame=10.0) == 1.0
    with pytest.raises(InvalidBoundsError):
        ms_to_s(1, 0)


def test_phn_converts_ms_and_honours_frame_size():
    head = "MillisecondsPerFrame: {}\nEND OF HEADER\n500.0 1500.0 un\n\n"
    iv = parse_phn(head.format("1.0"))
    assert (iv[0].start_s, iv[0].end_s) == (0.5, 1.5)
    iv10 = parse_phn(head.format("10.0"))
    assert (iv10[0].start_s, iv10[0].end_s) == (5.0, 15.0)


@pytest.mark.parametrize("body", ["0 0 x\n", "10 5 x\n", "a b x\n", "1 2\n"])
def test_phn_rejects_bad_rows(body):
    with pytest.raises((InvalidBoundsError, CorpusFormatError)):
        parse_phn("MillisecondsPerFrame: 1.0\nEND OF HEADER\n" + body + "\n")


def test_phn_requires_header_and_rejects_dense_overlap():
    with pytest.raises(CorpusFormatError):
        parse_phn("0 10 x\n\n")
    with pytest.raises(InvalidBoundsError):
        parse_phn(
            "MillisecondsPerFrame: 1.0\nEND OF HEADER\n0 100 a\n50 200 b\n", dense=True
        )


# ------------------------------------------------------------------ bounds
def test_valid_timed_example_accepted():
    e = ex([ann(0.5, 1.5)], {"revision": "pos"})
    assert e.events_as_dataset_dicts()[0]["start_ms"] == 500


@pytest.mark.parametrize("s,e", [(1.5, 0.5), (1.0, 1.0), (-0.1, 0.5)])
def test_annotation_order_rejected(s, e):
    with pytest.raises(InvalidBoundsError):
        ann(s, e)


def test_annotation_beyond_audio_rejected():
    with pytest.raises(InvalidBoundsError):
        ex([ann(1.0, 2.5)], {"revision": "pos"})


def test_example_beyond_audio_or_without_duration_rejected():
    with pytest.raises(InvalidBoundsError):
        ex(end_s=2.5)
    with pytest.raises(InvalidBoundsError):
        ex(audio_duration_s=None)


def test_same_class_overlap_rejected_but_other_class_allowed():
    with pytest.raises(InvalidBoundsError):
        ex([ann(0.2, 1.0), ann(0.9, 1.5)], {"revision": "pos"})
    ex(
        [ann(0.2, 1.0), ann(0.9, 1.5, "repetition")],
        {"revision": "pos", "repetition": "pos"},
    )


def test_untimed_example_cannot_carry_times():
    with pytest.raises(InvalidBoundsError):
        ex(temporal_unit="untimed", start_s=0.0)


def test_wav_info_checks_layout(tmp_path):
    assert wav_info(make_wav(tmp_path / "a.wav", 1.25)) == 1.25
    with pytest.raises(CorpusFormatError):
        wav_info(make_wav(tmp_path / "b.wav", 1, rate=8000))
    with pytest.raises(CorpusFormatError):
        wav_info(make_wav(tmp_path / "c.wav", 1, channels=2))


# ------------------------------------------------------------------ coverage
def test_pos_requires_matching_annotation_and_vice_versa():
    with pytest.raises(ValueError):
        ex(coverage={"revision": "pos"})  # pos without any event
    with pytest.raises(ValueError):
        ex([ann(0.1, 0.5)])  # event while coverage says unknown


def test_negative_requires_documented_basis():
    with pytest.raises(ValueError):
        ex(coverage={"block": "neg"})
    e = ex(
        coverage={"block": "neg"},
        negative_basis={"block": "manual declares full coverage"},
    )
    assert e.coverage["block"] == "neg"


def test_coverage_must_list_every_class():
    with pytest.raises(ValueError):
        dataclasses.replace(ex(), coverage={"revision": "unknown"})


def test_invalid_negative_state_and_provenance_rejected():
    with pytest.raises(ValueError):
        ex(coverage={"block": "negative"})
    with pytest.raises(ValueError):
        ex(provenance="gold")


def test_speaker_id_never_invented():
    with pytest.raises(ValueError):
        ex(speaker_id="  ")
    assert ex().speaker_id is None


# ----------------------------------------------------- dataset.py contract kept
def test_partial_coverage_is_rejected_by_dataset_contract():
    e = ex([ann(0.5, 1.5)], {"revision": "pos"})
    declared = e.manifest_human_annotation()
    assert declared["coverage"] == "partial"
    rec = dict(
        recording_id="r1", speaker_id="s", duration_s=2.0, human_annotation=declared
    )
    with pytest.raises(ValueError, match="coverage='complete'"):
        build_recording_windows(rec, Path("."), label_source="human")


def test_complete_only_when_every_class_pos_or_neg_and_human():
    cov = {c: "neg" for c in CLASSES}
    basis = {c: "declared" for c in CLASSES}
    e = ex(coverage=cov, negative_basis=basis)
    assert e.manifest_human_annotation()["coverage"] == "complete"
    assert (
        ex(
            coverage=cov, negative_basis=basis, provenance="heuristic"
        ).manifest_human_annotation()["coverage"]
        == "partial"
    )


# ------------------------------------------------------------------ mappings
@pytest.mark.parametrize(
    "label,cls",
    [
        ("palabra_cortada", "revision"),
        ("palabra_cortada", "block"),
        ("pausa", "block"),
        ("pausa", "rhetorical_pause"),
        ("pausa", "neutral_pause"),
        ("pausa_larga", "rhetorical_pause"),
        ("silencio", "neutral_pause"),
        ("silencio", "block"),
        ("sil", "neutral_pause"),
    ],
)
def test_forbidden_mappings_rejected(label, cls):
    with pytest.raises(ForbiddenMappingError):
        Annotation(
            "eslora_tag",
            label,
            "human",
            proposed_class=cls,
            mapping_strength="approximate",
            justification="because",
        )


def test_mapping_needs_justification_and_known_class():
    with pytest.raises(ForbiddenMappingError):
        Annotation(
            "t",
            "alargamiento",
            "human",
            proposed_class="prolongation",
            mapping_strength="approximate",
            justification=" ",
        )
    with pytest.raises(ForbiddenMappingError):
        Annotation(
            "t",
            "x",
            "human",
            proposed_class="fluent",
            mapping_strength="exact",
            justification="j",
        )


@pytest.mark.parametrize(
    "kind", ["dime_xls_dialogue_act", "dime_xls_cargo_abono", "dime_xls_modalidad"]
)
@pytest.mark.parametrize("label", ["abandonada", "repet-refr", "correccion"])
def test_xls_columns_never_become_labels(kind, label):
    with pytest.raises(ForbiddenMappingError):
        Annotation(
            kind,
            label,
            "human",
            proposed_class="revision",
            mapping_strength="exact",
            justification="looks similar",
        )


def test_xls_reader_always_refuses(tmp_path):
    f = tmp_path / "d12dimedamsl.xls"
    f.write_bytes(b"x")
    with pytest.raises(XlsNotLabelsError):
        refuse_xls_as_labels(f)


# ---- guard hardening (REVISION3 section 1): variants of forbidden labels / kinds ------------------
FORBIDDEN_VARIANTS = [
    ("palabra_cortada", "revision"),
    ("palabra cortada", "revision"),
    ("palabra-cortada", "block"),
    ("palabracortada", "revision"),
    ("PALABRA CORTADA", "revision"),
    ("Palabra Cortada.", "block"),
    ("pausa_corta", "rhetorical_pause"),
    ("Pausa Larga", "block"),
    ("pausa-larga", "neutral_pause"),
    ("Pausa", "block"),
    ("silence", "neutral_pause"),
    ("Silencio", "rhetorical_pause"),
    (".sil", "block"),
    ("<sil>", "neutral_pause"),
    ("<SIL>", "block"),
    ("sp", "neutral_pause"),
    ("fragmento", "revision"),
    ("pausá larga", "block"),  # combining accent
]


@pytest.mark.parametrize("label,cls", FORBIDDEN_VARIANTS)
def test_forbidden_mapping_variants_are_refused_on_annotations(label, cls):
    with pytest.raises(ForbiddenMappingError):
        Annotation("t", label, "human", proposed_class=cls, mapping_strength="exact", justification="x")


@pytest.mark.parametrize("label,cls", FORBIDDEN_VARIANTS)
def test_forbidden_mapping_variants_are_refused_on_the_example_level_fields(label, cls):
    other = ann(0.2, 0.4, cls="repetition")  # a legitimate annotation of another class
    with pytest.raises(ForbiddenMappingError):
        ex([other], source_label=label, proposed_mrcd_mapping=cls, coverage={"repetition": "pos"})


def test_example_mapping_must_exist_among_the_annotations_and_be_a_known_class():
    with pytest.raises(ForbiddenMappingError):
        ex([], proposed_mrcd_mapping="revision")  # no annotation backs it
    with pytest.raises(ForbiddenMappingError):
        ex([ann(0.2, 0.4)], proposed_mrcd_mapping="not_a_class", coverage={"revision": "pos"})
    ok = ex([ann(0.2, 0.4)], source_label="modification repair", proposed_mrcd_mapping="revision",
            coverage={"revision": "pos"})
    assert ok.proposed_mrcd_mapping == "revision"


@pytest.mark.parametrize(
    "kind", ["dime_xls_dialogue_act", "DIME_XLS_dialogue_act", "dime_xls_other", "dime_xls_acto", "xls", "XLS", "DIME-XLS-x"]
)
def test_xls_source_kinds_are_forbidden_whatever_the_case_or_suffix(kind):
    with pytest.raises(ForbiddenMappingError):
        Annotation(kind, "whatever", "human", proposed_class="revision", mapping_strength="exact", justification="x")


def test_legitimate_labels_and_kinds_still_map():
    a = Annotation("dime_phn_repair", "modification repair", "human", proposed_class="revision",
                   mapping_strength="exact", justification="x")
    assert a.proposed_class == "revision"
    b = Annotation("eslora_tag", "alargamiento", "human", proposed_class="prolongation",
                   mapping_strength="exact", justification="x")
    assert b.proposed_class == "prolongation"


# ---- guard v3 (REVISION4 section 2): POSITIVE allow-list, NFKC + confusables, plurals/suffixes, type checks ----
BYPASSES = [
    "pаusa",  # Cyrillic a
    "ｐａｕｓａ",  # full-width
    "pauſa",  # long s
    "pausas",
    "pausa 2",
    "pausa2",
    "silent pausa",
    "fragmentos",
    "pause",
    "Pausas largas",
    "silencios",
    "sil 3",
    "sp1",
]


@pytest.mark.parametrize("label", BYPASSES)
@pytest.mark.parametrize("cls", ["block", "rhetorical_pause", "neutral_pause", "revision"])
def test_reviewer_bypasses_are_refused_on_annotations_and_examples(label, cls):
    with pytest.raises(ForbiddenMappingError):
        Annotation("eslora_tag", label, "human", proposed_class=cls, mapping_strength="exact", justification="x")
    with pytest.raises(ForbiddenMappingError):
        ex([ann(0.2, 0.4)], source_label=label, proposed_mrcd_mapping=cls, coverage={"revision": "pos"})


@pytest.mark.parametrize("bad", [("pausa", "x"), {"l": "pausa"}, ["pausa"], [["pausa"]], b"pausa", 3, None])
def test_non_string_fields_are_refused_cleanly(bad):
    with pytest.raises(ForbiddenMappingError):
        Annotation("eslora_tag", bad, "human", proposed_class="block", mapping_strength="exact", justification="x")
    with pytest.raises(ForbiddenMappingError):
        Annotation(bad, "alargamiento", "human", proposed_class="prolongation", mapping_strength="exact", justification="x")
    with pytest.raises(ForbiddenMappingError):
        Annotation("eslora_tag", "alargamiento", "human", proposed_class="prolongation", mapping_strength="exact",
                   justification=bad)


def test_unlisted_pairs_cannot_carry_a_class_but_can_exist_unmapped():
    for kind, label, cls in [
        ("eslora_tag", "vacilacion", "filler_word"),  # real ESLORA tag, never audited
        ("dime_phn_repair", "abandoned", "revision"),
        ("dihana_tag", "r:repeticion", "revision"),  # listed label, wrong class
        ("eslora_tag", "r:repeticion", "repetition"),  # listed label, wrong source kind
        ("t", "modification repair", "revision"),
    ]:
        with pytest.raises(ForbiddenMappingError):
            Annotation(kind, label, "human", proposed_class=cls, mapping_strength="exact", justification="x")
    for label in ("pausa", "pausas", "pаusa", "vacilacion"):
        a = Annotation("eslora_tag", label, "human")  # unmapped: no class
        assert a.proposed_class is None


@pytest.mark.parametrize(
    "kind,label,cls",
    [(k, lab, c) for k, labs in __import__("mrcd.corpora.base", fromlist=["x"]).ALLOWED_MAPPINGS.items()
     for lab, cs in labs.items() for c in cs],
)
def test_every_allow_listed_pair_maps_and_none_is_in_a_forbidden_family(kind, label, cls):
    from mrcd.corpora.base import CANONICAL_KINDS, CANONICAL_SPELLINGS, forbidden_classes

    assert not forbidden_classes(label)
    # the allow-list keys are canon forms; the stored strings must be the canonical spellings
    a = Annotation(
        CANONICAL_KINDS[kind], CANONICAL_SPELLINGS[kind][label], "human",
        proposed_class=cls, mapping_strength="exact", justification="x",
    )
    assert a.proposed_class == cls
    assert (a.source_kind, a.source_label) == (CANONICAL_KINDS[kind], CANONICAL_SPELLINGS[kind][label])


@pytest.mark.parametrize(
    "label", ["silabeo", "respaldo", "Aspirada (sp)", "sillon", "pasa", "fragil", "alargamiento", "vacilacion", "repeticion"]
)
def test_no_false_positive_of_the_family_check_on_legitimate_labels(label):
    from mrcd.corpora.base import forbidden_classes

    assert forbidden_classes(label) == frozenset()


def test_family_check_flags_plurals_suffixes_and_homoglyphs():
    from mrcd.corpora.base import forbidden_classes

    for label in ("pausas", "pausa 2", "pаusa", "ｐａｕｓａ", "pauſa", "fragmentos", "SP 1"):
        assert forbidden_classes(label), label


# ---- revision 6: homoglyph / full-width / invisible variants of LEGIT labels are refused, not stored ----
@pytest.mark.parametrize(
    "kind,label,cls",
    [
        ("dihana_tag", "f:\u0430", "filler_word"),  # Cyrillic a
        ("dihana_tag", "\uff46:\uff41", "filler_word"),  # full-width
        ("dihana_tag", "f:a\u200b", "filler_word"),  # zero-width space
        ("dihana_tag", "F:\u0410", "filler_word"),  # Cyrillic capital A
        ("dihana_tag", "r:repetici\u0301on", "repetition"),  # NFD accent: not the stored canonical form
        ("eslora_tag", "alargami\u0435nto", "prolongation"),  # Cyrillic e
        ("ESLORA_TAG\u200b", "alargamiento", "prolongation"),
        ("dime_phn_repair", "modification\u00a0repair", "revision"),  # NBSP (NFKC changes it)
    ],
)
def test_homoglyph_variants_of_legit_labels_are_refused_when_mapped(kind, label, cls):
    with pytest.raises(ForbiddenMappingError):
        Annotation(kind, label, "human", proposed_class=cls, mapping_strength="exact", justification="x")


def test_plain_legit_labels_with_accents_are_still_stored_verbatim():
    from mrcd.corpora.base import is_plain_label

    assert is_plain_label("r:repetici\u00f3n") and is_plain_label("dihana_tag") and is_plain_label("modification repair")
    a = Annotation("dihana_tag", "r:repeticion", "human", proposed_class="repetition",
                   mapping_strength="exact", justification="x")
    assert a.source_label == "r:repeticion"


def test_unlisted_pair_raises_and_is_not_downgraded_to_unmapped():
    with pytest.raises(ForbiddenMappingError, match="rejected"):
        Annotation("eslora_tag", "vacilacion", "human", proposed_class="filler_word",
                   mapping_strength="exact", justification="x")


def test_fold_label_strips_invisible_characters_by_itself():
    """REVISION6 mutant N9: the Cf stripping is a redundant layer in canon_label, so pin fold_label directly."""
    from mrcd.corpora.base import fold_label

    assert fold_label("pa\u200busa") == "pausa"
    assert fold_label("pa\u00adusa") == "pausa"
    assert fold_label("pa\ufeffusa") == "pausa"


@pytest.mark.parametrize("label", ["silencios", "silencio largo", "silenciado", "SILENCIO 2"])
def test_silence_family_stem_flags_variants_beyond_the_exact_names(label):
    """REVISION6 mutant N12: only the 'silenc' stem catches these (the exact table has 'silencio' and 'silence')."""
    from mrcd.corpora.base import forbidden_classes

    assert {"block", "rhetorical_pause", "neutral_pause"} <= forbidden_classes(label)


# ----------------------------------------------------------------- REVISION7 F-4: invisible chars and variants
INVISIBLE = ["͏", "️", "︀", " ", " ", " ", "⠀", "​", " ", "⁠"]


@pytest.mark.parametrize("ch", INVISIBLE)
def test_is_plain_label_rejects_invisible_and_format_characters(ch):
    from mrcd.corpora.base import is_plain_label

    assert not is_plain_label("f" + ch + ":a")
    assert not is_plain_label("f:a" + ch)


def test_is_plain_label_still_accepts_real_labels():
    from mrcd.corpora.base import is_plain_label

    for lab in ("f:a", "r:repeticion", "alargamiento", "modification repair", "<sil>", "vacilación", "d:abrir"):
        assert is_plain_label(lab)


@pytest.mark.parametrize("ch", INVISIBLE)
def test_invisible_char_variants_of_allowed_pairs_are_refused_at_annotation_level(ch):
    with pytest.raises(ForbiddenMappingError):
        Annotation("dihana_tag", "f" + ch + ":a", "human", proposed_class="filler_word",
                   mapping_strength="exact", justification="x")


@pytest.mark.parametrize("kind,label", [("dihana_tag", v) for v in ("FA", "f.a", "f:a.", "<f:a>", "F:A", "f: a", " f:a")]
                         + [("DIHANA_TAG", "f:a"), ("dihana-tag", "f:a")])
def test_ascii_variants_of_allowed_pairs_are_refused(kind, label):
    with pytest.raises(ForbiddenMappingError):
        Annotation(kind, label, "human", proposed_class="filler_word", mapping_strength="exact", justification="x")


def test_example_level_label_must_be_the_canonical_spelling_REVISION7_mutant_a():
    """Pins the Example-level check on its own: the Annotation is canonical, the Example label is not.
    Mutation check: removing the plain/canonical test in Example._check_example_mapping makes this fail."""
    ann_ok = Annotation("dime_phn_repair", "modification repair", "human", proposed_class="revision",
                        mapping_strength="exact", justification="x", start_s=0.2, end_s=0.4)
    ok = ex([ann_ok], source_label="modification repair", proposed_mrcd_mapping="revision", coverage={"revision": "pos"})
    assert ok.source_label == "modification repair"
    for bad in ("modification͏ repair", "modification repair⠀", "modification repair", "modification repair"):
        with pytest.raises(ForbiddenMappingError):
            ex([ann_ok], source_label=bad, proposed_mrcd_mapping="revision", coverage={"revision": "pos"})
