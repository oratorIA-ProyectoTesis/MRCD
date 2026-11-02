"""Tests for the ambiguity study: occurrences, label/input separation, alignment, features, model."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mrcd.context_ambiguity.align import map_equal, map_span
from mrcd.context_ambiguity.features import build_vocab, text_features
from mrcd.context_ambiguity.model import CtxModel, best_threshold, select_and_fit
from mrcd.context_ambiguity.occurrences import (
    f_marks_between,
    find_occurrences,
    norm,
    tokens_of_turn,
)
from mrcd.context_eval.pool import HistoricalFoldError, assert_no_historical

XML = (
    '<t> <f tipo="e"/> <d tipo="rellenar"> pues </d> quisiera saber <d tipo="aceptar">bueno</d> , '
    'el precio de este tren <d tipo="editar">o sea</d> , a ver <f tipo="m"/> si hay</t>'
)


def _turn(xml: str = XML):
    t = ET.fromstring(xml)
    toks = tokens_of_turn(t)
    return t, toks, find_occurrences(toks, f_marks_between(t))


def test_norm_strips_accents_case_and_punct():
    assert norm("Bueno,") == "bueno"
    assert norm("¿Sí?") == "si"
    assert norm(",") == ""


def test_occurrences_and_reference_types():
    _, toks, occ = _turn()
    got = [(o.form, o.d_type) for o in occ]
    assert got == [
        ("pues", "rellenar"),
        ("bueno", "aceptar"),
        ("este", None),
        ("o sea", "editar"),
        ("a ver", None),
    ]


def test_multiword_form_is_one_occurrence_inside_one_d_element():
    _, toks, occ = _turn()
    o = [x for x in occ if x.form == "o sea"][0]
    assert o.end - o.start == 2 and o.d_type == "editar"


def test_input_text_never_contains_marks_or_types():
    t, toks, _ = _turn()
    text = " ".join(x.text for x in toks)
    for leak in ("rellenar", "aceptar", "editar", "tipo", "<", ">"):
        assert leak not in text
    assert text == " ".join("".join(t.itertext()).split())


def test_f_marks_do_not_enter_text_but_are_located():
    t, toks, occ = _turn()
    pues = occ[0]
    assert pues.f_before == 1  # the filled-pause mark precedes "pues"
    a_ver = [x for x in occ if x.form == "a ver"][0]
    assert a_ver.f_after == 1


def test_occurrence_split_by_two_d_elements_is_not_a_single_mark():
    _, _, occ = _turn('<t><d tipo="rellenar">o</d> <d tipo="editar">sea</d> que</t>')
    assert occ[0].form == "o sea" and occ[0].d_type is None


def test_alignment_recovers_equal_blocks_and_flags_deleted():
    ref = ["pues", "quisiera", "saber", "el", "precio"]
    asr = ["quisiera", "saber", "el", "precio"]
    m = map_equal(ref, asr)
    assert map_span(m, 0, 1) is None  # "pues" deleted by the ASR
    assert map_span(m, 1, 3) == (0, 2)


def test_alignment_rejects_changed_word():
    m = map_equal(["a", "ver", "si"], ["haber", "si"])
    assert map_span(m, 0, 2) is None


def test_features_use_vocab_and_hide_unknown_words():
    w = ["si", "bueno", "quisiera", "xyzzy"]
    f = text_features(w, 1, 2, "bueno", build_vocab([w, w, w]), None)
    assert f["L1=si"] == 1.0 and f["R1=quisiera"] == 1.0
    f2 = text_features(w, 1, 2, "bueno", frozenset({"si"}), None)
    assert "R1=<unk>" in f2 and "R1=quisiera" not in f2
    assert not any(k.startswith("pb=") for k in f)


def test_punctuation_features_only_when_requested():
    f = text_features(["bueno"], 0, 1, "bueno", frozenset(), (",", "."))
    assert f["pb=,"] == 1.0 and f["pa=."] == 1.0


def test_best_threshold_and_degenerate_fold():
    y = np.array([0, 0, 1, 1])
    thr, f1 = best_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert f1 == 1.0 and 0.2 < thr <= 0.8
    m = CtxModel().fit([{"form=a": 1.0}, {"form=b": 1.0}], [1, 1])
    assert m.score([{"form=a": 1.0}]).tolist() == [1.0]


def test_select_and_fit_learns_a_separable_context_and_is_deterministic():
    rng = np.random.default_rng(0)
    feats, y, g = [], [], []
    for i in range(120):
        lab = int(i % 2 == 0)
        feats.append(
            {
                "form=x": 1.0,
                ("R1=a" if lab else "R1=b"): 1.0,
                "noise": float(rng.normal()),
            }
        )
        y.append(lab)
        g.append(f"s{i % 12}")
    m1, i1 = select_and_fit(feats, y, g)
    m2, i2 = select_and_fit(feats, y, g)
    assert i1["C"] == i2["C"] and i1["inner_f1"] > 0.95
    assert (m1.predict(feats) == np.array(y)).mean() > 0.95


def test_historical_fold_guard():
    with pytest.raises(HistoricalFoldError):
        assert_no_historical([1, 2, 5])
    assert_no_historical([1, 2, 3, 4])
