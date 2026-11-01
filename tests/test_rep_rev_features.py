"""Reference sub-types, simple rules and context features (text only)."""

import numpy as np
import pytest

from mrcd.context_eval.reference import turn_reference
from mrcd.context_eval.survival import turn_survives
from mrcd.context_eval.textfeat import (
    NAMES,
    featurize,
    rep_features,
    rev_features,
    rule_rep,
    rule_rev,
    tokenize,
)


def T(inner: str) -> str:
    return f"<t> {inner} </t>"


A_TURN = T('el precio <r tipo="repetición"> <m> de </m> <c> de </c> </r> los trenes')
B_TURN = T(
    'obtener <r tipo="repetición"> <m> el <l tipo="cortada" palabra="horario"> horaro </l> </m> <c> el horario </c> </r>'
)
C_TURN = T(
    'desde <r tipo="repetición"> <m> <l tipo="mal_pronunciada" palabra="salga"> saga </l> </m> <c> salga </c> </r> hoy'
)
REV_TURN = T(
    'el viaje <r tipo="sustitución"> <m> que va </m> <s> o sea </s> <c> el tren </c> </r> de hoy'
)
REV_SILENT = T('el horario <r tipo="sustitución"> <m> de </m> <c> del </c> </r> tren')


def test_reference_subtypes_follow_the_audit_definitions():
    a = turn_reference(A_TURN)
    assert a["rep_literal"] == 1 and a["rep_full"] == 1 and a["revision"] == 0
    b = turn_reference(B_TURN)
    assert (
        b["rep_literal"] is None
        and b["rep_full"] == 1
        and b["rep_subtypes"] == ["B_cut_word_restart"]
    )
    c = turn_reference(C_TURN)
    assert c["rep_literal"] is None and c["rep_subtypes"] == [
        "C_mispronunciation_corrected"
    ]
    assert turn_reference(T("hola buenas tardes"))["rep_literal"] == 0


def test_revision_subtypes_and_independence_from_repetition():
    r = turn_reference(REV_TURN)
    assert (
        r["revision"] == 1
        and r["rep_full"] == 0
        and r["rev_subtypes"] == ["R1_explicit_edit_signal"]
    )
    assert turn_reference(REV_SILENT)["rev_subtypes"] == [
        "R4_silent_substitution_insertion_deletion"
    ]
    both = turn_reference(
        T(
            '<r tipo="repetición"><m>no</m><c>no</c></r> <r tipo="inserción"><m>a</m><c>a las</c></r>'
        )
    )
    assert both["rep_literal"] == 1 and both["revision"] == 1


def test_a_mark_survives_only_if_the_asr_keeps_the_duplicate():
    marks = turn_reference(A_TURN)["rep_marks"]
    assert turn_survives("repetition", marks, "el precio de de los trenes")
    assert not turn_survives(
        "repetition", marks, "el precio de los trenes"
    )  # ASR collapsed the repeat
    assert not turn_survives("repetition", marks, "")


def test_rule_rep_matches_the_baseline_rule():
    assert rule_rep("el precio de de los trenes") == 1
    assert rule_rep("para ir a ir a madrid") == 1  # 2-gram
    assert rule_rep("el precio de los trenes") == 0
    assert (
        rule_rep("no, no deseo nada") == 1
    )  # surface repeat: the rule cannot tell emphasis from repetition
    assert rule_rep("") == 0


def test_rule_rev_agreed_triggers():
    assert rule_rev("el tren, perdón, el avión") == 1
    assert rule_rev("quiero decir el martes") == 1
    assert (
        rule_rev("el pre precio del tren") == 1
    )  # cut fragment followed by a longer word
    assert rule_rev("el precio del tren") == 0
    assert rule_rev("no es lo mismo") == 0


def test_tokenizer_keeps_punctuation_as_flags_between_words():
    w, pb, end = tokenize("Sí, sí... puede ser.")
    assert w == ["si", "si", "puede", "ser"]
    assert pb[1] == "," and pb[2] == "..." and end == "."


def test_features_see_context_before_and_after_the_candidate():
    f = rep_features("quiero saber el precio de de los trenes")
    assert f["g0_n1"] == 1 and f["has_cand"] == 1 and f["func_only"] == 1
    assert (
        f["after_content"] == 0 and f["after_func"] == 1
    )  # token after the second copy is "los"
    f2 = rep_features("quiero saber el precio de de trenes")
    assert f2["after_content"] == 1 and f2["after_func"] == 0
    assert f2["prev_func"] == 0  # "precio" precedes the candidate
    assert rep_features("de de")["after_end"] == 1
    assert rep_features("de de")["at_start"] == 1


def test_features_cover_gaps_fragments_fillers_and_punctuation():
    assert rep_features("el precio de el precio")["gap1"] == 1
    assert rep_features("el eh el horario")["filler_between"] == 1
    assert rep_features("el pre precio")["frag0"] == 1
    assert (
        rep_features("no, no")["punct_between"] == 1
        and rep_features("no no")["punct_between"] == 0
    )
    assert rep_features("no no no")["run3"] == 1
    assert rep_features("sí sí")["polarity"] == 1


def test_revision_features():
    f = rev_features("el viaje que va, o sea, el tren")
    assert f["edit_expl"] == 1 and f["edit_corr"] == 0 and f["edit_prev_punct"] == 1
    f = rev_features("el tren perdón el avión")
    assert f["edit_corr"] == 1 and f["edit_prev_func"] == 0 and f["edit_after_len"] > 0
    assert rev_features("a las seis de la tarde sobre las cuatro")["restart_anchor"] > 0
    assert rev_features("hola")["edit_any"] == 0


def test_featurize_shape_and_text_only_invariance():
    for task, names in NAMES.items():
        X = featurize(["el precio de de los trenes", "", "hola"], task)
        assert X.shape == (3, len(names)) and np.isfinite(X).all()
    # same words with different case / accents give identical features (condition A vs B surface differences)
    assert rep_features("Sí, sí puede ser") == rep_features("si, si puede SER")


def test_empty_text_gives_all_zero_candidates():
    f = rep_features("")
    assert f["has_cand"] == 0 and f["log_len"] == 0
    with pytest.raises(KeyError):
        featurize(["x"], "unknown_task")


# ------------------------------------------------------------------ parity with the audit (skipped without local data)
def _pool():
    import pickle
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "results/ejecucion_material4_20261030/experimentos/_cache/pool.pkl"
    if not p.exists():
        pytest.skip("pool cache not available")
    return pickle.loads(p.read_bytes())


def test_subtype_counts_reproduce_the_audit_on_folds_1_to_3_and_fold_5_is_absent():
    import json
    from collections import Counter
    from pathlib import Path

    audit = Path(__file__).resolve().parents[1] / "results/auditoria_etiquetas_20261030/dihana_rep_rev_breakdown_dev.json"
    if not audit.exists():
        pytest.skip("audit output not available")
    ref = json.loads(audit.read_text("utf8"))
    pool = _pool()
    assert {u["fold"] for u in pool} == {1, 2, 3, 4}
    dev = [u for u in pool if u["fold"] <= 3]
    rep = Counter(m["subtype"] for u in dev for m in u["ref"]["rep_marks"])
    rev = Counter(m["subtype"] for u in dev for m in u["ref"]["rev_marks"])
    assert dict(rep) == ref["repetition"]["taxonomy"]
    assert dict(rev) == ref["revision"]["taxonomy"]
    # speakers and dialogues never cross the experiment's outer folds
    from mrcd.context_eval.partition import assert_no_crossing, group_folds

    f = group_folds([u["speaker"] for u in pool], 5, 13)
    assert_no_crossing([u["speaker"] for u in pool], f, "speaker")
    assert_no_crossing([u["dialogue"] for u in pool], f, "dialogue")


def test_mark_survives_threshold_is_80_percent_coverage():
    """Pins the 0.8 coverage threshold (a mutant lowering it to 0 survived the previous suite)."""
    from mrcd.context_eval.survival import mark_survives, seq_cov

    mt = "alfa bravo charlie delta echo"
    ct = "foxtrot golf hotel india juliet"
    mark = {"span": f"{mt} {ct}", "mt": mt.split(), "ct": ct.split(), "subtype": "R1_explicit_edit_signal"}
    at_threshold = "alfa bravo charlie delta foxtrot golf hotel india"  # 4/5, 4/5, 8/10 = exactly 0.8
    below = "alfa bravo charlie foxtrot golf hotel india"  # mt 3/5 = 0.6
    assert seq_cov(at_threshold.split(), mt.split())[0] == pytest.approx(0.8)
    assert mark_survives("revision", mark, at_threshold)
    assert not mark_survives("revision", mark, below)
    only_correction = "foxtrot golf hotel india juliet"  # reparandum absent
    assert not mark_survives("revision", mark, only_correction)
