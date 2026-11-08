"""Plan B text normalisation and speaker split (scripts/verbatim_ft/vtext.py). Synthetic XML only, no corpus."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "verbatim_ft"))
import vtext as V  # noqa: E402

from mrcd.context_eval.reference import _norm_tokens  # noqa: E402
from mrcd.context_eval.transcript_reference import is_nonlexical_filler  # noqa: E402


def test_fillers_are_reconstructed_and_repetitions_kept():
    x = (
        '<t> <f tipo="e" /> Sí ¿ me podría decir el precio <p /> <r tipo="repetición"> <m> en </m> <c> en </c> </r> '
        "primera clase ? </t>"
    )
    assert (
        V.verbatim_target(x) == "eh Sí ¿me podría decir el precio en en primera clase?"
    )


def test_cut_word_prolongation_noise_and_edit_signal():
    x = (
        '<t> <n fuente="hablante" tipo="labios" /> el día <r tipo="repetición"> <m> el '
        '<l tipo="cortada" palabra="diecinueve"> diez </l> </m> <s> <f tipo="m" /> </s> <c> el diecinueve </c> </r> '
        'd<a>e</a> junio , <f tipo="sin_identificar" /> <N> gracias </N> . </t>'
    )
    assert (
        V.verbatim_target(x) == "el día el diez mm el diecinueve de junio, eh gracias."
    )


@pytest.mark.parametrize(
    "kind,word", [("a", "ah"), ("e", "eh"), ("m", "mm"), ("sin_identificar", "eh")]
)
def test_every_filler_word_is_a_nonlexical_filler_for_the_metrics(kind, word):
    assert V.verbatim_target(f'<t> <f tipo="{kind}" /> no </t>') == f"{word} no"
    assert is_nonlexical_filler(word, "no")


def test_unknown_filler_type_is_rejected():
    with pytest.raises(ValueError):
        V.verbatim_target('<t> <f tipo="x" /> no </t>')


def test_removing_the_fillers_gives_back_the_corpus_input_view():
    x = '<t> <f tipo="e" /> quería <r tipo="repetición"> <m> ir </m> <c> ir </c> </r> a Bilbao <f tipo="a" /> . </t>'
    toks = _norm_tokens(V.verbatim_target(x))
    no_fill = [
        t
        for i, t in enumerate(toks)
        if not is_nonlexical_filler(t, toks[i + 1] if i + 1 < len(toks) else None)
    ]
    assert no_fill == _norm_tokens(V.filler_free(x))


def _speakers(n_pos=40, n_neg=50):
    return {
        **{f"P{i:03d}": True for i in range(n_pos)},
        **{f"N{i:03d}": False for i in range(n_neg)},
    }


def test_split_is_speaker_disjoint_stratified_and_deterministic():
    sp = _speakers()
    a, b = V.make_split(sp), V.make_split(dict(reversed(list(sp.items()))))
    assert a == b
    assert a["train"] | a["dev"] | a["val"] == set(sp)
    assert sum(sp[s] for s in a["val"]) == 8 and sum(not sp[s] for s in a["val"]) == 10
    assert sum(sp[s] for s in a["dev"]) == 3 and sum(not sp[s] for s in a["dev"]) == 4


def test_split_depends_on_the_seed_only_through_the_hash():
    sp = _speakers()
    assert V.make_split(sp, "other")["val"] != V.make_split(sp)["val"]
