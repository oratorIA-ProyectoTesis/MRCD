"""Plan A (o2026_12): repetition candidates + demotion-only contextual layer, and the filler subclass split."""

from mrcd.service import contextual as CX


def w(text, start, end, prob=0.9, punct=""):
    from mrcd.extractors.linguistic import normalize

    return {
        "text": text,
        "norm": normalize(text),
        "start": start,
        "end": end,
        "prob": prob,
        "punct_after": punct,
    }


def seq(tokens, t0=0.0, dur=0.3, gap=0.05):
    out, t = [], t0
    for tok in tokens:
        out.append(w(tok, t, t + dur))
        t += dur + gap
    return out


# ------------------------------------------------------------------------------------------- generator parity
def test_generator_is_the_confirmatory_rule():
    # filled pauses are removed before matching ("el eh el" is a repetition), interval = second copy
    words = seq(["quería", "el", "eh", "el", "horario"])
    c = CX.classify_repetitions(words)
    assert [(x["pos"], x["n"], x["tokens"]) for x in c] == [(2, 1, ["el"])]
    assert c[0]["start_s"] == words[3]["start"] and c[0]["state"] == CX.DISFLUENT
    # 2-token repeat; accents and case are normalised like norm_tokens
    c = CX.classify_repetitions(seq(["Por", "él", "por", "el", "tren"]))
    assert [(x["n"], x["tokens"]) for x in c] == [(2, ["por", "el"])]
    # conjunction "e" before i-/hi- is not a filler and stays a token
    assert CX.lexical_stream(seq(["e", "hijos"]))[0][0] == "e"
    assert CX.classify_repetitions(seq(["la", "casa", "grande"])) == []


def test_layer_never_creates_or_moves_candidates():
    words = seq(["si", "si", "quería", "quería", "ir", "ir"])
    full = CX.classify_repetitions(words)
    none = CX.classify_repetitions(words, rules=())
    assert [(c["pos"], c["n"], c["start_s"], c["end_s"]) for c in full] == [
        (c["pos"], c["n"], c["start_s"], c["end_s"]) for c in none
    ]
    assert all(c["state"] == CX.DISFLUENT for c in none)
    assert {c["state"] for c in full} <= set(CX.CONTEXT_STATES)


# ------------------------------------------------------------------------------------------- demotion rules
def test_r4_polarity_reduplication_is_functional():
    c = CX.classify_repetitions(seq(["si", "si", "a", "madrid"]))
    assert c[0]["rule"] == "polarity" and c[0]["state"] == CX.FUNCTIONAL
    # not a polarity item, or n > 1: stays disfluent
    assert (
        CX.classify_repetitions(seq(["el", "el", "tren"]))[0]["state"] == CX.DISFLUENT
    )
    c = CX.classify_repetitions(seq(["si", "es", "si", "es"]))
    assert all(x["state"] == CX.DISFLUENT for x in c)


def test_r2_gap_over_engine_repetition_gap_is_uncertain():
    words = [w("tren", 0.0, 0.3), w("tren", 2.0, 2.3)]  # 1.7 s of silence > 1.5 s
    c = CX.classify_repetitions(words)
    assert c[0]["rule"] == "gap" and c[0]["state"] == CX.UNCERTAIN
    words = [w("tren", 0.0, 0.3), w("tren", 1.7, 2.0)]  # 1.4 s: kept
    assert CX.classify_repetitions(words)[0]["state"] == CX.DISFLUENT


def test_r3_seam_only_when_copies_come_from_different_chunks():
    assert CX.chunk_seams(7.9) == []
    assert CX.chunk_seams(20.0) == [8.0, 14.0, 20.0]  # the 2 s tail chunk is decoded too
    words = [w("tren", 7.6, 7.9), w("tren", 8.0, 8.3)]  # midpoints 7.75 < 8.0 <= 8.15
    assert CX.classify_repetitions(words, duration_s=20.0)[0]["rule"] == "seam"
    assert CX.classify_repetitions(words, duration_s=None)[0]["state"] == CX.DISFLUENT
    words = [w("tren", 8.1, 8.3), w("tren", 8.4, 8.6)]  # both after the seam
    assert CX.classify_repetitions(words, duration_s=20.0)[0]["state"] == CX.DISFLUENT


def test_r1_loop_wins_over_later_rules():
    c = CX.classify_repetitions(seq(["si"] * 6))
    assert c and all(x["rule"] == "loop" and x["state"] == CX.UNCERTAIN for x in c)
    one = [x for x in c if x["n"] == 1]
    assert one and all("polarity" in x["matched_rules"] for x in one)  # logged; the first rule sets the state
    # a genuine short repetition is not a loop
    assert (
        CX.classify_repetitions(seq(["no", "no", "no", "quiero"]))[0]["rule"]
        == "polarity"
    )


def test_merge_and_stats():
    c = CX.classify_repetitions(seq(["el", "el", "el", "tren", "si", "si"]))
    ev = CX.merge_disfluent(c)
    assert (
        len(ev) == 1
    )  # the two overlapping "el" candidates become one event; "si si" is functional
    st = CX.demotion_stats(c)
    assert st["n_candidates"] == 3 and st["by_state"] == {
        "disfluent": 2,
        "functional": 1,
        "uncertain": 0,
    }
    assert st["demoted_by_rule"]["polarity"] == 1


# ------------------------------------------------------------------------------------------- filler subclasses
def test_lexical_lexicon_is_the_existing_one():
    from mrcd.constants import FILLER_LEXICON

    assert CX.LEXICAL_FILLERS <= FILLER_LEXICON["es"]
    assert CX.LEXICAL_FILLERS == {
        "este",
        "esteee",
        "o sea",
        "bueno",
        "tipo",
        "pues",
        "a ver",
        "digamos",
    }
    assert not any(CX.NONLEXICAL_FILLER_RE.match(x) for x in CX.LEXICAL_FILLERS)
    assert {"mira", "mire", "entonces"}.isdisjoint(
        CX.LEXICAL_FILLERS
    )  # not extended (protocol section 2)


def test_lexical_filler_needs_hesitation_evidence():
    # "pues" glued to its neighbours: no gap, short, no comma -> uncertain
    words = [w("quería", 0.0, 0.40), w("pues", 0.42, 0.60), w("ir", 0.62, 0.80)]
    c = CX.classify_lexical_fillers(words)
    assert [(x["token"], x["state"]) for x in c] == [("pues", CX.UNCERTAIN)]
    # same item followed by a 0.2 s gap -> disfluent
    words[2] = w("ir", 0.80, 1.0)
    assert CX.classify_lexical_fillers(words)[0]["state"] == CX.DISFLUENT
    # comma after the item -> disfluent
    words = [
        w("quería", 0.0, 0.40),
        w("bueno,", 0.42, 0.60, punct=","),
        w("ir", 0.62, 0.80),
    ]
    assert CX.classify_lexical_fillers(words)[0]["state"] == CX.DISFLUENT
    # two-word item, turn-initial (missing neighbour = 1.0 s gap, as in the auto-labeler)
    words = [w("o", 0.0, 0.1), w("sea", 0.12, 0.3), w("mañana", 0.32, 0.7)]
    c = CX.classify_lexical_fillers(words)
    assert [(x["token"], x["state"]) for x in c] == [("o sea", CX.DISFLUENT)]


def test_non_lexical_tokens_are_not_lexical_fillers():
    assert CX.classify_lexical_fillers(seq(["eh", "mmm", "quería"])) == []
    st = CX.lexical_stats(CX.classify_lexical_fillers(seq(["pues", "bueno"])))
    assert st["n_candidates"] == 2
