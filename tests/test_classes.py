"""Un test por clase crítica (escenas sintéticas con verdad conocida)."""
from core.annotation.auto_labeler import label_recording
from core.extractors.acoustic import extract_acoustic
from tests.synth import build

W = lambda t, d=0.35, a=140, b=120: (t, d, a, b)   # palabra
P = lambda d: (None, d, 0, 0)                      # pausa


def run(plan, **kw):
    audio, words, kin = build(plan, **kw.pop("build_kw", {}))
    ac = extract_acoustic(audio, use_silero=False)
    return label_recording(ac, words, kin, **kw)


def cats(cands, source="auto"):
    return [c.category for c in cands if c.source == source]


def test_prolongation_vocalic():
    c = run([W("hoy"), W("es"), ("muuuy", 0.8, 150, 150), W("importante,"), W("claro.")])
    p = [x for x in c if x.category == "prolongation"]
    assert p and p[0].evidence["kind"] == "vocalic" and p[0].end - p[0].start >= 0.35


def test_prolongation_fricative():
    c = run([W("hoy"), W("es"), ("sssí", 0.55, "fric", 0.2), W("importante,"), W("claro.")])
    p = [x for x in c if x.category == "prolongation"]
    assert p and p[0].evidence["kind"] == "fricative"


def test_no_prolongation_at_sentence_end():
    c = run([W("hoy"), W("es"), ("muuuy.", 0.8, 150, 150), P(0.3), W("bien")])
    assert "prolongation" not in cats(c)


def test_repetition_identical_and_partial():
    c = run([W("y"), W("pero"), W("pero"), W("luego"), ("pe-", 0.15, 140, 140), W("pensamos"), W("bien.")])
    reps = [x for x in c if x.category == "repetition"]
    assert len(reps) == 2 and any(r.evidence["partial"] for r in reps)


def test_revision_needs_incomplete_clause():
    c = run([W("fuimos"), W("a"), W("la"), W("o", 0.15), W("sea", 0.2), W("al"), W("cine."), P(0.3),
             W("Terminó."), W("es", 0.15), W("decir", 0.25), W("todo"), W("bien.")])
    revs = [x for x in c if x.category == "revision"]
    assert len(revs) == 1 and revs[0].evidence["edit_term"] == "o sea"


def test_block_with_tension():
    c = run([W("vamos"), W("a"), W("ver"), W("la", 0.2), P(0.9), W("casa"), W("nueva.")],
            build_kw=dict(tense_after="la"))
    b = [x for x in c if x.category == "block"]
    assert b and b[0].confidence >= 0.8 and b[0].evidence["prev_word"] == "la"


def test_rhetorical_pause_relaxed_boundary_falling_tone():
    c = run([W("esto"), W("cambia"), ("todo.", 0.45, 130, 90), P(1.3), W("Ahora"), W("bien.")])
    r = [x for x in c if x.category == "rhetorical_pause"]
    assert r and r[0].confidence >= 0.8


def test_boundary_pause_with_tense_face_is_not_rhetorical():
    c = run([W("esto"), W("cambia"), ("todo.", 0.45, 130, 90), P(1.3), W("Ahora"), W("bien.")],
            build_kw=dict(relaxed=False, tense_after="todo."))
    assert "rhetorical_pause" not in cats(c)


def test_recall_mode_only_low_conf_source():
    plan = [W("vamos"), W("a"), W("la", 0.2), P(0.45), W("casa"), W("nueva.")]
    assert not [x for x in run(plan) if x.category == "block"]
    low = [x for x in run(plan, recall_mode=True) if x.source == "auto_low_conf"]
    assert low and all(0.35 <= x.confidence <= 0.5 for x in low)


def test_no_prolongation_when_asr_span_includes_silence():
    """Timestamp largo del ASR sobre una palabra corta seguida de pausa: NO es alargamiento."""
    from core.extractors.linguistic import Word
    from tests.synth import build
    audio, words, kin = build([W("vamos"), W("a"), W("ver"), ("en", 0.25, 140, 128), P(0.7), W("casa"), W("nueva.")])
    ac = extract_acoustic(audio, use_silero=False)
    w = next(x for x in words if x.norm == "en")
    stretched = [Word(x.text, x.norm, x.start, x.end + (0.7 if x is w else 0), x.prob, x.punct_after) for x in words]
    cands = label_recording(ac, stretched, kin)
    assert not [c for c in cands if c.category == "prolongation" and c.evidence.get("word") == "en"]
