"""Loop metrics and the experimental chunk decoders of the S2 experiment (synthetic cases, no model)."""

from types import SimpleNamespace

import numpy as np

from mrcd.asr_recovery.noloop import (
    B_LADDER,
    LOOP_CLIP_EXCESS,
    is_loop_clip,
    loop_excess,
    looped_text,
    run_excess,
    transcribe_chunked_branch,
)


def test_run_excess_counts_only_long_runs():
    assert run_excess("no no no".split()) == 0  # 3 repeats: ordinary speech
    assert run_excess("no no no no".split()) == 3
    assert (
        run_excess(("la de " * 10).split()) == 18
    )  # bigram x10 -> 9 extra repeats * 2
    assert run_excess("a b c d e f".split()) == 0


def test_loop_excess_is_relative_to_the_reference():
    hyp = ("no " * 60).split()
    assert loop_excess(hyp, ["no"]) == 59
    # a speaker who really says it 60 times is not a recognizer loop
    assert loop_excess(hyp, hyp) == 0
    assert is_loop_clip(hyp, ["si"]) and not is_loop_clip(hyp, hyp)


def test_loop_clip_threshold_boundary():
    below = (["x"] * (LOOP_CLIP_EXCESS - 1 + 1)) + ["y"]  # run of 20 -> excess 19
    at = (["x"] * (LOOP_CLIP_EXCESS + 1)) + ["y"]  # excess 20
    assert not is_loop_clip(below, [])
    assert is_loop_clip(at, [])


def test_looped_text_detector():
    assert looped_text("no no no no no no")  # 1-gram x6 (>= 5)
    assert looped_text("que se entramo " * 5)  # 3-gram x5 (>= 4)
    assert not looped_text("hola buenos dias quisiera preguntar por el horario")
    assert not looped_text("no no no")  # below the thresholds


class _Seg:
    def __init__(self, words):
        self.words = [
            SimpleNamespace(word=w, start=0.1 * i, end=0.1 * i + 0.05, probability=0.9)
            for i, w in enumerate(words)
        ]


class _FakeModel:
    """Looped on the first call for the chunk, clean on a prompt-free retry (or never clean if ``always_loop``)."""

    def __init__(self, always_loop=False):
        self.calls = []
        self.always_loop = always_loop

    def transcribe(self, audio, **kw):
        self.calls.append(dict(kw))
        looped = self.always_loop or kw.get("initial_prompt", "x") is not None
        words = ["no"] * 12 if looped else ["hola", "mundo"]
        return [_Seg(words)], None


def _asr(model):
    return SimpleNamespace(model=model)


BASE = dict(initial_prompt="eh", temperature=0.0, beam_size=5)


def test_branch_without_ladder_never_redecodes_and_keeps_text():
    m = _FakeModel()
    words, diag = transcribe_chunked_branch(
        _asr(m), np.zeros(16000 * 3, np.float32), BASE, 8.0, 2.0, None
    )
    assert [w.norm for w in words] == ["no"] * 12
    assert len(m.calls) == 1 and diag["redecoded"] == [] and diag["looped_chunks"] == 0


def test_ladder_redecodes_looped_chunk_and_records_it():
    m = _FakeModel()
    words, diag = transcribe_chunked_branch(
        _asr(m), np.zeros(16000 * 3, np.float32), BASE, 8.0, 2.0, B_LADDER
    )
    assert [w.norm for w in words] == ["hola", "mundo"]
    assert (
        len(m.calls) == 2
        and m.calls[1]["initial_prompt"] is None
        and m.calls[1]["temperature"] == 0.0
    )
    assert diag["looped_chunks"] == 1 and diag["unresolved"] == 0
    assert diag["redecoded"] == [
        {"chunk": 0, "attempt": 1, "words_before": 12, "words_after": 2}
    ]


def test_ladder_keeps_original_when_nothing_is_clean_and_says_so():
    m = _FakeModel(always_loop=True)
    words, diag = transcribe_chunked_branch(
        _asr(m), np.zeros(16000 * 3, np.float32), BASE, 8.0, 2.0, B_LADDER
    )
    assert [w.norm for w in words] == [
        "no"
    ] * 12  # original kept, never silently replaced
    assert len(m.calls) == 1 + len(B_LADDER)
    assert m.calls[2]["temperature"] == 0.4 and m.calls[2]["best_of"] == 5
    assert diag["unresolved"] == 1 and diag["redecoded"][0]["kept_original"] is True


def test_chunk_grid_and_overlap_region_match_the_engine_semantics():
    m = _FakeModel()
    m.always_loop = False
    # 20 s -> chunks start at 0, 6, 12, 18 (8 s, 2 s overlap)
    words, diag = transcribe_chunked_branch(
        _asr(m),
        np.zeros(16000 * 20, np.float32),
        dict(BASE, initial_prompt=None),
        8.0,
        2.0,
        None,
    )
    assert diag["chunks"] == 4
    assert all(words[i].start <= words[i + 1].start for i in range(len(words) - 1))
