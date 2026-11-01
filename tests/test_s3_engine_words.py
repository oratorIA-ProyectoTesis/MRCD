"""analyze(return_words=True) exposes ASR words; the default output stays bit-identical to the golden file."""

import json

import pytest

from tests.test_engine_terminal import (
    GOLDEN,
    WORDS,
    flat,
    make_engine,
    needs_ckpt,
    synth_audio,
)


@needs_ckpt
def test_default_has_no_words_and_matches_golden():
    r = make_engine(WORDS).analyze(synth_audio(), keep_windows=True)
    assert r.words is None
    g = json.loads(GOLDEN.read_text("utf8"))
    got = flat(r)
    assert got["windows"] == g["windows"] and got["events"] == g["events"]
    assert got["transcript"] == g["transcript"]
    assert "words" not in r.to_dict()


@needs_ckpt
def test_return_words_only_adds_the_words():
    eng = make_engine(WORDS)
    a = eng.analyze(synth_audio(), keep_windows=True)
    b = eng.analyze(synth_audio(), keep_windows=True, return_words=True)
    assert flat(a) == flat(b)
    assert [w["text"] for w in b.words] == [w.text for w in WORDS]
    assert b.words[0]["start"] == 1.0 and "norm" in b.words[0]
