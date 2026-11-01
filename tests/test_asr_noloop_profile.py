"""Opt-in loop-safe ASR profiles (p0_noloop, p1_noloop): registered config = used config, text never replaced without a record."""

import json
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mrcd.asr_profiles import (
    LOOP_REDECODE_LADDER,
    asr_cache_key,
    chunk_looped,
    get_profile,
    loop_redecode_warning,
)
from mrcd.extractors.linguistic import VerbatimASR


# ------------------------------------------------------------------ defaults and fingerprints are untouched
PINNED = {  # describe() sha256 and asr_cache_key, measured BEFORE the S2 change
    "p0": (
        "0f0c4f9bf6fb673c16f4087c27fdea661ce8f6eb3efbbe2df67e68d0586f962c",
        "edfba15b65a4d173405dd46c5bb59c20912c70233fe64b2aa25aec181ada5d8a",
    ),
    "p1_disfluent_prompt": (
        "1b18c943a67cd817e3b0b872a1a46dd81be557ec8bf0e86da628d4a93c00a2ee",
        "ba122431db833a3dba9f9aa1ff6370271f2b3112f65090d8ced76175c4041460",
    ),
    "p0_cuda": (
        "33cee0b6229b18be419828502e7333d5c040d06fd25d9c41c2610d50bcaa1f17",
        "24acb7d5c715752ca86e025656591b9bab0f9d1acdad9f14dca22cb88bdf2d88",
    ),
    "p1_cuda": (
        "e3f0c436783bdcf1dbcb334b1b21dd7fc16bf628a44c89d1d096331ac1914fba",
        "b37ad9a179aa9323e8164507942493165704ede3c1b925521c669a9963fa0581",
    ),
}


def _key(name):
    return asr_cache_key(
        name,
        audio_sha256="a" * 64,
        model_bin_sha256="b" * 64,
        compute_type="int8",
        device="cpu",
        omp_num_threads="4",
    )


@pytest.mark.parametrize("name", sorted(PINNED))
def test_existing_profiles_keep_their_description_and_cache_key(name):
    d = hashlib.sha256(
        json.dumps(get_profile(name).describe(), sort_keys=True).encode()
    ).hexdigest()
    assert (d, _key(name)) == PINNED[name]
    assert "loop_redecode" not in get_profile(name).describe()
    assert get_profile(name).loop_redecode is False


def test_default_is_still_p0():
    assert get_profile(None).name == "p0" and not get_profile(None).experimental


# ------------------------------------------------------------------ the new profiles
@pytest.mark.parametrize(
    "name,base", [("p0_noloop", "p0"), ("p1_noloop", "p1_disfluent_prompt")]
)
def test_noloop_profile_differs_from_its_base_only_in_name_flags_and_loop_option(
    name, base
):
    p, b = get_profile(name), get_profile(base)
    assert p.experimental and p.loop_redecode and "experimental" in p.validation
    assert p.decode_kwargs("es") == b.decode_kwargs(
        "es"
    )  # first-pass decoding is exactly the base profile's
    assert (p.chunk_s, p.overlap_s, p.initial_prompt, p.device) == (
        b.chunk_s,
        b.overlap_s,
        b.initial_prompt,
        b.device,
    )
    assert (
        replace(
            p,
            name=b.name,
            experimental=b.experimental,
            validation=b.validation,
            description=b.description,
            loop_redecode=False,
        )
        == b
    )


@pytest.mark.parametrize(
    "name,base", [("p0_noloop", "p0"), ("p1_noloop", "p1_disfluent_prompt")]
)
def test_noloop_fingerprint_and_cache_key_are_distinct_and_carry_the_ladder(name, base):
    assert _key(name) != _key(base)
    assert get_profile(name).cache_namespace() != get_profile(base).cache_namespace()
    d = get_profile(name).describe()
    assert d["loop_redecode"]["enabled"] is True
    assert d["loop_redecode"]["ladder"] == [dict(a) for a in LOOP_REDECODE_LADDER]


def test_custom_instance_cannot_pose_as_a_noloop_profile():
    fake = replace(get_profile("p0_noloop"), loop_redecode=False)
    with pytest.raises(ValueError):
        get_profile(fake)


# ------------------------------------------------------------------ detector with synthetic cases
def test_chunk_looped_synthetic_cases():
    assert chunk_looped("no no no no no no no")
    assert chunk_looped("la de la de la de la de la de")
    assert chunk_looped("y que se entramo " * 6)
    assert not chunk_looped("")
    assert not chunk_looped("no no no")  # ordinary emphatic repetition
    assert not chunk_looped("eh yo quería, quería preguntar por el, por el horario")


# ------------------------------------------------------------------ behaviour with a fake model
class _W:
    def __init__(self, w, i):
        self.word, self.start, self.end, self.probability = (
            w,
            0.1 * i,
            0.1 * i + 0.05,
            0.9,
        )


class _Seg:
    def __init__(self, words):
        self.words = [_W(w, i) for i, w in enumerate(words)]


class FakeModel:
    """Loops while an initial prompt is present (or always, with ``always``); records every call's kwargs."""

    def __init__(self, always=False):
        self.calls, self.always = [], always

    def transcribe(self, audio, **kw):
        self.calls.append(dict(kw))
        looped = self.always or kw.get("initial_prompt") is not None
        return [_Seg(["no"] * 10 if looped else ["hola", "buenos", "dias"])], None


def _asr(profile, model):
    a = VerbatimASR.__new__(VerbatimASR)
    a.model, a.language, a.profile = model, "es", get_profile(profile)
    return a


AUDIO = np.zeros(16000 * 2, np.float32)


def test_noloop_replaces_a_looped_chunk_and_records_it():
    m = FakeModel()
    a = _asr("p0_noloop", m)
    words = a.transcribe(AUDIO)
    assert [w.norm for w in words] == ["hola", "buenos", "dias"]
    assert len(m.calls) == 2
    # configuration registered == configuration used
    assert m.calls[0] == get_profile("p0_noloop").decode_kwargs("es")
    assert m.calls[1] == {
        **get_profile("p0_noloop").decode_kwargs("es"),
        **LOOP_REDECODE_LADDER[0],
    }
    assert a.redecode_events == [
        {
            "attempt": 1,
            "audio_s": 2.0,
            "chunk_offset_s": 0.0,
            "words_before": 10,
            "words_after": 3,
            "reproducible": True,
        }
    ]


def test_noloop_second_rung_uses_the_fixed_temperature_and_best_of():
    m = FakeModel(always=True)
    a = _asr("p0_noloop", m)
    words = a.transcribe(AUDIO)
    assert len(m.calls) == 1 + len(LOOP_REDECODE_LADDER)
    assert (
        m.calls[2]["temperature"] == 0.4
        and m.calls[2]["best_of"] == 5
        and m.calls[2]["initial_prompt"] is None
    )
    # nothing clean: the ORIGINAL text is kept, and the failure is recorded
    assert [w.norm for w in words] == ["no"] * 10
    assert (
        a.redecode_events[0]["attempt"] is None
        and a.redecode_events[0]["kept_original"] is True
    )


def test_noloop_does_not_touch_clean_chunks():
    m = FakeModel(always=False)
    a = _asr("p0_noloop", m)
    m.transcribe = lambda audio, **kw: (
        m.calls.append(dict(kw)) or ([_Seg(["hola", "mundo"])], None)
    )
    words = a.transcribe(AUDIO)
    assert [w.norm for w in words] == ["hola", "mundo"] and len(m.calls) == 1
    assert not getattr(a, "redecode_events", [])


@pytest.mark.parametrize("name", ["p0", "p1_disfluent_prompt"])
def test_default_and_p1_never_redecode_even_if_the_text_loops(name):
    m = FakeModel()
    a = _asr(name, m)
    words = a.transcribe(AUDIO)
    assert [w.norm for w in words] == ["no"] * 10 and len(m.calls) == 1
    assert not hasattr(a, "redecode_events")


def test_chunked_resets_the_event_log_per_call():
    m = FakeModel()
    a = _asr("p1_noloop", m)
    a.transcribe_chunked(AUDIO)
    assert len(a.redecode_events) == 1
    a.transcribe_chunked(AUDIO)
    assert len(a.redecode_events) == 1  # not accumulated across analyses


def test_warning_text_mentions_replacement_and_unresolved():
    assert loop_redecode_warning([]) is None
    w = loop_redecode_warning([{"attempt": 1}, {"attempt": None}])
    assert "1 chunk(s) were re-decoded" in w and "1 chunk(s) still looked looped" in w


def test_engine_metadata_lists_events_and_transcript_comes_from_them():
    from mrcd.engine import DisfluencyEngine

    m = FakeModel()
    a = _asr("p0_noloop", m)
    a.model_size = "small"
    a.device, a.compute_type = "cpu", "int8"
    a._model_info_done, a._model_bin_sha256, a._model_dir = True, "b" * 64, "x"
    words = a.transcribe_chunked(AUDIO)
    eng = DisfluencyEngine.__new__(DisfluencyEngine)
    eng._asr = a
    meta = DisfluencyEngine._asr_profile_meta(
        eng, AUDIO, " ".join(w.text for w in words), 8.0
    )
    lr = meta["loop_redecode"]
    assert (
        lr["enabled"]
        and lr["n_replaced"] == 1
        and lr["n_unresolved"] == 0
        and len(lr["events"]) == 1
    )
    assert meta["experimental"] is True and meta["name"] == "p0_noloop"
    assert SimpleNamespace(**meta).asr_fingerprint["profile_fingerprint"]
