"""Opt-in verbatim ASR profile ``cw2`` (CrisperWhisper 2.0) and its opt-in decision route.

Seams: the profile registry (``mrcd.asr_profiles``), the tag mapping and route (``mrcd.verbatim``), and
``VerbatimASR`` with an injected backend. No model is loaded; nothing here needs a GPU.
"""

import hashlib
import json
import re
import types

import numpy as np
import pytest

from mrcd import verbatim as V
from mrcd.asr_profiles import DEFAULT_PROFILE, PROFILES, asr_cache_key, get_profile

# Golden values captured on main BEFORE the cw2 change: the pre-existing profiles must stay bit-identical.
GOLDEN_DESCRIBE = {
    "p0": "0f0c4f9bf6fb673c16f4087c27fdea661ce8f6eb3efbbe2df67e68d0586f962c",
    "p1_disfluent_prompt": "1b18c943a67cd817e3b0b872a1a46dd81be557ec8bf0e86da628d4a93c00a2ee",
    "p0_cuda": "33cee0b6229b18be419828502e7333d5c040d06fd25d9c41c2610d50bcaa1f17",
    "p1_cuda": "e3f0c436783bdcf1dbcb334b1b21dd7fc16bf628a44c89d1d096331ac1914fba",
    "p0_noloop": "bb94c5b68aaa22001f9e4e015454b214f87b89e1679e27a18b2afe69440b2d4e",
    "p1_noloop": "83d55e17bef6ed0b46a3371deed6009701b2711d80aa6b7cb5b7b451219b0f07",
}
GOLDEN_KEY = {
    "p0": "edfba15b65a4d173405dd46c5bb59c20912c70233fe64b2aa25aec181ada5d8a",
    "p1_disfluent_prompt": "ba122431db833a3dba9f9aa1ff6370271f2b3112f65090d8ced76175c4041460",
    "p0_cuda": "24acb7d5c715752ca86e025656591b9bab0f9d1acdad9f14dca22cb88bdf2d88",
    "p1_cuda": "b37ad9a179aa9323e8164507942493165704ede3c1b925521c669a9963fa0581",
    "p0_noloop": "c2aecafec8b92a9ff7eab4dde1df2c54d43fe3bdf0594cf8388982178c7fafd6",
    "p1_noloop": "3b8f93670102517e91f57df27c03ac29eeb4e60678275576ea7cb73353ad68ba",
}


def test_default_profile_is_still_p0():
    assert DEFAULT_PROFILE == "p0"
    assert get_profile(None).name == "p0"
    assert not PROFILES["p0"].experimental


@pytest.mark.parametrize("name", sorted(GOLDEN_DESCRIBE))
def test_existing_profiles_are_bit_identical(name):
    d = json.dumps(PROFILES[name].describe(), sort_keys=True, ensure_ascii=True)
    assert hashlib.sha256(d.encode()).hexdigest() == GOLDEN_DESCRIBE[name]
    key = asr_cache_key(
        name,
        audio_sha256="a" * 64,
        model_bin_sha256="b" * 64,
        compute_type="int8",
        device="cpu",
        omp_num_threads="4",
    )
    assert key == GOLDEN_KEY[name]


def test_cw2_is_registered_opt_in_and_pinned():
    p = get_profile("cw2")
    assert p.experimental and p.backend == "crisperwhisper"
    assert p.model_id == "nyralabs/CrisperWhisper2.0_large"
    assert re.fullmatch(r"[0-9a-f]{40}", p.model_revision)
    assert "non-commercial" in p.license_note.lower()
    d = p.describe()
    assert d["backend"] == "crisperwhisper" and d["model_revision"] == p.model_revision
    assert "non-commercial" in d["license_note"].lower()


def test_cw2_cache_key_depends_on_revision_and_differs_from_p0():
    kw = dict(
        audio_sha256="a" * 64,
        model_bin_sha256=None,
        compute_type="float16",
        device="cuda",
    )
    k = asr_cache_key("cw2", **kw)
    assert k != asr_cache_key("p0", **kw)
    from dataclasses import replace

    other = replace(PROFILES["cw2"], name="cw2_other_rev", model_revision="0" * 40)
    assert asr_cache_key(other, **kw) != k  # the pinned revision is part of the key
    with pytest.raises(ValueError):
        get_profile(
            replace(PROFILES["cw2"], model_revision="0" * 40)
        )  # cannot pose as the registered name


def test_engine_default_asr_profile_unchanged():
    import inspect

    from mrcd.engine import DisfluencyEngine

    assert (
        inspect.signature(DisfluencyEngine.load).parameters["asr_profile"].default
        == "p0"
    )


# ------------------------------------------------------------------ tag mapping
def test_declared_tag_mapping_is_total_over_the_filled_pause_set():
    assert set(V.FP_TAG_NORM) == {
        "uh",
        "um",
        "eh",
        "em",
        "er",
        "ah",
        "ahh",
        "hm",
        "hmm",
        "mm",
        "mmm",
    }
    assert (
        V.FP_TAG_NORM["uh"] == "eh"
        and V.FP_TAG_NORM["um"] == "em"
        and V.FP_TAG_NORM["hmm"] == "mm"
    )


def test_tags_become_filler_words_and_unknown_tags_are_dropped_and_counted():
    toks, dropped = V.map_tags("[UH] no, no [LAUGH] sé [Um]")
    assert toks == ["eh", "no,", "no", "sé", "em"]
    assert dropped == ["laugh"]


def test_strip_tags_and_filled_pause_detection():
    assert V.strip_tags("de [UH] de casa") == "de de casa"
    assert V.has_filled_pause_tag("[UH] no") and not V.has_filled_pause_tag(
        "[LAUGH] no"
    )
    assert not V.has_filled_pause_tag("uh no")  # a bare word is not a tag


# ------------------------------------------------------------------ route (declared fusion: OR)
def test_route_name_and_rule_are_declared():
    assert V.ROUTE_NAME == "verbatim_or"
    assert "OR" in V.ROUTE_RULE


@pytest.mark.parametrize(
    "text, clf, rep, fill",
    [
        (
            "quiero ir a ir a casa",
            {"repetition": 0, "filler_word": 0},
            1,
            0,
        ),  # verbatim rule fires, classifier silent
        (
            "quiero ir a casa",
            {"repetition": 1, "filler_word": 0},
            1,
            0,
        ),  # classifier fires, rule silent
        ("quiero ir a casa", {"repetition": 0, "filler_word": 0}, 0, 0),
        (
            "[UH]",
            {"repetition": 0, "filler_word": 0},
            0,
            1,
        ),  # tag-only turn is filler evidence
        (
            "de [UH] de casa",
            {"repetition": 0, "filler_word": 0},
            1,
            1,
        ),  # tag does not break adjacency
        (
            "eh quiero",
            {"repetition": 0, "filler_word": 0},
            0,
            1,
        ),  # lexical filled pause
        (
            "quiero este billete",
            {"repetition": 0, "filler_word": 0},
            0,
            0,
        ),  # 'este' is excluded by the rule
        ("quiero", {"repetition": 0, "filler_word": 1}, 0, 1),
    ],
)
def test_route_decisions(text, clf, rep, fill):
    out = V.route_decisions(text, clf)
    assert (out["repetition"], out["filler_word"]) == (rep, fill)


def test_route_does_not_invent_classes():
    assert set(V.route_decisions("a a", {"repetition": 0, "filler_word": 0})) == {
        "repetition",
        "filler_word",
    }


# ------------------------------------------------------------------ VerbatimASR with an injected backend
class _FakeCW:
    def transcribe(self, audio, **kw):
        self.kw = kw
        w = [
            types.SimpleNamespace(word="[UH]", start=0.0, end=0.3),
            types.SimpleNamespace(word="no,", start=0.4, end=0.6),
            types.SimpleNamespace(word="[LAUGH]", start=0.6, end=0.8),
            types.SimpleNamespace(word="no", start=0.9, end=1.0),
        ]
        return types.SimpleNamespace(text="[UH] no, [LAUGH] no", words=w)


def test_verbatim_asr_cw2_maps_words_through_the_declared_tag_map(monkeypatch):
    from mrcd.extractors import linguistic as L

    fake = _FakeCW()
    monkeypatch.setattr(
        V,
        "load_crisper",
        lambda profile, device, compute_type: (fake, "cuda", "float16"),
    )
    asr = L.VerbatimASR(profile="cw2")
    assert asr.profile.name == "cw2" and asr.device == "cuda"
    words = asr.transcribe(np.zeros(16000, np.float32))
    assert [w.norm for w in words] == ["eh", "no", "no"]  # [LAUGH] dropped
    assert words[0].text == "eh" and words[1].punct_after == ","
    assert (
        fake.kw["mode"] == "verbatim"
        and fake.kw["language"] == "es"
        and fake.kw["word_timestamps"] is True
    )
    assert asr.dropped_tags == ["laugh"]
