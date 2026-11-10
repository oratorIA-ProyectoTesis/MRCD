"""Opt-in ASR profile ``ft_small_v1`` (Plan B fine-tune on CTranslate2 int8, Plan F). No real model is loaded."""

import hashlib

import pytest

from mrcd import ft_whisper as FT
from mrcd.asr_profiles import DEFAULT_PROFILE, PROFILES, asr_cache_key, get_profile


def _fake_dir(path, payload: bytes):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.bin").write_bytes(payload)
    (path / "tokenizer.json").write_text("{}")
    return path


def test_ft_small_v1_is_opt_in_experimental_and_p0_stays_the_default():
    p = get_profile("ft_small_v1")
    assert DEFAULT_PROFILE == "p0" and p.experimental
    assert (
        p.backend == "faster_whisper_local"
        and PROFILES["p0"].backend == "faster_whisper"
    )
    assert (p.device, p.compute_type) == ("cpu", "int8")
    assert "MIT" in p.license_note and "CrisperWhisper" in p.license_note
    # the historical profiles keep their description untouched
    assert (
        "backend" not in PROFILES["p0"].describe()
        and "decode_note" not in PROFILES["p1_disfluent_prompt"].describe()
    )


def test_ft_small_v1_describe_is_truthful_about_the_real_decode():
    d = get_profile("ft_small_v1").describe()
    assert d["initial_prompt"] == "" and "NO prompt" in d["decode_note"]
    assert (d["beam_size"], d["temperature"], d["chunk_s"], d["overlap_s"]) == (
        5,
        0.0,
        8.0,
        2.0,
    )
    assert (
        d["condition_on_previous_text"] is False
        and d["vad_filter"] is False
        and d["suppress_tokens"] == []
    )
    assert (d["device"], d["compute_type"]) == ("cpu", "int8")
    assert d["backend"] == "faster_whisper_local" and d["model_revision"] is None
    for needle in (
        "MRCD_FT_CT2_DIR",
        "model.bin",
        "p0 faster-whisper call without a prompt",
    ):
        assert needle in d["decode_note"]
    # decode kwargs equal the Plan B measured configuration (variant p_none: p0 settings, no prompt)
    from mrcd.asr_recovery import decode as D

    kw, plan_b = get_profile("ft_small_v1").decode_kwargs(), D.decode_kwargs("p_none")
    for k, v in plan_b.items():
        if k == "initial_prompt":
            assert not kw[k]  # "" and None both mean no prompt
        else:
            assert kw[k] == v, k


def test_cache_key_differs_from_p0_and_from_another_model_bin():
    kw = dict(audio_sha256="a" * 64, compute_type="int8", device="cpu")
    a = asr_cache_key("ft_small_v1", model_bin_sha256="1" * 64, **kw)
    assert a != asr_cache_key("p0", model_bin_sha256="1" * 64, **kw)
    assert a != asr_cache_key("ft_small_v1", model_bin_sha256="2" * 64, **kw)


def test_resolve_model_dir_env_beats_default_and_never_hardcodes_a_drive(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(FT.ENV_DIR, str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="not a CTranslate2"):
        FT.resolve_model_dir()
    good = _fake_dir(tmp_path / "ct2", b"x")
    monkeypatch.setenv(FT.ENV_DIR, str(good))
    assert FT.resolve_model_dir() == good
    for c in (
        FT.default_dir_candidates()
    ):  # repository-relative, whatever the checkout location
        assert c.parts[-3:] == FT.DEFAULT_DIR_PARTS
    import inspect

    assert "C:\\" not in inspect.getsource(FT) and "zArnes" not in inspect.getsource(FT)


def test_resolve_model_dir_without_env_or_default_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv(FT.ENV_DIR, raising=False)
    monkeypatch.setattr(FT, "default_dir_candidates", lambda: [])
    with pytest.raises(RuntimeError, match="MRCD_FT_CT2_DIR"):
        FT.resolve_model_dir()


def test_model_info_hashes_model_bin_lazily_and_differs_across_dirs(tmp_path):
    from mrcd.extractors.linguistic import VerbatimASR

    def asr_for(d):
        a = VerbatimASR.__new__(VerbatimASR)  # no weights: only the lazy hashing path
        a.profile = get_profile("ft_small_v1")
        a.model_size, a._model_bin_sha256, a._model_dir, a._model_info_done = (
            str(d),
            None,
            None,
            False,
        )
        a.device, a.compute_type = "cpu", "int8"
        return a

    da, db = (
        _fake_dir(tmp_path / "a", b"fine-tune-A"),
        _fake_dir(tmp_path / "b", b"fine-tune-B"),
    )
    ia, ib = asr_for(da).model_info(), asr_for(db).model_info()
    assert ia["model_bin_sha256"] == hashlib.sha256(b"fine-tune-A").hexdigest()
    assert (
        ia["model_bin_sha256"] != ib["model_bin_sha256"]
        and "model_info_error" not in ia
    )


def test_verbatim_asr_loads_the_env_directory_on_cpu_int8_and_rejects_cuda(
    monkeypatch, tmp_path
):
    import faster_whisper

    seen = {}

    class FakeModel:
        def __init__(self, path, device, compute_type):
            seen.update(path=path, device=device, compute_type=compute_type)

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    good = _fake_dir(tmp_path / "ct2", b"w")
    monkeypatch.setenv(FT.ENV_DIR, str(good))
    from mrcd.extractors.linguistic import VerbatimASR

    asr = VerbatimASR(profile="ft_small_v1")
    assert seen == {"path": str(good), "device": "cpu", "compute_type": "int8"}
    assert asr.profile.name == "ft_small_v1" and asr.model_size == str(good)
    with pytest.raises(ValueError, match="requires device='cpu'"):
        VerbatimASR(profile="ft_small_v1", device="cuda")


def test_filler_token_convention_matches_plan_b_targets():
    from mrcd.service.contextual import NONLEXICAL_FILLER_RE

    assert FT.FT_FILLER_TOKENS == {
        "a": "ah",
        "e": "eh",
        "m": "mm",
        "sin_identificar": "eh",
    }
    assert all(NONLEXICAL_FILLER_RE.match(t) for t in FT.FT_FILLER_TOKENS.values())
