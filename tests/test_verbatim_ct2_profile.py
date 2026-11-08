"""Opt-in profile ``cw2_ct2`` (CrisperWhisper 2.0 large on CTranslate2, Plan C). No model is loaded; no GPU needed."""

import numpy as np
import pytest

from mrcd import verbatim_ct2 as C
from mrcd.asr_profiles import DEFAULT_PROFILE, PROFILES, asr_cache_key, get_profile


def test_cw2_ct2_is_opt_in_experimental_and_pinned_to_the_cw2_revision():
    p = get_profile("cw2_ct2")
    cw2 = PROFILES["cw2"]
    assert DEFAULT_PROFILE == "p0" and p.experimental
    assert p.backend == "crisperwhisper_ct2" and cw2.backend == "crisperwhisper"
    assert (p.model_id, p.model_revision) == (cw2.model_id, cw2.model_revision)
    assert p.device == "cuda" and p.compute_type == "float16"
    assert "Non-Commercial" in p.license_note and "Derivative Work" in p.license_note
    d = p.describe()
    assert d["backend"] == "crisperwhisper_ct2" and d["license_note"] == p.license_note


def test_cw2_ct2_describe_is_truthful_about_the_real_decode():
    d = get_profile("cw2_ct2").describe()
    assert d["beam_size"] == 1 and d["temperature"] == 0.0  # greedy, not the ASRProfile default of 5
    assert (d["chunk_s"], d["overlap_s"]) == (30.0, 4.0)
    note = d["decode_note"]
    for needle in ("[verbatim_1]", "generation_config.json", "max_length=448", "30 s windows"):
        assert needle in note
    assert C.MAX_LENGTH == 448
    assert "decode_note" not in PROFILES["cw2"].describe()  # other profiles unchanged


def test_cw2_ct2_model_info_hashes_model_bin_lazily(tmp_path):
    import hashlib

    from mrcd.extractors.linguistic import VerbatimASR

    (tmp_path / "model.bin").write_bytes(b"conversion-A")
    asr = VerbatimASR.__new__(VerbatimASR)  # no weights: only the lazy hashing path
    asr.profile = get_profile("cw2_ct2")
    asr._model_dir, asr._model_bin_sha256, asr._model_info_done = str(tmp_path), None, False
    asr.model_size, asr.device, asr.compute_type = asr.profile.model_id, "cuda", "float16"
    info = asr.model_info()
    assert info["model_bin_sha256"] == hashlib.sha256(b"conversion-A").hexdigest()
    assert info["model_dir"] == str(tmp_path) and "model_info_error" not in info
    other = tmp_path / "b"
    other.mkdir()
    (other / "model.bin").write_bytes(b"conversion-B")
    asr2 = VerbatimASR.__new__(VerbatimASR)
    asr2.profile, asr2._model_dir = asr.profile, str(other)
    asr2._model_bin_sha256, asr2._model_info_done = None, False
    asr2.model_size, asr2.device, asr2.compute_type = asr.model_size, "cuda", "float16"
    assert asr2.model_info()["model_bin_sha256"] != info["model_bin_sha256"]


def test_cw2_ct2_cache_key_differs_from_cw2():
    kw = dict(
        audio_sha256="a" * 64,
        model_bin_sha256=None,
        compute_type="float16",
        device="cuda",
    )
    assert asr_cache_key("cw2_ct2", **kw) != asr_cache_key("cw2", **kw)


def test_special_ids_drop_control_and_mode_tokens_but_keep_bracketed_tags():
    vocab = {
        "<|startoftranscript|>": 1,
        "<|es|>": 2,
        "[verbatim_1]": 3,
        "[intended_2]": 4,
        "<ctx>": 5,
        "[UH]": 6,
        "[laughter]": 7,
        "hola": 8,
    }
    assert C.special_ids(vocab) == {1, 2, 3, 4, 5}


def test_place_tags_puts_each_tag_after_the_words_emitted_before_it():
    eot = 100
    words = [{"tokens": [10, 11]}, {"tokens": [12]}, {"tokens": [13]}]
    kept = [
        200,
        10,
        11,
        201,
        12,
        13,
        202,
    ]  # tag, word0 (2 tokens), tag, word1, word2, tag
    order = [(w, t) for _, w, t in C.place_tags(kept, eot, words)]
    assert order == [(None, 200), (0, 0), (None, 201), (1, 0), (2, 0), (None, 202)]


def test_place_tags_without_tags_keeps_word_order():
    words = [{"tokens": [1]}, {"tokens": [2, 3]}]
    assert [w for _, w, _ in C.place_tags([1, 2, 3], 100, words)] == [0, 1]


def test_resolve_model_dir_requires_an_explicit_local_conversion(monkeypatch, tmp_path):
    monkeypatch.delenv(C.ENV_DIR, raising=False)
    with pytest.raises(RuntimeError, match="MRCD_CW2_CT2_DIR"):
        C.resolve_model_dir()
    with pytest.raises(RuntimeError, match="not a CTranslate2"):
        C.resolve_model_dir(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"")
    (tmp_path / "tokenizer.json").write_text("{}")
    assert C.resolve_model_dir(tmp_path) == tmp_path


def test_loader_refuses_a_non_cuda_device():
    with pytest.raises(ValueError, match="requires device='cuda'"):
        C.load_crisper_ct2(PROFILES["cw2_ct2"], device="cpu")


class _FakeCT2:
    model_dir = "fake"

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return [
            ("[UH]", 0.0, 0.0, 1.0),
            (" no,", 0.1, 0.3, 0.9),
            ("[laughter]", 0.3, 0.3, 1.0),
            (" no", 0.4, 0.6, 0.8),
        ]


def test_verbatim_asr_cw2_ct2_maps_tags_like_cw2_and_skips_8s_chunking(monkeypatch):
    from mrcd.extractors import linguistic as L

    fake = _FakeCT2()
    monkeypatch.setattr(
        C,
        "load_crisper_ct2",
        lambda profile, device, compute_type: (fake, "cuda", "float16"),
    )
    asr = L.VerbatimASR(profile="cw2_ct2")
    assert asr.profile.name == "cw2_ct2" and asr.device == "cuda"
    words = asr.transcribe_chunked(np.zeros(20 * 16000, np.float32), chunk_s=8.0)
    assert (
        fake.calls == 1
    )  # 20 s audio, one call: the backend windows by itself (30 s), not MRCD's 8 s chunks
    assert [w.norm for w in words] == ["eh", "no", "no"]
    assert (
        words[0].prob == 1.0
        and words[1].prob == pytest.approx(0.9)
        and words[1].punct_after == ","
    )
    assert asr.dropped_tags == ["laughter"]
