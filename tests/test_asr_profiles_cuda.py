"""Opt-in GPU ASR profiles (p0_cuda, p1_cuda): registered, experimental, distinct fingerprint, clean rejection."""

from __future__ import annotations

import sys
import types

import pytest

from mrcd.asr_profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    ASRDeviceUnavailable,
    asr_cache_key,
    get_profile,
)
from mrcd.extractors.linguistic import VerbatimASR

# pinned on 2026-11-01 BEFORE the GPU profiles were added: adding them must not move these
P0_NAMESPACE = "p0-bd0bddb4329c"
P0_KEY = "3e3d5b59262bb8e899ce0f09b6abf6e865c6ad1d4c3732a2610c227b04e5a83e"
P1_KEY = "3ceddf8ec9c693e4bf157e2203c7d68058d7f97c3149e5488ab1dfef2307be28"
KW = dict(audio_sha256="a" * 64, model_bin_sha256="b", compute_type="int8", device="cpu", omp_num_threads="4")


def test_registered_experimental_and_default_unchanged():
    assert DEFAULT_PROFILE == "p0"
    for name, base in (("p0_cuda", "p0"), ("p1_cuda", "p1_disfluent_prompt")):
        p = PROFILES[name]
        assert p.experimental is True and p.device == "cuda" and p.compute_type == "float16"
        # only device/compute_type/name/labels differ from the CPU twin
        assert p.initial_prompt == PROFILES[base].initial_prompt
        assert p.decode_kwargs() == PROFILES[base].decode_kwargs()
        assert (p.chunk_s, p.overlap_s) == (PROFILES[base].chunk_s, PROFILES[base].overlap_s)
        assert p.describe()["device"] == "cuda" and p.describe()["experimental"] is True
    assert get_profile(None).name == "p0"


def test_cpu_profiles_fingerprint_and_description_unchanged():
    assert get_profile("p0").cache_namespace() == P0_NAMESPACE
    assert asr_cache_key("p0", **KW) == P0_KEY
    assert asr_cache_key("p1_disfluent_prompt", **KW) == P1_KEY
    assert "device" not in get_profile("p0").describe()
    assert "device" not in get_profile("p1_disfluent_prompt").describe()
    assert not PROFILES["p0"].experimental and PROFILES["p0"].device is None


def test_gpu_fingerprint_differs_even_with_same_runtime_fields():
    keys = {n: asr_cache_key(n, **KW) for n in ("p0", "p0_cuda", "p1_disfluent_prompt", "p1_cuda")}
    assert len(set(keys.values())) == 4
    assert get_profile("p0_cuda").cache_namespace() != get_profile("p0").cache_namespace()
    # the same profile on a different device also differs
    assert asr_cache_key("p0_cuda", **{**KW, "device": "cuda", "compute_type": "float16"}) != keys["p0_cuda"]


def _fake_faster_whisper(monkeypatch):
    called = []

    class Boom:
        def __init__(self, *a, **k):
            called.append((a, k))

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Boom))
    return called


@pytest.mark.parametrize("name", ["p0_cuda", "p1_cuda"])
def test_clean_rejection_without_cuda(monkeypatch, name):
    called = _fake_faster_whisper(monkeypatch)
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 0))
    with pytest.raises(ASRDeviceUnavailable, match="requires CUDA"):
        VerbatimASR("small", profile=name)
    assert called == []  # the model was never loaded, no CPU fallback


def test_clean_rejection_when_cuda_probe_fails(monkeypatch):
    class Model:
        def __init__(self, *a, **k):
            self.k = k

        def transcribe(self, *a, **k):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1))
    with pytest.raises(ASRDeviceUnavailable, match="no CPU fallback"):
        VerbatimASR("small", profile="p0_cuda")


def test_conflicting_device_is_rejected(monkeypatch):
    _fake_faster_whisper(monkeypatch)
    with pytest.raises(ValueError, match="requires device='cuda'"):
        VerbatimASR("small", device="cpu", profile="p1_cuda")


def test_gpu_profile_requests_cuda_float16(monkeypatch):
    seen = {}

    class Model:
        def __init__(self, size, device, compute_type):
            seen.update(device=device, compute_type=compute_type)

        def transcribe(self, *a, **k):
            return iter(()), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1))
    asr = VerbatimASR("small", profile="p0_cuda")
    assert seen == {"device": "cuda", "compute_type": "float16"}
    assert (asr.device, asr.compute_type, asr.profile.name) == ("cuda", "float16", "p0_cuda")


def test_cpu_profile_default_path_unchanged(monkeypatch):
    seen = {}

    class Model:
        def __init__(self, size, device, compute_type):
            seen.update(device=device, compute_type=compute_type)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    VerbatimASR("small", device="cpu")
    assert seen == {"device": "cpu", "compute_type": "int8"}
