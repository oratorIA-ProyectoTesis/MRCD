"""Deterministic boundaries for the reconstructed model and shared adapters."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import subprocess
import sys
from pathlib import Path

from core.constants import LABELS
from core.contracts import (FEATURE_SCHEMA_VERSION, FUSION_IMPLEMENTATION, RAW_FEATURE_SCHEMA_VERSION,
                            feature_contract, raw_feature_contract, validate_features,
                            validate_raw_features)
from core.inference import InferenceEngine
from core.models.fusion import MODES, CrossModalFusion, count_params
from core.normalization import apply_stats, fit_stats


def sample(n: int = 2) -> dict:
    rng = np.random.default_rng(5)
    return {
        "ac": rng.normal(size=(n, 300, 5)).astype(np.float32),
        "pros": rng.normal(size=(n, 9)).astype(np.float32),
        "ling": rng.normal(size=(n, 40, 12)).astype(np.float32),
        "lpos": np.tile(np.arange(40), (n, 1)),
        "lmask": np.ones((n, 40), bool),
        "kin": rng.normal(size=(n, 30, 12)).astype(np.float32),
        "kmask": np.ones((n, 30), bool),
        "has_video": np.ones(n, bool),
    }


def torch_batch(data: dict) -> dict:
    return {key: torch.as_tensor(value) for key, value in data.items()}


def test_versioned_feature_contract_rejects_schema_and_shape_drift():
    data = {**sample(), "y": np.array([0, 1]), "labels": np.array(LABELS),
            "label_source": np.array("auto"), "feature_schema": np.array(FEATURE_SCHEMA_VERSION)}
    assert validate_features(data, require_labels=True) == 2
    with pytest.raises(ValueError, match="unsupported feature schema"):
        validate_features({**data, "feature_schema": np.array("future")})
    with pytest.raises(ValueError, match="kmask"):
        validate_features({**data, "kmask": np.ones((2, 29), bool)})
    with pytest.raises(ValueError, match="taxonomy"):
        validate_features({**data, "labels": np.array(list(reversed(LABELS)))}, require_labels=True)
    with pytest.raises(ValueError, match="has_video from >=24 real visual frames"):
        validate_features({key: value for key, value in data.items() if key != "has_video"})
    with pytest.raises(ValueError, match="requires has_video"):
        CrossModalFusion(len(LABELS))(torch_batch({key: value for key, value in data.items()
                                                   if key in ("ac", "pros", "ling", "lpos", "lmask", "kin", "kmask")}))


def test_raw_extraction_contract_rejects_legacy_and_stale_inputs(tmp_path):
    import json
    from core.dataset import build_recording_windows

    raw = {"raw_feature_schema": np.array(RAW_FEATURE_SCHEMA_VERSION),
           "raw_contract": np.array(json.dumps(raw_feature_contract())),
           "raw_extractor": np.array("extract_and_label"),
           "raw_extractor_config": np.array(json.dumps({"asr_model": "test", "variety": "es-PE",
                                                         "vad_backend": "test", "face_tracking_method": "none"})),
           "ac_frames": np.zeros((300, 5), np.float32),
           "word_feats": np.zeros((0, 10), np.float32),
           "k_times": np.zeros(0), "k_vectors": np.zeros((0, 12), np.float32)}
    assert validate_raw_features(raw)["extractor"] == "extract_and_label"
    with pytest.raises(ValueError, match="raw extractor contract differs"):
        validate_raw_features({**raw, "raw_contract": np.array(json.dumps({"version": "old"}))})
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, ac_frames=raw["ac_frames"])
    with pytest.raises(ValueError, match="re-extract"):
        build_recording_windows({"features_npz": legacy.name}, tmp_path)


def test_extraction_resume_refuses_unversioned_npz_before_loading_models(tmp_path, monkeypatch):
    import json
    import scripts.extract_and_label as extractor

    features = tmp_path / "features"
    features.mkdir()
    np.savez_compressed(features / "r.npz", ac_frames=np.zeros((300, 5), np.float32))
    index = tmp_path / "index.json"
    index.write_text(json.dumps([{"recording_id": "r"}]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"recordings": [{"recording_id": "r", "features_npz": "features/r.npz"}]}))
    monkeypatch.setattr(extractor, "ROOT", tmp_path)
    monkeypatch.setattr(extractor, "FEAT", features)
    monkeypatch.setattr(extractor, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["extract_and_label.py", "--index", str(index)])
    with pytest.raises(ValueError, match="re-extract"):
        extractor.main()


def test_mask_aware_stats_ignore_padding_and_keep_it_zero():
    data = sample()
    data["ling"][:] = 1000
    data["ling"][:, 0] = np.array([1] * 12, np.float32)
    data["ling"][:, 1] = np.array([3] * 12, np.float32)
    data["lmask"][:] = False
    data["lmask"][:, :2] = True
    data["kin"][:] = 1000
    data["kmask"][:] = False
    stats = fit_stats(data, np.array([0]))
    assert stats["ling"][0][0] == pytest.approx(2)
    np.testing.assert_allclose(stats["kin"][0], 0)
    normalized = apply_stats(data, stats)
    assert np.all(normalized["ling"][~data["lmask"]] == 0)
    assert np.all(normalized["kin"] == 0)


def test_all_modes_remain_finite_with_both_auxiliary_modalities_missing():
    data = sample()
    data["lmask"][:] = False
    data["kmask"][:] = False
    for mode in MODES:
        net = CrossModalFusion(len(LABELS), mode=mode).eval()
        with torch.no_grad():
            logits, weights = net(torch_batch(data), return_attn=True)
        assert logits.shape == (2, len(LABELS)) and torch.isfinite(logits).all()
        assert torch.all(weights["syntactic"] == 0)
        assert torch.all(weights["kinesic"] == 0)
        assert torch.allclose(sum(weights.values()), torch.ones(2))
        assert count_params(net) > 0


def test_masked_values_and_excluded_modes_cannot_change_prediction():
    data = sample()
    data["lmask"][:] = False
    data["kmask"][:] = False
    changed = {**data, "ling": data["ling"] + 1000, "kin": data["kin"] - 1000}
    for mode in MODES:
        net = CrossModalFusion(len(LABELS), mode=mode).eval()
        with torch.no_grad():
            assert torch.allclose(net(torch_batch(data)), net(torch_batch(changed)), atol=1e-6)


def test_interpolated_visual_tokens_do_not_bypass_real_frame_availability():
    data = sample()
    data["has_video"][:] = False
    assert data["kmask"].any()  # An interpolated grid is not real-frame coverage.
    changed = {**data, "kin": data["kin"] + 1000}
    net = CrossModalFusion(len(LABELS), mode="trimodal").eval()
    with torch.no_grad():
        logits, weights = net(torch_batch(data), return_attn=True)
        changed_logits = net(torch_batch(changed))
    assert torch.all(weights["kinesic"] == 0)
    assert torch.allclose(logits, changed_logits, atol=1e-6)
    normalized = apply_stats(data, fit_stats(data))
    assert np.all(normalized["kin"] == 0)


def test_batch_and_live_adapters_use_the_same_inference_class(tmp_path):
    import scripts.infer_video as batch_adapter
    import scripts.live_demo as live_adapter

    assert batch_adapter.InferenceEngine is InferenceEngine
    assert live_adapter.InferenceEngine is InferenceEngine
    data = sample()
    norm = fit_stats(data)
    net = CrossModalFusion(len(LABELS), mode="trimodal").eval()
    ckpt = tmp_path / "model.pt"
    torch.save({"state_dict": net.state_dict(), "mode": "trimodal", "labels": list(LABELS),
                "dims": {"ac": 5, "pros": 9, "ling": 12, "d": 64},
                "norm": {key: (mu.tolist(), sd.tolist()) for key, (mu, sd) in norm.items()},
                "feature_contract": feature_contract(),
                "fusion_implementation": FUSION_IMPLEMENTATION,
                "provenance": {"label_source": "auto"}}, ckpt)
    engine = InferenceEngine.from_checkpoint(ckpt, "cpu")
    assert engine.provenance["contract_status"] == "versioned"
    assert engine.predict(data).shape == (2, len(LABELS))
    with pytest.raises(ValueError, match="width differs"):
        engine.predict({**data, "ling": data["ling"][:, :, :-1]})


def test_provider_dispatch_never_crosses_credentials(monkeypatch):
    import scripts.llm_judge as judge

    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(key, key)
    cases = (("claude-test", "anthropic", judge.call_anthropic),
             ("gpt-test", "openai", judge.call_openai),
             ("gemini-test", "gemini", judge.call_gemini),
             ("llama-test", "groq", judge.call_groq))
    for model, name, caller in cases:
        provider, credential = judge.pick_provider(model)
        assert provider == name and credential == judge.PROVIDER_KEYS[name]
        assert judge.caller_for(provider) is caller
    assert judge.pick_provider("qwen2.5vl:7b", "ollama") == ("ollama", "ollama")
    assert judge.caller_for("ollama") is judge.call_ollama
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        judge.pick_provider("gemini-test")


def test_feature_core_does_not_import_script_adapters():
    """The common use case must stay importable without the CLI/web adapters."""
    import ast
    from pathlib import Path
    core_file = Path(__file__).resolve().parents[1] / "core/inference.py"
    imports = [node.module for node in ast.walk(ast.parse(core_file.read_text()))
               if isinstance(node, ast.ImportFrom)]
    assert not any(name and name.startswith("scripts") for name in imports)


def test_general_feature_worker_tracks_a_stable_face_and_reports_method(tmp_path, monkeypatch):
    import core.extractors.kinesic as kinesic
    import scripts.extract_and_label as extract

    (tmp_path / "clip.mp4").write_bytes(b"stub")
    monkeypatch.setattr(extract, "ROOT", tmp_path)
    monkeypatch.setattr(extract, "load_wav", lambda path: (np.zeros(16000, np.float32), 16000))
    monkeypatch.setattr(extract, "extract_acoustic", lambda audio, sr: object())
    calls = []

    class FaceExtractor:
        def __init__(self, num_faces):
            calls.append(("init", num_faces))

        def process_video_tracked(self, path):
            calls.append(("tracked", path))
            return object(), {"selected": 4, "dominance_margin": 0.5, "tracks": []}

        def close(self):
            calls.append(("closed",))

    monkeypatch.setattr(kinesic, "KinesicExtractor", FaceExtractor)
    rid, acoustic, visual, timings = extract.signal_worker(
        {"recording_id": "clip", "wav": "clip.wav", "mp4": "clip.mp4"}, True)
    assert rid == "clip" and acoustic is not None and visual is not None
    assert timings["face_tracking"]["selected"] == 4
    assert [item[0] for item in calls] == ["init", "tracked", "closed"]
    assert calls[0][1] == 3


def test_synthetic_train_checkpoint_inference_roundtrip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = sample(4)
    npz = tmp_path / "train.npz"
    np.savez_compressed(npz, **data, y=np.array([0, 1, 2, 3]),
                        labels=np.array(LABELS), label_source=np.array("auto"),
                        feature_schema=np.array(FEATURE_SCHEMA_VERSION))
    out = tmp_path / "models"
    run = subprocess.run([sys.executable, str(root / "scripts/train_checkpoints.py"),
                          "--data", str(npz), "--out", str(out), "--epochs", "1",
                          "--bs", "2", "--d", "16", "--modes", "audio_text"],
                         cwd=root, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    engine = InferenceEngine.from_checkpoint(out / "ckpt_audio_text.pt", "cpu")
    probabilities = engine.predict(data)
    assert probabilities.shape == (4, len(LABELS))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert engine.provenance["contract_status"] == "versioned"


def test_unversioned_legacy_windows_cannot_claim_current_checkpoint_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    npz = tmp_path / "legacy.npz"
    np.savez_compressed(npz, **sample(2), y=np.array([0, 1]),
                        labels=np.array(LABELS), label_source=np.array("auto"))
    out = tmp_path / "models"
    run = subprocess.run([sys.executable, str(root / "scripts/train_checkpoints.py"),
                          "--data", str(npz), "--out", str(out), "--epochs", "1",
                          "--modes", "audio_only"],
                         cwd=root, capture_output=True, text=True, timeout=30)
    assert run.returncode != 0
    assert "unversioned windows cannot be certified" in run.stdout + run.stderr
    assert not out.exists()


def test_subset_labels_cannot_emit_versioned_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    npz = tmp_path / "subset.npz"
    np.savez_compressed(npz, **sample(2), y=np.array([0, 1]),
                        labels=np.array(LABELS[:3]), label_source=np.array("auto"),
                        feature_schema=np.array(FEATURE_SCHEMA_VERSION))
    out = tmp_path / "models"
    run = subprocess.run([sys.executable, str(root / "scripts/train_checkpoints.py"),
                          "--data", str(npz), "--out", str(out), "--epochs", "1",
                          "--modes", "audio_only"],
                         cwd=root, capture_output=True, text=True, timeout=30)
    assert run.returncode != 0
    assert "versioned labels differ" in run.stdout + run.stderr
    assert not out.exists()


def test_case_report_rejects_checkpoint_renamed_to_wrong_mode(tmp_path):
    from scripts.infer_video import load_reporting_engine

    data = sample()
    stats = fit_stats(data)
    net = CrossModalFusion(len(LABELS), mode="audio_text")
    path = tmp_path / "ckpt_trimodal.pt"
    torch.save({"state_dict": net.state_dict(), "mode": "audio_text", "labels": list(LABELS),
                "dims": {"ac": 5, "pros": 9, "ling": 12, "d": 64},
                "norm": {key: (mu.tolist(), sd.tolist()) for key, (mu, sd) in stats.items()},
                "feature_contract": feature_contract(),
                "fusion_implementation": FUSION_IMPLEMENTATION,
                "provenance": {"label_source": "auto"}}, path)
    with pytest.raises(ValueError, match="does not match requested 'trimodal'"):
        load_reporting_engine(path, "trimodal", "cpu")


def test_gold_window_metadata_preserves_zero_frame_fallback(tmp_path):
    import json
    from core.dataset import build_recording_windows, gold_window_metadata

    events = [{"category": "block", "n_frames": 0, "provider": "openai", "model": "gpt-4o",
               "input_fingerprint": "sha256", "fallback_note": "sin_fotogramas",
               "interval_identity_version": "ms-v1"}]
    meta = gold_window_metadata(events, np.array([1]), 3)
    assert meta["judge_n_frames"][1] == 0
    assert not meta["judge_visual_evidence_used"][1]
    assert meta["judge_input_fingerprint"][1] == "sha256"
    assert meta["judge_fallback_note"][1] == "sin_fotogramas"
    raw = tmp_path / "r.npz"
    np.savez_compressed(raw, raw_feature_schema=np.array(RAW_FEATURE_SCHEMA_VERSION),
                        raw_contract=np.array(json.dumps(raw_feature_contract())),
                        raw_extractor=np.array("extract_and_label"),
                        raw_extractor_config=np.array(json.dumps({"asr_model": "test", "variety": "es-PE",
                                                                  "vad_backend": "test", "face_tracking_method": "none"})),
                        ac_frames=np.zeros((300, 5), np.float32),
                        word_feats=np.zeros((0, 10), np.float32),
                        k_times=np.zeros(0), k_vectors=np.zeros((0, 12), np.float32))
    (tmp_path / "r.words.json").write_text("[]")
    (tmp_path / "r.gold.json").write_text(json.dumps([dict(events[0], start_ms=0, end_ms=1000)]))
    rec = {"features_npz": "r.npz", "words_json": "r.words.json",
           "events_json": "r.events.json", "duration_s": 3.0,
           "speaker_id": "s", "recording_id": "r"}
    windows = build_recording_windows(rec, tmp_path, "gold")
    assert windows["judge_n_frames"].tolist() == [0]
    assert windows["judge_visual_evidence_used"].tolist() == [False]
    assert windows["judge_fallback_note"].tolist() == ["sin_fotogramas"]
