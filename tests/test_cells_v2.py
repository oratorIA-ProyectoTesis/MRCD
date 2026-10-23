"""Additive v2 features/model/training: synthetic local signals only."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from core.features_v2 import build_cell_features
from core.cell_contracts import cell_feature_contract, validate_cell_features
from core.cell_normalization import fit_cell_stats, apply_cell_stats
from core.models.cell_fusion import CellFusionV2, tensor_batch
from core.training_v2 import masked_loss, supervision_counts, train_step, train_cells
from core.training_v2 import masked_metrics
from core.cell_dataset import build_dataset, load_dataset, content_digest
from core.cell_inference import CellInference
from core.targets import project_targets


def raw_signals(duration=1500):
    times = np.arange(0, duration + 11, 10, dtype=np.float64) / 1000
    frames = np.zeros((len(times), 5), np.float32)
    frames[:, 0], frames[:, 2], frames[:, 3] = -1, .9, 1
    return dict(ac_times=times, ac_frames=frames, f0=np.full(len(times), 100.),
                k_times=np.zeros(0), k_vectors=np.zeros((0, 12)), k_detected=np.zeros(0, bool))


def human(start=100, end=200, context=True, label="filler_word"):
    return dict(start_ms=start, end_ms=end, label=label, type_known=True, is_disfluent=context)


def projected(duration=1500):
    return project_targets(duration, [human(), human(600, 700, False)], [[0, duration]])


def synthetic_manifest(tmp_path, duration=1500, events=None, coverage=None):
    np.savez(tmp_path / "raw.npz", **raw_signals(duration))
    (tmp_path / "words.json").write_text("[]")
    (tmp_path / "events.json").write_text(json.dumps(events if events is not None else [human(), human(600, 700, False)]))
    manifest = dict(source_kind="synthetic", recordings=[dict(
        recording_id="synthetic-r1", speaker_id="synthetic-s1", split="train",
        duration_ms=duration, source_identity="synthetic-local-signal-fixture",
        raw_npz="raw.npz", words_json="words.json", events_json="events.json",
        reviewed_coverage=coverage if coverage is not None else [[0, duration]])])
    file_path = tmp_path / "input.json"
    file_path.write_text(json.dumps(manifest))
    return file_path


def test_duration_not_frame_count_and_frame_at_D_never_becomes_evidence():
    raw = raw_signals(501)
    raw["ac_frames"][raw["ac_times"] >= .501] = 1e30
    features = build_cell_features(raw, [], 501)
    assert features["ac"].shape == (2, 300, 5)
    assert features["ac_support"].sum(axis=1).tolist() == [51, 51]
    assert not np.any(features["ac"][features["ac_mask"]] == 1e30)
    assert features["ac_target"].sum(axis=1).tolist() == [50, 1]
    assert features["ac_mask"][..., 1][features["ac_support"]].all()  # 100Hz = 0 semitones, observed
    assert validate_cell_features(features) == 2
    sparse = raw_signals(100)
    assert build_cell_features(sparse, [], 3250)["ac"].shape[0] == 7


def test_pitch_delta_requires_adjacent_real_voiced_samples():
    raw = raw_signals(500)
    raw["f0"][2] = np.nan
    features = build_cell_features(raw, [], 500)
    observed = features["ac_support"][0]
    mask = features["ac_mask"][0, observed]
    assert mask[:, 1].sum() == 49
    assert not mask[0, 4] and not mask[2, 4] and not mask[3, 4]


def test_missing_pause_is_not_observed_zero_and_censored_duration_not_exact():
    raw = raw_signals(500)
    f = build_cell_features(raw, [], 500)
    assert f["pros"][0, 0] == 0 and f["pros_mask"][0, 0]
    assert not f["pros_mask"][0, 1]
    raw["ac_frames"][:, 3] = np.nan
    assert not build_cell_features(raw, [], 500)["pros_mask"].any()
    raw["ac_frames"][:, 3] = 0
    assert not build_cell_features(raw, [], 500)["pros_mask"][0, 0]


def test_prosody_selects_pause_overlapping_owned_not_longest_context_pause():
    raw = raw_signals(3000)
    raw["ac_frames"][10:20, 3] = 0  # owned first cell
    raw["ac_frames"][80:140, 3] = 0  # larger but not owned first cell
    f = build_cell_features(raw, [], 3000)
    assert f["pros"][0, 0] == pytest.approx(.1)
    assert f["pros_mask"][0, 5] and f["pros_mask"][0, 2]
    assert f["pros"].shape[1] == 8  # reset validity is a mask, not a duplicate feature


def test_linguistic_target_first_context_boundary_neighbor_isolation_and_overflow():
    words = [dict(text="hola", start_ms=i * 100, end_ms=i * 100 + 50, prob=.9, id=f"w{i}") for i in range(120)]
    f = build_cell_features(raw_signals(12000), words, 12000)
    row = 10  # center5250ms: context [250,10250), owned[5000,5500)
    selected = f["word_indices"][row][f["lmask"][row]].tolist()
    assert set(range(50, 55)).issubset(selected) and len(selected) == 40
    assert selected == sorted(selected)
    assert f["ltarget"][row].sum() == 5
    assert f["word_indices"][0][0] == 0 and not f["ling_mask"][0, 0, 8]
    modified = deepcopy(words)
    modified[-1]["text"] = "unrelated outside first context"
    f2 = build_cell_features(raw_signals(12000), modified, 12000)
    np.testing.assert_array_equal(f["ling"][0], f2["ling"][0])
    crowded = [dict(text="x", start_ms=i, end_ms=i + 1) for i in range(41)]
    with pytest.raises(ValueError, match="target_word_overflow"):
        build_cell_features(raw_signals(500), crowded, 500)


def test_visual_real_detection_threshold_no_extrapolation_audio_fallback():
    raw = raw_signals(3250)
    raw.update(k_times=np.arange(34) / 10, k_vectors=np.zeros((34, 12)), k_detected=np.ones(34, bool))
    f = build_cell_features(raw, [], 3250)
    assert f["real_frame_count"][3] == 30 and f["has_video"][3]
    assert f["real_frame_count"][0] == 18 and not f["has_video"][0]
    raw["k_detected"][:] = False
    raw["k_detected"][:23] = True
    assert not build_cell_features(raw, [], 3250)["has_video"].any()
    assert build_cell_features(raw_signals(200), [], 200)["ac_target"].any()


def test_masked_extremes_nan_and_unobserved_rows_do_not_change_stats_or_logits():
    torch.manual_seed(13)
    f = build_cell_features(raw_signals(), [], 1500)
    targets = projected()
    stats = fit_cell_stats(f, targets, indices=[0, 1])
    poisoned = deepcopy(f)
    for key in ("ac", "pros", "ling", "kin"):
        poisoned[key][~poisoned[key + "_mask"]] = np.nan
    altered = fit_cell_stats(poisoned, targets, indices=[0, 1])
    for key in stats:
        np.testing.assert_array_equal(stats[key]["count"], altered[key]["count"])
        np.testing.assert_array_equal(stats[key]["mean"], altered[key]["mean"])
    assert stats["kin"]["count"].sum() == 0 and np.all(stats["kin"]["std"] == 1)
    model = CellFusionV2(mode="trimodal", d_model=16).eval()
    a = model(tensor_batch(apply_cell_stats(f, stats)))
    b = model(tensor_batch(apply_cell_stats(poisoned, stats)))
    torch.testing.assert_close(a["type_logits"], b["type_logits"])


def test_model_consumes_target_and_position_and_no_context_only_eligibility():
    torch.manual_seed(13)
    f = build_cell_features(raw_signals(), [dict(text="eh", start_ms=100, end_ms=150)], 1500)
    model = CellFusionV2(mode="audio_text", d_model=16).eval()
    original = model(tensor_batch(f))
    assert original["type_logits"].shape == (3, 7) and original["context_logits"].shape == (3, 7, 2)
    shifted = deepcopy(f)
    shifted["lpos"][shifted["lmask"]] += 1
    assert not torch.allclose(original["type_logits"], model(tensor_batch(shifted))["type_logits"])
    shifted = deepcopy(f)
    shifted["ac_target"][:] = False
    shifted["ltarget"][:] = False
    result = model(tensor_batch(shifted))
    assert not result["input_evaluable"].any()
    assert not torch.allclose(original["type_logits"], result["type_logits"])


def test_masked_labels_never_change_loss_or_counts_and_adamw_skips_all_masked_batch():
    f = build_cell_features(raw_signals(), [], 1500)
    t = projected()
    model = CellFusionV2(d_model=16)
    output = model(tensor_batch(f))
    t["type_mask"][:, 1] = False
    t["context_mask"][:] = False
    changed = deepcopy(t)
    changed["multi_hot"][:, 1] = ~changed["multi_hot"][:, 1]
    changed["context_target"][:] = 99  # Out-of-class but representable in U4a int8.
    assert supervision_counts(t) == supervision_counts(changed)
    torch.testing.assert_close(masked_loss(output, t), masked_loss(output, changed))
    assert masked_metrics(output, t) == masked_metrics(output, changed)
    t["type_mask"][:] = False
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.2)
    assert train_step(model, optimizer, f, t)["stepped"] is False
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_build_train_strict_load_predict_and_partial_capability(tmp_path):
    file_path = synthetic_manifest(tmp_path)
    dataset = tmp_path / "cells"
    build_dataset(file_path, dataset)
    f, t, metadata = load_dataset(dataset)
    assert metadata["source_kind"] == "synthetic" and f["ac"].shape[0] == 3
    checkpoint = tmp_path / "cells.pt"
    report = train_cells(dataset, checkpoint, epochs=1, d_model=16, batch_size=2)
    assert report["optimizer_steps"] == 2 and not report["modalities"]["kin"]
    assert report["type_discrimination"][0] and report["context_discrimination"][0]
    assert not any(report["context_discrimination"][1:])
    loaded = CellInference.from_checkpoint(checkpoint)
    result = loaded.predict(f)
    assert result["type_probabilities"].shape == (3, 7)
    assert result["score_kind"] == "uncalibrated_sigmoid_conditional_softmax"
    assert np.isnan(result["type_probabilities"][:, 1:]).all()
    assert not loaded.capabilities["modalities"]["kin"]
    with pytest.raises(FileExistsError):
        train_cells(dataset, checkpoint, epochs=1)
    with pytest.raises(FileExistsError):
        build_dataset(file_path, dataset)
    with pytest.raises(FileNotFoundError):
        CellInference.from_checkpoint(tmp_path / "absent.pt")
    saved = torch.load(checkpoint, weights_only=True)
    saved["feature_contract"]["linguistic_interval"] = "legacy-start-centered"
    torch.save(saved, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="contract"):
        CellInference.from_checkpoint(tmp_path / "wrong.pt")
    with np.load(dataset / "arrays.npz", allow_pickle=False) as archive:
        assert all(archive[key].dtype != object for key in archive.files)


def test_all_masked_dataset_rejects_before_checkpoint_and_single_context_is_not_discrimination(tmp_path):
    manifest = synthetic_manifest(tmp_path, events=[], coverage=[])
    build_dataset(manifest, tmp_path / "unknown")
    checkpoint = tmp_path / "no.pt"
    with pytest.raises(ValueError, match="supervision"):
        train_cells(tmp_path / "unknown", checkpoint, epochs=1, d_model=16)
    assert not checkpoint.exists()
    manifest = synthetic_manifest(tmp_path, events=[human()])
    build_dataset(manifest, tmp_path / "single-state")
    report = train_cells(tmp_path / "single-state", tmp_path / "single.pt", epochs=1, d_model=16)
    assert report["context_state_counts"][0] == [0, 1]
    assert not report["context_discrimination"][0]


def test_cli_build_train_runs_only_on_explicit_synthetic_paths(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    root = Path(__file__).resolve().parents[1]
    for script, args in [("build_cells.py", ["--manifest", str(manifest), "--out", str(tmp_path / "cli-cells")]),
                         ("train_cells.py", ["--dataset", str(tmp_path / "cli-cells"), "--out", str(tmp_path / "cli.pt"), "--epochs", "1", "--d-model", "16"])]:
        result = subprocess.run([sys.executable, str(root / "scripts" / script), *args], cwd=tmp_path,
                                text=True, capture_output=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
    assert CellInference.from_checkpoint(tmp_path / "cli.pt").capabilities["optimizer_steps"] > 0


def test_strict_feature_contract_rejects_same_shape_legacy_and_unknown_fields():
    f = build_cell_features(raw_signals(), [], 1500)
    f["feature_contract"]["version"] = "1.2.0"
    with pytest.raises(ValueError, match="contract"):
        validate_cell_features(f)
    with pytest.raises(ValueError, match="live"):
        CellFusionV2(live=True)


@pytest.mark.parametrize("tamper", ["statistics", "missing_statistics", "counts", "modalities", "weights", "mode"])
def test_strict_loader_rejects_corrupt_same_dimension_envelopes(tmp_path, tamper):
    manifest = synthetic_manifest(tmp_path)
    build_dataset(manifest, tmp_path / "cells")
    train_cells(tmp_path / "cells", tmp_path / "good.pt", d_model=16)
    saved = torch.load(tmp_path / "good.pt", weights_only=True)
    if tamper == "statistics":
        saved["stats"]["ac"]["std"][0] = 0
    elif tamper == "missing_statistics":
        del saved["stats"]["kin"]
    elif tamper == "counts":
        saved["capabilities"]["type_positive_counts"][0] = -1
        saved["capabilities"]["type_discrimination"][0] = False
    elif tamper == "modalities":
        saved["capabilities"]["modalities"]["kin"] = True
    elif tamper == "weights":
        next(iter(saved["state_dict"].values())).fill_(float("nan"))
    else:
        saved["mode"] = "trimodal"
    torch.save(saved, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        CellInference.from_checkpoint(tmp_path / "bad.pt")


def test_context_only_and_all_empty_inputs_have_no_owned_eligibility_and_finite_logits():
    raw = raw_signals(1000)
    for key in ("ac_times", "ac_frames", "f0"):
        raw[key] = raw[key][75:]
    f = build_cell_features(raw, [], 1000)
    model = CellFusionV2(d_model=16).eval()
    result = model(tensor_batch(f))
    assert result["input_evaluable"].tolist() == [False, True]
    for key in ("ac", "pros", "ling", "kin"):
        f[key + "_mask"][:] = False
        f[key][:] = np.nan
    result = model(tensor_batch(f))
    assert not result["input_evaluable"].any() and torch.isfinite(result["type_logits"]).all()


def test_frame_at_duration_cannot_change_pitch_reference_prosody_or_normalization():
    raw = raw_signals(500)
    original = build_cell_features(raw, [], 500)
    raw["f0"][raw["ac_times"] >= .5] = 1e99
    raw["ac_frames"][raw["ac_times"] >= .5] = np.nan
    altered = build_cell_features(raw, [], 500)
    for key in ("ac", "ac_mask", "pros", "pros_mask"):
        np.testing.assert_array_equal(original[key], altered[key])


def test_training_stats_ignore_unselected_and_unusable_supervision_rows():
    f = build_cell_features(raw_signals(), [], 1500)
    t = projected()
    before = fit_cell_stats(f, t, indices=[0])
    t["type_mask"][1:] = False
    t["context_mask"][1:] = False
    f["ac"][1:] = 1e15
    after = fit_cell_stats(f, t, indices=[0, 1, 2])
    for field in ("mean", "std", "count"):
        np.testing.assert_array_equal(before["ac"][field], after["ac"][field])


def test_delta_does_not_claim_a_predecessor_outside_requested_evidence():
    f = build_cell_features(raw_signals(3250), [], 3250)
    first = np.flatnonzero(f["ac_support"][3])[0]
    assert not f["ac_mask"][3, first, 4]
    assert f["ac_mask"][3, first + 1, 4]


def test_multi_type_context_conflict_and_masked_metrics_are_preserved():
    events = [human(), human(110, 210, False), human(120, 220, True, "repetition")]
    targets = project_targets(500, events, [[0, 500]])
    assert targets["multi_hot"][0, [0, 2]].all()
    assert not targets["context_mask"][0, 0] and targets["context_mask"][0, 2]
    output = dict(type_logits=torch.zeros(1, 7), context_logits=torch.zeros(1, 7, 2), input_evaluable=torch.ones(1, dtype=torch.bool))
    before = masked_metrics(output, targets)
    targets["context_target"][0, 0] = 99
    assert masked_metrics(output, targets) == before


def test_dataset_rejects_tampered_arrays_and_revalidates_raw_target_provenance(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    build_dataset(manifest, tmp_path / "cells")
    sidecar = tmp_path / "cells/manifest.json"
    saved = json.loads(sidecar.read_text())
    original = deepcopy(saved)
    saved["recordings"][0]["targets"]["source_events"][0]["label"] = "repetition"
    sidecar.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "cells")
    with pytest.raises(ValueError):
        CellInference.from_checkpoint(tmp_path / "cells/arrays.npz")
    sidecar.write_text(json.dumps(original))
    with np.load(tmp_path / "cells/arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["ac"][0, 0, 0] += 1
    np.savez(tmp_path / "cells/arrays.npz", **arrays)
    with pytest.raises(ValueError, match="hash"):
        load_dataset(tmp_path / "cells")


@pytest.mark.parametrize("field,value", [("duration_ms", 1501), ("split", "selection"), ("source_identity", "")])
def test_dataset_load_rejects_inconsistent_retained_source_metadata(tmp_path, field, value):
    manifest = synthetic_manifest(tmp_path)
    build_dataset(manifest, tmp_path / "cells")
    sidecar = tmp_path / "cells/manifest.json"
    saved = json.loads(sidecar.read_text())
    saved["recordings"][0][field] = value
    sidecar.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "cells")


def test_persisted_dev_rows_never_fit_statistics_or_training_counts(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    source = json.loads(manifest.read_text())
    dev = deepcopy(source["recordings"][0])
    dev.update(recording_id="synthetic-dev", speaker_id="synthetic-other-speaker", split="dev", raw_npz="dev.npz")
    raw = raw_signals()
    raw["ac_frames"][:, 0] = 1e9
    np.savez(tmp_path / "dev.npz", **raw)
    source["recordings"].append(dev)
    manifest.write_text(json.dumps(source))
    build_dataset(manifest, tmp_path / "cells")
    report = train_cells(tmp_path / "cells", tmp_path / "model.pt", d_model=16)
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    assert checkpoint["stats"]["ac"]["mean"][0] == pytest.approx(-1)
    assert report["useful_training_rows"] == 3 and report["type_eligible"] == 21


def test_trimodal_metadata_never_activates_untrained_visual_weights(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    build_dataset(manifest, tmp_path / "cells")
    train_cells(tmp_path / "cells", tmp_path / "trimodal.pt", mode="trimodal", d_model=16)
    loaded = CellInference.from_checkpoint(tmp_path / "trimodal.pt")
    raw = raw_signals(3250)
    raw.update(k_times=np.arange(34) / 10, k_vectors=np.zeros((34, 12)), k_detected=np.ones(34, bool))
    first = loaded.predict(build_cell_features(raw, [], 3250))
    raw["k_vectors"][:] = 1e6
    changed = loaded.predict(build_cell_features(raw, [], 3250))
    np.testing.assert_array_equal(first["type_probabilities"], changed["type_probabilities"])
    assert not loaded.capabilities["modalities"]["kin"]


def test_synthetic_legacy_checkpoint_and_live_load_are_explicitly_rejected(tmp_path):
    torch.save(dict(state_dict={}, mode="audio_text", labels=["fluent"], dims=dict(ac=5), norm={}), tmp_path / "v1.pt")
    with pytest.raises(ValueError, match="contract"):
        CellInference.from_checkpoint(tmp_path / "v1.pt")
    with pytest.raises(ValueError, match="live"):
        CellInference.from_checkpoint(tmp_path / "missing.pt", live=True)


@pytest.mark.parametrize("edge,unavailable", [("left", list(range(8))), ("right", [0, 5]), ("both", list(range(8)))])
def test_censored_pause_anchors_mask_only_the_features_that_require_them(edge, unavailable):
    raw = raw_signals(3000)
    raw["ac_frames"][10:20, 3], raw["f0"][10:20] = 0, np.nan
    indices = [9] if edge == "left" else [20] if edge == "right" else [9, 20]
    raw["ac_frames"][indices, 3], raw["f0"][indices] = np.nan, np.nan
    f = build_cell_features(raw, [], 3000)
    if edge == "right":
        assert f["pros_mask"][0, [1, 2, 3, 4, 6, 7]].all()
    assert not f["pros_mask"][0, unavailable].any()


@pytest.mark.parametrize("tamper", ["target", "bounds", "position", "identity", "context"])
def test_retained_active_word_geometry_and_identity_cannot_invent_owned_input(tamper):
    raw = raw_signals(1000)
    raw.update(ac_times=np.zeros(0), ac_frames=np.zeros((0, 5)), f0=np.zeros(0))
    f = build_cell_features(raw, [dict(text="hola", start_ms=750, end_ms=800)], 1000)
    if tamper == "target":
        f["ltarget"][0, 0] = True
    elif tamper == "bounds":
        f["word_intervals"][0, 0] = [-100, 2000]
    elif tamper == "position":
        f["lpos"][0, 0] = 999
    elif tamper == "identity":
        f["word_indices"][0, 0] = -1
    else:
        # Consistent geometry, but the word lies outside the first cell's 10s context.
        f = build_cell_features(raw_signals(12000), [dict(text="x", start_ms=9000, end_ms=9100)], 12000)
        f["lmask"][0, 0] = True
        f["word_indices"][0, 0], f["word_intervals"][0, 0], f["lpos"][0, 0] = 0, [9000, 9100], 8.8
    with pytest.raises(ValueError):
        validate_cell_features(f)


@pytest.mark.parametrize("tamper", ["coverage", "words", "input_hashes", "manifest_hash", "word_identity"])
def test_portable_dataset_replays_retained_evidence_and_validates_source_hash_identifiers(tmp_path, tamper):
    manifest = synthetic_manifest(tmp_path)
    (tmp_path / "words.json").write_text(json.dumps([dict(text="hola", start_ms=100, end_ms=150, id="w0")]))
    build_dataset(manifest, tmp_path / "cells")
    sidecar = tmp_path / "cells/manifest.json"
    saved = json.loads(sidecar.read_text())
    rec = saved["recordings"][0]
    if tamper == "coverage":
        rec["reviewed_coverage"] = []
    elif tamper == "words":
        rec["words"] = [dict(text="x", start_ms=-10, end_ms=99999)]
    elif tamper == "input_hashes":
        del rec["input_sha256"]
    elif tamper == "manifest_hash":
        saved["input_manifest_sha256"] = "not-a-hash"
    else:
        rec["words"][0]["id"] = "different-identity-with-the-same-features"
    sidecar.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "cells")


@pytest.mark.parametrize("tamper", ["zero_steps", "zero_type_steps", "zero_context_steps", "context_without_positive", "count_exceeds_rows", "mode_and_provenance"])
def test_checkpoint_rejects_cross_field_supervision_and_mode_contradictions(tmp_path, tamper):
    manifest = synthetic_manifest(tmp_path)
    (tmp_path / "words.json").write_text(json.dumps([dict(text="eh", start_ms=100, end_ms=200, prob=.9)]))
    build_dataset(manifest, tmp_path / "cells")
    train_cells(tmp_path / "cells", tmp_path / "good.pt", d_model=16)
    saved = torch.load(tmp_path / "good.pt", weights_only=True)
    cap = saved["capabilities"]
    if tamper == "zero_steps":
        cap["type_optimizer_steps"] = cap["context_optimizer_steps"] = 0
    elif tamper in ("zero_type_steps", "zero_context_steps"):
        cap[tamper.removeprefix("zero_").removesuffix("_steps") + "_optimizer_steps"] = 0
    elif tamper == "context_without_positive":
        cap["context_state_counts"][1] = [1, 1]
        cap["context_eligible"] += 2
        cap["context_discrimination"][1] = True
    elif tamper == "count_exceeds_rows":
        cap["type_positive_counts"][0] = 1000000
        cap["type_eligible"] += 999998
    else:
        saved["mode"] = saved["provenance"]["mode"] = "audio_only"
    torch.save(saved, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        CellInference.from_checkpoint(tmp_path / "bad.pt")


def test_supported_checkpoint_predicts_empty_media_without_attention_reshape(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    build_dataset(manifest, tmp_path / "cells")
    train_cells(tmp_path / "cells", tmp_path / "good.pt", d_model=16)
    loaded = CellInference.from_checkpoint(tmp_path / "good.pt")
    result = loaded.predict(build_cell_features(raw_signals(0), [], 0))
    assert result["type_probabilities"].shape == (0, 7)
    assert result["context_probabilities"].shape == (0, 7, 2)
    assert result["input_evaluable"].shape == (0,) and result["unavailable_reason"] == []


@pytest.mark.parametrize("tamper", ["coverage", "words"])
def test_replay_rejects_inconsistent_arrays_even_if_retained_content_hash_is_recomputed(tmp_path, tamper):
    manifest = synthetic_manifest(tmp_path)
    (tmp_path / "words.json").write_text(json.dumps([dict(text="hola", start_ms=100, end_ms=150)]))
    build_dataset(manifest, tmp_path / "cells")
    sidecar = tmp_path / "cells/manifest.json"
    saved = json.loads(sidecar.read_text())
    rec = saved["recordings"][0]
    if tamper == "coverage":
        rec["reviewed_coverage"] = []
        rec["retained_input_sha256"]["reviewed_coverage"] = content_digest([])
    else:
        rec["words"][0].update(start_ms=750, end_ms=800)
        rec["retained_input_sha256"]["words"] = content_digest(rec["words"])
    sidecar.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        load_dataset(tmp_path / "cells")


def test_portable_dataset_does_not_require_original_raw_json_or_manifest_files(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    build_dataset(manifest, tmp_path / "cells")
    for file_name in ("input.json", "raw.npz", "words.json", "events.json"):
        (tmp_path / file_name).unlink()
    f, t, metadata = load_dataset(tmp_path / "cells")
    assert len(f["ac"]) == 3 and t["type_mask"].sum() == 21
    assert metadata["source_kind"] == "synthetic"


def test_neutral_masked_word_storage_is_not_an_active_word_and_owned_rule_has_priority():
    f = build_cell_features(raw_signals(500), [], 500)
    f["word_indices"][:] = 0
    f["ling"][:] = np.nan
    assert validate_cell_features(f) == 1
    long_word = build_cell_features(raw_signals(12000), [dict(text="x", start_ms=0, end_ms=12000)], 12000)
    assert long_word["ltarget"][0, 0] and validate_cell_features(long_word) == 24


def test_context_probabilities_require_supported_type_and_empty_training_never_writes(tmp_path):
    manifest = synthetic_manifest(tmp_path, duration=1000)
    build_dataset(manifest, tmp_path / "two-positives")
    report = train_cells(tmp_path / "two-positives", tmp_path / "known.pt", d_model=16)
    assert report["context_discrimination"][0] and not report["type_discrimination"][0]
    features, _, _ = load_dataset(tmp_path / "two-positives")
    result = CellInference.from_checkpoint(tmp_path / "known.pt").predict(features)
    assert not result["context_supported"][0] and np.isnan(result["context_probabilities"][:, 0]).all()
    manifest = synthetic_manifest(tmp_path, duration=0, events=[], coverage=[])
    build_dataset(manifest, tmp_path / "empty")
    with pytest.raises(ValueError, match="supervision"):
        train_cells(tmp_path / "empty", tmp_path / "not-created.pt", d_model=16)
    assert not (tmp_path / "not-created.pt").exists()
