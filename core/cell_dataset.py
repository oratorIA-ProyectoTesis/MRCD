"""Explicit NPZ+JSON v2 datasets from raw signals/original intervals, never old y."""
from copy import deepcopy
from hashlib import sha256
import json
import re
from pathlib import Path
import numpy as np
from core.cell_contracts import SHAPES, cell_feature_contract, validate_cell_features
from core.features_v2 import build_cell_features
from core.targets import project_targets, validate_targets

DATASET_VERSION = "mrcd-cell-dataset-v2"
TARGET_ARRAYS = ("multi_hot", "type_mask", "fully_reviewed", "type_ambiguity", "softmax_target", "softmax_mask",
                 "context_multi_hot", "context_conflict", "context_target", "context_mask")


def _source_metadata(rec):
    if not isinstance(rec, dict) or any(not isinstance(rec.get(key), str) or not rec[key].strip()
                                       for key in ("recording_id", "speaker_id", "source_identity")):
        raise ValueError("explicit recording/speaker/source identities required")
    if rec.get("split") not in ("train", "dev", "test"):
        raise ValueError("invalid persisted source split")
    if type(rec.get("duration_ms")) is not int or rec["duration_ms"] < 0:
        raise ValueError("invalid authoritative source duration_ms")


def digest(file_path):
    return sha256(Path(file_path).read_bytes()).hexdigest()


def content_digest(value):
    """Portable self-consistency binding, not an external authenticity signature."""
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _retained_bindings(words, events, coverage):
    return {key: content_digest(value) for key, value in
            (("words", words), ("events", events), ("reviewed_coverage", coverage))}


def _validate_portable_record(rec):
    _source_metadata(rec)
    if any(not isinstance(rec.get(key), str) or not rec[key].strip() for key in ("raw_npz", "words_json", "events_json")):
        raise ValueError("retained explicit input file references missing")
    hashes = rec.get("input_sha256")
    if (not isinstance(hashes, dict) or set(hashes) != {"raw_npz", "words_json", "events_json"}
            or any(not _is_hash(value) for value in hashes.values())):
        raise ValueError("retained input file SHA-256 identifiers missing or malformed")
    if (not isinstance(rec.get("words"), list) or any(not isinstance(word, dict) for word in rec["words"])
            or not isinstance(rec.get("reviewed_coverage"), list) or not isinstance(rec.get("targets"), dict)
            or not isinstance(rec["targets"].get("source_events"), list)):
        raise ValueError("retained words/coverage/original-event schema malformed")
    expected = _retained_bindings(rec["words"], rec["targets"]["source_events"], rec["reviewed_coverage"])
    if rec.get("retained_input_sha256") != expected:
        raise ValueError("retained input content binding mismatch; rebuild at a new path")


def build_dataset(manifest_path, out_dir):
    manifest_path, out_dir = Path(manifest_path), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    source = json.loads(manifest_path.read_text())
    if source.get("source_kind") not in ("synthetic", "human") or not source.get("recordings"):
        raise ValueError("explicit synthetic/human source_kind and recordings required")
    feature_rows, target_rows, records, offset = [], [], [], 0
    seen = set()
    for rec in source["recordings"]:
        _source_metadata(rec)
        for field in ("recording_id", "speaker_id", "split", "duration_ms", "source_identity", "raw_npz", "words_json", "events_json", "reviewed_coverage"):
            if field not in rec:
                raise ValueError(f"manifest recording missing {field}")
        if not rec["speaker_id"] or not rec["source_identity"] or rec["split"] not in ("train", "dev", "test"):
            raise ValueError("explicit speaker/source identity and persisted split required")
        if rec["recording_id"] in seen:
            raise ValueError("duplicate recording_id")
        seen.add(rec["recording_id"])
        paths = {key: manifest_path.parent / rec[key] for key in ("raw_npz", "words_json", "events_json")}
        with np.load(paths["raw_npz"], allow_pickle=False) as archive:
            raw = {key: archive[key] for key in ("ac_times", "ac_frames", "f0", "k_times", "k_vectors", "k_detected")}
        words = json.loads(paths["words_json"].read_text())
        events = json.loads(paths["events_json"].read_text())
        features = build_cell_features(raw, words, rec["duration_ms"])
        targets = project_targets(rec["duration_ms"], events, rec["reviewed_coverage"])
        validate_targets(targets)
        n = validate_cell_features(features)
        metadata = deepcopy(rec)
        metadata.update(offset=offset, n_cells=n, words=words,
                        input_sha256={key: digest(value) for key, value in paths.items()},
                        retained_input_sha256=_retained_bindings(words, events, rec["reviewed_coverage"]),
                        targets={key: value for key, value in targets.items() if key not in TARGET_ARRAYS})
        records.append(metadata)
        feature_rows.append(features)
        target_rows.append(targets)
        offset += n
    arrays = {key: np.concatenate([row[key] for row in feature_rows]) for key in SHAPES}
    arrays.update({"target_" + key: np.concatenate([row[key] for row in target_rows]) for key in TARGET_ARRAYS})
    metadata = dict(dataset_version=DATASET_VERSION, feature_contract=cell_feature_contract(), source_kind=source["source_kind"],
                    input_manifest_sha256=digest(manifest_path), recordings=records)
    json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
    # Validate all inputs before creating output; existing directories are refused.
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(out_dir / "arrays.npz", **arrays)
        metadata["arrays_sha256"] = digest(out_dir / "arrays.npz")
        (out_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except Exception:
        for child in out_dir.iterdir():
            child.unlink()
        out_dir.rmdir()
        raise
    return metadata


def load_dataset(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if (metadata.get("dataset_version") != DATASET_VERSION or metadata.get("feature_contract") != cell_feature_contract()
            or metadata.get("source_kind") not in ("synthetic", "human")):
        raise ValueError("v2 dataset contract mismatch; rebuild from explicit original evidence")
    if (not _is_hash(metadata.get("input_manifest_sha256")) or not _is_hash(metadata.get("arrays_sha256"))
            or not isinstance(metadata.get("recordings"), list) or not metadata["recordings"]):
        raise ValueError("dataset source manifest/hash schema malformed")
    if digest(directory / "arrays.npz") != metadata.get("arrays_sha256"):
        raise ValueError("v2 dataset arrays hash mismatch")
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        features = {key: archive[key] for key in SHAPES}
        targets = {key: archive["target_" + key] for key in TARGET_ARRAYS}
    features["feature_contract"] = cell_feature_contract()
    n, offset = validate_cell_features(features), 0
    seen = set()
    for rec in metadata["recordings"]:
        _validate_portable_record(rec)
        if rec["recording_id"] in seen:
            raise ValueError("duplicate retained recording identity")
        seen.add(rec["recording_id"])
        if (type(rec.get("offset")) is not int or type(rec.get("n_cells")) is not int
                or rec["offset"] != offset or rec["n_cells"] < 0):
            raise ValueError("v2 dataset row alignment mismatch")
        end = offset + rec["n_cells"]
        view = dict(rec["targets"], **{key: value[offset:end] for key, value in targets.items()})
        if rec["duration_ms"] != view["media_duration_ms"]:
            raise ValueError("retained source and targets disagree on actual duration")
        validate_targets(view)  # Revalidate U4a from retained original intervals/coverage.
        replay = project_targets(rec["duration_ms"], view["source_events"], rec["reviewed_coverage"])
        if replay["reviewed_coverage"] != view["reviewed_coverage"] or any(
                not np.array_equal(replay[key], view[key]) for key in TARGET_ARRAYS):
            raise ValueError("outer reviewed coverage and retained target evidence disagree")
        empty_raw = dict(ac_times=np.zeros(0), ac_frames=np.zeros((0, 5)), f0=np.zeros(0),
                         k_times=np.zeros(0), k_vectors=np.zeros((0, 12)), k_detected=np.zeros(0, bool))
        word_replay = build_cell_features(empty_raw, rec["words"], rec["duration_ms"])
        # Replay word membership/truncation/identity and observed measurements;
        # do not demand a particular value in unobserved/padded numeric storage.
        for key in ("lmask", "ling_mask", "ltarget", "word_source_count"):
            if not np.array_equal(features[key][offset:end], word_replay[key]):
                raise ValueError(f"retained words disagree with {key}")
        for key in ("word_indices", "word_intervals", "lpos", "ling"):
            observed = word_replay["ling_mask"] if key == "ling" else word_replay["lmask"]
            if not np.allclose(features[key][offset:end][observed], word_replay[key][observed], rtol=0, atol=1e-7):
                raise ValueError(f"retained word identity/measurement binding mismatch: {key}")
        for key in ("media_duration_ms", "owned_start_ms", "owned_end_ms", "nominal_center_ms"):
            expected = [cell[key] for cell in view["cells"]]
            if not np.array_equal(features[key][offset:end], expected):
                raise ValueError("v2 features/targets temporal alignment mismatch")
        offset = end
    if offset != n:
        raise ValueError("v2 dataset rows not accounted for by sources")
    return features, targets, metadata
