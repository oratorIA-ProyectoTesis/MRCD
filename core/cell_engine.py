"""Offline product adapter: one global feature/grid build, segment-owned inference."""
from __future__ import annotations

from copy import deepcopy
import time
import numpy as np

from core.cell_contracts import cell_feature_contract
from core.cell_decode import (CELL_RESULT_SCHEMA, SCORE_KIND, threshold_policy, partition,
                              decode_cells, ledger_coverage, validate_cell_result)
from core.features_v2 import build_cell_features
from core.temporal import decision_cells, evidence_windows, integer_ms, TEMPORAL_CONTRACT


def raw_from_prepared(prepared, analysis):
    samples, sr, duration = (integer_ms(analysis.get(k), k) for k in ("samples", "sample_rate", "duration_ms"))
    if (samples < 0 or sr != 16000 or duration != round(samples * 1000 / sr)
            or not np.isclose(prepared.duration_s, samples / sr, rtol=0, atol=1e-9)
            or analysis.get("origin_ms", 0) != 0):
        raise ValueError("whole-recording samples/rate/duration/global origin disagree")
    track = prepared.track
    return dict(raw=dict(ac_times=np.asarray(track.times), ac_frames=track.frame_matrix(), f0=np.asarray(track.f0),
                         k_times=np.zeros(0), k_vectors=np.zeros((0, 12)), k_detected=np.zeros(0, bool)),
                words=[w.to_dict() for w in prepared.words], duration_ms=duration,
                visual_reason="product_visual_acquisition_not_integrated")


def assign_cells_to_segments(cells, segments, duration_ms):
    return partition(cells, segments, duration_ms)


def _select(features, indices):
    return {key: value[indices] if isinstance(value, np.ndarray) else value for key, value in features.items()}


class CellAnalysisEngine:
    def __init__(self, inference, *, threshold_policy=None):
        self.inference = inference
        self.policy = globals()["threshold_policy"](threshold_policy)

    def analyze(self, prepared, *, analysis, recording_id, segments, batch_size=128, stage=lambda name: None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        started = time.perf_counter()
        source = raw_from_prepared(prepared, analysis)
        duration = source["duration_ms"]
        grid = evidence_windows(decision_cells(duration))
        assignments = partition(grid, segments, duration)
        stage("cell_features_global")
        features = build_cell_features(source["raw"], source["words"], duration)
        capabilities = deepcopy(self.inference.capabilities)
        owners = {i: sid for sid, indices in assignments.items() for i in indices}
        cells = []
        for i, owned in enumerate(grid):
            support = dict(acoustic_frames=int(features["ac_support"][i].sum()),
                           owned_acoustic_frames=int(features["ac_target"][i].sum()),
                           word_indices=features["word_indices"][i][features["lmask"][i]].tolist(),
                           real_visual_frames=0, visual_reason=source["visual_reason"],
                           evidence_start_ms=owned["observed_start_ms"], evidence_end_ms=owned["observed_end_ms"])
            cells.append(dict(owned, cell_id=f"{recording_id}:cell:{i}", recording_id=recording_id, segment_id=owners[i],
                              status="failed", input_evaluable=False, available=False, failure_reason="not_executed",
                              type_probabilities=[None] * 7, context_probabilities=[[None, None] for _ in range(7)],
                              type_decisions=["unavailable"] * 7, contextual_states=[None] * 7,
                              type_reasons=["not_executed"] * 7, context_reasons=["not_executed"] * 7,
                              capabilities=deepcopy(capabilities), support_refs=support,
                              quality=dict(acoustic=bool(features["ac_mask"][i].any()), linguistic=bool(features["lmask"][i].any()), visual=False)))
        # Segment ownership only schedules batches; context never changes with batching.
        for sid, indices in assignments.items():
            for offset in range(0, len(indices), batch_size):
                batch = indices[offset:offset + batch_size]
                stage(f"cell_inference:{sid}:{offset}")  # cancellation must propagate
                try:
                    output = self.inference.predict(_select(features, batch))
                except RuntimeError as exc:
                    for i in batch:
                        cells[i]["failure_reason"] = f"inference_failed:{str(exc)[:200]}"
                        cells[i]["type_reasons"] = ["inference_failed"] * 7
                        cells[i]["context_reasons"] = ["inference_failed"] * 7
                    continue
                types, contexts = np.asarray(output["type_probabilities"]), np.asarray(output["context_probabilities"])
                available = np.asarray(output["input_evaluable"])
                ts, cs = np.asarray(output["type_supported"]), np.asarray(output["context_supported"])
                if (types.shape != (len(batch), 7) or contexts.shape != (len(batch), 7, 2)
                        or available.shape != (len(batch),) or available.dtype != bool or ts.shape != (7,) or cs.shape != (7,)
                        or ts.dtype != bool or cs.dtype != bool or np.any(cs & ~ts)
                        or ts.tolist() != capabilities["type_discrimination"] or cs.tolist() != capabilities["context_discrimination"]):
                    raise ValueError("inference output/capability contract mismatch")
                for row, i in enumerate(batch):
                    cell = cells[i]
                    cell.update(status="succeeded", failure_reason=None, input_evaluable=bool(available[row]), available=bool(available[row]))
                    for k in range(7):
                        type_ok, context_ok = bool(available[row] and ts[k]), bool(available[row] and cs[k])
                        p, cp = types[row, k], contexts[row, k]
                        if type_ok and (not np.isfinite(p) or not 0 <= p <= 1):
                            raise ValueError("supported type returned invalid score")
                        if context_ok and (not np.isfinite(cp).all() or np.any((cp < 0) | (cp > 1)) or not np.isclose(cp.sum(), 1., atol=1e-5)):
                            raise ValueError("supported context returned invalid probabilities")
                        cell["type_probabilities"][k] = float(p) if type_ok else None
                        cell["context_probabilities"][k] = [float(v) for v in cp] if context_ok else [None, None]
                        cell["type_decisions"][k] = ("positive" if p >= self.policy["positive"] else "negative" if p <= self.policy["negative"] else "abstained") if type_ok else "unavailable"
                        cell["contextual_states"][k] = ("uncertain" if max(cp) < self.policy["context"] else "disfluent" if cp[1] >= cp[0] else "functional") if context_ok else None
                        cell["type_reasons"][k] = None if type_ok else "no_owned_input" if not available[row] else "type_head_unavailable"
                        cell["context_reasons"][k] = None if context_ok else "no_owned_input" if not available[row] else "context_head_unavailable"
        statuses = {sid: "succeeded" if all(cells[i]["status"] == "succeeded" for i in indices)
                    else "partial" if any(cells[i]["status"] == "succeeded" for i in indices) else "failed"
                    for sid, indices in assignments.items()}
        result = dict(result_schema=CELL_RESULT_SCHEMA, prediction_kind="cell_type_context", recording_id=recording_id,
                      duration_ms=duration, temporal_contract=TEMPORAL_CONTRACT, cells=cells, execution_cores=deepcopy(segments),
                      segments=statuses, coverage=ledger_coverage(cells), threshold_policy=dict(self.policy),
                      events=decode_cells(cells, recording_id=recording_id), words=source["words"],
                      capabilities=capabilities, score_kind=SCORE_KIND, feature_contract=cell_feature_contract(),
                      visual_reason=source["visual_reason"], timings=dict(prepared.timings, cell_analysis_s=time.perf_counter() - started),
                      warning="Uncalibrated cell projection; synthetic supervision is not human validation. HU059 refinement not implemented.")
        validate_cell_result(result)
        return result
