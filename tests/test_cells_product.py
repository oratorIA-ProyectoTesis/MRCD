"""U4c integration: synthetic prepared signals/checkpoints, no ASR or services."""
from copy import deepcopy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from core.cell_engine import CellAnalysisEngine, raw_from_prepared, assign_cells_to_segments
from core.cell_decode import CELL_RESULT_SCHEMA, validate_cell_result, decode_cells, ledger_coverage
from core.legacy_windows import legacy_window_predictions
from core.cell_dataset import build_dataset
from core.training_v2 import train_cells
from core.cell_inference import CellInference
from core.engine import PreparedRecording, MRCDEngine
from core.extractors.acoustic import AcousticTrack
from core.extractors.linguistic import Word
from core.temporal import decision_cells
from core.constants import LABELS
from core import evaluation
from app import systems, worker, exports
from app.api import create_app
from app.store import Store


def prep(duration):
    times = np.arange(0, duration + 10, 10) / 1000
    n = len(times)
    track = AcousticTrack(times, np.full(n, .1), np.full(n, 100.), np.full(n, .9), np.ones(n))
    return PreparedRecording(track, [], duration / 1000, dict(asr_s=0., acoustic_s=0., cache_hit=True))


def analysis(duration):
    return dict(samples=duration * 16, sample_rate=16000, duration_ms=duration, sha256="a" * 64)


def cores(duration=6250):
    return [dict(id=f"s{i}", core_start_ms=a, core_end_ms=b) for i, (a, b) in enumerate([(0, 2300), (2300, 4600), (4600, duration)])]


class FakeCellInference:
    mode = "audio_text"
    capabilities = dict(type_discrimination=[True, False, True, False, False, False, False],
                        context_discrimination=[True, False, True, False, False, False, False],
                        modalities=dict(ac=True, pros=True, ling=False, kin=False), source_kind="synthetic")
    provenance = dict(implementation="synthetic-fake")

    def __init__(self, failure=None):
        self.failure = failure

    def predict(self, features):
        if self.failure and np.any(features["owned_start_ms"] == self.failure):
            raise RuntimeError("synthetic recoverable inference failure")
        n = len(features["ac"])
        types, contexts = np.full((n, 7), np.nan), np.full((n, 7, 2), np.nan)
        types[:, [0, 2]] = .9
        contexts[:, [0, 2]] = [.1, .9]
        return dict(type_probabilities=types, context_probabilities=contexts, input_evaluable=np.ones(n, bool),
                    type_supported=np.asarray(self.capabilities["type_discrimination"]),
                    context_supported=np.asarray(self.capabilities["context_discrimination"]))


def result(inference=None, batch_size=128, duration=6250):
    return CellAnalysisEngine(inference or FakeCellInference()).analyze(
        prep(duration), analysis=analysis(duration), recording_id="r", segments=cores(duration) if duration > 4600 else
        ([dict(id="s0", core_start_ms=0, core_end_ms=duration)] if duration else []), batch_size=batch_size)


def test_global_grid_misaligned_cores_and_actual_tail_midpoint_partial_coverage():
    assignments = assign_cells_to_segments(decision_cells(6250), cores(), 6250)
    assert [len(assignments[s["id"]]) for s in cores()] == [5, 4, 4]
    res = result(FakeCellInference(failure=2500))
    assert res["result_schema"] == CELL_RESULT_SCHEMA == "mrcd-cell-result-v2"
    assert len(res["cells"]) == 13
    assert res["coverage"] == [[0, 2500], [4500, 6250]]
    last = res["cells"][-1]
    assert [last[k] for k in ("owned_start_ms", "owned_end_ms", "nominal_center_ms", "requested_start_ms", "requested_end_ms", "observed_start_ms", "observed_end_ms")] == [6000, 6250, 6250, 4750, 7750, 4750, 6250]
    run = dict(parent="r", status="partial", data=dict(result=res))
    assert evaluation.run_coverage(run, cores())["available_ms"] == 4250
    assert validate_cell_result(res) == 13


@pytest.mark.parametrize("tamper", ["gap", "overlap", "duplicate"])
def test_cores_must_partition_actual_duration(tamper):
    segments = cores()
    if tamper == "gap": segments[1]["core_start_ms"] += 1
    elif tamper == "overlap": segments[1]["core_start_ms"] -= 1
    else: segments[1]["id"] = segments[0]["id"]
    with pytest.raises(ValueError): assign_cells_to_segments(decision_cells(6250), segments, 6250)


@pytest.mark.parametrize("tamper", ["duplicate", "missing", "claimed", "availability", "clock"])
def test_exclusive_v2_coverage_rejects_inconsistent_complete_ledger(tamper):
    res = result()
    if tamper == "duplicate": res["cells"][1] = deepcopy(res["cells"][0])
    elif tamper == "missing": res["cells"].pop()
    elif tamper == "claimed": res["coverage"] = [[0, 6000]]
    elif tamper == "availability": res["cells"][0]["available"] = False
    else: res["cells"][-1]["observed_end_ms"] = 7750
    with pytest.raises(ValueError): evaluation.run_coverage(dict(parent="r", status="succeeded", data=dict(result=res)), cores())


@pytest.mark.parametrize("duration", [0, 125, 500, 501, 6250])
def test_short_tail_zero_and_batch_parity(duration):
    a, b = result(batch_size=1, duration=duration), result(batch_size=99, duration=duration)
    assert a["coverage"] == b["coverage"] == ([[0, duration]] if duration else [])
    assert a["events"] == b["events"]
    assert len(a["cells"]) == (duration + 499) // 500
    json.dumps(a, allow_nan=False)


def test_prepared_clock_and_duration_are_global_seconds_not_frame_row_count():
    p = prep(6250)
    p.words = [Word("eh", "eh", 2., 2.1, .9, "")]
    raw = raw_from_prepared(p, analysis(6250))
    assert raw["raw"]["ac_times"][1] == .01 and raw["words"][0]["start"] == 2.
    bad = analysis(6250); bad["samples"] -= 16
    with pytest.raises(ValueError): raw_from_prepared(p, bad)
    bad = analysis(6250); bad["origin_ms"] = 1100
    with pytest.raises(ValueError): raw_from_prepared(p, bad)


def test_global_decode_multilabel_context_gap_nulls_abstention_and_cancel():
    res = result()
    cells = deepcopy(res["cells"])
    cells[2]["contextual_states"][0] = None
    cells[2]["context_reasons"][0] = "context_head_unavailable"
    events = decode_cells(cells, recording_id="r")
    filler = [event for event in events if event["label"] == "filler_word"]
    assert len(filler) == 3 and filler[1]["is_disfluent"] is None
    assert len([e for e in events if e["label"] == "repetition"]) == 1
    assert all(event["boundary_resolution_ms"] == 500 for event in events)
    assert cells[0]["type_probabilities"][1] is None
    strict = CellAnalysisEngine(FakeCellInference(), threshold_policy=dict(version="cell-threshold-v1", positive=.99, negative=.2, context=.99, calibrated=False))
    abstained = strict.analyze(prep(6250), analysis=analysis(6250), recording_id="r", segments=cores())
    assert abstained["coverage"] == [[0, 6250]]
    assert all(e["type_decision"] == "abstained" and e["is_disfluent"] is None for e in abstained["events"])
    def cancelled(name): raise worker.Cancelled
    with pytest.raises(worker.Cancelled):
        CellAnalysisEngine(FakeCellInference()).analyze(prep(6250), analysis=analysis(6250), recording_id="r", segments=cores(), stage=cancelled)


@pytest.fixture
def checkpoint(tmp_path):
    from tests.test_cells_v2 import synthetic_manifest
    manifest = synthetic_manifest(tmp_path)
    build_dataset(manifest, tmp_path / "cells")
    train_cells(tmp_path / "cells", tmp_path / "model.pt", d_model=16)
    return tmp_path / "model.pt"


def test_worker_api_dispatch_persisted_kind_strict_json_and_unsafe_consumer_guards(tmp_path, monkeypatch, checkpoint):
    monkeypatch.setenv("MRCD_DATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(systems, "_loaded", {})
    calls = []
    monkeypatch.setattr(systems, "prepared", lambda *args: calls.append(1) or prep(6250))
    monkeypatch.setattr(worker.media, "load", lambda _: np.zeros(100000, np.float32))
    store = Store(f"sqlite:///{tmp_path / 'app.db'}")
    user = store.create("user", dict(name="researcher", role="researcher", token_sha256=hashlib.sha256(b"synthetic-token").hexdigest()), status="active")
    rec = store.create("recording", dict(sha256="b" * 64, speaker_id="s", media=dict(analysis=analysis(6250), original=dict(has_video=False)), segments=cores()), owner=user["id"], status="ready")
    client = TestClient(create_app(store))
    headers = dict(Authorization="Bearer synthetic-token")
    response = client.post("/api/analysis-runs", headers=headers, json=dict(recording_id=rec["id"], config=dict(engine_kind="cell_v2", checkpoint=str(checkpoint))))
    assert response.status_code == 202, response.text
    run_id = response.json()["runs"][0]["id"]
    assert store.get(run_id)["data"]["config"]["engine_kind"] == "cell_v2"
    worker.execute(store, store.get(run_id), systems.SYSTEMS)
    run = store.get(run_id)
    assert run["status"] == "succeeded", run["data"].get("error")
    assert run["data"]["result"]["result_schema"] == CELL_RESULT_SCHEMA and len(calls) == 1
    exported = client.get(f"/api/exports/{run_id}", headers=headers)
    assert exported.status_code == 200 and "NaN" not in exported.text
    assert client.get(f"/api/exports/{run_id}?format=csv", headers=headers).status_code == 422
    assert client.get(f"/api/exports/{run_id}?format=eaf", headers=headers).status_code == 422
    report = client.get(f"/api/exports/{run_id}?format=md", headers=headers)
    assert report.status_code == 200 and "context" in report.text.lower()
    with pytest.raises(ValueError, match="task"):
        evaluation.evaluate_runs({}, {"mrcd": [run]}, {})


def test_checkpoint_cache_uses_actual_loaded_content_not_only_path(tmp_path, monkeypatch, checkpoint):
    monkeypatch.setattr(systems, "_loaded", {})
    monkeypatch.setattr(systems, "prepared", lambda *args: prep(6250))
    job = worker.Job("run", dict(id="r", data=dict(media=dict(analysis=analysis(6250)), segments=cores())), {},
                     dict(engine_kind="cell_v2", checkpoint=str(checkpoint)), np.zeros(100000), tmp_path, lambda _: None)
    first = systems.mrcd(job)
    state = torch.load(checkpoint, weights_only=True)
    state["state_dict"]["type_head.bias"][0] += 2
    torch.save(state, checkpoint)
    second = systems.mrcd(job)
    assert first["model_version"]["checkpoint_sha256"] != second["model_version"]["checkpoint_sha256"]
    assert first["cells"][0]["type_probabilities"][0] != second["cells"][0]["type_probabilities"][0]
    job.config["checkpoint"] = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError): systems.mrcd(job)
    job.config["live"] = True
    with pytest.raises(ValueError, match="live"): systems.mrcd(job)


def test_shared_legacy_support_is_not_fake_stride_localization_or_event_truth():
    probabilities = np.zeros((2, len(LABELS)))
    probabilities[:, LABELS.index("filler_word")] = 1
    groups = legacy_window_predictions(probabilities, [0, .5], list(LABELS), duration_ms=3500)
    assert groups[0]["start_ms"] == 0 and groups[0]["end_ms"] == 3500
    assert [[w["start_ms"], w["end_ms"]] for w in groups[0]["windows"]] == [[0, 3000], [500, 3500]]
    assert groups[0]["prediction_kind"] == "window_classification"
    from scripts.infer_video import merge_events
    assert merge_events(np.array([0, .5]), probabilities.argmax(1), probabilities, list(LABELS), duration_ms=3500) == groups
    net = SimpleNamespace(mode="audio_only", ling=None, pros=None)
    inference = SimpleNamespace(net=net, labels=list(LABELS), provenance={}, device="cpu",
                                predict=lambda data: np.tile(probabilities[0], (len(data["ac"]), 1)))
    output = MRCDEngine(inference, None).analyze(np.zeros(16000 * 20), prepared=prep(20000))
    assert output["prediction_kind"] == "window_classification"
    assert output["events"][0]["end_ms"] == 20000
    run = dict(parent="r", status="succeeded", data=dict(result=output))
    available = evaluation.run_coverage(run, [dict(id="s0", core_start_ms=0, core_end_ms=20000)])
    assert available["available_ms"] is None and "legacy" in available["reasons"][0]
    with pytest.raises(ValueError): exports.eaf(groups, "synthetic.wav", "test")


@pytest.mark.parametrize("tamper", ["order", "gap", "arrays", "context_nan", "capability", "schema", "events"])
def test_ledger_and_decoder_reject_corrupt_semantics(tamper):
    res = result()
    if tamper == "order": res["cells"][0], res["cells"][1] = res["cells"][1], res["cells"][0]
    elif tamper == "gap": res["cells"][1]["owned_start_ms"] += 1
    elif tamper == "arrays": res["cells"][0]["type_decisions"].pop()
    elif tamper == "context_nan": res["cells"][0]["context_probabilities"][0][0] = float("nan")
    elif tamper == "capability": res["cells"][0]["capabilities"]["type_discrimination"][0] = False
    elif tamper == "schema": res["temporal_contract"] = "invented"
    else: res["events"][0]["is_disfluent"] = False
    with pytest.raises(ValueError): validate_cell_result(res)
    if tamper in ("order", "gap", "arrays"):
        with pytest.raises(ValueError): decode_cells(res["cells"], recording_id="r")


def test_no_owned_input_and_unsupported_types_remain_distinct_from_negative_or_threshold_abstention():
    class Missing(FakeCellInference):
        def predict(self, features):
            out = super().predict(features)
            out["input_evaluable"][:] = False
            out["type_probabilities"][:] = np.nan
            out["context_probabilities"][:] = np.nan
            return out
    res = result(Missing())
    assert res["coverage"] == [] and res["events"] == []
    assert res["cells"][0]["type_reasons"][0] == "no_owned_input"
    assert result()["cells"][0]["type_reasons"][1] == "type_head_unavailable"
    assert res["cells"][0]["status"] == "succeeded"


def test_decoder_does_not_bridge_negative_failed_or_functional_cells():
    cells = result()["cells"]
    cells[2]["type_decisions"][0] = "negative"
    cells[4].update(status="failed", input_evaluable=False, available=False)
    cells[6]["contextual_states"][0] = "functional"
    events = [e for e in decode_cells(cells, recording_id="r") if e["label"] == "filler_word"]
    assert [(e["start_ms"], e["end_ms"]) for e in events] == [(0, 1000), (1500, 2000), (2500, 3000), (3000, 3500), (3500, 6250)]
    assert events[3]["is_disfluent"] is False


def test_product_summary_renderer_distinguishes_context_unknown_functional_and_legacy():
    source = (Path(__file__).parents[1] / "app/static/app.js").read_text()
    function = "function resultSemantics(res) {" + source.split("function resultSemantics(res) {", 1)[1].split("// END RESULT SEMANTICS", 1)[0]
    data = result()
    data["events"][0]["is_disfluent"] = None
    data["events"][0]["contextual_state"] = None
    js = "const PAUSES=new Set(['rhetorical_pause','neutral_pause']);\n" + function
    outputs = json.loads(subprocess.run(["node", "-e", js + "\nconsole.log(JSON.stringify([resultSemantics(" + json.dumps(data) + "),resultSemantics({prediction_kind:'window_classification',events:[{label:'filler_word',decision:'candidate'}]})]));"], capture_output=True, text=True, check=True).stdout)
    assert outputs[0]["dis"] == 1 and outputs[0]["unavailable"] == 1 and outputs[0]["guarded"]
    assert outputs[1]["dis"] is None and outputs[1]["candidates"] == 1 and outputs[1]["guarded"]
    assert outputs[0]["unavailable_types"] == 5 and outputs[0]["unavailable_contexts"] == 5


@pytest.mark.parametrize("marker", ["unknown", "missing"])
def test_unknown_or_missing_cell_schema_cannot_fall_back_to_legacy_cores(marker):
    res = result()
    if marker == "unknown": res["result_schema"] = "mrcd-cell-result-v99"
    else: res.pop("result_schema")
    with pytest.raises(ValueError): evaluation.run_coverage(dict(parent="r", status="succeeded", data=dict(result=res)), cores())


@pytest.mark.parametrize("bad", ["sum", "negative", "nan", "reverse", "duration"])
def test_legacy_shared_decoder_rejects_invalid_distributions_and_support(bad):
    p = np.full((2, len(LABELS)), 1 / len(LABELS))
    starts, duration = [0., .5], 3500
    if bad == "sum": p *= .8
    elif bad == "negative": p[0, 0] = -.1
    elif bad == "nan": p[0, 0] = np.nan
    elif bad == "reverse": starts = [.5, 0.]
    else: duration = 500
    with pytest.raises(ValueError): legacy_window_predictions(p, starts, list(LABELS), duration_ms=duration)


@pytest.mark.parametrize("outcome", ["partial", "failed", "cancelled", "invalid", "empty"])
def test_worker_cell_outcomes_are_cell_based_and_cancellation_is_not_swallowed(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("MRCD_DATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(worker.media, "load", lambda _: np.zeros(100000, np.float32))
    store = Store(f"sqlite:///{tmp_path / 'app.db'}")
    rec = store.create("recording", dict(sha256="a" * 64, media=dict(analysis=analysis(6250)), segments=cores()), status="ready")
    run = store.create("run", dict(system="mrcd", config=dict(engine_kind="cell_v2")), parent=rec["id"], status="running")
    class AllFailed(FakeCellInference):
        def predict(self, features): raise RuntimeError("synthetic all failed")
    class Invalid(FakeCellInference):
        def predict(self, features):
            output = super().predict(features)
            output["type_probabilities"][0, 0] = np.nan
            return output
    def adapter(job):
        if outcome == "cancelled":
            store.update(run["id"], status="cancel_requested")
        inf = AllFailed() if outcome == "failed" else Invalid() if outcome == "invalid" else FakeCellInference(failure=2500)
        d = 0 if outcome == "empty" else 6250
        return CellAnalysisEngine(inf).analyze(prep(d), analysis=analysis(d), recording_id=rec["id"], segments=[] if not d else cores(), stage=job.stage)
    worker.execute(store, run, dict(mrcd=adapter))
    saved = store.get(run["id"])
    assert saved["status"] == ("failed" if outcome == "invalid" else "succeeded" if outcome == "empty" else outcome), saved["data"].get("error")
    if outcome in ("invalid", "cancelled"):
        assert "result" not in saved["data"]
    elif outcome == "partial":
        assert saved["data"]["result"]["coverage"] == [[0, 2500], [4500, 6250]]


def test_http_unsupported_empty_new_results_do_not_create_assisted_tasks_or_exports(tmp_path, monkeypatch):
    from app.ops import create_tasks
    monkeypatch.setenv("MRCD_DATA", str(tmp_path / "appdata"))
    store = Store(f"sqlite:///{tmp_path / 'app.db'}")
    researcher = store.create("user", dict(name="r", role="researcher", token_sha256=hashlib.sha256(b"token").hexdigest()), status="active")
    annotator = store.create("user", dict(name="a", role="annotator", token_sha256=hashlib.sha256(b"annotation-token").hexdigest()), status="active")
    rec = store.create("recording", dict(sha256="a" * 64, speaker_id="s", media=dict(analysis=analysis(6250)), segments=cores()), owner=researcher["id"], status="ready")
    run = store.create("run", dict(system="mrcd", comparison="test", result=dict(result_schema=CELL_RESULT_SCHEMA, events=[])), parent=rec["id"], status="succeeded")
    client = TestClient(create_app(store)); headers = dict(Authorization="Bearer token")
    response = client.post("/api/review/tasks", headers=headers, json=dict(recording_id=rec["id"], start_ms=0, end_ms=5000, mode="assisted", split="dev", run_id=run["id"], assignee=annotator["id"]))
    assert response.status_code == 422 and "task" in response.text
    with pytest.raises(ValueError): create_tasks(store, rec["id"], [annotator["id"]], mode="assisted", split="dev", run_id=run["id"])
    assert store.find("task") == []
    for format in ("csv", "eaf"):
        assert client.get(f"/api/exports/{run['id']}?format={format}", headers=headers).status_code == 422
    assert client.get("/api/experiments/comparisons/test", headers=headers).status_code == 422


def test_strict_product_loader_rejects_incompatible_before_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(systems, "_loaded", {})
    def forbidden(*args): raise AssertionError("preparation must not run on invalid model")
    monkeypatch.setattr(systems, "prepared", forbidden)
    incompatible = tmp_path / "legacy.pt"
    torch.save(dict(checkpoint_version="legacy"), incompatible)
    job = worker.Job("run", dict(id="r", data=dict(media=dict(analysis=analysis(6250)), segments=cores())), {},
                     dict(engine_kind="cell_v2", checkpoint=str(incompatible)), np.zeros(100000), tmp_path, lambda _: None)
    with pytest.raises(ValueError, match="contract"): systems.mrcd(job)
    job.config = dict(engine_kind="cell_v2")
    with pytest.raises(ValueError, match="explicit"): systems.mrcd(job)


@pytest.mark.parametrize("start,end,expected", [(490, 510, [0, 1000]), (500, 510, [500, 1000]), (2400, 2600, [2000, 3000])])
def test_oracle_decoder_recovers_cell_projection_not_arbitrary_human_timestamps(start, end, expected):
    from core.targets import project_targets
    source = [dict(start_ms=start, end_ms=end, label="filler_word", type_known=True, is_disfluent=True)]
    projected = project_targets(3250, source, [[0, 3250]])
    cells = result(duration=3250)["cells"]
    for i, cell in enumerate(cells):
        cell["type_decisions"] = ["negative"] * 7
        cell["type_decisions"][0] = "positive" if projected["multi_hot"][i, 0] else "negative"
    events = decode_cells(cells, recording_id="r")
    assert [[e["start_ms"], e["end_ms"]] for e in events] == [expected]
    assert projected["source_events"] == source
    assert events[0]["boundary_source"] == "owned_cells_unrefined"


@pytest.mark.parametrize("marker", [None, "", "unrecognized-cell-kind"])
def test_recognizable_cell_ledger_cannot_downgrade_after_both_top_markers_change(marker):
    data = result(FakeCellInference(failure=2500))
    assert evaluation.run_coverage(dict(parent="r", status="partial", data=dict(result=data)), cores())["available_ms"] == 4250
    data.pop("result_schema")
    if marker is None: data.pop("prediction_kind")
    else: data["prediction_kind"] = marker
    with pytest.raises(ValueError): evaluation.run_coverage(dict(parent="r", status="partial", data=dict(result=data)), cores())


@pytest.mark.parametrize("signal", ["contract", "events", "empty_contract", "unknown_kind"])
def test_recognizable_window_payload_cannot_downgrade_to_old_core_coverage(signal):
    p = np.zeros((1, len(LABELS))); p[:, LABELS.index("filler_word")] = 1
    data = dict(events=legacy_window_predictions(p, [0.], list(LABELS), duration_ms=6250), segments={s["id"]: "succeeded" for s in cores()})
    if signal == "contract": data["window_contract"] = "legacy-forward3000-stride500-v1"
    elif signal == "empty_contract": data["window_contract"] = ""
    elif signal == "unknown_kind": data["prediction_kind"] = "new-unknown-window-kind"
    with pytest.raises(ValueError): evaluation.run_coverage(dict(parent="r", status="succeeded", data=dict(result=data)), cores())


def test_genuine_historical_result_retains_core_availability_and_original_bytes():
    data = dict(events=[dict(label="filler_word", start_ms=100, end_ms=500, decision="event")],
                segments={"s0": "succeeded", "s1": "failed", "s2": "succeeded"})
    before = deepcopy(data)
    coverage = evaluation.run_coverage(dict(parent="r", status="partial", data=dict(result=data)), cores())
    assert coverage["available_ms"] == 3950 and coverage["coverage"] == {"r": [[0, 2300], [4600, 6250]]}
    assert data == before


def test_uncertainty_filter_and_summary_share_type_and_context_semantics():
    class Uncertain(FakeCellInference):
        def predict(self, features):
            output = super().predict(features)
            output["context_probabilities"][:, [0, 2]] = [.5, .5]
            return output
    data = result(Uncertain())
    source = (Path(__file__).parents[1] / "app/static/app.js").read_text()
    summary = "function resultSemantics(res) {" + source.split("function resultSemantics(res) {", 1)[1].split("// END RESULT SEMANTICS", 1)[0]
    filtered = "const visible = events.filter(" + source.split("const visible = events.filter(", 1)[1].split("const words = res.words || []", 1)[0]
    js = "const PAUSES=new Set(['rhetorical_pause','neutral_pause']);\n" + summary
    js += "\nfunction filter(res,f) {const events=res.events||[],semantics=resultSemantics(res);" + filtered + "\nreturn visible;}"
    js += "\nconst res=" + json.dumps(data) + ";console.log(JSON.stringify({summary:resultSemantics(res),visible:filter(res,{label:'',kind:'uncertain'}),old:filter({events:[{decision:'uncertain'},{decision:'event'}]},{kind:'uncertain'})}));"
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out["summary"]["unc"] == len(out["visible"]) == 2
    assert all(e["type_decision"] == "positive" and e["decision"] == "event" for e in out["visible"])
    assert out["old"] == [dict(decision="uncertain")]


def test_qa_new_window_groups_are_candidates_without_rates_and_old_report_is_unchanged(tmp_path, monkeypatch):
    import sys
    from scripts import infer_video, qa_report
    from core.models.fusion import MODES
    p = np.zeros((2, len(LABELS))); p[:, LABELS.index("filler_word")] = 1
    events = infer_video.merge_events(np.array([0., .5]), p.argmax(1), p, list(LABELS), duration_ms=3500)
    windows = [dict(timestamp_start=t, timestamp_end=t+3, transcription="", features=dict(f0_reset=0., max_delta_mouth_press=None, is_dangling=False), features_extra=dict(sil_dur=0.), predictions={m:dict(**{"class": "filler_word"}, prob=1.) for m in MODES}) for t in [0., .5]]
    artifact = dict(prediction_kind="window_classification", window_contract="legacy-forward3000-stride500-v1", duration_s=3.5, labels=list(LABELS), window_s=3., stride_s=.5, windows=windows, events={m:deepcopy(events) for m in MODES}, window_disagreement_rate=0.)
    case = tmp_path / "case"; case.mkdir()
    monkeypatch.setattr(qa_report, "ROOT", tmp_path)
    monkeypatch.setattr(qa_report, "CASE", case)
    input_file, report = tmp_path / "results.json", tmp_path / "report.md"
    input_file.write_text(json.dumps(artifact)); original = input_file.read_bytes()
    monkeypatch.setattr(sys, "argv", ["qa_report.py", "--vid", "synthetic", "--results", str(input_file), "--out", str(report)])
    qa_report.main()
    text = report.read_text()
    assert "Tasa por minuto" not in text and "eventos/min" not in text and "Eventos detectados" not in text
    assert "candidate" in text.lower() and "unavailable" in text.lower()
    assert input_file.read_bytes() == original and not list(case.iterdir())
    # A genuinely historical artifact has no new schema/contract/event tags.
    artifact.pop("prediction_kind"); artifact.pop("window_contract")
    artifact["events"] = {m:[dict(**{"class": "filler_word"}, start=0., end=3.5, n_windows=2, peak_prob=1.)] for m in MODES}
    input_file.write_text(json.dumps(artifact)); historical = input_file.read_bytes()
    qa_report.main()
    assert "Tasa por minuto (trimodal): **17.1** eventos/min." in report.read_text()
    assert input_file.read_bytes() == historical
