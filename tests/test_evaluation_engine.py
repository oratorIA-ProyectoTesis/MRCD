"""Evaluator, decoding and extraction-refactor guarantees (no ASR/model downloads)."""

import numpy as np
import pytest

from core import evaluation as ev
from core.constants import LABELS, SAMPLE_RATE
from core.engine import decode, window_tensors
from core.extractors.acoustic import extract_acoustic
from core.extractors.linguistic import Word
from core.live.window_builder import build_window


def e(a, b, label="filler_word", **kw):
    return {"start_ms": a, "end_ms": b, "label": label, **kw}


def test_long_event_cannot_match_five_references():
    ref = [e(i * 1000, i * 1000 + 400) for i in range(5)]
    r = ev.evaluate(ref, [e(0, 5000)])
    assert r["per_class"]["filler_word"]["tp"] == 0 and r["per_class"]["filler_word"]["fn"] == 5
    r = ev.evaluate(ref, [e(0, 5000)], iou_threshold=0.01)
    assert r["per_class"]["filler_word"]["tp"] == 1  # one-to-one even when every pair overlaps


def test_class_and_boundary_errors_are_not_hidden():
    ref = [e(0, 1000), e(5000, 6000, "repetition")]
    hyp = [e(0, 1000, "revision"), e(5600, 7000, "repetition")]
    r = ev.evaluate(ref, hyp)
    assert r["errors"] == {"class": 1, "interval": 1, "unsupported": 0, "missed": 2}
    assert r["boundaries"]["matched_proportion"] == 0


def test_pauses_are_reported_apart_from_disfluencies():
    ref = [e(0, 500), e(1000, 2000, "neutral_pause")]
    hyp = [e(0, 500), e(1000, 2000, "neutral_pause"), e(3000, 4000, "rhetorical_pause")]
    r = ev.evaluate(ref, hyp, evaluable_ms=60000)
    assert r["micro"]["disfluency"]["f1"] == 1.0 and r["micro"]["pause"]["fp"] == 1
    assert r["false_alarms_per_min"] == 0  # pause false alarms are not disfluency false alarms


def test_unreviewed_audio_is_not_negative_and_uncertain_is_not_an_event():
    cov = {"r": [(0, 10000)]}
    hyp = [e(20000, 20500, recording_id="r"), e(100, 600, recording_id="r", decision="uncertain")]
    r = ev.evaluate([], hyp, coverage=cov)
    assert r["n_hyp"] == 0 and r["n_uncertain"] == 1


def test_tolerance_metric_is_secondary_and_separate():
    r = ev.evaluate([e(0, 200)], [e(60, 150)])
    assert r["micro"]["disfluency"]["tp"] == 0 and r["tolerance"]["disfluency"]["tp"] == 1


def test_agreement_counts_omissions_in_kappa_units():
    a = [e(0, 500), e(1000, 1500, "repetition")]
    b = [e(0, 520)]
    g = ev.agreement(a, b)
    assert g["matched"] == 1 and g["existence_agreement"] == pytest.approx(2 / 3)
    assert ("repetition", "none") in {
        (x, y)
        for x in g["confusion"]["labels"]
        for y in g["confusion"]["labels"]
        if g["confusion"]["matrix"][g["confusion"]["labels"].index(x)][g["confusion"]["labels"].index(y)]
    }


def test_speaker_bootstrap_and_paired_difference():
    ref = [e(i * 1000, i * 1000 + 500, speaker_id=f"s{i % 3}") for i in range(9)]
    good, bad = list(ref), ref[:3]
    f1 = lambda r, h: ev.evaluate(r, h)["micro"]["disfluency"]["f1"]  # noqa: E731
    ci = ev.bootstrap_ci(ref, [good], f1, n=50)
    assert ci["low"] == ci["high"] == 1.0 and ci["n_speakers"] == 3
    diff = ev.bootstrap_ci(ref, [good, bad], lambda r, a, b: f1(r, a) - f1(r, b), n=50)
    assert diff["low"] > 0


def test_calibration_threshold_and_coverage_use_fixed_units():
    scores, correct = [0.9, 0.8, 0.6, 0.4], [1, 1, 0, 0]
    assert ev.pick_threshold(scores, correct, max_risk=0.0) == 0.8
    cr = ev.coverage_risk(scores, correct, 0.8)
    assert cr == {"threshold": 0.8, "n_units": 4, "coverage": 0.5, "risk": 0.0, "abstained": 2}
    assert ev.calibration(np.array([0.9, 0.1]), np.array([1, 0]))["brier"] == pytest.approx(0.01)
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    assert ev.fit_temperature(probs, np.array([0, 1, 0, 1])) > 1  # overconfident -> soften


def test_active_selection_refuses_test_and_sus_scoring():
    with pytest.raises(ValueError):
        ev.select_for_review([{"split": "test", "kind": "uncertain"}], 1)
    picked = ev.select_for_review([{"kind": "disagreement", "speaker_id": s} for s in "aab"], 2, quotas=(1, 0, 0))
    assert sorted(c["speaker_id"] for c in picked) == ["a", "b"]  # round-robin across speakers
    assert ev.sus_score([5, 1] * 5) == 100 and ev.sus_score([3] * 10) == 50


def test_decode_tiles_by_stride_and_abstains_below_threshold():
    labels = list(LABELS)
    probs = np.zeros((5, len(labels)))
    probs[[0, 1], labels.index("filler_word")] = 0.9
    probs[2, labels.index("fluent")] = 1
    probs[[3, 4], labels.index("block")] = 0.4
    events = decode(probs, np.arange(5) * 0.5, labels, thresholds={"block": 0.5})
    assert [(x["label"], x["decision"], x["start_ms"], x["end_ms"]) for x in events] == [
        ("filler_word", "event", 0, 1000),
        ("block", "uncertain", 1500, 2500),
    ]


def test_vectorized_windows_equal_per_window_builder():
    """Refactor is computational only: batch tensors equal the old per-window path."""
    rng = np.random.default_rng(0)
    t = np.arange(16 * SAMPLE_RATE) / SAMPLE_RATE
    audio = (
        0.2 * np.sin(2 * np.pi * 140 * t) * (np.sin(2 * np.pi * 0.4 * t) > 0) + 0.01 * rng.standard_normal(len(t))
    ).astype(np.float32)
    words = [
        Word("eh", "eh", 1.0, 1.3, 0.9, ","),
        Word("bueno", "bueno", 4.0, 4.4, 0.9, ""),
        Word("casa", "casa", 9.0, 9.5, 0.9, "."),
    ]
    starts = [0.0, 2.5, 6.0]
    frames = extract_acoustic(audio, use_silero=False).frame_matrix()
    batch = window_tensors(frames, words, starts)
    for i, s in enumerate(starts):
        single = build_window(audio, words, s, use_silero=False)
        for key, value in single.items():
            np.testing.assert_array_equal(value[0], batch[key][i])


def snapshot_fixture(items, *, versioned=True):
    coverage = {}
    for item in items:
        coverage.setdefault(item["recording_id"], []).append([item["start_ms"], item["end_ms"]])
    # Test fixtures intentionally include overlapping spans; reference_view must union them.
    for rid, spans in coverage.items():
        merged = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        coverage[rid] = merged
    data = {"items": items}
    if versioned:
        data.update(snapshot_schema="mrcd-snapshot-v2", coverage=coverage,
                    reference_kind="human_adjudicated", taxonomy_version="mrcd-taxonomy-v1")
    return data


def region(start, end, *, events=(), contextual=(), rid="r", speaker="s"):
    return {"recording_id": rid, "speaker_id": speaker, "start_ms": start, "end_ms": end,
            "events": list(events), "contextual": list(contextual)}


def run_fixture(name, intervals, events=(), *, rid="r", status="partial", explicit=None):
    segments = [{"id": f"s{i}", "core_start_ms": a, "core_end_ms": b,
                 "start_ms": max(0, a - 5000), "end_ms": b + 5000} for i, (a, b) in enumerate(intervals)]
    result = {"events": list(events), "segments": {s["id"]: "succeeded" for s in segments}}
    if explicit is not None:
        result["coverage"] = explicit
    return {"id": name, "parent": rid, "status": status, "data": {"system": name, "result": result}}, segments


def test_reference_view_unions_human_regions_and_subtracts_only_human_unknowns():
    positive = e(200, 600)
    unknown = e(1200, 1800, decision="uncertain", label=None)
    snap = snapshot_fixture([region(0, 2000, events=[positive, unknown]), region(1000, 3000)])
    view = ev.reference_view(snap, {"r": "s"})
    assert view["coverage"] == {"r": [[0, 1200], [1800, 3000]]}
    assert ev.coverage_duration(view["coverage"]) == 2400
    scored = ev.evaluate(view["reference"], [{**positive, "recording_id": "r"}], coverage=view["coverage"])
    assert scored["micro"]["disfluency"]["tp"] == 1 and scored["evaluable_ms"] == 2400
    assert scored["protocol"]["boundary_policy"] == "fully_contained_v2"


@pytest.mark.parametrize("decision", ["uncertain", "not_evaluable"])
def test_human_hole_excludes_prediction_without_shrinking_for_model_abstention(decision):
    snap = snapshot_fixture([region(0, 2000, events=[e(800, 1200, decision=decision)])])
    view = ev.reference_view(snap, {"r": "s"})
    result = ev.evaluate(view["reference"], [e(900, 1100, recording_id="r")], coverage=view["coverage"])
    assert result["evaluable_ms"] == 1600 and result["micro"]["disfluency"]["fp"] == 0
    assert result["exclusions"]["prediction"]["count"] == 1
    abstained = ev.evaluate([e(100, 300, recording_id="r")], [e(100, 300, recording_id="r", decision="uncertain")], coverage={"r": [[0, 2000]]})
    assert abstained["evaluable_ms"] == 2000 and abstained["n_uncertain"] == 1
    assert abstained["micro"]["disfluency"]["fn"] == 1 and abstained["micro"]["disfluency"]["f1"] == 0


def test_partial_run_scores_only_successful_core_and_intersects_explicit_coverage():
    run, segments = run_fixture("a", [(0, 30_000), (30_000, 60_000)], [e(1000, 1400)])
    run["data"]["result"]["segments"]["s1"] = "failed"
    available = ev.run_coverage(run, segments)
    assert available["coverage"] == {"r": [[0, 30_000]]}
    result = ev.evaluate([e(1000, 1400, recording_id="r"), e(40_000, 40_400, recording_id="r")],
        [e(1000, 1400, recording_id="r")], coverage=available["coverage"])
    assert result["evaluable_ms"] == 30_000 and result["micro"]["disfluency"]["tp"] == 1
    assert result["micro"]["disfluency"]["fn"] == 0 and result["exclusions"]["reference"]["count"] == 1
    run["data"]["result"]["coverage"] = [[5000, 10_000], [9000, 12_000]]
    assert ev.run_coverage(run, segments)["coverage"] == {"r": [[5000, 12_000]]}
    run["data"]["result"]["segments"]["s0"] = "completed"
    assert ev.run_coverage(run, segments)["coverage"] == {"r": []}


@pytest.mark.parametrize("predictions,fp,rate", [([e(1000, 1500, recording_id="r")], 1, 1), ([], 0, 0)])
def test_functional_context_is_an_evaluable_negative_not_an_exclusion(predictions, fp, rate):
    snap = snapshot_fixture([region(0, 60_000, contextual=[{"start_ms": 900, "end_ms": 1600, "expression": "eh", "is_disfluent": False}])])
    view = ev.reference_view(snap, {"r": "s"})
    result = ev.evaluate(view["reference"], predictions, coverage=view["coverage"], legitimate=view["legitimate"])
    assert result["evaluable_ms"] == 60_000 and result["micro"]["disfluency"]["fp"] == fp
    assert result["false_alarms_per_min"] == rate and result["false_alarms_on_legitimate"] == fp
    if not predictions:
        assert result["micro"]["disfluency"]["f1"] is None
        assert result["micro"]["disfluency"]["reasons"]["f1"] == "no_reference_or_predictions"


def test_empty_availability_is_unavailable_not_zero_counts_or_false_negatives():
    result = ev.evaluate([e(1000, 1400, recording_id="r")], [], coverage={"r": []})
    assert result["evaluable_ms"] == 0 and result["reason"] == "no_evaluable_coverage"
    assert result["n_ref"] is None and result["false_alarms_per_min"] is None
    assert all(result["micro"]["disfluency"][key] is None for key in ("tp", "fp", "fn", "f1", "support"))


def test_fully_contained_policy_excludes_both_boundary_crossers_and_includes_exact_edge():
    crossing = e(900, 1300, recording_id="r")
    exact = e(1000, 1300, recording_id="r")
    result = ev.evaluate([crossing, exact], [crossing, exact], coverage={"r": [[1000, 3000]]})
    assert result["micro"]["disfluency"]["tp"] == 1 and result["n_ref"] == result["n_hyp"] == 1
    assert result["exclusions"]["reference"]["boundary_or_hole"] == result["exclusions"]["prediction"]["boundary_or_hole"] == 1


def test_v2_coverage_must_match_items_and_legacy_adapter_does_not_mutate_snapshot():
    import copy
    snap = snapshot_fixture([region(0, 2000)])
    snap["coverage"]["r"] = [[0, 3000]]
    with pytest.raises(ValueError, match="coverage"):
        ev.reference_view(snap, {"r": "s"})
    old = snapshot_fixture([region(0, 2000)], versioned=False)
    before = copy.deepcopy(old)
    view = ev.reference_view(old, {"r": "s"})
    assert view["adapter"] == "legacy_items_v1" and old == before
    assert view["coverage"] == {"r": [[0, 2000]]}


def test_paired_metrics_recompute_both_systems_on_common_coverage():
    snap = snapshot_fixture([region(0, 6000, events=[e(500, 900), e(2500, 2900), e(4500, 4900)])])
    a, seg_a = run_fixture("a", [(0, 4000)], [e(500, 900)])
    b, _ = run_fixture("b", [(2000, 6000)], [e(2500, 2900), e(4500, 4900)])
    # Both runs share the recording segmentation; explicit coverage narrows each system.
    segments = [{"id": "s0", "core_start_ms": 0, "core_end_ms": 6000}]
    a["data"]["result"]["coverage"], b["data"]["result"]["coverage"] = [[0, 4000]], [[2000, 6000]]
    recordings = {"r": {"data": {"speaker_id": "s", "segments": segments}}}
    result = ev.evaluate_runs(snap, {"a": [a], "b": [b]}, recordings, bootstrap=10)
    assert (result["results"]["a"]["micro"]["disfluency"]["tp"], result["results"]["a"]["micro"]["disfluency"]["fn"]) == (1, 1)
    assert (result["results"]["b"]["micro"]["disfluency"]["tp"], result["results"]["b"]["micro"]["disfluency"]["fn"]) == (2, 0)
    pair = result["paired_results"]["b-a"]
    assert pair["evaluable_ms"] == 2000 and pair["delta_f1"] == 1
    assert pair["results"]["a"]["micro"]["disfluency"]["fn"] == 1 and pair["results"]["a"]["micro"]["disfluency"]["tp"] == 0
    assert pair["results"]["b"]["micro"]["disfluency"]["tp"] == 1
    assert pair["bootstrap_ci"]["low"] == pair["bootstrap_ci"]["high"] == 1


def test_coverage_cluster_bootstrap_includes_negative_only_speakers_and_recalculates_duration():
    coverage = {"rA": [[0, 10_000]], "rB": [[0, 20_000]], "rC": [[0, 30_000]]}
    speakers = {"rA": "A", "rB": "B", "rC": "C"}
    ref = [e(1000, 1500, recording_id="rA", speaker_id="A")]
    draw = ev.resample_clusters(ref, [ref], coverage, speakers, ["B", "C", "C"])
    assert ev.coverage_duration(draw["coverage"]) == 80_000 and draw["reference"] == []
    assert len(draw["coverage"]) == 3
    stat = lambda r, h, **kw: ev.evaluate(r, h, **kw)["micro"]["disfluency"]["f1"]
    ci = ev.bootstrap_ci(ref, [ref], stat, n=30, coverage=coverage, recording_speakers=speakers)
    assert ci["n_speakers"] == 3 and ci["requested"] == 30 and ci["valid"] + ci["omitted"] == 30
    assert ci["omitted"] > 0 and ci["low"] == ci["high"] == 1
    empty = ev.bootstrap_ci([], [[]], stat, n=5, coverage={"rA": []}, recording_speakers=speakers)
    assert empty["valid"] == 0 and empty["omitted"] == 5 and empty["reason"] == "no_evaluable_coverage"


@pytest.mark.parametrize("same_id", [True, False])
def test_evaluation_selection_rejects_duplicate_run_or_system_recording(same_id):
    snap = snapshot_fixture([region(0, 2000)])
    first, segments = run_fixture("a", [(0, 2000)])
    second = {**first, "id": first["id"] if same_id else "other-run"}
    with pytest.raises(ValueError, match="duplicate|multiple"):
        ev.evaluate_runs(snap, {"a": [first, second]}, {"r": {"data": {"speaker_id": "s", "segments": segments}}}, bootstrap=0)


def test_reference_view_rejects_duplicate_positive_events_without_rejecting_distinct_overlaps():
    first = region(0, 2000, events=[e(500, 1500)])
    duplicate = region(1000, 3000, events=[e(500, 1500)])
    with pytest.raises(ValueError, match="duplicate"):
        ev.reference_view(snapshot_fixture([first, duplicate]), {"r": "s"})
    overlap = region(1000, 3000, events=[e(600, 1600, "repetition")])
    assert len(ev.reference_view(snapshot_fixture([first, overlap]), {"r": "s"})["reference"]) == 2


def test_pause_only_protocol_has_undefined_disfluency_comparison_not_internal_error():
    snap = snapshot_fixture([region(0, 2000, events=[e(500, 900, "neutral_pause")])])
    run, segments = run_fixture("a", [(0, 2000)], [e(500, 900, "neutral_pause")])
    result = ev.evaluate_runs(snap, {"a": [run]}, {"r": {"data": {"speaker_id": "s", "segments": segments}}},
        protocol={"classes": ["neutral_pause"]}, bootstrap=2)
    assert result["results"]["a"]["micro"]["pause"]["f1"] == 1
    assert result["bootstrap_ci"]["a"]["valid"] == 0
    assert result["bootstrap_ci"]["a"]["reason"] == "statistic_undefined_in_all_replicates"


def test_duration_is_derived_from_union_and_cannot_be_supplied_as_a_contradictory_denominator():
    with pytest.raises(ValueError, match="evaluable_ms"):
        ev.evaluate([], [], coverage={"r": [[0, 2000], [1000, 3000]]}, evaluable_ms=4000)
    result = ev.evaluate([], [], coverage={"r": [[0, 2000], [1000, 3000]]})
    assert result["evaluable_ms"] == 3000 and result["false_alarms_per_min"] == 0


def test_cluster_copies_have_unique_event_and_recording_identities():
    ref = [e(100, 300, event_id="original-human-event", recording_id="r", speaker_id="A")]
    sampled = ev.resample_clusters(ref, [ref], {"r": [[0, 1000]]}, {"r": "A"}, ["A", "A", "A"])
    assert len({event["event_id"] for event in sampled["reference"]}) == 3
    assert len({event["recording_id"] for event in sampled["reference"]}) == 3
    assert ev.coverage_duration(sampled["coverage"]) == 3000


def test_explicit_denominator_must_match_human_masked_eligible_duration():
    reference = [e(1200, 1800, recording_id="r", decision="uncertain")]
    result = ev.evaluate(reference, [], coverage={"r": [[0, 3000]]}, evaluable_ms=2400)
    assert result["evaluable_ms"] == 2400
    with pytest.raises(ValueError, match="evaluable_ms"):
        ev.evaluate(reference, [], coverage={"r": [[0, 3000]]}, evaluable_ms=3000)


def test_paired_ci_compatibility_alias_cannot_overwrite_a_named_system_ci():
    snap = snapshot_fixture([region(0, 2000, events=[e(500, 900)])])
    good, segments = run_fixture("a", [(0, 2000)], [e(500, 900)])
    bad, _ = run_fixture("b", [(0, 2000)], [])
    named_like_pair, _ = run_fixture("b-a", [(0, 2000)], [e(500, 900)])
    result = ev.evaluate_runs(snap, {"a": [good], "b": [bad], "b-a": [named_like_pair]},
        {"r": {"data": {"speaker_id": "s", "segments": segments}}}, bootstrap=2)
    assert result["bootstrap_ci"]["b-a"]["low"] == 1  # own CI for the named system
    assert result["paired_results"]["b-a"]["bootstrap_ci"]["low"] == -1


def test_system_agreement_has_fixed_symmetric_coverage_without_changing_human_mask_semantics():
    uncertain = e(1000, 1500, recording_id="r", decision="uncertain")
    functional = e(2000, 2500, recording_id="r", is_disfluent=False)
    positive = e(3000, 3500, recording_id="r")
    a, b = [uncertain, functional, positive], [e(1000, 1500, recording_id="r"), positive]
    coverage = {"r": [[0, 6000]]}
    forward = ev.system_agreement(a, b, coverage=coverage)
    reverse = ev.system_agreement(b, a, coverage=coverage)
    assert forward["per_class"]["filler_word"]["f1"] == reverse["per_class"]["filler_word"]["f1"] == pytest.approx(2 / 3)
    assert forward["coverage"] == reverse["coverage"] == coverage
    assert forward["evaluable_ms"] == reverse["evaluable_ms"] == 6000
    human = ev.evaluate([uncertain, positive], b, coverage=coverage)
    assert human["evaluable_ms"] == 5500 and human["per_class"]["filler_word"]["f1"] == 1
