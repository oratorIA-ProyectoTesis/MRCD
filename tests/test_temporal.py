"""Pure temporal/target oracles: no files, feature extraction, weights, or services."""

from copy import deepcopy

import numpy as np
import pytest

from core.constants import TAXONOMY
from core.temporal import TEMPORAL_CONTRACT, decision_cells, evidence_windows, validate_temporal_grid
from core.targets import TARGET_CONTRACT, project_targets, validate_targets


def event(start, end, label="filler_word", **fields):
    return {"start_ms": start, "end_ms": end, "label": label,
            "type_known": label is not None and fields.get("decision", "event") == "event", **fields}


def column(label):
    return TAXONOMY.index(label)


@pytest.mark.parametrize("duration", [0, 125, 200, 500, 501, 2999, 3001, 3250])
def test_exact_owned_cover_and_centered_requested_observed_evidence(duration):
    cells = decision_cells(duration)
    assert len(cells) == (duration + 499) // 500
    assert validate_temporal_grid(cells, duration) == len(cells)
    windows = evidence_windows(cells)
    assert validate_temporal_grid(windows, duration) == len(cells)
    if not duration:
        assert cells == windows == []
    else:
        assert cells[0]["owned_start_ms"] == 0 and cells[-1]["owned_end_ms"] == duration
        assert sum(c["owned_end_ms"] - c["owned_start_ms"] for c in cells) == duration
    for index, (cell, window) in enumerate(zip(cells, windows)):
        assert cell["temporal_contract"] == TEMPORAL_CONTRACT == "cell500-centered3000-v2"
        assert cell["media_duration_ms"] == duration
        assert cell["index"] == index and cell["nominal_center_ms"] == 500 * index + 250
        assert window["requested_start_ms"] == 500 * index - 1250
        assert window["requested_end_ms"] == 500 * index + 1750
        assert window["observed_start_ms"] == max(0, window["requested_start_ms"])
        assert window["observed_end_ms"] == min(duration, window["requested_end_ms"])
        assert 0 <= window["observed_start_ms"] < window["observed_end_ms"] <= duration
        assert window["pad_left_ms"] + window["observed_end_ms"] - window["observed_start_ms"] + window["pad_right_ms"] == 3000
        if index:
            assert cells[index - 1]["owned_end_ms"] == cell["owned_start_ms"]


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), 1.5, True])
def test_duration_must_be_a_real_nonnegative_integer_not_derived_row_count(duration):
    with pytest.raises(ValueError, match="duration"):
        decision_cells(duration)


@pytest.mark.parametrize("field,value", [
    ("owned_start_ms", -1), ("owned_start_ms", 600), ("owned_end_ms", 0),
    ("owned_end_ms", 600), ("nominal_center_ms", float("nan")),
    ("nominal_center_ms", 200), ("media_duration_ms", 999), ("index", 2),
    ("temporal_contract", "legacy"),
])
def test_grid_rejects_malformed_nonfinite_reversed_inconsistent_rows(field, value):
    cells = decision_cells(1000)
    cells[0][field] = value
    with pytest.raises(ValueError):
        validate_temporal_grid(cells, 1000)


@pytest.mark.parametrize("mutation", ["overlap", "gap", "reverse_order", "missing_tail", "empty_nonzero", "extra_cell", "evidence_past_duration", "incomplete_evidence"])
def test_grid_requires_order_nonoverlap_exact_adjacency_and_real_observed_support(mutation):
    cells = evidence_windows(decision_cells(501))
    if mutation == "overlap":
        cells[1]["owned_start_ms"] = 499
    elif mutation == "gap":
        cells[0]["owned_end_ms"] = 499
    elif mutation == "reverse_order":
        cells.reverse()
    elif mutation == "missing_tail":
        cells.pop()
    elif mutation == "empty_nonzero":
        cells = []
    elif mutation == "extra_cell":
        cells.append(deepcopy(cells[-1]))
    elif mutation == "evidence_past_duration":
        cells[-1]["observed_end_ms"] = 502
    else:
        cells[-1].pop("observed_end_ms")
    with pytest.raises(ValueError):
        validate_temporal_grid(cells, 501)


def test_no_unversioned_config_or_inconsistent_declared_duration_is_silently_accepted():
    with pytest.raises(ValueError):
        decision_cells(1000, stride_ms=250)
    with pytest.raises(ValueError):
        evidence_windows(decision_cells(1000), context_ms=1000)
    with pytest.raises(ValueError):
        validate_temporal_grid(decision_cells(501), 500)


def test_endpoint_touching_is_not_positive_overlap_and_short_events_survive():
    projected = project_targets(1000, [event(499, 500), event(500, 501, "repetition")], [[0, 1000]])
    assert projected["multi_hot"][:, column("filler_word")].tolist() == [True, False]
    assert projected["multi_hot"][:, column("repetition")].tolist() == [False, True]
    assert validate_targets(projected) == 2


def test_event_2400_2600_owns_only_its_two_cells_not_an_earlier_evidence_window():
    projected = project_targets(3250, [event(2400, 2600, is_disfluent=True)], [[0, 3250]])
    assert np.flatnonzero(projected["multi_hot"][:, column("filler_word")]).tolist() == [4, 5]
    assert projected["event_indices"][0] == []
    assert projected["event_indices"][4] == projected["event_indices"][5] == [0]


def test_quantization_oracle_recovers_cell_projection_not_original_timestamps():
    raw = [event(490, 510)]
    projected = project_targets(1000, raw, [[0, 1000]])
    selected = [cell for cell, positive in zip(projected["cells"], projected["multi_hot"][:, column("filler_word")]) if positive]
    assert [[cell["owned_start_ms"], cell["owned_end_ms"]] for cell in selected] == [[0, 500], [500, 1000]]
    assert projected["source_events"] == raw and raw[0]["end_ms"] - raw[0]["start_ms"] == 20


def test_distinct_phenomena_preserve_multihot_but_never_overwrite_softmax_with_a_dominant_type():
    projected = project_targets(500, [event(100, 300, is_disfluent=True), event(200, 400, "repetition", is_disfluent=True)], [[0, 500]])
    assert projected["multi_hot"][0, [column("filler_word"), column("repetition")]].all()
    assert projected["type_mask"][0].all() and projected["type_ambiguity"][0]
    assert not projected["softmax_mask"][0] and projected["softmax_target"][0] == -1


def test_functional_filler_retains_phenomenology_and_explicit_functional_context_never_fluent():
    projected = project_targets(500, [event(100, 200, is_disfluent=False)], [[0, 500]])
    filler = column("filler_word")
    assert projected["multi_hot"][0, filler] and projected["type_mask"][0, filler]
    assert projected["softmax_target"][0] == filler + 1 and projected["softmax_mask"][0]
    assert projected["context_mask"][0, filler] and projected["context_target"][0, filler] == 0
    assert projected["context_multi_hot"][0, filler].tolist() == [True, False]


@pytest.mark.parametrize("fields", [{}, {"is_disfluent": None}, {"decision": "uncertain", "type_known": True}])
def test_context_is_not_invented_from_event_presence_or_an_uncertain_decision(fields):
    projected = project_targets(500, [event(100, 200, **fields)], [[0, 500]])
    filler = column("filler_word")
    assert projected["multi_hot"][0, filler]
    assert not projected["context_mask"][0, filler] and projected["context_target"][0, filler] == -1


@pytest.mark.parametrize("fields", [{}, {"type_known": False}, {"decision": "event", "is_disfluent": True}])
def test_a_merely_proposed_label_without_explicit_known_human_type_is_not_positive_evidence(fields):
    proposed = {"start_ms": 100, "end_ms": 200, "label": "filler_word", **fields}
    projected = project_targets(500, [proposed], [[0, 500]])
    assert not projected["multi_hot"].any() and not projected["type_mask"].any()
    assert projected["type_ambiguity"][0] and not projected["softmax_mask"][0]
    assert not projected["context_mask"].any()


def test_conflicting_contexts_mask_only_context_head_and_preserve_both_known_states():
    projected = project_targets(500, [event(100, 200, is_disfluent=False), event(300, 400, is_disfluent=True)], [[0, 500]])
    filler = column("filler_word")
    assert projected["context_multi_hot"][0, filler].tolist() == [True, True]
    assert projected["context_conflict"][0, filler] and not projected["context_mask"][0, filler]
    assert projected["softmax_mask"][0] and projected["multi_hot"][0, filler]


@pytest.mark.parametrize("coverage,unknown", [([[0, 250]], []), ([[0, 500]], [event(200, 300, None, decision="uncertain")]), ([[0, 500]], [event(200, 300, decision="not_evaluable")])])
def test_partial_or_unknown_cells_never_supply_fluent_or_unobserved_type_negatives(coverage, unknown):
    projected = project_targets(500, unknown, coverage)
    assert not projected["fully_reviewed"][0] and not projected["type_mask"][0].any()
    assert not projected["softmax_mask"][0] and projected["softmax_target"][0] == -1
    assert not projected["context_mask"][0].any()


def test_known_positive_survives_partial_review_without_unknown_other_labels_becoming_negative():
    projected = project_targets(500, [event(50, 100, is_disfluent=True)], [[0, 250]])
    filler = column("filler_word")
    assert projected["multi_hot"][0, filler] and projected["type_mask"][0, filler]
    assert projected["type_mask"][0].sum() == 1 and not projected["softmax_mask"][0]
    assert projected["context_multi_hot"][0, filler, 1] and not projected["context_mask"][0, filler]


def test_known_positive_survives_unknown_type_but_no_negative_or_single_softmax_is_fabricated():
    projected = project_targets(500, [event(50, 100, is_disfluent=True), event(200, 300, None, decision="uncertain")], [[0, 500]])
    filler = column("filler_word")
    assert projected["multi_hot"][0, filler] and projected["type_mask"][0, filler]
    assert projected["type_mask"][0].sum() == 1 and projected["type_ambiguity"][0]
    assert not projected["softmax_mask"][0] and not projected["context_mask"][0].any()


def test_fluent_requires_a_fully_reviewed_negative_cell_and_empty_media_is_explicit():
    projected = project_targets(501, [], [[0, 500]])
    assert projected["softmax_target"].tolist() == [0, -1]
    assert projected["softmax_mask"].tolist() == [True, False]
    assert projected["type_mask"][0].all() and not projected["type_mask"][1].any()
    empty = project_targets(0, [], [])
    assert empty["multi_hot"].shape == empty["type_mask"].shape == (0, len(TAXONOMY))
    assert empty["context_multi_hot"].shape == (0, len(TAXONOMY), 2)
    assert empty["softmax_target"].shape == (0,) and validate_targets(empty) == 0


@pytest.mark.parametrize("raw", [event(-1, 100), event(200, 100), event(0, 501), event(float("nan"), 100), event(0, float("inf")), event(0, 100, "fluent"), event(0, 100, None, type_known=True), event(0, 100, is_disfluent=0)])
def test_source_intervals_and_semantics_are_validated_not_clipped_or_reinterpreted(raw):
    with pytest.raises(ValueError):
        project_targets(500, [raw], [[0, 500]])


def test_projection_keeps_original_raw_data_and_is_a_distinct_masked_contract():
    raw = [event(100, 200, is_disfluent=False, event_id="synthetic-human-1", note="retain original")]
    coverage = [[250, 500], [0, 250]]
    before, coverage_before = deepcopy(raw), deepcopy(coverage)
    projected = project_targets(500, raw, coverage)
    assert raw == before and coverage == coverage_before
    assert projected["target_contract"] == TARGET_CONTRACT == "mrcd-target-projection-v2"
    assert projected["temporal_contract"] == TEMPORAL_CONTRACT and projected["requires_masked_training"] is True
    assert projected["source_events"] == raw and projected["reviewed_coverage"] == [[0, 500]]
    projected["source_events"][0]["note"] = "mutated output copy"
    assert raw == before


@pytest.mark.parametrize("broken", ["missing_mask", "wrong_shape", "nonfinite", "wrong_contract", "fluent_with_unknown", "context_without_type"])
def test_projected_arrays_require_aligned_shapes_masks_and_registered_contract(broken):
    projected = project_targets(500, [event(100, 200, is_disfluent=True)], [[0, 500]])
    if broken == "missing_mask":
        projected.pop("type_mask")
    elif broken == "wrong_shape":
        projected["context_mask"] = np.ones((1, 1), dtype=bool)
    elif broken == "nonfinite":
        projected["softmax_target"] = np.array([float("nan")])
    elif broken == "wrong_contract":
        projected["temporal_contract"] = "legacy"
    elif broken == "fluent_with_unknown":
        projected["fully_reviewed"][0] = False
        projected["softmax_target"][0] = 0
    else:
        projected["multi_hot"][0, column("filler_word")] = False
    with pytest.raises(ValueError):
        validate_targets(projected)


@pytest.mark.parametrize("coverage", [[[0, 501]], [[-1, 500]], [[100, 100]], [[300, 200]],
                                       [[0, float("nan")]], [[0]], [[False, 500]]])
def test_invalid_reviewed_coverage_is_not_clipped_or_silently_ignored(coverage):
    with pytest.raises(ValueError):
        project_targets(500, [], coverage)


def test_explicit_grid_and_source_inputs_remain_unchanged_and_returned_cells_are_copies():
    cells = evidence_windows(decision_cells(501))
    before = deepcopy(cells)
    projected = project_targets(501, [], [[0, 501]], cells=cells)
    assert cells == before
    projected["cells"][0]["owned_end_ms"] = 1
    assert cells == before


@pytest.mark.parametrize("field,value", [("type_known", 1), ("decision", "accepted"),
                                        ("type_known", True)])
def test_invalid_or_contradictory_evidence_flags_are_not_coerced(field, value):
    raw = event(100, 200, decision="not_evaluable")
    raw[field] = value
    with pytest.raises(ValueError):
        project_targets(500, [raw], [[0, 500]])


def test_known_type_with_uncertain_context_does_not_erase_other_observed_type_negatives():
    projected = project_targets(500, [event(100, 200, decision="uncertain", type_known=True)], [[0, 500]])
    assert projected["fully_reviewed"][0] and projected["type_mask"][0].all()
    assert projected["softmax_mask"][0] and projected["multi_hot"][0, column("filler_word")]
    assert not projected["context_multi_hot"].any() and not projected["context_mask"].any()


def test_unknown_context_mark_masks_same_type_aggregate_without_erasing_known_context_evidence():
    projected = project_targets(500, [event(100, 200, is_disfluent=False), event(300, 400)], [[0, 500]])
    filler = column("filler_word")
    assert projected["context_multi_hot"][0, filler].tolist() == [True, False]
    assert not projected["context_mask"][0, filler] and projected["context_target"][0, filler] == -1
    assert projected["softmax_mask"][0]


def test_projected_schema_requires_complete_evidence_not_only_owned_cells():
    projected = project_targets(500, [], [[0, 500]])
    for cell in projected["cells"]:
        for name in ("requested_start_ms", "requested_end_ms", "observed_start_ms", "observed_end_ms", "pad_left_ms", "pad_right_ms"):
            cell.pop(name)
    # A bare owned grid is valid input to the projector, not its complete output.
    assert "observed_end_ms" not in projected["cells"][0]
    with pytest.raises(ValueError):
        validate_targets(projected)
