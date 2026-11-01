"""La ausencia de anotación no certifica ventanas humanas negativas."""

import copy

import numpy as np
import pytest

from mrcd.constants import BACKGROUND, KINESIC_DIM, LABEL2ID, TAXONOMY
from mrcd.dataset import build_recording_windows


@pytest.fixture
def recording(tmp_path):
    np.savez(
        tmp_path / "features.npz",
        ac_frames=np.zeros((600, 5), np.float32),
        word_feats=np.zeros((0, 10), np.float32),
        k_times=np.zeros(0),
        k_vectors=np.zeros((0, KINESIC_DIM)),
    )
    (tmp_path / "words.json").write_text("[]", encoding="utf-8")
    (tmp_path / "events.json").write_text("[]", encoding="utf-8")
    (tmp_path / "recording.eaf").write_text(
        '<ANNOTATION_DOCUMENT><TIER TIER_ID="human_disfluency"/></ANNOTATION_DOCUMENT>',
        encoding="utf-8",
    )
    return dict(
        recording_id="ejemplo", speaker_id="hablante", duration_s=6.0,
        features_npz="features.npz", words_json="words.json",
        events_json="events.json", eaf="recording.eaf", label_source="auto",
    )


@pytest.mark.parametrize("annotation", [
    None, {}, {"coverage": "unknown", "provenance": "fuente"},
    {"coverage": "positive_only", "provenance": "fuente"},
    {"coverage": "partial", "provenance": "fuente"},
    {"coverage": "complete"}, {"coverage": "complete", "provenance": "  "},
    {"coverage": "complete", "provenance": True},
])
def test_unknown_or_incomplete_coverage_cannot_create_negatives(tmp_path, recording, annotation):
    recording["human_annotation"] = annotation
    # La validación debe preceder incluso a la carga de características.
    recording["features_npz"] = "inexistente.npz"
    with pytest.raises(ValueError, match="human_annotation.*coverage='complete'"):
        build_recording_windows(recording, tmp_path, "human")


def _reviewed(recording):
    recording["human_annotation"] = {
        "coverage": "complete", "provenance": "registro-de-revisión/ejemplo",
    }
    return recording


def test_complete_review_with_no_events_is_valid_and_preserves_metadata(tmp_path, recording):
    _reviewed(recording)
    original = copy.deepcopy(recording)
    result = build_recording_windows(recording, tmp_path, "human")
    assert len(result["y"]) == 7
    assert np.all(result["y"] == LABEL2ID[BACKGROUND])
    assert not result["multi"].any()
    assert recording == original


@pytest.mark.parametrize("xml", [
    "<ANNOTATION_DOCUMENT/>",
    '<ANNOTATION_DOCUMENT><TIER TIER_ID="auto_candidates"/></ANNOTATION_DOCUMENT>',
])
def test_complete_declaration_does_not_replace_missing_human_tier(tmp_path, recording, xml):
    (tmp_path / "recording.eaf").write_text(xml, encoding="utf-8")
    with pytest.raises(ValueError, match="Falta el tier humano"):
        build_recording_windows(_reviewed(recording), tmp_path, "human")


def _write_event(tmp_path, category, start=500, end=1000):
    (tmp_path / "recording.eaf").write_text(
        f'<ANNOTATION_DOCUMENT><TIME_ORDER>'
        f'<TIME_SLOT TIME_SLOT_ID="a" TIME_VALUE="{start}"/>'
        f'<TIME_SLOT TIME_SLOT_ID="b" TIME_VALUE="{end}"/></TIME_ORDER>'
        '<TIER TIER_ID="human_disfluency"><ANNOTATION><ALIGNABLE_ANNOTATION '
        'TIME_SLOT_REF1="a" TIME_SLOT_REF2="b"><ANNOTATION_VALUE>'
        f'{category}</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION></TIER>'
        '</ANNOTATION_DOCUMENT>', encoding="utf-8",
    )


def test_reviewed_positive_and_negative_windows_keep_human_semantics(tmp_path, recording):
    _write_event(tmp_path, TAXONOMY[0])
    result = build_recording_windows(_reviewed(recording), tmp_path, "human")
    assert result["y"][0] == LABEL2ID[TAXONOMY[0]]
    assert result["multi"][0, 0] == 1
    assert result["y"][-1] == LABEL2ID[BACKGROUND]


@pytest.mark.parametrize("category,start,end", [
    ("desconocida", 500, 1000), ("", 500, 1000),
    (TAXONOMY[0], 1000, 500), (TAXONOMY[0], -1, 500),
    (TAXONOMY[0], 500, 7000),
])
def test_invalid_human_events_are_not_silently_converted_to_negatives(
    tmp_path, recording, category, start, end,
):
    _write_event(tmp_path, category, start, end)
    with pytest.raises(ValueError):
        build_recording_windows(_reviewed(recording), tmp_path, "human")


def test_missing_eaf_is_not_an_empty_review(tmp_path, recording):
    recording["eaf"] = "inexistente.eaf"
    with pytest.raises(ValueError, match="EAF existente"):
        build_recording_windows(_reviewed(recording), tmp_path, "human")


def test_auto_labels_do_not_require_human_coverage(tmp_path, recording):
    result = build_recording_windows(recording, tmp_path, "auto")
    assert np.all(result["y"] == LABEL2ID[BACKGROUND])
