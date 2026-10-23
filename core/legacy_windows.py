"""Honest shared v1 classifier support; NOT localized disfluency ground truth."""
import numpy as np

LEGACY_WINDOW_CONTRACT = "legacy-forward3000-stride500-v1"


def legacy_window_predictions(probs, starts, labels, *, duration_ms, thresholds=None, score_kind="softmax_uncalibrated"):
    probs, starts = np.asarray(probs), np.asarray(starts)
    if (probs.shape != (len(starts), len(labels)) or starts.ndim != 1 or not np.isfinite(starts).all()
            or np.any(starts < 0) or np.any(np.diff(starts) <= 0) or not np.isfinite(probs).all()
            or np.any((probs < 0) | (probs > 1)) or not np.allclose(probs.sum(1), 1., atol=1e-5)
            or type(duration_ms) is not int or duration_ms < 0):
        raise ValueError("invalid legacy probabilities/support clock")
    thresholds = thresholds or {}
    groups, active = [], None
    for i, start in enumerate(starts):
        a, b = round(float(start) * 1000), min(duration_ms, round(float(start) * 1000) + 3000)
        if not 0 <= a < b:
            raise ValueError("legacy window starts outside actual duration")
        k = int(probs[i].argmax())
        label, p = labels[k], float(probs[i, k])
        if label in ("fluent", "background"):
            active = None
            continue
        window = dict(index=i, start_ms=a, end_ms=b, probabilities=probs[i].tolist(), labels=list(labels),
                      logits=None, logits_reason="legacy_predict_exposes_probabilities_only")
        decision = "candidate" if p >= thresholds.get(label, .5) else "uncertain"
        if active is not None and active["label"] == label and active["decision"] == decision and a - active["windows"][-1]["start_ms"] == 500:
            active["windows"].append(window)
            active.update(end_ms=b, end=b / 1000, n_windows=active["n_windows"] + 1, score=max(active["score"], p), peak_prob=max(active["peak_prob"], p))
        else:
            active = dict(prediction_kind="window_classification", window_contract=LEGACY_WINDOW_CONTRACT,
                          label=label, **{"class": label}, start_ms=a, end_ms=b, start=a / 1000, end=b / 1000,
                          n_windows=1, peak_prob=p, score=p, score_kind=score_kind, decision=decision,
                          contextual_state=None, is_disfluent=None, windows=[window],
                          temporal_evaluation_reason="legacy_window_support_not_temporal_events", boundary_source="classifier_evidence_support")
            groups.append(active)
    return groups
