"""terminal_window / span / Vc head / metadata of DisfluencyEngine.

V0 (the default) must stay bit-identical to the output of commit 33c0e58, stored in
``tests/data/golden_v0_synth.json`` (exact float reprs).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mrcd.constants import SAMPLE_RATE
from mrcd.engine import DisfluencyEngine
from mrcd.extractors.linguistic import Word

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "models" / "ckpt_audio_text.pt"
GOLDEN = Path(__file__).parent / "data" / "golden_v0_synth.json"
AMERESCO = REPO / "results" / "baselines_20261030" / "ameresco"
ASR_CACHE = REPO / "results" / "mejoras_20261030" / "asr_cache" / "ameresco"
VC = (
    REPO
    / "results"
    / "mejoras_20261030"
    / "checkpoints"
    / "ckpt_head_dihana_f123_zero.pt"
)
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="base checkpoint not present")
needs_vc = pytest.mark.skipif(
    not (CKPT.exists() and VC.exists()), reason="Vc checkpoint not present"
)
needs_pilot = pytest.mark.skipif(
    not (
        CKPT.exists() and (AMERESCO / "manifest.json").exists() and ASR_CACHE.exists()
    ),
    reason="AMERESCO clips / frozen ASR cache not present",
)

WORDS = [
    Word("bueno", "bueno", 1.0, 1.4, 0.9, ","),
    Word("eh", "eh", 3.0, 3.3, 0.8, ""),
    Word("seguimos", "seguimos", 5.2, 5.9, 0.95, "."),
    Word("entonces", "entonces", 9.0, 9.6, 0.9, ""),
    Word("este", "este", 12.0, 12.3, 0.7, ""),
]


def synth_audio(seed: int = 0, secs: int = 18) -> np.ndarray:
    """Deterministic 18 s signal; the golden file was generated from exactly this."""
    rng = np.random.default_rng(seed)
    t = np.arange(secs * SAMPLE_RATE) / SAMPLE_RATE
    f = 140 + 30 * np.sin(2 * np.pi * 0.4 * t)
    ph = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    env = (np.sin(2 * np.pi * 0.35 * t) > -0.3).astype(float)
    return (
        0.2 * env * (np.sin(ph) + 0.4 * np.sin(2 * ph))
        + 0.003 * rng.standard_normal(len(t))
    ).astype(np.float32)


class CachedASR:
    compute_type = "int8"
    device = "cpu"
    model_size = "stub"

    def __init__(self, words):
        self.words = words

    def transcribe_chunked(self, audio, chunk_s=8.0):
        return self.words

    def transcribe(self, audio):
        return self.words


def make_engine(words, head=None):
    from mrcd.inference import load_ckpt

    net, norm, labels, prov = load_ckpt(CKPT, "cpu")
    eng = DisfluencyEngine(net, norm, list(labels), CachedASR(words), prov, head=head)
    eng._device = "cpu"
    return eng


def flat(r):
    return {
        "windows": json.loads(
            json.dumps(
                [
                    [w.start, w.end, w.label, w.confidence, w.probabilities, w.text]
                    for w in r.windows
                ]
            )
        ),
        "events": json.loads(
            json.dumps(
                [
                    [
                        e.label,
                        e.start,
                        e.end,
                        e.confidence,
                        e.n_windows,
                        e.text,
                        list(e.modalities),
                    ]
                    for e in r.events
                ]
            )
        ),
        "transcript": r.transcript,
    }


@pytest.fixture(scope="module")
def v0_default():
    return make_engine(WORDS).analyze(synth_audio(), keep_windows=True)


# ------------------------------------------------------------------ V0 unchanged
@needs_ckpt
def test_default_output_is_bit_identical_to_the_pre_change_golden(v0_default):
    g = json.loads(GOLDEN.read_text("utf8"))
    got = flat(v0_default)
    assert got["windows"] == g["windows"]
    assert got["events"] == g["events"]
    assert got["transcript"] == g["transcript"]
    assert v0_default.duration == g["duration"] and v0_default.mode == g["mode"]
    assert v0_default.head is None and not any(w.terminal for w in v0_default.windows)


@needs_ckpt
def test_terminal_false_equals_default_and_true_extends_with_an_unchanged_grid(
    v0_default,
):
    eng = make_engine(WORDS)
    off = eng.analyze(synth_audio(), keep_windows=True, terminal_window=False)
    assert flat(off) == flat(v0_default)
    on = eng.analyze(synth_audio(), keep_windows=True, terminal_window=True)
    # 18 s: grid starts 0..15 step 0.5 end exactly at 15 = dur - 3 -> no extra window
    assert len(on.windows) == len(off.windows) and flat(on) == flat(off)


@needs_ckpt
def test_terminal_window_added_when_the_grid_does_not_end_at_the_end():
    audio = synth_audio(secs=18)[: int(17.7 * SAMPLE_RATE)]
    eng = make_engine(WORDS)
    off = eng.analyze(audio, keep_windows=True)
    on = eng.analyze(audio, keep_windows=True, terminal_window=True)
    assert len(on.windows) == len(off.windows) + 1
    term = on.windows[-1]
    assert term.terminal and not any(w.terminal for w in on.windows[:-1])
    assert abs(term.start - (len(audio) / SAMPLE_RATE - 3.0)) < 1e-9
    # the shared grid windows are bit-identical (features are extracted once over the whole audio)
    assert [(w.start, w.probabilities) for w in on.windows[:-1]] == [
        (w.start, w.probabilities) for w in off.windows
    ]
    # events never overlap, also around the terminal window
    ends = [e.end for e in on.events]
    starts = [e.start for e in on.events]
    assert all(starts[i + 1] >= ends[i] - 1e-9 for i in range(len(on.events) - 1))
    assert on.events[-1].end <= term.start + 0.5 + 1e-9
    assert on.metadata["window_plan"]["terminal_window"] is True


@needs_ckpt
def test_span_matches_the_pilot_plan_and_validates():
    audio = synth_audio()
    eng = make_engine(WORDS)
    r = eng.analyze(audio, keep_windows=True, terminal_window=True, span=(5.0, 13.7))
    starts = [w.start for w in r.windows]
    assert starts[0] == 5.0 and abs(starts[-1] - 10.7) < 1e-9 and r.windows[-1].terminal
    assert starts[:-1] == [5.0 + 0.5 * k for k in range(len(starts) - 1)]
    for bad in [(-1.0, 8.0), (5.0, 7.0), (5.0, 99.0)]:
        with pytest.raises(ValueError):
            eng.analyze(audio, span=bad)


@needs_ckpt
def test_metadata_records_the_effective_environment(v0_default):
    env = v0_default.metadata["environment"]
    for key in (
        "omp_num_threads",
        "effective",
        "set_before_numpy",
        "cpu_threads",
        "faster_whisper",
        "ctranslate2",
    ):
        assert key in env, key
    assert "torch" in env["cpu_threads"]
    assert v0_default.to_dict()["metadata"] == v0_default.metadata
    json.dumps(v0_default.to_dict())  # must stay plain JSON


# ------------------------------------------------------------------ pilot reproduction
def _pilot_rows():
    man = json.loads((AMERESCO / "manifest.json").read_text("utf8"))
    ref = {
        r["unit_id"]: int(r["prolongation_present"])
        for r in csv.DictReader(
            (AMERESCO / "references.csv").open(encoding="utf-8-sig")
        )
    }
    return [(m, ref[m["unit_id"]]) for m in man if m["in_pilot32"]]


def pilot_confusion(terminal: bool, n: int | None = None):
    import soundfile as sf

    from mrcd.inference import load_ckpt

    net, norm, labels, prov = load_ckpt(CKPT, "cpu")
    pi = list(labels).index("prolongation")
    tp = fp = fn = tn = 0
    for m, y in _pilot_rows()[:n]:
        audio, _ = sf.read(m["wav"], dtype="float32")
        words = [
            Word(**w)
            for w in json.loads((ASR_CACHE / f"{m['unit_id']}.json").read_text("utf8"))[
                "words"
            ]
        ]
        eng = DisfluencyEngine(net, norm, list(labels), CachedASR(words), prov)
        eng._device = "cpu"
        r = eng.analyze(
            audio,
            keep_windows=True,
            terminal_window=terminal,
            span=(m["target_start_s"], m["target_end_s"]),
        )
        pred = int(any(w.label == "prolongation" for w in r.windows))
        tp, fp, fn, tn = (
            tp + (y and pred),
            fp + ((not y) and pred),
            fn + (y and not pred),
            tn + ((not y) and (not pred)),
        )
    return tp, fp, fn, tn


@needs_pilot
def test_pilot_subset_terminal_window_only_adds_a_window_at_b_minus_3():
    import soundfile as sf

    m, _ = _pilot_rows()[0]
    audio, _ = sf.read(m["wav"], dtype="float32")
    words = [
        Word(**w)
        for w in json.loads((ASR_CACHE / f"{m['unit_id']}.json").read_text("utf8"))[
            "words"
        ]
    ]
    eng = make_engine(words)
    a, b = m["target_start_s"], m["target_end_s"]
    off = eng.analyze(audio, keep_windows=True, span=(a, b))
    on = eng.analyze(audio, keep_windows=True, span=(a, b), terminal_window=True)
    assert [w.start for w in on.windows[:-1]] == [w.start for w in off.windows]
    assert abs(on.windows[-1].start - (b - 3.0)) < 1e-9 and on.windows[-1].terminal


@needs_pilot
@pytest.mark.skipif(
    os.environ.get("MRCD_SLOW") != "1",
    reason="set MRCD_SLOW=1: runs the 32 pilot clips (~2 min)",
)
def test_pilot_32_clips_reproduce_5_3_11_13_with_terminal_and_3_1_13_15_without():
    assert pilot_confusion(True) == (5, 3, 11, 13)
    assert pilot_confusion(False) == (3, 1, 13, 15)


# ------------------------------------------------------------------ Vc (experimental, opt-in)
@needs_vc
def test_vc_requires_the_pinned_hash_and_rejects_a_modified_checkpoint(tmp_path):
    from mrcd import adaptation as A
    from mrcd.reproducibility import file_sha256

    base = file_sha256(CKPT)
    head = A.load_vc(base)
    assert head.sha256 == A.VC_SHA256 and head.classes == (
        "filler_word",
        "prolongation",
        "repetition",
    )
    bad = tmp_path / "vc_modified.pt"
    data = bytearray(VC.read_bytes())
    data[len(data) // 2] ^= 0x01
    bad.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        A.load_vc(base, path=bad)
    # a verbatim copy elsewhere is fine (the hash, not the location, is the identity)
    ok = tmp_path / "vc_copy.pt"
    shutil.copy(VC, ok)
    assert A.load_vc(base, path=ok).sha256 == A.VC_SHA256


@needs_vc
def test_hash_is_checked_before_unpickling(tmp_path, monkeypatch):
    import torch

    from mrcd import adaptation as A

    bad = tmp_path / "x.pt"
    bad.write_bytes(b"not a checkpoint")
    called = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="sha256"):
        A.load_vc(path=bad)
    assert not called, (
        "torch.load (pickle) must not run on bytes whose hash was not verified"
    )


@needs_vc
def test_custom_head_needs_an_explicit_sha256_and_the_right_base(tmp_path):
    from mrcd import adaptation as A

    with pytest.raises(ValueError, match="expected_sha256"):
        A.load_head(VC, expected_sha256="")
    with pytest.raises(ValueError, match="different base"):
        A.load_head(VC, expected_sha256=A.VC_SHA256, base_sha256="0" * 64)
    with pytest.raises(ValueError):
        DisfluencyEngine.load(CKPT, head="not-vc")


@needs_vc
def test_vc_does_not_change_v0_outputs_and_marks_everything_experimental(v0_default):
    from mrcd import adaptation as A
    from mrcd.reproducibility import file_sha256

    head = A.load_vc(file_sha256(CKPT))
    r = make_engine(WORDS, head=head).analyze(synth_audio(), keep_windows=True)
    # V0 windows/events/transcript are the same as without the head
    assert flat(r) == flat(v0_default)
    h = r.head
    assert h["experimental"] is True and h["scores_calibrated"] is False
    p = h["provenance"]
    assert (
        p["sha256"] == A.VC_SHA256
        and p["train_folds"] == [1, 2, 3]
        and p["selection_fold"] == 4
    )
    assert "turn" in p["task"] and p["classes"] == [
        "filler_word",
        "prolongation",
        "repetition",
    ]
    assert p["scores_are_calibrated_probabilities"] is False
    assert set(h["classes"]) == {"filler_word", "prolongation", "repetition"}
    for c in h["classes"].values():
        assert {"score", "threshold", "present"} <= set(c) and 0.0 <= c["score"] <= 1.0
    # classes without a compatible reference are NOT covered by Vc and not presented as validated
    assert set(h["unvalidated_classes"]) == {
        "block",
        "rhetorical_pause",
        "neutral_pause",
        "revision",
    }
    assert all(
        v["validated"] is False and v["source"] == "V0"
        for v in h["unvalidated_classes"].values()
    )
    assert not set(h["classes"]) & set(h["unvalidated_classes"])
    assert r.to_dict()["head"] == h and r.metadata["head"] == "vc"
    json.dumps(r.to_dict())
    assert all(
        w.head_scores is not None
        and w.head_scores["experimental"] is True
        and w.head_scores["calibrated"] is False
        and set(w.head_scores["scores"]) == set(h["classes"])
        for w in r.windows
    )


# ------------------------------------------------------------------ OMP warning at load
def test_omp_warning_text_only_when_unset_at_import():
    from mrcd.reproducibility import omp_status, omp_warning

    unset = {"omp_num_threads_at_import": None, "numpy_imported_before_mrcd": False}
    assert "not set" in omp_warning(omp_status(import_state=unset, environ={}))
    late = omp_warning(omp_status(import_state=unset, environ={"OMP_NUM_THREADS": "4"}))
    assert "after numpy" in late
    fine = {"omp_num_threads_at_import": "4", "numpy_imported_before_mrcd": False}
    assert (
        omp_warning(omp_status(import_state=fine, environ={"OMP_NUM_THREADS": "4"}))
        is None
    )
    st = omp_status(import_state=fine, environ={"OMP_NUM_THREADS": "4"})
    assert st["effective"] is True and st["set_before_numpy"] is True
    other = omp_status(
        import_state={
            "omp_num_threads_at_import": "8",
            "numpy_imported_before_mrcd": False,
        },
        environ={"OMP_NUM_THREADS": "8"},
    )
    assert other["effective"] is False and "differs from the pinned value" in omp_warning(other)
    unverifiable = omp_status(
        import_state={
            "omp_num_threads_at_import": "4",
            "numpy_imported_before_mrcd": True,
        },
        environ={"OMP_NUM_THREADS": "4"},
    )
    assert (
        unverifiable["effective"] is None and unverifiable["set_before_numpy"] is None
    )


@needs_vc
def test_experimental_marker_is_carried_by_events_windows_and_class_entries():
    from mrcd import adaptation as A
    from mrcd.reproducibility import file_sha256

    head = A.load_vc(file_sha256(CKPT))
    r = make_engine(WORDS, head=head).analyze(synth_audio(), keep_windows=True)
    assert r.events, "the synthetic scene must produce events"
    labelled = [e for e in r.events if e.label in r.head["classes"] or e.label in r.head["unvalidated_classes"]]
    assert labelled and all(e.validation for e in labelled)
    for e in labelled:
        if e.label in r.head["unvalidated_classes"]:
            assert "not validated" in e.validation
    assert all(c["experimental"] is True and c["calibrated"] is False for c in r.head["classes"].values())
    assert all("reference_tune_f1" in c for c in r.head["classes"].values())
    dumped = r.to_dict()["events"]
    assert all("validation" in d for d, e in zip(dumped, r.events) if e.validation)
    # without a head nothing is added (V0 dict unchanged)
    v0 = make_engine(WORDS).analyze(synth_audio())
    assert all("validation" not in d for d in v0.to_dict()["events"]) and v0.head is None


def test_vc_uncovered_reasons_hold_no_test_fold_figures():
    from mrcd.adaptation import VC_UNCOVERED

    assert not any("fold 5" in why or "F1 0." in why for why in VC_UNCOVERED.values())


@needs_ckpt
def test_terminal_tolerance_is_at_least_one_sample_and_one_acoustic_hop():
    base = synth_audio(secs=18)
    eng = make_engine(WORDS)
    # 16.0001 s: the would-be terminal start is 0.0001 s (1.6 samples) after the last grid start
    near = eng.analyze(base[: int(round(16.0001 * SAMPLE_RATE))], keep_windows=True, terminal_window=True)
    plain = eng.analyze(base[: int(round(16.0001 * SAMPLE_RATE))], keep_windows=True)
    assert len(near.windows) == len(plain.windows) and not any(w.terminal for w in near.windows)
    far = eng.analyze(base[: int(round(16.05 * SAMPLE_RATE))], keep_windows=True, terminal_window=True)
    assert far.windows[-1].terminal and abs(far.windows[-1].start - 13.05) < 1e-9
    from mrcd.constants import ACOUSTIC_HOP_S

    assert ACOUSTIC_HOP_S >= 1.0 / SAMPLE_RATE
