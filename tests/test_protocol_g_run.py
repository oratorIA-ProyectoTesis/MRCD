"""Refusal conditions, freezes, lock, leakage check and the single-shot core of the protocol G runner.

Everything runs in a temporary git repository with synthetic files; no corpus, no model, no ASR."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.protocol_g import common as C
from scripts.protocol_g import leakage as LK
from scripts.protocol_g import run_protocol_g as RUN
from scripts.protocol_g.analysis import Config
from scripts.protocol_g.build_reference_gates import build_gates
from scripts.protocol_g.ctc_align import AlignOutput, AlignmentError, TokenTimes
from tests.protocol_g_synth import (
    base_events,
    sidecar,
    synth_inputs,
    write_sheet,
)

RECS = ["anna_es_1", "anna_es_2", "beto_es_1"]
ITEMS = [f"it{i:03d}" for i in range(60)]
ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}
WHISPER_BYTES = b"fake whisper weights"
PROBE = {
    "default_decision_profile": "o2026_11",
    "decision_profiles": ["v0", "o2026_11", "o2026_11b", "o2026_12"],
    "evidence": {
        "EVIDENCE_GAP_S": 0.15,
        "EVIDENCE_DUR_S": 0.35,
        "EVIDENCE_PUNCT": [",", "...", "…"],
    },
    "lexical_fillers": ["a ver", "bueno", "este"],
    "libs": dict(C.EXPECTED_LIBS),
    "o2026_13_asr_profile": "ft_small_v1",
}


def sh(repo: Path, *a: str):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, env=ENV)


def commit(repo: Path, msg="step"):
    sh(repo, "add", "-A", "-f")
    sh(repo, "commit", "-q", "--allow-empty", "-m", msg)


def probe(**over):
    return lambda: {**PROBE, **over}


class World:
    def __init__(self, root: Path):
        self.root = root
        self.repo = root / "repo"
        self.corpus = root / "corpus"
        self.paths = C.Paths(out=self.repo / "results" / "protocol_g", repo=self.repo)


def build_world(
    tmp_path: Path, monkeypatch, *, tamper_gates=False, draft=False, skip=()
) -> World:
    w = World(tmp_path)
    repo, paths = w.repo, w.paths
    repo.mkdir()
    sh(repo, "init", "-q")
    monkeypatch.setattr(C, "ALIGNER_WEIGHTS_SHA256", "ab" * 32)
    monkeypatch.setattr(
        C, "WHISPER_SMALL_MODEL_BIN_SHA256", hashlib.sha256(WHISPER_BYTES).hexdigest()
    )
    for rel, txt in {
        "mrcd/__init__.py": "VALUE = 1\n",
        "pyproject.toml": "[project]\nname='x'\n",
        "scripts/annotation/agreement.py": "# kit\n",
        "scripts/annotation/markers.py": "# kit\n",
        "scripts/confirmatoria/run_confirmatoria.py": "# F\n",
        "scripts/protocol_g/x.py": "# harness\n",
        "tests/test_protocol_g_x.py": "# test\n",
        "docs/annotation_20261108/GUIA_ANOTACION.md": "guide\n",
        "docs/annotation_20261108/PROTOCOL_ANNOTATION.md": "protocol\n",
        "docs/annotation_20261108/ANNEX_20261108_SELF_ANNOTATION.md": "annex\n",
        "docs/annotation_20261108/manifest_nocando_calibration.csv": "item_id,file,sha256,start_s,end_s,speaker_if_in_name\n",
        "docs/notes.md": "nothing relevant\n",
    }.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, "utf-8", newline="\n")
    # corpus + manifest
    base = w.corpus / "02_NOCANDO_ES"
    (base / "audio").mkdir(parents=True)
    (base / "transcripciones").mkdir()
    rows = []
    for ri, rec in enumerate(RECS):
        b = f"mp3-bytes-{rec}".encode()
        (base / "audio" / f"{rec}.mp3").write_bytes(b)
        (base / "transcripciones" / f"{rec}_ts.txt").write_bytes(
            f"transcript text of {rec}".encode()
        )
        for k in range(20):
            rows.append(
                [
                    ITEMS[ri * 20 + k],
                    f"{rec}.mp3",
                    hashlib.sha256(b).hexdigest(),
                    20 * k,
                    20 * k + 8,
                    rec.split("_")[0],
                ]
            )
    mp = repo / "docs/annotation_20261108/manifest_nocando_unlabeled.csv"
    with mp.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(
            ["item_id", "file", "sha256", "start_s", "end_s", "speaker_if_in_name"]
        )
        wr.writerows(rows)
    paths.out.mkdir(parents=True)
    paths.protocol.write_text(
        ("# Protocol G - DRAFT\n" if draft else "# Protocol G\n") + "frozen text\n",
        "utf-8",
        newline="\n",
    )
    commit(repo, "docs")
    docs = {
        "guide": repo / "docs/annotation_20261108/GUIA_ANOTACION.md",
        "protocol_annotation": repo / "docs/annotation_20261108/PROTOCOL_ANNOTATION.md",
        "annex": repo / "docs/annotation_20261108/ANNEX_20261108_SELF_ANNOTATION.md",
        "manifest_unlabeled": mp,
        "manifest_calibration": repo
        / "docs/annotation_20261108/manifest_nocando_calibration.csv",
    }
    RUN.freeze_stage1(paths, docs, w.corpus, "ab" * 32)
    if "osf1" not in skip:
        paths.osf1.write_text(
            json.dumps(
                {
                    "url": "https://osf.io/x",
                    "utc": "2026-11-09T00:00:00Z",
                    "stage1_commit": "abc",
                    "protocol_sha256": C.H.sha256_text_file(paths.protocol),
                }
            ),
            "utf-8",
        )
    commit(repo, "stage1")
    d = repo / "docs/annotation_20261108"
    ev = base_events(ITEMS)
    write_sheet(d / "pass1_orig.csv", "author", ITEMS, ev)
    write_sheet(d / "pass2_orig.csv", "author", ITEMS, ev)
    write_sheet(
        d / "rater2_orig.csv",
        "rater2",
        ITEMS[:45],
        {k: v for k, v in ev.items() if k in ITEMS[:45]},
    )
    write_sheet(d / "adjudicated.csv", "author", ITEMS, ev)
    for n in ("pass1_orig", "pass2_orig", "rater2_orig", "adjudicated"):
        sidecar(d / f"{n}.csv")
    g = build_gates(d / "pass1_orig.csv", d / "pass2_orig.csv", d / "rater2_orig.csv")
    if tamper_gates:
        g["classes"]["repetition"]["reference_status"] = "ok-tampered"
    if "gates" not in skip:
        paths.gates.write_text(C.dumps(g) + "\n", "utf-8", newline="\n")
    commit(repo, "annotations")
    wh = w.root / "model.bin"
    wh.write_bytes(WHISPER_BYTES)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    if "freeze" not in skip:
        RUN.freeze_system(
            paths,
            head[:12],
            whisper_model_bin=wh,
            ft_model_bin=None,
            ft_branch_merged=False,
            probe=probe(),
        )
    commit(repo, "system freeze")
    roles = {
        k: d / f"{n}.csv"
        for k, n in (
            ("adjudicated", "adjudicated"),
            ("pass1", "pass1_orig"),
            ("pass2", "pass2_orig"),
            ("rater2", "rater2_orig"),
        )
    }
    RUN.freeze_stage2(paths, roles, [])
    if "osf2" not in skip:
        paths.osf2.write_text(
            json.dumps(
                {
                    "url": "https://osf.io/y",
                    "utc_upload": "2026-11-10T00:00:00Z",
                    "stage2_sha256": C.H.sha256_text_file(paths.stage2),
                }
            ),
            "utf-8",
        )
    commit(repo, "stage2")
    return w


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    w = build_world(tmp_path_factory.mktemp("world"), mp)
    yield w
    mp.undo()


@pytest.fixture()
def pin(monkeypatch):
    monkeypatch.setattr(C, "ALIGNER_WEIGHTS_SHA256", "ab" * 32)
    monkeypatch.setattr(
        C, "WHISPER_SMALL_MODEL_BIN_SHA256", hashlib.sha256(WHISPER_BYTES).hexdigest()
    )


@pytest.fixture()
def clone(world, tmp_path, pin):
    """A private copy of the frozen world (repo and corpus) to break one thing at a time."""
    dst = tmp_path / "w"
    shutil.copytree(world.root, dst, ignore=shutil.ignore_patterns("model.bin"))
    w = World(dst)
    return w


def pre(w: World, **kw):
    kw.setdefault("probe", probe())
    kw.setdefault("env_probe", lambda: {m: None for m in C.REQUIRED_IMPORTS})
    kw.setdefault("corpora_root", w.corpus)
    return RUN.preflight(w.paths, **kw)


def refuses(code: str, w: World, **kw):
    with pytest.raises(C.Refusal) as e:
        pre(w, **kw)
    assert e.value.code == code, str(e.value)


def test_a_complete_frozen_state_passes_every_precondition(world, pin):
    st = pre(world)
    assert (
        st["leakage"]["passed"] and st["system_freeze"]["od4"]["ft_included"] is False
    )
    assert st["library_provenance_differences"] == {}


def test_library_version_difference_is_recorded_not_refused(world, pin):
    st = pre(world, probe=probe(libs={**C.EXPECTED_LIBS, "faster-whisper": "9.9"}))
    assert "faster-whisper" in st["library_provenance_differences"]


# ---------------------------------------------------------------------------------------------- stage 1
def test_refuses_without_protocol(clone):
    clone.paths.protocol.unlink()
    refuses("no_protocol", clone)


def test_refuses_a_protocol_that_is_still_a_draft(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch, draft=True)
    refuses("protocol_is_draft", w)


def test_refuses_when_the_protocol_hash_differs_from_the_stage1_hash(clone):
    clone.paths.protocol.write_text("# Protocol G\nedited after the freeze\n", "utf-8")
    refuses("protocol_hash", clone)


def test_refuses_without_the_osf_stage1_registration(tmp_path, monkeypatch):
    refuses("no_osf_stage1", build_world(tmp_path, monkeypatch, skip=("osf1",)))


def test_refuses_an_osf_registration_citing_another_protocol_hash(clone):
    clone.paths.osf1.write_text(
        json.dumps(
            {"url": "u", "utc": "t", "stage1_commit": "c", "protocol_sha256": "0" * 64}
        ),
        "utf-8",
    )
    refuses("osf_stage1_hash", clone)


def test_refuses_an_incomplete_osf_registration(clone):
    clone.paths.osf1.write_text(json.dumps({"url": "u"}), "utf-8")
    refuses("osf_stage1_incomplete", clone)


def test_refuses_when_a_stage1_document_changed(clone):
    (clone.repo / "docs/annotation_20261108/GUIA_ANOTACION.md").write_text(
        "guide changed\n", "utf-8"
    )
    refuses("stage1_files", clone)


def test_refuses_when_corpus_data_differ_from_stage1(clone):
    (
        clone.corpus / "02_NOCANDO_ES" / "transcripciones" / "anna_es_1_ts.txt"
    ).write_bytes(b"other")
    refuses("data_mismatch", clone)
    (clone.corpus / "02_NOCANDO_ES" / "audio" / "anna_es_1.mp3").write_bytes(b"other")
    refuses("data_mismatch", clone)


# ---------------------------------------------------------------------------------------------- stage 2
def test_refuses_without_the_osf_stage2_record(tmp_path, monkeypatch):
    refuses("no_osf_stage2", build_world(tmp_path, monkeypatch, skip=("osf2",)))


def test_refuses_an_osf_stage2_record_citing_another_hash(clone):
    clone.paths.osf2.write_text(
        json.dumps({"url": "u", "utc_upload": "t", "stage2_sha256": "0" * 64}), "utf-8"
    )
    refuses("osf_stage2", clone)


def test_refuses_without_reference_gates(tmp_path, monkeypatch):
    refuses("no_gates", build_world(tmp_path, monkeypatch, skip=("gates",)))


def test_refuses_gates_that_differ_from_the_stage2_hash(clone):
    clone.paths.gates.write_text(
        clone.paths.gates.read_text("utf-8").replace("pass", "pxss", 1), "utf-8"
    )
    refuses("gates_hash", clone)


def test_refuses_gates_that_the_generator_does_not_reproduce(tmp_path, monkeypatch):
    refuses("gates_regeneration", build_world(tmp_path, monkeypatch, tamper_gates=True))


def test_refuses_tampered_or_unhashed_annotations(clone):
    f = clone.repo / "docs/annotation_20261108/adjudicated.csv"
    side = f.with_name(f.name + ".sha256")
    side.rename(side.with_name("moved"))
    refuses("annotation_unhashed", clone)
    side.with_name("moved").rename(side)
    f.write_text(f.read_text("utf-8") + "\n", "utf-8")
    refuses("annotation_hash_mismatch", clone)


def test_refuses_annotations_that_are_not_committed(clone):
    sh(clone.repo, "rm", "-q", "--cached", "docs/annotation_20261108/pass1_orig.csv")
    refuses("not_committed", clone)


def test_refuses_without_the_system_freeze(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch, skip=("freeze",))
    refuses("no_system_freeze", w)


def test_refuses_when_the_harness_changed_after_the_stage2_freeze(clone):
    (clone.repo / "scripts/protocol_g/x.py").write_text("# harness edited\n", "utf-8")
    refuses("harness_hash", clone)


def test_refuses_a_dirty_or_changed_system_tree(clone):
    (clone.repo / "mrcd/__init__.py").write_text("VALUE = 2\n", "utf-8")
    refuses("system_dirty", clone)
    commit(clone.repo, "change the system after the freeze")
    refuses("system_tree", clone)


def test_refuses_a_changed_default_profile_or_constants(world, pin):
    with pytest.raises(C.Refusal) as e:
        pre(world, probe=probe(default_decision_profile="o2026_12"))
    assert e.value.code == "default_profile"
    with pytest.raises(C.Refusal) as e:
        pre(world, probe=probe(evidence={**PROBE["evidence"], "EVIDENCE_GAP_S": 0.2}))
    assert e.value.code == "constants"
    with pytest.raises(C.Refusal) as e:
        pre(world, probe=probe(lexical_fillers=["bueno"]))
    assert e.value.code == "constants"


def test_refuses_a_system_commit_that_is_not_the_frozen_one(world, pin):
    with pytest.raises(C.Refusal) as e:
        pre(world, system_commit="0" * 12)
    assert e.value.code == "system_commit"


def test_refuses_on_leakage_in_a_tracked_file_or_an_ignored_results_file(clone):
    (clone.repo / "docs/notes.md").write_text("see anna_es_1 for details\n", "utf-8")
    commit(clone.repo, "notes mention a stem")
    refuses("leakage", clone)
    sh(clone.repo, "rm", "-q", "-f", "docs/notes.md")
    commit(clone.repo, "remove")
    cache = clone.repo / "results" / "some_cache"
    cache.mkdir()
    digest = json.loads(clone.paths.stage1.read_text("utf-8"))["transcripts"][
        "anna_es_1"
    ]
    (cache / "x.bin").write_bytes(
        b"\x00\x01" + digest.upper().encode() + b"\x02"
    )  # untracked, upper-case digest
    refuses("leakage", clone)


# ---------------------------------------------------------------------------------------------- freeze-system
def test_freeze_system_refuses_wrong_profile_constants_weights_commit_and_dirty_tree(
    tmp_path, monkeypatch
):
    w = build_world(tmp_path, monkeypatch, skip=("freeze", "osf2"))
    wh = w.root / "model.bin"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True
    ).stdout.strip()
    kw = dict(whisper_model_bin=wh, ft_model_bin=None, ft_branch_merged=False)
    for over, code in (
        ({"probe": probe(default_decision_profile="v0")}, "default_profile"),
        (
            {"probe": probe(evidence={**PROBE["evidence"], "EVIDENCE_DUR_S": 0.5})},
            "constants",
        ),
        ({"whisper_model_bin": None}, "whisper_weights"),
        ({"system_commit": "f" * 12}, "system_commit"),
    ):
        args = dict(system_commit=head[:12], probe=probe(), **kw)
        args.update(over)
        with pytest.raises(C.Refusal) as e:
            RUN.freeze_system(w.paths, args.pop("system_commit"), **args)
        assert e.value.code == code
    (w.repo / "mrcd/__init__.py").write_text("VALUE = 9\n", "utf-8")
    with pytest.raises(C.Refusal) as e:
        RUN.freeze_system(w.paths, head[:12], probe=probe(), **kw)
    assert e.value.code == "system_dirty"


def test_od4_is_decided_by_merge_status_and_pinned_weights_alone(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch, skip=("freeze", "osf2"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True
    ).stdout.strip()
    ft = w.root / "ft.bin"
    ft.write_bytes(b"ft weights")
    r = RUN.freeze_system(
        w.paths,
        head[:12],
        whisper_model_bin=w.root / "model.bin",
        ft_model_bin=ft,
        ft_branch_merged=True,
        probe=probe(),
    )
    assert (
        r["od4"]["ft_included"] is True
        and r["od4"]["ft_model_bin_sha256"] == hashlib.sha256(b"ft weights").hexdigest()
    )
    with pytest.raises(C.Refusal) as e:  # a second freeze is refused
        RUN.freeze_system(
            w.paths,
            head[:12],
            whisper_model_bin=w.root / "model.bin",
            ft_model_bin=None,
            ft_branch_merged=False,
            probe=probe(),
        )
    assert e.value.code == "already_frozen"


def test_freeze_system_refuses_a_merged_ft_branch_with_another_asr_profile_name(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch, skip=("freeze", "osf2"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True
    ).stdout.strip()
    ft = w.root / "ft.bin"
    ft.write_bytes(b"ft weights")
    with pytest.raises(C.Refusal) as e:
        RUN.freeze_system(
            w.paths,
            head[:12],
            whisper_model_bin=w.root / "model.bin",
            ft_model_bin=ft,
            ft_branch_merged=True,
            probe=probe(o2026_13_asr_profile="ft_small_v2"),
        )
    assert e.value.code == "ft_profile" and not w.paths.freeze.exists()


def test_freeze_stage1_is_one_time_and_checks_data_and_weights(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch)
    with pytest.raises(C.Refusal) as e:
        RUN.freeze_stage1(w.paths, {}, w.corpus, "ab" * 32)
    assert e.value.code == "already_frozen"
    s1 = json.loads(w.paths.stage1.read_text("utf-8"))
    assert set(s1["transcripts"]) == set(RECS) and len(s1["mp3"]) == 3


# ---------------------------------------------------------------------------------------------- lock / single shot
def test_lock_is_created_exclusively_and_a_second_run_is_refused(tmp_path):
    paths = C.Paths(out=tmp_path / "out", repo=tmp_path)
    RUN.acquire_lock(paths, {"utc": "t"}, None)
    assert json.loads(paths.lock.read_text("utf-8"))["status"] == "started"
    with pytest.raises(C.Refusal) as e:
        RUN.acquire_lock(paths, {"utc": "t"}, None)
    assert e.value.code == "lock_exists"
    with pytest.raises(
        C.Refusal
    ):  # a crash can only be continued with a non-empty annex
        RUN.acquire_lock(paths, {}, str(tmp_path / "missing.md"), check_git=False)
    annex = tmp_path / "ANNEX_crash.md"
    annex.write_text("crash declared", "utf-8")
    rec = RUN.acquire_lock(
        paths, {}, str(annex), check_git=False
    )  # continuation allowed, nothing else changes
    assert rec["annex"] == "ANNEX_crash.md" and len(rec["annex_sha256"]) == 64
    assert json.loads(paths.lock.read_text("utf-8"))["continuations"][0] == rec


def test_a_finished_run_can_never_be_repeated(tmp_path):
    paths = C.Paths(out=tmp_path / "out", repo=tmp_path)
    RUN.acquire_lock(paths, {}, None)
    RUN.finish(paths, {"x": 1})
    assert json.loads(paths.lock.read_text("utf-8"))["status"] == "completed"
    annex = tmp_path / "a.md"
    annex.write_text("x", "utf-8")
    with pytest.raises(C.Refusal) as e:
        RUN.acquire_lock(paths, {}, str(annex), check_git=False)
    assert e.value.code == "already_run"


# ---------------------------------------------------------------------------------------------- leakage
def test_leakage_whole_token_digest_case_and_allow_list(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    sh(repo, "init", "-q")
    files = {
        "a.txt": "maria_es_10 and xmaria_es_1 are different tokens",
        "b.txt": "the stem maria_es_1.mp3 appears here",
        "docs/annotation_20261108/ok.csv": "maria_es_1",
        "results/protocol_g/ok.json": "maria_es_1",
        "results/protocol_g_draft/ok.md": "maria_es_1",
        "scripts/protocol_g/ok.py": "maria_es_1",
        "tests/test_protocol_g_ok.py": "maria_es_1",
        "tests/test_other.py": "n/a",
    }
    for rel, txt in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, "utf-8")
    commit(repo)
    dig = "ab" * 32
    (repo / "results" / "cache").mkdir(parents=True)
    (repo / "results" / "cache" / "ignored.bin").write_bytes(dig.upper().encode())
    r = LK.leakage_check(repo, ["maria_es_1"], [dig])
    hits = {h["path"] for h in r["hits"]}
    assert hits == {"b.txt", "results/cache/ignored.bin"} and not r["passed"]
    assert "a.txt" not in hits  # neighbours that are token characters do not match
    assert LK.leakage_check(repo, ["nothing_es_9"], ["cd" * 32])["passed"]


# ---------------------------------------------------------------------------------------------- core with fakes
class FakeSource:
    SENTINEL = "sentinel corpus words that must never reach the result"

    def __init__(self, items, fail_align=(), bad_layout=()):
        self.items = items
        self.fail = set(fail_align)
        self.bad = set(bad_layout)

    def audio(self, rec):
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(
            buf, np.zeros(16000 * 5, np.float32), 16000, format="WAV", subtype="PCM_16"
        )
        return buf.getvalue()

    def transcript(self, rec):
        if rec in self.bad:
            return f"{self.SENTINEL} plain text without the declared markup"
        return (
            f'<recording id="{rec}" speaker="sentinelname"><transcript>'
            f'<segment id="1">{self.SENTINEL} la casa la casa eh</segment>'
            f'<segment id="2">{rec}</segment></transcript></recording>'
        )


class FakeAligner:
    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = 0

    def align(self, audio, tokens):
        self.calls += 1
        if self.fail and self.calls in self.fail:
            raise AlignmentError("synthetic failure")
        return AlignOutput(
            [
                TokenTimes(t, 0.1 * i, 0.1 * i + 0.08, 0.9, True)
                for i, t in enumerate(tokens)
            ]
        )


class FakeBackend:
    def __init__(self, inp, cw2_fail=False, fail_p0_on=()):
        self.inp, self.cw2_fail, self.fail_p0_on = inp, cw2_fail, set(fail_p0_on)

    def decode_recording(self, rec, wav):
        return {
            "events": {
                s: self.inp.events[s].get(rec, {})
                for s in ("default", "V0", "o2026_12")
            },
            "words": {a: self.inp.words[a].get(rec, []) for a in ("p0", "p1")},
            "failed_systems": [],
            "failed_asr": ["p0"] if rec in self.fail_p0_on else [],
        }

    def decode_cw2(self, rec, wav):
        if self.cw2_fail:
            raise RuntimeError("CUDA is not available on this machine")
        return {"words": self.inp.words["cw2"].get(rec, []), "error": None}


def run_fake(inp, **kw):
    items = inp.items
    fa = kw.pop("fail_align", ())
    bad = kw.pop("bad_layout", ())
    return RUN.run_core(
        items,
        inp.human,
        inp.gates,
        FakeSource(items, bad_layout=bad),
        FakeBackend(inp, **kw),
        FakeAligner(fa),
        inp.classifier,
        Config(n_boot_test=300, n_boot_ci=100),
        extra_humans={"pass1": inp.human, "pass2": inp.human},
    )


def test_run_core_end_to_end_with_fakes_and_no_corpus_text_in_the_result():
    inp = synth_inputs()
    res = run_fake(inp)
    res.pop("_ctx_cols")
    text = C.dumps(res)
    assert (
        FakeSource.SENTINEL.split()[0] not in text
        and "casa" not in text
        and "sentinelname" not in text
    )  # no transcript, transcript attribute or ASR text is serialised
    assert res["reference_R_T"]["n_recordings_aligner_failed"] == 0
    assert list(res)[:3] == [
        "cell",
        "reference_R_T",
        "systems",
    ]  # R_T counts and gate precede every metric
    assert (
        "S-RATER-pass1" in res["sensitivities"] and "S-P0ANCHOR" in res["sensitivities"]
    )
    assert res["cw2_arm"]["members"]["PGC2"]["status"] == "ok"


def test_cw2_failure_leaves_the_primary_result_byte_identical():
    inp = synth_inputs()
    ok = run_fake(inp)
    bad = run_fake(inp, cw2_fail=True)
    assert (
        bad["cw2_arm"]["status"] == "not_available"
        and "cw2" in bad["cw2_arm"]["reason"]
    )
    for k in ("primary", "secondary", "absolute", "reference_R_T", "cell"):
        a, b = dict(ok[k]), dict(bad[k])
        assert C.dumps(a) == C.dumps(b), k


def test_transcript_with_another_layout_makes_r_t_not_available_for_that_recording():
    res = run_fake(synth_inputs(), bad_layout={"sp0_es_1"})
    r_t = res["reference_R_T"]
    assert r_t["n_recordings_aligner_failed"] == 1 and r_t["interpretable"] is True
    rec = r_t["per_recording"]["sp0_es_1"]
    assert rec["available"] is False and "TranscriptFormatError" in rec["reason"]
    assert FakeSource.SENTINEL.split()[0] not in C.dumps({k: v for k, v in res.items() if k != "_ctx_cols"})
    assert res["sensitivities"]["S-P0ANCHOR"]  # the sensitivity copes with the missing transcript


def test_aligner_failure_marks_the_recording_not_available():
    res = run_fake(synth_inputs(), fail_align={1})
    r_t = res["reference_R_T"]
    assert r_t["n_recordings_aligner_failed"] == 1 and r_t["interpretable"] is True
    assert sum(1 for v in r_t["per_recording"].values() if not v["available"]) == 1
    res2 = run_fake(synth_inputs(), fail_align={1, 2})
    assert res2["reference_R_T"]["interpretable"] is False
    assert res2["primary"]["members"]["PG1"]["status"] == "not_available"


def test_p0_decode_failure_makes_p0_dependent_arms_not_available_in_the_run():
    inp = synth_inputs()
    res = run_fake(inp, fail_p0_on={"sp0_es_1"})
    assert res["primary"]["members"]["PG2"]["status"] == "not_available"
    assert res["asr"]["p0"]["n_failed"] == 1
    # the synthetic transcript does not match the p0 words: the sensitivity's own 70 % exact-match gate excludes everything
    assert res["sensitivities"]["S-P0ANCHOR"]["PG1"]["status"] == "not_available"


def test_power_curve_uses_every_recording_and_the_full_verdict_rule():
    counts = [6, 5, 0, 5, 6, 0, 5, 6]  # zero-event recordings stay resampling units
    spk = ["a", "a", "b", "b", "c", "c", "d", "d"]
    r = RUN.power_curve(counts, [0.3], [0.0], n_sims=20, n_boot=300, speakers=spk)
    assert r["n_recordings"] == 8 and r["n_events"] == 33 and r["loso_included"] is True
    assert RUN.POWER_GRID == (0.40, 0.50, 0.60, 0.70, 0.80) and RUN.POWER_RHOS == (0.0, 0.1, 0.3)
    # at the threshold itself the rule (Holm with the two other members at p = 1) almost never rejects
    r8 = RUN.power_curve(counts, [0.8], [0.0], n_sims=20, n_boot=300, speakers=spk)
    assert r8["power"]["rho=0.0|true_survival=0.8"] <= 0.1


def test_power_curve_is_deterministic_and_monotone_in_the_effect():
    r1 = RUN.power_curve(
        [6, 5, 4, 5, 6, 4, 5, 6], [0.4, 0.8], [0.0], n_sims=40, n_boot=300
    )
    r2 = RUN.power_curve(
        [6, 5, 4, 5, 6, 4, 5, 6], [0.4, 0.8], [0.0], n_sims=40, n_boot=300
    )
    assert r1 == r2
    assert (
        r1["power"]["rho=0.0|true_survival=0.4"]
        >= r1["power"]["rho=0.0|true_survival=0.8"]
    )


def test_harness_files_cover_the_runner_helpers_and_tests(clone):
    files = RUN.harness_files(clone.paths)
    assert (
        "scripts/protocol_g/x.py" in files
        and "tests/test_protocol_g_x.py" in files
        and "scripts/annotation/agreement.py" in files
    )


def test_run_command_refuses_before_creating_the_lock(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    with pytest.raises(SystemExit) as e:
        RUN.main(["--out", str(out), "run", "--system-commit", "a" * 12, "--corpora-root", str(tmp_path)])
    assert "REFUSED [no_protocol]" in str(e.value)
    assert not (out / C.LOCK_NAME).exists() and not (out / C.RESULT_NAME).exists()


def test_gates_command_writes_a_deterministic_file(tmp_path):
    a = write_sheet(tmp_path / "p1.csv", "author", ITEMS, base_events(ITEMS))
    b = write_sheet(tmp_path / "p2.csv", "author", ITEMS, base_events(ITEMS))
    out = tmp_path / "o"
    out.mkdir()
    for p in (a, b):
        sidecar(p)
    from scripts.protocol_g import build_reference_gates as G

    assert G.main(["--pass1", str(a), "--pass2", str(b), "--out", str(out / "g.json")]) == 0
    assert json.loads((out / "g.json").read_text("utf-8"))["classes"]["repetition"]["reference_status"] == "conditional"  # no rater 2


# ---------------------------------------------------------------------------------------------- revision 4: leakage
REPO_ROOT = Path(__file__).resolve().parents[1]
F_MANIFEST = "results/confirmatoria_20261107/PROTOCOL.sha256.json"


def _real_manifest_stems():
    man = RUN.R.load_manifest(
        REPO_ROOT / "docs/annotation_20261108/manifest_nocando_unlabeled.csv"
    )
    return sorted({i.recording for i in man})


def test_the_real_repository_tree_passes_the_leakage_check():
    """The real check on the real tree (19 stems; the transcript digests exist only after stage 1). Prints nothing."""
    stems = _real_manifest_stems()
    assert len(stems) == 19
    r = LK.leakage_check(REPO_ROOT, stems, [])
    assert r["passed"], [h["path"] for h in r["hits"]]
    assert r["n_files_scanned"] > 100


def test_the_protocol_f_hash_manifest_is_allowed_by_its_exact_path_only():
    assert F_MANIFEST in C.ALLOW_LIST
    # a sibling file under the same directory is not allowed
    assert not LK._allowed(
        "results/confirmatoria_20261107/CONFIRMATORY_RESULT.json", C.ALLOW_LIST
    )
    assert LK._allowed(F_MANIFEST, C.ALLOW_LIST)


def test_leakage_matches_a_stem_followed_by_underscore_and_scans_file_names(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    sh(repo, "init", "-q")
    for rel, txt in {
        "a.txt": "anna_es_1_ts.txt is the transcript",
        "docs/anna_es_2_notes.md": "no stem in the content",
        "b.txt": "anna_es_10 and xanna_es_1 are other tokens",
    }.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, "utf-8")
    commit(repo)
    r = LK.leakage_check(repo, ["anna_es_1", "anna_es_2"], [])
    assert {h["path"] for h in r["hits"]} == {"a.txt", "docs/anna_es_2_notes.md"}


# ---------------------------------------------------------------------------------------------- revision 4: parse
def test_self_closing_segment_is_an_empty_segment_and_swallows_nothing():
    segs = RUN.R.transcript_segments(
        '<r><transcript><segment id="1"/><segment id="2">hola</segment><segment id="3" /></transcript></r>'
    )
    assert segs == ["", "hola", ""]


# ---------------------------------------------------------------------------------------------- revision 4: probe
def test_the_real_system_probe_sees_the_decision_profiles_of_the_installed_system():
    p = RUN.system_probe()  # no stub: imports mrcd at the checked-out commit
    assert {"o2026_11", "o2026_12"} <= set(p["decision_profiles"])
    assert p["default_decision_profile"] == C.EXPECTED_DEFAULT_PROFILE
    # o2026_13 exists only once feat/plan-f-ft-profile is merged; the ASR profile is then resolved, never None
    if "o2026_13" in p["decision_profiles"]:
        assert p["o2026_13_asr_profile"] == RUN.FT_ASR_PROFILE
    else:
        assert p["o2026_13_asr_profile"] is None


# ---------------------------------------------------------------------------------------------- revision 4: weights
def _weights_world(tmp_path, monkeypatch):
    wb = tmp_path / "model.bin"
    wb.write_bytes(WHISPER_BYTES)
    snap = tmp_path / "snap"
    snap.mkdir()
    aux = {}
    for name in list(C.ALIGNER_AUX_FILES_SHA256) + ["pytorch_model.bin"]:
        b = f"fake {name}".encode()
        (snap / name).write_bytes(b)
        aux[name] = hashlib.sha256(b).hexdigest()
    monkeypatch.setattr(C, "ALIGNER_WEIGHTS_SHA256", aux.pop("pytorch_model.bin"))
    monkeypatch.setattr(C, "ALIGNER_AUX_FILES_SHA256", aux)
    monkeypatch.setattr(
        C, "WHISPER_SMALL_MODEL_BIN_SHA256", hashlib.sha256(WHISPER_BYTES).hexdigest()
    )
    cw = tmp_path / "cw2"
    cw.mkdir()
    (cw / "w.bin").write_bytes(b"cw2")
    man = RUN.EC.snapshot_manifest(cw)["combined_sha256"]
    monkeypatch.setattr(C, "CW2_WEIGHTS_MANIFEST_SHA256", man)
    sf = {
        "asr": {
            "whisper_small_ct2_int8_model_bin_sha256": C.WHISPER_SMALL_MODEL_BIN_SHA256,
            "cw2": {"weights_manifest_sha256": man},
        },
        "od4": {"ft_model_bin_sha256": None},
    }
    loc = {
        "whisper_model_bin": wb,
        "aligner_snapshot": snap,
        "cw2_snapshot": cw,
        "ft_model_bin": None,
    }
    return sf, loc, snap, wb, cw


def test_weights_are_hashed_and_every_mismatch_refuses(tmp_path, monkeypatch):
    sf, loc, snap, wb, cw = _weights_world(tmp_path, monkeypatch)
    rep = RUN.EC.verify_weights(sf, lambda: loc, include_ft=False)
    assert rep == {
        "whisper_small_model_bin": "match",
        "aligner_files": "match",
        "cw2_snapshot": "match",
    }
    (snap / "vocab.json").write_bytes(b"tampered")  # an aux file of the aligner
    with pytest.raises(C.Refusal) as e:
        RUN.EC.verify_weights(sf, lambda: loc, include_ft=False)
    assert e.value.code == "aligner_weights"
    (snap / "vocab.json").write_bytes(b"fake vocab.json")
    wb.write_bytes(b"other weights")
    with pytest.raises(C.Refusal) as e:
        RUN.EC.verify_weights(sf, lambda: loc, include_ft=False)
    assert e.value.code == "whisper_weights"
    wb.write_bytes(WHISPER_BYTES)
    (cw / "w.bin").write_bytes(b"other")
    with pytest.raises(C.Refusal) as e:
        RUN.EC.verify_weights(sf, lambda: loc, include_ft=False)
    assert e.value.code == "cw2_weights"


def test_an_absent_cw2_snapshot_refuses_unless_explicitly_allowed(
    tmp_path, monkeypatch
):
    sf, loc, snap, wb, cw = _weights_world(tmp_path, monkeypatch)
    none_loc = lambda: {**loc, "cw2_snapshot": None}  # noqa: E731
    with pytest.raises(C.Refusal) as e:
        RUN.EC.verify_weights(sf, none_loc, include_ft=False)
    assert e.value.code == "cw2_weights"
    assert "--allow-cw2-unavailable" in str(e.value)
    rep = RUN.EC.verify_weights(
        sf, none_loc, include_ft=False, allow_cw2_unavailable=True
    )
    assert rep["cw2_snapshot"] == "not_available_locally"
    assert rep["cw2_unavailable_allowed"] is True
    assert "allow-cw2-unavailable" in rep["cw2_unavailable_reason"]
    # a present snapshot never records the escape hatch, even when the flag is given
    rep = RUN.EC.verify_weights(sf, lambda: loc, include_ft=False, allow_cw2_unavailable=True)
    assert rep["cw2_snapshot"] == "match" and "cw2_unavailable_allowed" not in rep


def test_the_cw2_flag_is_a_cli_option_and_reaches_preflight(world, pin, monkeypatch):
    seen = {}

    def fake(sf, locate, include_ft, allow_cw2_unavailable=False, **k):
        seen["allow"] = allow_cw2_unavailable
        return {"stub": "match"}

    monkeypatch.setattr(RUN.EC, "verify_weights", fake)
    pre(world, weights=lambda: {})
    assert seen["allow"] is False  # default: refuse
    pre(world, weights=lambda: {}, allow_cw2_unavailable=True)
    assert seen["allow"] is True
    src = Path(RUN.__file__).read_text("utf8")
    assert src.count('"--allow-cw2-unavailable"') == 2  # check/verify and run


def test_missing_weights_refuse_other_than_cw2(tmp_path, monkeypatch):
    sf, loc, snap, wb, cw = _weights_world(tmp_path, monkeypatch)
    with pytest.raises(C.Refusal) as e:
        RUN.EC.verify_weights(
            sf, lambda: {**loc, "aligner_snapshot": None}, include_ft=False
        )
    assert e.value.code == "aligner_weights"
    with pytest.raises(C.Refusal) as e:  # OD-4 includes ft but no model.bin was given
        RUN.EC.verify_weights(sf, lambda: loc, include_ft=True)
    assert e.value.code == "ft_weights"
    ft = tmp_path / "ft.bin"
    ft.write_bytes(b"ft")
    sf["od4"]["ft_model_bin_sha256"] = hashlib.sha256(b"ft").hexdigest()
    got = RUN.EC.verify_weights(sf, lambda: {**loc, "ft_model_bin": ft}, include_ft=True)
    assert got["ft_model_bin"] == "match"
    ft.write_bytes(b"other ft")
    with pytest.raises(C.Refusal) as e:
        RUN.EC.verify_weights(sf, lambda: {**loc, "ft_model_bin": ft}, include_ft=True)
    assert e.value.code == "ft_weights"


def test_a_hub_cache_snapshot_of_another_revision_refuses(tmp_path, monkeypatch):
    sf, loc, snap, wb, cw = _weights_world(tmp_path, monkeypatch)
    other = tmp_path / "hub" / "snapshots" / ("0" * 40)
    shutil.copytree(snap, other)
    with pytest.raises(C.Refusal) as e:
        RUN.EC.check_aligner_snapshot(other)
    assert e.value.code == "aligner_revision"


def test_preflight_refuses_when_transformers_or_torch_cannot_be_imported(world, pin):
    with pytest.raises(C.Refusal) as e:
        pre(
            world,
            env_probe=lambda: {"transformers": "ModuleNotFoundError: x", "torch": None},
        )
    assert e.value.code == "environment"
    assert "transformers" in C.EXPECTED_LIBS and "torch" in C.EXPECTED_LIBS


def test_preflight_runs_the_weight_check(world, pin, monkeypatch):
    def boom(sf, locate, include_ft, **k):
        raise C.Refusal("whisper_weights", "x")

    monkeypatch.setattr(RUN.EC, "verify_weights", boom)
    refuses("whisper_weights", world, weights=lambda: {})


# ---------------------------------------------------------------------------------------------- revision 4: harness set
def test_a_harness_file_added_after_stage2_refuses(clone):
    (clone.repo / "scripts/protocol_g/extra.py").write_text("# added later\n", "utf-8")
    commit(clone.repo, "late addition")
    refuses("harness_set", clone)


def test_the_synthetic_fixture_module_is_part_of_the_harness_set(clone):
    (clone.repo / "tests").mkdir(exist_ok=True)
    (clone.repo / "tests/protocol_g_synth.py").write_text("# fixtures\n", "utf-8")
    (clone.repo / "tests/other_helper.py").write_text("# not harness\n", "utf-8")
    files = RUN.harness_files(clone.paths)
    assert "tests/protocol_g_synth.py" in files and "tests/other_helper.py" not in files


# ---------------------------------------------------------------------------------------------- revision 4: lock / annex
def test_sanitized_argv_has_no_absolute_path(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    argv = [
        "--out",
        str(repo / "results" / "protocol_g"),
        "run",
        "--corpora-root",
        "C:\\Users\\someone\\corpora",
        f"--continue-after-crash={repo / 'docs' / 'ANNEX.md'}",
        "--system-commit",
        "abcdef123456",
    ]
    out = RUN.EC.sanitize_argv(argv, repo)
    text = " ".join(out)
    assert "someone" not in text and str(tmp_path) not in text and "C:\\" not in text
    assert out[1] == "results/protocol_g"
    assert out[5] == "--continue-after-crash=docs/ANNEX.md"
    assert out[-1] == "abcdef123456"


def test_continue_after_crash_needs_a_committed_annex_and_records_its_hash(clone):
    paths = clone.paths
    RUN.acquire_lock(paths, {"utc": "t"}, None, check_git=False)
    annex = clone.repo / "docs" / "ANNEX_crash.md"
    annex.write_text("a crash happened; nothing else changes\n", "utf-8")
    with pytest.raises(C.Refusal) as e:  # present but not committed
        RUN.acquire_lock(paths, {}, str(annex))
    assert e.value.code == "not_committed"
    commit(clone.repo, "annex")
    rec = RUN.acquire_lock(paths, {}, str(annex))
    assert rec["annex"] == "docs/ANNEX_crash.md"
    assert rec["annex_sha256"] == C.H.sha256_text_file(annex)
    outside = clone.root / "outside.md"
    outside.write_text("x", "utf-8")
    with pytest.raises(C.Refusal) as e:
        RUN.acquire_lock(paths, {}, str(outside))
    assert e.value.code == "annex"


# ---------------------------------------------------------------------------------------------- revision 4: post-lock paths
class FakeEmitter:
    device, device_fallback = "cpu", False

    def selftest(self, seconds=1.0):
        return (50, 8)


class RaisingBackend:
    """Decode fails for named recordings with a runtime error; every other recording decodes to nothing."""

    def __init__(self, bad=(), cw2_bad=()):
        self.bad, self.cw2_bad = set(bad), set(cw2_bad)

    def decode_recording(self, rec, wav):
        if rec in self.bad:
            raise RuntimeError("synthetic decode crash")
        return {
            "events": {},
            "words": {"p0": [], "p1": []},
            "failed_systems": [],
            "failed_asr": [],
        }

    def decode_cw2(self, rec, wav):
        if rec in self.cw2_bad:
            raise RuntimeError("CUDA out of memory")
        return {"words": [], "error": None}


def _manifest_items(w: World):
    return RUN.R.load_manifest(
        w.repo / "docs/annotation_20261108/manifest_nocando_unlabeled.csv"
    )


def _objs(w: World, st, *, backend, source=None, aligner=None):
    items = _manifest_items(w)
    ann = lambda k: w.repo / st["stage2"]["annotations"][k]["path"]  # noqa: E731
    return RUN.RunObjects(
        items,
        RUN.R.load_human(ann("adjudicated"), items),
        {},
        json.loads(w.paths.gates.read_text("utf-8")),
        Config(n_boot_test=100, n_boot_ci=50),
        FakeEmitter(),
        aligner or FakeAligner(),
        backend,
        synth_inputs().classifier,
        source or FakeSource(items),
        False,
    )


def test_one_recording_that_crashes_after_the_lock_is_recording_failed_not_a_crash(
    clone,
):
    st = pre(clone)
    objs = _objs(clone, st, backend=RaisingBackend(bad={"anna_es_1"}))
    final = RUN.run_locked(
        clone.paths,
        st,
        objs,
        argv=["run", "--corpora-root", str(clone.corpus)],
        continue_annex=None,
        check_git=False,
        progress=lambda *_: None,
    )
    lock = json.loads(clone.paths.lock.read_text("utf-8"))
    assert lock["status"] == "completed" and clone.paths.result.is_file()
    assert str(clone.corpus) not in json.dumps(lock)
    assert "<absolute-path-redacted>" in json.dumps(lock)
    assert final["asr"]["p0"]["failed"] == ["anna_es_1"]
    assert final["systems"]["default"]["failed"] == ["anna_es_1"]
    assert FakeSource.SENTINEL.split()[0] not in clone.paths.result.read_text("utf-8")


def test_an_audio_or_aligner_failure_of_one_recording_never_escapes(clone):
    class BadAudio(FakeSource):
        def audio(self, rec):
            if rec == "beto_es_1":
                raise C.Refusal("data_mismatch", "differs from the manifest sha256")
            return super().audio(rec)

    st = pre(clone)
    objs = _objs(
        clone,
        st,
        backend=RaisingBackend(),
        source=BadAudio(_manifest_items(clone)),
        aligner=FakeAligner(fail={1}),
    )
    final = RUN.run_locked(
        clone.paths,
        st,
        objs,
        argv=[],
        continue_annex=None,
        check_git=False,
        progress=lambda *_: None,
    )
    assert final["reference_R_T"]["n_recordings_aligner_failed"] == 2
    assert "beto_es_1" in final["asr"]["p0"]["failed"]


def test_the_environment_is_built_before_the_lock_so_a_failure_spends_nothing(
    clone, monkeypatch
):
    import argparse

    from scripts.protocol_g import ctc_align

    head = json.loads(clone.paths.freeze.read_text("utf-8"))["system_commit"]
    monkeypatch.setattr(RUN, "_paths", lambda a: clone.paths)
    monkeypatch.setattr(RUN, "system_probe", probe())
    monkeypatch.setattr(
        RUN.EC, "import_probe", lambda: {m: None for m in C.REQUIRED_IMPORTS}
    )
    monkeypatch.setattr(RUN.EC, "verify_weights", lambda *a, **k: {"stub": "match"})
    monkeypatch.setattr(ctc_align, "locate_snapshot", lambda: clone.root)

    def no_transformers(snapshot, device):
        raise ModuleNotFoundError("No module named 'transformers'")

    monkeypatch.setitem(RUN.REAL_FACTORIES, "emitter", no_transformers)
    ns = argparse.Namespace(
        out=str(clone.paths.out),
        corpora_root=str(clone.corpus),
        system_commit=head[:12],
        continue_after_crash=None,
        ft_model_bin=None,
    )
    with pytest.raises(ModuleNotFoundError):
        RUN.cmd_run(ns)
    assert not clone.paths.lock.exists() and not clone.paths.result.exists()
    # with every object constructible the run proceeds to the lock (and stops there on purpose)
    monkeypatch.setitem(
        RUN.REAL_FACTORIES, "emitter", lambda snapshot, device: FakeEmitter()
    )
    monkeypatch.setitem(
        RUN.REAL_FACTORIES, "backend", lambda include_ft: RaisingBackend()
    )
    monkeypatch.setitem(
        RUN.REAL_FACTORIES, "classifier", lambda: synth_inputs().classifier
    )

    def stop(*a, **k):
        raise RuntimeError("stop after the lock")

    monkeypatch.setattr(RUN, "run_core", stop)
    with pytest.raises(RuntimeError, match="stop after the lock"):
        RUN.cmd_run(ns)
    assert clone.paths.lock.exists()


def test_check_builds_everything_and_runs_a_synthetic_forward_pass(clone):
    st = pre(clone)
    objs = RUN.build_run_objects(
        clone.paths,
        st,
        clone.corpus,
        clone.root,
        factories={
            "emitter": lambda snapshot, device: FakeEmitter(),
            "backend": lambda include_ft: RaisingBackend(),
            "classifier": lambda: synth_inputs().classifier,
        },
    )
    assert len(objs.items) == 60 and objs.include_ft is False
    assert RUN.dry_construction(objs) == {"emission_shape": [50, 8], "device": "cpu"}
    from scripts.protocol_g import ctc_align

    assert hasattr(ctc_align.Wav2Vec2Emitter, "selftest")


# ---------------------------------------------------------------------------------------------- revision 4: cw2, decomposition
def test_one_failed_cw2_recording_is_tolerated_two_are_not():
    inp = synth_inputs()

    def run(cw2_bad):
        fb = FakeBackend(inp)

        def cw2(rec, wav):
            if rec in cw2_bad:
                raise RuntimeError("CUDA out of memory")
            return {"words": inp.words["cw2"].get(rec, []), "error": None}

        fb.decode_cw2 = cw2
        res = RUN.run_core(
            inp.items,
            inp.human,
            inp.gates,
            FakeSource(inp.items),
            fb,
            FakeAligner(),
            inp.classifier,
            Config(n_boot_test=200, n_boot_ci=50),
        )
        return res

    one = run({"sp0_es_1"})
    members = one["cw2_arm"]["members"]
    assert members["PGC2"]["status"] == "ok"
    assert members["PGC2"]["excluded_recordings"] == ["sp0_es_1"]
    assert "decode failed" not in (members["PGC1"].get("reason") or "")
    two = run({"sp0_es_1", "sp0_es_2"})
    assert two["cw2_arm"]["status"] == "not_available"
    assert "more than 1" in two["cw2_arm"]["reason"]
    assert all(m["verdict"] == "not_evaluable" for m in two["cw2_arm"]["members"].values())


def test_the_decomposition_below_the_floor_is_counts_only_never_a_number():
    d = run_fake(synth_inputs())["descriptive"]["decomposition"]  # 12 R_T positives on S_dec (< 20)
    assert d["status"] == "not_available" and "origin" not in d


def test_the_decomposition_is_reported_with_paired_intervals_and_is_exact():
    from scripts.protocol_g import analysis as AN
    from tests.protocol_g_synth import fast_cfg

    inp = synth_inputs()
    res = AN.evaluate(inp, fast_cfg())
    d = res["descriptive"]["decomposition"]
    assert d["set"] == "S_dec" and d["origin_on_S_lit_see"] == "PG1"
    for k in ("origin", "convention", "total"):
        assert len(d[k]["ci95"]) == 2
    assert d["identity_residual"] == pytest.approx(0.0, abs=1e-12)
    pgs5 = res["secondary"]["members"]["PGS5"]["test"]["estimate"]
    assert d["total"]["estimate"] == pytest.approx(pgs5)


def test_an_order_flip_carries_its_paired_interval():
    from scripts.protocol_g import analysis as AN

    rng = np.random.default_rng(1)
    g = 8
    ci = np.stack(
        [rng.integers(8, 12, g), rng.integers(1, 4, g), rng.integers(1, 4, g)], 1
    ).astype(float)
    cj = np.stack(
        [rng.integers(5, 9, g), rng.integers(1, 4, g), rng.integers(1, 4, g)], 1
    ).astype(float)
    idx = np.random.default_rng(13).integers(0, g, (200, g))
    f = AN._flip_with_interval("a", "b", ("R_T", ci, cj), ("H_lit", cj, ci), idx)
    assert f["pair"] == ["a", "b"]
    for k in ("f1_diff_R_T", "f1_diff_H_lit"):
        lo, hi = f[k]["ci95"]
        assert lo <= hi
    assert f["f1_diff_R_T"]["estimate"] == pytest.approx(
        -f["f1_diff_H_lit"]["estimate"]
    )
