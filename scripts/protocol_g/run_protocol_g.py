#!/usr/bin/env python
"""Protocol G harness (results/protocol_g_draft/PROTOCOL_G_DRAFT.md; frozen text: results/protocol_g/PROTOCOL_G.md).

Sub-commands (mirroring protocol F, plus the two-stage freeze of section 14)
----------------------------------------------------------------------------
gates          REFERENCE_GATES_G.json from the committed annotation CSVs (no system output exists)
freeze-stage1  hash PROTOCOL_G.md, kit files, manifests, the 19 mp3, the 19 transcripts, the aligner weights -> stage1 file
freeze-system  SYSTEM_FREEZE_G.json (system commit, tree hash, default profile, constants, pins, OD-4 record)
freeze-stage2  hash annotations, gates, harness, tests, system freeze -> stage2 file
leakage        the leakage check on its own
check          every precondition of ``run`` without running anything (alias: verify)
run            THE run: refuses unless every precondition holds; RUN_G.lock.json is created with O_EXCL first
power-sim      descriptive power simulation for PG2 (section 10.1); uses no ASR output

Nothing in this file touches audio, transcripts or models outside ``run`` / ``freeze-*`` (and those only on the
command of the author). Model and ASR calls sit behind ``Backend`` / ``ForcedAligner`` / ``CellSource`` so tests inject fakes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from scripts.protocol_g import common as C
from scripts.protocol_g import env_check as EC
from scripts.protocol_g import refs as R
from scripts.protocol_g import stats_g as S
from scripts.protocol_g.analysis import Config, Ctx, Inputs, evaluate, sensitivities
from scripts.protocol_g.build_reference_gates import build_gates
from scripts.protocol_g.ctc_align import AlignmentError, ForcedAligner, anchor_to_words
from scripts.protocol_g.leakage import leakage_check

H = C.H
Refusal = C.Refusal

# Corpus layout (protocol 3.3; same base as protocol F): <MRCD_CORPORA_ROOT>/02_NOCANDO_ES/audio/<manifest file> and
# <MRCD_CORPORA_ROOT>/02_NOCANDO_ES/transcripciones/<stem>_ts.txt
CORPUS_DIR = "02_NOCANDO_ES"
AUDIO_SUBDIR, TRANSCRIPT_SUBDIR, TRANSCRIPT_SUFFIX = (
    "audio",
    "transcripciones",
    "_ts.txt",
)
# protocol 5 / OD-4: the ASR profile id of o2026_13 = the key "ft_small_v1" of mrcd.asr_profiles.PROFILES and the value of
# mrcd.service.decision_profile.required_asr_profile("o2026_13") on the plan-F branch; freeze-system refuses otherwise
FT_ASR_PROFILE = "ft_small_v1"
SYSTEM_OF_PROFILE = {
    "o2026_11": "default",
    "v0": "V0",
    "o2026_12": "o2026_12",
    "o2026_13": "o2026_13",
}
ROLE_KEYS = ("adjudicated", "pass1", "pass2", "rater2")
STAGE1_DOCS = (
    "guide",
    "protocol_annotation",
    "annex",
    "manifest_unlabeled",
    "manifest_calibration",
)


# ------------------------------------------------------------------------------------------------ interfaces
class CellSource(Protocol):
    """Gives audio and transcript text per recording. The real one reads the corpus; tests inject synthetic data."""

    def audio(self, recording: str) -> bytes: ...
    def transcript(self, recording: str) -> str: ...


class Backend(Protocol):
    def decode_recording(self, recording: str, wav: bytes) -> dict:
        """-> {"events": {system: {cls: [(s,e)]}}, "words": {asr: [word dict]}, "failed_systems": [..], "failed_asr": [..]}"""

    def decode_cw2(self, recording: str, wav: bytes) -> dict:
        """-> {"words": [word dict] | None, "error": str | None}"""


def wav_to_array(wav: bytes) -> np.ndarray:
    import io

    import soundfile as sf

    a, _sr = sf.read(io.BytesIO(wav), dtype="float32")
    return a if a.ndim == 1 else a.mean(1)


# ------------------------------------------------------------------------------------------------ core (testable)
def run_core(
    items: list,
    human: R.HumanRef,
    gates: dict,
    source: CellSource,
    backend: Backend,
    aligner,
    classifier,
    cfg: Config,
    *,
    extra_humans: dict | None = None,
    progress=lambda *_: None,
) -> dict:
    """The single execution: R_T alignment (counts + gate BEFORE any metric), systems, cw2 arm, evaluation, sensitivities."""
    recs = sorted(R.recordings_of(items))
    # 1. R_T: aligner first
    texts, rt = {}, {}
    for rec in recs:
        progress(f"[align] {rec}")
        try:
            texts[rec] = R.transcript_segments(source.transcript(rec))
            toks, _fl, _seg = R.segment_tokens(texts[rec])
            out = aligner.align(wav_to_array(source.audio(rec)), toks)
            rt[rec] = R.rt_from_alignment(texts[rec], out)
        except (AlignmentError, ValueError) as exc:
            rt[rec] = R.RTRec(False, f"{type(exc).__name__}: {exc}")
        except (
            Exception,
            SystemExit,
        ) as exc:  # noqa: BLE001 - any aligner / audio-load failure makes R_T not_available for the recording
            rt[rec] = R.RTRec(False, f"{type(exc).__name__}: {exc}")
    # 2. systems, one decode per (recording, ASR profile)
    events = {s: {} for s in ("default", "o2026_12", "V0")}
    if cfg.include_ft:
        events["o2026_13"] = {}
    words = {"p0": {}, "p1": {}}
    if cfg.include_ft:
        words["ft"] = {}
    failed_sys = {s: set() for s in events}
    failed_asr = {a: set() for a in words}
    for rec in recs:
        progress(f"[systems] {rec}")
        try:
            r = backend.decode_recording(rec, source.audio(rec))
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - section 6: "recording failed" for EVERY system and ASR arm
            progress(f"[systems] {rec} failed: {type(exc).__name__}")
            r = {
                "events": {},
                "words": {},
                "failed_systems": list(events),
                "failed_asr": list(words),
            }
        for s in events:
            events[s][rec] = (
                (r["events"].get(s) or {}) if s not in r["failed_systems"] else {}
            )
            if s in r["failed_systems"]:
                failed_sys[s].add(rec)
        for a in words:
            words[a][rec] = r["words"].get(a) or []
            if a in r["failed_asr"] or a not in r["words"]:
                failed_asr[a].add(rec)
    # 3. S-P0ANCHOR timing (needs the p0 words; declared circular for p0_rule)
    rt_anchor = {}
    for rec in recs:
        if rec in failed_asr["p0"]:
            rt_anchor[rec] = R.RTRec(False, "p0 decode failed")
            continue
        if rec not in texts:
            rt_anchor[rec] = R.RTRec(False, rt[rec].reason)
            continue
        try:
            toks, _fl, _seg = R.segment_tokens(texts[rec])
            wt = R.asr_tokens(words["p0"][rec])
            total = max([e for _t, _s, e in wt] + [0.0])
            out = anchor_to_words(toks, wt, total)
            rec_rt = R.rt_from_alignment(texts[rec], out)
            gate = R.p0_anchor_gate(out, texts[rec])
            rec_rt.gate, rec_rt.gate_passed = gate, gate["passed"]
            rt_anchor[rec] = rec_rt
        except Exception as exc:  # noqa: BLE001 - the sensitivity loses the recording, nothing else
            rt_anchor[rec] = R.RTRec(False, f"{type(exc).__name__}: {exc}")
    # 4. cw2 arm, after the primary systems; any failure -> not_available, primary untouched
    if cfg.include_cw2:
        words["cw2"], failed_asr["cw2"] = {}, set()
        for rec in recs:
            progress(f"[cw2] {rec}")
            try:
                r = backend.decode_cw2(rec, source.audio(rec))
            except (Exception, SystemExit) as exc:  # noqa: BLE001
                r = {"words": None, "error": f"{type(exc).__name__}: {exc}"}
            if r.get("error") or r.get("words") is None:
                failed_asr["cw2"].add(rec)
                words["cw2"][rec] = []
            else:
                words["cw2"][rec] = r["words"]
    inp = Inputs(
        items,
        human,
        rt,
        events,
        words,
        failed_sys,
        failed_asr,
        gates,
        classifier,
        rt_anchor,
    )
    res = evaluate(inp, cfg)
    sens = sensitivities(inp, cfg)
    for name, hum in (
        extra_humans or {}
    ).items():  # S-RATER: pass 1 alone / pass 2 alone instead of the adjudicated file
        from scripts.protocol_g.analysis import _headline

        sens[f"S-RATER-{name}"] = _headline(
            evaluate(
                Inputs(
                    items,
                    hum,
                    rt,
                    events,
                    words,
                    failed_sys,
                    failed_asr,
                    gates,
                    classifier,
                    rt_anchor,
                ),
                cfg,
                descriptive=False,
            )
        )
    res["sensitivities"] = sens
    res["_ctx_cols"] = {k: v.tolist() for k, v in Ctx(inp, cfg)._v.items()}
    return res


# ------------------------------------------------------------------------------------------------ git helpers (repo-aware)
def _g(paths: C.Paths, *a: str) -> str:
    return H.git(*a, cwd=paths.repo)


def _committed(paths: C.Paths, p: Path) -> bool:
    return H.committed_clean(p, paths.repo)


def need_committed(paths: C.Paths, p: Path, what: str, check_git: bool = True) -> None:
    if check_git and not _committed(paths, p):
        raise Refusal(
            "not_committed", f"{what} ({p.name}) must be committed and unmodified"
        )


def read_json(p: Path, code: str, what: str) -> dict:
    if not p.is_file():
        raise Refusal(code, f"{what} missing: {p.name}")
    return json.loads(p.read_text("utf8"))


# ------------------------------------------------------------------------------------------------ system probe
def system_probe() -> dict:
    """What the installed system says about itself (imports mrcd at the checked-out commit)."""
    from importlib import metadata

    from mrcd.service import contextual as CX
    from mrcd.service import contract as CT
    from mrcd.service import decision_profile as DP

    libs = {}
    for pkg in (
        "faster-whisper",
        "ctranslate2",
        "torch",
        "transformers",
        "numpy",
        "scipy",
        "soundfile",
        "librosa",
    ):
        try:
            libs[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            libs[pkg] = None
    ft_profile = None
    # the profile names live in ``mrcd.service.contract.DECISION_PROFILE_NAMES`` and ``decision_profile.DECISION_PROFILES``
    # (``o2026_13`` appears in them only once ``feat/plan-f-ft-profile`` is merged)
    names = list(dict.fromkeys([*CT.DECISION_PROFILE_NAMES, *DP.DECISION_PROFILES]))
    if "o2026_13" in names:
        from mrcd import asr_profiles as AP

        want = DP.required_asr_profile("o2026_13")
        ft_profile = want if want in AP.PROFILES else None
    return {
        "default_decision_profile": DP.DEFAULT_DECISION_PROFILE,
        "decision_profiles": names,
        "o2026_13_asr_profile": ft_profile,
        "evidence": {
            "EVIDENCE_GAP_S": CX.EVIDENCE_GAP_S,
            "EVIDENCE_DUR_S": CX.EVIDENCE_DUR_S,
            "EVIDENCE_PUNCT": sorted(CX.EVIDENCE_PUNCT),
        },
        "lexical_fillers": sorted(CX.LEXICAL_FILLERS),
        "libs": libs,
    }


def _evidence_ok(ev: dict) -> bool:
    exp = C.EXPECTED_EVIDENCE
    return (
        float(ev.get("EVIDENCE_GAP_S")) == exp["EVIDENCE_GAP_S"]
        and float(ev.get("EVIDENCE_DUR_S")) == exp["EVIDENCE_DUR_S"]
        and sorted(ev.get("EVIDENCE_PUNCT") or []) == sorted(exp["EVIDENCE_PUNCT"])
    )


# ------------------------------------------------------------------------------------------------ freezes
def harness_files(paths: C.Paths) -> list[str]:
    out = []
    for pat in ("scripts/protocol_g", "tests"):
        d = paths.repo / pat
        if d.is_dir():
            for f in sorted(d.rglob("*.py")):
                r = f.relative_to(paths.repo).as_posix()
                if pat == "scripts/protocol_g" or Path(r).name.startswith(
                    ("test_protocol_g", "protocol_g_synth")
                ):
                    out.append(r)
    out += [
        "scripts/confirmatoria/run_confirmatoria.py",
        "scripts/annotation/agreement.py",
        "scripts/annotation/markers.py",
    ]
    return sorted(set(out))


def freeze_stage1(
    paths: C.Paths,
    docs: dict[str, Path],
    corpora_root: Path,
    aligner_weights_sha256: str,
) -> dict:
    """Digests only; transcripts are hashed from bytes and never displayed."""
    if paths.stage1.exists():
        raise Refusal(
            "already_frozen",
            f"{paths.stage1.name} exists: stage 1 is frozen (write a dated annex instead)",
        )
    if not paths.protocol.is_file():
        raise Refusal("no_protocol", "PROTOCOL_G.md missing")
    if aligner_weights_sha256 != C.ALIGNER_WEIGHTS_SHA256:
        raise Refusal(
            "aligner_weights", "aligner weights hash differs from the pinned value"
        )
    man = R.load_manifest(docs["manifest_unlabeled"])
    base = corpora_root / CORPUS_DIR
    mp3 = {}
    for stem, rows in sorted({i.recording: i for i in man}.items()):
        f = base / AUDIO_SUBDIR / rows.file
        h = H.sha256_file(f)
        if h != rows.sha256:
            raise Refusal(
                "mp3_mismatch", f"{rows.file} differs from the manifest sha256"
            )
        mp3[rows.file] = h
    tr = {
        stem: H.sha256_file(base / TRANSCRIPT_SUBDIR / f"{stem}{TRANSCRIPT_SUFFIX}")
        for stem in sorted({i.recording for i in man})
    }
    rec = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": H.sha256_text_file(paths.protocol),
        "files": {
            C.rel(p, paths.repo): H.sha256_text_file(p) for k, p in sorted(docs.items())
        },
        "kit_files": {
            f: H.sha256_text_file(paths.repo / f)
            for f in (
                "scripts/annotation/agreement.py",
                "scripts/annotation/markers.py",
            )
        },
        "mp3": mp3,
        "transcripts": tr,
        "aligner": {
            "model": C.ALIGNER_MODEL,
            "revision": C.ALIGNER_REVISION,
            "weights_sha256": aligner_weights_sha256,
            "files_sha256": C.aligner_expected_files(),
        },
        "note": "text files hashed after CRLF -> LF; mp3 and transcripts as raw bytes",
    }
    paths.stage1.write_text(C.dumps(rec) + "\n", "utf8")
    return rec


def freeze_system(
    paths: C.Paths,
    system_commit: str,
    *,
    whisper_model_bin: Path | None,
    ft_model_bin: Path | None,
    ft_branch_merged: bool,
    cw2_manifest_sha256: str = C.CW2_WEIGHTS_MANIFEST_SHA256,
    probe=system_probe,
) -> dict:
    if paths.freeze.exists():
        raise Refusal("already_frozen", f"{paths.freeze.name} exists")
    head = _g(paths, "rev-parse", "HEAD").strip()
    if len(system_commit) < 12 or not head.startswith(system_commit):
        raise Refusal(
            "system_commit",
            f"--system-commit {system_commit} (>= 12 hex chars) must be a prefix of HEAD {head}",
        )
    if not H.system_state_clean(paths.repo):
        raise Refusal(
            "system_dirty", "system paths have uncommitted or untracked changes"
        )
    p = probe()
    if p["default_decision_profile"] != C.EXPECTED_DEFAULT_PROFILE:
        raise Refusal(
            "default_profile",
            f"DEFAULT_DECISION_PROFILE resolves to {p['default_decision_profile']!r}, protocol tests {C.EXPECTED_DEFAULT_PROFILE!r}",
        )
    if not _evidence_ok(p["evidence"]):
        raise Refusal(
            "constants",
            "lexical-filler evidence constants differ from the protocol (section 5)",
        )
    wh = H.sha256_file(whisper_model_bin) if whisper_model_bin else None
    if wh != C.WHISPER_SMALL_MODEL_BIN_SHA256:
        raise Refusal(
            "whisper_weights",
            "Whisper small CT2 model.bin sha256 missing or differs from the pinned value",
        )
    ft_sha = H.sha256_file(ft_model_bin) if ft_model_bin else None
    if ft_branch_merged and p.get("o2026_13_asr_profile") != FT_ASR_PROFILE:
        raise Refusal(
            "ft_profile",
            f"o2026_13 resolves to ASR profile {p.get('o2026_13_asr_profile')!r}, protocol tests {FT_ASR_PROFILE!r}",
        )
    od4_included = bool(ft_branch_merged and ft_sha)
    tree = H.system_tree(paths.repo)
    rec = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system_commit": head,
        "system_paths": C.SYSTEM_PATHS,
        "system_tree_sha256": tree["sha256"],
        "system_tree_n_files": tree["n_files"],
        "default_decision_profile": p["default_decision_profile"],
        "decision_profiles": p["decision_profiles"],
        "evidence": p["evidence"],
        "lexical_fillers": p["lexical_fillers"],
        "asr": {
            "whisper_small_ct2_int8_model_bin_sha256": wh,
            "cw2": {
                "revision": C.CW2_REVISION,
                "weights_manifest_sha256": cw2_manifest_sha256,
            },
            "ft_small_v1": {
                "model_bin_sha256": ft_sha,
                "asr_profile_of_o2026_13": p.get("o2026_13_asr_profile"),
            },
        },
        "od4": {
            "branch_merged_at_freeze": bool(ft_branch_merged),
            "ft_model_bin_sha256": ft_sha,
            "ft_included": od4_included,
            "rule": "included iff merged AND model.bin pinned at the stage-2 freeze; decided by merge status alone",
        },
        "libs": p["libs"],
        "libs_expected": C.EXPECTED_LIBS,
        "environment_is_pinned": False,
    }
    paths.freeze.write_text(C.dumps(rec) + "\n", "utf8")
    return rec


def freeze_stage2(
    paths: C.Paths, roles: dict[str, Path | None], extra_agreement_jsons: list[Path]
) -> dict:
    if paths.stage2.exists():
        raise Refusal("already_frozen", f"{paths.stage2.name} exists")
    s1 = read_json(paths.stage1, "no_stage1", "stage-1 hash file")
    ann = {}
    for k, p in roles.items():
        if p is None:
            continue
        C.check_sidecar(p)
        ann[k] = {"path": C.rel(p, paths.repo), "sha256": H.sha256_file(p)}
    rec = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage1_file_sha256": H.sha256_text_file(paths.stage1),
        "stage1_protocol_sha256": s1["protocol_sha256"],
        "stage1_hashes_repeated": {
            k: s1[k] for k in ("files", "kit_files", "mp3", "transcripts", "aligner")
        },
        "annotations": ann,
        "agreement_jsons": {
            C.rel(p, paths.repo): H.sha256_file(p) for p in extra_agreement_jsons
        },
        "gates_sha256": H.sha256_file(paths.gates) if paths.gates.is_file() else None,
        "system_freeze_sha256": H.sha256_text_file(paths.freeze)
        if paths.freeze.is_file()
        else None,
        "harness": {
            f: H.sha256_text_file(paths.repo / f)
            for f in harness_files(paths)
            if (paths.repo / f).is_file()
        },
        "od4_record": (
            json.loads(paths.freeze.read_text("utf8")).get("od4")
            if paths.freeze.is_file()
            else None
        ),
    }
    paths.stage2.write_text(C.dumps(rec) + "\n", "utf8")
    return rec


# ------------------------------------------------------------------------------------------------ preflight
def regenerate_gates(paths: C.Paths, s2: dict) -> str:
    a = s2["annotations"]
    g = build_gates(
        paths.repo / a["pass1"]["path"],
        paths.repo / a["pass2"]["path"],
        (paths.repo / a["rater2"]["path"]) if "rater2" in a else None,
    )
    return C.dumps(g) + "\n"


def preflight(
    paths: C.Paths,
    *,
    system_commit: str | None = None,
    corpora_root: Path | None = None,
    check_git: bool = True,
    probe=system_probe,
    leak_check=True,
    env_probe=EC.import_probe,
    weights=None,
    allow_cw2_unavailable: bool = False,
) -> dict:
    """Every precondition of ``run``. Raises ``Refusal`` (with a code) on the first failure.

    ``env_probe`` really imports the packages the run needs (transformers, torch); ``weights`` is a locator (see
    ``env_check.real_locator``) whose files are HASHED and compared with the frozen record. ``check`` and ``run`` always
    pass both; ``None`` / a stub is for synthetic tests only."""
    # ---- stage 1: protocol (non-draft) + hashes + OSF registration
    if not paths.protocol.is_file():
        raise Refusal("no_protocol", "PROTOCOL_G.md missing")
    head_text = "\n".join(paths.protocol.read_text("utf8").splitlines()[:10])
    if "DRAFT" in head_text:
        raise Refusal(
            "protocol_is_draft",
            "PROTOCOL_G.md still carries the DRAFT marker in its header",
        )
    s1 = read_json(paths.stage1, "no_stage1", "PROTOCOL_G.stage1.sha256.json")
    psha = H.sha256_text_file(paths.protocol)
    if psha != s1["protocol_sha256"]:
        raise Refusal(
            "protocol_hash", "PROTOCOL_G.md does not match its stage-1 sha256"
        )
    osf1 = read_json(paths.osf1, "no_osf_stage1", "OSF_REGISTRATION_STAGE1.json")
    for k in ("url", "utc", "stage1_commit", "protocol_sha256"):
        if not osf1.get(k):
            raise Refusal(
                "osf_stage1_incomplete", f"OSF_REGISTRATION_STAGE1.json lacks {k!r}"
            )
    if osf1["protocol_sha256"] != psha:
        raise Refusal(
            "osf_stage1_hash",
            "OSF stage-1 registration cites a different PROTOCOL_G.md sha256",
        )
    for f, h in {**s1["files"], **s1["kit_files"]}.items():
        p = paths.repo / f
        if not p.is_file() or H.sha256_text_file(p) != h:
            raise Refusal("stage1_files", f"{f} differs from its stage-1 hash")
    if corpora_root is not None:
        base = corpora_root / CORPUS_DIR
        for f, h in s1["mp3"].items():
            if H.sha256_file(base / AUDIO_SUBDIR / f) != h:
                raise Refusal("data_mismatch", f"{f} differs from its stage-1 hash")
        for stem, h in s1["transcripts"].items():
            if (
                H.sha256_file(base / TRANSCRIPT_SUBDIR / f"{stem}{TRANSCRIPT_SUFFIX}")
                != h
            ):
                raise Refusal(
                    "data_mismatch",
                    f"transcript of {stem} differs from its stage-1 hash",
                )
    # ---- stage 2: annotations, gates, harness, system freeze
    s2 = read_json(paths.stage2, "no_stage2", "PROTOCOL_G.stage2.sha256.json")
    if s2["stage1_file_sha256"] != H.sha256_text_file(paths.stage1) or any(
        s2["stage1_hashes_repeated"][k] != s1[k] for k in s2["stage1_hashes_repeated"]
    ):
        raise Refusal(
            "stage2_stage1", "stage-2 file does not repeat the stage-1 hashes unchanged"
        )
    osf2 = read_json(paths.osf2, "no_osf_stage2", "OSF_REGISTRATION.json")
    if (
        not osf2.get("url")
        or not osf2.get("utc_upload")
        or osf2.get("stage2_sha256") != H.sha256_text_file(paths.stage2)
    ):
        raise Refusal(
            "osf_stage2",
            "OSF_REGISTRATION.json lacks the stage-2 upload record or cites another stage-2 hash",
        )
    if (
        "adjudicated" not in s2["annotations"]
        or "pass1" not in s2["annotations"]
        or "pass2" not in s2["annotations"]
    ):
        raise Refusal(
            "annotations_missing",
            "stage-2 file lacks adjudicated / pass1 / pass2 annotation files",
        )
    for k, rec in s2["annotations"].items():
        p = paths.repo / rec["path"]
        if not p.is_file():
            raise Refusal("annotations_missing", f"{rec['path']} missing")
        C.check_sidecar(p)
        if H.sha256_file(p) != rec["sha256"]:
            raise Refusal(
                "annotation_hash_mismatch",
                f"{rec['path']} differs from the stage-2 hash",
            )
        need_committed(paths, p, "annotation file", check_git)
        need_committed(
            paths, p.with_name(p.name + ".sha256"), "annotation sidecar", check_git
        )
    if not paths.gates.is_file():
        raise Refusal("no_gates", "REFERENCE_GATES_G.json missing")
    if H.sha256_file(paths.gates) != s2["gates_sha256"]:
        raise Refusal(
            "gates_hash", "REFERENCE_GATES_G.json differs from the stage-2 hash"
        )
    if paths.gates.read_text("utf8") != regenerate_gates(paths, s2):
        raise Refusal(
            "gates_regeneration",
            "REFERENCE_GATES_G.json is not what the generator computes from the committed annotations",
        )
    for f, h in s2["harness"].items():
        p = paths.repo / f
        if not p.is_file() or H.sha256_text_file(p) != h:
            raise Refusal("harness_hash", f"{f} differs from its stage-2 hash")
    # the SET of harness files is recomputed now: a file added after the stage-2 freeze (or one dropped) refuses
    now_set = {f for f in harness_files(paths) if (paths.repo / f).is_file()}
    if now_set != set(s2["harness"]):
        diff = sorted(now_set ^ set(s2["harness"]))
        raise Refusal(
            "harness_set",
            f"the harness file set differs from the stage-2 list: {diff[:5]}",
        )
    if not paths.freeze.is_file():
        raise Refusal("no_system_freeze", "SYSTEM_FREEZE_G.json missing")
    sf = json.loads(paths.freeze.read_text("utf8"))
    if H.sha256_text_file(paths.freeze) != s2["system_freeze_sha256"]:
        raise Refusal(
            "system_freeze_hash", "SYSTEM_FREEZE_G.json differs from the stage-2 hash"
        )
    for p in (
        paths.protocol,
        paths.stage1,
        paths.stage2,
        paths.osf1,
        paths.osf2,
        paths.gates,
        paths.freeze,
    ):
        need_committed(paths, p, "frozen file", check_git)
    for f in s2["harness"]:
        need_committed(paths, paths.repo / f, "harness file", check_git)
    if s1["aligner"].get("files_sha256") != C.aligner_expected_files():
        raise Refusal(
            "aligner_pin",
            "the stage-1 aligner file hashes differ from the pinned values of the harness",
        )
    # ---- system
    if system_commit is not None and (
        len(system_commit) < 12 or not sf["system_commit"].startswith(system_commit)
    ):
        raise Refusal(
            "system_commit",
            "--system-commit must be a (>= 12 hex chars) prefix of the frozen system commit",
        )
    if check_git:
        if not H.is_ancestor(sf["system_commit"], paths.repo):
            raise Refusal(
                "not_ancestor", "the frozen system commit is not an ancestor of HEAD"
            )
        if not H.system_state_clean(paths.repo):
            raise Refusal(
                "system_dirty", "system paths have uncommitted or untracked changes"
            )
        if H.system_tree(paths.repo)["sha256"] != sf["system_tree_sha256"]:
            raise Refusal(
                "system_tree", "system tree hash differs from the frozen value"
            )
    pr = probe()
    if (
        pr["default_decision_profile"] != C.EXPECTED_DEFAULT_PROFILE
        or sf["default_decision_profile"] != C.EXPECTED_DEFAULT_PROFILE
    ):
        raise Refusal(
            "default_profile",
            f"resolved default decision profile is {pr['default_decision_profile']!r}; a change of the tested default is a protocol change",
        )
    if not _evidence_ok(pr["evidence"]) or not _evidence_ok(sf["evidence"]):
        raise Refusal(
            "constants", "lexical-filler evidence constants differ from the protocol"
        )
    if sorted(pr["lexical_fillers"]) != sorted(sf["lexical_fillers"]):
        raise Refusal("constants", "LEXICAL_FILLERS differ from the frozen list")
    if (
        sf["asr"]["whisper_small_ct2_int8_model_bin_sha256"]
        != C.WHISPER_SMALL_MODEL_BIN_SHA256
    ):
        raise Refusal(
            "whisper_weights",
            "frozen Whisper small weights hash differs from the pinned value",
        )
    if (
        sf["asr"]["cw2"]["revision"] != C.CW2_REVISION
        or sf["asr"]["cw2"]["weights_manifest_sha256"] != C.CW2_WEIGHTS_MANIFEST_SHA256
    ):
        raise Refusal(
            "cw2_pin",
            "frozen cw2 revision / weights manifest differ from the pinned values",
        )
    od4 = sf["od4"]
    if od4["ft_included"] != bool(
        od4["branch_merged_at_freeze"] and od4["ft_model_bin_sha256"]
    ):
        raise Refusal(
            "od4",
            "OD-4 record is inconsistent (included iff merged AND model.bin pinned)",
        )
    if json.dumps(od4, sort_keys=True) != json.dumps(s2["od4_record"], sort_keys=True):
        raise Refusal("od4", "OD-4 record differs from the stage-2 file")
    if od4["ft_included"] and pr.get("o2026_13_asr_profile") != FT_ASR_PROFILE:
        raise Refusal(
            "ft_profile",
            f"o2026_13 resolves to ASR profile {pr.get('o2026_13_asr_profile')!r}, protocol tests {FT_ASR_PROFILE!r}",
        )
    lib_diff = {
        k: {"expected": v, "frozen": sf["libs"].get(k), "now": pr["libs"].get(k)}
        for k, v in C.EXPECTED_LIBS.items()
        if pr["libs"].get(k) != v
    }
    # ---- run environment (imports) and the weight files that will actually be loaded
    if env_probe is not None:
        bad = {m: e for m, e in env_probe().items() if e}
        if bad:
            raise Refusal(
                "environment",
                f"required packages not importable in this environment: {sorted(bad)}",
            )
    weights_report = None
    if weights is not None:
        weights_report = EC.verify_weights(
            sf,
            weights,
            include_ft=bool(od4["ft_included"]),
            allow_cw2_unavailable=allow_cw2_unavailable,
        )
    # ---- leakage
    leak = None
    if leak_check:
        man = R.load_manifest(
            paths.repo
            / next(k for k in s1["files"] if "manifest_nocando_unlabeled" in k)
        )
        leak = leakage_check(
            paths.repo,
            sorted({i.recording for i in man}),
            list(s1["transcripts"].values()),
        )
        if not leak["passed"]:
            raise Refusal(
                "leakage",
                f"leakage check failed: {[h['path'] for h in leak['hits']][:5]}",
            )
    return {
        "stage1": s1,
        "stage2": s2,
        "system_freeze": sf,
        "osf1": osf1,
        "osf2": osf2,
        "protocol_sha256": psha,
        "library_provenance_differences": lib_diff,
        "weights": weights_report,
        "leakage": leak,
        "head": _g(paths, "rev-parse", "HEAD").strip() if check_git else None,
    }


# ------------------------------------------------------------------------------------------------ lock
def _continuation(paths: C.Paths, continue_annex: str, check_git: bool) -> dict:
    """The crash annex must be a committed, non-empty file inside the repository; its sha256 is recorded."""
    p = Path(continue_annex)
    if not p.is_file() or p.stat().st_size == 0:
        raise Refusal("annex", "--continue-after-crash needs an existing, non-empty annex file")
    try:
        relp = C.rel(p, paths.repo)
    except ValueError:
        raise Refusal("annex", "the crash annex must live inside the repository") from None
    need_committed(paths, p, "crash annex", check_git)
    return {
        "annex": relp,
        "annex_sha256": H.sha256_text_file(p),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def acquire_lock(
    paths: C.Paths,
    payload: dict,
    continue_annex: str | None,
    check_git: bool = True,
) -> dict | None:
    """Create ``RUN_G.lock.json`` with O_EXCL. Returns the continuation record (annex path + sha256) on a crash
    recovery, else ``None``. Commit the lock immediately after creation (``git add -f``), see section 14."""
    paths.out.mkdir(parents=True, exist_ok=True)
    if paths.result.exists():
        raise Refusal(
            "already_run",
            "a result already exists: the evaluation was run once and cannot be repeated",
        )
    if paths.lock.exists():
        lock = json.loads(paths.lock.read_text("utf8"))
        st = lock.get("status")
        if st == "started" and continue_annex:
            rec = _continuation(paths, continue_annex, check_git)
            lock.setdefault("continuations", []).append(rec)
            paths.lock.write_text(json.dumps(lock, indent=1), "utf8")
            return rec  # crash recovery: the run restarts from the beginning with identical inputs
        raise Refusal(
            "lock_exists",
            f"{paths.lock.name} exists (status {st}): the run has been started; a rerun needs a dated annex",
        )
    fd = os.open(paths.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf8") as f:
        json.dump({**payload, "status": "started"}, f, indent=1)
    return None


def finish(paths: C.Paths, result: dict) -> None:
    paths.result.write_text(C.dumps(result) + "\n", "utf8")
    lock = json.loads(paths.lock.read_text("utf8"))
    lock.update({"status": "completed", "result_sha256": H.sha256_file(paths.result)})
    paths.lock.write_text(json.dumps(lock, indent=1), "utf8")


# ------------------------------------------------------------------------------------------------ real implementations
class RealSource:
    def __init__(self, root: Path, items: list):
        self.base = Path(root) / CORPUS_DIR
        self.files = {i.recording: i for i in items}

    def audio(self, rec: str) -> bytes:
        it = self.files[rec]
        p = self.base / AUDIO_SUBDIR / it.file
        if H.sha256_file(p) != it.sha256:
            raise Refusal(
                "data_mismatch", f"{it.file} differs from the manifest sha256"
            )
        return H.load_audio_wav_bytes(p)

    def transcript(self, rec: str) -> str:
        return R.read_transcript(
            (self.base / TRANSCRIPT_SUBDIR / f"{rec}{TRANSCRIPT_SUFFIX}").read_bytes()
        )


class MrcdBackend:
    """Real systems through ``MrcdAnalyzer``. One decode per (recording, ASR profile), shared by the profiles on it."""

    def __init__(
        self,
        analyzer,
        options_cls,
        cw2_analyzer_factory,
        cw2_options_factory,
        include_ft: bool,
    ):
        self.an, self.opt = analyzer, options_cls
        self.cw2_factory, self.cw2_opt = cw2_analyzer_factory, cw2_options_factory
        self.include_ft = include_ft
        self._cw2 = None

    @staticmethod
    def _words(ws) -> list[dict]:
        return [
            {
                "text": w.text,
                "norm": w.norm,
                "start": float(w.start),
                "end": float(w.end),
                "punct_after": getattr(w, "punct_after", ""),
            }
            for w in (ws or [])
        ]

    def _group(
        self,
        asr_id: str,
        profiles: list[str],
        wav: bytes,
        asr_profile: str | None = None,
    ) -> dict:
        cache = H.CachingASR(None)

        def wrapper(inner):
            object.__setattr__(cache, "_inner", inner)
            return cache

        out = {"events": {}, "failed": []}
        for prof in profiles:
            kw = {
                "decision_profile": prof,
                "include_text": False,
                "output_scope": "evaluated",
            }
            if asr_profile:
                kw["asr_profile"] = asr_profile
            doc = self.an.analyze_bytes(wav, self.opt(**kw), asr_wrapper=wrapper)
            sid = SYSTEM_OF_PROFILE[prof]
            if doc.get("error"):
                out["failed"].append(sid)
            else:
                out["events"][sid] = R.doc_events_g(doc)
        out["words"] = (
            self._words(cache.first_words) if cache.first_words is not None else None
        )
        return out

    def decode_recording(self, rec: str, wav: bytes) -> dict:
        res = {"events": {}, "words": {}, "failed_systems": [], "failed_asr": []}
        groups = [
            ("p0", ["o2026_11", "v0"], None),
            ("p1", ["o2026_12"], "p1_disfluent_prompt"),
        ]
        if self.include_ft:
            groups.append(("ft", ["o2026_13"], FT_ASR_PROFILE))
        for asr_id, profs, asr_profile in groups:
            g = self._group(asr_id, profs, wav, asr_profile)
            res["events"].update(g["events"])
            res["failed_systems"] += g["failed"]
            if g["words"] is None:
                res["failed_asr"].append(asr_id)
            else:
                res["words"][asr_id] = g["words"]
        return res

    def decode_cw2(self, rec: str, wav: bytes) -> dict:
        if self._cw2 is None:
            self._cw2 = (
                self.cw2_factory()
            )  # CUDA float16, no CPU fallback (annex F); raises if unavailable
        cache = H.CachingASR(None)

        def wrapper(inner):
            object.__setattr__(cache, "_inner", inner)
            return cache

        doc = self._cw2.analyze_bytes(
            wav,
            self.cw2_opt(
                decision_profile="v0", include_text=False, output_scope="evaluated"
            ),
            asr_wrapper=wrapper,
        )
        if doc.get("error") or cache.first_words is None:
            return {"words": None, "error": str(doc.get("error") or "no ASR output")}
        return {"words": self._words(cache.first_words), "error": None}


# ------------------------------------------------------------------------------------------------ commands
def _paths(a) -> C.Paths:
    return C.Paths(out=Path(a.out) if getattr(a, "out", None) else C.Paths().out)


def cmd_gates(a) -> int:
    paths = _paths(a)
    files = [a.pass1, a.pass2] + ([a.rater2] if a.rater2 else [])
    for p in files:
        need_committed(paths, Path(p), "annotation file", not a.no_git_check)
    g = build_gates(Path(a.pass1), Path(a.pass2), Path(a.rater2) if a.rater2 else None)
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.gates.write_text(C.dumps(g) + "\n", "utf8")
    print("wrote", paths.gates)
    return 0


def cmd_leakage(a) -> int:
    paths = _paths(a)
    s1 = read_json(paths.stage1, "no_stage1", "stage-1 hash file")
    man = R.load_manifest(
        paths.repo / next(k for k in s1["files"] if "manifest_nocando_unlabeled" in k)
    )
    r = leakage_check(
        paths.repo, sorted({i.recording for i in man}), list(s1["transcripts"].values())
    )
    print(
        json.dumps({k: r[k] for k in ("passed", "n_files_scanned")}),
        [h["path"] for h in r["hits"]],
    )
    return 0 if r["passed"] else 1


@dataclass
class RunObjects:
    """Everything ``run`` needs, built BEFORE the lock so that a construction failure cannot spend the single shot."""

    items: list
    human: object
    extra: dict
    gates: dict
    cfg: Config
    emitter: object
    aligner: object
    backend: object
    classifier: object
    source: object
    include_ft: bool


def _real_emitter(snapshot: Path, device: str):
    from scripts.protocol_g.ctc_align import Wav2Vec2Emitter

    return Wav2Vec2Emitter(snapshot, device=device)


def _real_backend(include_ft: bool):
    from functools import partial

    from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer

    return MrcdBackend(
        MrcdAnalyzer(max_duration_s=3600.0, device="cpu"),
        AnalysisOptions,
        lambda: MrcdAnalyzer(max_duration_s=3600.0, device="cuda"),
        partial(AnalysisOptions, asr_profile="cw2"),
        include_ft,
    )


def _real_classifier():
    from mrcd.service.contextual import classify_lexical_fillers

    return classify_lexical_fillers


REAL_FACTORIES = {
    "emitter": _real_emitter,
    "backend": _real_backend,
    "classifier": _real_classifier,
}


def build_run_objects(
    paths: C.Paths,
    st: dict,
    root: Path | None,
    snapshot: Path,
    *,
    factories: dict | None = None,
    device: str = "cuda",
) -> RunObjects:
    """Parse the manifest and annotation files and construct the aligner emitter, the analyzer / backends and the
    classifier. No audio, transcript or ASR call happens here."""
    fac = factories or REAL_FACTORIES
    s2 = st["stage2"]
    man_rel = next(
        k for k in st["stage1"]["files"] if "manifest_nocando_unlabeled" in k
    )
    items = R.load_manifest(paths.repo / man_rel)
    ann = lambda k: paths.repo / s2["annotations"][k]["path"]  # noqa: E731
    human = R.load_human(ann("adjudicated"), items)
    extra = {
        "pass1": R.load_human(ann("pass1"), items),
        "pass2": R.load_human(ann("pass2"), items),
    }
    gates = json.loads(paths.gates.read_text("utf8"))
    include_ft = bool(st["system_freeze"]["od4"]["ft_included"])
    emitter = fac["emitter"](snapshot, device)
    return RunObjects(
        items,
        human,
        extra,
        gates,
        Config(include_ft=include_ft),
        emitter,
        ForcedAligner(emitter),
        fac["backend"](include_ft),
        fac["classifier"](),
        RealSource(root if root is not None else Path("."), items),
        include_ft,
    )


def dry_construction(objs: RunObjects) -> dict:
    """~1 s synthetic-zeros forward pass of the aligner (no corpus audio)."""
    shape = objs.emitter.selftest()
    return {
        "emission_shape": list(shape),
        "device": getattr(objs.emitter, "device", None),
    }


def run_locked(
    paths: C.Paths,
    st: dict,
    objs: RunObjects,
    *,
    argv: list[str],
    continue_annex: str | None,
    check_git: bool = True,
    progress=print,
) -> dict:
    """THE run, after every object exists: take the lock, run, write the result. Data / runtime errors of one recording
    are that recording's "recording failed" outcome (section 6) inside ``run_core``; nothing else is caught.
    Commit the lock and the result immediately (``git add -f``), section 14."""
    cont = acquire_lock(
        paths,
        {
            "system_commit": st["system_freeze"]["system_commit"],
            "git_head_at_run": st["head"],
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_sha256": st["protocol_sha256"],
            "argv": EC.sanitize_argv(argv, paths.repo),
            "cw2_unavailable": (st.get("weights") or {}).get("cw2_unavailable_reason"),
        },
        continue_annex,
        check_git,
    )
    res = run_core(
        objs.items,
        objs.human,
        objs.gates,
        objs.source,
        objs.backend,
        objs.aligner,
        objs.classifier,
        objs.cfg,
        extra_humans=objs.extra,
        progress=progress,
    )
    cols = res.pop("_ctx_cols")
    loc = paths.out / "local"
    loc.mkdir(parents=True, exist_ok=True)
    import csv

    with (loc / "item_predictions.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        names = sorted(cols)
        w.writerow(["item_id", "recording", "speaker", *names])
        for i, it in enumerate(objs.items):
            w.writerow(
                [
                    it.item_id,
                    it.recording,
                    it.speaker,
                    *[
                        "" if cols[n][i] != cols[n][i] else int(cols[n][i])
                        for n in names
                    ],
                ]
            )
    gates = objs.gates
    final = {
        "label": "PROTOCOL G CONFIRMATORY RESULT against a human reference (see PROTOCOL_G.md); verdicts are 'supported on these recordings', never 'demonstrated'",
        "system_commit": st["system_freeze"]["system_commit"],
        "git_head_at_run": st["head"],
        "protocol_sha256": st["protocol_sha256"],
        "stage2_sha256": H.sha256_text_file(paths.stage2),
        "gates_sha256": H.sha256_file(paths.gates),
        "aligner": {
            "model": C.ALIGNER_MODEL,
            "revision": C.ALIGNER_REVISION,
            "weights_sha256": C.ALIGNER_WEIGHTS_SHA256,
            "device": getattr(objs.emitter, "device", None),
            "device_fallback": getattr(objs.emitter, "device_fallback", None),
        },
        "od4_ft_included": objs.include_ft,
        "weights_checked_before_the_lock": st.get("weights"),
        "library_provenance_differences": st["library_provenance_differences"],
        "environment": {
            **H.env_versions(),
            "transformers": _ver("transformers"),
            "environment_is_pinned": False,
        },
        "continued_after_crash": cont,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_quality_gates": {
            "classes": {
                c: {
                    "qg1": g["qg1"],
                    "qg3": g["qg3"],
                    "reference_status": g["reference_status"],
                }
                for c, g in gates["classes"].items()
            },
            "markers": gates["markers"],
            "qg2_repetition_state": gates["qg2_repetition_state"],
        },
        **res,
    }
    finish(paths, final)
    return final


def _ft_bin(a) -> Path | None:
    v = getattr(a, "ft_model_bin", None)
    return Path(v) if v else None


def cmd_check(a) -> int:
    paths = _paths(a)
    root = C.corpora_root(a.corpora_root) if a.check_data else None
    st = preflight(
        paths,
        system_commit=a.system_commit,
        corpora_root=root,
        probe=system_probe,  # looked up at call time (tests substitute it)
        env_probe=EC.import_probe,
        weights=EC.real_locator(_ft_bin(a)),
        allow_cw2_unavailable=bool(getattr(a, "allow_cw2_unavailable", False)),
    )
    print("OK: preflight (hashes, freeze, leakage, environment, weight files)")
    from scripts.protocol_g.ctc_align import locate_snapshot

    try:
        snapshot = locate_snapshot()
    except Exception as exc:  # noqa: BLE001
        print(
            f"SKIPPED dry construction: the aligner snapshot is not available locally ({type(exc).__name__}); "
            "nothing is downloaded"
        )
        return 0
    objs = build_run_objects(paths, st, root, snapshot, device=a.device)
    info = dry_construction(objs)
    print("OK: dry construction and synthetic forward pass", json.dumps(info))
    return 0


def cmd_run(a) -> int:
    paths = _paths(a)
    root = C.corpora_root(a.corpora_root)
    st = preflight(
        paths,
        system_commit=a.system_commit,
        corpora_root=root,
        probe=system_probe,  # looked up at call time (tests substitute it)
        env_probe=EC.import_probe,
        weights=EC.real_locator(_ft_bin(a)),
        allow_cw2_unavailable=bool(getattr(a, "allow_cw2_unavailable", False)),
    )
    from scripts.protocol_g.ctc_align import locate_snapshot

    # everything that can fail for an environment reason is built BEFORE the lock
    objs = build_run_objects(paths, st, root, locate_snapshot(), device="cuda")
    run_locked(
        paths,
        st,
        objs,
        argv=sys.argv[1:],
        continue_annex=a.continue_after_crash,
    )
    print("done:", paths.result)
    return 0


def _ver(pkg: str):
    from importlib import metadata

    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def cmd_freeze_stage1(a) -> int:
    paths = _paths(a)
    docs = {k: Path(getattr(a, k)) for k in STAGE1_DOCS}
    freeze_stage1(paths, docs, C.corpora_root(a.corpora_root), a.aligner_weights_sha256)
    print("wrote", paths.stage1)
    return 0


def cmd_freeze_system(a) -> int:
    paths = _paths(a)
    freeze_system(
        paths,
        a.system_commit,
        whisper_model_bin=Path(a.whisper_model_bin) if a.whisper_model_bin else None,
        ft_model_bin=Path(a.ft_model_bin) if a.ft_model_bin else None,
        ft_branch_merged=a.ft_branch_merged,
    )
    print("wrote", paths.freeze, "- commit it BEFORE freeze-stage2")
    return 0


def cmd_freeze_stage2(a) -> int:
    paths = _paths(a)
    roles = {k: (Path(getattr(a, k)) if getattr(a, k) else None) for k in ROLE_KEYS}
    freeze_stage2(paths, roles, [Path(p) for p in a.agreement_json])
    print("wrote", paths.stage2)
    return 0


POWER_GRID = (0.40, 0.50, 0.60, 0.70, 0.80)
POWER_RHOS = (0.0, 0.1, 0.3)
POWER_SEED = 20261108


def power_curve(
    events_per_recording: list[int],
    grid=POWER_GRID,
    rhos=POWER_RHOS,
    n_sims: int = 1000,
    n_boot: int = C.N_BOOT_TEST,
    seed: int = POWER_SEED,
    speakers: list[str] | None = None,
) -> dict:
    """Descriptive power of PG2 (protocol 10.1). Per simulated cell: survival per recording ~ beta-binomial (n_r events,
    mean = true survival, intra-recording correlation rho; rho = 0 is binomial; Beta(p*k, (1-p)*k) with k = (1-rho)/rho),
    then the SAME test and verdict rule as the run: recording bootstrap (seed 13, ``n_boot`` resamples, every recording
    of the cell a unit, zero-event recordings included), Holm over the fixed family of 3 with the two other primary
    members taken at p = 1 (so the PG2 Holm p is 3 p: the least favourable case), upper 95 % bound < 0.80, estimate
    < 0.80 and, when ``speakers`` is given, the LOSO rule (>= n - 1 speakers). A simulated cell whose test is not
    computable counts as a non-rejection."""
    rng = np.random.default_rng(seed)
    n = np.asarray(events_per_recording, int)
    out = {}
    for rho in rhos:
        for p in grid:
            rej = 0
            for _ in range(n_sims):
                if rho == 0:
                    q = np.full(len(n), p)
                else:
                    k = (1 - rho) / rho
                    q = rng.beta(p * k, (1 - p) * k, len(n))
                surv = rng.binomial(n, q)
                c = np.stack([surv, n], axis=1).astype(float)
                t = S.test_stat(
                    c,
                    S.rate_of,
                    null=C.SURVIVAL_T,
                    predicted="lt",
                    two_sided=False,
                    n_boot=n_boot,
                    n_ci=100,
                )
                if C.is_na(t):
                    continue
                lo = (
                    S.loso(c, S.rate_of, speakers, null=C.SURVIVAL_T, predicted="lt")
                    if speakers is not None
                    else {"passes": True}
                )
                ph = S.holm_fixed([t["p"], 1.0, 1.0])[0]
                v = S.verdict(t, lo, ph, family_size=3)
                rej += v["verdict"] == "supported_on_these_recordings"
            out[f"rho={rho}|true_survival={p}"] = rej / n_sims
    return {
        "seed": seed,
        "n_sims": n_sims,
        "n_boot": n_boot,
        "n_recordings": int(len(n)),
        "n_events": int(n.sum()),
        "loso_included": speakers is not None,
        "power": out,
        "note": "descriptive; changes no rule, threshold, floor or family; uses no ASR output",
    }


def cmd_power_sim(a) -> int:
    """Counts of H_disf REP events per recording from the committed adjudicated file, over EVERY recording of the
    manifest (zero-event recordings included, as in the run), with the manifest speakers for LOSO."""
    paths = _paths(a)
    items = R.load_manifest(Path(a.manifest))
    h = R.load_human(Path(a.adjudicated), items)
    rec_spk = R.recordings_of(items)
    recs = sorted(rec_spk)
    evs = R.human_event_list(h, "repetition", ("disfluent",))
    counts = [sum(1 for e in evs if e["recording"] == r) for r in recs]
    res = power_curve(
        counts, POWER_GRID, POWER_RHOS, a.n_sims, speakers=[rec_spk[r] for r in recs]
    )
    paths.out.mkdir(parents=True, exist_ok=True)
    (paths.out / "POWER_SIM_G.json").write_text(C.dumps(res) + "\n", "utf8")
    print("wrote", paths.out / "POWER_SIM_G.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", default=None, help="output directory (default results/protocol_g)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(n, *, corpora=False, syscommit=False):
        p = sub.add_parser(n)
        if corpora:
            p.add_argument(
                "--corpora-root", default=None, help="corpus root or $MRCD_CORPORA_ROOT"
            )
        if syscommit:
            p.add_argument("--system-commit", required=True)
        return p

    p = add("gates")
    p.add_argument("--pass1", required=True)
    p.add_argument("--pass2", required=True)
    p.add_argument("--rater2", default=None)
    p.add_argument("--no-git-check", action="store_true")
    add("leakage")
    for n in ("check", "verify"):
        p = add(n, corpora=True)
        p.add_argument("--system-commit", default=None)
        p.add_argument("--check-data", action="store_true")
        p.add_argument("--ft-model-bin", default=None)
        p.add_argument("--device", default="cuda")
        p.add_argument("--allow-cw2-unavailable", action="store_true")
    p = add("run", corpora=True, syscommit=True)
    p.add_argument(
        "--continue-after-crash",
        default=None,
        help="a COMMITTED crash annex inside the repository; its sha256 is recorded in the lock",
    )
    p.add_argument("--ft-model-bin", default=None)
    p.add_argument("--allow-cw2-unavailable", action="store_true")
    p = add("freeze-stage1", corpora=True)
    for k in STAGE1_DOCS:
        p.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    p.add_argument("--aligner-weights-sha256", required=True)
    p = add("freeze-system", syscommit=True)
    p.add_argument("--whisper-model-bin", default=None)
    p.add_argument("--ft-model-bin", default=None)
    p.add_argument("--ft-branch-merged", action="store_true")
    p = add("freeze-stage2")
    for k in ROLE_KEYS:
        p.add_argument("--" + k, default=None)
    p.add_argument("--agreement-json", action="append", default=[])
    p = add("power-sim")
    p.add_argument("--adjudicated", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--n-sims", type=int, default=1000)
    a = ap.parse_args(argv)
    return {
        "gates": cmd_gates,
        "leakage": cmd_leakage,
        "check": cmd_check,
        "verify": cmd_check,
        "run": cmd_run,
        "freeze-stage1": cmd_freeze_stage1,
        "freeze-system": cmd_freeze_system,
        "freeze-stage2": cmd_freeze_stage2,
        "power-sim": cmd_power_sim,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
