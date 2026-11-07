#!/usr/bin/env python
"""Confirmatory harness (protocol: results/confirmatoria_20261107/PROTOCOL_CONFIRMATORIO.md).

Sub-commands
------------
freeze-protocol   one-time: hash the protocol, this file and the data bytes -> PROTOCOL.sha256.json
freeze-system     one-time, AFTER the improvements are merged: hash the system tree at --system-commit -> SYSTEM_FREEZE.json
verify            check every hash without running anything
run               THE confirmatory run. Refuses unless protocol hash, harness hash, system hash, data hashes and the
                  committed state all match the frozen values; writes RUN.lock.json first (atomic, O_EXCL) so the run
                  can happen once.
dry-run           development check on SYNTHETIC data with a stub system; writes only to --scratch, never to the official
                  output directory and never touches the lock.

This file is self-contained on purpose: the reference derivation and the statistics are vendored (numpy only), so the
frozen protocol does not depend on files of ``mrcd/`` that the system freeze is allowed to change. Parity with
``mrcd.context_eval.transcript_reference`` / ``metrics`` is asserted in ``tests/test_confirmatoria_harness.py``.

Nothing here reads, fits or tunes anything: every number comes from one execution over fixed cells.
"""

from __future__ import annotations

import os

# The ASR text depends on this value and a library cannot set it in time: set before numpy is imported (project rule).
if os.environ.get("OMP_NUM_THREADS") not in (None, "4"):
    raise SystemExit("OMP_NUM_THREADS must be unset or 4 (protocol section 9)")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import unicodedata  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "confirmatoria_20261107"
PROTOCOL_MD = OUT_DIR / "PROTOCOL_CONFIRMATORIO.md"
PROTOCOL_FREEZE = OUT_DIR / "PROTOCOL.sha256.json"
SYSTEM_FREEZE = OUT_DIR / "SYSTEM_FREEZE.json"
LOCK_NAME = "RUN.lock.json"
RESULT_NAME = "CONFIRMATORY_RESULT.json"
SELF = Path(__file__).resolve()

SEED = 13
N_BOOT_CI = 2000
N_BOOT_TEST = 5000
GATE_MIN_POS = 30
GATE_MIN_RECORDINGS = 3
GATE_MIN_SPEAKERS = 3
MAX_FAILURE_RATE = 0.10
FLOOR_F1_REPETITION = 0.30
FLOOR_RECALL_FILLER = 0.40
SYSTEM_PATHS = ["mrcd", "models", "pyproject.toml"]

# --------------------------------------------------------------------------------------------- hashing / git


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text_file(p: Path) -> str:
    """Hash of a text file with CRLF normalised to LF (a Windows checkout must match the frozen value)."""
    return sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git(*args: str, cwd: Path | None = None) -> str:
    cwd = cwd or REPO
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def system_tree(repo: Path | None = None) -> dict:
    repo = repo or REPO
    """Hash of every tracked file of the system paths (code, packaged profiles, checkpoint, pyproject)."""
    files = [
        f
        for f in git("ls-files", "-z", "--", *SYSTEM_PATHS, cwd=repo).split("\0")
        if f and not f.endswith(".pyc")
    ]
    entries = []
    for f in sorted(files):
        p = repo / f
        if p.is_file():
            entries.append((f, sha256_file(p)))
    digest = sha256_bytes("".join(f"{f}\0{h}\n" for f, h in entries).encode())
    return {"sha256": digest, "n_files": len(entries)}


def system_state_clean(repo: Path | None = None) -> bool:
    repo = repo or REPO
    out = git(
        "status", "--porcelain", "--untracked-files=all", "--", *SYSTEM_PATHS, cwd=repo
    )
    return out.strip() == ""


def is_ancestor(commit: str, repo: Path | None = None) -> bool:
    repo = repo or REPO
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=repo
    )
    return r.returncode == 0


def committed_clean(path: Path, repo: Path | None = None) -> bool:
    repo = repo or REPO
    rel = path.resolve().relative_to(repo.resolve()).as_posix()
    tracked = git("ls-files", "--", rel, cwd=repo).strip()
    dirty = git("status", "--porcelain", "--", rel, cwd=repo).strip()
    return bool(tracked) and not dirty


def env_versions() -> dict:
    from importlib import metadata

    out = {
        "python": sys.version.split()[0],
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    for pkg in (
        "numpy",
        "torch",
        "faster-whisper",
        "ctranslate2",
        "librosa",
        "soundfile",
        "scipy",
    ):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


# ------------------------------------------------------------------------------- reference (vendored, frozen)

#: Non-lexical filled pauses; identical to ``mrcd.context_eval.transcript_reference.NONLEXICAL_FILLER_RE``.
NONLEXICAL_FILLER_RE = re.compile(r"^(e+h*|eh+|ehm+|em+|mm+|hm+|ah+|aa+h*)$")
MAX_N = 4


def norm_tokens(s: str) -> list[str]:
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9ñ ]+", " ", s).split()


def is_filler(tok: str, nxt: str | None) -> bool:
    if not NONLEXICAL_FILLER_RE.match(tok):
        return False
    if tok == "e" and nxt is not None and (nxt.startswith("i") or nxt.startswith("hi")):
        return False  # conjunction "e"
    return True


def split_fillers(toks: list[str]) -> tuple[list[str], list[str]]:
    keep, fill = [], []
    for i, t in enumerate(toks):
        (
            fill if is_filler(t, toks[i + 1] if i + 1 < len(toks) else None) else keep
        ).append(t)
    return keep, fill


def repeat_positions(tokens: list[str], max_n: int = MAX_N) -> list[tuple[int, int]]:
    """(start index of the SECOND occurrence, n) of every immediate exact repetition of 1..max_n tokens."""
    out = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - 2 * n + 1):
            if tokens[i : i + n] == tokens[i + n : i + 2 * n]:
                out.append((i + n, n))
    return out


def reference_of(text: str) -> dict:
    """Reference of one unit, derived ONLY from the orthographic transcription (protocol section 3)."""
    toks = norm_tokens(text)
    keep, fill = split_fillers(toks)
    return {
        "rep_literal": int(bool(repeat_positions(keep))),
        "n_fillers": len(fill),
        "n_tokens": len(keep),
    }


# ------------------------------------------------------------------------------------- statistics (vendored)


def _group_index(groups) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def group_counts(y, p, groups) -> np.ndarray:
    """(G, 3): tp, fp, fn per group."""
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    inv, g = _group_index(groups)
    c = np.zeros((g, 3))
    np.add.at(c[:, 0], inv, (y == 1) & (p == 1))
    np.add.at(c[:, 1], inv, (y == 0) & (p == 1))
    np.add.at(c[:, 2], inv, (y == 1) & (p == 0))
    return c


def f1_of(c) -> np.ndarray:
    c = np.asarray(c, float)
    den = 2 * c[..., 0] + c[..., 1] + c[..., 2]
    return np.where(den > 0, 2 * c[..., 0] / np.maximum(den, 1), np.nan)


def recall_of(c) -> np.ndarray:
    c = np.asarray(c, float)
    den = c[..., 0] + c[..., 2]
    return np.where(den > 0, c[..., 0] / np.maximum(den, 1), np.nan)


def precision_of(c) -> np.ndarray:
    c = np.asarray(c, float)
    den = c[..., 0] + c[..., 1]
    return np.where(den > 0, c[..., 0] / np.maximum(den, 1), np.nan)


def point(c, fn) -> float:
    v = float(fn(np.asarray(c).sum(0)))
    return v


def boot_ci(c, fn, n_boot: int = N_BOOT_CI, seed: int = SEED) -> list[float]:
    if len(c) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    v = fn(c[idx].sum(1))
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan")] * 2
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_delta(c1, c0, fn, n_boot: int = N_BOOT_TEST, seed: int = SEED) -> dict:
    """fn(c1) - fn(c0) with cluster bootstrap. p_two = repo formula; p_one = P(d<=0) with +1 correction (H1: d>0)."""
    d0 = float(fn(c1.sum(0)) - fn(c0.sum(0))) if len(c1) else float("nan")
    nan = {
        "delta": d0,
        "ci": [float("nan")] * 2,
        "p_two": float("nan"),
        "p_one_gt": float("nan"),
        "n_boot": 0,
    }
    if len(c1) < 2:
        return nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c1), (n_boot, len(c1)))
    d = fn(c1[idx].sum(1)) - fn(c0[idx].sum(1))
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return nan
    b = len(d)
    le, ge = (np.sum(d <= 0) + 1) / (b + 1), (np.sum(d >= 0) + 1) / (b + 1)
    return {
        "delta": d0,
        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_two": float(min(1.0, 2 * min(le, ge))),
        "p_one_gt": float(le),
        "n_boot": int(b),
    }


def holm(pvals) -> list[float]:
    p = np.asarray(pvals, float)
    ok = np.flatnonzero(np.isfinite(p))
    out = np.full(len(p), np.nan)
    m = len(
        p
    )  # family size is fixed by the protocol: a test that could not be computed still counts
    order = ok[np.argsort(p[ok], kind="stable")]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return [float(x) for x in out]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


# --------------------------------------------------------------------------------------------- cells / parsing


def parse_textgrid_first_tier(text: str) -> list[tuple[float, float, str]]:
    """Intervals of the FIRST interval tier of a Praat TextGrid (long or short format)."""
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    out: list[tuple[float, float, str]] = []
    # long format: "xmin = ..", "xmax = ..", "text = ".."" inside "intervals [n]:" blocks of the first tier
    if any(ln.startswith("intervals [") for ln in lines):
        in_first, seen_tier = False, 0
        i = 0
        while i < len(lines):
            ln = lines[i]
            if re.match(r"item \[\d+\]", ln):
                seen_tier += 1
                in_first = seen_tier == 1
            elif in_first and ln.startswith("intervals ["):
                xmin = float(lines[i + 1].split("=")[1])
                xmax = float(lines[i + 2].split("=")[1])
                m = re.match(r'text\s*=\s*"(.*)"\s*$', lines[i + 3], re.S)
                txt = (
                    m.group(1)
                    if m
                    else lines[i + 3].split("=", 1)[1].strip().strip('"')
                )
                out.append((xmin, xmax, txt.replace('""', '"')))
                i += 3
            i += 1
        return out
    # short format: header lines, then per tier: "IntervalTier", name, xmin, xmax, n, then n x (xmin, xmax, text)
    k = next(i for i, ln in enumerate(lines) if ln == '"IntervalTier"')
    n = int(lines[k + 4])
    base = k + 5
    for j in range(n):
        a, b, t = lines[base + 3 * j : base + 3 * j + 3]
        out.append((float(a), float(b), t.strip().strip('"').replace('""', '"')))
    return out


def nocando_speaker(rec_id: str) -> str:
    """Speaker = file stem without the trailing ``_es_<n>``: ``berta_es_2`` -> ``berta``."""
    return re.sub(r"_es_\d+$", "", rec_id)


class Cell:
    """One evaluation cell: recordings (audio loader) and reference units."""

    def __init__(self, name: str, recordings: dict, units: list[dict]):
        self.name, self.recordings, self.units = (
            name,
            recordings,
            units,
        )  # recordings: id -> {"audio": callable}


def load_audio_wav_bytes(path: Path) -> bytes:
    """mp3/wav -> mono 16 kHz PCM_16 WAV bytes (same recipe as scripts/baselines/ameresco_prepare.py)."""
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def decode_textgrid(raw: bytes) -> str:
    """BOM -> UTF-16; else strict UTF-8; else Latin-1 (a UTF-16 try without BOM would accept any even-length bytes)."""
    import codecs

    if raw[:2] in (codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE):
        return raw.decode("utf-16")
    if raw[:3] == codecs.BOM_UTF8:
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def build_nocando_cell(root: Path) -> Cell:
    base = root / "02_NOCANDO_ES"
    units, recs = [], {}
    for tg in sorted((base / "alineamientos").glob("*.TextGrid")):
        rec = tg.stem
        mp3 = base / "audio" / f"{rec}.mp3"
        if not mp3.is_file():
            raise FileNotFoundError(f"no audio for {rec}")
        recs[rec] = {"audio": (lambda p=mp3: load_audio_wav_bytes(p))}
        raw = tg.read_bytes()
        txt = decode_textgrid(raw)
        k = 0
        for xmin, xmax, t in parse_textgrid_first_tier(txt):
            if not t.strip():
                continue
            ref = reference_of(t)
            units.append(
                {
                    "unit_id": f"{rec}:{k:05d}",
                    "recording": rec,
                    "speaker": nocando_speaker(rec),
                    "start": xmin,
                    "end": xmax,
                    **ref,
                }
            )
            k += 1
    return Cell("nocando", recs, units)


def build_ameresco_second_look(root: Path) -> Cell:
    """Out-of-pool AMERESCO segments (no automatic exclusion reason, duration outside 3-8 s). Second look, not confirmatory."""
    base = root / "12_AMERESCO"
    rows = list(
        csv.DictReader(
            (base / "source_annotation_index.csv").open(encoding="utf-8-sig")
        )
    )
    recs, units = {}, []
    for r in rows:
        d = int(r["end_ms"]) - int(r["start_ms"])
        if r["automatic_exclusion_reasons"] or 3000 <= d <= 8000:
            continue  # excluded rows and the contaminated 98-segment pool
        rid = r["recording_id"]
        if rid not in recs:
            mp3 = next((base / rid).glob("*.mp3"))
            recs[rid] = {"audio": (lambda p=mp3: load_audio_wav_bytes(p))}
        units.append(
            {
                "unit_id": f"{rid}:{r['source_annotation_id']}",
                "recording": rid,
                "speaker": f"{rid}:{r['source_tier']}",
                "start": int(r["start_ms"]) / 1000,
                "end": int(r["end_ms"]) / 1000,
                "prolong_ref": int(
                    r["source_alargamiento_present"] in ("1", "True", "true", "yes")
                ),
            }
        )
    return Cell("ameresco_second_look", recs, units)


# ----------------------------------------------------------------------------------------- running the systems


class CachingASR:
    """Decodes each distinct (method, audio, args) once; the second profile of the same file replays it, so both
    profiles are scored on IDENTICAL ASR output and the cost is paid once. Everything else is delegated."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "first_words", None)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _call(self, method, audio, *a, **k):
        key = (
            method,
            sha256_bytes(np.ascontiguousarray(audio).tobytes()),
            repr(a),
            repr(sorted(k.items())),
        )
        if key not in self._cache:
            self._cache[key] = getattr(self._inner, method)(audio, *a, **k)
            if self.first_words is None:
                object.__setattr__(self, "first_words", self._cache[key])
        return self._cache[key]

    def transcribe(self, audio, *a, **k):
        return self._call("transcribe", audio, *a, **k)

    def transcribe_chunked(self, audio, *a, **k):
        return self._call("transcribe_chunked", audio, *a, **k)


def words_to_tokens(words) -> list[tuple[str, float, float]]:
    out = []
    for w in words or []:
        for t in norm_tokens(getattr(w, "text", "")):
            out.append((t, float(w.start), float(w.end)))
    return out


def rule_events(words) -> dict[str, list[tuple[float, float]]]:
    """Frozen lexical comparator on the ASR words (same rules as the reference, applied to the system's own text)."""
    toks = words_to_tokens(words)
    fillers, rest = [], []
    for i, (t, s, e) in enumerate(toks):
        nxt = toks[i + 1][0] if i + 1 < len(toks) else None
        (fillers if is_filler(t, nxt) else rest).append((t, s, e))
    rep = []
    for pos, n in repeat_positions([t for t, _, _ in rest]):
        rep.append((rest[pos][1], rest[pos + n - 1][2]))
    return {"repetition": rep, "filler_word": [(s, e) for _, s, e in fillers]}


def doc_events(doc: dict) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for ev in doc.get("events") or []:
        if ev.get("in_scope", True) is False:
            continue
        out.setdefault(ev["class"], []).append(
            (float(ev["start_s"]), float(ev["end_s"]))
        )
    return out


def run_recording(
    analyzer, options_cls, audio_bytes: bytes, profiles: list[str]
) -> dict:
    """Run every profile on one recording sharing one ASR decode. Returns events per profile, words and errors."""
    cache = CachingASR(None)

    def wrapper(inner):
        object.__setattr__(cache, "_inner", inner)
        return cache

    res = {"events": {}, "errors": {}, "words": None, "doc_meta": {}}
    for prof in profiles:
        doc = analyzer.analyze_bytes(
            audio_bytes,
            options_cls(
                decision_profile=prof, include_text=False, output_scope="evaluated"
            ),
            asr_wrapper=wrapper,
        )
        if doc.get("error"):
            res["errors"][prof] = (
                doc["error"].get("code", "error")
                if isinstance(doc["error"], dict)
                else str(doc["error"])
            )
            res["events"][prof] = {}
        else:
            res["events"][prof] = doc_events(doc)
            res["doc_meta"][prof] = {
                "provenance": doc.get("provenance"),
                "schema_version": doc.get("schema_version"),
                "n_events": len(doc.get("events") or []),
            }
    res["words"] = cache.first_words
    return res


def assign_to_units(
    units: list[dict], events: list[tuple[float, float]], pad: float = 0.0
) -> tuple[set, int]:
    """Max-overlap assignment of events to the units of ONE recording. Returns (unit indexes hit, dropped events)."""
    hit, dropped = set(), 0
    for s, e in events:
        e = max(e, s + 0.02)  # word timestamps can be zero-length: keep them assignable
        best, best_ov = None, 0.0
        for i, u in units:
            ov = min(e, u["end"] + pad) - max(s, u["start"] - pad)
            if ov > best_ov:
                best, best_ov = i, ov
        if best is None:
            dropped += 1
        else:
            hit.add(best)
    return hit, dropped


def score_cell(
    cell: Cell, analyzer, options_cls, profiles: list[str], progress=print
) -> dict:
    """Run the systems on every recording of ``cell`` and return per-unit predictions per system and class."""
    by_rec: dict[str, list] = {}
    for i, u in enumerate(cell.units):
        by_rec.setdefault(u["recording"], []).append((i, u))
    systems = list(profiles) + ["rule"]
    pred = {
        (s, c): np.zeros(len(cell.units), int)
        for s in systems
        for c in ("repetition", "filler_word", "prolongation")
    }
    failures, dropped = {}, {s: 0 for s in systems}
    meta = {}
    for rid in sorted(cell.recordings):
        units = by_rec.get(rid, [])
        progress(f"[{cell.name}] {rid} ({len(units)} units)")
        r = run_recording(
            analyzer, options_cls, cell.recordings[rid]["audio"](), profiles
        )
        if r["errors"]:
            failures[rid] = r[
                "errors"
            ]  # a failed recording keeps all-zero predictions: counted as no detection
        meta[rid] = r["doc_meta"]
        per_sys = {p: r["events"][p] for p in profiles}
        per_sys["rule"] = rule_events(r["words"])
        for s, evs in per_sys.items():
            for c, lst in evs.items():
                if (s, c) not in pred:
                    continue
                hit, dr = assign_to_units(units, lst)
                dropped[s] += dr
                for i in hit:
                    pred[(s, c)][i] = 1
    return {
        "pred": pred,
        "failures": failures,
        "dropped_events": dropped,
        "doc_meta": meta,
    }


# ------------------------------------------------------------------------------------------------ analysis


def evaluate_nocando(cell: Cell, run: dict, default: str, v0: str) -> dict:
    u = cell.units
    rec = np.array([x["recording"] for x in u])
    spk = np.array([x["speaker"] for x in u])
    y_rep = np.array([x["rep_literal"] for x in u])
    y_fill = np.array([int(x["n_fillers"] > 0) for x in u])
    P = run["pred"]
    out: dict = {
        "n_units": len(u),
        "n_recordings": len(set(rec)),
        "n_speakers": len(set(spk)),
        "classes": {},
    }

    def gate(y):
        pos = y == 1
        g = {
            "positives": int(pos.sum()),
            "recordings_with_positives": int(len(set(rec[pos]))),
            "speakers_with_positives": int(len(set(spk[pos]))),
        }
        g["evaluable"] = bool(
            g["positives"] >= GATE_MIN_POS
            and g["recordings_with_positives"] >= GATE_MIN_RECORDINGS
            and g["speakers_with_positives"] >= GATE_MIN_SPEAKERS
        )
        return g

    tests: list[dict] = []
    # ---- repetition (F1 over all units)
    gr = gate(y_rep)
    rep = {"gate": gr, "metrics": {}, "tests": {}}
    sys_pred = {
        "default": P[(default, "repetition")],
        "v0": P[(v0, "repetition")],
        "rule": P[("rule", "repetition")],
        "always_present": np.ones(len(u), int),
    }
    if gr["evaluable"]:
        C = {k: group_counts(y_rep, v, rec) for k, v in sys_pred.items()}
        for k, c in C.items():
            tot = c.sum(0)
            rep["metrics"][k] = {
                "tp": int(tot[0]),
                "fp": int(tot[1]),
                "fn": int(tot[2]),
                "precision": point(c, precision_of),
                "recall": point(c, recall_of),
                "f1": point(c, f1_of),
                "f1_ci95": boot_ci(c, f1_of),
                "recall_ci95": boot_ci(c, recall_of),
                "precision_ci95": boot_ci(c, precision_of),
            }
        rep["tests"]["H1_default_vs_always_present"] = {
            **paired_delta(C["default"], C["always_present"], f1_of),
            "expected": "gt0",
            "one_sided": True,
        }
        rep["tests"]["H2_default_vs_rule"] = {
            **paired_delta(C["default"], C["rule"], f1_of),
            "expected": "two_sided",
            "one_sided": False,
        }
        rep["metrics"]["default_events_equal_v0_events"] = bool(
            np.array_equal(sys_pred["default"], sys_pred["v0"])
        )
        rep["metrics"]["leave_one_speaker_out_f1_default"] = {
            s: float(
                f1_of(
                    group_counts(
                        y_rep[spk != s], sys_pred["default"][spk != s], rec[spk != s]
                    ).sum(0)
                )
            )
            for s in sorted(set(spk))
        }
    out["classes"]["repetition"] = rep
    # ---- filler (recall only: absence of a transcribed filler is not a verified negative)
    gf = gate(y_fill)
    fil = {
        "gate": gf,
        "metrics": {},
        "tests": {},
        "note": "recall on transcript-marked non-lexical filled pauses; no precision, no F1",
    }
    fp = {
        "default": P[(default, "filler_word")],
        "v0": P[(v0, "filler_word")],
        "rule": P[("rule", "filler_word")],
    }
    if gf["evaluable"]:
        pos = y_fill == 1
        C = {k: group_counts(y_fill[pos], v[pos], rec[pos]) for k, v in fp.items()}
        for k, c in C.items():
            tot = c.sum(0)
            fil["metrics"][k] = {
                "tp": int(tot[0]),
                "fn": int(tot[2]),
                "recall": point(c, recall_of),
                "recall_ci95": boot_ci(c, recall_of),
            }
        fil["tests"]["H3_default_vs_v0"] = {
            **paired_delta(C["default"], C["v0"], recall_of),
            "expected": "two_sided",
            "one_sided": False,
        }
        fil["tests"]["H4_default_vs_rule"] = {
            **paired_delta(C["default"], C["rule"], recall_of),
            "expected": "two_sided",
            "one_sided": False,
        }
        dur = {r: max(x["end"] for x in u if x["recording"] == r) for r in set(rec)}
        minutes = sum(dur.values()) / 60
        fil["descriptive"] = {
            "audio_minutes_covered_by_units": minutes,
            "default_filler_units_per_minute": float(fp["default"].sum() / minutes)
            if minutes
            else None,
            "reference_units_with_filler_per_minute": float(y_fill.sum() / minutes)
            if minutes
            else None,
        }
    out["classes"]["filler_word"] = fil
    # ---- Holm family fixed at 4
    names = [
        "H1_default_vs_always_present",
        "H2_default_vs_rule",
        "H3_default_vs_v0",
        "H4_default_vs_rule",
    ]
    where = {
        "H1_default_vs_always_present": ("repetition", "H1_default_vs_always_present"),
        "H2_default_vs_rule": ("repetition", "H2_default_vs_rule"),
        "H3_default_vs_v0": ("filler_word", "H3_default_vs_v0"),
        "H4_default_vs_rule": ("filler_word", "H4_default_vs_rule"),
    }
    plist = []
    for n in [
        "H1_default_vs_always_present",
        "H2_default_vs_rule",
        "H3_default_vs_v0",
        "H4_default_vs_rule",
    ]:
        cls, key = where[n]
        t = out["classes"][cls]["tests"].get(key)
        plist.append(
            float("nan")
            if t is None
            else (t["p_one_gt"] if t["one_sided"] else t["p_two"])
        )
    adj = holm(plist)
    for (n_key, (cls, key)), p, a in zip(where.items(), plist, adj):
        t = out["classes"][cls]["tests"].get(key)
        if t is None:
            out["classes"][cls]["tests"][key] = {
                "status": "NOT EVALUABLE (gate failed)",
                "p_raw": None,
                "p_holm": None,
                "verdict": "not evaluable",
            }
            continue
        t["p_raw"], t["p_holm"] = p, a
        ci_ok = (
            (t["ci"][0] > 0)
            if t["expected"] == "gt0"
            else (t["ci"][0] > 0 or t["ci"][1] < 0)
        )
        sign_ok = t["delta"] > 0 if t["expected"] == "gt0" else True
        t["direction"] = (
            "default_better"
            if t["delta"] > 0
            else "default_worse"
            if t["delta"] < 0
            else "none"
        )
        t["verdict"] = (
            "supported" if (a < 0.05 and ci_ok and sign_ok) else "not demonstrated"
        )
    out["holm_family"] = {"size": 4, "members": names, "p_raw": plist, "p_holm": adj}
    # ---- absolute criteria (CI-based, not part of the Holm family)
    ab = {}
    m = rep["metrics"].get("default")
    ab["A1_repetition_f1_ci_lower_ge_floor"] = {
        "floor": FLOOR_F1_REPETITION,
        "lower": None if m is None else m["f1_ci95"][0],
        "met": None if m is None else bool(m["f1_ci95"][0] >= FLOOR_F1_REPETITION),
    }
    m = fil["metrics"].get("default")
    ab["A2_filler_recall_ci_lower_ge_floor"] = {
        "floor": FLOOR_RECALL_FILLER,
        "lower": None if m is None else m["recall_ci95"][0],
        "met": None if m is None else bool(m["recall_ci95"][0] >= FLOOR_RECALL_FILLER),
    }
    out["absolute_criteria"] = ab
    out["failures_recordings"] = run["failures"]
    n_rec = max(1, len(set(rec)))
    out["recording_failure_rate"] = len(run["failures"]) / n_rec
    out["interpretable"] = bool(out["recording_failure_rate"] <= MAX_FAILURE_RATE)

    out["dropped_events_no_unit"] = run["dropped_events"]
    return out


def evaluate_ameresco_second_look(cell: Cell, run: dict, default: str, v0: str) -> dict:
    u = cell.units
    y = np.array([x["prolong_ref"] for x in u])
    out = {
        "status": "SECOND LOOK - NOT CONFIRMATORY (same conversations and speakers as the contaminated 98-segment pool)",
        "n_units": len(u),
        "marked_positive": int(y.sum()),
        "recordings": sorted(set(x["recording"] for x in u)),
        "systems": {},
    }
    for k, prof in (("default", default), ("v0", v0)):
        p = run["pred"][(prof, "prolongation")]
        tp = int(((y == 1) & (p == 1)).sum())
        n = int((y == 1).sum())
        out["systems"][k] = {
            "recall_marked": tp / n if n else None,
            "recall_exact_ci95_segments_independent_assumption": clopper_pearson(tp, n),
            "tp": tp,
            "n_marked": n,
            "flag_rate_in_unmarked_segments": float(
                ((y == 0) & (p == 1)).sum() / max(1, (y == 0).sum())
            ),
            "flag_rate_note": "unmarked is NOT verified negative (AMERESCO negative coverage unknown): this is not a false-positive rate",
        }
    out["failures_recordings"] = run["failures"]
    return out


# --------------------------------------------------------------------------------------- freezing / verification


def data_fingerprint(root: Path) -> dict:
    files = sorted((root / "02_NOCANDO_ES" / "audio").glob("*.mp3")) + sorted(
        (root / "02_NOCANDO_ES" / "alineamientos").glob("*.TextGrid")
    )
    files += [root / "12_AMERESCO" / "source_annotation_index.csv"] + sorted(
        (root / "12_AMERESCO").glob("*/*.mp3")
    )
    return {
        f.relative_to(root).as_posix(): {
            "sha256": sha256_file(f),
            "bytes": f.stat().st_size,
        }
        for f in files
    }


def cmd_freeze_protocol(a) -> int:
    if PROTOCOL_FREEZE.exists():
        raise SystemExit(
            f"{PROTOCOL_FREEZE} exists: the protocol is already frozen (write a dated annex instead)"
        )
    root = corpora_root(a)
    PROTOCOL_FREEZE.write_text(
        json.dumps(
            {
                "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol_file": PROTOCOL_MD.name,
                "protocol_sha256": sha256_text_file(PROTOCOL_MD),
                "harness_file": SELF.relative_to(REPO).as_posix(),
                "harness_sha256": sha256_text_file(SELF),
                "note": "sha256 of text files computed after CRLF -> LF normalisation; data files hashed as raw bytes",
                "data": data_fingerprint(root),
            },
            indent=1,
        ),
        "utf8",
    )
    print("wrote", PROTOCOL_FREEZE)
    return 0


def default_profile_name() -> str:
    from mrcd.service.decision_profile import DEFAULT_DECISION_PROFILE

    return DEFAULT_DECISION_PROFILE


def cmd_freeze_system(a) -> int:
    if SYSTEM_FREEZE.exists():
        raise SystemExit(f"{SYSTEM_FREEZE} exists: the system is already frozen")
    check_protocol_hashes()
    head = git("rev-parse", "HEAD").strip()
    if not head.startswith(a.system_commit) or len(a.system_commit) < 12:
        raise SystemExit(
            f"--system-commit {a.system_commit} (>= 12 hex chars) must be a prefix of HEAD {head}"
        )
    if not system_state_clean():
        raise SystemExit("system paths have uncommitted or untracked changes")
    tree = system_tree()
    SYSTEM_FREEZE.write_text(
        json.dumps(
            {
                "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "system_commit": head,
                "system_paths": SYSTEM_PATHS,
                "system_tree_sha256": tree["sha256"],
                "system_tree_n_files": tree["n_files"],
                "protocol_sha256": sha256_text_file(PROTOCOL_MD),
                "harness_sha256": sha256_text_file(SELF),
                "default_decision_profile": default_profile_name(),
                "environment": env_versions(),
            },
            indent=1,
        ),
        "utf8",
    )
    print("wrote", SYSTEM_FREEZE, "- commit it BEFORE running")
    return 0


def check_protocol_hashes() -> dict:
    if not PROTOCOL_FREEZE.is_file():
        raise SystemExit("PROTOCOL.sha256.json missing: protocol not frozen")
    fz = json.loads(PROTOCOL_FREEZE.read_text("utf8"))
    if sha256_text_file(PROTOCOL_MD) != fz["protocol_sha256"]:
        raise SystemExit("PROTOCOL_CONFIRMATORIO.md does not match its frozen sha256")
    if sha256_text_file(SELF) != fz["harness_sha256"]:
        raise SystemExit("this harness does not match its frozen sha256")
    return fz


def corpora_root(a) -> Path:
    r = a.corpora_root or os.environ.get("MRCD_CORPORA_ROOT")
    if not r:
        raise SystemExit(
            "set MRCD_CORPORA_ROOT or pass --corpora-root (no default path is built in)"
        )
    return Path(r)


def verify_all(a, *, check_data: bool = True) -> dict:
    fz = check_protocol_hashes()
    if not SYSTEM_FREEZE.is_file():
        raise SystemExit(
            "SYSTEM_FREEZE.json missing: run freeze-system after the improvements are merged"
        )
    sf = json.loads(SYSTEM_FREEZE.read_text("utf8"))
    if (
        sf["protocol_sha256"] != fz["protocol_sha256"]
        or sf["harness_sha256"] != fz["harness_sha256"]
    ):
        raise SystemExit(
            "SYSTEM_FREEZE.json was made against a different protocol/harness"
        )
    head = git("rev-parse", "HEAD").strip()
    if len(a.system_commit) < 12 or not sf["system_commit"].startswith(a.system_commit):
        raise SystemExit(
            "--system-commit must be a (>= 12 hex chars) prefix of the frozen system commit"
        )
    if not is_ancestor(sf["system_commit"]):
        raise SystemExit(
            "the frozen system commit is not an ancestor of HEAD (history was rewritten)"
        )
    if not system_state_clean():
        raise SystemExit("system paths have uncommitted or untracked changes")
    if system_tree()["sha256"] != sf["system_tree_sha256"]:
        raise SystemExit("system tree hash differs from the frozen value")
    for p in (PROTOCOL_MD, PROTOCOL_FREEZE, SYSTEM_FREEZE, SELF):
        if not committed_clean(p):
            raise SystemExit(
                f"{p.name} must be committed and unmodified before the run"
            )
    if check_data:
        root = corpora_root(a)
        now = data_fingerprint(root)
        if now != fz["data"]:
            bad = sorted(
                k for k in set(now) | set(fz["data"]) if now.get(k) != fz["data"].get(k)
            )
            raise SystemExit(
                f"data files differ from the frozen fingerprint: {bad[:5]}"
            )
    return {"protocol_freeze": fz, "system_freeze": sf, "head": head}


def cmd_verify(a) -> int:
    verify_all(a)
    print(
        "OK: protocol, harness, system tree, data and committed state match the frozen values"
    )
    return 0


def acquire_lock(out: Path, payload: dict, continue_annex: str | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK_NAME
    if (out / RESULT_NAME).exists():
        raise SystemExit(
            "a confirmatory result already exists: the evaluation was run once and cannot be repeated"
        )
    if lock.exists():
        st = json.loads(lock.read_text("utf8")).get("status")
        if (
            st == "started"
            and continue_annex
            and Path(continue_annex).is_file()
            and Path(continue_annex).stat().st_size > 0
        ):
            return  # crash recovery: only cached per-recording outputs are reused, nothing in the code or the data changes
        raise SystemExit(
            f"{lock} exists (status {st}): the evaluation has been started; a rerun needs a dated annex"
        )
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf8") as f:
        json.dump({**payload, "status": "started"}, f, indent=1)


def run_all(
    analyzer, options_cls, cells: dict, default: str, out: Path, progress=print
) -> dict:
    profiles = [default] if default == "v0" else [default, "v0"]
    result = {}
    nc = score_cell(cells["nocando"], analyzer, options_cls, profiles, progress)
    result["confirmatory_nocando"] = evaluate_nocando(
        cells["nocando"], nc, default, "v0"
    )
    if "ameresco_second_look" in cells:
        am = score_cell(
            cells["ameresco_second_look"], analyzer, options_cls, profiles, progress
        )
        result["second_look_ameresco"] = evaluate_ameresco_second_look(
            cells["ameresco_second_look"], am, default, "v0"
        )
    # local-only per-unit file (ids and 0/1; no text)
    loc = out / "local"
    loc.mkdir(parents=True, exist_ok=True)
    with (loc / "nocando_unit_predictions.csv").open(
        "w", newline="", encoding="utf8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "unit_id",
                "recording",
                "speaker",
                "y_rep",
                "y_fill",
                "default_rep",
                "v0_rep",
                "rule_rep",
                "default_fill",
                "v0_fill",
                "rule_fill",
            ]
        )
        P = nc["pred"]
        for i, u in enumerate(cells["nocando"].units):
            w.writerow(
                [
                    u["unit_id"],
                    u["recording"],
                    u["speaker"],
                    u["rep_literal"],
                    int(u["n_fillers"] > 0),
                    P[(default, "repetition")][i],
                    P[("v0", "repetition")][i],
                    P[("rule", "repetition")][i],
                    P[(default, "filler_word")][i],
                    P[("v0", "filler_word")][i],
                    P[("rule", "filler_word")][i],
                ]
            )
    return result


def cmd_run(a) -> int:
    state = verify_all(a)
    root = corpora_root(a)
    out = OUT_DIR
    acquire_lock(
        out,
        {
            "system_commit": state["system_freeze"]["system_commit"],
        "git_head_at_run": state["head"],
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_sha256": state["protocol_freeze"]["protocol_sha256"],
            "argv": sys.argv[1:],
        },
        a.continue_after_crash,
    )
    from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer

    default = default_profile_name()
    analyzer = MrcdAnalyzer(max_duration_s=3600.0, device=a.device)
    cells = {
        "nocando": build_nocando_cell(root),
        "ameresco_second_look": build_ameresco_second_look(root),
    }
    res = run_all(analyzer, AnalysisOptions, cells, default, out)
    final = {
        "label": "CONFIRMATORY (cell nocando) + SECOND LOOK (cell ameresco_second_look); see protocol section 1 for what each can claim",
        "system_commit": state["system_freeze"]["system_commit"],
        "git_head_at_run": state["head"],
        "system_commit_param": a.system_commit,
        "system_tree_sha256": state["system_freeze"]["system_tree_sha256"],
        "default_decision_profile": default,
        "protocol_sha256": state["protocol_freeze"]["protocol_sha256"],
        "harness_sha256": state["protocol_freeze"]["harness_sha256"],
        "environment": env_versions(),
        "continued_after_crash_annex": a.continue_after_crash,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **res,
    }
    (out / RESULT_NAME).write_text(json.dumps(final, indent=1, default=float), "utf8")
    lock = json.loads((out / LOCK_NAME).read_text("utf8"))
    lock.update(
        {"status": "completed", "result_sha256": sha256_file(out / RESULT_NAME)}
    )
    (out / LOCK_NAME).write_text(json.dumps(lock, indent=1), "utf8")
    print("done:", out / RESULT_NAME)
    return 0


# ---------------------------------------------------------------------------------------------------- dry run


def synthetic_cell(scratch: Path) -> Cell:
    """Synthetic TextGrid + noise audio: exercises the parser, reference, assignment, gate and statistics."""
    rng = np.random.default_rng(0)
    phrases = [
        "la casa la casa es grande",
        "eh y luego fuimos",
        "no no quiero ir",
        "bueno entonces nada",
        "hm pues eso",
        "el el perro corre",
        "vamos al parque",
        "e indigenas y hijos",
        "ehm no se",
        "claro que si",
    ]
    units, recs = [], {}
    for r, spk in (
        ("anna_es_1", "anna"),
        ("anna_es_2", "anna"),
        ("beto_es_1", "beto"),
        ("cleo_es_1", "cleo"),
        ("dani_es_1", "dani"),
    ):
        t, body = 0.0, []
        for k in range(30):
            txt = phrases[int(rng.integers(0, len(phrases)))]
            body.append((t, t + 2.0, txt))
            t += 2.0
        tg = scratch / f"{r}.TextGrid"
        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            "xmin = 0",
            f"xmax = {t}",
            "tiers? <exists>",
            "size = 1",
            "item []:",
            "    item [1]:",
            '        class = "IntervalTier"',
            '        name = "transcript"',
            "        xmin = 0",
            f"        xmax = {t}",
            f"        intervals: size = {len(body)}",
        ]
        for j, (s, e, txt) in enumerate(body, 1):
            lines += [
                f"        intervals [{j}]:",
                f"            xmin = {s}",
                f"            xmax = {e}",
                f'            text = "{txt}"',
            ]
        tg.write_text("\n".join(lines) + "\n", "utf8")
        for j, (s, e, txt) in enumerate(
            parse_textgrid_first_tier(tg.read_text("utf8"))
        ):
            units.append(
                {
                    "unit_id": f"{r}:{j:05d}",
                    "recording": r,
                    "speaker": spk,
                    "start": s,
                    "end": e,
                    **reference_of(txt),
                }
            )
        wav = np.clip(rng.normal(0, 0.05, int(t * 16000)), -1, 1).astype(np.float32)
        buf = io.BytesIO()
        import soundfile as sf

        sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
        recs[r] = {"audio": (lambda b=buf.getvalue(): b)}
    return Cell("nocando", recs, units)


class _StubOptions:
    def __init__(
        self, decision_profile="o2026_11", include_text=False, output_scope="evaluated"
    ):
        self.decision_profile = decision_profile


class _StubWord:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


class _StubASR:
    def transcribe_chunked(self, audio, chunk_s=8.0):
        d = len(audio) / 16000
        return [
            _StubWord(w, i * 0.5, i * 0.5 + 0.4)
            for i, w in enumerate((["eh", "la", "la", "casa"] * int(d))[: int(d * 2)])
        ]


class _StubAnalyzer:
    """Stands in for MrcdAnalyzer in the dry run: deterministic events, goes through the real asr_wrapper path."""

    def analyze_bytes(self, data, options, asr_wrapper=None):
        import soundfile as sf

        audio, _ = sf.read(io.BytesIO(data), dtype="float32")
        asr = asr_wrapper(_StubASR()) if asr_wrapper else _StubASR()
        asr.transcribe_chunked(audio, chunk_s=8.0)
        d = len(audio) / 16000
        step = 3.0 if options.decision_profile == "v0" else 2.0
        evs = []
        for i, t in enumerate(np.arange(0.5, d - 1, step)):
            c = ("repetition", "filler_word", "prolongation")[i % 3]
            evs.append(
                {
                    "class": c,
                    "start_s": float(t),
                    "end_s": float(t) + 0.8,
                    "in_scope": True,
                }
            )
        return {
            "error": None,
            "events": evs,
            "provenance": None,
            "schema_version": "stub",
        }


def cmd_dry_run(a) -> int:
    scratch = Path(a.scratch).resolve()
    if (
        OUT_DIR.resolve() in (scratch, *scratch.parents)
        or scratch in OUT_DIR.resolve().parents
        or REPO in (scratch,)
    ):
        raise SystemExit(
            "--scratch must be outside the official output directory and the repository root"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    cell = synthetic_cell(scratch)
    res = run_all(_StubAnalyzer(), _StubOptions, {"nocando": cell}, "o2026_11", scratch)
    res["DRY_RUN"] = (
        "SYNTHETIC DATA AND STUB SYSTEM: no scientific content, never a result"
    )
    (scratch / "DRY_RUN_RESULT.json").write_text(
        json.dumps(res, indent=1, default=float), "utf8"
    )
    print("dry run written to", scratch)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("freeze-protocol", "freeze-system", "verify", "run", "dry-run"):
        p = sub.add_parser(n)
        p.add_argument(
            "--corpora-root",
            default=None,
            help="root of the corpus repository (or $MRCD_CORPORA_ROOT)",
        )
        if n in ("freeze-system", "verify", "run"):
            p.add_argument(
                "--system-commit",
                required=True,
                help="commit sha of the frozen system (>= 12 hex chars); recorded in the output",
            )
        if n == "run":
            p.add_argument("--device", default="cpu", choices=["cpu"])
            p.add_argument(
                "--continue-after-crash",
                default=None,
                help="dated annex describing an infrastructure crash (lock status 'started')",
            )
        if n == "dry-run":
            p.add_argument("--scratch", required=True)
    a = ap.parse_args(argv)
    return {
        "freeze-protocol": cmd_freeze_protocol,
        "freeze-system": cmd_freeze_system,
        "verify": cmd_verify,
        "run": cmd_run,
        "dry-run": cmd_dry_run,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
