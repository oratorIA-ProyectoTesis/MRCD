"""Shared code of the service-improvement evaluation (results/mejora_servicio_20261102).

DIHANA folds 1-4 only (``asr_recovery.common.load_pool`` asserts it); fold 5 is never loaded. CHM150 is read only by
the confirmation script, after the decision is frozen.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"  # before numpy (project rule)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import wave  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from mrcd.asr_recovery import common as RC  # noqa: E402
from mrcd.asr_recovery import stats as ST  # noqa: E402
from mrcd.context_eval.metrics import holm  # noqa: E402

OUT = REPO / "results" / "mejora_servicio_20261102"
SR = 16000
SEED = 13
N_BOOT = 5000
#: digital silence between concatenated turns (> the 0.6 s pause that closes a segment). 2.0 s was chosen on the
#: bundled sample clip (mrcd/service/web/sample_es.wav cut at its pauses), never on DIHANA: 1.0/1.5 s merged more
#: neighbouring utterances into one ASR segment.
GAP_S = 2.0
MAX_SESSION_S = 120.0  # a session holds turns of ONE speaker, in corpus order, up to this much audio
MIN_SESSION_S = (
    13.0  # current service minimum: shorter sessions are zero-padded at the end to it
)
CLASSES = ("filler_word", "prolongation", "repetition")
LABEL_COL = {"filler_word": "filler_present", "prolongation": "lengthening_present"}
WAV_DIR = RC.WAV


def sha_file(p) -> str:
    return RC.sha(p)


def turns_meta() -> dict:
    return {
        r["turn_id"]: r for r in csv.DictReader(RC.TURNS.open(encoding="utf-8-sig"))
    }


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16 kHz mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


def wav_bytes(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------- frame (M1)
def frame() -> dict:
    """The label-defined P1 frame (1019 turns of DIHANA folds 1-4): ids, rep label, weight, speaker, outer fold."""
    pool = RC.load_pool()
    outer = RC.outer_folds(pool)
    fr = json.loads((RC.OUT / "subset.json").read_text("utf8"))["frame"]
    by = {u["unit_id"]: i for i, u in enumerate(pool)}
    T = turns_meta()
    rows = []
    for s in fr:
        i = by[s["unit_id"]]
        v = pool[i]["ref"]["rep_literal"]
        if v is None:  # indeterminate turns are excluded, never negatives
            continue
        t = T[s["unit_id"]]
        rows.append(
            {
                "unit_id": s["unit_id"],
                "speaker": pool[i]["speaker"],
                "fold": int(outer[i]),
                "w": float(s["w"]),
                "y_repetition": int(v),
                "y_filler_word": int(t["filler_present"]),
                "y_prolongation": int(t["lengthening_present"]),
            }
        )
    return {"rows": rows}


def build_sessions(rows: list[dict], durations: dict) -> list[dict]:
    """Deterministic sessions: per speaker, turns sorted by id (dialogue then turn order), packed greedily up to
    MAX_SESSION_S of audio, separated by GAP_S of silence; a session shorter than MIN_SESSION_S is padded at the end."""
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r)
    sessions = []
    for spk in sorted(by_spk):
        cur, t = [], 0.0
        for r in sorted(by_spk[spk], key=lambda x: x["unit_id"]):
            d = durations[r["unit_id"]]
            if cur and t + GAP_S + d > MAX_SESSION_S:
                sessions.append(cur)
                cur, t = [], 0.0
            off = 0.0 if not cur else t + GAP_S
            cur.append({"unit_id": r["unit_id"], "offset_s": round(off, 6), "dur_s": d})
            t = off + d
        if cur:
            sessions.append(cur)
    out = []
    for k, turns in enumerate(sessions):
        spk = next(r["speaker"] for r in rows if r["unit_id"] == turns[0]["unit_id"])
        fold = next(r["fold"] for r in rows if r["unit_id"] == turns[0]["unit_id"])
        end = turns[-1]["offset_s"] + turns[-1]["dur_s"]
        out.append(
            {
                "session_id": f"s{k:04d}_{spk}",
                "speaker": spk,
                "fold": fold,
                "turns": turns,
                "speech_end_s": round(end, 6),
                "total_s": round(max(end, MIN_SESSION_S), 6),
            }
        )
    return out


def session_pcm(sess: dict, wav_dir: Path = WAV_DIR, read=read_pcm16) -> np.ndarray:
    n = int(round(sess["total_s"] * SR))
    pcm = np.zeros(n, np.int16)
    for t in sess["turns"]:
        x = read(wav_dir / f"{t['unit_id']}.wav")
        a = int(round(t["offset_s"] * SR))
        pcm[a : a + len(x)] = x[: max(0, n - a)]
    return pcm


# ---------------------------------------------------------------------------------------------------- event mapping
def assign_events(
    turns: list[dict], events: list[dict], mode: str = "max_overlap"
) -> dict:
    """{unit_id: set(classes)} from in-scope events of one document.

    ``max_overlap`` (primary): each event goes to the ONE turn it overlaps most (ties: earlier turn); an event that
    overlaps no turn is unassigned. ``any_overlap`` (sensitivity): an event marks every turn it overlaps."""
    out = {t["unit_id"]: set() for t in turns}
    spans = [(t["unit_id"], t["offset_s"], t["offset_s"] + t["dur_s"]) for t in turns]
    for e in events:
        if not e.get("in_scope", True):
            continue
        a, b = float(e["start_s"]), float(e["end_s"])
        ov = [(min(b, te) - max(a, ts), uid) for uid, ts, te in spans]
        if b <= a:  # zero-length interval: the turn that contains its point
            ov = [(1.0 if ts <= a <= te else 0.0, uid) for uid, ts, te in spans]
        hits = [(o, uid) for o, uid in ov if o > 0]
        if not hits:
            continue
        if mode == "max_overlap":
            best = max(o for o, _ in hits)
            uid = next(u for o, u in hits if o == best)
            out[uid].add(e["class"])
        else:
            for _, uid in hits:
                out[uid].add(e["class"])
    return out


# ---------------------------------------------------------------------------------------------------- statistics
def describe(y, p, g, w) -> dict:
    m = ST.wconfusion(y, p, w)
    m["f1_ci95"] = ST.boot_wf1_ci(y, p, g, w, 2000, SEED)
    m["n_groups"] = int(len(set(np.asarray(g).tolist())))
    return m


def paired_dist(y, p1, p0, g, w, n_boot: int = N_BOOT, seed: int = SEED):
    """Point weighted-F1 difference and its speaker-bootstrap distribution (same resampling as stats.paired_wf1_delta)."""
    y, p1, p0, g = (np.asarray(a) for a in (y, p1, p0, g))
    w = np.asarray(w, float)
    C1, C0 = ST._gcounts(y, p1, g, w), ST._gcounts(y, p0, g, w)
    d0 = float(ST._f1(C1.sum(0)) - ST._f1(C0.sum(0)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(C1), (n_boot, len(C1)))
    d = ST._f1(C1[idx].sum(1)) - ST._f1(C0[idx].sum(1))
    return d0, d[np.isfinite(d)]


def p_two_sided(d: np.ndarray) -> float:
    B = len(d)
    return float(
        min(
            1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1))
        )
    )


def ci(d: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(np.percentile(d, 100 * a)), float(np.percentile(d, 100 * (1 - a)))]


def holm_ci_family(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """Holm over the rows (each has ``delta``, ``dist``); adds ``p_boot``, ``p_holm``, the CI at the Holm step level
    (1 - alpha/(m - rank)) and ``holm_lower_gt_0``. Step-down: once a rank fails, later ranks fail."""
    m = len(rows)
    for r in rows:
        r["p_boot"] = p_two_sided(r["dist"])
    adj = holm([r["p_boot"] for r in rows])
    order = sorted(range(m), key=lambda i: rows[i]["p_boot"])
    still = True
    for rank, i in enumerate(order):
        r = rows[i]
        lev = 1 - alpha / (m - rank)
        r["p_holm"] = adj[i]
        r["holm_level"] = lev
        r["ci_holm"] = ci(r["dist"], lev)
        r["ci95"] = ci(r["dist"], 0.95)
        ok = still and r["p_holm"] < alpha and r["ci_holm"][0] > 0 and r["delta"] > 0
        r["holm_lower_gt_0"] = bool(ok)
        still = still and ok
    return rows


def wrate(flag, g, w, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Weighted rate (sum w*flag / sum w) with speaker-bootstrap CI."""
    flag, w = np.asarray(flag, float), np.asarray(w, float)
    _, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    k = int(inv.max()) + 1
    num, den = np.zeros(k), np.zeros(k)
    np.add.at(num, inv, w * flag)
    np.add.at(den, inv, w)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (n_boot, k))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return {
        "rate": float(num.sum() / den.sum()),
        "ci95": ci(r),
        "_num": num,
        "_den": den,
    }


def paired_rate_diff(f1, f0, g, w, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    a, b = wrate(f1, g, w), wrate(f0, g, w)
    k = len(a["_num"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (n_boot, k))
    d = a["_num"][idx].sum(1) / a["_den"][idx].sum(1) - b["_num"][idx].sum(1) / b[
        "_den"
    ][idx].sum(1)
    return {
        "delta": a["rate"] - b["rate"],
        "ci95": ci(d),
        "rate_1": a["rate"],
        "rate_0": b["rate"],
    }


def doc_digest(doc: dict) -> str:
    d = {k: v for k, v in doc.items() if k != "generated_at"}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def compact_doc(doc: dict) -> dict:
    """What is kept per document (no text): status, error code, events, warning codes."""
    return {
        "status": doc.get("status"),
        "error": (doc.get("error") or {}).get("code"),
        "error_details": (doc.get("error") or {}).get("details"),
        "events": [
            {
                "class": e["class"],
                "start_s": e["start_s"],
                "end_s": e["end_s"],
                "kind": e["score"]["kind"],
                "score": e["score"]["value"],
                "in_scope": e.get("in_scope", True),
            }
            for e in doc.get("events", [])
        ],
        "warnings": [w["code"] for w in doc.get("warnings", [])],
        "digest": doc_digest(doc),
    }
