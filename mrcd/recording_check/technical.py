"""Aggregate per-recording technical reports into the pre-registered technical summary.

Inputs
------
* ``reports``: dicts produced by ``verify_recording.verify_file`` (numbers only).
* ``runs`` (optional): one summary per recording from running the engine outside this module::

      {"recording_id": "VB_S01_PRES1", "status": "ok" | "failed" | "not_run",
       "wall_s": 41.2, "audio_s": 183.0, "warnings": ["..."], "events_total": 17}

  ``events_total`` is only a sanity number (zero or absurd values flag a broken run). It is NOT accuracy: there are
  no human labels in this track and none may be created from system output.

Output states that no accuracy claim is possible. Missing measurements are reported as unavailable, never as 0.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from mrcd.recording_check import stats
from mrcd.recording_check.config import CONFIG_VERSION, TECHNICAL as T, config_hash

RECORDING_RE = re.compile(r"^VB_S(?P<spk>\d{2})_(?P<task>PRES|ESP)(?P<n>\d)$")
NO_ACCURACY_STATEMENT = (
    "No human labels exist for these recordings. No precision, recall, F1 or any accuracy figure can be computed or "
    "claimed from this report; system outputs are never a reference for themselves."
)


def parse_recording_id(rec: str) -> dict:
    m = RECORDING_RE.match(rec)
    if not m:
        raise ValueError(f"recording id {rec!r} does not match VB_S<nn>_<PRES|ESP><k>")
    return {
        "speaker": f"VB_S{m.group('spk')}",
        "task": m.group("task"),
        "n": int(m.group("n")),
    }


def _ci(
    counts: dict[str, np.ndarray], stat=lambda v: None if v[1] == 0 else v[0] / v[1]
) -> dict:
    return stats.speaker_bootstrap(
        counts, stat, B=T["bootstrap_B"], seed=T["seed"], level=T["ci_level"]
    )


def _add(d: dict, key: str, vec) -> None:
    d[key] = d.get(key, 0) + np.asarray(vec, float)


def _decision(ci: dict, threshold: float) -> str:
    if ci["point"] is None:
        return "no disponible"
    if ci["lo"] is not None and ci["lo"] >= threshold:
        return "cumple (limite inferior del IC >= umbral)"
    if ci["point"] >= threshold:
        return "cumple solo en la estimacion puntual (el IC incluye valores bajo el umbral)"
    return "no cumple"


def summarize(reports: Sequence[dict], runs: Sequence[dict] = ()) -> dict:
    by_rec = {}
    for r in reports:
        rec = r["file"].rsplit(".", 1)[0]
        info = parse_recording_id(rec)
        by_rec[rec] = (info, r)
    speakers = {i["speaker"] for i, _ in by_rec.values()}
    total_s = sum((r.get("duration_s") or 0.0) for _, r in by_rec.values())
    out: dict[str, Any] = {
        "config_version": CONFIG_VERSION,
        "config_hash": config_hash(),
        "accuracy_claim_allowed": False,
        "statement": NO_ACCURACY_STATEMENT,
        "design": {
            "n_recordings": len(by_rec),
            "n_speakers": len(speakers),
            "total_minutes": total_s / 60,
            "meets_min_speakers": len(speakers) >= T["min_speakers"],
            "meets_min_minutes": total_s / 60 >= T["min_total_speech_minutes"],
        },
    }

    # T1 / T2 / T3: video
    adm: dict[str, np.ndarray] = {}
    sync: dict[str, np.ndarray] = {}
    face: list[float] = []
    unavailable_gate = []
    offsets_gate, offsets_clap = [], []
    reasons: dict[str, int] = {}
    for rec, (info, r) in by_rec.items():
        g = r.get("face_gate")
        if not g or g.get("admitted_fraction") is None:
            unavailable_gate.append(rec)
            continue
        _add(adm, info["speaker"], [g["admitted"], g["segments"]])
        _add(
            sync,
            info["speaker"],
            [1.0 if g.get("video_sync_reason") == "ok" else 0.0, 1.0],
        )
        if g.get("face_present_fraction") is not None:
            face.append(g["face_present_fraction"])
        if g.get("video_offset_ms") is not None:
            offsets_gate.append(g["video_offset_ms"])
        for k, v in g.get("primary_reasons", {}).items():
            reasons[k] = reasons.get(k, 0) + v
        c = (r.get("clap_sync") or {}).get("start_offset_ms")
        if c is not None:
            offsets_clap.append(c)
    t1 = _ci(adm)
    loo = []
    for k in adm:
        rest = sum(v for kk, v in adm.items() if kk != k) if len(adm) > 1 else None
        if rest is not None and rest[1] > 0:
            loo.append(rest[0] / rest[1])
    t2 = _ci(sync)
    out["T1_gate_admission"] = {
        "admitted_fraction": t1,
        "threshold": T["T1_min_admitted_fraction"],
        "decision": _decision(t1, T["T1_min_admitted_fraction"]),
        "primary_exclusion_reasons_segments": reasons,
        "recordings_without_gate_result": unavailable_gate,
        "leave_one_speaker_out": {"min": min(loo), "max": max(loo)} if loo else None,
        "note": "frozen S6 gate (mrcd/av/config.py); counts of 3 s segments; speaker-bootstrap CI",
    }
    out["T2_av_sync"] = {
        "videos_with_sync_ok": t2,
        "threshold": T["T2_min_sync_ok_fraction"],
        "decision": _decision(t2, T["T2_min_sync_ok_fraction"]),
        "gate_offset_ms": stats.quantiles(offsets_gate, (0.5, 0.95)),
        "clap_offset_ms_start": stats.quantiles(offsets_clap, (0.5, 0.95)),
        "positive_means": "video later than audio",
    }
    med_face = float(np.median(face)) if face else None
    out["T3_face_visibility"] = {
        "median_face_present_fraction": med_face,
        "threshold": T["T3_min_median_face_present_fraction"],
        "decision": "no disponible"
        if med_face is None
        else (
            "cumple"
            if med_face >= T["T3_min_median_face_present_fraction"]
            else "no cumple"
        ),
        "note": "geometric detection of a frontal face only; no emotion, intent or psychological state is inferred",
    }

    # T4: microphone / audio chain
    audio_keys = (
        "audio_peak_level",
        "audio_clipping",
        "audio_noise_floor",
        "audio_sample_rate",
    )
    a_ok: dict[str, np.ndarray] = {}
    per_check: dict[str, dict] = {}
    for rec, (info, r) in by_rec.items():
        v = r.get("verdicts", {})
        known = [v.get(k) for k in audio_keys if k in v]
        if not known or any(x is None for x in known):
            per_check.setdefault("unknown_recordings", {"n": 0})["n"] += 1
            continue
        _add(a_ok, info["speaker"], [1.0 if all(known) else 0.0, 1.0])
    t4 = _ci(a_ok)
    for k in audio_keys:
        vals = [
            r["verdicts"].get(k)
            for _, r in by_rec.values()
            if k in r.get("verdicts", {})
        ]
        per_check[k] = {
            "pass": sum(x is True for x in vals),
            "fail": sum(x is False for x in vals),
            "unknown": sum(x is None for x in vals),
        }
    out["T4_audio_chain"] = {
        "recordings_passing_all_audio_checks": t4,
        "threshold": T["T4_min_audio_ok_fraction"],
        "decision": _decision(t4, T["T4_min_audio_ok_fraction"]),
        "per_check": per_check,
        "levels": {
            "peak_dbfs": stats.quantiles(
                [(r.get("levels") or {}).get("peak_dbfs") for _, r in by_rec.values()]
            ),
            "noise_floor_dbfs": stats.quantiles(
                [
                    (r.get("levels") or {}).get("noise_floor_dbfs")
                    for _, r in by_rec.values()
                ]
            ),
        },
    }

    # T5 / latency: engine robustness on unseen speakers
    if runs:
        ok: dict[str, np.ndarray] = {}
        rtf, failures, warns = [], [], {}
        events_per_min = []
        for s in runs:
            info = parse_recording_id(s["recording_id"])
            good = s.get("status") == "ok"
            _add(ok, info["speaker"], [1.0 if good else 0.0, 1.0])
            if not good:
                failures.append(
                    {"recording_id": s["recording_id"], "status": s.get("status")}
                )
            if good and s.get("wall_s") and s.get("audio_s"):
                rtf.append(s["wall_s"] / s["audio_s"])
            if good and s.get("events_total") is not None and s.get("audio_s"):
                events_per_min.append(s["events_total"] / (s["audio_s"] / 60))
            for w in s.get("warnings", []):
                warns[w] = warns.get(w, 0) + 1
        t5 = _ci(ok)
        out["T5_robustness"] = {
            "analysis_success": t5,
            "threshold": T["T5_min_analysis_success_fraction"],
            "decision": _decision(t5, T["T5_min_analysis_success_fraction"]),
            "failures": failures,
            "warnings": warns,
            "real_time_factor_wall_over_audio": stats.quantiles(rtf, (0.5, 0.95)),
            "events_per_minute_sanity": {
                **stats.quantiles(events_per_min, (0.05, 0.5, 0.95)),
                "note": "sanity range of system output, NOT accuracy",
            },
        }
    else:
        out["T5_robustness"] = {
            "available": False,
            "reason": "no engine run summaries supplied",
        }
    return out


def render_markdown(res: dict) -> str:
    def f(x, nd=3):
        return "n/d" if x is None else f"{x:.{nd}f}"

    def ci(b):
        return f"{f(b['point'])} [{f(b['lo'])}, {f(b['hi'])}] ({b['n_speakers']} hablantes)"

    d = res["design"]
    L = ["# Informe tecnico de grabaciones propias", "", f"> {res['statement']}", ""]
    L.append(
        f"Diseno: {d['n_recordings']} grabaciones, {d['n_speakers']} hablantes, {d['total_minutes']:.1f} min (hablantes >= minimo: {d['meets_min_speakers']}; minutos >= minimo: {d['meets_min_minutes']})."
    )
    L.append("")
    L.append("| Criterio | Resultado | Umbral | Decision |")
    L.append("|---|---|---|---|")
    t = res["T1_gate_admission"]
    L.append(
        f"| T1 admision de la compuerta S6 | {ci(t['admitted_fraction'])} | {t['threshold']} | {t['decision']} |"
    )
    t = res["T2_av_sync"]
    L.append(
        f"| T2 sincronia AV verificable | {ci(t['videos_with_sync_ok'])} | {t['threshold']} | {t['decision']} |"
    )
    t = res["T3_face_visibility"]
    L.append(
        f"| T3 rostro visible (mediana) | {f(t['median_face_present_fraction'])} | {t['threshold']} | {t['decision']} |"
    )
    t = res["T4_audio_chain"]
    L.append(
        f"| T4 cadena de audio | {ci(t['recordings_passing_all_audio_checks'])} | {t['threshold']} | {t['decision']} |"
    )
    t = res["T5_robustness"]
    if t.get("available") is False:
        L.append(f"| T5 robustez | n/d: {t['reason']} | | |")
    else:
        L.append(
            f"| T5 analisis completado | {ci(t['analysis_success'])} | {t['threshold']} | {t['decision']} |"
        )
    return "\n".join(L) + "\n"
