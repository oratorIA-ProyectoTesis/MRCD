"""Per-stage computational cost of the full local pipeline, file in -> service JSON out (Task 3.4). SERIAL, CPU-only.

Stages (same calls as ``DisfluencyEngine.analyze`` / ``_evaluate``, instrumented; equality with ``engine.analyze`` is asserted):
  load_prep   read bytes + decode_wav (service audio_io: header checks, sha256)
  asr_p0 / asr_p1   VerbatimASR.transcribe_chunked(8 s / 2 s), profile p0 / p1_disfluent_prompt
  acoustic    extract_acoustic (Silero VAD + pYIN + energy) + speaker_f0_center
  linguistic  word_features + linguistic_windows
  head_v0     acoustic/prosody window tensors + predict() (checkpoint forward, batch 1 as in the engine)
  head_vc     EXPERIMENTAL Vc: predict_pooled + window_scores + unit_decision (extra on top of V0)
  decision    windows -> events (``_merge``)
  report      build_analysis + validate_document + json.dumps (service document)
Rounds: round 0 = cold (first pass after the models were loaded; reported separately), rounds 1..R measured, median/min/max.
usage: OMP_NUM_THREADS=4 python cost_pipeline.py --out DIR [--rounds 3]
"""

import sys
import pathlib as _p

sys.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
import _pin_omp  # noqa: F401,E402
import argparse
import json
import os
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import _sys  # noqa: E402

DATA = REPO / "data/annotation_batch_01"
# (name, source wav, start_s, duration_s) -> real recordings of the repo; crops are plain slices, no processing
FILES = [
    ("c05s_ak4Ox02", "ak4Oxthf_fc_s02.wav", 0, 5),
    ("c09s_ak4Ox03", "ak4Oxthf_fc_s03.wav", 0, 9),
    ("c13s_ak4Ox04", "ak4Oxthf_fc_s04.wav", 0, 13.5),
    ("c17s_ak4Ox04", "ak4Oxthf_fc_s04.wav", 14, 17),
    ("w22s_MTRsIn", "MTRsInkqv5E_s04.wav", 0, None),
    ("w26s_p2ZWJP", "p2ZWJP75kwA_s01.wav", 0, None),
    ("w27s_ak4Ox03", "ak4Oxthf_fc_s03.wav", 0, None),
    ("w34s_ak4Ox04", "ak4Oxthf_fc_s04.wav", 0, None),
    ("w36s_DNrrpE", "DNrrpE-sYnU_s00.wav", 0, None),
    ("w43s_XAJ2Dc", "XAJ2DCyHjvg_s04.wav", 0, None),
    ("w47s_ak4Ox00", "ak4Oxthf_fc_s00.wav", 0, None),
    ("w51s_pCF39B", "pCF39B2D0sM_s01.wav", 0, None),
    ("c60s_RkP96R", "RkP96RIRA4o_s05.wav", 0, 60),
]


class T:
    def __init__(self):
        self.d, self.cpu = {}, {}

    def stage(self, name):
        outer = self

        class C:
            def __enter__(s):
                s.t, s.c = time.perf_counter(), time.process_time()
                return s

            def __exit__(s, *a):
                outer.d[name] = outer.d.get(name, 0.0) + time.perf_counter() - s.t
                outer.cpu[name] = outer.cpu.get(name, 0.0) + time.process_time() - s.c

        return C()


def pipeline(engine, vc_head, wav_bytes, svc_opts):
    from mrcd.constants import SAMPLE_RATE, STRIDE_S, WINDOW_S
    from mrcd.asr_profiles import get_profile
    from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
    from mrcd.engine import AnalysisResult, Window
    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.extractors.linguistic import word_features
    from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
    from mrcd.inference import predict
    from mrcd.live.window_builder import window_text
    from mrcd.service.analyzer import build_analysis
    from mrcd.service.audio_io import decode_wav
    from mrcd.service.contract import validate_document
    from mrcd.adaptation import predict_pooled

    tm = T()
    with tm.stage("load_prep"):
        dec = decode_wav(wav_bytes)
        audio = dec.samples
    out = {"duration_s": dec.duration_s}
    if dec.duration_s < engine.MIN_AUDIO_S:
        out["error"] = "audio_too_short"
        out["stages_s"] = dict(tm.d)
        return out
    res = {}
    for prof, name in (
        ("p1_disfluent_prompt", "asr_p1"),
        ("p0", "asr_p0"),
    ):  # p0 last: its words feed the rest
        engine._asr.profile = get_profile(prof)
        with tm.stage(name):
            res[prof] = engine._asr.transcribe_chunked(audio, chunk_s=8.0)
    words = res["p0"]
    with tm.stage("acoustic"):
        frames = extract_acoustic(audio, use_silero=True).frame_matrix()
        center = speaker_f0_center(frames)
    duration = len(audio) / SAMPLE_RATE
    starts = np.arange(0.0, max(duration - WINDOW_S, 0.0) + 1e-9, STRIDE_S)
    with tm.stage("linguistic"):
        ling, lpos, lmask = linguistic_windows(
            [w.to_dict() for w in words],
            word_features(words),
            starts,
            context=engine._context,
        )
    with tm.stage("head_v0"):
        tensors = {
            "ac": acoustic_windows(center_f0(frames, center), starts),
            "ling": ling[:, :, : engine._ling_dim],
            "lpos": lpos,
            "lmask": lmask,
        }
        if engine._include_pros:
            tensors["pros"] = prosody_windows(frames, starts, center)
        probs = predict(engine._net, engine._norm, tensors, engine._device, bs=1)
    with tm.stage("head_vc"):
        p2, pooled = predict_pooled(
            engine._net, engine._norm, tensors, engine._device, bs=1
        )
        scores = vc_head.window_scores(pooled)
        vc_head.unit_decision(scores)
    with tm.stage("decision"):
        wins = []
        for k, (t0, p) in enumerate(zip(starts, probs)):
            i = int(p.argmax())
            wins.append(
                Window(
                    start=float(t0),
                    end=float(t0) + WINDOW_S,
                    label=engine._labels[i],
                    confidence=float(p[i]),
                    probabilities={l: float(v) for l, v in zip(engine._labels, p)},
                    text=window_text(words, float(t0)),
                    terminal=False,
                    head_scores=None,
                )
            )
        events = engine._merge(wins)
    with tm.stage("report"):
        meta = engine._metadata(False, None, 8.0)
        meta["asr_profile"] = engine._asr_profile_meta(
            audio, " ".join(w.text for w in words), 8.0
        )
        result = AnalysisResult(
            events=events,
            windows=[],
            transcript=" ".join(w.text for w in words),
            duration=duration,
            mode=engine._net.mode,
            warning=engine.warning,
            metadata=meta,
            head=None,
        )
        doc = build_analysis(
            result,
            options=svc_opts,
            audio=dec,
            engine_info={
                "checkpoint": {
                    "file": "ckpt_audio_text.pt",
                    "sha256": "n/a",
                    "mode": engine._net.mode,
                    "labels": list(engine.labels),
                    "checkpoint_notice": engine.warning,
                }
            },
        )
        errs = validate_document(doc)
        n_json = len(json.dumps(doc))
    out.update(
        stages_s=dict(tm.d),
        stages_cpu_s=dict(tm.cpu),
        n_words=len(words),
        n_events=len(events),
        n_windows=len(starts),
        n_words_p1=len(res["p1_disfluent_prompt"]),
        schema_errors=len(errs),
        json_bytes=n_json,
        events=[(e.label, e.start, e.end) for e in events],
    )
    engine._asr.profile = get_profile("p0")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inp = out / "inputs"
    inp.mkdir(exist_ok=True)
    meta = {
        "hardware": _sys.hardware(),
        "code_sha256_mrcd": _sys.code_sha(REPO),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ckpt_sha256": _sys.sha256_file(REPO / "models/ckpt_audio_text.pt"),
        "rounds": a.rounds,
        "files": [],
    }
    paths = []
    for name, src, st, dur in FILES:
        x, sr = sf.read(DATA / src, dtype="float32")
        assert sr == 16000 and x.ndim == 1
        x = x[int(st * sr) : int((st + dur) * sr)] if dur else x
        p = inp / f"{name}.wav"
        sf.write(p, x, sr, subtype="PCM_16")
        paths.append((name, p))
        meta["files"].append(
            {
                "name": name,
                "source": src,
                "start_s": st,
                "duration_s": len(x) / sr,
                "sha256": _sys.sha256_file(p),
            }
        )
    from mrcd import DisfluencyEngine
    from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer

    meta["background_start"] = _sys.background_report(3)
    t0, c0 = time.perf_counter(), time.process_time()
    eng = DisfluencyEngine.load(
        REPO / "models/ckpt_audio_text.pt",
        whisper_size="small",
        device="cpu",
        head="vc",
    )
    meta["cold_start_load_s"] = {
        "wall": time.perf_counter() - t0,
        "cpu": time.process_time() - c0,
        "includes": "checkpoint + Whisper small + Vc head, local files",
    }
    meta["mem_after_load"] = _sys.mem_mb()
    vc = eng._head
    opts = AnalysisOptions()
    runs = []
    for rnd in range(a.rounds + 1):  # round 0 = cold
        bgr = _sys.background_report(3)
        for name, p in paths:
            data = p.read_bytes()
            t1 = time.perf_counter()
            r = pipeline(eng, vc, data, opts)
            r.update(
                file=name,
                round=rnd,
                total_s=time.perf_counter() - t1,
                mem=_sys.mem_mb(),
                background_before_round=bgr["system_cpu_pct"],
            )
            runs.append(r)
            print(rnd, name, round(r["total_s"], 1), r.get("error", ""), flush=True)
    eq = {}
    for name, p in paths:
        if sf.info(p).duration >= eng.MIN_AUDIO_S and any(
            r["file"] == name and r["round"] == min(1, a.rounds) for r in runs
        ):
            x, _ = sf.read(p, dtype="float32")
            ref = eng.analyze(x)
            st = next(
                r for r in runs if r["file"] == name and r["round"] == min(1, a.rounds)
            )
            eq[name] = [(e.label, e.start, e.end) for e in ref.events] == [
                tuple(e) for e in st["events"]
            ]
    meta["staged_events_equal_engine_analyze"] = eq
    del eng
    svc = MrcdAnalyzer(REPO / "models/ckpt_audio_text.pt", device="cpu")
    e2e = []
    for rnd in range(a.rounds + 1):
        for name, p in paths:
            data = p.read_bytes()
            t1 = time.perf_counter()
            doc = svc.analyze_bytes(data)
            e2e.append(
                {
                    "file": name,
                    "round": rnd,
                    "s": time.perf_counter() - t1,
                    "status": doc.get("status"),
                    "error": (doc.get("error") or {}).get("code"),
                    "cold_engine_load_included": rnd == 0 and name == paths[0][0],
                }
            )
    meta["mem_end"] = _sys.mem_mb()
    (out / "cost_runs.json").write_text(
        json.dumps(
            {"meta": meta, "runs": runs, "service_e2e": e2e}, indent=1, default=str
        ),
        "utf8",
    )
    print("DONE", meta["mem_end"])


if __name__ == "__main__":
    main()
