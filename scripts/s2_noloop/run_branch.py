"""Decode the S2 union of units (U_W + U_S) with one branch on one device. Resumable, atomic writes, one JSON per unit.

usage: python run_branch.py --branch C0|P1|A|B|C|D_A|D_B|D_C --device cuda|cpu [--limit N] [--out-suffix X] [--ids FILE]
On cuda the CUDA 12 DLL dirs of the separate GPU venv (MRCD_GPU_NVIDIA_DIR) must be on PATH (see run_gpu.sh). OMP_NUM_THREADS=4 is pinned before numpy.
"""

import os
import sys

NUMPY_FIRST = "numpy" in sys.modules
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if NUMPY_FIRST:
    raise SystemExit("numpy imported before the OMP pin")
from pathlib import Path as _P  # noqa: E402

_REPO = _P(os.environ.get("MRCD_REPO") or _P(__file__).resolve().parents[2])
sys.path.insert(0, str(_REPO))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import soundfile as sf  # noqa: E402

from mrcd.asr_recovery.noloop import (
    B_LADDER,
    P2_FALLBACK,
    SEED,
    transcribe_chunked_branch,
)  # noqa: E402

REPO = _REPO
S2 = REPO / "results/mejora_objetivos_20261101/S2"
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
)
WAV_DIHANA = REPO / "results/baselines_20261030/dihana/wav16k"


def union_units() -> list[dict]:
    uw = json.loads(
        (REPO / "results/cierre_oe1_oe3_20261031/3b/asr/units.json").read_text("utf8")
    )
    frame = json.loads(
        (REPO / "results/asr_recovery_20261031/subset.json").read_text("utf8")
    )["frame"]
    seen, out = set(), []
    for u in uw:
        seen.add(u["unit_id"])
        out.append({"unit_id": u["unit_id"], "wav": u["wav"], "sets": ["W"]})
    by = {u["unit_id"]: u for u in out}
    for s in frame:
        if s["unit_id"] in by:
            by[s["unit_id"]]["sets"].append("S")
        else:
            out.append(
                {
                    "unit_id": s["unit_id"],
                    "wav": str(WAV_DIHANA / f"{s['unit_id']}.wav"),
                    "sets": ["S"],
                }
            )
    return out


def branch_spec(name: str):
    """-> (profile_name, kind, kwargs_override, chunk_s, overlap_s, ladder). kind 'engine' = VerbatimASR.transcribe_chunked itself."""
    if name == "C0":
        return "p0", "engine", {}, 8.0, 2.0, None
    if name == "P1":
        return "p1_disfluent_prompt", "engine", {}, 8.0, 2.0, None
    if name == "A":
        return "p0", "custom", dict(P2_FALLBACK), 8.0, 2.0, None
    if name == "B":
        return "p0", "custom", {}, 8.0, 2.0, B_LADDER
    if name == "C":
        return "p0", "custom", {}, 13.0, 2.0, None
    if (
        name == "C0kw"
    ):  # equivalence check of the custom decoder against the engine path
        return "p0", "custom", {}, 8.0, 2.0, None
    if name.startswith("D_"):  # P1 prompt + mitigation X
        x = name[2:]
        _, _, ov, ch, ov_s, lad = branch_spec(x)
        return "p1_disfluent_prompt", "custom", ov, ch, ov_s, lad
    raise SystemExit(f"unknown branch {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    ap.add_argument("--device", required=True, choices=("cuda", "cpu"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--ids", default=None, help="json list of unit ids to decode")
    a = ap.parse_args()
    import ctranslate2  # noqa: E402

    from mrcd.extractors.linguistic import VerbatimASR  # noqa: E402

    prof_name, kind, override, chunk_s, overlap_s, ladder = branch_spec(a.branch)
    if a.device == "cuda":
        asr = VerbatimASR(
            str(MODEL), profile="p0_cuda" if prof_name == "p0" else "p1_cuda"
        )
    else:
        asr = VerbatimASR(
            str(MODEL), device="cpu", compute_type="int8", profile=prof_name
        )
    prof = asr.profile
    base = {**prof.decode_kwargs("es"), **override}
    units = union_units()
    if a.ids:
        keep = set(json.loads(Path(a.ids).read_text("utf8")))
        units = [u for u in units if u["unit_id"] in keep]
    if a.limit:
        units = units[: a.limit]
    d = S2 / "asr" / (a.device + a.out_suffix) / a.branch
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "branch": a.branch,
        "device": asr.device,
        "compute_type": asr.compute_type,
        "profile": prof.name,
        "kwargs": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in base.items()
        },
        "chunk_s": chunk_s,
        "overlap_s": overlap_s,
        "ladder": [dict(x) for x in ladder] if ladder else None,
        "seed": SEED,
        "model_info": asr.model_info(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "n_units": len(units),
    }
    (d / "_meta.json").write_text(json.dumps(meta, indent=1, default=str), "utf8")
    t0 = time.perf_counter()
    done = 0
    for u in units:
        p = d / f"{u['unit_id']}.json"
        if p.exists():
            continue
        audio, sr = sf.read(u["wav"], dtype="float32")
        assert sr == 16000 and audio.ndim == 1, u["wav"]
        ctranslate2.set_random_seed(SEED)
        t1 = time.perf_counter()
        if kind == "engine":
            words, diag = (
                asr.transcribe_chunked(audio, chunk_s=chunk_s, overlap_s=overlap_s),
                None,
            )
        else:
            words, diag = transcribe_chunked_branch(
                asr, audio, base, chunk_s, overlap_s, ladder
            )
        rec = {
            "unit_id": u["unit_id"],
            "branch": a.branch,
            "duration_s": len(audio) / 16000,
            "wall_s": time.perf_counter() - t1,
            "diag": diag,
            "words": [w.to_dict() for w in words],
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        done += 1
        if done % 100 == 1:
            print(
                a.device,
                a.branch,
                done,
                "/",
                len(units),
                round(time.perf_counter() - t0),
                "s",
                flush=True,
            )
    print("DONE", a.device, a.branch, round(time.perf_counter() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
