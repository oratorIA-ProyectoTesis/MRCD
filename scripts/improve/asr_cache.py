"""Freeze ASR caches per corpus with OMP_NUM_THREADS=4 pinned BEFORE ctranslate2 is imported.

ASR input = the unit WAV exactly as stored (NO padding: padding is a downstream decision under study).
Cache file per unit: results/mejoras_20261030/asr_cache/<corpus>/<unit_id>.json (atomic write, resumable).
Usage: python asr_cache.py CORPUS SHARD NSHARDS [rev]       (transcribe)
       python asr_cache.py CORPUS manifest              (hash everything, write manifest.json)
"""
import _pin_env  # MUST be first: OMP_NUM_THREADS=4 before numpy (see _pin_env.py)
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import REPO  # noqa  (adds repo to sys.path)
from mrcd.reproducibility import pin_asr_threads, asr_environment
PIN = pin_asr_threads(4)          # env was already set by _pin_env before numpy; this only records it
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
assert os.environ["OMP_NUM_THREADS"] == "4" and PIN["action"] == "kept", PIN
from common import CACHE, WHISPER, manifest, sha256_file  # noqa
import soundfile as sf  # noqa

CHUNK_S, OVERLAP_S = 8.0, 2.0


PILOT_CANARY = (Path("C:/zArnes-tesis/mrcd/results/reproduccion_piloto_20261030/ameresco_000_asr.json"),
                Path("C:/zArnes-tesis/mrcd/results/baselines_20261030/ameresco/inputs/BAQ_001_03_16_a12.wav"))


def canary(asr):
    """Abort unless the ASR reproduces the pilot transcript of one AMERESCO clip (guards against a silently ineffective OMP pin)."""
    ref = json.loads(PILOT_CANARY[0].read_text("utf8"))
    a, _ = sf.read(PILOT_CANARY[1], dtype="float32")
    got = [w.text for w in asr.transcribe_chunked(a, chunk_s=CHUNK_S, overlap_s=OVERLAP_S)]
    if got != [w["text"] for w in ref]:
        raise SystemExit("CANARY FAILED: ASR does not reproduce the pilot transcript; OMP pin ineffective in this process")
    print("canary ok (pilot transcript reproduced)", flush=True)


def transcribe(corpus, shard, nshard, rev=False, dev_fwd=False):
    from mrcd.extractors.linguistic import VerbatimASR
    order = manifest(corpus)[::-1] if rev else (manifest(corpus)[2463:] if dev_fwd is True else manifest(corpus)[dev_fwd:] if dev_fwd else manifest(corpus))
    man = [m for i, m in enumerate(order) if i % nshard == shard]
    d = CACHE / corpus; d.mkdir(parents=True, exist_ok=True)
    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
    canary(asr)
    t0 = time.perf_counter()
    for k, m in enumerate(man):
        p = d / f"{m['unit_id']}.json"
        if p.exists(): continue
        audio, sr = sf.read(m["wav"], dtype="float32"); assert sr == 16000 and audio.ndim == 1
        tic = time.perf_counter()
        words = asr.transcribe_chunked(audio, chunk_s=CHUNK_S, overlap_s=OVERLAP_S)
        rec = {"unit_id": m["unit_id"], "n_samples": int(len(audio)), "chunk_s": CHUNK_S, "overlap_s": OVERLAP_S,
               "omp_num_threads": os.environ["OMP_NUM_THREADS"], "sec": time.perf_counter() - tic,
               "words": [w.to_dict() for w in words]}
        tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8"); tmp.replace(p)
        if k % 50 == 0: print(f"{corpus} shard {shard} {k+1}/{len(man)} {time.perf_counter()-t0:.0f}s", flush=True)
    print("DONE", corpus, shard, flush=True)


def write_manifest(corpus):
    man = manifest(corpus)
    d = CACHE / corpus
    units = {}
    for m in man:
        p = d / f"{m['unit_id']}.json"
        if not p.exists(): continue
        units[m["unit_id"]] = {"wav": m["wav"], "wav_sha256": sha256_file(m["wav"]), "cache_sha256": sha256_file(p)}
    out = {"corpus": corpus, "n_units": len(units), "n_manifest_units": len(man), "chunk_s": CHUNK_S, "overlap_s": OVERLAP_S,
           "asr_environment": asr_environment(WHISPER), "pin_report": PIN, "padding": "none (ASR on stored WAV)",
           "compute_type": "int8", "device": "cpu", "units": units}
    (d / "manifest.json").write_text(json.dumps(out, indent=1), "utf8")
    print(corpus, "manifest", len(units), "of", len(man))


if __name__ == "__main__":
    c = sys.argv[1]
    if sys.argv[2] == "manifest": write_manifest(c)
    else: transcribe(c, int(sys.argv[2]), int(sys.argv[3]), len(sys.argv) > 4 and sys.argv[4] == 'rev', (True if len(sys.argv) > 4 and sys.argv[4] == 'devfwd' else (int(sys.argv[4][3:]) if len(sys.argv) > 4 and sys.argv[4].startswith('off') else False)))
