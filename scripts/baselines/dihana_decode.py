"""DIHANA .ul adapter. Originals are never modified; output = new 16 kHz PCM16 WAVs in results/baselines_20261030/dihana/wav16k/.

Usage:
  python dihana_decode.py verify   # objective codec checks (writes codec_verification.json)
  python dihana_decode.py convert  # decode all 6279 turns with the confirmed codec
Codec hypotheses tested objectively (not assumed from the extension): mu-law, A-law, linear int16 LE, linear int16 BE, all at 8 kHz.
"""
import csv, json, re, sys, time, unicodedata
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
sys.path.insert(0, str(Path(__file__).parent))
from bl_common import CORPORA, OUT, SEED, WHISPER, write_config, sha256_file

ROOT = CORPORA / "01_DIHANA" / "extraido" / "dihana"
D = OUT / "dihana"


def ulaw_decode(b: np.ndarray) -> np.ndarray:
    u = (~b.astype(np.uint8)).astype(np.int32)
    sign = u & 0x80; exp = (u >> 4) & 7; man = u & 15
    s = (((man << 3) + 0x84) << exp) - 0x84
    return np.where(sign != 0, -s, s).astype(np.int16)


def alaw_decode(b: np.ndarray) -> np.ndarray:
    a = (b.astype(np.uint8) ^ 0x55).astype(np.int32)
    sign = a & 0x80; exp = (a >> 4) & 7; man = a & 15
    s = np.where(exp == 0, (man << 4) + 8, ((man << 4) + 0x108) << np.maximum(exp - 1, 0))
    return np.where(sign != 0, s, -s).astype(np.int16)  # A-law: sign bit set = positive


def decode(raw: bytes, codec: str) -> np.ndarray:
    if codec == "ulaw": return ulaw_decode(np.frombuffer(raw, np.uint8))
    if codec == "alaw": return alaw_decode(np.frombuffer(raw, np.uint8))
    if codec == "lin16le": return np.frombuffer(raw[: len(raw) // 2 * 2], "<i2")
    if codec == "lin16be": return np.frombuffer(raw[: len(raw) // 2 * 2], ">i2")
    raise ValueError(codec)


def to16k(x16: np.ndarray) -> np.ndarray:
    y = x16.astype(np.float32) / 32768.0
    return resample_poly(y, 2, 1).astype(np.float32)  # 8k -> 16k, deterministic polyphase


def norm_words(s):
    s = unicodedata.normalize("NFD", s.lower()); s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", s)


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p = d[:]; d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def turns():
    with (D / "turns.csv").open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def verify():
    rows = turns()
    rng = np.random.default_rng(SEED)
    # (1) global statistics over all 6279 files
    stats = {}
    hist = np.zeros(256, np.int64); dur_words = []
    for r in rows:
        raw = (ROOT / r["ul_rel"]).read_bytes()
        hist += np.bincount(np.frombuffer(raw, np.uint8), minlength=256)
        nw = len(norm_words(r["ref_text_marks_stripped"]))
        dur_words.append((len(raw), nw))
    top = np.argsort(-hist)[:8]
    stats["byte_hist_top8"] = {f"0x{int(k):02x}": round(float(hist[k] / hist.sum()), 4) for k in top}
    stats["frac_bytes_in_mulaw_silence_zone_0x7e7f_0xfe_0xff"] = float(hist[[0x7e, 0x7f, 0xfe, 0xff]].sum() / hist.sum())
    stats["frac_bytes_in_alaw_silence_zone_0x54_0x55_0xd4_0xd5"] = float(hist[[0x54, 0x55, 0xd4, 0xd5]].sum() / hist.sum())
    arr = np.array(dur_words, float)
    arr = arr[arr[:, 1] >= 3]
    rate = {}
    for name, bps in [("ulaw/alaw 8k (1 B/sample)", 1), ("lin16 8k (2 B/sample)", 2)]:
        dur = arr[:, 0] / bps / 8000.0
        w = arr[:, 1] / dur
        rate[name] = {"median_words_per_s": float(np.median(w)), "p05": float(np.percentile(w, 5)), "p95": float(np.percentile(w, 95)),
                      "total_hours": float(arr[:, 0].sum() / bps / 8000 / 3600)}
    stats["speech_rate_by_hypothesis"] = rate
    stats["official_total_hours_readme"] = 5.5
    stats["total_hours_all_turns_ulaw8k"] = float(sum(len((ROOT / r['ul_rel']).read_bytes()) for r in rows) / 8000 / 3600)
    # (2) ASR plausibility on a fixed sample drawn from DEV folds only
    dev = [r for r in rows if r["split"] == "dev" and 12 <= len(norm_words(r["ref_text_marks_stripped"])) <= 40]
    pick = [dev[i] for i in rng.choice(len(dev), 24, replace=False)]
    from faster_whisper import WhisperModel
    m = WhisperModel(str(WHISPER), device="cpu", compute_type="int8", cpu_threads=8)
    res = {}
    for codec in ["ulaw", "alaw", "lin16le", "lin16be"]:
        E = N = 0; per = []
        for r in pick:
            raw = (ROOT / r["ul_rel"]).read_bytes()
            y = to16k(decode(raw, codec))
            segs, _ = m.transcribe(y, language="es", beam_size=1, temperature=0.0, condition_on_previous_text=False)
            hyp = " ".join(s.text for s in segs)
            e, n = wer(r["ref_text_marks_stripped"], hyp); E += e; N += n
            per.append({"turn_id": r["turn_id"], "wer": e / n, "hyp": hyp[:120]})
        res[codec] = {"WER": E / N, "n_turns": len(pick), "n_ref_words": N, "examples": per[:3]}
    # also: ulaw read as 16 kHz-rate (wrong rate) to check rate sensitivity
    E = N = 0
    for r in pick:
        raw = (ROOT / r["ul_rel"]).read_bytes()
        y = ulaw_decode(np.frombuffer(raw, np.uint8)).astype(np.float32) / 32768.0
        segs, _ = m.transcribe(y, language="es", beam_size=1, temperature=0.0, condition_on_previous_text=False)  # treated as 16 kHz
        e, n = wer(r["ref_text_marks_stripped"], " ".join(s.text for s in segs)); E += e; N += n
    res["ulaw_misread_as_16kHz"] = {"WER": E / N}
    stats["asr_wer_by_hypothesis"] = res
    stats["sample_turn_ids"] = [r["turn_id"] for r in pick]
    best = min(["ulaw", "alaw", "lin16le", "lin16be"], key=lambda c: res[c]["WER"])
    stats["confirmed_codec"] = best if res[best]["WER"] < 0.5 else None
    (D / "codec_verification.json").write_text(json.dumps(stats, indent=1, ensure_ascii=True))
    write_config(D / "codec_run", "dihana_codec_verify", __file__, {"sample": "24 dev-fold turns, 12-40 ref words"}, {"turns.csv": sha256_file(D / "turns.csv")})
    print(json.dumps({k: v for k, v in stats.items() if k != "sample_turn_ids"}, indent=1, ensure_ascii=True)[:3500])


def convert():
    cv = json.loads((D / "codec_verification.json").read_text())["confirmed_codec"]
    if cv is None:
        raise SystemExit("codec not confirmed; refusing to convert")
    (D / "wav16k").mkdir(exist_ok=True)
    rows = turns(); man = []
    for r in rows:
        raw = (ROOT / r["ul_rel"]).read_bytes()
        x = to16k(decode(raw, cv))
        p = D / "wav16k" / f"{r['turn_id']}.wav"
        sf.write(p, x, 16000, subtype="PCM_16")
        man.append({"turn_id": r["turn_id"], "ul_sha256": __import__("hashlib").sha256(raw).hexdigest(), "wav_sha256": sha256_file(p), "dur_s": len(x) / 16000})
    (D / "wav16k_manifest.json").write_text(json.dumps(man))
    write_config(D / "convert_run", "dihana_decode_convert", __file__, {"codec": cv, "src_rate_hz": 8000, "dst_rate_hz": 16000,
                 "resampler": "scipy.signal.resample_poly(up=2,down=1)", "n": len(man)}, {"wav16k_manifest.json": sha256_file(D / "wav16k_manifest.json")})
    print("converted", len(man), "hours", sum(m["dur_s"] for m in man) / 3600)


if __name__ == "__main__":
    {"verify": verify, "convert": convert}[sys.argv[1]]()
