#!/usr/bin/env python3
"""Run an independent Gemini judge over the existing blind adjudication queue."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.llm_judge as J  # noqa: E402

CASE = ROOT / "data/case"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    a = ap.parse_args()
    J.load_env_file()
    is_groq = a.model.startswith(("llama-", "mixtral-", "gemma-"))
    key_name = "GROQ_API_KEY" if is_groq else "GEMINI_API_KEY"
    key = os.environ.get(key_name)
    if not key:
        raise SystemExit(f"Falta {key_name}. Configúrala antes de ejecutar el segundo juez.")

    queue = json.loads((CASE / f"{a.vid}.adjudication_queue.json").read_text(encoding="utf-8"))
    meta = json.loads((CASE / f"{a.vid}.meta.json").read_text(encoding="utf-8"))
    words = json.loads((ROOT / meta["words_json"]).read_text(encoding="utf-8"))
    z = np.load(ROOT / meta["features_npz"], allow_pickle=False)
    mp4 = ROOT / meta["mp4"]
    provider_name = "groq" if is_groq else "gemini"
    out = CASE / f"{a.vid}.adjudication_{provider_name}.jsonl"
    previous = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()] if out.exists() else []
    done = {r["window_index"] for r in previous if r.get("label")}
    todo = [c for c in queue if c["window_index"] not in done]
    print(f"[gemini] {len(queue)} en cola · {len(done)} ya juzgadas · {len(todo)} pendientes · modelo {a.model}")
    if not todo:
        return

    errors = ROOT / "data/judge/errors_gemini.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        for c in todo:
            start, end = float(c["start"]), float(c["end"])
            try:
                prev, seg, nxt = J.context_text(words, start, end)
                prompt = J.build_prompt(prev, seg, nxt, J.acoustic_summary(z, start, end))
                images = J.frames_at(mp4, [start, (start + end) / 2, end])
                if any(name in prompt.lower() for name in ("audio_only", "audio_text", "trimodal", "gpt-4o")):
                    raise AssertionError("fuga de predicciones o juez previo en el prompt")
                result = None
                for attempt in range(3):
                    try:
                        caller = J.call_groq if is_groq else J.call_gemini
                        result = J.ask_judge(caller, a.model, prompt, images, key, J.CLASSES)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 ** attempt)
                value, n_frames, note = result
                rec = {"window_index": c["window_index"], "start": start, "end": end,
                       "label": value["classification"], "confidence": value.get("confidence"),
                       "rationale": value.get("rationale"), "model": a.model, "blind": True,
                       "n_frames": n_frames, "note": note}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            except Exception as exc:
                J.log_failure(errors, window_index=c["window_index"], start=start, end=end,
                              kind=type(exc).__name__, reason=str(exc)[:300],
                              raw=getattr(exc, "raw", "")[:500])
                print(f"[gemini] error en ventana {c['window_index']}: {type(exc).__name__}: {str(exc)[:160]}")
    print(f"[gemini] listo -> {out}")


if __name__ == "__main__":
    main()