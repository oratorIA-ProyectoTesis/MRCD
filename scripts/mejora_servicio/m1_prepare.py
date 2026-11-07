"""M1 preparation (before the freeze): the session plan over the P1 frame and the per-fold refits of rep_ctx_p1.

    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\m1_prepare.py

* ``m1/sessions.json``: deterministic concatenation plan (ids, offsets, folds; no audio, no text).
* ``m1/routes/fold_<k>.json``: rep_ctx_p1 refit with the packaged recipe (``scripts/s3/train_final.py``:
  ``fusion_text.select_fit_w`` on the turn-level P1 text of the frame) on the frame turns of the speakers NOT in outer
  fold k. Only training labels are read; nothing is evaluated here.
"""

import common as K  # noqa: F401  (pins OMP before numpy)

import json
import wave

import numpy as np

from mrcd import fusion_text as F
from mrcd.context_eval.textfeat import featurize

M1 = K.OUT / "m1"


def wav_seconds(uid: str) -> float:
    with wave.open(str(K.WAV_DIR / f"{uid}.wav")) as w:
        return w.getnframes() / w.getframerate()


def p1_text(uid: str) -> str:
    d = json.loads((K.RC.OUT / "asr" / "P1" / f"{uid}.json").read_text("utf8"))
    return " ".join(w["text"] for w in d["words"])


def main():
    (M1 / "routes").mkdir(parents=True, exist_ok=True)
    fr = K.frame()["rows"]
    durs = {r["unit_id"]: wav_seconds(r["unit_id"]) for r in fr}
    sessions = K.build_sessions(fr, durs)
    (M1 / "frame.json").write_text(json.dumps(fr, indent=0), "utf8")
    (M1 / "sessions.json").write_text(json.dumps(sessions, indent=0), "utf8")
    ids = [r["unit_id"] for r in fr]
    y = np.array([r["y_repetition"] for r in fr])
    w = np.array([r["w"] for r in fr])
    g = np.array([r["speaker"] for r in fr])
    f = np.array([r["fold"] for r in fr])
    X = featurize([p1_text(u) for u in ids], "rep_literal")
    for k in sorted(set(f.tolist())):
        tr = f != k
        sel = F.select_fit_w(X[tr], y[tr], g[tr], w[tr])
        route = F.route_from_fit(
            sel,
            "rep_literal",
            "p1_disfluent_prompt",
            F.NEVER,
            f"ctx_LR on P1 text, frame turns of outer folds != {k} (DIHANA folds 1-4); evaluation refit",
        )
        (M1 / "routes" / f"fold_{k}.json").write_text(route.to_json(), "utf8")
    print(
        json.dumps(
            {
                "frame_turns": len(fr),
                "positives": int(y.sum()),
                "speakers": len(set(g.tolist())),
                "sessions": len(sessions),
                "audio_h": round(sum(s["total_s"] for s in sessions) / 3600, 3),
                "speech_h": round(sum(durs.values()) / 3600, 3),
                "padded_sessions": sum(
                    1 for s in sessions if s["total_s"] > s["speech_end_s"]
                ),
                "folds": {int(k): int((f == k).sum()) for k in sorted(set(f.tolist()))},
            }
        )
    )


if __name__ == "__main__":
    main()
