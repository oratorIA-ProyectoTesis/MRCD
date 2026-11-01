"""Comparability / leakage check of the control prompt C1 against P1 (run BEFORE any C1 decode; no ASR is run).

Writes ``c1_prompt_check.json``: length (words, characters, Whisper tokens), repetition and filler checks, and
n-gram overlap (n = 3, 4, 5) of C1 and of P1 with (a) the reference text of EVERY pool turn (folds 1-4) and
(b) the 1025 units of the frozen subset.
"""

import _pin  # noqa: F401  (MUST be first)
import json
import re

from mrcd.asr_recovery.common import OUT, WHISPER, load_pool, sha_text
from mrcd.asr_recovery.decode import C1_PROMPT, P1_PROMPT
from mrcd.context_eval.reference import _norm_tokens

FILLERS = {"eh", "este", "mmm", "mm", "ah", "o", "sea", "bueno", "digo", "perdon", "perdone", "pues", "ehh", "em"}


def toks(s):
    return _norm_tokens(s)


def grams(t, n):
    return {" ".join(t[i : i + n]) for i in range(len(t) - n + 1)}


def adjacent_repeats(t):
    return sum(a == b for a, b in zip(t, t[1:]))


def main():
    from mrcd.extractors.linguistic import VerbatimASR

    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
    tok = asr.model.hf_tokenizer
    pool = load_pool()
    frame = {s["unit_id"] for s in json.loads((OUT / "subset.json").read_text("utf8"))["frame"]}
    rep = {}
    for name, p in (("P1", P1_PROMPT), ("C1", C1_PROMPT)):
        t = toks(p)
        d = {
            "sha256": sha_text(p),
            "words": len(t),
            "chars": len(p),
            "whisper_tokens": len(tok.encode(" " + p, add_special_tokens=False).ids),
            "adjacent_identical_word_pairs": adjacent_repeats(t),
            "ellipses": p.count("..."),
            "filler_tokens_present": sorted(set(t) & FILLERS),
            "has_horario": "horario" in t or "horarios" in t,
            "has_preguntar": "preguntar" in t,
        }
        for n in (3, 4, 5):
            g = grams(t, n)
            for label, units in (("pool", pool), ("frame", [u for u in pool if u["unit_id"] in frame])):
                hit = sorted(
                    {x for u in units for x in g & grams(toks(u["text_A"]), n)}
                )
                nt = sum(1 for u in units if g & grams(toks(u["text_A"]), n))
                d[f"{n}gram_vs_{label}"] = {"distinct_ngrams_hit": len(hit), "turns_hit": nt, "examples": hit[:8]}
        rep[name] = d
    (OUT / "c1_prompt_check.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), "utf8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
