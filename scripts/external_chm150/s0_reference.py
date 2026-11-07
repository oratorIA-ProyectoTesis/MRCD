"""CHM150 step 0: import the corpus and derive the transcript reference (no system output is read or produced).

Writes OUT/reference.json (LOCAL ONLY: contains transcription-derived marks; never committed) and
OUT/reference_summary.json (counts only). Usage: python -I s0_reference.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pin  # noqa: E402,F401  (MUST be first)

import json  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

from mrcd.context_eval.transcript_reference import utterance_reference  # noqa: E402
from mrcd.corpora.chm150 import import_chm150  # noqa: E402
from mrcd.engine import DisfluencyEngine  # noqa: E402


def main():
    ex = import_chm150(_pin.CORPUS_ROOT)
    refs = {}
    for e in ex:
        r = utterance_reference(e.original_text)
        refs[e.recording_id] = {
            "speaker": e.group_id,
            "sex": e.extra["sex"],
            "audio_relpath": e.extra["audio_relpath"],
            "audio_sha256": e.source_files[0].sha256,
            "duration_s": e.audio_duration_s,
            **r,
        }
    dur = np.array([r["duration_s"] for r in refs.values()])
    pos = [u for u, r in refs.items() if r["rep_literal"]]
    fill = Counter()
    for r in refs.values():
        fill.update(r["fillers"])
    spk = {r["speaker"] for r in refs.values()}
    summary = {
        "n_utterances": len(refs),
        "n_speakers": len(spk),
        "speakers_by_sex": dict(Counter(s[0] for s in spk)),
        "transcription_sha256": ex[0].source_files[1].sha256,
        "duration_s": {
            "total_h": float(dur.sum() / 3600),
            "min": float(dur.min()),
            "median": float(np.median(dur)),
            "p95": float(np.percentile(dur, 95)),
            "max": float(dur.max()),
            "n_lt_3s": int((dur < 3.0).sum()),
            "n_ge_engine_min": int((dur >= DisfluencyEngine.MIN_AUDIO_S).sum()),
            "engine_min_audio_s": float(DisfluencyEngine.MIN_AUDIO_S),
        },
        "rep_literal": {
            "positive_utterances": len(pos),
            "positive_speakers": len({refs[u]["speaker"] for u in pos}),
            "marks": int(sum(refs[u]["n_rep_marks"] for u in pos)),
            "marks_by_n": dict(
                Counter(str(m["n"]) for u in pos for m in refs[u]["rep_marks"])
            ),
            "positive_without_filler_removal": int(
                sum(r["rep_literal_without_filler_removal"] for r in refs.values())
            ),
            "indeterminate": 0,
        },
        "nonlexical_filled_pauses_by_form": dict(fill.most_common()),
        "utterances_with_filled_pause": int(
            sum(r["n_fillers"] > 0 for r in refs.values())
        ),
        "este_tokens_not_evaluated": int(sum(r["n_este"] for r in refs.values())),
    }
    _pin.OUT.mkdir(parents=True, exist_ok=True)
    (_pin.OUT / "reference.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=0), "utf8"
    )
    (_pin.OUT / "reference_summary.json").write_text(
        json.dumps(summary, indent=1), "utf8"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
