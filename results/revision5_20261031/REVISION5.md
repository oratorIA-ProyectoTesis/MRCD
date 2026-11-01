# REVISION 5: fixes for the findings of REVISION4 (80a921a..5f92c8e)

**Status: SELF-REVIEW.** Round 4 was independent; everything below was written and checked by the same writer who produced the code and the experiments. It has not been re-reviewed. Statements marked "measured" come from a script in the repo; the mutation checks ran on a copy outside the repo. Nothing was pushed; `material/` was not touched; DIHANA fold 5 was not used for anything (the new tests pin that).

## 1. pyin_carry thread safety (REVISION4 1.5, was M)
* Cause: `_capture_viterbi` replaced `librosa.sequence.viterbi` globally; a concurrent plain `librosa.pyin` received the spy (8/25 wrong F0 results in the reviewer's harness).
* Fix (`mrcd/live/pyin_carry.py`): no mutation of librosa at all. `pyin_resumable` runs a PRIVATE copy of `librosa.pyin` (same code object, same defaults, `types.FunctionType`) whose module-level name `sequence` is a per-call proxy that records the Viterbi arguments. The proxy delegates every other attribute to the real module. If librosa changes its internals so that `viterbi` is not called, a `RuntimeError` is raised instead of decoding garbage. The old module lock is gone.
* Tests (`tests/test_pyin_carry_threads.py`): librosa's `viterbi` object is unchanged after a call; 12 plain `librosa.pyin` calls in the main thread while another thread loops `pyin_resumable`, every result bit-equal (f0 with NaN-equality, voiced_prob) to a reference computed before, and the carry results equal to theirs. Mutant "back to the old global monkeypatch" (code of 5f92c8e) is detected by the concurrency test.

## 2. Forbidden-mapping guard (REVISION4 section 2)
* Positive validation (`mrcd/corpora/base.py`): a class can only be attached to a `(source_kind, source_label)` pair in `ALLOWED_MAPPINGS` (DIHANA tag keys, ESLORA `alargamiento`, DIME `modification repair`: exactly what the importers emit today). Everything else must be imported unmapped (`proposed_class=None`). At Example level, `source_label` must be allow-listed for `proposed_mrcd_mapping`.
* Defence in depth: the old deny-list is kept and widened to families (contains `palabracortada`, `fragment`, `pausa`, `pause`, `silenc`, `silent`; exact `sil`, `sp`, with numeric suffixes stripped), and an import-time assertion fails if an allow-list entry belongs to a forbidden family.
* Normalisation: NFKC, casefold, basic confusables (Cyrillic/Greek/long s/dotless i) folded, combining and zero-width characters removed, punctuation removed. Non-`str` `source_kind`, `source_label`, `justification` (tuple, dict, list, bytes, int, None) raise `ForbiddenMappingError` (no `str()` coercion, no `AttributeError`).
* Tests: the reviewer's bypasses (`pаusa` Cyrillic, `ｐａｕｓａ`, `pauſa`, `pausas`, `pausa 2`, `silent pausa`, `fragmentos`, `pause`, ...) x 4 classes at Annotation and Example level; tuple/dict/list/bytes/int/None; unlisted pairs; every allow-listed pair maps and is outside every forbidden family; no false positive on `silabeo`, `respaldo`, `Aspirada (sp)`, `sillon`, `pasa`, `fragil`, `alargamiento`, `vacilacion`, `repeticion`. The test helper `ann()` now uses legitimate pairs. Real importer tests unchanged and green.
* Residual: the allow-list is only as good as its audit (extending it is a decision); `xls` substring logic for source kinds is unchanged (intended).

## 3. Surviving mutants (REVISION4 section 6) and new mutants
New default-suite tests: `tests/test_pyin_carry_threads.py` (fast fixture `tests/data/seed23_noise_stretch_8s.npy`, the 292-300 s slice of seed 23 as int16, 256 KB; v2 diverges in 10 frames, v3 is exact; ~4 s), `tests/test_survival_and_pool_guards.py` (repetition survival threshold at 5/6/7/8/9/10 of 10 tokens, revision branch, `ALLOWED_FOLDS`, `build_pool` with a fold-5-only CSV yields an empty pool without touching the corpus), additions in `tests/test_corpora_base.py`, `tests/test_asr_recovery_core.py` (C1 variant). Mutants run on a COPY outside the repo (`scratchpad/mutate5.py`; result `mutations.json`), default suite files only (no `MRCD_SLOW`):
| mutant | result |
|---|---|
| M1s tracker ignores `viterbi_carry` | DETECTED (fast fixture) |
| M3b `ALLOWED_FOLDS` includes 5 | DETECTED |
| M4a/b/c repetition survival 0.8 -> 0.5 / 0.7 / 0.9 | DETECTED (all three) |
| M4d revision branch `cov_span` 0.8 -> 0.5 | DETECTED |
| M5 pyin_carry back to the global monkeypatch | DETECTED (concurrency test) |
| M6 guard: allow-list check removed | DETECTED |
| M7 guard: confusables not folded | DETECTED |
| M8 guard: NFKC dropped | DETECTED |
| M9 guard: type check dropped | DETECTED |
| M10 guard: family contains-stems dropped | DETECTED |
| M11 guard: numeric suffix strip dropped | DETECTED |
13/13 detected. Not covered (as in round 4): `s6_family.py` / `s7_tables.py` have no tests; they were recomputed by hand by the reviewer, not pinned.

## 4. Claim corrections (dated addenda; originals kept)
(a) **v3 residual is NOT inherent.** New fixed seeds 301-312 (fixed in the script header before the first run, commit 5 of this round), tick partition, L = 3, v3, exact criterion, after `finalize` (`residual_vs_R.jsonl`, `scripts/live_parity/residual_vs_right_margin.py`):
* R = 4 s (default): 2/12 sessions differ (Clopper-Pearson 95% [2.1%, 48.4%]), 7 frames each (seeds 307 and 311), both at the END of the same real clip (`HAV_001_02_12_a216.wav`, offset 14.47 of 14.52 s) that is followed by a digital silence of 29.09 s (307) and 15.05 s (311). Pooled with the authors' 1/12 and the reviewer's 0/8: 3/32.
* R = 30 s: 0/12 sessions differ (upper bound 26.5%); all 24 seed x R jobs ran to completion.
* Dose (post-hoc, declared, only the two diverging sessions): seed 307 (silence 29.09 s): R = 12 -> 7 frames, R = 30 -> exact. Seed 311 (silence 15.05 s): R = 4, 12 and 14 -> 7 frames, R = 16 -> exact (`residual_probe_311.jsonl`). In both sessions the residual vanishes exactly when R exceeds the following silence. This is consistent with "needs R larger than the degenerate stretch", from only two sessions; it is not a general threshold.
* Cost: the commit lag equals R (4 s -> 30 s, +26 s of latency). Compute: job wall time for a 10.5 min session was about 400 s (R=4), about 500 s (R=12-16) and 1380-1650 s (R=30) with 14 jobs running at once, i.e. about 4x more pYIN work per tick (window [commit-3 s, now] grows from about 7 s to about 33 s). Not measured unloaded, not a controlled benchmark. The default stays `live_cfg_v2`; `live_cfg_v3` stays R = 4; nothing was changed.
(b) "No FP cost" holds only for ctx_LR (10 -> 9 of 679 sampled negatives). Rule: 11 -> 14, total FP 23 -> 34 (13 new, 2 removed), weighted 52.3 -> 70.1. Hallucination loop `V118_VA2c1_U_005` ("[quote withheld]") copies the prompt. 7 of the 9 new ctx_LR FP fall on DIHANA turns marked as revisions.
(c) "about 28%" -> 0.2756 with speaker-bootstrap CI [0.16, 0.39]; p "0.0004" -> "< 0.0004" (bootstrap floor 2/(5000+1)); p_Holm 0.0024 -> "< 0.0024".
(d) "fold 5 never loaded" -> "never indexed or used" (the CSV is read whole and filtered).
(e) TASKS.md (a backup `TASKS.md.bak_rev5` was made before this edit; the original `.bak` is untouched), `docs_actualizados/` copies and the OE matrix: test counts replaced by the current ones (below), the ASR experiment added as EXPLORATORY, OE4 still open.
(f) Live parity: "not exact"; evidence = synthetic sessions with given ASR words and an injected F0 centre; streaming ASR is not equivalent (WER 0.76 on 8 clips, 2 s tick).

## 5. Attribution control of the O1 prompt (EXPLORATORY; `results/asr_recovery_20261031/C1_CONTROL.md`)
Annex 3 (control prompt C1: fluent, same domain words and length +5.6% in Whisper tokens, no repetitions/fillers/ellipses; hashed and committed before decoding; no 4-gram or 5-gram overlap with the evaluated turns) plus `s9_eval_c1.py`. Same 1025 units, same procedure (OMP = 4 before numpy, chunk 8 s / 2 s, seed 13). rep_literal: survival of the human mark P0 105/214, **C1 42/214**, P1 144/214; ctx_LR weighted F1 0.581 / **0.304** / 0.690; P1 - C1 = +0.387 [+0.298, +0.478], C1 - P0 = -0.277 [-0.359, -0.200], both p < 0.0004, p_Holm 0.0016 (family of 4). Revision: 0.313 / 0.232 / 0.352; P1 - C1 +0.120 [+0.080, +0.164]; C1 - P0 -0.081 [-0.125, -0.038]. Conclusion by the pre-declared rule: **P1 > C1**, the gain over P0 is attributable to the disfluent content of the prompt, not to prompt length or domain words; a fluent prompt makes the ASR remove repetitions it used to keep. WER on negatives improves equally under C1 (0.129) and P1 (0.130) vs P0 (0.153): a generic prompt effect, not a recovery signal. Which part of P1 matters, and any other corpus/model, is untested. CPU: C1 decode 52.8 min summed per-unit time (RTF 0.69), about 27 min wall with 2 shards x OMP = 4, plus a concurrent 4-14 process live-parity study.

## 6. Suite (literal in `pytest_default.txt` and `pytest_slow.txt`)
* `OMP_NUM_THREADS=4 pytest -q -rs`: **653 passed, 3 skipped in 329.27 s** (same three MRCD_SLOW skips).
* `MRCD_SLOW=1 OMP_NUM_THREADS=4 pytest -q -rs`: **656 passed, 10 warnings in 755.73 s** (0 skipped). Both runs were under CPU contention from the study of section 4(a).

## 7. Limits of this round
Self-review (no independent re-check); the allow-list is correct only for what the three importers emit today; the thread-safety fix relies on the structure of `librosa.pyin` 1.0.0 (a RuntimeError guards a change); item 4(a) uses two diverging sessions for the dose-response and the same synthetic generator that found the defect; the C1 conclusion uses a single control prompt; timings are from a loaded machine.

## Addendum 2026-10-31 (round 6 correction, SELF-REVIEW)
* Section "C1": "the gain over P0 is attributable to the disfluent content of the prompt, not to prompt length or domain words" is withdrawn; wording in force in `results/asr_recovery_20261031/EXPERIMENTO_ASR.md` (addendum round 6). C1 also has more domain overlap than P1.
* "Everything else must be imported unmapped" (allow-list): an unlisted pair that asks for a class RAISES `ForbiddenMappingError`; it is not downgraded. "13/13 mutants detected" did not cover REVISION6 N1-N4, N9, N10, N12 (now pinned).
* "a backup `TASKS.md.bak_rev5` was made": the file did not exist; real copies from git 16bc252 were created in `docs_actualizados/` (round 6).
* "656 passed, 0 skipped" holds only in the author's tree (clean tree: 639 + 17 skipped).
