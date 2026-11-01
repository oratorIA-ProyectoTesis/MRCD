# ASR recovery of repetitions / revisions (EXPLORATORY, DIHANA folds 1-4, fold 5 never loaded)

Protocol frozen before any decode: `PROTOCOL.md` (sha256 `3523e36a...6e71`, 2026-10-31T07:09:56Z); annexes `PROTOCOL_ANNEX_20261031_1_SUBSET.md`, `PROTOCOL_ANNEX_20261031_2_NOTES.md`. Full numbers: `TABLES.md`; examples: `ERRORS.md`; per-unit predictions and `config.json`: `run_o1/`, `run_o2_o3/`; ASR word lists: `asr/P0|P1|P2/`; code: `mrcd/asr_recovery/`, `scripts/asr_recovery/`, tests `tests/test_asr_recovery_*.py`.

## Headline
* **O1 (verbatim prompt P1): higher survival of literal repetitions on this frame (exploratory; the word "recovers" is a shorthand, see the qualifiers in `PERFIL_P1.md` and `C1_CONTROL.md`: a fluent control did not reproduce the gain, the cause is unknown, ctx_LR 0.690 is a DIHANA text-model score, not an engine result).** Survival of the human mark 105/214 -> 144/214 (49.1 % -> 67.3 %, +18.2 pp, 95 % CI [13.2, 23.3]; 40 of 109 destroyed marks become visible, 1 lost). ctx_LR weighted F1 0.581 -> 0.690 (+0.110, CI [0.064, 0.159], p_Holm = 0.0024 over the 6 pre-declared tests): about 28 % of the A->B loss on the same frame (rule: 30 %). False positives on sampled negatives did not rise (ctx_LR 10 -> 9 of 679; rule 11 -> 14). WER against the reference improved (0.153 -> 0.130 on negatives), but the fluent control C1 reaches 0.129: WER is equal under C1 and is NOT a recovery signal. Still exploratory: one corpus, telephone, weights extrapolate from a sampled frame, best of 2 variants (P2 is practically P1).
* **O1 on revisions: weak, not demonstrated.** Survival 32.9 % -> 36.0 % (+3.1 pp, CI [-1.3, 7.9]); ctx_LR F1 0.313 -> 0.352 (+0.039, CI [0.005, 0.072], p = 0.021, p_Holm = 0.104). The rule got slightly worse (0.152 -> 0.133, n.s.). Revision stays poorly detected even with perfect text (F1_A = 0.389).
* **O2 (cached ASR signals) and O3 (acoustic DTW): no recovery.** Alone they are far worse than ctx_LR (rep_literal F1 0.356 / 0.233 / 0.236 vs 0.537). Combined with ctx_LR: rep_literal +0.011 / +0.019 / +0.006 (all n.s.); revision -0.022 / -0.019 / -0.038 (n.s. or negative). O3 acoustic does fire on some destroyed repetitions (recall 0.25-0.32 vs 0.10 for ctx_LR) but with 213-247 new false positives.

## Family (pre-declared, Holm over 6; best variant per O vs control ctx_LR on B / P0 text; p optimistic because the best of 2 is chosen)
| task | O | best | F1 | delta [95 % CI] | p | p_Holm | verdict |
|---|---|---|---|---|---|---|---|
| rep_literal | O1 | P1 | 0.690 | +0.110 [+0.064, +0.159] | 0.0004 | 0.0024 | supported after Holm (exploratory) |
| rep_literal | O2 | C+S | 0.548 | +0.011 [-0.014, +0.038] | 0.460 | 0.662 | not demonstrated |
| rep_literal | O3 | C+AC_asr | 0.556 | +0.019 [-0.008, +0.046] | 0.166 | 0.662 | not demonstrated |
| revision | O1 | P1 | 0.352 | +0.039 [+0.005, +0.072] | 0.021 | 0.104 | not demonstrated |
| revision | O2 | C+S | 0.264 | -0.022 [-0.055, +0.014] | 0.228 | 0.662 | not demonstrated |
| revision | O3 | C+AC_asr | 0.267 | -0.019 [-0.048, +0.009] | 0.184 | 0.662 | not demonstrated |

## O1 details (frame: 346 positive turns of the union + 679 sampled negatives, 122 speakers; negatives weighted by inverse sampling fraction)
| task | variant | survival | w-F1 ctx_LR | w-F1 rule | recall ctx_LR on P0-destroyed | new FP / removed FP (ctx_LR) | share of A-P0 recovered (ctx_LR / rule) |
|---|---|---|---|---|---|---|---|
| rep_literal | A (ref.) | - | 0.979 | 0.932 | - | - | - |
| rep_literal | P0 | 0.491 | 0.581 | 0.577 | 0.101 | - | - |
| rep_literal | P1 | 0.673 | 0.690 | 0.682 | 0.422 | 9 / 6 | 0.28 / 0.30 |
| rep_literal | P2 | 0.673 | 0.690 | 0.684 | 0.422 | 9 / 6 | 0.28 / 0.30 |
| revision | A (ref.) | - | 0.389 | 0.237 | - | - | - |
| revision | P0 | 0.329 | 0.313 | 0.152 | 0.370 | - | - |
| revision | P1 | 0.360 | 0.352 | 0.133 | 0.426 | 13 / 9 | 0.51 / -0.22 |

P0 re-decode equals the frozen cache text in 1025/1025 units. ctx_LR was NOT retrained on the new text (thresholds tuned on P0 text). P1 changed the text of 544/1025 units (WER P0 vs P1 on negatives 0.060); 2 outputs echo the prompt, empty outputs 2 -> 3, loop hallucinations (>= 5 equal tokens) 4 / 4 / 3 for P0 / P1 / P2. Cost: summed decode time 105 vs 99 min (+7 %).
Recovered examples are mostly textbook (`si, si`, `[quote withheld]`, `[quote withheld]`). Of the 9 new FP, several are real repetitions that DIHANA files under other labels (`[quote withheld]`, `de, del`: sub-type R4; cut-word restarts) and one is a hallucination loop, so part of the FP cost is a label-definition artefact.

## O2 / O3 details (all label-defined turns of folds 1-4; same outer partition; ctx_LR(B) reproduces the parent decisions with 0 mismatches)
* rep_literal (214 pos; ctx_LR B F1 0.537, A 0.970): O2 S 0.356, C+S 0.548; O3 AC_asr 0.233, AC_vad 0.236, C+AC_asr 0.556, C+AC_vad 0.543. Recovered share of A-B: 0.025 (C+S), 0.044 / 0.014 (O3 combos). Recall on destroyed positives: 0.10 (ctx_LR), 0.09 (C+S), 0.11 (C+AC_asr), 0.25 / 0.32 (acoustic only, at precision about 0.20).
* revision (161 pos; ctx_LR B 0.286, A 0.345): S 0.237, C+S 0.264, AC_asr 0.220, AC_vad 0.223, C+AC_asr 0.267, C+AC_vad 0.247. No variant beats ctx_LR(B).
* Reading (not tested further): a 20 ms MFCC/DTW score separates repeats only weakly in telephone speech with short words; the VAD segmentation (no ASR) does as well as the ASR one, so the information does not depend on the ASR, but it is too weak to survive the false positives from natural sound/function-word repetitions.

## Discarded
O1 P2 (identical to P1, no added value); O2 signals alone and O3 acoustic alone (worse than ctx_LR); O2/O3 for revision (negative); O1 for revision as a confirmed gain.

## Limits
One corpus (telephone, peninsular, train-information domain), no inter-annotator agreement; labels from DIHANA `<r>` marks, with B/C/D sub-types excluded from rep_literal and revision mixing sub-types; survival is a heuristic; the O1 frame covers only speakers with a positive and uses sampling weights (weighted F1 is an estimate, bootstrapped by speaker but not over the negative sampling); the positive subset is not a random sample of turns; best-of-2 selection makes p-values optimistic; ctx_LR not retrained on P1 text (a retrained model could do better or worse); one Whisper size/quantisation and one prompt; the P1 prompt was written once without test text but with the target phenomenon in mind (domain-neutral wording); R1-R4 sub-type breakdown not produced. Fold 5 was never loaded. The rep_literal gain must be checked on an independent corpus before any claim.

---

## Addendum 2026-10-31 (round 5, self-review of the round-4 findings; the original text above is kept unchanged)
The round-4 review was independent; this addendum was written by the same writer as the experiment, so it is a self-correction, not a new independent check.

1. **Attribution (headline "O1 recovers ... verbatim prompt P1")**: narrowed. P1 confounds style, length and domain words. The control prompt C1 (fluent, same length and domain, no repetitions; `C1_CONTROL.md`, annex 3 hashed and committed before decoding) does NOT reproduce the gain: survival 105 -> 42 of 214 under C1 versus 144 under P1; ctx_LR w-F1 0.581 (P0) / 0.304 (C1) / 0.690 (P1); P1 - C1 = +0.387 [+0.298, +0.478], p < 0.0004, p_Holm 0.0016 (family of 4). [Revised in round 6: the original sentence here said the gain was "attributable to the disfluent content of the prompt"; see addendum below.] In this exploratory DIHANA experiment, a fluent control of equal length and domain did not reproduce the gain observed with P1; a written, fluent prompt suppresses repetitions and P1 preserves them; which feature matters (oral vs written register, punctuation, domain overlap, the filler list already present in P0) is not proven. Wording to use: "exploratory: a disfluent prompt recovers, +0.11 w-F1 [0.06, 0.16] on this frame; a fluent prompt of the same length and domain removes repetitions".
2. **"False positives on sampled negatives did not rise"**: true only for ctx_LR (10 -> 9 of 679; weighted FP 52.4 -> 55.4). For the rule it is false: FP on sampled negatives 11 -> 14 and total FP 23 -> 34 (13 new, 2 removed); weighted rule FP 52.3 -> 70.1. "Without FP cost" must not be written for the rule. Of the 9 new ctx_LR FP, 7 fall on turns that DIHANA marks as revisions (rep_literal label 0 but revision positives of the exhaustive frame; P1 inserted or recovered a duplicate such as "[quote withheld]"); 2 are on sampled negatives, one of them a prompt-induced hallucination loop in `V118_VA2c1_U_005` ("[quote withheld]", copying the prompt's "por el, por el"). Whether the 7 duplicates are real repetitions cannot be decided without audio.
3. **"about 28 % of the A->B loss"**: point estimate 0.2756 on the same frame, speaker-bootstrap interval [0.16, 0.39] (round-4 recomputation). State it with the interval.
4. **p = 0.0004** is the floor of the 5000-draw bootstrap (2/(5000+1)); write "p < 0.0004". p_Holm = 0.0024 therefore reads "< 0.0024".
5. **Fold 5**: "fold 5 never loaded" (title line, PROTOCOL.md section 0, which is frozen and not edited) is replaced by "fold 5 is never indexed or used". `build_pool` reads the whole `turns.csv` and filters rows with `fold in ALLOWED_FOLDS`; no fold-5 unit is in the pool. Round-5 tests pin `ALLOWED_FOLDS == (1,2,3,4)` and that a CSV with only fold-5 rows yields an empty pool without touching the corpus.
6. **WER**: the improvement (0.153 -> 0.130 on negatives) is a generic prompt effect: C1 gives 0.129 while destroying repetitions. It is not evidence of better recovery.
7. **Prompt text**: "contains no text from any DIHANA turn" is read as "no sentence-level overlap": P1 shares 3-grams with 8 DIHANA turns (2 in the frame; one 4-gram "si si es el" in one turn). Excluding the 2 overlapping frame units changes w-F1 from 0.581 -> 0.690 to 0.580 -> 0.687 (round-4 check).
8. Protocol order is evidenced by file mtimes for annexes 1-2 (the `results/` tree is git-ignored). Annex 3 (C1) was hashed and committed to git before its decode (commit 248037f).

---

## Addendum 2026-10-31 (round 6 correction; SELF-REVIEW by the same writer, pending an independent reviewer)
REVISION6 (C3.1, C3.2) found that "the gain is attributable to the disfluent content of the prompt" (item 1 of the round-5 addendum) over-claims. The sentence in the current text was replaced; this addendum records the wording now in force.

**Wording in force:** In this exploratory DIHANA experiment, a fluent control of equal length and domain did not reproduce the gain observed with P1; a written, fluent prompt suppresses repetitions and P1 preserves them; which feature matters (oral vs written register, punctuation, domain overlap, the filler list already present in P0) is not proven.

Qualifiers: (a) P0 is itself a prompt with a filler list (`"eh, este, o sea, mmm, bueno, tipo, ah, este..."`), so P1 vs P0 contrasts a repetition/self-correction prompt with a filler-list prompt, not "disfluent vs no prompt"; (b) there is no no-prompt control; (c) C1 has MORE domain overlap with the evaluated turns than P1 (42 vs 2 frame turns sharing a 3-gram), so it is a closer-domain control, not a weaker one; (d) one control prompt and one length only; C1 was written after the P1 result was known; (e) not a confirmatory result.
Numbers are unchanged (C1_CONTROL.md). The two overlapping round-5 addenda are kept as history; where they differ, this one prevails.
