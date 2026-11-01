# C1 control prompt: does a fluent control reproduce the O1 gain of P1? (EXPLORATORY; annex 3)

Dated 2026-10-31. Protocol: `PROTOCOL_ANNEX_20261031_3_C1_CONTROL.md` (sha256 `f9f1537d...7a23`, hashed 09:03:47Z and committed to git as 248037f BEFORE the first C1 decode; decode started 09:04Z). Script `scripts/asr_recovery/s9_eval_c1.py`; outputs `run_c1/{results.json,predictions.csv,config.json}`; word lists `asr/C1/`; prompt properties `c1_prompt_check.json`. This is self-review work (the round-4 review was independent; this experiment answers its finding 3.7 and was run by the same writer, with the decision rule fixed in advance).

## Result in one paragraph
A fluent, domain-flavoured control prompt of the same length (C1) does NOT reproduce the P1 gain; it does the opposite. It makes Whisper clean up even more repetitions than the production prompt P0: survival of the human literal-repetition mark 105/214 (P0) -> 42/214 (C1), versus 144/214 (P1). [Revised in round 6; the original sentence claimed attribution to the disfluent content of the prompt.] In this exploratory DIHANA experiment, a fluent control of equal length and domain did not reproduce the gain observed with P1; a written, fluent prompt suppresses repetitions and P1 preserves them; which feature matters (oral vs written register, punctuation, domain overlap, the filler list already present in P0) is not proven. Decision rule of annex 3 section 5: **P1 > C1** (both tasks, supported after Holm). What this does not show: which part of P1 (repetitions, ellipses, fillers, corrections) carries the effect, and whether it transfers beyond this corpus/model.

## Numbers (frame: 346 positives + 679 sampled negatives, 122 speakers, weighted; ctx_LR = parent per-fold model trained on P0 text and NOT retrained)
| task | variant | survival of marks (95% CI by speaker) | w-F1 ctx_LR [95% CI] | w-F1 rule | FP on 679 sampled negatives (ctx_LR / rule) | WER vs reference A, negatives |
|---|---|---|---|---|---|---|
| rep_literal | P0 | 105/214 = 0.491 [0.422, 0.551] | 0.581 [0.511, 0.644] | 0.577 | 10 / 11 | 0.153 |
| rep_literal | C1 | 42/214 = 0.196 [0.134, 0.264] | 0.304 [0.231, 0.378] | 0.307 | 7 / 10 | 0.129 |
| rep_literal | P1 | 144/214 = 0.673 [0.614, 0.730] | 0.690 [0.619, 0.756] | 0.682 | 9 / 14 | 0.130 |
| revision | P0 | 53/161 = 0.329 [0.255, 0.405] | 0.313 [0.250, 0.381] | 0.152 | 38 / 6 (revision task negatives) | 0.153 |
| revision | C1 | 19/161 = 0.118 [0.071, 0.172] | 0.232 [0.165, 0.303] | 0.075 | 28 / 5 | 0.129 |
| revision | P1 | 58/161 = 0.360 [0.288, 0.436] | 0.352 [0.287, 0.419] | 0.133 | 36 / 6 | 0.130 |

Pre-declared family (Holm over 4; paired weighted F1 difference of ctx_LR on identical units, speaker bootstrap 5000, seed 13; p at the bootstrap floor is reported as "< 0.0004"):
| task | contrast | delta [95% CI] | p | p_Holm | verdict |
|---|---|---|---|---|---|
| rep_literal | P1 - C1 | +0.387 [+0.298, +0.478] | < 0.0004 | 0.0016 | supported after Holm |
| rep_literal | C1 - P0 | -0.277 [-0.359, -0.200] | < 0.0004 | 0.0016 | supported after Holm (C1 is worse than P0) |
| revision | P1 - C1 | +0.120 [+0.080, +0.164] | < 0.0004 | 0.0016 | supported after Holm |
| revision | C1 - P0 | -0.081 [-0.125, -0.038] | 0.0008 | 0.0016 | supported after Holm (C1 is worse than P0) |
Descriptive: rule, rep_literal, P1 - C1 +0.375 [+0.286, +0.465]; C1 - P0 -0.270 [-0.345, -0.199]. Paired survival, rep_literal: C1 vs P0 gained 0 / lost 63 (-29.4 pp [-35.8, -22.4]); P1 vs C1 gained 102 / lost 0 (+47.7 pp [+39.1, +56.0]). Revision: C1 vs P0 gained 0 / lost 34; P1 vs C1 gained 39 / lost 0.
"Share of the P1 gain reproduced by C1" is not meaningful (C1 moves the other way: -2.5 [-4.6, -1.5] for ctx_LR on rep_literal).

## Other readings
* **The WER improvement is not evidence of better recovery.** C1 lowers WER against the reference exactly as much as P1 (0.153 -> 0.129 / 0.130) while it destroys repetitions. The WER drop is a generic prompt effect (length, domain words, fewer loop hallucinations) and, with a reference that contains the repetitions and not the fillers, rewards cleaning. Adjacent identical word pairs on the 679 sampled negatives: P0 116 (7 turns; mostly loop hallucinations), C1 5 (5 turns), P1 17 (7 turns).
* The prompt acts as a style instruction in both directions: a fluent prompt suppresses repetitions, a disfluent one preserves them. This also confirms the round-4 caution that P1's text carries the style that is being measured; the P1 gain is real relative to P0 and to C1 on this frame, and "keep what is said" is a hypothesis, not a demonstrated mechanism.
* False positives: C1 has fewer ctx_LR FP (7 vs 10 vs 9) because it removes the repetitions that a detector would fire on, at the cost of recall (weighted recall 0.201 vs 0.509 vs 0.664). The rule's FP on sampled negatives: P0 11, C1 10, P1 14 (see EXPERIMENTO_ASR addendum for the P1 FP, which rose).
* Prompt echo (>= 4 consecutive prompt tokens): C1 output echoes the C1 prompt in 1 unit; P1 output echoes the P1 prompt in 2 units; empty outputs 2 / 3 / 3 (P0 / C1 / P1).

## Cost
C1 decode: 1025 units, summed per-unit time 52.8 min, per-unit RTF 0.69 (2 shards x OMP=4, run alongside a 4-process live-parity study), wall clock about 27 min (09:04Z - 09:31Z), within the 90 min budget.

## Limits
One control prompt (one draw of style/length/domain); P1 vs C1 differ in several ways at once (disfluent features, characters, specific words), so which feature of P1 matters is untested; same weighted frame, one corpus, one Whisper size/quantisation; ctx_LR thresholds tuned on P0 text; not a confirmatory result.

## Addendum 2026-10-31 (round 6 correction; SELF-REVIEW, pending an independent reviewer)
The statistical result (P1 > C1 by the pre-declared rule of annex 3) stands; its interpretation is narrowed. **Wording in force:** In this exploratory DIHANA experiment, a fluent control of equal length and domain did not reproduce the gain observed with P1; a written, fluent prompt suppresses repetitions and P1 preserves them; which feature matters (oral vs written register, punctuation, domain overlap, the filler list already present in P0) is not proven.

Qualifiers: (a) P0 is itself a prompt with a filler list (`"eh, este, o sea, mmm, bueno, tipo, ah, este..."`), so P1 vs P0 contrasts a repetition/self-correction prompt with a filler-list prompt, not "disfluent vs no prompt"; (b) there is no no-prompt control; (c) C1 has MORE domain overlap with the evaluated turns than P1 (42 vs 2 frame turns sharing a 3-gram), so it is a closer-domain control, not a weaker one; (d) one control prompt and one length only; C1 was written after the P1 result was known; (e) not a confirmatory result.
Annex 3 rule 5 says "the gain is attributable to the verbatim content of the prompt beyond a generic prompt"; that rule text is frozen (hashed) and is not edited, but a P1 > C1 outcome is read here only as the wording above.
