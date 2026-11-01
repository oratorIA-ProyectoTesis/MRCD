# INFORME FINAL: material4 integration, corrections after round-3 review (2026-10-31)

Review status, stated plainly: the round-3 review (`revision3/REVISION3.md`) was INDEPENDENT (a different agent). The corrections in this task (live_cfg_v3, dashboard, importer guard, tests, addenda, pause sensitivity) were made by the same writer who produced the work under review: they are SELF-REVIEWED. No new independent review was run on them. Everything here is technical evidence; nothing closes OE4 and nothing is confirmatory.

## (a) What was implemented / integrated
* Corpus importers `mrcd/corpora/` (DIME, ESLORA, DIHANA): coverage semantics (absence = unknown), hashed sources, bounds, leak checks, group-crossing checks. Hardened this round: canonical forbidden-mapping guard that also validates Example-level fields and any XLS kind (commit 9e448a7).
* Contextual repetition/revision experiment (`mrcd/context_eval/`, `scripts/rep_rev/`): speaker-grouped nested CV, Holm, paired bootstrap. EXPLORATORY. Round-3 qualifications added as an addendum; quantile-bin calibration added (fdbb871).
* Live path: `SessionFeatureTracker.finalize`, `LiveScorer`, `live_cfg_v1/v2`; NEW `live_cfg_v3` (exact Viterbi resume, `mrcd/live/pyin_carry.py`, 18288ca); dashboard de-duplication of final/revises rows (6aa53b8). The default config stays `live_cfg_v2`.
* Block / pause sensitivity measurement (cbc6ec2). No engine behaviour changed.

## (b) Data and labels really usable
| resource | usable for | not usable for |
|---|---|---|
| DIHANA folds 1-4 (human XML marks) | text-level reference for filler, lengthening mark, repetition (approximate), reformulation (approximate); dev/tune/nested CV. Fold 5 is historical and untouched here | block, pauses (no reference); anything as an untouched test |
| EPIC, AMERESCO | earlier filler / prolongation comparisons (inspected, not unseen) | repetition, revision |
| DIME 20 dialogues + utt1/utt74 | import/format verification, throughput, ASR smoke test, silver diagnostics | training or gold evaluation (no class labels; XLS are dialogue acts, not disfluencies) |
| ESLORA public text | text tag statistics, parser tests | audio classes (no audio, no times) |
| PerLA, PUCP, SOPHIAS, DIME repair-label directory | nothing retrieved / not granted | everything |

Absence of a label is never a negative. No new human annotation was made.

## (c) Improvements and deteriorations against each comparator
DIHANA folds 1-4 pool, nested CV, 180 speakers, F1, EXPLORATORY; full tables in `experimentos/TABLES.md`.

| task (support) | cond | rule | V0 | Vc | ctx_LR | paired ctx_LR - rule [95% CI], Holm p |
|---|---|---|---|---|---|---|
| rep_literal (214 pos / 84 spk) | A (circular) | 0.923 | n/a | n/a | 0.970 | +0.046 [0.016, 0.082], .0072 |
| rep_literal | B (Whisper-small text) | 0.541 | 0.487 | 0.500 (fold 4 only, favourable) | 0.537 | -0.004 [-0.036, 0.028], .81: no difference |
| revision (161 pos / 91 spk) | A | 0.208 | n/a | n/a | 0.345 | +0.137 [0.048, 0.223], .0072 |
| revision | B | 0.146 | 0.116 | n/a | 0.286 | +0.140 [0.073, 0.207], .0016 |

Reading, with the round-3 limits: repetition has NO improvement over a one-line rule on ASR text (the rule also beats V0 by +0.054 [0.011, 0.097], descriptive). The revision gain is a screening gain: a length-only model reaches F1 0.232 vs 0.286 for ctx_LR (length-only beats the rule by +0.087 [0.015, 0.155]; the contextual excess +0.053 [0.017, 0.091] is post hoc and unadjusted). Condition A depends on the comma convention (comma-separated copies marked 0/19). "ASR destroys 51/63/67%" is a property of this Whisper-small transcript (102 of 109 are collapsed repetitions). Calibration: gain not demonstrated (equal-width and quantile bins; all CIs contain 0).

## (d) Exploratory
Everything in (c); the pause sensitivity; the utt74 route check (n = 1: proves the path runs, nothing about detection); the live study on synthetic stress sessions. Confirmed: nothing.

## (e) The seven classes
| class | status | basis |
|---|---|---|
| filler_word | reference compatible; exploratory | DIHANA `f` (turn level), EPIC |
| prolongation | partial approximation; exploratory | DIHANA `a` lengthening mark, AMERESCO; ESLORA text tags without audio |
| repetition | reference compatible for literal repeats, partial approximation for the full mark; exploratory | DIHANA sub-type A (78% of marks); no gain over a rule on ASR text |
| revision | partial approximation (58% silent edits); exploratory; utt74 = 1 exact example | grouped DIHANA mark |
| block | no reference | no annotated blocks; emission follows turn duration (2/60 to 31/60 units by duration bin) |
| rhetorical_pause | no reference | emission is largely a padding artefact on short turns (`pausas/PAUSAS.md`) |
| neutral_pause | no reference | same; absence of a mark is not "neutral" |

Per-resource coverage matrix: `audit/MATRIZ_ETIQUETAS_COBERTURA.md`. Classes without reference are never averaged as F1 = 0.

### Live parity (correction of the earlier "exact" claim)
12 new 10.5 min sessions (seeds 101-112 fixed before running) x 3 partitions, exact frame criterion, Clopper-Pearson 95%: live_cfg_v2 diverges in 7/12 sessions (58%, [27.7%, 84.8%]; 1-19 frames of ~63 000, up to 24 st); live_cfg_v3 in 1/12 (8%, [0.2%, 38.5%]; 1 frame, 5.5 st). Root cause: restart of pYIN's Viterbi from a uniform prior at the slice start (left dependence inside noise stretches, margin-independent); v3 resumes from the exact forward scores. Residual of v3: first frames of a long digital silence (3 frames in seed 23, 1 in seed 111), unaffected by R = 6, 8, 12. Details: addendum of `paridad/PARIDAD_LIVE_V2.md`. Synthetic stress sessions, Spanish only, no real microphone.

## (f) Tests, literal
* `OMP_NUM_THREADS=4 pytest -q -rs`: **525 passed, 3 skipped** (214.63 s; `logs/pytest_final.log`). Skipped, literal: `tests\test_engine_terminal.py:268: set MRCD_SLOW=1: runs the 32 pilot clips (~2 min)`; `tests\test_live_stream_partitions.py:447: set MRCD_SLOW=1: ~10.5 min session (~6 min CPU)`; `tests\test_live_stream_partitions.py:541: set MRCD_SLOW=1: ~30 s`.
* `MRCD_SLOW=1 OMP_NUM_THREADS=4 pytest -q -rs`: **528 passed, 0 skipped** (777.58 s; `logs/pytest_final_slow.log`).
* New tests: dashboard upsert (runs the page JS with node), 18 forbidden-label variants at annotation and Example level, XLS kinds, survival 0.8 threshold (mutant verified caught), last partial Silero chunk padding (mutant verified caught on a real-clip cut), R=4 pin, `pyin_resumable` equality, v3 tracker partitions, v3 scorer final rows, seed-23 reproduction (v2 diverges, v3 exact; MRCD_SLOW).
* Not run: real microphone, browser rendering, Speechmatics, re-hash of `material/` against the inventory, v3 on the 32-clip pilot table, a new independent review.

## (g) Pending and access dependencies
DIME repair-label directory (HTTP 403, owners IIMAS-UNAM); ESLORA audio and labelled data (request to the owners, not sent; licences conflict); PerLA (auth wall) and PUCP (challenge): nothing retrieved; SOPHIAS (DUA signature and GHIA authorisation, TASKS T03/T04/W01); DIME licence terms. Technical: v3 is not the default (decision needed); events from live rows (merge) not built; Vc not wired in the live path; low-severity importer items (non-string scan, ZIP size cap); independent review of this task's corrections.

## (h) Reproduction (repo root, `.venv`)
```
OMP_NUM_THREADS=4 python -m pytest -q -rs
MRCD_SLOW=1 OMP_NUM_THREADS=4 python -m pytest -q -rs
python scripts/live_parity/divergence_rate_v3.py --out paridad/divergence_rate_fresh.jsonl --workers 18   # about 1 h
python scripts/live_parity/divergence_rate_v3.py --known --out paridad/divergence_rate_known.jsonl --workers 6
python scripts/live_parity/right_margin_v3.py 111 8
python scripts/rep_rev/calibration_quantile.py
python scripts/improve/pause_sensitivity.py 60
python scripts/live_demo.py --live-config live_cfg_v3
```
Earlier experiment/importer commands: `experimentos/EXPERIMENTOS.md` section 9 and `importadores/IMPORTADORES.md`.

## (i) Commits (80a921a..HEAD; nothing pushed)
This task: 18288ca live_cfg_v3 pYIN resume; 0fb2f06 divergence study script (seeds fixed); 6aa53b8 dashboard de-dup; 9e448a7 forbidden-mapping guard; b1204b2 surviving-mutation and v3/seed-23 tests; fdbb871 EXPERIMENTOS addendum + quantile calibration; cbc6ec2 pause sensitivity; cb052a8 PARIDAD addendum; then the commit of this report (see `git log`). Earlier in the range: 5eb616f, 2f57039, 3db0d70, 069c315, 73c2047, 9ed9c9f, cccdf80, 77b312c, 85be0e8, 0f1fdb8, 4013b5f, de2d8ce, e570ab4, 0034d7d, 434c4e0.

## Limits
One reference corpus (telephone, peninsular Spanish, train domain), unknown inter-annotator agreement, synthetic live sessions, protocol-freeze order evidenced by file mtimes only, `results/` is git-ignored (only force-added reports are versioned). OE4 is NOT complete.

---

## Addendum 2026-10-31 (round 5, SELF-REVIEW of the round-4 findings; original text kept)
* Tests now: `OMP_NUM_THREADS=4 pytest -q -rs` 653 passed, 3 skipped; `MRCD_SLOW=1 ...` 656 passed, 0 skipped (replaces 525/528 above; round-4 reviewer saw 555+3 / 558).
* Withdrawn: "residual of v3 inherent / unaffected by R" (see PARIDAD addendum); "no FP cost" (only ctx_LR; rule 11 -> 14, total 23 -> 34); "28%" now 0.28 [0.16, 0.39]; p "< 0.0004"; "fold 5 never loaded" -> "never indexed or used"; "survival 0.8 threshold caught" now pinned for both branches; live parity "not exact".
* Fixed: pYIN monkeypatch removed; positive allow-list guard; default-suite tests catch removal of `viterbi_carry`, `ALLOWED_FOLDS` with 5, survival threshold changes (13/13 mutants detected on a copy).
* Added EXPLORATORY experiment: ASR prompt recovery with control prompt C1 (`results/asr_recovery_20261031/C1_CONTROL.md`): P1 > C1 on this frame; wording and qualifiers in the round-6 addendum below (the attribution to "disfluent content" is withdrawn). Full list: `results/revision5_20261031/REVISION5.md`.

---

## Addendum 2026-10-31 (round 6 corrections; SELF-REVIEW by the same writer, NOT an independent review; no independent reviewer has checked them yet)
Responds to `results/revision6_20261031/REVISION6.md`. Code commits: 35caddd (corpora guard, tests), ebb4d10 (pyin_resumable).

* **C1/P1 claim.** Wording in force: In this exploratory DIHANA experiment, a fluent control of equal length and domain did not reproduce the gain observed with P1; a written, fluent prompt suppresses repetitions and P1 preserves them; which feature matters (oral vs written register, punctuation, domain overlap, the filler list already present in P0) is not proven.
  Qualifiers: (a) P0 is itself a prompt with a filler list (`"eh, este, o sea, mmm, bueno, tipo, ah, este..."`), so P1 vs P0 contrasts a repetition/self-correction prompt with a filler-list prompt, not "disfluent vs no prompt"; (b) there is no no-prompt control; (c) C1 has MORE domain overlap with the evaluated turns than P1 (42 vs 2 frame turns sharing a 3-gram), so it is a closer-domain control, not a weaker one; (d) one control prompt and one length only; C1 was written after the P1 result was known; (e) not a confirmatory result.
* **Allow-list behaviour (decided: fail loud).** A (source_kind, source_label) pair that is NOT in `ALLOWED_MAPPINGS` and asks for a class makes `Annotation(...)` RAISE `ForbiddenMappingError`; it is NOT imported as "unmapped" and nothing counts such cases. The earlier phrase "everything else is imported unmapped" (REVISION5) is withdrawn. A new legitimate label stops the import until the importer emits it with `proposed_class=None` or the list is extended (audit decision). Mapped pairs whose strings only match the list through homoglyph, full-width, NFD or invisible characters are also rejected (`is_plain_label`), so the stored `source_label` is always the canonical string.
* **DIHANA test gap closed.** `r:borrado` first appears at XML 243 of 900 and the old parity test read only 60 files. A synthetic fixture with every mapped tag now runs in the default suite; removing `rborrado` from the list makes it fail (verified on a temp copy).
* **Mutants (REVISION6 N1-N4, N9, N10, N12).** New tests pin: repetition n-gram sizes 1-4 (and that 5 is not counted), fragment prefix <= 3 characters, revision correction coverage 0.8, the smoothed bootstrap p (2/(B+1) for a perfect gain), `fold_label` stripping invisible characters, the `silenc` stem. Each mutant was re-applied on a temp copy outside the repo (log: `results/revision6_20261031/logs_fix/mutations7.json`); all 12 mutants in that log were detected by the new tests (N1, N1b, N2, N2b, N3, N3b, N4, N9, N10, N12, plus N13 `is_plain_label` disabled and N14 non-mono check removed). This covers only these mutants; it says nothing about other mutants (REVISION6 N5-N8 and N11 were already detected).
* **`pyin_resumable`:** non-1-D input now raises `ValueError` ("mono audio only") instead of a numba TypingError; a default-suite test checks bit-equality with `librosa.pyin` on one noisy signal.
* **Live dose-response (limits).** Rests on 2 synthetic sessions (seeds 307 and 311); the intermediate R values were chosen post hoc after seeing the failures; seed 307 has no point between R = 12 and 30. "0/12 at R = 30" holds BY CONSTRUCTION of the generator (it draws silences of 0.5-30 s, so R = 30 covers all of them) and says nothing about real recordings with longer degenerate stretches. "The residual disappears when R exceeds the silence" is a hypothesis for the synthetic generator.
* **Test-suite counts (two trees).** Author's local tree: 656 passed, 0 skipped (`MRCD_SLOW=1`); default 653 passed + 3 skipped. Clean checkout (REVISION6): 639 passed + 17 skipped (default) and 641 passed + 15 skipped (`MRCD_SLOW=1`), because the extra skips need git-ignored data (DIHANA `turns.csv`, AMERESCO clips/ASR cache, checkpoint-linked clips). "656 passed, 0 skipped" is therefore true only where those files exist. Counts after this round (more tests) are in the logs of `results/revision6_20261031/logs_fix/`.
* **Backups.** The `.bak_rev5` files cited in REVISION5 did not exist. Real copies were created from git (`16bc252`, the state before the round-5 edits): `docs_actualizados/TASKS.md.bak_rev5`, `docs_actualizados/MATRIZ_OE1_OE4.md.bak_rev5`, `INFORME_FINAL.md.bak_rev5`. They are reconstructions, not the original transient backups.
* **PREREGISTRO.md** is not frozen or current (see its header note).
* **Suites after round 6 (author's local tree, literal logs in `results/revision6_20261031/logs_fix/`):** `OMP_NUM_THREADS=4 pytest -q -rs` 698 passed, 9 skipped (191.9 s); `MRCD_SLOW=1` 707 passed, 0 skipped (528.6 s). 51 tests were added; counts in a clean checkout were not re-measured this round.

## Addendum 2026-10-31 (round 7, SELF-REVIEW by the author after the independent REVISION7; these fixes have NOT been re-reviewed independently)
* **Correction of the previous addendum:** the claim "the stored `source_label` is always the canonical string" was FALSE for round 6 (REVISION7 F-4: 742 of 2,943 fuzz variants were accepted, e.g. U+034F, U+FE00-FE0F, U+1680, U+2028/2029, U+2800, plus 49 ASCII variants such as `FA`, `f.a`, `<f:a>`). It is true only from this round: `is_plain_label` now rejects every C*, M*, Zs/Zl/Zp (except ASCII space), Sk and So character, and `Annotation`/`Example` additionally require the exact canonical spelling (`CANONICAL_SPELLINGS`). Real importer counts are unchanged: 44 keys, 178,272 annotations, 51,448 example counters (identical JSON, `results/revision7_20261031/logs_fix/import_counts_fix.json` vs the reviewer's `logs/import_counts_4443d71.json`; counting only, DIHANA fold 5 was imported for counting, no decision or training used it).
* **"P1 echoed its prompt in 2 of 1025" is withdrawn:** the detector flagged 2 units: 1 loop (`V118_VA2c1_U_005`) and 1 false positive (`B501_BA7a2_U_007`). The detector is a heuristic (limits and changes in `results/asr_recovery_20261031/PERFIL_P1.md`, addendum R7).
* **"Bit-identical" qualifier:** holds only on an unloaded machine with `OMP_NUM_THREADS=4` exported before numpy is imported. Two concurrent slow suites produced 6 failures each (REVISION7 F-3); alone they pass. Slow decode tests now skip under > 50 % CPU load (`MRCD_SKIP_LOAD_CHECK=1` forces them). Never run a slow suite concurrently with another job.
* **Surviving mutants of REVISION7 F-9 pinned:** (a) Example-level canonical-label check removed, (b) `chunk_s` dropped from the fingerprint. Each is killed by a new test, verified on a temp copy outside the repo (`results/revision7_20261031/logs_fix/mutations_r7fix.json`). This says nothing about other mutants.
* **Metadata:** `asr_cache.key` is a deprecated alias of `asr_fingerprint.profile_fingerprint` (the engine has no cache). P1 events and live rows carry `asr_profile` / `asr_experimental`; an ASR without `.profile` is `unknown`, not p0; spoofed `ASRProfile` instances are rejected; `model_info` failures are logged.
* **Test counts and who measured them (all with `OMP_NUM_THREADS=4`):**
  * Author, local tree, after round 7 (logs in `results/revision7_20261031/logs_fix/`): default 741 passed + 9 skipped (195.8 s, `pytest_default.log`); `MRCD_SLOW=1`, run alone with no concurrent load, 750 passed + 0 skipped (518.9 s, `pytest_slow.log`). 43 tests were added since round 6.
  * Author, local tree, round 6 (superseded by the line above): 698 passed + 9 skipped default, 707 passed + 0 skipped with `MRCD_SLOW=1`; the independent reviewer re-measured exactly these two numbers on the same tree (REVISION7 C-8).
  * Independent reviewer, clean worktree at 4443d71 (no git-ignored data), default: 681 passed + 26 skipped (REVISION7 C-8). A clean tree was NOT re-measured by the author after round 7. The earlier 639 + 17 / 641 + 15 clean-tree figures (REVISION6) are historical.
* **Do not cite the `.bak_rev5` files.** `INFORME_FINAL.md.bak_rev5`, `docs_actualizados/TASKS.md.bak_rev5` and `docs_actualizados/MATRIZ_OE1_OE4.md.bak_rev5` are reconstructions from `16bc252` and contain the withdrawn wording ("la mejora se atribuye al contenido disfluente del prompt"). Their content was deliberately not edited. Annex 3 rule 5 (frozen) also still reads "attributable to the verbatim content"; it is quarantined by the C1 qualifiers above.
* **EXPERIMENTO_ASR.md headline:** the WER improvement (0.153 -> 0.130) is equal under the fluent control C1 (0.129) and is not evidence of recovery; ctx_LR 0.690 is a DIHANA text-model score, not an engine result.
