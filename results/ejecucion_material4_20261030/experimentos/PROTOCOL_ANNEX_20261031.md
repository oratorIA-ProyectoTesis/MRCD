# Annex to PROTOCOL.md (dated 2026-10-31; PROTOCOL.md itself is unchanged, sha256 in PROTOCOL.sha256.json)

Entries are written after the main run (`run_main/`) and record every difference between plan and execution.

1. **Vc on folds 1-3 reported as an in-sample row.** The protocol said comparisons against Vc on folds 1-3 would not be computed. The tables show one descriptive row "Vc (all units; IN-SAMPLE for folds 1-3)" so that its raw behaviour is visible; it is labelled in-sample and enters no paired comparison. All paired Vc comparisons use fold-4 units only (Vc-favourable).
2. **Post-hoc diagnostics (NOT part of the Holm family, NOT used to choose anything).** Added after seeing the main results because the revision gain looked suspiciously tied to turn length: (a) `posthoc_diagnostics.py`: ctx_LR restricted to length / punctuation / filler-count features under the same nested protocol, and in-sample standardised coefficients; (b) `posthoc_error_causes.py`: text-pattern categories of the rule's false positives and V0 / rule / ctx_LR recall on positives visible vs destroyed in the ASR. Both are descriptive and cannot change any verdict.
3. **Block / rhetorical_pause / neutral_pause.** No implementation defect was corrected and no new sensitivity was measured: any fix would need edits to `mrcd/windowing.py` or `mrcd/engine.py`, outside the write scope of this task. The earlier measurements (`mejoras_20261030/properties/properties_noref.json`, summarised in `MEJORAS.md` section 3) are reused and cited.
4. **utt74 window grid.** The unit is 3.339 s (>= 3 s), so no zero padding was applied and two windows (0.00 and 0.34 s) were scored; the recipe is the V0 recipe of `scripts/baselines/run_mrcd.py` re-implemented in `scripts/rep_rev/utt74_check.py`.
5. **Pool construction is slow (about 80 s)** and cached in `_cache/pool.pkl` (derived data, ignored by git with the rest of `results/`).
6. No other deviation. In particular: no feature, grid value, threshold rule, model, comparison or gate was changed after the main run; the main run was executed once.

7. **Update 2026-10-31: item 3 is superseded.** The block / pause sensitivity was measured afterwards (`../pausas/PAUSAS.md`, DIHANA folds 1-3, 240 units). No defect was demonstrated and `mrcd/` was not changed.
