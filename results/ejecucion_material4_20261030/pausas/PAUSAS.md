# block / rhetorical_pause / neutral_pause: technical sensitivity (NOT validity, NOT F1)

Date 2026-10-31. Script `scripts/improve/pause_sensitivity.py` (output `pause_sensitivity.json`, log `../pausas_run.log`). These three classes have NO human reference anywhere in the available material (DIHANA `p` = silent pause without duration or function; ESLORA pause tags have no audio/times; no annotated blocks). Nothing below says whether an emission is correct. It only says what the current engine's emission depends on.

## Design
Units: 240 DIHANA user turns of folds 1-3 only (dev; fold 4 and fold 5 untouched), seeded sample (seed 20261030) stratified by duration, 60 per bin (<1.5, 1.5-3, 3-6, >=6 s). ASR words from the frozen cache and never changed. Checkpoint `models/ckpt_audio_text.pt` (V0), default recipe. One variable per condition, paired on the same units:

| condition | what changes |
|---|---|
| zero | V0 behaviour: short turns zero-padded to one 3 s window, long turns unpadded |
| noise | the same padding made of noise at the unit's own noise floor (existing option `pad_mode="noise"`) |
| zero+1.5 | 1.5 s MORE digital zeros appended (any duration) |
| noise+1.5 | 1.5 s more noise-floor padding |
| noctx | linguistic context off (`context=False`), zero padding |

"Emitted" = the argmax of at least one window of the unit is the class. Counts are units emitting / 240 (Clopper-Pearson intervals are in the JSON).

## Results (all 240 units)
| class | zero | noise | zero+1.5 | noise+1.5 | noctx |
|---|---|---|---|---|---|
| block | 50 | 53 | 65 | 65 | 57 |
| rhetorical_pause | 100 | 42 | 179 | 74 | 93 |
| neutral_pause | 91 | 124 | 185 | 226 | 98 |

Paired changes against `zero` (units gained / lost): rhetorical_pause noise 0/58, zero+1.5 79/0, noise+1.5 30/56, noctx 15/22; neutral_pause noise 44/11, zero+1.5 95/1, noise+1.5 135/0, noctx 26/19; block noise 6/3, zero+1.5 15/0, noise+1.5 18/3, noctx 20/13.

By duration (units emitting of 60; zero -> noise -> zero+1.5 -> noise+1.5 -> noctx):
* < 1.5 s: rhetorical 44 -> 6 -> 54 -> 5 -> 49; neutral 13 -> 31 -> 49 -> 60 -> 11; block 2 -> 4 -> 2 -> 4 -> 0.
* 1.5-3 s: rhetorical 30 -> 10 -> 48 -> 20 -> 27; neutral 15 -> 30 -> 33 -> 57 -> 21; block 3 -> 4 -> 7 -> 6 -> 1.
* 3-6 s (no padding in `zero`): rhetorical 9 -> 9 -> 36 -> 20 -> 5; neutral 20 -> 20 -> 47 -> 52 -> 20; block 14 -> 14 -> 22 -> 21 -> 20.
* >= 6 s: rhetorical 17 -> 17 -> 41 -> 29 -> 12; neutral 43 -> 43 -> 56 -> 57 -> 46; block 31 -> 31 -> 34 -> 34 -> 36.

Windows of the V0 run that emit a class and reach past the end of the real audio (i.e. contain padding): rhetorical_pause 74 of 120 (62%), neutral_pause 28 of 161, block 5 of 85.

## What it shows
1. **Tail content, not speech, drives much of the emission.** Appending 1.5 s of silence to turns that were not padded at all (>= 3 s) raises rhetorical_pause from 26/120 to 77/120 units in the two longest bins and neutral_pause from 63/120 to 103/120 (zero+1.5). The same turns, same words, same speech.
2. **The padding TYPE decides between the two pause classes in short turns.** Zero -> noise-floor padding moves 38/60 (< 1.5 s) rhetorical emissions away and adds 18 neutral ones. The "rhetorical" label on a short padded turn is therefore a padding artefact of the recipe, not a property of the speech.
3. **Duration is confounded with all of it:** block grows with turn length (2/60 -> 31/60) independently of padding (the `noise` condition changes it by at most 4 units per bin), so block emission is mostly a duration effect plus a small padding effect.
4. **Context (linguistic) matters less but not zero:** turning it off changes the emission of 33 (block), 37 (rhetorical) and 45 (neutral) of 240 units, in both directions (net +7 block, -7 rhetorical, +7 neutral), i.e. it reshuffles which units emit more than it adds or removes emissions.

## Defects, decision
No implementation bug was demonstrated. Appending silence makes a window contain a pause; the model reading it as a pause is the intended function of the head, and the recipe (zero padding of short turns, V0) is a documented behaviour kept for comparison (`mrcd/windowing.py`). The honest conclusion is a property of the evaluation recipe: pause-class emission on turns shorter than a window (62% of DIHANA user turns) mixes speech and padding, and the same holds for any session that ends in silence. **Nothing in `mrcd/` was changed and the historical default is intact.** Two options already exist for anyone who needs a less pad-sensitive reading: `pad_mode="noise"` and `abstain_below_s` in `mrcd.windowing` (both opt-in, not wired as defaults). Whether noise padding is "better" cannot be decided without a reference: it moves emissions instead of removing them (rhetorical -58, neutral +33 net).

Limits: 240 units of one corpus (telephone, train-information domain), one checkpoint, argmax emission at the engine's grid (no thresholds), one pad length (1.5 s), words fixed by the cache so ASR effects are not included; it does not say any emission is right or wrong.
