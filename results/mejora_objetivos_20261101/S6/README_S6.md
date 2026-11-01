# S6: video evaluable on EPIC v2 (2026-11-01)

**Self-reviewed by the author; an independent adversarial review was run on 2026-11-01 (`results/mejora_objetivos_20261101/REVISION_S/fix_logs/`), findings applied.** EXPLORATORY: all 39 EPIC speakers were seen in earlier development. Defaults unchanged (V0, P0, `live_cfg_v2`); nothing was wired into the service or the app.

## Conclusion (explicit)

**No concluyente: the video channel cannot be evaluated on EPIC v2 with an adequate sample, so no contribution is measured and none is excluded.** The measured reason is coverage, not classifier performance:

- Under the frozen gate (protocol + amendment A1) only **69 of 1533 units (4.5 %)** are admitted, from **2 of 74 videos / 2 speakers**, and **0 of the 173 FPAUSE positives**. A post-hoc corrected gate (A2, merged face tracks) admits 66 units, again 0 positives. The paired ablation on admitted units has no positives and was **not computable**.
- Independently of synchrony, a face is detected in only **31.7 % of all frames** (362,103 frames); only 14 positives (8 speakers) under A1 and 19 (10 speakers) under A2 have face coverage >= 0.8. The adequacy rule (>= 40 positives, >= 10 speakers) cannot be met on EPIC with this tooling.
- Synchrony is also not verifiable for most videos: a video-level offset could be estimated for 29/74 videos, and only 2 pass (|offset| <= 120 ms, CI width <= 200 ms). Median unit lip-audio peak r is 0.25 (units with face coverage >= 0.8), the offset estimates pile up at the +-400 ms search limit, so with 384x288 parliamentary footage and MediaPipe lip aperture the lip-energy coupling is weak or the footage is cut/wide. This is a statement about this corpus + tool, not about video in general.

## What was run (commands, serial; S1 had finished: `S1/README_S1.md` existed at 01:30 before any heavy step; the only earlier use was a 5 s single-video smoke test while S1 ran, no measurable overlap)

```
python scripts/s6_video/s6_01_units.py                      # 1533 units, 173 FPAUSE, 74 videos, 39 speakers
python scripts/s6_video/s6_02_faces.py                      # 74 videos, 362,103 frames, 4048 s (11.2 ms/frame)
python scripts/s6_video/s6_03_gate.py --tag main            # frozen gate (A1)
python scripts/s6_video/s6_03_gate.py --tag a2 --merge      # amendment A2
python scripts/s6_video/s6_03_gate.py --tag sens_r0.2 --unit-min-r 0.2   # sensitivity (coverage only)
python scripts/s6_video/s6_03_gate.py --tag sens_r0.4 --unit-min-r 0.4
python scripts/s6_video/s6_04_features.py --tag a2 --subset face_only    # exploratory dataset (A3)
python scripts/s6_video/s6_05_ablation.py --dataset a2_face_only
```

Protocol, amendments and hashes: `PROTOCOL.md`, `PROTOCOL.sha256` (frozen and committed before the gate ran; A1 before the full gate run; A2 and A3 are **post-hoc and disclosed**, written after seeing the A1 result). Versions and input hashes: `versions.json` (python 3.12.10, numpy 2.5.3, scikit-learn 1.9.1, mediapipe 1.0.1, opencv 5.0.0, faster-whisper 1.2.1; face landmarker sha256 `64184e22...c9ff`; tracks manifest sha256 `d33cc91a...cdd`). Code: `mrcd/av/{config,gate,faces,features,ablation,evidence}.py`, `scripts/s6_video/`, tests `tests/test_av_*.py`.

## Gate coverage (admitted / total; counts only)

| Run | Admitted | Positives admitted | Speakers | Videos with sync OK |
|---|---|---|---|---|
| A1 frozen (r >= 0.30) | 69 / 1533 | 0 / 173 | 2 | 2 / 74 |
| A1 sensitivity r >= 0.20 | 81 | 0 | 2 | 2 |
| A1 sensitivity r >= 0.40 | 44 | 0 | 2 | 2 |
| A2 merged tracks (post-hoc) | 66 | 0 | 2 | 2 |

Primary exclusion reason (A1; each unit counted once, first failing criterion): face not visible enough 1150; video sync unverified 189 (45 videos with too few units, 27 with out-of-range or imprecise offset); unit < 1.5 s 93; lip-audio r below threshold 27; not above surrogate null 4; second face 1; admitted 69. Per video and per speaker: `coverage_by_video_*.csv`, `coverage_by_speaker_*.csv`. Synthetic tests cover known offset (sign and value), desync exclusion, no face (reported as unavailable, never 0), two faces, unrelated audio, short units, identity swap, fragmented tracks.

## Exploratory pipeline check (A3; NOT gate-admitted, synchrony NOT verified; cannot support any verdict)

Units: A2 units passing C0-C3 only (face visible, one face, consistent identity), 301 eligible; negatives capped at 3x positives by sha1(unit_id) -> **76 units, 19 positives, 10 positive speakers, 24 speakers**; ASR P0 on CPU only for those. Speaker-grouped 5-fold x 5 repeats, balanced logistic regression, threshold 0.5. Counts are means over repeats; CI = 95 % speaker bootstrap (B = 2000). Chance F1 (all positive) = 0.40.

| Model | TP | FP | FN | P | R | F1 [CI] | AP | AUC |
|---|---|---|---|---|---|---|---|---|
| A+T (reference) | 17.0 | 10.0 | 2.0 | 0.64 | 0.89 | 0.74 [0.46, 0.91] | 0.71 | 0.88 |
| V alone (control) | 10.6 | 38.0 | 8.4 | 0.22 | 0.56 | 0.31 [0.13, 0.52] | 0.22 | 0.41 |
| A+T+V concat (F1) | 14.4 | 14.2 | 4.6 | 0.51 | 0.76 | 0.61 [0.34, 0.81] | 0.64 | 0.83 |
| A+T+V gated (F2) | 17.0 | 9.8 | 2.0 | 0.64 | 0.89 | 0.75 [0.46, 0.91] | 0.72 | 0.89 |
| A+T+V residualised (F3) | 15.2 | 13.0 | 3.8 | 0.55 | 0.80 | 0.65 [0.37, 0.84] | 0.63 | 0.83 |

Paired difference vs A+T on the same 76 units (Holm over 3):

| Variant | dF1 [95 % CI] | p (boot) | p (Holm) | dAP |
|---|---|---|---|---|
| F1 concat | -0.134 [-0.222, +0.002] | 0.060 | 0.177 | -0.076 |
| F2 gated | +0.004 [0.000, +0.013] | 0.712 | 0.712 | +0.008 |
| F3 resid | -0.095 [-0.167, +0.003] | 0.059 | 0.177 | -0.088 |

Without the two dominant speakers (68 units, 15 positives, 8 positive speakers): A+T F1 0.63; dF1 F1 -0.092 [-0.168, -0.009] (Holm 0.078), F2 -0.004 (Holm 0.75), F3 -0.056 [-0.187, +0.021] (Holm 0.35). V alone has AUC 0.41 (0.46 without them): no signal. F2 is numerically identical to A+T because its gate confidence (local r) is mostly low, so it mutes V; its "+0.004" is not a gain. Adding V by concatenation or residualisation lowered F1 (overfitting with 76 rows), a negative result recorded as such. By the frozen rule (adequacy fails: 19 < 40 positives; 10 positive speakers meets the speaker minimum) every variant is **no concluyente** (`ablation_results_a2_face_only.json`).

## Identity leakage probe (visual features, 265 eligible units, 13 speakers with >= 8 units)

Linear speaker-identification balanced accuracy: raw 0.62, speaker-normalised 0.43, normalised with permuted speaker labels 0.09, chance 0.077. Per-speaker normalisation removes part but **not all** identity information (still ~5.6x chance); the units of one speaker also share sessions, so some of it is session/camera, not only the person. This is why the V block would need further control before any claim, and why the identity-free requirement is only partly met.

## Integration (task 5)

`mrcd/av/evidence.py` provides the opt-in experimental `av_gated` evidence (gated observable features, or `available: False` with a reason such as `face_not_visible_enough` or `video_file_not_found`, never zeros; always marked experimental; tested). It is **not wired into `mrcd.service`** (S1's path) and no default changed. Given a 4.5 % admission rate and no demonstrated gain, wiring is not recommended now.

## Limits

EPIC is parliamentary speech (2004, 21 sessions, L1 Spanish, 384x288/352x288, wide shots and cutaways); all 39 speakers were seen in development, so no speaker-independent test exists; sentence times are automatic WebMAUS and the label is filled-pause presence per sentence (negatives are negatives for FPAUSE only); thresholds were set a priori and not calibrated; A2 and A3 are post-hoc; the 3-unit minimum for a video offset gives crude CIs; text features come from our own ASR (P0 loses some filled pauses); a visible face can be another person in a parliament shot; licence CC BY-NC 4.0 (local use only: no video, frames, per-frame landmarks or EPIC text are committed; only counts and aggregate metrics are).

## What an independent test would need

A corpus with a frontal, close-up, continuous, high-resolution recording of the speaker (>= 25 fps, face >= ~150 px), unseen speakers (>= 10 with positives and >= 40 positives), human-verified filled-pause annotations with word-level times, and a pre-registered gate and analysis run once. CHM150 or DIME-style labelled video with a webcam would fit; EPIC and the DIME screen-recording AVIs (no face) do not.

## Files

`PROTOCOL.md`, `PROTOCOL.sha256`, `versions.json`, `gate_summary_{main,a2,sens_r0.2,sens_r0.4}.json`, `coverage_by_{video,speaker}_*.csv`, `dataset_summary_a2_face_only.json`, `ablation_results_a2_face_only.json`, `pytest_full.log`. Local only (gitignored `work/`): units, tracks (per-frame scalars), ASR word caches, datasets.

## Addendum 2026-11-01 (independent review S; nothing above was edited)
* **The A3 verdict rule changed AFTER the A3 run.** Commit `c18672a` (2026-11-01 03:00) made the rule stricter: an inadequate sample never yields a verdict (and the identity probe now covers all eligible units). A3 was run before this commit. The direction is conservative (it can only turn a verdict into "no concluyente"), the A3 outcome is "no concluyente" for every variant either way (`adequate_sample` false), but the change was made after seeing the result and was not mentioned above.
* Reviewer re-computation matched: gate coverage 69/1533 (A1), 0/173 positives, face coverage >= 0.8 on 14 positives (8 speakers) in A1 and 19 (10) in A2; the identity-probe numbers match the JSON.
