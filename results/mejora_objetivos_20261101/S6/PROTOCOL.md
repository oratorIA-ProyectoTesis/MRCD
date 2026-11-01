# S6 protocol: evaluable video for filled-pause presence (EPIC v2), frozen before any metric

Status: FROZEN 2026-11-01 (hash in `PROTOCOL.sha256`, committed before the gate, features or classifiers were run on the 74 videos). Self-review only; no independent review. EXPLORATORY: all 39 EPIC speakers were seen in earlier development, so nothing here is a speaker-independent or confirmatory test.

## 1. Question and non-claims

Does an automatically admitted, synchronised visual evidence block add anything to audio + ASR-text for detecting **presence of a filled pause (FPAUSE) per EPIC sentence**? "Evaluable" means measured with verifiable technical gates; it does not mean the video helps.

Non-claims: the visual channel never infers psychological state (no anxiety, doubt, intention, diagnosis); features are geometric (lip aperture, jaw opening, head movement). Absence of video/face is "not available: reason", never 0. The gate is technical and label-free (it never receives the label). No new human annotation, no automatic judge as reference. EPIC v2 is CC BY-NC 4.0: local use only; no video, frames, per-frame landmarks or EPIC text leave `results/.../S6/work/` (gitignored).

## 2. Units and labels

- Unit = EPIC v2 aligned sentence (automatic WebMAUS times) of the 74 local Spanish-original videos: 1533 sentences, 39 speakers (`work/units.csv`, built by `scripts/s6_video/s6_01_units.py`).
- Positive = source notation ` -xxx- ` (FPAUSE) present: 173 sentences (51 videos, 31 speakers; 2 speakers, `epic_actor_010` and `epic_actor_173`, hold 71).
- Negative = no FPAUSE mark. Negative coverage for FPAUSE is accredited in `results/auditoria_corpus_20261030/MATRIZ_CORPUS.md` (EPIC FPAUSE row, "negative coverage: yes (documented)"). It is a negative **for filled pause only** (not for repetition, revision, prolongation, block). Label noise: the transcription convention marks vocalic filled pauses only, may miss some, and sentence boundaries are automatic.
- Source text is used only to derive the label. A+T text features come from our own ASR (P0, faster-whisper small, CPU int8) on the audio, never from the EPIC transcript (that would leak the label).
- Evaluation units = gate-admitted units. If admitted negatives exceed 3x admitted positives, negatives are subsampled deterministically by ascending sha1(unit_id) (ASR cost bound; independent of features and models).

## 3. Technical gate (fixed values in `mrcd/av/config.py`, hash in PROTOCOL.sha256)

Per unit, all required (no label involved). Reasons are recorded per unit and per video; coverage = admitted/total by video and speaker.

| Criterion | Value | Justification |
|---|---|---|
| C0 duration | >= 1.5 s | Lip-audio correlation needs >= ~35 frames at 25 fps. |
| C1 face | dominant tracked face present and semi-frontal (|yaw| <= 45, |pitch| <= 35, repo cone) in >= 80 % of unit frames | Lip aperture is unreliable off-axis; X = 80 % leaves room for short tracking gaps. |
| C2 active speaker | no second face present simultaneously in > 25 % of frames | Parliament shots often show several people; with two faces we do not claim to know who speaks, and we do not choose by lip-audio correlation (selection bias). |
| C3 identity consistency | landmark-ratio descriptor drift between unit halves <= 3 robust SDs | Detects track swaps within a unit. It is not biometric identification. |
| C4 synchrony (video) | video offset = median of unit local offsets (units with local peak r >= 0.20, >= 5 units); 90 % bootstrap CI (500 resamples over units); require \|offset\| <= 120 ms and CI width <= 200 ms | ITU-R BT.1359 detectability thresholds are about -125/+45 ms; 120 ms is a tolerant magnitude cut. Offset resolution 40 ms (25 fps), parabolic refinement. Search +-400 ms. |
| C5 lip-audio correlation (unit) | Pearson r(lip aperture, log-RMS envelope) at the video offset (+-1 frame) >= 0.30 AND surrogate p <= 0.05 | r >= 0.30 is a moderate effect; the surrogate null pairs the unit's lip series with envelopes from other times of the same video (100 surrogates, same lag window), controlling for chance coupling and lag search. |

Thresholds are a-priori engineering choices, not tuned. Sensitivity (coverage only, never selection) is reported for r in {0.2, 0.4}. Sign convention: offset > 0 means video late.

## 4. Features (observable only)

- V (12): lip-aperture mean/std/p90/p10, aperture velocity RMS, steady-mouth fraction and longest steady run, jaw-open mean/std, mouth-pucker mean, head angular speed, head pose deviation. Computed on the active-track frames of the unit.
- Speaker normalisation: per-speaker robust median/MAD of the speaker's own frames in admitted units (label-free, `speaker_baseline_min_frames` = 250); head speed normalised the same way. A speaker with fewer frames is excluded with reason `speaker_baseline_insufficient`. Remaining scaling (imputation median, z-score, ridge residualisation) is fitted on the training folds only. Deviation to disclose: the held-out speaker's baseline uses that speaker's own unlabeled admitted frames (transductive, label-free); a deployed system would need a warm-up period of that speaker's video.
- Speaker identity is never a feature. Identity-leakage probe: linear (multinomial logistic) speaker-identification balanced accuracy on V features raw vs speaker-normalised, speakers with >= 8 admitted units, stratified 4-fold at unit level, chance = 1/K.
- A (12): duration, speech/voiced fractions, log-RMS mean/std, F0 median relative to the speaker's median, F0 SD, mean |dF0|, longest voiced stretch and its F0 SD, longest silence, silences/s (Silero VAD, pYIN, repo extractor).
- T (12): ASR (P0) word rate, word probability mean/min/low fraction, lexical-filler rate, literal repeats/s, max/mean inter-word gap, max/mean log word duration, truncations/s, sentence-punctuation rate (repo `word_features`). Units with zero ASR words keep zero counts and NaN statistics (imputed with the training median).

## 5. Comparisons (max 3 fusion variants)

Classifier everywhere: class-balanced logistic regression (C = 1), fixed threshold 0.5 (no tuning). Same units, same speaker-grouped folds for all models.

- AT: A + T (reference; in-scheme retraining of the equivalent text/audio head, because the checkpoints in `models_stale` have incompatible feature schemas and V0 was trained on DIHANA windows).
- V: V alone (control; descriptive, outside the Holm family).
- F1_concat: [A, T, V_speaker-normalised] (late concatenation).
- F2_gated: [A, T, c * V], c = clip(r_at_video_offset / 0.6, 0, 1) (confidence gating by gate quality).
- F3_resid: [A, T, V - Ridge(A, T)] (identity/audio residualisation: only the part of V not linearly predictable from A and T on the training fold).

## 6. Validation

Speaker-grouped 5-fold CV (whole speakers per fold, greedy positive balancing), 5 repeats with seeds 20261101+r; point estimates are means over repeats of pooled counts. Report P/R/F1, TP/FP/FN, support, AP, AUC. DIHANA fold 5 is not used. Sensitivity: whole analysis repeated without the two dominant speakers (re-splitting and retraining), and results per speaker where supports allow.

## 7. Statistics

Primary metric: pooled F1. Paired difference F1(variant) - F1(AT) on the same units. IC 95 % by speaker bootstrap (B = 2000, speakers resampled with replacement, all repeats recomputed on the same resample). Two-sided bootstrap p-values; Holm correction over the family {F1_concat, F2_gated, F3_resid} vs AT (3 comparisons). AP differences are secondary and not corrected.

## 8. Decision rule for the conclusion

- Adequate sample: >= 40 admitted positives and >= 10 speakers with a positive; otherwise the conclusion is "no concluyente" regardless of numbers.
- "aporte medido" for a variant: Holm p < 0.05, lower CI bound of dF1 > 0, dF1 > 0 also without the two dominant speakers, adequate sample.
- "sin aporte": CI of dF1 inside +-0.05 (equivalence margin) and Holm p >= 0.05.
- Otherwise "no concluyente". Overall conclusion = "aporte medido" if any variant qualifies, "sin aporte" if all do, else "no concluyente". Negative results are reported.

## 9. Compute plan

Face tracking at native 25 fps over all 74 videos (about 7 ms/frame measured in 3b), run serially after S1 finishes or shows no growing Python CPU load (checked every ~10 min, up to 120 min; any overlap with S1 is recorded in the README). ASR only on admitted units. No GPU, no network.

## 10. Known limits (declared in advance)

EPIC is parliamentary speech (21 sessions, 2004, L1 Spanish); all speakers seen in development; timing is automatic WebMAUS; labels are filled-pause presence per sentence; CC BY-NC 4.0; one run on one corpus; thresholds not calibrated; a face may be another person in a parliament shot (C2/C3 only partly mitigate); no independent test exists locally (0 untouched speakers).

## Amendment A1 (2026-11-01, before the gate was run on the 74 videos)

Reason: a pipeline smoke test on ONE 22 s video (3 sentences; no metric, no label used) showed that requiring >= 5 units per video with per-unit peak r >= 0.20 leaves short videos without any offset estimate, which would exclude them for a reason unrelated to synchrony quality. Changes to C4 only (everything else unchanged):

- Video offset = peak (parabolic refinement) of the frame-weighted mean of the lag-correlation curves of the units that pass C1 and C2 (no selection on correlation strength). Requires >= 3 contributing units and >= 150 pooled frames (6 s); otherwise the video is `offset_unestimable_few_units` (excluded with that reason, never admitted by default).
- CI = 90 % percentile bootstrap over contributing units (500 resamples). With only 3 units this CI is crude; this is a known weakness and is reported. Admission still requires |offset| <= 120 ms and CI width <= 200 ms.
- The per-unit C5 test (r >= 0.30 at the video offset +-1 frame and surrogate p <= 0.05) is unchanged.
- `mrcd/av/config.py` version `s6-gate-1.1-A1`; the new hash is in PROTOCOL.sha256.

## Amendment A2 (2026-11-01, AFTER the A1 gate was run; post-hoc, disclosed)

What was seen first: the A1 gate (`gate_summary_main.json`) admitted 69/1533 units, 0 positives, 2 speakers, 2 videos; 1150 units failed C1 (`face_not_visible_enough`). Diagnosis of the stored tracks (no labels, no classifier): a face is detected in only 31.7 % of all frames, yet the single best track is semi-frontal in a median 9.3 % of frames of its video, with a median of 6 tracks per video. So part of the failure is tracker fragmentation (the same person re-acquired under a new track id counts as absence under the single-active-track rule), not only absence of faces.

Change (C1/C2 only, mechanism not outcome-tuned; thresholds unchanged): with `merge_tracks = true`, coverage = fraction of unit frames where EXACTLY ONE face is present and it is semi-frontal; second-face share = fraction of frames with two or more faces present; the lip and shape series are taken from the sole face of each frame; C3 drift is computed on that merged series. C0, C4 (A1 estimator) and C5 are unchanged.

Status: the A1 result stays the protocol-conformant (frozen) gate outcome and is reported first. A2 is a post-hoc corrected gate: any ablation on A2-admitted units is exploratory and labelled as such. The decision rule (section 8) and adequacy thresholds are not changed.

## Amendment A3 (2026-11-01, after the A1 and A2 gate runs; exploratory-descriptive, disclosed)

What was seen: A2 admitted 66/1533 units and 0 positives (2 videos pass the video-level sync test). Independently of synchrony, only 14 positives (8 speakers) under A1 and 19 positives (10 speakers) under A2 have face coverage >= 0.8, far below the adequacy threshold (>= 40 positives, >= 10 speakers). The paired ablation on gate-admitted units is therefore NOT computable (0 positives).

A3 runs the same paired ablation (same features, folds, variants, Holm, bootstrap) on the A2 units that pass C0-C3 only (face visible, a single face, consistent identity) and whose synchrony is NOT verified (sync criteria C4/C5 are dropped). Purpose: exercise the full pipeline on real data and give a descriptive number. These are NOT gate-admitted units: a result here cannot support "aporte medido"; the conclusion is capped by the adequacy rule (n_pos < 40 gives "no concluyente"). The F2 gate confidence uses the unit's local peak r (no video offset exists). Everything else as in the protocol.
