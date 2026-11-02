# Annex 2026-11-01 (1): clarification of the "current default" comparator (no protocol change)

* The dry-run log line "prolongation F1 0.6135 vs default 0.155" printed the paired DELTA (optimised minus V0 argmax), not the V0 F1. V0 argmax F1 on the identical units is 0.4583 (TP/FP/FN 341/180/626), identical to S3 E4 and to the earlier studies (0.45-0.54).
* The comparator `V0_argmax` is `mrcd.decision.any_window_argmax` (turn positive if ANY window argmax equals the class), which is the engine's event-emission rule (window argmax in `mrcd/engine.py`). The frozen "current product default" therefore equals the real product behaviour; no second comparator is needed. V0thr and the S3 best are reported as fixed comparators as frozen.
* Alignment check: comparator TP/FP/FN equal S3 e4; labels asserted equal to the S3 predictions for filler and repetition. No code or protocol change.
