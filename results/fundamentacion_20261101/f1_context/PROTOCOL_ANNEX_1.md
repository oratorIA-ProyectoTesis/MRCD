# Protocol annex 1 (2026-11-01): deviations, declared before the report

1. Vc (head trained on folds 1-3) was NOT evaluated on fold 4. Reason: V0/Vc detect the filled-pause class, the head needs the noise-padded feature shards and its own loader, and the comparison would be cross-construct and descriptive only. V0 is reported on all folds 1-4 (it was not trained on DIHANA).
2. Logistic regression solver: `lbfgs` (liblinear rejected the 64-bit sparse indices on this platform). Same L2 model, same C grid.
3. Alignment found 15 of 371 occurrences not recovered by the ASR (P0); the P1 subset is the 1025-unit set of the repetition study and is used only for the recovery count.
4. No other change to PROTOCOL.md (sha256 in PROTOCOL.sha256.json).
