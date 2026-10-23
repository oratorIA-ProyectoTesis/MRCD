# MRCD implementation and evidence status

This public summary describes the delivered software scope. Synthetic regressions and independent code reviews are software evidence, not scientific accuracy or operational acceptance.

## Delivery status

| Unit | Status at this commit |
| --- | --- |
| Campaign access, reference integrity and evaluable masks (U1–U3) | Delivered software slice; synthetic verification recorded. |
| Temporal cells and target projection (U4a) | Delivered software slice; synthetic verification recorded. |
| Offline cell features/model/training (U4b) | Delivered software slice; synthetic verification recorded. |
| Versioned product results and legacy guards (U4c) | Not included at this commit. |
| Explicit human semantics and review lineage (U5a) | Not included at this commit. |

## Limits and deferred work

- U5b task-aware evaluation is incomplete and excluded from this delivery. Later planned units are not claimed complete.
- Human annotator recruitment and missing EAF acquisition/import are deferred. No annotations, consent, or results are fabricated.
- Real corpus governance, frozen partitions, calibrated thresholds, compatible human-trained weights, and genuine scientific evaluation remain outstanding.
- Browser roundtrip, PostgreSQL concurrency, real-media performance and operational acceptance are not established by synthetic tests.
- Legacy weights are not relabeled as trained v2 weights. Unavailable evidence remains unavailable rather than zero.
- Test prerequisites depend on the selected suite: application development requirements do not replace scientific dependencies; human-interface regression tests additionally require Node.js.

Local operational commands, private filesystem paths and raw verification logs are intentionally not published.
