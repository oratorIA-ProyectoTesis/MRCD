# live_cfg comparison tables (generated)

| config | sessions | tick median s (median of sessions) | tick p95 s | ticks > 0.5 s % | peak RSS MB | latency median/p95 s | sessions diverging (final) | total differing frames | max frames/session | windows w/o row | duplicate rows | finalize s | load-valid runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| live_cfg_v1 | 6 | 0.169 | 0.195 | 0.0 | 1816.8 | 3.00/3.00 | 2/6 | 24 | 22 | 36 | 0 | 0.0 | 5/6 |
| live_cfg_v2 | 6 | 0.200 | 0.275 | 0.0 | 1816.8 | 4.00/4.00 | 2/6 | 24 | 22 | 0 | 0 | 3.2 | 6/6 |
| live_cfg_v3 | 6 | 0.192 | 0.262 | 0.0 | 1816.8 | 4.00/4.00 | 0/6 | 0 | 0 | 0 | 0 | 3.1 | 6/6 |

Per session (committed_bad before finalize / final_bad after): 501:v1=2/2; 501:v2=2/2; 501:v3=0/0; 502:v1=22/22; 502:v2=22/22; 502:v3=0/0; 503:v1=0/0; 503:v2=0/0; 503:v3=0/0; 504:v1=0/0; 504:v2=0/0; 504:v3=0/0; 505:v1=0/0; 505:v2=0/0; 505:v3=0/0; 506:v1=0/0; 506:v2=0/0; 506:v3=0/0

Background CPU % before each session run: [1.77, 3.2, 3.8, 3.89, 3.9, 4.71, 5.44, 5.74, 7.75, 8.23, 8.25, 9.0, 9.09, 9.12, 9.37, 12.91, 14.75, 16.45]

Silence scenarios (120 s):

| scenario | config | rows | windows w/o row | gaps | tick median s |
|---|---|---|---|---|---|
| digital_zeros | live_cfg_v1 | 229 | 6 | 0 | 0.086 |
| digital_zeros | live_cfg_v2 | 462 | 0 | 0 | 0.096 |
| digital_zeros | live_cfg_v3 | 462 | 0 | 0 | 0.089 |
| noise_-70dBFS | live_cfg_v1 | 229 | 6 | 0 | 0.206 |
| noise_-70dBFS | live_cfg_v2 | 462 | 0 | 0 | 0.237 |
| noise_-70dBFS | live_cfg_v3 | 462 | 0 | 0 | 0.222 |

## Criteria (rule: docs/LIVE_CFG_V3_DECISION_RULE.md)

- C1 completeness: **NOT MET**
- C2 parity (v2 shows differences AND v3 <= v2, <= 50 % frames): **MET**
- C3 latency within 0.05 s: **MET**
- C4 cost (needs load-valid runs) [NOT EVALUABLE: some runs had background CPU >= 15 %]: **NOT MET**
- C5 lost/duplicated = 0: **MET**
- C6 silence equal: **MET**

**Outcome: criteria not all met: default stays live_cfg_v2; v3 stays experimental.**