# Comparison with fresh control

Scope: train; control: hybrid_control_repeat.
Historical H remains unchanged in the frozen suite summary; this table uses the fresh control.

| Case | Flags | BIRD correct | BIRD accuracy | Delta fresh H pp | Ambrosia correct | Ambrosia accuracy | Delta fresh H pp | Ambrosia F1 | Mean accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid_plus_2 | 2,14,32,35 | 80/131 | 61.07 | +6.11 | 103/136 | 75.74 | +0.74 | 50.75 | 68.40 |
| hybrid_plus_25 | 14,25,32,35 | 76/131 | 58.02 | +3.05 | 103/136 | 75.74 | +0.74 | 50.75 | 66.88 |
| hybrid_plus_30 | 14,30,32,35 | 76/131 | 58.02 | +3.05 | 103/136 | 75.74 | +0.74 | 50.75 | 66.88 |
| hybrid_plus_21 | 14,21,32,35 | 76/131 | 58.02 | +3.05 | 102/136 | 75.00 | +0.00 | 48.48 | 66.51 |
| hybrid_plus_29 | 14,29,32,35 | 76/131 | 58.02 | +3.05 | 102/136 | 75.00 | +0.00 | 48.48 | 66.51 |
| hybrid_plus_23 | 14,23,32,35 | 75/131 | 57.25 | +2.29 | 103/136 | 75.74 | +0.74 | 49.23 | 66.49 |
| core_without_tsv | 14,32 | 72/131 | 54.96 | +0.00 | 105/136 | 77.21 | +2.21 | 52.31 | 66.08 |
| hybrid_plus_15 | 14,15,32,35 | 74/131 | 56.49 | +1.53 | 102/136 | 75.00 | +0.00 | 48.48 | 65.74 |
| core_without_rules | 32,35 | 72/131 | 54.96 | +0.00 | 103/136 | 75.74 | +0.74 | 49.23 | 65.35 |
| hybrid_control_repeat | 14,32,35 | 72/131 | 54.96 | +0.00 | 102/136 | 75.00 | +0.00 | 48.48 | 64.98 |
| hybrid_plus_28 | 14,28,32,35 | 72/131 | 54.96 | +0.00 | 102/136 | 75.00 | +0.00 | 48.48 | 64.98 |
| hybrid_plus_16 | 14,16,32,35 | 72/131 | 54.96 | +0.00 | 100/136 | 73.53 | -1.47 | 48.48 | 64.25 |

## Identical taxonomy messages and changed verdicts

| Comparison | Identical / current messages | Changed verdicts |
| --- | ---: | ---: |
| historical_H_vs_fresh_H | 86/86 | 7 |
| hybrid_plus_2_vs_fresh_H | 86/86 | 1 |
| hybrid_plus_25_vs_fresh_H | 86/86 | 1 |
| hybrid_plus_30_vs_fresh_H | 86/86 | 1 |
| hybrid_plus_21_vs_fresh_H | 86/86 | 0 |
| hybrid_plus_29_vs_fresh_H | 86/86 | 0 |
| hybrid_plus_23_vs_fresh_H | 86/86 | 1 |
| core_without_tsv_vs_fresh_H | 0/86 | 0 |
| hybrid_plus_15_vs_fresh_H | 86/86 | 0 |
| core_without_rules_vs_fresh_H | 86/86 | 1 |
| hybrid_plus_28_vs_fresh_H | 86/86 | 0 |
| hybrid_plus_16_vs_fresh_H | 86/86 | 0 |

No causal attribution to FEAT_14: it changes the SQL prompt, not the taxonomy prompt.
One trial per new mask is insufficient to establish a stable optimum. Question-level paired changes and source hashes are in the JSON report.
