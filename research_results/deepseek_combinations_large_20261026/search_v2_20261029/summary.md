# Static hybrid search v2

Search: train; confirmation: dev; full is exploratory, not an independent test.
Historical controls are replayed on identical question IDs, without new API calls.

| Scope | Case | Features | Benchmark | Status | Saved / total | Accuracy | Delta baseline pp | Delta H pp | F1 | Recall | FP | Errors | Tokens | Calls | Balance delta |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | baseline | all false | bird | historical | 131/131 | 51.15 | 0.00 | -5.34 | — | — | — | — | — | — | — |
| train | feat_14 | 14 | bird | historical | 131/131 | 54.20 | 3.05 | -2.29 | — | — | — | — | — | — | — |
| train | feat_32 | 32 | bird | historical | 131/131 | 52.67 | 1.53 | -3.82 | — | — | — | — | — | — | — |
| train | feat_35 | 35 | bird | historical | 131/131 | 54.20 | 3.05 | -2.29 | — | — | — | — | — | — | — |
| train | sql_rules | 14,35 | bird | historical | 131/131 | 56.49 | 5.34 | 0.00 | — | — | — | — | — | — | — |
| train | sql_ambiguity | 14,32,35 | bird | historical | 131/131 | 56.49 | 5.34 | 0.00 | — | — | — | — | — | — | — |
| train | core_without_tsv | 14,32 | bird | completed | 131/131 | 54.96 | 3.82 | -1.53 | — | — | — | 0 | 506311 | 263 | 0.0800 USD |
| train | core_without_rules | 32,35 | bird | completed | 131/131 | 54.96 | 3.82 | -1.53 | — | — | — | 0 | 408392 | 263 | 0.1000 USD |
| train | hybrid_plus_2 | 2,14,32,35 | bird | completed | 131/131 | 61.07 | 9.92 | 4.58 | — | — | — | 0 | 813283 | 263 | 0.2400 USD |
| train | hybrid_plus_15 | 14,15,32,35 | bird | completed | 131/131 | 56.49 | 5.34 | 0.00 | — | — | — | 0 | 499747 | 263 | 0.4200 USD |
| train | hybrid_plus_16 | 14,16,32,35 | bird | completed | 131/131 | 54.96 | 3.82 | -1.53 | — | — | — | 1 | 499156 | 263 | 0.0400 USD |
| train | hybrid_plus_21 | 14,21,32,35 | bird | completed | 131/131 | 58.02 | 6.87 | 1.53 | — | — | — | 0 | 507218 | 263 | 0.0400 USD |
| train | hybrid_plus_23 | 14,23,32,35 | bird | completed | 131/131 | 57.25 | 6.11 | 0.76 | — | — | — | 0 | 564863 | 293 | 0.0500 USD |
| train | hybrid_plus_25 | 14,25,32,35 | bird | completed | 131/131 | 58.02 | 6.87 | 1.53 | — | — | — | 0 | 579305 | 394 | 0.0600 USD |
| train | hybrid_plus_28 | 14,28,32,35 | bird | completed | 131/131 | 54.96 | 3.82 | -1.53 | — | — | — | 0 | 957380 | 525 | 0.0900 USD |
| train | hybrid_plus_29 | 14,29,32,35 | bird | completed | 131/131 | 58.02 | 6.87 | 1.53 | — | — | — | 0 | 467484 | 263 | 0.0400 USD |
| train | hybrid_plus_30 | 14,30,32,35 | bird | completed | 131/131 | 58.02 | 6.87 | 1.53 | — | — | — | 0 | 598074 | 394 | 0.3300 USD |
| train | hybrid_control_repeat | 14,32,35 | bird | completed | 131/131 | 54.96 | 3.82 | -1.53 | — | — | — | 0 | 499345 | 263 | 0.1200 USD |
| train | baseline | all false | ambrosia | historical | 136/136 | 68.38 | 0.00 | -10.29 | 0.00 | 0.00 | 1 | — | — | — | — |
| train | feat_14 | 14 | ambrosia | historical | 136/136 | 69.12 | 0.74 | -9.56 | 12.50 | 7.14 | 3 | — | — | — | — |
| train | feat_32 | 32 | ambrosia | historical | 136/136 | 75.74 | 7.35 | -2.94 | 49.23 | 38.10 | 7 | — | — | — | — |
| train | feat_35 | 35 | ambrosia | historical | 136/136 | 69.12 | 0.74 | -9.56 | 8.70 | 4.76 | 2 | — | — | — | — |
| train | sql_rules | 14,35 | ambrosia | historical | 136/136 | 67.65 | -0.74 | -11.03 | 4.35 | 2.38 | 3 | — | — | — | — |
| train | sql_ambiguity | 14,32,35 | ambrosia | historical | 136/136 | 78.68 | 10.29 | 0.00 | 55.38 | 42.86 | 5 | — | — | — | — |
| train | core_without_tsv | 14,32 | ambrosia | completed | 136/136 | 77.21 | 8.82 | -1.47 | 52.31 | 40.48 | 6 | 0 | 335562 | 200 | 0.0500 USD |
| train | core_without_rules | 32,35 | ambrosia | completed | 136/136 | 75.74 | 7.35 | -2.94 | 49.23 | 38.10 | 7 | 0 | 255103 | 200 | 0.4000 USD |
| train | hybrid_plus_2 | 2,14,32,35 | ambrosia | completed | 136/136 | 75.74 | 7.35 | -2.94 | 50.75 | 40.48 | 8 | 0 | 426717 | 198 | 0.3100 USD |
| train | hybrid_plus_15 | 14,15,32,35 | ambrosia | completed | 136/136 | 75.00 | 6.62 | -3.68 | 48.48 | 38.10 | 8 | 0 | 333162 | 199 | 0.0800 USD |
| train | hybrid_plus_16 | 14,16,32,35 | ambrosia | completed | 136/136 | 73.53 | 5.15 | -5.15 | 48.48 | 38.10 | 8 | 3 | 333820 | 199 | 0.0100 USD |
| train | hybrid_plus_21 | 14,21,32,35 | ambrosia | completed | 136/136 | 75.00 | 6.62 | -3.68 | 48.48 | 38.10 | 8 | 0 | 335031 | 199 | 0.0200 USD |
| train | hybrid_plus_23 | 14,23,32,35 | ambrosia | completed | 136/136 | 75.74 | 7.35 | -2.94 | 49.23 | 38.10 | 7 | 0 | 377393 | 306 | 0.0300 USD |
| train | hybrid_plus_25 | 14,25,32,35 | ambrosia | completed | 136/136 | 75.74 | 7.35 | -2.94 | 50.75 | 40.48 | 8 | 0 | 382601 | 313 | 0.0500 USD |
| train | hybrid_plus_28 | 14,28,32,35 | ambrosia | completed | 136/136 | 75.00 | 6.62 | -3.68 | 48.48 | 38.10 | 8 | 0 | 679156 | 423 | 0.0800 USD |
| train | hybrid_plus_29 | 14,29,32,35 | ambrosia | completed | 136/136 | 75.00 | 6.62 | -3.68 | 48.48 | 38.10 | 8 | 0 | 283464 | 199 | 0.0200 USD |
| train | hybrid_plus_30 | 14,30,32,35 | ambrosia | completed | 136/136 | 75.74 | 7.35 | -2.94 | 50.75 | 40.48 | 8 | 0 | 395262 | 309 | 0.3000 USD |
| train | hybrid_control_repeat | 14,32,35 | ambrosia | completed | 136/136 | 75.00 | 6.62 | -3.68 | 48.48 | 38.10 | 8 | 0 | 333282 | 199 | 0.1900 USD |
| dev | baseline | all false | bird | historical | 55/55 | 36.36 | 0.00 | -1.82 | — | — | — | — | — | — | — |
| dev | feat_14 | 14 | bird | historical | 55/55 | 38.18 | 1.82 | 0.00 | — | — | — | — | — | — | — |
| dev | feat_32 | 32 | bird | historical | 55/55 | 36.36 | 0.00 | -1.82 | — | — | — | — | — | — | — |
| dev | feat_35 | 35 | bird | historical | 55/55 | 43.64 | 7.27 | 5.45 | — | — | — | — | — | — | — |
| dev | sql_rules | 14,35 | bird | historical | 55/55 | 38.18 | 1.82 | 0.00 | — | — | — | — | — | — | — |
| dev | sql_ambiguity | 14,32,35 | bird | historical | 55/55 | 38.18 | 1.82 | 0.00 | — | — | — | — | — | — | — |
| dev | baseline | all false | ambrosia | historical | 46/46 | 73.91 | 0.00 | -4.35 | 14.29 | 7.69 | 0 | — | — | — | — |
| dev | feat_14 | 14 | ambrosia | historical | 46/46 | 73.91 | 0.00 | -4.35 | 14.29 | 7.69 | 0 | — | — | — | — |
| dev | feat_32 | 32 | ambrosia | historical | 46/46 | 78.26 | 4.35 | 0.00 | 54.55 | 46.15 | 3 | — | — | — | — |
| dev | feat_35 | 35 | ambrosia | historical | 46/46 | 73.91 | 0.00 | -4.35 | 14.29 | 7.69 | 0 | — | — | — | — |
| dev | sql_rules | 14,35 | ambrosia | historical | 46/46 | 73.91 | 0.00 | -4.35 | 14.29 | 7.69 | 0 | — | — | — | — |
| dev | sql_ambiguity | 14,32,35 | ambrosia | historical | 46/46 | 78.26 | 4.35 | 0.00 | 58.33 | 53.85 | 4 | — | — | — | — |
| full | baseline | all false | bird | historical | 241/241 | 43.15 | 0.00 | -5.39 | — | — | — | — | — | — | — |
| full | feat_14 | 14 | bird | historical | 241/241 | 46.89 | 3.73 | -1.66 | — | — | — | — | — | — | — |
| full | feat_32 | 32 | bird | historical | 241/241 | 44.81 | 1.66 | -3.73 | — | — | — | — | — | — | — |
| full | feat_35 | 35 | bird | historical | 241/241 | 47.30 | 4.15 | -1.24 | — | — | — | — | — | — | — |
| full | sql_rules | 14,35 | bird | historical | 241/241 | 48.13 | 4.98 | -0.41 | — | — | — | — | — | — | — |
| full | sql_ambiguity | 14,32,35 | bird | historical | 241/241 | 48.55 | 5.39 | 0.00 | — | — | — | — | — | — | — |
| full | baseline | all false | ambrosia | historical | 227/227 | 70.93 | 0.00 | -9.25 | 5.71 | 2.99 | 1 | — | — | — | — |
| full | feat_14 | 14 | ambrosia | historical | 227/227 | 70.93 | 0.00 | -9.25 | 13.16 | 7.46 | 4 | — | — | — | — |
| full | feat_32 | 32 | ambrosia | historical | 227/227 | 77.53 | 6.61 | -2.64 | 50.49 | 38.81 | 10 | — | — | — | — |
| full | feat_35 | 35 | ambrosia | historical | 227/227 | 71.37 | 0.44 | -8.81 | 10.96 | 5.97 | 2 | — | — | — | — |
| full | sql_rules | 14,35 | ambrosia | historical | 227/227 | 70.48 | -0.44 | -9.69 | 8.22 | 4.48 | 3 | — | — | — | — |
| full | sql_ambiguity | 14,32,35 | ambrosia | historical | 227/227 | 80.18 | 9.25 | 0.00 | 57.94 | 46.27 | 9 | — | — | — | — |

## Joint score (both benchmarks required)

| Scope | Case | Flags | BIRD | Ambrosia | Mean accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| train | hybrid_plus_2 | 2,14,32,35 | 61.07 | 75.74 | 68.40 |
| train | sql_ambiguity | 14,32,35 | 56.49 | 78.68 | 67.58 |
| train | hybrid_plus_25 | 14,25,32,35 | 58.02 | 75.74 | 66.88 |
| train | hybrid_plus_30 | 14,30,32,35 | 58.02 | 75.74 | 66.88 |
| train | hybrid_plus_21 | 14,21,32,35 | 58.02 | 75.00 | 66.51 |
| train | hybrid_plus_29 | 14,29,32,35 | 58.02 | 75.00 | 66.51 |
| train | hybrid_plus_23 | 14,23,32,35 | 57.25 | 75.74 | 66.49 |
| train | core_without_tsv | 14,32 | 54.96 | 77.21 | 66.08 |
| train | hybrid_plus_15 | 14,15,32,35 | 56.49 | 75.00 | 65.74 |
| train | core_without_rules | 32,35 | 54.96 | 75.74 | 65.35 |
| train | hybrid_control_repeat | 14,32,35 | 54.96 | 75.00 | 64.98 |
| train | hybrid_plus_28 | 14,28,32,35 | 54.96 | 75.00 | 64.98 |
| train | hybrid_plus_16 | 14,16,32,35 | 54.96 | 73.53 | 64.25 |
| train | feat_32 | 32 | 52.67 | 75.74 | 64.20 |
| train | sql_rules | 14,35 | 56.49 | 67.65 | 62.07 |
| train | feat_14 | 14 | 54.20 | 69.12 | 61.66 |
| train | feat_35 | 35 | 54.20 | 69.12 | 61.66 |
| train | baseline | all false | 51.15 | 68.38 | 59.76 |
| dev | feat_35 | 35 | 43.64 | 73.91 | 58.77 |
| dev | sql_ambiguity | 14,32,35 | 38.18 | 78.26 | 58.22 |
| dev | feat_32 | 32 | 36.36 | 78.26 | 57.31 |
| dev | feat_14 | 14 | 38.18 | 73.91 | 56.05 |
| dev | sql_rules | 14,35 | 38.18 | 73.91 | 56.05 |
| dev | baseline | all false | 36.36 | 73.91 | 55.14 |
| full | sql_ambiguity | 14,32,35 | 48.55 | 80.18 | 64.36 |
| full | feat_32 | 32 | 44.81 | 77.53 | 61.17 |
| full | feat_35 | 35 | 47.30 | 71.37 | 59.33 |
| full | sql_rules | 14,35 | 48.13 | 70.48 | 59.31 |
| full | feat_14 | 14 | 46.89 | 70.93 | 58.91 |
| full | baseline | all false | 43.15 | 70.93 | 57.04 |

## Best measured mean score by number of enabled algorithm flags

| Scope | X flags | Case | Mean accuracy |
| --- | ---: | --- | ---: |
| train | 0 | baseline | 59.76 |
| train | 1 | feat_32 | 64.20 |
| train | 2 | core_without_tsv | 66.08 |
| train | 3 | sql_ambiguity | 67.58 |
| train | 4 | hybrid_plus_2 | 68.40 |
| dev | 0 | baseline | 55.14 |
| dev | 1 | feat_35 | 58.77 |
| dev | 2 | sql_rules | 56.05 |
| dev | 3 | sql_ambiguity | 58.22 |
| full | 0 | baseline | 57.04 |
| full | 1 | feat_32 | 61.17 |
| full | 2 | sql_rules | 59.31 |
| full | 3 | sql_ambiguity | 64.36 |

Tokens include all recorded probes, build calls and attempts, not failed/unrecorded HTTP retries.
Per-parent deltas and rescued/harmed counts are in summary.json. Historical per-question costs are unknown.
A stage result is not a certificate of global optimality. No automatic next-stage paid runs.
