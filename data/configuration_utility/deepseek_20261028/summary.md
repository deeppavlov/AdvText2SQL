# Configuration utility dataset

Offline retrospective dataset; not an independent final test.

| Benchmark | Rows | DBs | Hybrid only correct | Economical only correct | Both correct | Neither correct | Quality labels | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bird | 241 | 8 | 16 | 13 | 101 | 111 | 29 | 53.94% |
| ambrosia | 227 | 52 | 33 | 13 | 149 | 32 | 46 | 85.90% |

| Split | Benchmark | Rows | DBs | Quality labels | Gold ambiguous | Prompt-exposed families, rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | bird | 131 | 4 | 15 | 0 | 0 |
| train | ambrosia | 136 | 23 | 27 | 42 | 6 |
| dev | bird | 55 | 2 | 9 | 0 | 0 |
| dev | ambrosia | 46 | 15 | 12 | 13 | 6 |
| test | bird | 55 | 2 | 5 | 0 | 0 |
| test | ambrosia | 45 | 14 | 7 | 12 | 6 |

Only the inputs object is a model feature source. DB IDs, benchmark difficulty, gold labels,
families, split IDs and outcomes are audit/target fields, not classifier inputs.
utility = configuration_correct - control_correct; missing observations remain null.
Quality preference is null for ties. Cost-aware ties prefer the economical configuration
using the fixed run-level cost policy, not fabricated per-question costs.
All exact question token costs are unknown/masked. Recorded run totals include probe/build calls.
Source ambiguity labels use strict-v1: generation error is always incorrect.
Small sets were not appended. These observations do not provide independent multilabel feature targets.
Family normalization and high-similarity grouping are conservative heuristics, not semantic guarantees.
Configuration selection and prompt examples have already exposed this material; test is retrospective.
Family stress folds are separate diagnostics over Ambrosia, not part of main train/dev/test tuning.
