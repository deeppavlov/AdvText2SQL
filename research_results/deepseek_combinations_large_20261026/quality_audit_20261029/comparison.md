# Schema and SQL-output audit

Offline; no API or SQL execution. Original results remain unchanged.

| Benchmark | Questions | Reference-consistent | Mismatch | Review |
| --- | ---: | ---: | ---: | ---: |
| bird | 241 | 241 | 0 | 0 |
| ambrosia | 227 | 211 | 16 | 0 |

## Inconsistent databases

| Benchmark | Database | Affected questions | Issues |
| --- | --- | ---: | --- |
| ambrosia | vague_2cols_location | 16 | missing_tables:clientproperties,clients,properties; missing_tables:department,job_posting,office; missing_tables:job_posting,office; missing_tables:properties,transactions |

## Output-contract re-scoring (same train IDs, mismatches not excluded)

| Case | BIRD original / contract | Ambrosia original / contract | Invalid BIRD / Ambrosia | Contract mean |
| --- | ---: | ---: | ---: | ---: |
| hybrid_plus_2 | 61.07 / 61.07 | 75.74 / 72.79 | 0 / 6 | 66.93 |
| hybrid_plus_30 | 58.02 / 58.02 | 75.74 / 75.00 | 0 / 2 | 66.51 |
| hybrid_plus_25 | 58.02 / 58.02 | 75.74 / 73.53 | 0 / 3 | 65.77 |
| hybrid_plus_29 | 58.02 / 58.02 | 75.00 / 73.53 | 0 / 3 | 65.77 |
| hybrid_plus_21 | 58.02 / 58.02 | 75.00 / 72.79 | 0 / 3 | 65.40 |
| core_without_rules | 54.96 / 54.96 | 75.74 / 75.74 | 1 / 0 | 65.35 |
| hybrid_plus_15 | 56.49 / 56.49 | 75.00 / 73.53 | 0 / 3 | 65.01 |
| hybrid_plus_23 | 57.25 / 57.25 | 75.74 / 72.06 | 0 / 7 | 64.66 |
| core_without_tsv | 54.96 / 54.96 | 77.21 / 74.26 | 0 / 6 | 64.61 |
| hybrid_plus_16 | 54.96 / 54.96 | 73.53 / 73.53 | 0 / 0 | 64.25 |
| hybrid_plus_28 | 54.96 / 54.96 | 75.00 / 73.53 | 0 / 3 | 64.25 |
| hybrid_control_repeat | 54.96 / 54.96 | 75.00 / 73.53 | 0 / 3 | 64.25 |

## Separate reference-consistent train subset (offline replay, not new runs)

| Case | BIRD | Ambrosia | Mean |
| --- | ---: | ---: | ---: |
| hybrid_plus_2 | 61.07 (131) | 75.83 (120) | 68.45 |
| hybrid_plus_30 | 58.02 (131) | 75.83 (120) | 66.92 |
| hybrid_plus_25 | 58.02 (131) | 75.83 (120) | 66.92 |
| hybrid_plus_23 | 57.25 (131) | 75.83 (120) | 66.54 |
| hybrid_plus_29 | 58.02 (131) | 75.00 (120) | 66.51 |
| hybrid_plus_21 | 58.02 (131) | 75.00 (120) | 66.51 |
| core_without_tsv | 54.96 (131) | 77.50 (120) | 66.23 |
| hybrid_plus_15 | 56.49 (131) | 75.00 (120) | 65.74 |
| core_without_rules | 54.96 (131) | 75.83 (120) | 65.40 |
| hybrid_plus_16 | 54.96 (131) | 75.00 (120) | 64.98 |
| hybrid_plus_28 | 54.96 (131) | 75.00 (120) | 64.98 |
| hybrid_control_repeat | 54.96 (131) | 75.00 (120) | 64.98 |

Passing output shape does not prove SQL correctness. Syntax/execution remains the job of FEAT_16/evaluation.
A schema mismatch is a dataset/setup problem, not evidence that a feature is harmful.
