# DeepSeek: PK/FK hints added to the static hybrid

Ambrosia: strict-v1; generation error is always incorrect.

| Case | Flags | Benchmark | Status | Saved | Accuracy | Delta baseline, pp | Delta 14+32+35, pp | F1 | Tokens | Errors |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | all false | bird | completed | 241 | 43.15 | 0.00 | — | — | 488664 | 0 |
| baseline | all false | ambrosia | completed | 227 | 70.93 | 0.00 | — | 5.71 | 292465 | 0 |
| feat_35 | 35 | bird | completed | 241 | 47.30 | 4.15 | — | — | 472680 | 0 |
| feat_35 | 35 | ambrosia | completed | 227 | 71.37 | 0.44 | — | 10.96 | 289153 | 0 |
| sql_ambiguity | 14,32,35 | bird | completed | 241 | 48.55 | 5.39 | — | — | 979404 | 0 |
| sql_ambiguity | 14,32,35 | ambrosia | completed | 227 | 80.18 | 9.25 | — | 57.94 | 552999 | 0 |
| hybrid_with_relationships | 1,14,32,35 | bird | completed | 241 | 48.55 | 5.39 | 0.00 | — | 1003530 | 0 |
| hybrid_with_relationships | 1,14,32,35 | ambrosia | completed | 227 | 80.18 | 9.25 | 0.00 | 58.72 | 561885 | 0 |

Only FEAT_1 is added; all other feature settings are fixed.
Saved controls are reused without LLM calls. Original artifacts are unchanged.
This is exploratory model selection on previously used questions, not an independent final test.
