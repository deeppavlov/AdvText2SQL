# DeepSeek: staged combinations

Ambrosia: error is always incorrect (strict-v1).

| Case | Flags | Benchmark | Status | Saved | Accuracy | Delta baseline, pp | Delta parent, pp | F1 | Tokens | Errors |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| baseline | all false | bird | completed | 241 | 43.15 | 0.00 | — | — | 488664 | 0 |
| baseline | all false | ambrosia | completed | 227 | 70.93 | 0.00 | — | 5.71 | 292465 | 0 |
| feat_2 | 2 | bird | completed | 241 | 47.30 | 4.15 | — | — | 2134231 | 0 |
| feat_2 | 2 | ambrosia | completed | 227 | 70.48 | -0.44 | — | 10.67 | 679868 | 0 |
| feat_35 | 35 | bird | completed | 241 | 47.30 | 4.15 | — | — | 472680 | 0 |
| feat_35 | 35 | ambrosia | completed | 227 | 71.37 | 0.44 | — | 10.96 | 289153 | 0 |
| feat_14 | 14 | bird | completed | 241 | 46.89 | 3.73 | — | — | 652844 | 0 |
| feat_14 | 14 | ambrosia | completed | 227 | 70.93 | 0.00 | — | 13.16 | 439921 | 0 |
| feat_32 | 32 | bird | completed | 241 | 44.81 | 1.66 | — | — | 824766 | 0 |
| feat_32 | 32 | ambrosia | completed | 227 | 77.53 | 6.61 | — | 50.49 | 425914 | 0 |
| schema_2_35 | 2,35 | bird | completed | 241 | 46.06 | 2.90 | feat_2: -1.24; feat_35: -1.24 | — | 2129514 | 0 |
| schema_2_35 | 2,35 | ambrosia | completed | 227 | 70.04 | -0.88 | feat_2: -0.44; feat_35: -1.32 | 8.11 | 681230 | 0 |
| sql_rules | 14,35 | bird | completed | 241 | 48.13 | 4.98 | feat_35: +0.83 | — | 638050 | 0 |
| sql_rules | 14,35 | ambrosia | completed | 227 | 70.48 | -0.44 | feat_35: -0.88 | 8.22 | 444037 | 0 |
| sql_ambiguity | 14,32,35 | bird | completed | 241 | 48.55 | 5.39 | sql_rules: +0.41; feat_32: +3.73 | — | 979404 | 0 |
| sql_ambiguity | 14,32,35 | ambrosia | completed | 227 | 80.18 | 9.25 | sql_rules: +9.69; feat_32: +2.64 | 57.94 | 552999 | 0 |

Reference rows reuse saved answers; no paid baseline reruns.
Stages 2/3 use exploratory selection on BIRD, not an independent test.
References and source hashes are frozen in experiment.json. Original reports are unchanged.
