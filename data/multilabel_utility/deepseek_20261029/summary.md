# DeepSeek: weak multilabel benefit observations

468 questions, 60 DBs, 16 eligible heads.

Single-flag labels use all-false baseline. Utility = correct(feature) - correct(baseline).
Benefit target=1 only for observed gain; ties=0 means no observed gain, not proven uselessness.
Preference labels use only +1/-1 pairs. Invalid/missing/excluded labels stay null and masked.

| Feature | Eligible | BIRD: + / - / ties | Ambrosia: + / - / ties | Loss masks true | Exclusion |
| --- | --- | --- | --- | ---: | --- |
| FEAT_1 | True | 9 / 4 / 228 | 4 / 0 / 223 | 468 |  |
| FEAT_2 | True | 18 / 8 / 215 | 2 / 3 / 222 | 468 |  |
| FEAT_3 | False | 8 / 1 / 232 | 2 / 2 / 223 | 0 | dependency_not_enabled |
| FEAT_4 | False | 6 / 3 / 232 | 2 / 3 / 222 | 0 | dependency_not_enabled |
| FEAT_5 | False | 7 / 3 / 231 | 0 / 2 / 225 | 0 | no_effective_switch |
| FEAT_6 | False | 17 / 3 / 221 | 1 / 1 / 225 | 0 | incomplete_schema_context_in_recorded_bird_run |
| FEAT_7 | False | 9 / 1 / 231 | 2 / 2 / 223 | 0 | dependency_not_enabled |
| FEAT_8 | False | 8 / 3 / 230 | 1 / 2 / 224 | 0 | fixed_infrastructure |
| FEAT_9 | False | 6 / 1 / 234 | 2 / 3 / 222 | 0 | fixed_infrastructure |
| FEAT_10 | False | 5 / 3 / 233 | 3 / 2 / 222 | 0 | fixed_infrastructure |
| FEAT_11 | False | 7 / 4 / 230 | 2 / 2 / 223 | 0 | fixed_infrastructure |
| FEAT_12 | True | 9 / 5 / 227 | 1 / 2 / 224 | 468 |  |
| FEAT_13 | True | 6 / 2 / 233 | 1 / 2 / 224 | 468 |  |
| FEAT_14 | True | 16 / 7 / 218 | 3 / 3 / 221 | 468 |  |
| FEAT_15 | True | 7 / 2 / 232 | 3 / 2 / 222 | 468 |  |
| FEAT_16 | True | 5 / 2 / 234 | 2 / 4 / 221 | 468 |  |
| FEAT_17 | False | 7 / 4 / 230 | 2 / 1 / 224 | 0 | fixed_infrastructure |
| FEAT_18 | False | 8 / 2 / 231 | 1 / 2 / 224 | 0 | fixed_infrastructure |
| FEAT_19 | False | 6 / 2 / 233 | 1 / 3 / 223 | 0 | fixed_infrastructure |
| FEAT_20 | True | 7 / 2 / 232 | 3 / 21 / 203 | 468 |  |
| FEAT_21 | True | 7 / 5 / 229 | 1 / 3 / 223 | 468 |  |
| FEAT_23 | True | 11 / 9 / 221 | 1 / 3 / 223 | 468 |  |
| FEAT_25 | True | 13 / 5 / 223 | 2 / 3 / 221 | 467 |  |
| FEAT_26 | False | 7 / 3 / 231 | 1 / 3 / 223 | 0 | stateful_cache_without_question_history_features |
| FEAT_27 | False | 7 / 3 / 231 | 2 / 2 / 223 | 0 | dependency_not_enabled |
| FEAT_28 | True | 4 / 2 / 235 | 1 / 1 / 225 | 468 |  |
| FEAT_29 | True | 12 / 5 / 224 | 3 / 3 / 221 | 468 |  |
| FEAT_30 | True | 13 / 13 / 215 | 2 / 2 / 223 | 468 |  |
| FEAT_32 | True | 7 / 3 / 231 | 27 / 12 / 188 | 468 |  |
| FEAT_33 | False | 5 / 4 / 232 | 2 / 2 / 223 | 0 | dependency_not_enabled |
| FEAT_34 | False | 10 / 3 / 228 | 2 / 3 / 222 | 0 | dependency_not_enabled |
| FEAT_35 | True | 12 / 2 / 227 | 2 / 1 / 224 | 468 |  |
| FEAT_36 | False | 7 / 5 / 229 | 2 / 2 / 223 | 0 | dependency_not_enabled |

Raw + / - counts for excluded mechanisms are audit only; they are not training labels.

| Split | Questions | Observed targets | Positive targets | Preference targets |
| --- | ---: | ---: | ---: | ---: |
| train | 267 | 4271 | 117 | 187 |
| dev | 101 | 1616 | 57 | 102 |
| test | 100 | 1600 | 40 | 66 |

Conditional effects are exported separately; they do not overwrite independent targets.
These are single-run observational labels, not causal estimates or optimal combination masks.
Inputs: question/evidence/schema_stats only. Same frozen DB/family partitions as Phase 0.2.
Test has already been viewed in Phase 1 and is retrospective. No independent final test.
No per-question cost estimates are fabricated. BIRD and Ambrosia tasks remain separate.
