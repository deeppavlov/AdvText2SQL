# Static hybrid: offline audit

No API, SQL execution or new training.

Accuracy uses fixed denominators 241/227; blank/error output fails.
Rescued/harmed uses observed pairs only. Scores are single recorded outcomes.

| Feature | BIRD correct | Delta pp | Rescued / harmed | Ambrosia correct | Delta pp | Rescued / harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FEAT_1 | 109/241 | +2.07 | 9 / 4 | 165/227 | +1.76 | 4 / 0 |
| FEAT_2 | 114/241 | +4.15 | 18 / 8 | 160/227 | -0.44 | 2 / 3 |
| FEAT_3 | 111/241 | +2.90 | 8 / 1 | 161/227 | +0.00 | 2 / 2 |
| FEAT_4 | 107/241 | +1.24 | 6 / 3 | 160/227 | -0.44 | 2 / 3 |
| FEAT_5 | 108/241 | +1.66 | 7 / 3 | 159/227 | -0.88 | 0 / 2 |
| FEAT_6 | 118/241 | +5.81 | 17 / 3 | 161/227 | +0.00 | 1 / 1 |
| FEAT_7 | 112/241 | +3.32 | 9 / 1 | 161/227 | +0.00 | 2 / 2 |
| FEAT_8 | 109/241 | +2.07 | 8 / 3 | 160/227 | -0.44 | 1 / 2 |
| FEAT_9 | 109/241 | +2.07 | 6 / 1 | 160/227 | -0.44 | 2 / 3 |
| FEAT_10 | 106/241 | +0.83 | 5 / 3 | 162/227 | +0.44 | 3 / 2 |
| FEAT_11 | 107/241 | +1.24 | 7 / 4 | 161/227 | +0.00 | 2 / 2 |
| FEAT_12 | 108/241 | +1.66 | 9 / 5 | 160/227 | -0.44 | 1 / 2 |
| FEAT_13 | 108/241 | +1.66 | 6 / 2 | 160/227 | -0.44 | 1 / 2 |
| FEAT_14 | 113/241 | +3.73 | 16 / 7 | 161/227 | +0.00 | 3 / 3 |
| FEAT_15 | 109/241 | +2.07 | 7 / 2 | 162/227 | +0.44 | 3 / 2 |
| FEAT_16 | 107/241 | +1.24 | 5 / 2 | 159/227 | -0.88 | 2 / 4 |
| FEAT_17 | 107/241 | +1.24 | 7 / 4 | 162/227 | +0.44 | 2 / 1 |
| FEAT_18 | 110/241 | +2.49 | 8 / 2 | 160/227 | -0.44 | 1 / 2 |
| FEAT_19 | 108/241 | +1.66 | 6 / 2 | 159/227 | -0.88 | 1 / 3 |
| FEAT_20 | 109/241 | +2.07 | 7 / 2 | 143/227 | -7.93 | 3 / 21 |
| FEAT_21 | 106/241 | +0.83 | 7 / 5 | 159/227 | -0.88 | 1 / 3 |
| FEAT_23 | 106/241 | +0.83 | 11 / 9 | 159/227 | -0.88 | 1 / 3 |
| FEAT_25 | 112/241 | +3.32 | 13 / 5 | 159/227 | -0.88 | 2 / 3 |
| FEAT_26 | 108/241 | +1.66 | 7 / 3 | 159/227 | -0.88 | 1 / 3 |
| FEAT_27 | 108/241 | +1.66 | 7 / 3 | 161/227 | +0.00 | 2 / 2 |
| FEAT_28 | 106/241 | +0.83 | 4 / 2 | 161/227 | +0.00 | 1 / 1 |
| FEAT_29 | 111/241 | +2.90 | 12 / 5 | 161/227 | +0.00 | 3 / 3 |
| FEAT_30 | 104/241 | +0.00 | 13 / 13 | 161/227 | +0.00 | 2 / 2 |
| FEAT_32 | 108/241 | +1.66 | 7 / 3 | 176/227 | +6.61 | 27 / 12 |
| FEAT_33 | 105/241 | +0.41 | 5 / 4 | 161/227 | +0.00 | 2 / 2 |
| FEAT_34 | 111/241 | +2.90 | 10 / 3 | 160/227 | -0.44 | 2 / 3 |
| FEAT_35 | 114/241 | +4.15 | 12 / 2 | 162/227 | +0.44 | 2 / 1 |
| FEAT_36 | 106/241 | +0.83 | 7 / 5 | 161/227 | +0.00 | 2 / 2 |

FEAT_6: incomplete BIRD schema context. FEAT_25: one invalid Ambrosia output, excluded from paired utility.

## Proposed probes (not run)

| Stage | Name | Flags | Controls |
| --- | --- | --- | --- |
| core_factorial | core_without_tsv | 14,32 | feat_14, feat_32, sql_ambiguity |
| core_factorial | core_without_rules | 32,35 | feat_32, feat_35, sql_ambiguity |
| one_mechanism | hybrid_plus_2 | 2,14,32,35 | sql_ambiguity |
| one_mechanism | hybrid_plus_15 | 14,15,32,35 | sql_ambiguity |
| one_mechanism | hybrid_plus_16 | 14,16,32,35 | sql_ambiguity |
| one_mechanism | hybrid_plus_21 | 14,21,32,35 | sql_ambiguity |
| one_mechanism | hybrid_plus_23 | 14,23,32,35 | sql_ambiguity |
| one_mechanism | hybrid_plus_25 | 14,25,32,35 | sql_ambiguity |
| one_mechanism | hybrid_plus_28 | 14,28,32,35 | sql_ambiguity |
| one_mechanism | hybrid_plus_29 | 14,29,32,35 | sql_ambiguity |
| one_mechanism | hybrid_plus_30 | 14,30,32,35 | sql_ambiguity |
| statistics_bundle | statistics_compact | 2,7,14,32,35 | hybrid_plus_2 |
| statistics_bundle | statistics_types | 2,3,7,14,32,35 | statistics_compact |
| statistics_bundle | statistics_counts | 2,3,4,7,14,32,35 | statistics_types |
| repair_bundle | sanitize_validate | 14,15,16,32,35 | hybrid_plus_15, hybrid_plus_16 |
| repair_bundle | validated_retry | 14,15,16,17,32,35 | sanitize_validate |
| repair_bundle | validated_feedback | 14,15,16,17,27,32,35 | validated_retry |
| repair_bundle | judged_feedback | 14,15,16,17,20,27,32,35 | validated_feedback |
| workflow_interaction | validated_voting | 14,15,16,28,32,35 | hybrid_plus_28, sanitize_validate |
| workflow_interaction | routed_decomposition | 14,29,30,32,35 | hybrid_plus_29, hybrid_plus_30 |

Twenty proposed masks; running all on both full cohorts would be 40 benchmark runs / 9,360 question answers.
This is a maximum initial menu, not a recommendation to execute all. Stage and filter on train/dev.
Semantic audit and interpretation: ../../docs/10_static_hybrid_search.md
