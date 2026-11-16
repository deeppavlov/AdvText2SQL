# Configuration classifier v2: saved-outcome replay

| Split | Policy | BIRD correct | BIRD EX | Ambrosia correct | Ambrosia Accuracy | Ambrosia F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | all_false | 70/131 | 53.44% | 83/120 | 69.17% | 5.13% |
| train | task_anchor | 82/131 | 62.60% | 96/120 | 80.00% | 61.29% |
| train | hybrid_14_32_35 | 74/131 | 56.49% | 93/120 | 77.50% | 55.74% |
| train | classifier_v2 | 87/131 | 66.41% | 103/120 | 85.83% | 71.19% |
| dev | all_false | 24/55 | 43.64% | 34/46 | 73.91% | 14.29% |
| dev | task_anchor | 26/55 | 47.27% | 37/46 | 80.43% | 60.87% |
| dev | hybrid_14_32_35 | 26/55 | 47.27% | 36/46 | 78.26% | 58.33% |
| dev | classifier_v2 | 24/55 | 43.64% | 37/46 | 80.43% | 60.87% |

Train-only grouped CV chooses regularization, ranking loss and anchor margin.
Dev is reported after the model is frozen; no test rows are loaded.
This is replay of saved single-pass responses, not a fresh downstream LLM run.
Outputs are complete measured configurations, not independent feature labels.
Ambrosia scores ambiguity decisions; SQL execution accuracy is measured only for BIRD.
No paid requests, database queries, encoder updates or LoRA training.
