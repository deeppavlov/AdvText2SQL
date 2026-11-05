# Structured classifier: train/dev only

CV-selected candidate: `text_l2_01`.

| Part | Approach | BIRD correct | BIRD accuracy | Ambrosia correct | Ambrosia accuracy | Ambrosia F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | economical | 71/131 | 54.20% | 94/136 | 69.12% | 8.70% |
| train | rules | 74/131 | 56.49% | 92/136 | 67.65% | 4.35% |
| train | hybrid | 74/131 | 56.49% | 107/136 | 78.68% | 55.38% |
| train | structured | 81/131 | 61.83% | 114/136 | 83.82% | 64.52% |
| dev | economical | 24/55 | 43.64% | 34/46 | 73.91% | 14.29% |
| dev | rules | 21/55 | 38.18% | 34/46 | 73.91% | 14.29% |
| dev | hybrid | 21/55 | 38.18% | 36/46 | 78.26% | 58.33% |
| dev | structured | 21/55 | 38.18% | 34/46 | 73.91% | 45.45% |

Predictions are restricted to observed profiles 35, 14+35, 14+32+35.
This is a structured classifier over three masks, not independent selection of all 16 flags.
Scores are replayed from saved single-pass outcomes, not a new SQL/LLM benchmark.
Hyperparameters selected by four grouped folds inside train. Dev only reports transfer.
No test predictions, test quality evaluation, or paid requests were performed.
The existing test has already been viewed; future independent validation is still needed.
