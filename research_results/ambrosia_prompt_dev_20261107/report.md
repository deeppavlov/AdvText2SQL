# Ambrosia frozen prompt dev confirmation

14+32; same fixed prompts and questions in three task repeats; no teacher.
Accuracy measures ambiguity decisions, not SQL execution. Identical arms share results within a repeat.

| Repeat | Arm | Saved | Accuracy | Delta, pp | Rescued / harmed | F1 | Recall | FP / FN | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | original | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 1 | mipro_zero__r1 | 46/46 | 78.26% | -2.17 | 1 / 2 | 58.33% | 53.85% | 4 / 6 | 0 |
| 1 | mipro_zero__r2 | 46/46 | 78.26% | -2.17 | 1 / 2 | 61.54% | 61.54% | 5 / 5 | 0 |
| 1 | mipro_zero__r3 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 1 | gepa_zero__r1 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 1 | textgrad_zero__r1 | 46/46 | 73.91% | -6.52 | 4 / 7 | 25.00% | 15.38% | 1 / 11 | 0 |
| 1 | mipro_few__r1 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 2 | original | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 2 | mipro_zero__r1 | 46/46 | 76.09% | -4.35 | 1 / 3 | 56.00% | 53.85% | 5 / 6 | 0 |
| 2 | mipro_zero__r2 | 46/46 | 78.26% | -2.17 | 1 / 2 | 61.54% | 61.54% | 5 / 5 | 0 |
| 2 | mipro_zero__r3 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 2 | gepa_zero__r1 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 2 | textgrad_zero__r1 | 46/46 | 73.91% | -6.52 | 4 / 7 | 25.00% | 15.38% | 1 / 11 | 0 |
| 2 | mipro_few__r1 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 3 | original | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 3 | mipro_zero__r1 | 46/46 | 76.09% | -4.35 | 1 / 3 | 56.00% | 53.85% | 5 / 6 | 0 |
| 3 | mipro_zero__r2 | 46/46 | 78.26% | -2.17 | 1 / 2 | 61.54% | 61.54% | 5 / 5 | 0 |
| 3 | mipro_zero__r3 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 3 | gepa_zero__r1 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |
| 3 | textgrad_zero__r1 | 46/46 | 73.91% | -6.52 | 4 / 7 | 25.00% | 15.38% | 1 / 11 | 0 |
| 3 | mipro_few__r1 | 46/46 | 80.43% | +0.00 | 0 / 0 | 60.87% | 53.85% | 3 / 6 | 0 |

| Arm | Complete repeats | Mean accuracy | SD, pp | Mean delta, pp | Mean F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 3/3 | 80.43% | 0.00 | +0.00 | 60.87% |
| mipro_zero__r1 | 3/3 | 76.81% | 1.26 | -3.62 | 56.78% |
| mipro_zero__r2 | 3/3 | 78.26% | 0.00 | -2.17 | 61.54% |
| mipro_zero__r3 | 3/3 | 80.43% | 0.00 | +0.00 | 60.87% |
| gepa_zero__r1 | 3/3 | 80.43% | 0.00 | +0.00 | 60.87% |
| textgrad_zero__r1 | 3/3 | 73.91% | 0.00 | -6.52 | 25.00% |
| mipro_few__r1 | 3/3 | 80.43% | 0.00 | +0.00 | 60.87% |

Measured tokens: 1199987; unknown usage allowance: 0; accounted: 1199987.

Retrospective dev; no independent-test claim or automatic deployment. Shared arms must not be counted as independent observations or duplicate costs.
