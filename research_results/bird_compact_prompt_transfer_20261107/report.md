# BIRD compact prompt transfer — retrospective dev

Features: [2, 7, 14, 32, 35]. 55 fixed dev questions, three repeats, no teacher.
Original means the static compact control, not all-false. Identical arms share measurements within a repeat.

| Repeat | Arm | Saved | EX | Delta | Rescued / harmed | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | original | 55/55 | 38.18% | +0.00 pp | 0 / 0 | 0 / 2 |
| 1 | fixed_few | 55/55 | 41.82% | +3.64 pp | 5 / 3 | 0 / 1 |
| 1 | gepa_zero | 55/55 | 38.18% | +0.00 pp | 0 / 0 | 0 / 2 |
| 1 | mipro_zero | 55/55 | 38.18% | +0.00 pp | 0 / 0 | 0 / 2 |
| 1 | textgrad_zero | 55/55 | 45.45% | +7.27 pp | 7 / 3 | 1 / 3 |
| 1 | mipro_few | 55/55 | 45.45% | +7.27 pp | 6 / 2 | 0 / 2 |
| 2 | original | 55/55 | 40.00% | +0.00 pp | 0 / 0 | 1 / 2 |
| 2 | fixed_few | 55/55 | 38.18% | -1.82 pp | 2 / 3 | 0 / 2 |
| 2 | gepa_zero | 55/55 | 43.64% | +3.64 pp | 4 / 2 | 1 / 2 |
| 2 | mipro_zero | 55/55 | 41.82% | +1.82 pp | 6 / 5 | 0 / 0 |
| 2 | textgrad_zero | 55/55 | 38.18% | -1.82 pp | 1 / 2 | 0 / 4 |
| 2 | mipro_few | 55/55 | 47.27% | +7.27 pp | 6 / 2 | 0 / 0 |
| 3 | original | 55/55 | 38.18% | +0.00 pp | 0 / 0 | 1 / 2 |
| 3 | fixed_few | 55/55 | 47.27% | +9.09 pp | 6 / 1 | 1 / 2 |
| 3 | gepa_zero | 55/55 | 38.18% | +0.00 pp | 3 / 3 | 1 / 0 |
| 3 | mipro_zero | 55/55 | 49.09% | +10.91 pp | 7 / 1 | 0 / 0 |
| 3 | textgrad_zero | 55/55 | 40.00% | +1.82 pp | 1 / 0 | 1 / 1 |
| 3 | mipro_few | 55/55 | 43.64% | +5.45 pp | 5 / 2 | 0 / 2 |

| Arm | Complete repeats | Mean EX | SD | Mean paired delta |
| --- | ---: | ---: | ---: | ---: |
| original | 3/3 | 38.79% | 1.05 pp | 0.00 pp |
| fixed_few | 3/3 | 42.42% | 4.58 pp | 3.64 pp |
| gepa_zero | 3/3 | 40.00% | 3.15 pp | 1.21 pp |
| mipro_zero | 3/3 | 43.03% | 5.55 pp | 4.24 pp |
| textgrad_zero | 3/3 | 41.21% | 3.78 pp | 2.42 pp |
| mipro_few | 3/3 | 45.45% | 1.82 pp | 6.67 pp |

Measured tokens: 9731375; conservative unknown usage: 0; accounted budget: 9731375.

No full/test selection, independent-test claim or automatic deployment. SQL execution errors remain model outcomes; provider/DB outages stop the job.
