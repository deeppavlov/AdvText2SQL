# Prompt optimizer comparison

Search selection uses grouped train only. Existing dev is retrospective; no global optimum claim.

| Slot | Method | Repeat | Status | Original inner | Selected inner | Tokens | Calls task / teacher |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| sql | gepa_zero | 42 | completed | 54.41% | 54.41% | 3343634 | 1504 / 47 |
| sql | mipro_zero | 42 | completed | 50.00% | 50.00% | 1211130 | 562 / 12 |
| sql | textgrad_zero | 42 | completed | 52.94% | 58.82% | 2981043 | 1602 / 20 |
| sql | mipro_few | 42 | completed | 48.53% | 57.35% | 2913232 | 916 / 12 |
| sql | gepa_zero | 43 | completed | 50.00% | 55.88% | 3523244 | 1560 / 45 |
| sql | mipro_zero | 43 | completed | 52.94% | 55.88% | 1477279 | 712 / 14 |
| sql | textgrad_zero | 43 | completed | 50.00% | 57.35% | 2990568 | 1601 / 20 |
| sql | mipro_few | 43 | completed | 51.47% | 55.88% | 2234375 | 928 / 14 |
| sql | gepa_zero | 44 | completed | 50.00% | 54.41% | 3405557 | 1592 / 38 |
| sql | mipro_zero | 44 | completed | 51.47% | 57.35% | 1444722 | 680 / 13 |
| sql | textgrad_zero | 44 | completed | 51.47% | 54.41% | 3066401 | 1607 / 20 |
| sql | mipro_few | 44 | completed | 51.47% | 54.41% | 2301528 | 912 / 13 |
| ambiguity | gepa_zero | 42 | completed | 83.33% | 83.33% | 1003860 | 419 / 48 |
| ambiguity | mipro_zero | 42 | completed | 83.33% | 95.00% | 926593 | 519 / 14 |
| ambiguity | textgrad_zero | 42 | completed | 83.33% | 85.00% | 1729476 | 1059 / 22 |
| ambiguity | mipro_few | 42 | completed | 85.00% | 85.00% | 1260384 | 675 / 14 |
| ambiguity | gepa_zero | 43 | pending | — | — | 0 | 0 / 0 |
| ambiguity | mipro_zero | 43 | completed | 83.33% | 90.00% | 852763 | 481 / 14 |
| ambiguity | textgrad_zero | 43 | pending | — | — | 0 | 0 / 0 |
| ambiguity | mipro_few | 43 | pending | — | — | 0 | 0 / 0 |
| ambiguity | gepa_zero | 44 | pending | — | — | 0 | 0 / 0 |
| ambiguity | mipro_zero | 44 | completed | 83.33% | 83.33% | 851647 | 493 / 14 |
| ambiguity | textgrad_zero | 44 | pending | — | — | 0 | 0 / 0 |
| ambiguity | mipro_few | 44 | pending | — | — | 0 | 0 / 0 |

| Slot | Method | Completed repeats | Mean inner | SD across repeats |
| --- | --- | ---: | ---: | ---: |
| ambiguity | gepa_zero | 1/3 | 83.33% | — |
| ambiguity | mipro_few | 1/3 | 85.00% | — |
| ambiguity | mipro_zero | 3/3 | 89.44% | 5.85 pp |
| ambiguity | textgrad_zero | 1/3 | 85.00% | — |
| sql | gepa_zero | 3/3 | 54.90% | 0.85 pp |
| sql | mipro_few | 3/3 | 55.88% | 1.47 pp |
| sql | mipro_zero | 3/3 | 54.41% | 3.89 pp |
| sql | textgrad_zero | 3/3 | 56.86% | 2.25 pp |

SD describes optimizer/run variability on one fixed split, not uncertainty over new databases.
Use confirmation_report.md after the separately launched dev phase.
