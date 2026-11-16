# Anchor utility: fixed train-only experiment

| Method | BIRD OOF | Ambrosia OOF | BIRD gate | Ambrosia gate |
| --- | ---: | ---: | --- | --- |
| task_anchor | 82/131 | 96/120 | control | control |
| all_false | 70/131 | 83/120 | control | control |
| quality_base | 81/131 | 97/120 | False | False |
| quality_grounded | 79/131 | 93/120 | False | False |
| utility_base | 74/131 | 97/120 | False | False |
| utility_grounded | 72/131 | 96/120 | False | False |

Next validation nominations: {'bird': 'task_anchor', 'ambrosia': 'task_anchor'}

No outer-dev/test evaluation, paid requests, synthetic questions or source-label edits.
The gate is a predeclared practical screening rule, not statistical significance.
Fold heads are research artifacts, not a full-train production model.
