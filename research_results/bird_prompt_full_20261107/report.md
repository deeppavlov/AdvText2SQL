# BIRD full: frozen compact SQL prompt comparison

Same 241 questions, features [2, 7, 14, 32, 35]; fresh paired calls; one descriptive repeat, teacher=0.
Selected SQL candidate fixed before full from compact dev; source method permits zero demonstrations.

| Arm | Saved | Correct | EX | Delta, pp | Rescued / harmed | Generation / SQL errors | False ambiguity | Question tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 241/241 | 126 | 52.28% | +0.00 | 0 / 0 | 4 / 2 | 0 | 1601919 |
| selected_sql | 241/241 | 132 | 54.77% | +2.49 | 17 / 11 | 0 / 0 | 0 | 1619152 |

Measured total tokens: 3221071; unknown allowance: 0; accounted: 3221071. Root ledger is authoritative, arm usage excludes build/unknown overhead.

No full-driven prompt selection or automatic deployment. Full overlaps viewed training/dev; not an independent test. One paired pass does not establish run-to-run robustness.
