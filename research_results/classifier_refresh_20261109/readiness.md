# Classifier refresh: offline readiness

Audit only; no training labels, API calls or training.

| Split | BIRD | Ambrosia | Total |
| --- | ---: | ---: | ---: |
| train | 131 | 120 | 251 |
| dev | 55 | 46 | 101 |
| test | 55 | 45 | 100 |

| Benchmark | Fixed features | Train observations of selected arm | Verified feature-utility pairs |
| --- | --- | ---: | ---: |
| bird | 2+7+14+32+35 | 131 | 0 |
| ambrosia | 14+32 | 120 | 0 |

The two final full sources provide one chosen feature mask per task, not feature attribution.
Other historical archives still need a compatibility audit. No old labels were copied.

## Next requirements

- Version the catalog/constraints so FEAT_7 requires FEAT_2 and the five-feature BIRD control is representable
- Audit other historical archives before deciding which configuration observations need regeneration
- Keep model/prompt/evaluator compatibility explicit; old single-flag labels are not automatically compatible
- Build conditional configuration utility on train; unknown outcomes stay null with loss_mask=false
- Use dev for selection; previous test/full are already viewed and are not an independent final test
