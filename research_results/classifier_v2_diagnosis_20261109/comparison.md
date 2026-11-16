# Configuration v2: diagnostic input ablations

Fixed selected hyperparameters; 4 train-group folds; no new dev-model selection.

| Input | BIRD OOF correct | Ambrosia OOF correct | Macro accuracy |
| --- | ---: | ---: | ---: |
| text_schema | 81/131 | 97/120 | 71.33% |
| text_only | 81/131 | 96/120 | 70.92% |
| schema_only | 82/131 | 96/120 | 71.30% |
| intercept_only | 82/131 | 96/120 | 71.30% |

These are post-hoc diagnostic comparisons, not evidence for a new deployed winner.
Original model, dataset, checkpoints and generator are unchanged. No paid calls or test evaluation.
