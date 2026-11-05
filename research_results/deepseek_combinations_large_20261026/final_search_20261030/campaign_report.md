# Ограниченная серия статического гибрида

Только свежие guarded-прогоны; исторические ответы в рейтинг не входят.

| Train, повтор 1 | Фичи | BIRD | Ambrosia | Средняя Accuracy | Δ к свежему H | F1 Ambrosia | Токены | Готово |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_control | все false | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| hybrid_control_repeat | 14+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| hybrid_plus_2 | 2+14+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| core_without_tsv | 14+32 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| statistics_compact | 2+7+14+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| statistics_types | 2+3+7+14+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| statistics_counts | 2+3+4+7+14+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| sanitize_validate | 14+15+16+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| validated_retry | 14+15+16+17+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| validated_feedback | 14+15+16+17+27+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| judged_feedback | 14+15+16+17+20+27+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| validated_voting | 14+15+16+28+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| routed_decomposition | 14+29+30+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| statistics_repair | 2+14+15+16+17+20+27+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |
| statistics_routing | 2+14+29+30+32+35 | 0/131, pending | 0/120, pending | — | — | — | 0 | False |

## train: только три завершённых повтора

| Конфигурация | BIRD, среднее | Ambrosia, среднее | F1 Ambrosia | Средняя Accuracy | Δ к H | Положительных повторов | Описательный 95% интервал Δ | Токены/повтор |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |

## dev: только три завершённых повтора

| Конфигурация | BIRD, среднее | Ambrosia, среднее | F1 Ambrosia | Средняя Accuracy | Δ к H | Положительных повторов | Описательный 95% интервал Δ | Токены/повтор |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |

Состояние: screening.
Отсутствие устойчивого выигрыша — допустимый итог. Лучший наблюдаемый набор не является глобальным максимумом.
Dev уже используется для сравнения финалистов; независимый итоговый test этой серией не создаётся.
Интервалы учитывают пары повторов и кластеры БД, но при малом числе БД остаются описательными.
Сравнение включает итоговые ошибки pipeline; F1/FP/ошибки и стоимость нужно разбирать вместе с Accuracy.
Папка ambrosia_227 — имя из переиспользуемого runtime; фактические ID/размеры заданы в experiment.json.
