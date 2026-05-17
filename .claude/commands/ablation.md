# /ablation — Запуск и интерпретация ablation-тестов

## Контекст проекта

Проект: AdvText2SQL. Ablation методология: **leave-one-in**.
Каждая фича тестируется поверх base stack: `FEAT_5=true FEAT_12=true FEAT_18=true FEAT_19=true`.
Без base stack pipeline физически не работает (false-ambiguity cascade + rate-limit failures).

Датасеты: BIRD-small (22 q) и Ambrosia-small (24 q).
Статистический порог: 1 вопрос = 4.55% (BIRD) / 4.17% (AMB) — изменения <2 вопросов — шум.

## Запуск одной фичи

```bash
# Синтаксис: bash local/run_single.sh "FEAT_N [FEAT_M ...]" [ambrosia|bird|both]
bash local/run_single.sh "FEAT_5 FEAT_12 FEAT_18 FEAT_19 FEAT_17" bird
bash local/run_single.sh "FEAT_5 FEAT_12 FEAT_18 FEAT_19 FEAT_27" both
```

Первый флаг в списке используется для именования выходной директории.

## Полный ablation

```bash
bash local/ablation_full.sh
```

Прогоны пропускаются если `ablation_results/<dir>/summary.txt` существует.
Чтобы перезапустить конкретную фичу — удали её `summary.txt`.

## Добавление новой фичи в ablation

1. Добавить `run_feature` строку в `local/ablation_full.sh`:
   ```bash
   run_feature "FEAT_N: описание" "feat_N_slug" $BASE FEAT_N=true
   ```
2. Добавить `feat_N_slug` в summary-цикл в конце того же файла
3. Добавить `FEAT_N=false` в блоки `printf` и `unset` в обоих скриптах

## Интерпретация результатов

- Δ считается от baseline (base stack без доп. фич) после перезапуска v2
- Δ ≥ 2 вопроса (≥9.09% BIRD / ≥8.33% AMB) — потенциально значимый сигнал
- Δ = 1 вопрос (4.55% / 4.17%) — вероятно шум, нужен повторный прогон
- Одинаковые результаты у многих фич → проверить что pipeline не падает на ambiguity check

## Известные проблемы

- Результаты v1 (commit f997dec) недостоверны — методология была сломана (см. practices_doc.md)
- FEAT_11 (+16.67% AMB) — аномалия, требует повторной проверки
- FEAT_20, FEAT_27 показывают 0 в изоляции — работают только внутри retry-loop (FEAT_17)
