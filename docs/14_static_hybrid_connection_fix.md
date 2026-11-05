# Исправление preflight и сохранения диагностики

30.10.2026. Первый запуск controls остановился до генерации ответов.

| Событие | Подтверждение | Вывод |
| --- | --- | --- |
| OperationalError при проверке student_club | Трассировка guarded_hybrid_runtime | Проверка каталога не смогла завершиться; схема не признана корректной или некорректной |
| SSH не отвечает | В static_hybrid_tunnel.log есть Timeout и выход SSH с кодом 255 | Подтверждена нестабильность соединения; точная причина SQLAlchemy OperationalError старым кодом скрыта |
| FileNotFoundError для train/monitor_stop.json | Дочерний процесс не успел создать папку train | Вторичная ошибка записи перекрыла исходную ошибку соединения |
| В исходной final_search_20261030 нет query_results/raw_responses | Проверено дерево файлов | Сохранённых вопросов и платных LLM-ответов нет; model probe следует после preflight |

## Исправления

Новые `resilient_hybrid_campaign.py` / `resilient_hybrid_runtime.py` переиспользуют
исходную ограниченную серию. Замороженные campaign, guarded-runtime и v2 не меняются.

- Папка прогона создаётся родителем **до** запуска дочернего процесса. При раннем
  отказе imports/preflight monitor может сохранить диагностику.
- Каталог каждой БД читается в одной read-only connection. Прежний inspect(engine)
  с NullPool мог открывать соединения для отдельных операций/таблиц.
- Временные OperationalError/InterfaceError/TimeoutError получают максимум пять
  попыток с паузой 5 секунд. Это повтор подключения, а не повтор вопроса/LLM.
- Не совпавшая схема, отсутствие БД или отказ доступа не пропускаются; auth и
  известные ошибки доступа/имени БД завершают проверку без повторов.
- В metadata видны phase=schema_preflight, текущая БД и номер попытки. События
  записываются в schema_preflight.jsonl; вместо потенциально секретного текста
  исключения сохраняется фиксированный код причины.
- Неуспешный preflight получает status=preflight_failed. Существующие predictions
  сохраняются, прежняя metadata архивируется в attempt_history.

Сервер SSH или PostgreSQL этим исправлением не восстанавливается. Если связь не
вернётся за ограниченные попытки, серия остановится с понятной диагностикой до
платной генерации. Проверка схемы не ослаблена; gold SQL не исполняется в preflight.

## Версия и запуск

Исходная папка `final_search_20261030/` сохранена как неудачная попытка preflight.
Новая папка:
`research_results/deepseek_combinations_large_20261026/final_search_20261030_connectionfix/`.

Маски, model identity, cohorts, output contract и правила выбора финалистов
сохранены; меняются runtime подключения, диагностика и соответствующий protocol hash.
Старый manifest не переписывается, ответы между версиями не копируются.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File local/run_resilient_hybrid_campaign.ps1 --schema-policy exclude_mismatches --phase controls
```

Офлайн-проверка перед ручным стартом:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File local/run_resilient_hybrid_campaign.ps1 --schema-policy exclude_mismatches --dry-run
```

Все последующие фазы и повторы приведены в [протоколе 13](13_static_hybrid_confirmation.md),
его команды обновлены на новый launcher. SSH-лог новой версии:
`research_results/runs/static_hybrid_tunnel_connectionfix.log`.

## Проверка

60 офлайн-тестов прошли, включая 10 новых проверок: восстановление после
временного отказа, конечный retry budget, отсутствие retry для auth/mismatch,
неразглашение текста исключений, отсутствие model runtime при провале preflight,
сохранность checkpoint/history и исходного RuntimeError при раннем падении child.
Ruff проходит; исходные campaign/v2 source hashes совпадают. PowerShell dry-run
подтверждает 131/120, три стартовых маски и 753 обработки вопросов.
Новых платных вызовов при исправлении нет; доступность реальной БД будет
проверена при ручном старте через SSH.
