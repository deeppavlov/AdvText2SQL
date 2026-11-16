# Регрессионный контроль статического pipeline

A — исторический результат; B — тот же SQL с новым evaluator; C — новая генерация с новым evaluator.

| Бенчмарк | Условие | Ответы | Правильно | EX / Accuracy | F1 | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bird | A historical | 241/241 | 132.0 | 54.77% | — | — |
| bird | B rescored | 188/241 | — | — | — | — |
| bird | C fresh | 0/241 | — | — | — | — |
| ambrosia | A historical | 211/211 | 171.0 | 81.04% | 60.78% | 49.21% |
| ambrosia | B rescored | 0/211 | — | — | — | — |
| ambrosia | C fresh | 0/211 | — | — | — | — |

Incomplete coverage/reference errors block a full score and paid generation. No automatic deployment.
One retrospective pass does not establish generalization or remove LLM run-to-run variability.
