# Регрессионный контроль статического pipeline

A — исторический результат; B — тот же SQL с новым evaluator; C — новая генерация с новым evaluator.

| Бенчмарк | Условие | Ответы | Правильно | EX / Accuracy | F1 | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bird | A historical | 241/241 | 132.0 | 54.77% | — | — |
| bird | B rescored | 241/241 | 131.0 | 54.36% | — | — |
| bird | C fresh | 241/241 | 128.0 | 53.11% | — | — |
| ambrosia | A historical | 211/211 | 171.0 | 81.04% | 60.78% | 49.21% |
| ambrosia | B rescored | 211/211 | 171.0 | 81.04% | 60.78% | 49.21% |
| ambrosia | C fresh | 211/211 | 167.0 | 79.15% | 56.00% | 44.44% |

Incomplete coverage/reference errors block a full score and paid generation. No automatic deployment.
One retrospective pass does not establish generalization or remove LLM run-to-run variability.
