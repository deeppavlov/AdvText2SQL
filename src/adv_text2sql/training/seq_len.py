"""
training/seq_len.py — динамический подбор `max_seq_length` для FT под схему БД.

Зачем: системный промпт собирается из `schema_str + relationships_str +
column_stats_str` (см. `dataset_builder.build_system_prompt`). Для card_games
(115 колонок) это ~5.6k токенов, но `column_stats_str` растёт ~линейно с числом
колонок — для enterprise-схемы (500+ колонок) промпт пробивает 12288 токенов,
на которых обучались эксперименты. Если `max_seq_length` ниже фактической длины
промпта — SFTTrainer обрежет system prompt, и модель будет учиться на префиксе
без ответа (тот же класс бага, что ловили на дефолтных 4096).

Поэтому перед генерацией Colab-ноутбука оцениваем фактическую длину
последовательности из profile.json и подбираем `max_seq_length` под неё, а не
берём константу.

Оценка токенов — эвристика (символы / `CHARS_PER_TOKEN`), без загрузки
tokenizer: модуль вызывается из лёгкого CLI, где transformers может быть не
установлен. Эвристика консервативная (занижает CHARS_PER_TOKEN → завышает
токены → берём запас).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..profiler.profile import Profile
from .dataset_builder import build_system_prompt

# ── Калибровка эвристики ─────────────────────────────────────────────────────
# Смешанный RU/EN текст + SQL + спецсимволы схемы. 3.5 символа/токен —
# консервативно (реальный BPE на кириллице ~2-4, на ASCII-SQL ~4-5).
CHARS_PER_TOKEN: float = 3.5

# Запас на user-вопрос + assistant-SQL + overhead chat-template поверх системного
# промпта. Самый длинный пример в синтетике редко превышает это.
COMPLETION_MARGIN_TOKENS: int = 768

# Допустимые значения max_seq_length. vLLM/unsloth любят степени двойки, плюс
# 12288 — на нём шли успешные эксперименты.
SEQ_LEN_BUCKETS: tuple[int, ...] = (2048, 4096, 8192, 12288, 16384)

# Потолок для Qwen2.5-Coder-7B (4-bit LoRA) на A100-40GB. Выше — риск OOM
# на Colab; нужно усечение column_stats (см. overflow ниже).
HARD_CAP: int = 16384


@dataclass
class SeqLenRecommendation:
    """Результат подбора max_seq_length под конкретный профиль."""

    max_seq_len: int          # рекомендованное значение для generate_finetune_notebook
    estimated_tokens: int     # оценка длины самого длинного обучающего примера
    overflow: bool            # True если estimated_tokens > HARD_CAP (промпт усечётся)
    message: str              # человекочитаемое объяснение для CLI


def estimate_prompt_tokens(profile: Profile) -> int:
    """Оценка длины (в токенах) самого длинного обучающего примера для профиля.

    = токены системного промпта (доминирует, растёт со схемой)
      + COMPLETION_MARGIN_TOKENS (вопрос + SQL + chat-template overhead).
    """
    system_prompt = build_system_prompt(profile)
    system_tokens = int(len(system_prompt) / CHARS_PER_TOKEN)
    return system_tokens + COMPLETION_MARGIN_TOKENS


def recommend_max_seq_len(profile: Profile) -> SeqLenRecommendation:
    """Подобрать max_seq_length под фактический размер системного промпта профиля.

    Логику выбора bucket'а реализует человек — см. TODO(human) ниже.
    """
    estimated = estimate_prompt_tokens(profile)

    if estimated > HARD_CAP:
        # Даже максимальный bucket не вмещает промпт — обрежется при FT.
        return SeqLenRecommendation(
            max_seq_len=HARD_CAP,
            estimated_tokens=estimated,
            overflow=True,
            message=(
                f"~{estimated} токенов > потолка {HARD_CAP} → max_seq_len={HARD_CAP}; "
                f"промпт БУДЕТ усечён, сократи column_stats (top-K колонок)"
            ),
        )

    # Наименьший bucket, вмещающий промпт: не платим за лишний контекст и не режем.
    max_seq_len = next(b for b in SEQ_LEN_BUCKETS if b >= estimated)
    return SeqLenRecommendation(
        max_seq_len=max_seq_len,
        estimated_tokens=estimated,
        overflow=False,
        message=f"~{estimated} токенов → max_seq_len={max_seq_len}",
    )
