"""
_registry — общий загрузчик реестра экспериментов.

Единый источник правды: experiments/registry.json. Все скрипты-оркестраторы
(exp_generate, exp_build, compare_generators) читают отсюда, поэтому имена
папок/файлов согласованы на всех платформах.

Соглашение о путях (одинаково локально и на Drive):
    data/exp/<name>/raw_<lang>.jsonl        — сырые Q-SQL по языку
    data/exp/<name>/validated.jsonl         — объединённые валидные (все языки)
    data/exp/<name>/train.jsonl, val.jsonl  — финальный датасет
    data/exp/<name>/eval_predictions.json   — предсказания на тесте
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REGISTRY_PATH = Path("experiments/registry.json")
EXP_ROOT = Path("data/exp")


@dataclass
class Experiment:
    name: str
    llm_model: str
    languages: list[str]
    count_per_lang: int

    @property
    def dir(self) -> Path:
        return EXP_ROOT / self.name

    @property
    def validated_path(self) -> Path:
        return self.dir / "validated.jsonl"

    @property
    def train_path(self) -> Path:
        return self.dir / "train.jsonl"

    @property
    def predictions_path(self) -> Path:
        return self.dir / "eval_predictions.json"

    def raw_path(self, lang: str) -> Path:
        return self.dir / f"raw_{lang}.jsonl"


@dataclass
class Registry:
    db_id: str
    profile_path: str
    bird_path: str
    experiments: list[Experiment]


def load_registry(path: str | Path = REGISTRY_PATH) -> Registry:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    exps = [
        Experiment(
            name=e["name"],
            llm_model=e["llm_model"],
            languages=e["languages"],
            count_per_lang=e["count_per_lang"],
        )
        for e in data["experiments"]
    ]
    return Registry(
        db_id=data["db_id"],
        profile_path=data["profile_path"],
        bird_path=data["bird_path"],
        experiments=exps,
    )
