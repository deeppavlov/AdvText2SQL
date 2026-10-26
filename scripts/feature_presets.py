"""Fixed feature configurations used in reproducible comparisons."""

HYBRID_FEATURES = (2, 3, 4, 8, 12, 17, 18, 19, 27, 29, 30, 32, 33, 35)
IMPLEMENTED_FEATURES = tuple(n for n in range(1, 37) if n not in (22, 24, 31))
ABLATION_FEATURES = IMPLEMENTED_FEATURES
FEATURE_NOTES = {
    3: "Механизм не активируется без FEAT_2; FEAT_2 выключен",
    4: "Механизм не активируется без FEAT_2; FEAT_2 выключен",
    5: "Флаг не переключает light fallback в текущей реализации",
    7: "Механизм не активируется без FEAT_2; FEAT_2 выключен",
    8: "Инфраструктура: rollback при подготовке схемы",
    9: "Дамп схем через этот флаг текущим runner не вызывается",
    10: "Инфраструктура: измерение времени build и размера промпта",
    11: "Инфраструктура: структурированные логи",
    17: "Повторы при ошибке генерации; SQL-валидация FEAT_16 выключена",
    18: "Инфраструктура: backoff при rate limit",
    19: "Инфраструктура: дополнительная пауза между вопросами",
    26: "Изолированный кэш; обученные hints не подмешиваются, холодный старт не измеряет их пользу",
    27: "Feedback требует повторной генерации; FEAT_17 выключен",
    33: "Механизм не активируется без FEAT_32; FEAT_32 выключен",
    34: "Механизм не активируется без FEAT_32/33; оба флага выключены",
    36: "Механизм не активируется без FEAT_15; FEAT_15 выключен",
}


def ablation_cases(features):
    """All-false baseline, then exactly one enabled flag, always against baseline."""
    cases = [{"name": "baseline", "enabled": [], "control": None, "targets": []}]
    for number in sorted(set(features)):
        if number not in IMPLEMENTED_FEATURES:
            raise ValueError(f"FEAT_{number} is not implemented")
        cases.append({"name": f"feat_{number}", "enabled": [number], "control": "baseline",
                      "targets": [number], "note": FEATURE_NOTES.get(number, "")})
    return cases


def feature_flags(configuration, enabled_features=None):
    if configuration == "combination":
        values = list(enabled_features or [])
        enabled = set(values)
        if len(values) != len(enabled) or len(enabled) < 2 or not enabled <= set(IMPLEMENTED_FEATURES):
            raise ValueError("Combinations require at least two distinct implemented flags")
        return {f"FEAT_{number}": number in enabled for number in range(1, 37)}
    if configuration == "ablation":
        enabled = set(enabled_features or ())
        if len(enabled) != 1 or not enabled <= set(IMPLEMENTED_FEATURES):
            raise ValueError("Single-feature ablation requires exactly one implemented flag")
        return {f"FEAT_{number}": number in enabled for number in range(1, 37)}
    if enabled_features:
        raise ValueError("Explicit feature flags require configuration=ablation")
    if configuration not in ("baseline", "hybrid", "hybrid_no32_33"):
        raise ValueError(f"Unknown feature configuration: {configuration}")
    enabled = set(HYBRID_FEATURES) if configuration != "baseline" else set()
    if configuration == "hybrid_no32_33":
        enabled -= {32, 33}
    return {f"FEAT_{number}": number in enabled for number in range(1, 37)}
