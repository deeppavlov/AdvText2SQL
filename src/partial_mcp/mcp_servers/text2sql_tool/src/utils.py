import logging
from functools import partial, wraps
from typing import Any

logger = logging.getLogger("text2sql_tool")


def get_log(level: str, message: str, **details: Any) -> dict[str, Any]:
    """
    Формирует JSON-RPC сообщение для логирования.

    Args:
        level: Уровень логирования (например: "error", "warning", "info").
        message: Основное сообщение лога.
        **details: Дополнительные параметры, которые попадут в секцию "details".

    Returns:
        Словарь в формате JSON-RPC 2.0 с уведомлением.
    """
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": level,
            "logger": "text2sql_tool",
            "data": {
                level: message,
                **details,
            },
        },
    }


get_error = partial(get_log, level="error")
get_warning = partial(get_log, level="warning")
get_info = partial(get_log, level="info")


def print_result():
    """
    Декоратор: печатает результат работы функции, если включён debug.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(result)
            return result

        return wrapper

    return decorator
