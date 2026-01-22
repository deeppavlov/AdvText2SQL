import os

from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .text2sql_implementation import Text2SQLGenerator


async def generate_description_text2sql(
    tool_description: str, text2sql_agent: Text2SQLGenerator
) -> str:
    priority = "## Приоритет вызова \nСредний \n\n"

    const_part_of_description = (
        "## Условия вызова\n"
        "- Все используемые запросы в этот инструмент должны быть явными, термины должны быть понятными, "
        "единственным образом трактуемыми. -> Пояснение добавлено, чтобы инструмент не вызывался раньше, чем "
        "все термины будут расшифрованы\n"
        "- Сформирован запрос, по которому можно выполнить поиск в базе данных: параметры и значения, по которым "
        "можно выполнить поиск. -> Инструмент может работать только если готов запрос к БД\n\n"
        "## Возвращает\n"
        "- str | None: Таблица в формате markdown или None, если инструмент не справился с запросом"
    )

    prompt = """
    Тебе доступны описание функции генерации sql запросов на основе запроса пользователя и структура базы данных.
    Ты должен написать краткое описание функции с учетом данных (одно - два предложения),
    в одном-двух предложениях описать, что хранится в этой базе данных,
    написать краткое описание функции с учетом данных (одно - два предложения),
    а также ты должен написать два примера запросов, который пользователь может прислать.
    Напиши в следующем формате:
    ## Описание
        {Описание функции}
        {Описание базы данных}
    ## Примеры запросов
        {Примеры запросов}

    Описание функции: {tool_description}
    Структура базы данных: {schema}
    """

    client = OpenAIChatCompletionClient(
        api_key=os.environ["DESC_LLM_API_KEY"],
        base_url=os.environ["DESC_LLM_BASE_URL"],
        model=os.environ["DESC_LLM_MODEL_NAME"],
        temperature=0,
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

    schema = text2sql_agent.db_schema

    prompt = prompt.replace("{tool_description}", tool_description)
    prompt = prompt.replace("{schema}", schema)

    generated_description = await client.create(
        [UserMessage(content=prompt, source="user")]
    )

    description = (
        priority + generated_description.content + "\n\n" + const_part_of_description
    )

    return description
