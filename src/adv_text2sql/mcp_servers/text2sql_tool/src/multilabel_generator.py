"""Apply a classifier's feature mask with isolated flags and cached DB preparation."""

import asyncio
from functools import lru_cache
import importlib.util

from .multilabel_classifier import HEAD_FEATURES
from .selective_generator import schema_statistics_from_tsv

BUILD_FEATURES = frozenset((1, 2, 21, 23, 35))
STATE_FIELDS = ("db_schema", "db_relationships", "relationships_str", "column_stats",
                "column_stats_str", "system_prompt", "_db_id")


@lru_cache(maxsize=32)
def isolated_feature_class(features):
    if len(set(features)) != len(features) or not set(features) <= set(HEAD_FEATURES):
        raise ValueError("Only registered independent feature heads are supported")
    from . import text2sql_implementation as original
    name = original.__package__ + "._multilabel_" + "_".join(map(str, features))
    spec = importlib.util.spec_from_file_location(name, original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for number in range(1, 37):
        setattr(module, f"FEAT_{number}", number in features)
    module.MAX_RETRIES = 5  # Retry flag remains false, as in the training background.
    return module.Text2SQLGenerator


class MultilabelText2SQLGenerator:
    def __init__(self, db_uri, llm_client, selector, *, engine_factory=None, generator_factory=None):
        self.db_uri, self.llm_client, self.selector = db_uri, llm_client, selector
        self._engine_factory = engine_factory
        self._class_factory = generator_factory or isolated_feature_class
        self._templates = {}
        self._lock = asyncio.Lock()
        self.schema_stats = None
        self._preparation = self._new_tool((35,))

    def _new_tool(self, features):
        tool = self._class_factory(tuple(features))(db_uri=self.db_uri, llm_client=self.llm_client)
        if self._engine_factory is not None:
            tool.engine.dispose()
            tool.engine = self._engine_factory(self.db_uri)
        return tool

    @property
    def engine(self):
        return self._preparation.engine

    async def build(self):
        self.schema_stats = None
        await self._preparation.build()
        self.schema_stats = schema_statistics_from_tsv(self._preparation.db_schema)
        self._templates[(35,)] = self._preparation

    async def _template(self, features):
        key = tuple(n for n in features if n in BUILD_FEATURES)
        async with self._lock:
            if key not in self._templates:
                tool = self._new_tool(key)
                try:
                    await tool.build()
                except BaseException:
                    tool.engine.dispose()
                    raise
                self._templates[key] = tool
            return self._templates[key]

    async def query(self, user_query):
        if self.schema_stats is None:
            raise RuntimeError("Call await build() before query()")
        question, evidence = user_query, ""
        if user_query.startswith("question: ") and ", evidence (may be empty): " in user_query:
            question, evidence = user_query[len("question: "):].rsplit(", evidence (may be empty): ", 1)
        decision = self.selector.select_inputs({"question": question, "evidence": evidence, "schema_stats": self.schema_stats})
        features = tuple(decision["features"])
        if tuple(sorted(set(features))) != features or not set(features) <= set(HEAD_FEATURES):
            raise ValueError("Classifier emitted an invalid mask")
        template = await self._template(features)
        # Per-query instances keep mutable pruning/decomposition state from leaking across awaits.
        tool = self._new_tool(features)
        try:
            for field in STATE_FIELDS:
                setattr(tool, field, getattr(template, field))
            result = await tool.query(user_query)
            return {**result, "feature_selection": decision}
        finally:
            tool.engine.dispose()

    def close(self):
        unique = {id(tool): tool for tool in (self._preparation, *self._templates.values())}
        for tool in unique.values():
            tool.engine.dispose()
