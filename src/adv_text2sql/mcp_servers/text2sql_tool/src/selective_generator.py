"""Per-question routing around the legacy generator without shared flag mutation."""

import importlib.util

from .feature_selector import CONFIGURATIONS, RuleBasedFeatureSelector


def isolated_generator_class(configuration):
    """Each configuration owns its module globals, including across async awaits."""
    if configuration not in CONFIGURATIONS:
        raise ValueError("Only evaluated configurations are allowed")
    from . import text2sql_implementation as original

    name = original.__package__ + "._selector_" + configuration
    spec = importlib.util.spec_from_file_location(name, original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for number in range(1, 37):
        setattr(module, f"FEAT_{number}", number in CONFIGURATIONS[configuration])
    module.MAX_RETRIES = 5  # FEAT_17 remains false: one SQL generation attempt.
    return module.Text2SQLGenerator


def schema_statistics_from_tsv(schema):
    lines = schema.splitlines()
    if not lines or lines[0] != "table\tcolumn\ttype":
        raise ValueError("Selector requires the evaluated TSV schema representation")
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    if not rows or any(len(row) != 3 or not all(row) for row in rows):
        raise ValueError("Invalid or empty TSV schema")
    if len({(row[0], row[1]) for row in rows}) != len(rows):
        raise ValueError("Duplicate TSV columns")
    import re
    types = [row[2].casefold() for row in rows]
    return {
        "table_count": len({row[0] for row in rows}), "column_count": len(rows),
        "text_column_count": sum(bool(re.search(r"text|char|clob", t)) for t in types),
        "numeric_column_count": sum(bool(re.search(r"int|numeric|decimal|real|float|double", t)) for t in types),
        "temporal_column_count": sum(bool(re.search(r"date|time", t)) for t in types),
        "boolean_column_count": sum("bool" in t for t in types),
    }


class SelectiveText2SQLGenerator:
    """Same async build/query interface, with an explicit immutable selection result.

    Preparation uses only the economical TSV schema. Both configurations share
    that prepared schema; neither needs additional statistics or model calls.
    Pass a frozen selector loaded with RuleBasedFeatureSelector.from_file().
    """

    def __init__(self, db_uri, llm_client, selector=None, *, engine_factory=None, generator_factory=None):
        self.selector = selector or RuleBasedFeatureSelector()
        self._tools = {}
        factory = generator_factory or isolated_generator_class
        try:
            for configuration in CONFIGURATIONS:
                tool = factory(configuration)(db_uri=db_uri, llm_client=llm_client)
                self._tools[configuration] = tool
                if engine_factory is not None:
                    tool.engine.dispose()
                    tool.engine = engine_factory(db_uri)
        except BaseException:
            self.close()
            raise
        self.schema_stats = None

    @property
    def engine(self):
        return self._tools["economical"].engine

    async def build(self):
        self.schema_stats = None
        cheap = self._tools["economical"]
        await cheap.build()
        stats = schema_statistics_from_tsv(cheap.db_schema)
        # Exactly the same build-time flags (35) in both configurations.
        hybrid = self._tools["hybrid"]
        for field in ("db_schema", "db_relationships", "relationships_str", "column_stats",
                      "column_stats_str", "system_prompt", "_db_id"):
            setattr(hybrid, field, getattr(cheap, field))
        self.schema_stats = stats

    async def query(self, user_query, *, evidence=""):
        if self.schema_stats is None:
            raise RuntimeError("Call await build() before query()")
        question = user_query
        # Compatibility with BenchmarkBIRD(use_evidence=True), whose input is one string.
        if user_query.startswith("question: ") and ", evidence (may be empty): " in user_query:
            if evidence:
                raise ValueError("Evidence supplied twice")
            question, evidence = user_query[len("question: "):].rsplit(", evidence (may be empty): ", 1)
        decision = self.selector.select(question, evidence=evidence, schema_stats=self.schema_stats)
        model_query = f"question: {question}, evidence (may be empty): {evidence}" if evidence else user_query
        result = await self._tools[decision.configuration].query(model_query)
        return {**result, "feature_selection": decision.as_dict()}

    def close(self):
        for tool in self._tools.values():
            tool.engine.dispose()
