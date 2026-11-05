import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from adv_text2sql.mcp_servers.text2sql_tool.src import feature_selector as rules  # noqa: E402
from adv_text2sql.mcp_servers.text2sql_tool.src.selective_generator import (  # noqa: E402
    SelectiveText2SQLGenerator, isolated_generator_class, schema_statistics_from_tsv,
)
from evaluate_rule_selector import choose_policy, evaluate_policy, frozen_write  # noqa: E402


def registry():
    return {name: {"benchmarks": {b: {
        "questions": 10, "recorded_run_cost": {"mean_tokens_per_question": tokens, "recorded_calls": 20},
    } for b in ("bird", "ambrosia")}} for name, tokens in (("economical", 100), ("hybrid", 200))}


def record(identifier, question, cheap=True, full=True, split="dev", benchmark="bird", ambiguous=False):
    return {
        "id": identifier, "benchmark": benchmark, "split": split,
        "inputs": {"question": question, "evidence": "", "schema_stats": {"table_count": 2}},
        "analysis": {"gold_ambiguous": ambiguous},
        "outcomes": {name: {"observed": True, "correct": value, "generation_error": False,
                            "execution_error": False, "predicted_ambiguous": ambiguous if value else not ambiguous}
                     for name, value in (("economical", cheap), ("hybrid", full))},
    }


class RuleSelectorTests(unittest.TestCase):
    def test_decisions_use_signals_and_exact_evaluated_feature_sets(self):
        selector = rules.RuleBasedFeatureSelector("ambiguity_or_complexity")
        simple = selector.select("List customer names.")
        ambiguous = selector.select("Show employees and managers with offices or parking.")
        complex_query = selector.select("What is the average revenue per region?")
        self.assertEqual(simple.features, (35,))
        self.assertEqual(ambiguous.features, (14, 32, 35))
        self.assertEqual(complex_query.features, (14, 32, 35))
        self.assertEqual(sum(ambiguous.feature_flags.values()), 3)
        self.assertFalse(ambiguous.feature_flags["FEAT_33"])
        self.assertFalse(ambiguous.feature_flags["FEAT_17"])
        self.assertIn("mixed_coordination", ambiguous.reasons)

    def test_literal_values_do_not_trigger_rules_and_schema_policy_is_explicit(self):
        selector = rules.RuleBasedFeatureSelector("ambiguity_or_large_schema")
        self.assertEqual(selector.select('List stores named "All Good".').configuration, "economical")
        self.assertEqual(selector.select("Count users", schema_stats={"table_count": 20}).configuration, "hybrid")
        self.assertEqual(selector.select("Count users", schema_stats={"table_count": 2}).configuration, "economical")

    def test_inputs_reject_targets_identifiers_and_invalid_values(self):
        selector = rules.RuleBasedFeatureSelector()
        valid = {"question": "List users", "evidence": "", "schema_stats": {}}
        for field in ("db_id", "benchmark", "difficulty", "labels", "gold_ambiguous", "SQL", "split"):
            with self.assertRaises(ValueError):
                selector.select_inputs({**valid, field: "hidden"})
        for invalid in ({"table_count": -1}, {"table_count": True}, {"db_id": 10}):
            with self.assertRaises(ValueError):
                selector.select_inputs({**valid, "schema_stats": invalid})
        with self.assertRaises(ValueError):
            selector.select(" ")

    def test_non_input_outcome_changes_do_not_change_decisions(self):
        a = record("one", "Show every customer", cheap=False)
        b = copy.deepcopy(a)
        b.update(id="changed", benchmark="ambrosia", db_id="other", split="test")
        b["outcomes"]["hybrid"]["correct"] = False
        first, second = rules.RuleBasedFeatureSelector(), rules.RuleBasedFeatureSelector()
        self.assertEqual(first.select_inputs(a["inputs"]), second.select_inputs(b["inputs"]))

    def test_dev_selection_ignores_train_outcomes_and_accepts_no_test_rows(self):
        train = [record("train", "List users", split="train")]
        dev = [record("dev", "List users", cheap=False, full=True)]
        choice, _ = choose_policy(train, dev, registry())
        self.assertEqual(choice, "always_hybrid")
        train[0]["outcomes"]["hybrid"]["correct"] = False
        self.assertEqual(choose_policy(train, dev, registry())[0], choice)
        with self.assertRaises(ValueError):
            choose_policy(train, [record("test", "List users", split="test")], registry())

    def test_quality_tie_prefers_cheaper_static_fallback(self):
        train = [record("t", "List users", split="train")]
        dev = [record("d", "List users")]
        self.assertEqual(choose_policy(train, dev, registry())[0], "always_economical")

    def test_replay_tracks_rescues_losses_and_does_not_count_errors_as_tn(self):
        rows = [record("a", "Show all users", False, True, benchmark="ambrosia", ambiguous=True),
                record("b", "List names", True, False, benchmark="ambrosia")]
        report, _ = evaluate_policy(rows, "ambiguity", registry())
        metrics = report["benchmarks"]["ambrosia"]
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["vs_static"]["hybrid"], {"rescued": 1, "lost": 0, "net_correct": 1})
        self.assertEqual(metrics["ambiguity_metrics"]["f1"], 1)
        rows[1]["outcomes"]["economical"].update(correct=False, generation_error=True)
        metrics = evaluate_policy(rows, "ambiguity", registry())[0]["benchmarks"]["ambrosia"]
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["ambiguity_metrics"]["tn"], 0)
        self.assertEqual(metrics["generation_errors"], 1)
        self.assertFalse(metrics["exact_selected_cost_available"])
        self.assertEqual(metrics["estimated_tokens_from_run_means"], 300)

    def test_missing_observation_cannot_be_replayed(self):
        row = record("x", "List users")
        row["outcomes"]["hybrid"].update(observed=False, correct=None)
        with self.assertRaisesRegex(ValueError, "observed"):
            evaluate_policy([row], "always_economical", registry())

    def test_artifact_freezing_and_code_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            artifact = {"format_version": "rule-selector-selection-v1", "rule_version": rules.RULE_VERSION,
                        "selector_source_sha256": hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest(),
                        "selected_policy": "ambiguity"}
            frozen_write(path, json.dumps(artifact))
            frozen_write(path, json.dumps(artifact))
            self.assertEqual(rules.RuleBasedFeatureSelector.from_file(path).policy, "ambiguity")
            with self.assertRaisesRegex(ValueError, "Frozen"):
                frozen_write(path, "changed")
            artifact["selector_source_sha256"] = "wrong"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                rules.RuleBasedFeatureSelector.from_file(path)

    def test_runtime_schema_matches_dataset_statistics(self):
        from build_configuration_dataset import schema_statistics
        rows = [("a", "n", "INTEGER"), ("a", "v", "VARCHAR"), ("b", "d", "TIMESTAMP"), ("b", "b", "BOOLEAN")]
        schema = "table\tcolumn\ttype\n" + "\n".join("\t".join(r) for r in rows)
        self.assertEqual(schema_statistics_from_tsv(schema), schema_statistics(rows))
        for invalid in ("", "table\tcolumn\ttype", schema + "\na\tn\tINTEGER"):
            with self.assertRaises(ValueError):
                schema_statistics_from_tsv(invalid)


class RuntimeSelectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_queries_keep_isolated_flags_and_build_once(self):
        from adv_text2sql.mcp_servers.text2sql_tool.src import text2sql_implementation as original
        before = {name: value for name, value in vars(original).items() if name.startswith("FEAT_")}
        calls, built, engines = [], [], []

        class Engine:
            def __init__(self):
                self.disposed = False
                engines.append(self)

            def dispose(self):
                self.disposed = True

        def factory(configuration):
            base = isolated_generator_class(configuration)
            globals_ = base.query.__wrapped__.__globals__

            class Fake(base):
                def __init__(self, db_uri, llm_client):
                    self.db_uri, self.llm_client, self.engine = db_uri, llm_client, Engine()

                def _get_db_schema_tsv(self):
                    built.append(configuration)
                    return "table\tcolumn\ttype\na\tn\tINTEGER"

                async def _check_ambiguity(self, question):
                    await asyncio.sleep(0)
                    calls.append((configuration, "classic", globals_["FEAT_32"], question))
                    return {"status": "success", "ambiguous": False}

                async def _detect_ambiguity_by_taxonomy(self, question):
                    await asyncio.sleep(0)
                    calls.append((configuration, "taxonomy", globals_["FEAT_32"], question))
                    return {"is_ambiguous": False}

                async def generate_sql(self, *args, **kwargs):
                    await asyncio.sleep(0)
                    return {"status": "success", "sql_query": "SELECT 1" if not globals_["FEAT_14"] else "SELECT 2"}

            return Fake

        tool = SelectiveText2SQLGenerator("test://db", object(), rules.RuleBasedFeatureSelector("ambiguity"),
                                         generator_factory=factory)
        with self.assertRaisesRegex(RuntimeError, "build"):
            await tool.query("List users")
        await tool.build()
        cheap, hybrid = await asyncio.gather(tool.query("List users"), tool.query("Show every user"))
        self.assertEqual(cheap["query"], "SELECT 1")
        self.assertEqual(hybrid["query"], "SELECT 2")
        self.assertEqual(cheap["feature_selection"]["features"], [35])
        self.assertEqual(hybrid["feature_selection"]["features"], [14, 32, 35])
        self.assertEqual(built, ["economical"])
        self.assertIn(("economical", "classic", False, "List users"), calls)
        self.assertIn(("hybrid", "taxonomy", True, "Show every user"), calls)
        await tool.query("question: List names, evidence (may be empty): all means everything")
        self.assertEqual(calls[-1][0], "economical")  # evidence must not become question text
        self.assertEqual(before, {name: value for name, value in vars(original).items() if name.startswith("FEAT_")})
        tool.close()
        self.assertTrue(all(e.disposed for e in engines))


if __name__ == "__main__":
    unittest.main()
