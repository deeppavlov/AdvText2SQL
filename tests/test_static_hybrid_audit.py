"""Offline checks of the search evidence and actual interactions in the legacy generator."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from adv_text2sql.mcp_servers.text2sql_tool.src import text2sql_implementation as core  # noqa: E402
from audit_static_hybrid import BASE, FEATURES, KNOWN_COMBINATIONS, paired, proposed_probes  # noqa: E402


class AuditEvidenceTests(unittest.TestCase):
    def test_invalid_output_fails_score_but_is_not_an_observed_negative_pair(self):
        good = {"observed": True, "correct": True, "generation_error": False, "invalid_prediction": False}
        invalid = {"observed": False, "correct": None, "generation_error": None, "invalid_prediction": True}
        result = paired([{"observations": {"control": good, "treatment": invalid}}], "treatment", "control")
        self.assertEqual((result["accuracy"], result["delta_pp"]), (0, -100))
        self.assertEqual((result["missing_pairs"], result["harmed"], result["invalid_predictions"]), (1, 0, 1))

    def test_neutral_total_preserves_complementary_answers(self):
        yes = {"observed": True, "correct": True, "generation_error": False, "invalid_prediction": False}
        no = {**yes, "correct": False}
        rows = [{"observations": {"control": yes, "treatment": no}},
                {"observations": {"control": no, "treatment": yes}}]
        before = copy.deepcopy(rows)
        result = paired(rows, "treatment", "control")
        self.assertEqual((result["delta_pp"], result["rescued"], result["harmed"]), (0, 1, 1))
        self.assertEqual(rows, before)

    def test_probe_controls_exist_and_core_factorial_is_completed(self):
        known = {"baseline": (), **{f"feat_{n}": (n,) for n in FEATURES}, **KNOWN_COMBINATIONS}
        probes = proposed_probes()
        self.assertEqual(len(probes), 20)
        for probe in probes:
            self.assertTrue(set(probe["controls"]) <= set(known))
            f = set(probe["features"])
            self.assertTrue(f <= set(FEATURES))
            self.assertTrue(not f.intersection((3, 4, 7)) or 2 in f)
            self.assertTrue(27 not in f or 17 in f)
            self.assertFalse(28 in f and f.intersection((17, 20, 27, 30)))
            self.assertNotIn(tuple(probe["features"]), known.values())
            known[probe["name"]] = tuple(probe["features"])
        covered = {frozenset(f) for f in known.values() if set(f) <= set(BASE)}
        self.assertEqual(len(covered), 8)


class ActualFeatureSemanticsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.flags = patch.multiple(core, **{f"FEAT_{n}": False for n in FEATURES})
        self.flags.start()
        self.addCleanup(self.flags.stop)
        self.tool = object.__new__(core.Text2SQLGenerator)
        self.tool.db_schema = "FULL SCHEMA"
        self.tool.system_prompt = "SYSTEM WITH FULL SCHEMA"
        self.tool.relationships_str = "RELATIONSHIPS"
        self.tool.column_stats_str = "STATISTICS"

    async def test_tsv_overrides_heavy_and_light_flag_is_not_a_switch(self):
        with patch.object(self.tool, "_get_db_schema_tsv", return_value="TSV") as tsv, \
                patch.object(self.tool, "_get_db_schema_heavy", return_value="HEAVY") as heavy, \
                patch.object(self.tool, "_get_db_schema_light", return_value="LIGHT") as light:
            self.tool.db_uri = "unused/database"
            core.FEAT_35 = core.FEAT_6 = True
            await self.tool.build()
            self.assertEqual(self.tool.db_schema, "TSV")
            heavy.assert_not_called()
            core.FEAT_35 = core.FEAT_6 = False
            await self.tool.build()
            self.assertEqual(self.tool.db_schema, "LIGHT")
            tsv.assert_called_once()
            light.assert_called_once()

    async def capture_generation(self, **kwargs):
        messages = []

        async def llm(value):
            messages.extend(value)
            return SimpleNamespace(content="SELECT name FROM thing")

        self.tool._llm_call_with_retry = llm
        result = await self.tool.generate_sql("List names", **kwargs)
        self.assertEqual(result["status"], "success")
        return messages

    async def test_routed_moderate_keeps_strict_rules_but_simple_does_not(self):
        core.FEAT_29 = True
        basic = await self.capture_generation(complexity="moderate")
        core.FEAT_14 = True
        strict = await self.capture_generation(complexity="moderate")
        simple_strict = await self.capture_generation(complexity="simple")
        core.FEAT_14 = False
        simple_basic = await self.capture_generation(complexity="simple")
        self.assertNotEqual(basic[-1].content, strict[-1].content)
        self.assertEqual(simple_strict[-1].content, simple_basic[-1].content)

    async def test_pruning_adds_context_without_removing_full_system_schema(self):
        core.FEAT_25 = True

        async def prune(question):
            return "REDUCED SCHEMA"

        self.tool._prune_schema_for_question = prune
        messages = await self.capture_generation()
        self.assertIn("FULL SCHEMA", messages[0].content)
        self.assertIn("REDUCED SCHEMA", messages[1].content)

    async def test_voting_drops_computed_decomposition_and_bypasses_judge_retry(self):
        core.FEAT_28 = core.FEAT_30 = core.FEAT_17 = core.FEAT_20 = core.FEAT_27 = True
        calls = []

        async def check(question):
            return {"status": "success", "ambiguous": False}

        async def decompose(question):
            calls.append("decomposition_computed")
            return "PLAN"

        async def generate(question, **kwargs):
            calls.append(kwargs)
            return {"status": "success", "sql_query": "SELECT name FROM thing"}

        async def judge(*args):
            raise AssertionError("Voting unexpectedly used the judge")

        self.tool._check_ambiguity = check
        self.tool._decompose_question = decompose
        self.tool.generate_sql = generate
        self.tool._verify_sql_against_query = judge
        result = await self.tool.query("List names")
        self.assertEqual(result["status"], "success")
        self.assertEqual(calls, ["decomposition_computed", {"complexity": None},
                                 {"complexity": None}, {"complexity": None}])

    async def test_taxonomy_ignores_relationship_statistics_and_sql_system_enrichment(self):
        prompts = []

        async def llm(messages):
            prompts.append([m.content for m in messages])
            return SimpleNamespace(content='{"is_ambiguous": false}')

        self.tool._llm_call_with_retry = llm
        await self.tool._detect_ambiguity_by_taxonomy("List names")
        self.tool.system_prompt = "OTHER DESCRIPTIONS AND MINHASH"
        self.tool.relationships_str = "OTHER FK"
        self.tool.column_stats_str = "OTHER STATS"
        await self.tool._detect_ambiguity_by_taxonomy("List names")
        self.assertEqual(prompts[0], prompts[1])


if __name__ == "__main__":
    unittest.main()
