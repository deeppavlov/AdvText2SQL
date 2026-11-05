import asyncio
from contextvars import ContextVar
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import static_hybrid_search as search  # noqa: E402
import static_hybrid_runtime as runtime  # noqa: E402
from run_openrouter_baseline import ProviderUnavailable  # noqa: E402


class PlanningTests(unittest.TestCase):
    def test_default_is_only_two_core_cases_and_repetition_requires_explicit_selection(self):
        cases = search.case_registry()
        self.assertEqual([c["features"] for c in search.selected_cases(cases)], [[14, 32], [32, 35]])
        self.assertEqual(len(search.selected_cases(cases, "all")), 20)
        self.assertEqual(search.selected_cases(cases, names=["hybrid_control_repeat"])[0]["features"], [14, 32, 35])
        for names in (["unknown"], ["core_without_tsv", "core_without_tsv"]):
            with self.assertRaises(ValueError):
                search.selected_cases(cases, names=names)

    def test_dependencies_workflow_bypasses_and_partial_prompt_overlap(self):
        for flags in ([3, 35], [27, 35], [17, 28], [28, 30], [6, 35], [32, 33], [15, 36]):
            with self.assertRaises(ValueError):
                search.validate_features(flags)
        for flags in ([14, 29], [2, 3, 4, 7, 35], [15, 16, 17, 20, 27, 35]):
            search.validate_features(flags)

    def test_hash_guards_cover_masks_cohorts_references_and_protocol(self):
        manifest = {"protocol": {"files_sha256": {}}, "cohorts": {"train": [1]},
                    "references": {"result": True}, "cases": [{"features": [14, 32]}]}
        for key, digest in (("protocol", "protocol_sha256"), ("cohorts", "cohorts_sha256"),
                            ("references", "references_sha256"), ("cases", "cases_sha256")):
            manifest[digest] = search.fingerprint(manifest[key])
        search.validate_manifest(manifest)
        for key in ("protocol", "cohorts", "references", "cases"):
            changed = copy.deepcopy(manifest)
            changed[key] = {"tampered": True}
            with self.assertRaises(ValueError):
                search.validate_manifest(changed)

    def test_benchmark_scope_and_mask_all_have_distinct_checkpoint_identities(self):
        case = {"name": "example", "features": [14, 32]}
        manifest = {"protocol_sha256": "hash", "protocol": {"model": "test"},
                    "cohorts": {scope: {b: [{"question_id": i}] for b in search.BENCHMARKS}
                                for i, scope in enumerate(search.SCOPES)}}
        identities = [search.identity(manifest, scope, case, b) for scope in search.SCOPES for b in search.BENCHMARKS]
        self.assertEqual(len({json.dumps(i, sort_keys=True) for i in identities}), 6)
        self.assertEqual(sum(identities[0]["feature_flags"].values()), 2)

    def test_train_dependent_probe_requires_parent_but_dev_finalist_does_not(self):
        cases = search.case_registry()
        manifest = {"cases": cases, "references": {"features": {"sql_ambiguity": []}}}
        selected = search.selected_cases(cases, names=["statistics_compact"])
        with patch.object(search, "result", return_value={"status": "pending"}):
            with self.assertRaisesRegex(ValueError, "hybrid_plus_2"):
                search.check_controls(ROOT, manifest, "train", selected, search.BENCHMARKS)
            search.check_controls(ROOT, manifest, "dev", selected, search.BENCHMARKS)
            ordered = search.selected_cases(cases, names=["hybrid_plus_2", "statistics_compact"])
            search.check_controls(ROOT, manifest, "train", ordered, search.BENCHMARKS)


class ScoringTests(unittest.TestCase):
    def test_ambrosia_errors_are_failed_and_blank_output_never_true_negative(self):
        from research_metrics import ambrosia_report
        items = [{"question_id": 1, "SQL": "SELECT name FROM t"}, {"question_id": 2, "SQL": "ambiguous"}]
        predictions = {"1": "error", "2": "ambiguous"}
        report = ambrosia_report(items, predictions)
        result = search.scored_predictions(items, predictions, report, "ambrosia")
        self.assertEqual((result["accuracy"], result["errors"]), (50, 1))
        self.assertEqual(result["ambiguity_metrics"]["tn"], 0)
        for invalid in ({"1": "", "2": "ambiguous"}, {"1": "SELECT 1"}):
            with self.assertRaises(ValueError):
                search.scored_predictions(items, invalid, report, "ambrosia")

    def test_bird_predictions_coverage_and_report_scores_must_agree(self):
        items = [{"question_id": 1}]
        predictions = {"1": "SELECT name FROM t"}
        row = {"question_id": 1, "predicted_sql": predictions["1"], "score": True}
        report = {"overall_accuracy": 100, "results": [row]}
        self.assertEqual(search.scored_predictions(items, predictions, report, "bird")["accuracy"], 100)
        for invalid in ({**report, "results": [row, row]}, {**report, "overall_accuracy": 0},
                        {**report, "results": [{**row, "predicted_sql": "different"}]}):
            with self.assertRaises(ValueError):
                search.scored_predictions(items, predictions, invalid, "bird")


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_only_queries_remaining_ids_and_preserves_checkpoint(self):
        called = []

        class Tool:
            async def query(self, question):
                called.append(question)
                return {"status": "success", "query": "SELECT name FROM t"}

        items = [{"question_id": 1, "question": "First", "evidence": "", "db_id": "a"},
                 {"question_id": 2, "question": "Second", "evidence": "hint", "db_id": "b"}]
        with tempfile.TemporaryDirectory(dir=ROOT / "research_results", prefix=".search_test_") as tmp:
            folder = Path(tmp)
            predictions = {"1": "OLD SQL"}
            await runtime.predict(items, {"b": Tool()}, predictions, folder, "bird", ContextVar("test"),
                                  lambda value: None, sleep=lambda _: asyncio.sleep(0))
            self.assertEqual(called, ["question: Second, evidence (may be empty): hint"])
            self.assertEqual(runtime.checkpoint(folder, {"1", "2"})["1"], "OLD SQL")
            with self.assertRaises(ValueError):
                runtime.checkpoint(folder, {"2"})

    async def test_terminal_provider_error_cannot_be_converted_to_a_question_label(self):
        class Client:
            async def create(self, *args, **kwargs):
                error = RuntimeError("fake terminal error")
                error.status_code = 402
                raise error

        with tempfile.TemporaryDirectory(dir=ROOT / "research_results", prefix=".search_test_") as tmp:
            folder = Path(tmp)
            recorder = runtime.RecordingClient(Client(), folder, {}, ContextVar("call", default={"phase": "probe"}))
            with self.assertRaises(ProviderUnavailable):
                await recorder.create([])
            self.assertFalse((folder / "query_results.json").exists())

    async def test_recorded_costs_include_stage_and_question_without_gold(self):
        class Client:
            async def create(self, *args, **kwargs):
                return SimpleNamespace(content="SELECT name FROM t", usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4))

        context = ContextVar("call", default={"phase": "question", "question_id": "2", "db_id": "a"})
        with tempfile.TemporaryDirectory(dir=ROOT / "research_results", prefix=".search_test_") as tmp:
            folder = Path(tmp)
            recorder = runtime.RecordingClient(Client(), folder, {}, context)
            await recorder.create([])
            row = json.loads((folder / "raw_responses.jsonl").read_text(encoding="utf-8"))
            self.assertEqual((row["phase"], row["question_id"], row["prompt_tokens"]), ("question", "2", 12))
            self.assertNotIn("SQL", row)
            self.assertEqual(context.get()["recorded_calls"], 1)

    async def test_completed_child_is_skipped_and_no_future_cases_run_automatically(self):
        # Exercise the real suite orchestration with fake subprocess/monitor; no SSH/API.
        cases = search.case_registry()
        manifest = {"cases": cases, "protocol": {"model": {"model": "fake"}},
                    "references": {"features": {"feat_14": [], "feat_32": [], "feat_35": [], "sql_ambiguity": []}},
                    "cohorts": {"train": {b: [1] for b in search.BENCHMARKS}}}
        with tempfile.TemporaryDirectory(dir=ROOT / "research_results", prefix=".search_test_") as tmp:
            argv = ["search", "--output", tmp]
            calls = []
            completed = set()

            def result(output, frozen, scope, case, benchmark):
                return {"status": "completed" if (case["name"], benchmark) in completed else "pending"}

            def start(command, cwd):
                name = command[command.index("--case")+1]
                benchmark = command[command.index("--benchmark")+1]
                calls.append((name, benchmark))
                completed.add((name, benchmark))
                return object()

            with patch.object(sys, "argv", argv), patch.object(search, "make_manifest", return_value=manifest), \
                    patch.object(search, "summary"), patch.object(search, "result", side_effect=result), \
                    patch.object(search.subprocess, "Popen", side_effect=start), patch.object(search, "monitor"):
                search.main()
            self.assertEqual(len(calls), 4)
            self.assertEqual({name for name, _ in calls}, {"core_without_tsv", "core_without_rules"})
            with patch.object(sys, "argv", argv), patch.object(search, "validate_manifest"), \
                    patch.object(search, "validate_connection"), patch.object(search, "summary"), \
                    patch.object(search, "dotenv_values", return_value={"LLM_MODEL_NAME": "fake", "LLM_MODEL_VERSION": "v", "LLM_BASE_URL": "url"}), \
                    patch.object(search, "result", side_effect=result), patch.object(search.subprocess, "Popen") as popen:
                frozen = search.read(Path(tmp)/"experiment.json")
                frozen["protocol"]["model"].update(model_version="v", base_url="url")
                search.write_json(Path(tmp)/"experiment.json", frozen)
                search.main()
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
