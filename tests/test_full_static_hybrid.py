import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

from atomic_artifacts import write_json  # noqa: E402
import full_static_hybrid as full  # noqa: E402
import full_static_hybrid_runtime as runtime  # noqa: E402
import static_hybrid_search as search  # noqa: E402


def manifest():
    return {
        "protocol_sha256": "test",
        "protocol": {
            "model": {"model": "fake"},
            "environment": {},
            "full_evaluation": {
                "version": full.VERSION,
                "profiles": copy.deepcopy(full.PROFILES),
                "expected_counts": {"bird": 241, "ambrosia": 211},
            },
        },
        "cases": [{"name": n, "features": f} for n, f in full.PROFILES.items()],
        "cohorts": {
            "full": {
                b: [
                    {"question_id": i, "db_id": "db", "question": "q"}
                    for i in range(count)
                ]
                for b, count in [("bird", 241), ("ambrosia", 211)]
            }
        },
        "confirmation_source": {
            "path": "research_results/confirmation",
            "files_sha256": {},
        },
    }


class FullManifestTests(unittest.TestCase):
    def test_profiles_are_four_unique_masks_and_counts_match_full(self):
        m = manifest()
        m["protocol"]["confirmation_source_sha256"] = search.fingerprint(
            m["confirmation_source"]
        )
        with (
            patch.object(full.guarded, "validate"),
            patch.object(full.guarded, "gate"),
            patch.object(full.campaign, "environment", return_value={}),
        ):
            full.validate(m)
            for mutate in [
                "duplicate_question",
                "fifth_profile",
                "changed_flag",
                "train_scope",
            ]:
                bad = copy.deepcopy(m)
                if mutate == "duplicate_question":
                    bad["cohorts"]["full"]["bird"][1]["question_id"] = 0
                elif mutate == "fifth_profile":
                    bad["cases"].append({"name": "h2", "features": [2, 14, 32, 35]})
                elif mutate == "changed_flag":
                    bad["cases"][0]["features"] = [1]
                else:
                    bad["cohorts"]["train"] = bad["cohorts"]["full"]
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    full.validate(bad)

    def test_output_must_be_in_workspace_results(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(full.campaign, "ROOT", Path(tmp)),
        ):
            with self.assertRaises(ValueError):
                full.checked_output(Path(tmp).parent)
            with self.assertRaises(ValueError):
                full.checked_output(Path(tmp) / "research_results")

    def test_completed_runs_are_skipped_without_starting_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(env_file="fake", stall_seconds=900, error_limit=8)
            with (
                patch.object(
                    full.search, "result", return_value={"status": "completed"}
                ),
                patch.object(full, "report"),
                patch.object(full.subprocess, "Popen") as child,
            ):
                full.run_jobs(Path(tmp), manifest(), args)
            child.assert_not_called()
            self.assertEqual(
                search.read(Path(tmp) / "suite_status.json")["status"], "completed"
            )

    def test_report_is_descriptive_and_contains_paired_comparisons(self):
        def scored(output, m, scope, case, b):
            self.assertEqual(scope, "full")
            return {
                "status": "completed",
                "saved": 2,
                "questions": 2,
                "correct": 1,
                "accuracy": 50.0,
                "outcomes": {"1": True, "2": False},
                "prompt_tokens": 10,
                "completion_tokens": 2,
            }

        with patch.object(full.search, "result", side_effect=scored):
            result = full.report(Path("unused"), manifest())
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("best_observed_dev", result)
        self.assertEqual(
            result["rows"][0]["comparison_H"]["bird"], {"rescued": 0, "harmed": 0}
        )


class FullRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_failure_stops_before_paid_runtime_and_restores_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            m = manifest()
            write_json(folder / "experiment.json", m)
            args = SimpleNamespace(
                search=str(folder),
                scope="full",
                case="baseline_control",
                benchmark="bird",
                env_file="fake",
            )
            old_preflight = runtime.guarded.validate_live_schemas
            old_recorder = runtime.legacy.RecordingClient

            async def before_paid(inner):
                self.assertEqual(inner.scope, "full")
                self.assertIs(
                    runtime.legacy.RecordingClient, runtime.TransportRecordingClient
                )
                runtime.guarded.validate_live_schemas(m["cohorts"]["full"]["bird"], {})

            with (
                patch.object(full, "checked_output", return_value=folder),
                patch.object(full, "validate"),
                patch.object(runtime.guarded, "run", side_effect=before_paid),
                patch.object(
                    runtime.resilient,
                    "validate_live_schemas",
                    side_effect=RuntimeError("catalog unavailable"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "catalog unavailable"):
                    await runtime.run(args)
            self.assertIs(runtime.guarded.validate_live_schemas, old_preflight)
            self.assertIs(runtime.legacy.RecordingClient, old_recorder)
            meta = search.read(folder / "full/baseline_control/bird/metadata.json")
            self.assertEqual(meta["status"], "preflight_failed")
            self.assertEqual(meta["saved"], 0)

    async def test_completed_checkpoint_has_no_preflight_or_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            m = manifest()
            write_json(folder / "experiment.json", m)
            run = folder / "full/baseline_control/bird"
            run.mkdir(parents=True)
            write_json(
                run / "metadata.json",
                {
                    "configuration": search.identity(m, "full", m["cases"][0], "bird"),
                    "status": "completed",
                },
            )
            args = SimpleNamespace(
                search=str(folder),
                scope="full",
                case="baseline_control",
                benchmark="bird",
            )
            with (
                patch.object(full, "checked_output", return_value=folder),
                patch.object(full, "validate"),
                patch.object(
                    runtime.search, "result", return_value={"status": "completed"}
                ),
                patch.object(runtime.guarded, "run", new_callable=AsyncMock) as paid,
            ):
                await runtime.run(args)
            paid.assert_not_called()


if __name__ == "__main__":
    unittest.main()
