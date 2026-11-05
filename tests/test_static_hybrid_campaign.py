import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import static_hybrid_campaign as campaign  # noqa: E402
import static_hybrid_search as search  # noqa: E402


def row(name, bird=60, ambrosia=70, tokens=100, flags=(14, 32, 35)):
    return {
        "configuration": name,
        "completed": True,
        "features": list(flags),
        "tokens": tokens,
        "macro_accuracy": (bird + ambrosia) / 2,
        "benchmarks": {"bird": {"accuracy": bird}, "ambrosia": {"accuracy": ambrosia}},
    }


def small_manifest():
    cohorts = {
        b: [
            {"question_id": "1", "db_id": "db_a"},
            {"question_id": "2", "db_id": "db_b"},
        ]
        for b in search.BENCHMARKS
    }
    return {
        "cases": campaign.registry(),
        "cohorts": {s: copy.deepcopy(cohorts) for s in search.SCOPES},
        "protocol_sha256": "frozen",
    }


class CampaignPlanningTests(unittest.TestCase):
    def test_bounded_unique_masks_and_separate_repeat_checkpoints(self):
        cases = campaign.registry()
        self.assertEqual(len(campaign.SCREEN), 15)
        self.assertEqual(len(cases), 45)
        self.assertEqual(len({c["name"] for c in cases}), 45)
        self.assertEqual(len({tuple(c["features"]) for c in cases}), 15)
        for name in campaign.SCREEN:
            values = [
                campaign.case_for({"cases": cases}, name, r) for r in campaign.ROUNDS
            ]
            self.assertTrue(all(c["features"] == values[0]["features"] for c in values))
            self.assertEqual([c["round"] for c in values], [1, 2, 3])
        for c in cases:
            if c["features"]:
                search.validate_features(c["features"])

    def test_baseline_explicitly_disables_all_36_flags(self):
        manifest = small_manifest()
        manifest["protocol"] = {"model": {"model": "stub"}}
        case = campaign.case_for(manifest, campaign.BASELINE, 1)
        config = search.identity(manifest, "train", case, "bird")
        self.assertEqual(len(config["feature_flags"]), 36)
        self.assertFalse(any(config["feature_flags"].values()))

    def test_next_phase_cannot_pay_before_fresh_controls_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = small_manifest()
            for phase in ("schema", "statistics", "repair", "interactions", "combine"):
                with self.assertRaisesRegex(ValueError, "BOTH benchmarks"):
                    campaign.jobs_for(Path(folder), manifest, phase, 1)
            scope, cases = campaign.jobs_for(Path(folder), manifest, "controls", 1)
            self.assertEqual(scope, "train")
            self.assertEqual(
                [c["configuration"] for c in cases], campaign.GROUPS["controls"]
            )

    def test_no_dev_or_repeat_without_frozen_train_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            for phase in ("repeat", "dev"):
                with self.assertRaisesRegex(ValueError, "freeze-finalists"):
                    campaign.jobs_for(Path(folder), small_manifest(), phase, 2)

    def test_round_order_changes_and_dev_requires_all_train_rounds(self):
        names = [
            campaign.BASELINE,
            campaign.CONTROL,
            "hybrid_plus_2",
            "core_without_tsv",
        ]
        with (
            patch.object(campaign, "finalists", return_value=names),
            patch.object(campaign, "require_complete") as required,
        ):
            orders = []
            for r in campaign.ROUNDS:
                scope, cases = campaign.jobs_for(
                    Path("unused"), small_manifest(), "dev", r
                )
                self.assertEqual(scope, "dev")
                orders.append([c["configuration"] for c in cases])
            self.assertNotEqual(orders[0], orders[1])
            self.assertNotEqual(orders[0], orders[2])
            self.assertTrue(
                any(
                    call.kwargs.get("rounds") == campaign.ROUNDS
                    for call in required.call_args_list
                )
            )

    def test_combination_phase_requires_every_preceding_screen_case(self):
        with patch.object(campaign, "require_complete") as required:
            scope, cases = campaign.jobs_for(
                Path("unused"), small_manifest(), "combine", 1
            )
            self.assertEqual(scope, "train")
            self.assertEqual(len(cases), 2)
            self.assertEqual(len(required.call_args_list[-1].args[3]), 13)

    def test_selection_keeps_benchmark_leaders_without_using_dev(self):
        rows = [
            row("macro", 75, 85),
            row("bird", 95, 30),
            row("ambrosia", 20, 99),
            row(campaign.BASELINE, 100, 100),
            row(campaign.CONTROL, 100, 100),
        ]
        self.assertEqual(campaign.choose_finalists(rows), ["macro", "bird", "ambrosia"])
        rows[1]["completed"] = False
        self.assertNotIn("bird", campaign.choose_finalists(rows))

    def test_exact_ties_use_cost_then_mask_size(self):
        rows = [
            row("expensive", tokens=300),
            row("cheap", tokens=100),
            row("smaller", tokens=100, flags=(14, 32)),
        ]
        self.assertEqual(campaign.choose_finalists(rows), ["smaller", "cheap"])

    def test_freeze_cannot_follow_existing_dev_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "dev").mkdir()
            with self.assertRaisesRegex(ValueError, "Dev artifacts"):
                campaign.freeze_finalists(output, small_manifest())

    def test_finalists_identity_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "finalists.json").write_text(
                json.dumps({"sha256": "wrong", "candidates": ["other"]})
            )
            with self.assertRaisesRegex(ValueError, "identity changed"):
                campaign.finalists(output, small_manifest())

    def test_freeze_is_idempotent_and_later_screen_changes_are_rejected(self):
        manifest = small_manifest()
        rows = {
            n: row(n, bird=60 + i, ambrosia=70) for i, n in enumerate(campaign.SCREEN)
        }
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with (
                patch.object(campaign, "require_complete"),
                patch.object(
                    campaign,
                    "measurement",
                    side_effect=lambda out, m, scope, n, r: rows[n],
                ),
                patch.object(
                    campaign, "screen_hashes", return_value={"checkpoint": "original"}
                ),
            ):
                chosen = campaign.freeze_finalists(output, manifest)
                content = (output / "finalists.json").read_bytes()
                self.assertEqual(campaign.freeze_finalists(output, manifest), chosen)
                self.assertEqual((output / "finalists.json").read_bytes(), content)
                with patch.object(
                    campaign, "screen_hashes", return_value={"checkpoint": "changed"}
                ):
                    with self.assertRaisesRegex(ValueError, "Screen results changed"):
                        campaign.finalists(output, manifest)

    def test_pending_report_does_not_rank_historical_or_partial_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            report = campaign.campaign_report(
                Path(folder), small_manifest(), write=False
            )
            self.assertEqual(report["selection_status"], "screening")
            self.assertEqual(len(report["screen"]), 15)
            self.assertFalse(any(v["completed"] for v in report["screen"]))
            self.assertEqual(report["replicated"], {"train": [], "dev": []})
            self.assertFalse(list(Path(folder).iterdir()))


class RepeatedComparisonTests(unittest.TestCase):
    def values(self, first, second):
        return [
            {
                "benchmarks": {
                    b: {"outcomes": {"1": first, "2": second}}
                    for b in search.BENCHMARKS
                }
            }
            for _ in range(3)
        ]

    def test_identical_repeated_answers_have_zero_interval(self):
        values = self.values(True, False)
        result = campaign.paired_repeats(
            values, values, small_manifest()["cohorts"]["dev"], samples=100
        )
        self.assertEqual(result["descriptive_95_interval_pp"], [0, 0])
        self.assertEqual(result["positive_rounds"], 0)
        self.assertEqual(result["database_clusters"], {"bird": 2, "ambrosia": 2})

    def test_all_questions_improved_have_paired_positive_delta(self):
        result = campaign.paired_repeats(
            self.values(True, True),
            self.values(False, False),
            small_manifest()["cohorts"]["train"],
            samples=100,
        )
        self.assertEqual(result["mean_macro_delta_pp"], 100)
        self.assertEqual(result["positive_rounds"], 3)
        self.assertEqual(result["descriptive_95_interval_pp"], [100, 100])

    def test_mismatched_ids_and_insufficient_repeats_are_rejected(self):
        values = self.values(True, False)
        cohorts = small_manifest()["cohorts"]["dev"]
        with self.assertRaisesRegex(ValueError, "Three matched"):
            campaign.paired_repeats(values[:1], values[:1], cohorts)
        damaged = copy.deepcopy(values)
        damaged[0]["benchmarks"]["bird"]["outcomes"]["foreign"] = True
        with self.assertRaisesRegex(ValueError, "frozen cohort IDs"):
            campaign.paired_repeats(damaged, values, cohorts)


if __name__ == "__main__":
    unittest.main()
