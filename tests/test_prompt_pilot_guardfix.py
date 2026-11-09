import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prompt_pilot as base
import prompt_pilot_guardfix as fix


def manifest():
    return {
        "seed_instruction": base.slot_parts()[1],
        "cohorts": {
            "fit": [
                {
                    "db_id": "training_database",
                    "question": "What is the total number of entries in this particular table?",
                }
            ],
            "inner_val": [],
        },
    }


class GuardRevisionTests(unittest.TestCase):
    def test_normal_rules_mentioning_select_and_from_are_allowed(self):
        text = (
            "5. Для SELECT с ORDER BY используй GROUP BY вместо DISTINCT, либо добавь выражение в SELECT.\n"
            "6. Каждый alias в SELECT/WHERE/ORDER BY должен быть объявлен в FROM или JOIN этого SELECT-блока."
        )
        with self.assertRaises(ValueError):
            base.check_candidate(text, manifest())
        self.assertEqual(fix.corrected_check(text, manifest()), text)

    def test_all_four_real_v1_proposals_pass_corrected_guard(self):
        path = base.ROOT / base.DEFAULT_OUTPUT / "search/raw_responses.jsonl"
        if not path.exists():
            self.skipTest("Local response logs are intentionally excluded from Git")
        base.load_gepa()
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        m = base.search.read(path.parent.parent / "experiment.json")
        teachers = [
            json.loads(s)
            for s in path.read_text(encoding="utf-8").splitlines()
            if json.loads(s).get("phase") == "teacher"
        ]
        self.assertEqual(len(teachers), 4)
        for teacher in teachers:
            instruction = InstructionProposalSignature.output_extractor(
                teacher["raw_response"]
            )["new_instruction"]
            fix.corrected_check(instruction, m)

    def test_real_sql_snippets_still_rejected(self):
        for text in (
            "SELECT id FROM entries",
            "Use SELECT id FROM entries.",
            '1. SELECT "id" FROM "entries";',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                fix.corrected_check(text, manifest())

    def test_protected_slots_and_training_specific_text_remain_rejected(self):
        for text in (
            "Use {user_query}",
            "Always use training_database",
            "What is the total number of entries in this particular table?",
            "```SELECT 1```",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                fix.corrected_check(text, manifest())

    def test_seed_compiler_stays_exact_noop_and_patch_does_not_escape(self):
        original = base.check_candidate
        with fix.revision():
            self.assertIs(base.check_candidate, fix.corrected_check)
            self.assertEqual(
                base.compile_template(base.slot_parts()[1], "hybrid"),
                base.templates()["SQL_PROMPT_TEMPLATE"],
            )
            self.assertNotEqual(base.DEFAULT_OUTPUT, fix.ORIGINAL_OUTPUT)
        self.assertIs(base.check_candidate, original)

    def test_original_manifest_rejected_by_revised_entrypoint(self):
        p = base.ROOT / fix.ORIGINAL_OUTPUT / "experiment.json"
        if not p.exists():
            self.skipTest("Original pilot artifact absent")
        m = base.search.read(p)
        base.validate(m)
        with self.assertRaisesRegex(ValueError, "separate guardfix"):
            fix.validate(copy.deepcopy(m))


if __name__ == "__main__":
    unittest.main()
