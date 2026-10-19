"""CPU regression checks for the train-only, nested fitting controls."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from build_fit_subsets import (
    DEFAULT_SOURCE, MODES, causal_label_audit, check_isolation,
    choose_subsets, operation_class, unique_actor_records,
)
from protocol_pilot import check_output, read_records

ROOT = Path(__file__).resolve().parents[1]


class FitSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_records([DEFAULT_SOURCE])
        cls.subsets = choose_subsets(cls.source)

    def test_nested_same_data_for_both_lora_variants(self):
        ids = lambda name: {row["sample_id"] for row in self.subsets[name]}
        self.assertLessEqual(ids("fit9.train"), ids("fit96.train"))
        self.assertLessEqual(ids("fit96.quick12.train"), ids("fit96.train"))

    def test_expected_mode_and_json_denominators(self):
        for name, count in (("fit9.train", 9), ("fit96.train", 96), ("fit96.quick12.train", 12)):
            rows = self.subsets[name]
            self.assertEqual(len(rows), count)
            self.assertEqual(sum(row["mode"] in {"READ", "WRITE"} for row in rows), count * 2 // 3)
            for mode in MODES:
                self.assertEqual(sum(row["mode"] == mode for row in rows), count // 3)

    def test_actual_input_deduplication(self):
        for rows in self.subsets.values():
            self.assertEqual(len(rows), len({row["actor_input_digest"] for row in rows}))

    def test_write_coverage_is_explicit(self):
        for name in ("fit96.train", "fit96.quick12.train"):
            self.assertEqual({operation_class(row) for row in self.subsets[name] if row["mode"] == "WRITE"}, {"EMPTY", "WRITE", "KEEP", "CORRECT"})
        self.assertEqual({operation_class(row) for row in self.subsets["fit9.train"] if row["mode"] == "WRITE"}, {"WRITE", "KEEP", "CORRECT"})

    def test_all_existing_labels_pass_causal_and_protocol_checks(self):
        for row in self.source:
            causal_label_audit(row)
            self.assertTrue(check_output(row, row["target_text"], True)["passed"])

    def test_missing_eos_is_not_repaired_by_reference_match(self):
        for row in self.subsets["fit9.train"]:
            result = check_output(row, row["target_text"], False)
            self.assertFalse(result["eos"])
            self.assertFalse(result["passed"])

    def test_future_evidence_is_rejected(self):
        row = deepcopy(self.source[0])
        row["evidence"]["observed_until_step"] = row["decision_step"] + 1
        with self.assertRaisesRegex(ValueError, "future"):
            causal_label_audit(row)

    def test_wrong_role_binding_is_rejected(self):
        row = deepcopy(self.source[0])
        row["evaluation"]["target_side"] = "left" if row["evaluation"]["target_side"] == "right" else "right"
        with self.assertRaisesRegex(ValueError, "first requested"):
            causal_label_audit(row)

    def test_heldout_record_cannot_become_train(self):
        row = deepcopy(self.source[0])
        row["split"] = "val"
        with self.assertRaisesRegex(ValueError, "train labels"):
            causal_label_audit(row)
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_isolation(self.source, [self.source[0]])

    def test_conflicting_target_same_actor_input_rejected(self):
        row = deepcopy(self.source[0])
        row["target_text"] += " inconsistent"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            unique_actor_records([self.source[0], row])

    def test_deterministic_selection_and_source_not_mutated(self):
        before = json.dumps(self.source, sort_keys=True)
        repeated = choose_subsets(self.source)
        self.assertEqual(self.subsets, repeated)
        self.assertEqual(before, json.dumps(self.source, sort_keys=True))

    def test_absolute_permanent_observation_references(self):
        for rows in self.subsets.values():
            for row in rows:
                path = Path(row["observation"]["npz_path"])
                self.assertTrue(path.is_absolute())
                self.assertTrue(path.is_file())
                self.assertNotIn("/tmp/", str(path))


if __name__ == "__main__":
    unittest.main()
