"""CPU contracts only. No actor, simulator, renderer or GPU import."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SPEC = importlib.util.spec_from_file_location("capture_dynamic_protocol", Path(__file__).with_name("capture_dynamic_protocol.py"))
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def make_info(sent, previous_controls=0, previous_ticks=0, partial=None):
    count = len(sent) if partial is None else 1
    receipts = []
    cursor = previous_ticks
    for i, target in enumerate(sent[:count]):
        ticks = 15 if partial is None else partial
        receipts.append({"raw_model_action24": target.tolist(), "physics_step_start": cursor,
                         "physics_step_end": cursor + ticks, "requested_physics_steps": 15,
                         "executed_physics_steps": ticks, "actual_final_drive_command24": target.tolist(),
                         "endpoint_command_written": ticks == 15})
        cursor += ticks
    return {"executed_action_count": count, "valid_action_mask": np.arange(len(sent)) < count,
            "action_trace": receipts, "observation_valid": True, "episode_terminated": partial is not None,
            "episode_truncated": False, "step_count": previous_controls + count,
            "physics_step_count": cursor, "sim_time_s": cursor * 0.004,
            "failure_reason": None, "episode_metrics": {"verified_count": 0, "returned_count": 0,
                                                        "task_events": [], "success": False},
            "episode_contract": {"profile": "full50_sdk24"}}


class FakeSpec:
    @classmethod
    def from_mapping(cls, layout):
        return SimpleNamespace(canonical_instruction="same goal")


class FakeProcess:
    def is_alive(self):
        return False


class FakeEnv:
    def __init__(self, task, n, seeds):
        self.task, self.controls, self.ticks = task, 0, 0
        self.state = np.zeros(24, dtype=np.float32)
        self._processes = [FakeProcess()]
        self.closed = False

    def obs(self):
        x = np.tile(np.arange(640, dtype=np.uint16) % 256, (480, 1)).astype(np.uint8)
        image = np.stack([x, np.roll(x, self.controls, axis=1), x], axis=-1)
        return {**{k: image.copy() for k in capture.CAMERAS}, "state": self.state.copy(), "instruction": self.task["instruction"]}

    def get_obs(self):
        return [self.obs()]

    def step(self, batch):
        assert batch.ndim == 3 and batch.shape[0] == 1 and batch.shape[2] == 24
        sent = batch[0]
        info = make_info(sent, self.controls, self.ticks)
        self.controls, self.ticks = info["step_count"], info["physics_step_count"]
        self.state = sent[-1].copy()
        return [self.obs()], np.zeros((1, len(sent))), np.zeros((1, len(sent)), dtype=bool), np.zeros((1, len(sent)), dtype=bool), [info]

    def close(self, clear_cache=True):
        self.closed = True


class CaptureTests(unittest.TestCase):
    def reference(self):
        a = np.zeros((244, 24), dtype=np.float32)
        a[:, 12:18] = np.arange(244)[:, None] * 0.001
        a[:, 18:24] = 300 + np.arange(244)[:, None]
        return a

    def test_split_before_goal_and_unique_physical_layouts(self):
        families = capture.make_families()
        episodes = capture.episode_specs(families)
        self.assertEqual(len(episodes), 56)
        self.assertEqual({s: sum(e["split"] == s for e in episodes) for s in ("train", "val", "test")}, {"train": 32, "val": 12, "test": 12})
        for f in families:
            group = [e for e in episodes if e["family_id"] == f["family_id"]]
            self.assertEqual({e["target_side"] for e in group}, {"left", "right"})
            self.assertEqual({e["split"] for e in group}, {f["split"]})
        self.assertEqual(len({capture.digest(f["home_xy"]) for f in families}), 28)

    def test_exact_source_and_tail_hold_no_skip(self):
        initial = np.arange(24, dtype=np.float32)
        reference = self.reference()
        right, indices, name = capture.replay_actions(reference, initial, "right")
        np.testing.assert_array_equal(right[:244, 12:], reference[:, 12:])
        np.testing.assert_array_equal(right[:, :12], np.repeat(initial[None, :12], 256, axis=0))
        np.testing.assert_array_equal(right[244:], np.repeat(right[243:244], 12, axis=0))
        self.assertEqual(indices, list(range(244)) + [243] * 12)
        left, _, name = capture.replay_actions(reference, initial, "left")
        np.testing.assert_array_equal(left[:244, :6], -reference[:, 12:18])
        np.testing.assert_array_equal(left[:244, 6:12], reference[:, 18:24])
        np.testing.assert_array_equal(left[:, 12:], np.repeat(initial[None, 12:], 256, axis=0))
        self.assertIn("not_validated_expert", name)

    def test_causal_history_uses_only_executed_past_and_last_two(self):
        events = [capture.issue_event(i, "left") for i in (0, 32, 64, 96)]
        self.assertEqual(capture.history_at(events, 0, 0), [])
        at16 = capture.history_at(events, 16, 240)
        self.assertEqual(len(at16), 1)
        self.assertEqual(at16[0]["executed_until_step"], 16)
        at32 = capture.history_at(events, 32, 480)
        self.assertEqual(len(at32), 1)
        at80 = capture.history_at(events, 80, 1200)
        self.assertEqual(len(at80), 3)
        self.assertEqual(at80[-1]["executed_until_step"], 80)
        self.assertEqual(len(capture.history_text(at80).splitlines()), 2)
        self.assertTrue(all(e["issued_at_step"] < 80 and e["observed_until_step"] == 80 for e in at80))
        self.assertNotIn("verified", capture.history_text(at80))

    def test_clock_and_native_command_rejection(self):
        sent = np.arange(16 * 24, dtype=np.float32).reshape(16, 24)
        info = make_info(sent)
        self.assertEqual(capture.validate_chunk(info, sent, 0, 0), (16, 240))
        info["action_trace"][3]["raw_model_action24"][0] += 1
        with self.assertRaisesRegex(AssertionError, "differs"):
            capture.validate_chunk(info, sent, 0, 0)
        info = make_info(sent)
        info["action_trace"][4]["physics_step_start"] += 1
        with self.assertRaisesRegex(AssertionError, "discontinuous"):
            capture.validate_chunk(info, sent, 0, 0)

    def test_true_terminal_partial_preserved_no_padding(self):
        sent = np.zeros((16, 24), dtype=np.float32)
        info = make_info(sent, 32, 480, partial=3)
        self.assertEqual(capture.validate_chunk(info, sent, 32, 480), (1, 483))
        info["episode_terminated"] = False
        with self.assertRaisesRegex(AssertionError, "terminal"):
            capture.validate_chunk(info, sent, 32, 480)

    def test_source_loader_rejects_interior_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            rows = [{"index": i, "actions": [self.reference()[i].tolist()], "physics_ticks": 15} for i in range(244)]
            rows.append({"index": 244, "actions": [self.reference()[-1].tolist()], "physics_ticks": 3})
            path.write_text("\n".join(json.dumps(r) for r in rows))
            actions, receipt = capture.load_reference(path)
            self.assertEqual(actions.shape, (244, 24))
            self.assertEqual(receipt["excluded_partial_final_rows"], 1)
            rows[10]["physics_ticks"] = 3
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "final"):
                capture.load_reference(path)

    def test_plan_does_not_create_output_or_import_simulator(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "not_created"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(capture.main(["--plan", "--output", str(output), "--max-episodes", "2"]), 0)
            plan = json.loads(out.getvalue())
            self.assertFalse(output.exists())
            self.assertEqual(plan["expected_train_physical_contexts"], 288)
            self.assertEqual(plan["expected_full_episodes"], 56)
            self.assertFalse(plan["policy_loaded"])
            self.assertEqual(plan["mode"], "plan_only")

    def test_fake_episode_records_real_chunk_clock_history_and_close(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output=Path(temp), controls=32)
            spec = capture.episode_specs(capture.make_families(1, 1, 1))[0]
            folder = Path(temp) / "episode"
            result = capture.capture_episode(args, {}, spec, self.reference(), FakeEnv, FakeSpec, folder)
            self.assertEqual(result["executed_controls"], 32)
            self.assertEqual(result["physics_ticks"], 480)
            self.assertTrue(result["worker_shutdown_confirmed"])
            self.assertFalse(result["motor_supervision_eligible"])
            self.assertEqual([x["step"] for x in result["observations"]], [0, 32])
            self.assertEqual(result["observations"][0]["visible_history"], [])
            self.assertEqual(len(result["observations"][1]["visible_history"]), 1)
            rows = [json.loads(x) for x in (folder / "transitions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(np.asarray(rows[0]["issued_action_chunk"]).shape, (1, 16, 24))
            self.assertTrue(result["motion"]["dynamic_state_and_rgb_observed"])
            self.assertEqual(result["actual_monitor_summary"]["verified_count"], 0)
            self.assertNotIn("verified", result["observations"][-1]["history_text"])


if __name__ == "__main__":
    unittest.main()
