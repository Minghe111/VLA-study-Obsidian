# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU control/recording boundaries; no optimizer, model load or simulator."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from rlinf.data.datasets.recap.hierarchical_dataset import build_high_level_samples
from rlinf.data.datasets.recap.hierarchical_value import episode_value_examples
from rlinf.data.schema.hierarchical_types import load_episode_json
from rlinf.runners.hierarchical_recap_runner import (
    HierarchicalReCapRunner,
    audit_config,
)
from rlinf.workers.rollout.hf.hierarchical_recap_worker import (
    GeneratedText,
    HierarchicalReCapRolloutWorker,
)
from rlinf.workers.sft.fsdp_hierarchical_recap_worker import (
    FSDPHierarchicalReCapWorker,
    IndependentValueTrainer,
)


class ExternalPolicy:
    """Scripted external policy edge, never claimed as a learned VLA."""

    version = "external_test_policy"

    def __init__(self):
        self.contexts = []
        self.completed = 0

    def generate(self, observation, mode, context):
        self.contexts.append((mode, context.copy()))
        if mode == "READ":
            return GeneratedText('{"read_ids": ["location"]}', [1])
        if mode == "WRITE":
            return (
                GeneratedText(
                    '{"operations":[{"op":"WRITE","slot_id":"location","value":"initial left side","confirmed":false}]}',
                    [2, 1],
                )
                if self.completed == 0
                else GeneratedText('{"operations":[]}', [1])
            )
        self.completed += 1
        return GeneratedText("Move the observed sample to its recorded side.", [3, 1])

    def actions(self, observation, context):
        return np.full((16, 24), 0.25, dtype=np.float32)


class ExternalClockedEnvironment:
    """Physical transport edge exposing recorded observations and clock counts."""

    def __init__(self):
        self.count = 0
        self.reset_count = 0
        self.actions = []

    @staticmethod
    def observation():
        return {
            "full_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "left_wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "right_wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "state": np.zeros(24, dtype=np.float32),
            "instruction": "sample task",
        }

    def reset(self, env_seeds):
        self.count = 0
        self.reset_count += 1
        return [self.observation()]

    def step(self, actions):
        assert actions.shape == (1, 1, 24)
        self.actions.append(actions.copy())
        self.count += 1
        done = self.count == 3
        return (
            [self.observation()],
            np.asarray([[float(done)]]),
            np.asarray([[done]]),
            np.asarray([[False]]),
            [
                {
                    "executed_action_count": 1,
                    "observation_valid": True,
                    "physics_step_count": self.count * 15,
                    "episode_success": done,
                    "action_trace": [
                        {
                            "raw_model_action24": actions[0, 0].tolist(),
                            "actual_final_drive_command24": actions[0, 0].tolist(),
                            "executed_physics_steps": 15,
                            "endpoint_command_written": True,
                        }
                    ],
                }
            ],
        )


def test_rollout_handoff_recording_and_memory_isolation(tmp_path):
    actor, env = ExternalPolicy(), ExternalClockedEnvironment()
    worker = HierarchicalReCapRolloutWorker(
        actor,
        env,
        {"location": "Observed original location"},
        high_level_window=2,
        max_steps=3,
    )
    episode = worker.collect_episode(
        tmp_path / "first", episode_id="first", goal="Return the sample", seed=4
    )
    restored = load_episode_json(tmp_path / "first" / "episode.json")
    assert restored.to_dict() == episode.to_dict()
    assert [item.end_step - item.start_step for item in restored.decisions] == [2, 1]
    assert all(step.physics_dt == 0.06 for step in restored.steps)
    assert len(restored.steps) == 3 and restored.metadata["success"]
    assert restored.decisions[0].memory_before["entries"] == {}
    assert restored.decisions[1].selected_memory[0]["value"] == "initial left side"
    assert len(env.actions) == 3
    # A fresh policy edge permits WRITE again; the controller must reset its bank.
    worker.actor = ExternalPolicy()
    second = worker.collect_episode(
        tmp_path / "second", episode_id="second", goal="Return the sample", seed=5
    )
    assert second.decisions[0].memory_before["entries"] == {}
    assert second.decisions[0].memory_before["episode_id"] == "second"
    episode.metadata["hierarchical_advantages"] = {
        "IH": [True, False],
        "IA": [True, False, True],
        "value_version": "value_test",
    }
    assert len(build_high_level_samples(episode)) == 6
    actions, high = episode_value_examples(episode)
    assert len(actions) == 3 and len(high) == 2
    assert len(actions[0]["images"]) == 3
    assert '"phase":"pre_read"' in high[0]["prompt"]


def test_unfinished_generation_cannot_advance_physics(tmp_path):
    class IncompletePolicy(ExternalPolicy):
        def generate(self, observation, mode, context):
            return GeneratedText('{"read_ids": []}', [8], ended_with_eos=False)

    env = ExternalClockedEnvironment()
    worker = HierarchicalReCapRolloutWorker(
        IncompletePolicy(), env, {"location": "Observed location"}
    )
    with pytest.raises(ValueError, match="ending with EOS"):
        worker.collect_episode(
            tmp_path / "incomplete", episode_id="i", goal="Return", seed=0
        )
    assert env.actions == []
    assert not (tmp_path / "incomplete" / "episode.json").exists()
    assert (tmp_path / "incomplete" / "failed_episode.json").exists()


def test_training_guards_run_before_any_model_or_optimizer_access(tmp_path):
    worker = FSDPHierarchicalReCapWorker(None, None)
    with pytest.raises(PermissionError, match="disabled"):
        worker.train([], max_updates=1)
    value = IndependentValueTrainer(None, None)
    with pytest.raises(PermissionError, match="explicit"):
        value.fit([], max_updates=1)
    runner = HierarchicalReCapRunner(None, tmp_path)
    with pytest.raises(PermissionError, match="optimizer"):
        runner.run_round(0)
    assert not (tmp_path / "round_0000").exists()


def test_audit_preserves_contract_without_running_assets():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/offline_rl/config/hierarchical_recap_rm65b.yaml"
    )
    config = yaml.safe_load(path.read_text())
    report = audit_config(config)
    assert report["config_valid"] and not report["execution_performed"]
    assert not report["ready_for_execution"] and not report["training_enabled"]
    assert not report["ready_for_asset_loading"]
    assert report["readiness_scope"] == "local_asset_paths_and_package_discovery_only"
    assert not report["model_capabilities_checked"]
    config["rollout"]["high_level_window"] = 8
    with pytest.raises(ValueError, match="H16"):
        audit_config(config)
    config["rollout"]["high_level_window"] = 32
    config["rollout"]["eval_seeds"] = config["rollout"]["train_seeds"]
    with pytest.raises(ValueError, match="disjoint"):
        audit_config(config)


def test_verified_sft_gate_and_collator_conditioning(tmp_path):
    from rlinf.data.datasets.recap.hierarchical_collator import (
        NativeJointCollator,
        training_advantage,
    )
    from rlinf.runners.hierarchical_recap_runner import prepare_actor_samples

    worker = HierarchicalReCapRolloutWorker(
        ExternalPolicy(),
        ExternalClockedEnvironment(),
        {"location": "Observed original location"},
        high_level_window=2,
        max_steps=3,
    )
    episode = worker.collect_episode(
        tmp_path / "sft", episode_id="sft", goal="Return the sample", seed=9
    )
    weights = {"high_ce": 1.0, "fast_ce": 0.0, "fm": 1.0}
    with pytest.raises(ValueError, match="Verified SFT"):
        prepare_actor_samples(
            [episode], training_mode="verified_sft", loss_weights=weights
        )
    episode.metadata["supervision_kind"] = "verified_sft"
    episode.metadata["provenance"] = " "
    with pytest.raises(ValueError, match="nonempty provenance"):
        prepare_actor_samples(
            [episode], training_mode="verified_sft", loss_weights=weights
        )
    episode.metadata["provenance"] = {
        "verified_by": "test_fixture",
        "evidence": "scripted policy edge contract only",
    }
    high, actions = prepare_actor_samples(
        [episode], training_mode="verified_sft", loss_weights=weights
    )
    assert high[0]["IH"] is None and actions[0]["IA"] is None
    assert training_advantage(high[0], "IH", training_mode="verified_sft") is None
    assert training_advantage(actions[0], "IA", training_mode="verified_sft") is None
    with pytest.raises(ValueError, match="explicit boolean"):
        training_advantage(high[0], "IH", training_mode="recap")
    assert training_advantage({"IH": False}, "IH", training_mode="recap") == "negative"
    assert training_advantage({"IA": True}, "IA", training_mode="recap") == "positive"
    assert (
        NativeJointCollator(None, None, training_mode="verified_sft").training_mode
        == "verified_sft"
    )
    with pytest.raises(ValueError, match="training_mode"):
        NativeJointCollator(None, None, training_mode="infer_from_rollout")
    with pytest.raises(ValueError, match="Unlabelled rollout"):
        prepare_actor_samples([episode], training_mode="recap", loss_weights=weights)
    episode.metadata["hierarchical_advantages"] = {
        "IH": [True, True],
        "IA": [True, True, True],
        "AH": [0.0, 0.0],
        "AA": [0.0, 0.0, 0.0],
        "value_version": "fitted_test",
    }
    with pytest.raises(ValueError, match="constant or missing"):
        prepare_actor_samples([episode], training_mode="recap", loss_weights=weights)


def test_cli_warm_start_requires_both_execution_and_training_flags(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "hierarchical_recap_entry",
        root / "examples/offline_rl/hierarchical_recap/run.py",
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    arguments = [
        "--config",
        str(root / "examples/offline_rl/config/hierarchical_recap_rm65b.yaml"),
        "--stage",
        "warm-start",
        "--output",
        str(tmp_path / "forbidden"),
    ]
    with pytest.raises(SystemExit) as missing_execution:
        entry.main(arguments)
    assert missing_execution.value.code == 2
    with pytest.raises(SystemExit) as missing_training:
        entry.main(arguments + ["--execute"])
    assert missing_training.value.code == 2
    assert not (tmp_path / "forbidden").exists()


def test_receipts_and_value_phase_context_are_causal(tmp_path):
    import json
    from copy import deepcopy

    from rlinf.workers.rollout.hf.hierarchical_recap_worker import (
        validate_action_receipt,
    )

    worker = HierarchicalReCapRolloutWorker(
        ExternalPolicy(),
        ExternalClockedEnvironment(),
        {"location": "Original side"},
        high_level_window=2,
        max_steps=3,
    )
    episode = worker.collect_episode(
        tmp_path / "receipt", episode_id="receipt", goal="Return", seed=3
    )
    receipt = episode.steps[0].metadata["action_trace"][0]
    assert receipt["raw_model_action24"] == episode.steps[0].proposal_actions[0]
    assert receipt["actual_final_drive_command24"] == [0.25] * 24
    with np.load(
        episode.steps[0].next_observation["npz_path"], allow_pickle=False
    ) as data:
        assert np.array_equal(data["state"], np.zeros(24))
    actions, high = episode_value_examples(episode)
    from rlinf.data.schema.history_events import history_from_decisions

    assert json.loads(high[1]["prompt"])["visible_history"] == history_from_decisions(
        episode.decisions[:1], observed_until_step=episode.decisions[1].start_step
    )
    assert (
        json.loads(actions[0]["prompt"])["actor_memory"]
        == episode.decisions[0].actor_memory
    )
    altered = deepcopy(episode)
    altered.decisions[0].memory_after["entries"]["location"]["value"] = (
        "fabricated future result"
    )
    with pytest.raises(ValueError, match="post-WRITE bank"):
        episode_value_examples(altered)
    bad_receipt = deepcopy(receipt)
    bad_receipt["raw_model_action24"][0] = 50.0
    with pytest.raises(ValueError, match="proposal"):
        validate_action_receipt(
            {"action_trace": [bad_receipt]},
            np.asarray(episode.steps[0].proposal_actions),
            15,
        )
    partial = deepcopy(receipt)
    partial["executed_physics_steps"] = 3
    partial["endpoint_command_written"] = False
    assert (
        validate_action_receipt(
            {"action_trace": [partial]},
            np.asarray(episode.steps[0].proposal_actions),
            3,
        )
        is partial
    )


def test_value_minimum_budget_covers_both_phases():
    from rlinf.runners.hierarchical_recap_runner import balanced_value_batches

    low, high = [{"value_phase": "action"}], [{"value_phase": "pre_read"}]
    assert [
        batch[0]["value_phase"]
        for batch in balanced_value_batches(low, high, batch_size=1, updates=2)
    ] == ["action", "pre_read"]
    assert {
        item["value_phase"]
        for item in next(
            iter(balanced_value_batches(low, high, batch_size=2, updates=1))
        )
    } == {"action", "pre_read"}
    with pytest.raises(ValueError, match="both phases"):
        list(balanced_value_batches(low, high, batch_size=1, updates=1))


def test_generated_runtime_and_training_prefixes_match(tmp_path):
    from rlinf.models.embodiment.openpi.recap_tokenizer import (
        build_high_prefix,
        canonical_json,
    )

    actor = ExternalPolicy()
    worker = HierarchicalReCapRolloutWorker(
        actor,
        ExternalClockedEnvironment(),
        {"location": "Original side"},
        high_level_window=2,
        max_steps=3,
    )
    episode = worker.collect_episode(
        tmp_path / "prompts", episode_id="prompts", goal="Return", seed=11
    )
    episode.metadata["hierarchical_advantages"] = {
        "IH": [True, True],
        "IA": [True] * 3,
        "value_version": "test_value",
    }
    normalized_state = np.linspace(-1, 1, 32)
    for sample, (mode, runtime) in zip(
        build_high_level_samples(episode), actor.contexts
    ):
        training = sample["context"]
        runtime_prompt = build_high_prefix(
            mode,
            task=runtime["goal"],
            state=normalized_state,
            visible_memory=canonical_json(runtime.get("visible_memory", {})),
            memory_index=canonical_json(runtime["memory_index"]),
            history=runtime["history"],
            advantage="positive",
        )
        training_prompt = build_high_prefix(
            mode,
            task=training["goal"],
            state=normalized_state,
            visible_memory=canonical_json(
                training.get("actor_memory", training.get("selected_memory", {}))
            ),
            memory_index=canonical_json(training["catalogue"]),
            history=training["visible_history"],
            advantage="positive",
        )
        assert runtime_prompt.encode("utf-8") == training_prompt.encode("utf-8")
        assert runtime_prompt.startswith("Task: Return, State: 0 ")


def test_cumulative_replay_excludes_eval_anchors_and_identity_collisions(tmp_path):
    from rlinf.data.schema.hierarchical_types import save_episode_json
    from rlinf.runners.hierarchical_recap_runner import (
        NativeHierarchicalBackend,
        asset_versions,
    )

    config_path = (
        Path(__file__).resolve().parents[2]
        / "examples/offline_rl/config/hierarchical_recap_rm65b.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    worker = HierarchicalReCapRolloutWorker(
        ExternalPolicy(),
        ExternalClockedEnvironment(),
        {"location": "Original side"},
        high_level_window=2,
        max_steps=3,
    )
    first = worker.collect_episode(
        tmp_path / "prior", episode_id="uuid_prior", goal="Return", seed=11
    )
    first_path = tmp_path / "prior" / "episode.json"
    backend = NativeHierarchicalBackend(config)
    assert backend._replay_paths([first_path]) == [first_path.resolve()]
    worker.actor = ExternalPolicy()
    worker.collect_episode(
        tmp_path / "fresh", episode_id="uuid_fresh", goal="Return", seed=12
    )
    assert len(backend._replay_paths([tmp_path / "fresh" / "episode.json"])) == 2
    duplicate = tmp_path / "same_identity.json"
    save_episode_json(duplicate, first)
    with pytest.raises(ValueError, match="same env_id and episode_id"):
        backend._replay_paths([duplicate])
    first.metadata["split"] = "heldout_eval"
    heldout = tmp_path / "heldout.json"
    save_episode_json(heldout, first)
    with pytest.raises(ValueError, match="Held-out"):
        NativeHierarchicalBackend(config)._replay_paths([heldout])
    first.metadata["split"] = "train"
    first.metadata["supervision_kind"] = "verified_sft"
    anchor = tmp_path / "anchor.json"
    save_episode_json(anchor, first)
    with pytest.raises(ValueError, match="separate buffer"):
        NativeHierarchicalBackend(config)._replay_paths([anchor])
    config["tokenizer"]["fast_processor_path"] = None
    versions = asset_versions(config)
    assert versions["fast_processor_sha256"] == "unavailable"
    assert versions["prompt"] == "rm65b_recap_prompt_v3_history_events"
    assert versions["schema"] == "rm65b_hier_v1"


def test_typed_history_full_accumulation_reset_and_replay_evidence(tmp_path):
    from copy import deepcopy
    from rlinf.data.schema.history_events import HISTORY_VERSION, history_from_decisions

    class SixControlEnvironment(ExternalClockedEnvironment):
        def step(self, actions):
            observation,reward,terminated,truncated,infos=super().step(actions)
            done=self.count==6
            terminated[:]=done
            infos[0]["episode_success"]=done
            return observation,reward,terminated,truncated,infos

    actor=ExternalPolicy()
    worker=HierarchicalReCapRolloutWorker(actor,SixControlEnvironment(),{"location":"Original side"},high_level_window=1,max_steps=6)
    episode=worker.collect_episode(tmp_path/'full_history',episode_id='full',goal='Return',seed=41)
    assert episode.metadata['history_version']==HISTORY_VERSION
    assert len(episode.decisions[-1].tool_inputs['visible_history'])==5
    for index,decision in enumerate(episode.decisions):
        history=decision.tool_inputs['visible_history']
        assert history==history_from_decisions(episode.decisions[:index],observed_until_step=decision.start_step)
        assert all(set(event)=={'type','text','decision_id','issued_at_step','executed_until_step'} for event in history)
    samples=build_high_level_samples(episode,require_labels=False)
    assert len(samples)==18
    tampered=deepcopy(episode)
    tampered.decisions[-1].tool_inputs['visible_history'][0]['text']='fabricated prior command'
    with pytest.raises(ValueError,match='prior actual decision windows'):
        build_high_level_samples(tampered,require_labels=False)
    with pytest.raises(ValueError,match='prior actual decision windows'):
        episode_value_examples(tampered)
    worker.actor=ExternalPolicy()
    next_episode=worker.collect_episode(tmp_path/'reset_history',episode_id='reset',goal='Return',seed=42)
    assert next_episode.decisions[0].tool_inputs['visible_history']==[]
