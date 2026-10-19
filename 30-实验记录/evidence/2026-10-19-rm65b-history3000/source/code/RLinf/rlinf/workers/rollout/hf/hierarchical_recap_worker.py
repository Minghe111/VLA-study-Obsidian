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

"""Serial RM65-B hierarchy with one actor and episode-owned memory."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rlinf.data.schema.history_events import (
    HISTORY_VERSION,
    control_decision_id,
    history_from_context,
    history_from_decisions,
    normalize_history,
)


@dataclasses.dataclass(frozen=True)
class GeneratedText:
    """A complete actor response, including the generated token IDs."""

    text: str
    token_ids: list[int]
    ended_with_eos: bool = True


class HierarchicalActor(Protocol):
    """The text and motor methods must use the same actor instance."""

    version: str

    def generate(
        self, observation: dict, mode: str, context: dict
    ) -> GeneratedText: ...

    def actions(self, observation: dict, context: dict) -> np.ndarray: ...


def _object(text: str, expected: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{expected} must be one JSON object, got {text!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{expected} must be an object, got {type(value).__name__}")
    return value


def _complete(result: GeneratedText, mode: str) -> GeneratedText:
    if not result.ended_with_eos or not result.text.strip():
        raise ValueError(
            f"{mode} generation must produce nonempty text ending with EOS"
        )
    return result


def validate_action_receipt(info: dict, proposal: np.ndarray, duration: int) -> dict:
    """Keep issued targets, actual drive commands and measured observations distinct."""
    records = info.get("action_trace")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(
            "One executed K1 command requires exactly one native action_trace receipt"
        )
    receipt = records[0]
    raw = np.asarray(receipt.get("raw_model_action24"), dtype=np.float32)
    final = np.asarray(receipt.get("actual_final_drive_command24"), dtype=np.float32)
    if raw.shape != (24,) or not np.array_equal(raw, proposal[0]):
        raise ValueError("Native raw_model_action24 must equal the issued proposal[0]")
    if final.shape != (24,) or not np.isfinite(final).all():
        raise ValueError("Native final drive command must contain 24 finite values")
    if receipt.get("executed_physics_steps") != duration:
        raise ValueError("Native receipt duration differs from actual physics clock")
    if not isinstance(receipt.get("endpoint_command_written"), bool):
        raise ValueError(
            "Native receipt must explicitly report endpoint_command_written"
        )
    if receipt["endpoint_command_written"] != (duration == 15):
        raise ValueError(
            "Native endpoint flag disagrees with full versus partial 15-tick control"
        )
    return receipt


class ObservationStore:
    """Persist RGB and measured state outside the JSON episode manifest."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)

    def write(self, index: int, observation: dict) -> dict:
        cameras = ("full_image", "left_wrist_image", "right_wrist_image")
        data = {}
        for name in cameras:
            value = np.asarray(observation[name])
            if value.shape != (480, 640, 3) or value.dtype != np.uint8:
                raise ValueError(
                    f"Expected {name} uint8 RGB[480,640,3], got {value.shape}/{value.dtype}"
                )
            data[name] = value
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (24,) or not np.isfinite(state).all():
            raise ValueError("Expected 24 finite measured SDK coordinates")
        data["state"] = state
        path = self.root / f"{index:06d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)
        return {
            "npz_path": str(path.resolve()),
            "instruction": observation.get("instruction", ""),
        }


class HierarchicalReCapRolloutWorker:
    """Own memory and advance exactly one physical control per motor call.

    The environment has one slot and never auto-resets. High-level failures
    stop collection before a motor command; they are not replaced by a library
    subtask or counted as successful episodes.
    """

    def __init__(
        self,
        actor: HierarchicalActor,
        env: Any,
        catalogue: Any,
        *,
        high_level_window: int = 32,
        max_steps: int = 100,
        history_num_frames: int = 1,
        history_stride_steps: int = 8,
        strict_memory_protocol: bool = False,
    ):
        if high_level_window < 1 or max_steps < 1:
            raise ValueError("high_level_window and max_steps must be positive")
        self.actor, self.env, self.catalogue = actor, env, catalogue
        self.high_level_window, self.max_steps = high_level_window, max_steps
        if history_num_frames < 1 or history_stride_steps < 1:
            raise ValueError("MEM frame count and stride must be positive")
        self.history_num_frames = history_num_frames
        self.history_stride_steps = history_stride_steps
        self.strict_memory_protocol = strict_memory_protocol

    def _decision(self, observation: dict, memory: Any, step: int, history: Any):
        from rlinf.data.schema.hierarchical_types import HighLevelDecision

        history = normalize_history(history, observed_until_step=step)
        before = memory.critic_view()
        context = {
            "goal": observation["instruction"],
            "memory_index": memory.catalogue_view(),
            "history": history,
            "observed_until_step": step,
        }
        read = _complete(self.actor.generate(observation, "READ", context), "READ")
        read_obj = _object(read.text, "READ")
        if set(read_obj) != {"read_ids"} or not isinstance(read_obj["read_ids"], list):
            raise ValueError("READ schema must be {'read_ids': [slot IDs]}")
        read_ids = read_obj["read_ids"]
        if any(not isinstance(item, str) for item in read_ids) or len(
            set(read_ids)
        ) != len(read_ids):
            raise ValueError("READ slot IDs must be distinct strings")
        selected = memory.read(read_ids)
        context["visible_memory"] = selected
        write = _complete(self.actor.generate(observation, "WRITE", context), "WRITE")
        write_obj = _object(write.text, "WRITE")
        if set(write_obj) != {"operations"} or not isinstance(
            write_obj["operations"], list
        ):
            raise ValueError("WRITE schema must be {'operations': [memory operations]}")
        written_ids = []
        current_writes = []
        for operation in write_obj["operations"]:
            if not isinstance(operation, dict):
                raise ValueError("Each memory operation must be an object")
            current_writes.append(
                memory.apply(
                    operation,
                    source={
                        "kind": "actor_generated",
                        "step": step,
                        "policy_version": self.actor.version,
                    },
                )
            )
            written_ids.append(operation["slot_id"])
        actor_memory = memory.actor_view(read_ids, list(dict.fromkeys(written_ids)))
        context["visible_memory"] = actor_memory
        subtask = _complete(
            self.actor.generate(observation, "SUBTASK", context), "SUBTASK"
        )
        decision = HighLevelDecision(
            decision_id=control_decision_id(step),
            start_step=step,
            end_step=step,
            generated_subtask=subtask.text.strip(),
            generated_token_ids=subtask.token_ids,
            generated_read_text=read.text,
            generated_write_text=write.text,
            read_token_ids=read.token_ids,
            write_token_ids=write.token_ids,
            read_ids=read_ids,
            tool_inputs={
                "catalogue": context["memory_index"],
                "visible_history": history,
            },
            tool_outputs={"selected_memory": selected},
            memory_before=before,
            memory_after=memory.critic_view(),
            selected_memory=selected,
            current_writes=current_writes,
            actor_memory=actor_memory,
            policy_version=self.actor.version,
            versions={
                "actor": self.actor.version,
                **getattr(self.actor, "versions", {}),
                "memory": memory.version,
            },
        )
        return decision

    def collect_episode(
        self,
        output: Path,
        *,
        episode_id: str,
        goal: str,
        seed: int,
        evaluate: bool = False,
    ):
        """Collect one episode; callers must explicitly opt into simulation."""
        from rlinf.data.schema.hierarchical_types import (
            ActionStep,
            HierarchicalEpisode,
            save_episode_json,
        )
        from rlinf.data.storage.episode_memory import EpisodeMemory

        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        store = ObservationStore(output / "observations")
        memory = EpisodeMemory(
            self.catalogue,
            env_id="0",
            episode_id=episode_id,
            strict_protocol=self.strict_memory_protocol,
        )
        observations = self.env.reset(env_seeds=[seed])
        if len(observations) != 1:
            raise ValueError(
                "Serial hierarchical rollout requires exactly one environment"
            )
        observation = dict(observations[0])
        if (
            self.history_num_frames > 1
            and observation.get("instruction")
            and observation["instruction"] != goal
        ):
            raise ValueError(
                "MEM rollout goal must match the environment task instruction"
            )
        observation["instruction"] = goal
        previous = store.write(0, observation)
        history_buffer = None
        if self.history_num_frames > 1:
            from rlinf.data.storage.observation_history import CausalObservationHistory

            history_buffer = CausalObservationHistory(
                env_id="0",
                episode_id=episode_id,
                num_frames=self.history_num_frames,
                stride_steps=self.history_stride_steps,
            )
            previous.update(physics_step=0, env_id="0", episode_id=episode_id)
            history_buffer.append(
                observation, previous, physics_step=0, env_id="0", episode_id=episode_id
            )
            observation["mem_history"] = history_buffer.window()
            previous["mem_history"] = history_buffer.window(references=True)
        episode = HierarchicalEpisode(
            env_id="0",
            episode_id=episode_id,
            goal=goal,
            versions={
                "actor": self.actor.version,
                **getattr(self.actor, "versions", {}),
                "memory": memory.version,
            },
            metadata={
                "seed": seed,
                "control_period_s": 0.06,
                "physics_dt_s": 0.004,
                "high_level_window": self.high_level_window,
                "split": "heldout_eval" if evaluate else "train",
                "strict_memory_protocol": self.strict_memory_protocol,
                "mem_num_frames": self.history_num_frames,
                "mem_stride_steps": self.history_stride_steps,
                "history_version": HISTORY_VERSION,
                "history_policy": "all_prior_issued_windows_reject_over_budget_no_truncation",
            },
        )
        decision = None
        physics_count = 0
        try:
            for index in range(self.max_steps):
                if index % self.high_level_window == 0:
                    history = history_from_decisions(
                        episode.decisions, observed_until_step=index
                    )
                    decision = self._decision(observation, memory, index, history)
                    episode.decisions.append(decision)
                context = {
                    "goal": goal,
                    "subtask": decision.generated_subtask,
                    "visible_memory": decision.actor_memory,
                }
                proposal = np.asarray(
                    self.actor.actions(observation, context), dtype=np.float32
                )
                if (
                    proposal.ndim != 2
                    or proposal.shape[1] != 24
                    or not len(proposal)
                    or not np.isfinite(proposal).all()
                ):
                    raise ValueError(
                        f"Expected finite proposal[H,24], got {proposal.shape}"
                    )
                next_obs, rewards, terminated, truncated, infos = self.env.step(
                    proposal[None, :1]
                )
                info = infos[0]
                if info.get("executed_action_count") != 1 or not info.get(
                    "observation_valid", False
                ):
                    raise RuntimeError(
                        "Native backend did not execute one control with a readable observation"
                    )
                count = int(info["physics_step_count"])
                duration = count - physics_count
                physics_count = count
                natural_end = bool(np.asarray(terminated)[0, 0])
                if not 1 <= duration <= 15 or (duration != 15 and not natural_end):
                    raise RuntimeError(
                        f"Expected 15 physics steps, or an explicitly terminal partial control, got {duration}"
                    )
                receipt = validate_action_receipt(info, proposal, duration)
                following = dict(next_obs[0])
                following["instruction"] = goal
                current = store.write(index + 1, following)
                if history_buffer is not None:
                    current.update(
                        physics_step=physics_count, env_id="0", episode_id=episode_id
                    )
                    history_buffer.append(
                        following,
                        current,
                        physics_step=physics_count,
                        env_id="0",
                        episode_id=episode_id,
                    )
                    following["mem_history"] = history_buffer.window()
                    current["mem_history"] = history_buffer.window(references=True)
                done, timeout = (
                    bool(np.asarray(terminated)[0, 0]),
                    bool(np.asarray(truncated)[0, 0]),
                )
                limit = index + 1 == self.max_steps and not done
                episode.steps.append(
                    ActionStep(
                        step_index=index,
                        decision_id=decision.decision_id,
                        observation=previous,
                        next_observation=current,
                        proposal_actions=proposal.tolist(),
                        executed_action=proposal[0].tolist(),
                        physics_steps=duration,
                        env_reward=float(np.asarray(rewards)[0, 0]),
                        terminated=done,
                        truncated=timeout or limit,
                        termination_reason=str(
                            info.get("failure_reason")
                            or ("collector_time_limit" if limit else "")
                        ),
                        policy_version=self.actor.version,
                        condition_id=decision.decision_id,
                        versions=dict(episode.versions),
                        metadata={
                            "action_trace": info["action_trace"],
                            "actual_final_drive_command24": receipt[
                                "actual_final_drive_command24"
                            ],
                            "action_label_semantics": "issued_raw_command_not_measured_state",
                        },
                    )
                )
                decision.end_step = index + 1
                previous, observation = current, following
                if done or timeout or limit:
                    episode.metadata["success"] = bool(
                        info.get("episode_success", False)
                    )
                    break
            save_episode_json(output / "episode.json", episode)
        except Exception as exc:
            # Partial traces are evidence only and are never discovered as episode.json.
            episode.metadata["collection_error"] = f"{type(exc).__name__}: {exc}"
            with (output / "failed_episode.json").open("x", encoding="utf-8") as stream:
                json.dump(episode.to_dict(), stream, indent=2)
            raise
        return episode


class NativePi05Actor:
    """Use a native Torch actor for text and continuous action generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        version: str,
        max_new_tokens: int = 96,
        eos_token_id: int = 1,
        advantage_condition: bool = True,
        versions: dict[str, str] | None = None,
    ):
        self.model, self.tokenizer, self.version = model, tokenizer, version
        self.max_new_tokens, self.eos_token_id = max_new_tokens, eos_token_id
        self.advantage_condition = advantage_condition
        self.versions = dict(versions or {})

    @staticmethod
    def env_io(observation: dict, prompt: str) -> dict:
        return {
            "main_images": observation["full_image"][None],
            "wrist_images": np.stack(
                [observation["left_wrist_image"], observation["right_wrist_image"]]
            )[None],
            "states": np.asarray(observation["state"], dtype=np.float32)[None],
            "task_descriptions": [prompt],
        }

    def generate(self, observation: dict, mode: str, context: dict) -> GeneratedText:
        import torch

        from rlinf.models.embodiment.openpi.recap_tokenizer import (
            build_high_prefix,
            canonical_json,
        )

        observed_until = context.get("observed_until_step")
        history = history_from_context(context, observed_until_step=observed_until)
        if observed_until is None and any(event["type"] == "issued_subtask" for event in history):
            raise ValueError("Native typed history needs the actual observed control-step bound")
        base = self._base_observation(observation)
        prompt = build_high_prefix(
            mode=mode,
            task=context["goal"],
            state=base.state[0].detach().float().cpu().numpy(),
            visible_memory=canonical_json(context.get("visible_memory", {})),
            memory_index=canonical_json(context.get("memory_index", {})),
            history=history,
            advantage="positive" if self.advantage_condition else None,
        )
        ids = self.tokenizer.encode_prefix(prompt)
        maximum = self.model.max_token_len
        if len(ids) + self.max_new_tokens > maximum:
            raise ValueError(
                f"Text generation needs {len(ids) + self.max_new_tokens} tokens; budget is {maximum}"
            )
        token_ids = torch.as_tensor(ids, device=self.model.device, dtype=torch.long)[
            None
        ]
        obs = dataclasses.replace(
            base,
            tokenized_prompt=token_ids,
            tokenized_prompt_mask=torch.ones_like(token_ids, dtype=torch.bool),
        )
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate_text_tokens(
                obs, max_new_tokens=self.max_new_tokens, eos_token_id=self.eos_token_id
            )
        values = generated[0].detach().cpu().tolist()
        complete = self.eos_token_id in values
        if complete:
            values = values[: values.index(self.eos_token_id) + 1]
        return GeneratedText(self.tokenizer.decode_text(values), values, complete)

    def actions(self, observation: dict, context: dict) -> np.ndarray:
        import torch

        from rlinf.models.embodiment.openpi.recap_tokenizer import (
            build_action_prefix,
            canonical_json,
        )

        base = self._base_observation(observation)
        prompt = build_action_prefix(
            task=context["goal"],
            subtask=context["subtask"],
            state=base.state[0].detach().float().cpu().numpy(),
            visible_memory=canonical_json(context["visible_memory"]),
            advantage="positive" if self.advantage_condition else None,
        )
        self.model.eval()
        with torch.no_grad():
            ids = self.tokenizer.encode_prefix(prompt)
            tokens = torch.as_tensor(ids, device=self.model.device, dtype=torch.long)[
                None
            ]
            obs = dataclasses.replace(
                base,
                tokenized_prompt=tokens,
                tokenized_prompt_mask=torch.ones_like(tokens, dtype=torch.bool),
            )
            actions, _ = self.model._predict_eval(obs)
        return actions[0].detach().float().cpu().numpy()

    def _base_observation(self, observation: dict):
        from rlinf.models.embodiment.openpi.mem_observation import attach_mem_history

        base = self.model.env_obs_to_observation(self.env_io(observation, ""))
        return attach_mem_history(self.model, base, observation)
