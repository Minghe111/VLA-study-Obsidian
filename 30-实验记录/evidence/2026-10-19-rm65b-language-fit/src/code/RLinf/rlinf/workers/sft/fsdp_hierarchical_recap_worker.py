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

"""Joint actor optimization with explicit permission and checkpoint boundaries."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Iterable


def capture_training_rng() -> dict:
    """Use weights-only-loadable containers for Python, NumPy and Torch RNGs."""
    import random
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "version": 1,
        "python": random.getstate(),
        "numpy": {"generator": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_training_rng(state: dict) -> None:
    """Restore after model/optimizer construction so initialization cannot perturb it."""
    import random
    import numpy as np
    import torch

    if state.get("version") != 1:
        raise ValueError("Unsupported training RNG checkpoint version")
    numpy_state = state["numpy"]
    cuda = state["torch_cuda"]
    if cuda is not None and len(cuda) != torch.cuda.device_count():
        raise ValueError("CUDA RNG checkpoint device count differs")
    # Validate the built-in generator states before mutating global state.
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state((
        numpy_state["generator"], np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    random.setstate(state["python"])
    np.random.set_state((
        numpy_state["generator"], np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"])
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


class FSDPHierarchicalReCapWorker:
    """Train a native joint actor, optionally already wrapped by FSDP.

    Construction never initializes a model, distributed runtime or optimizer.
    The local serial runner supplies a native actor. Distributed callers supply
    an FSDP wrapper and an optimizer built after wrapping.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        *,
        gradient_accumulation: int = 1,
        max_grad_norm: float = 1.0,
        scheduler: Any = None,
        enable_training: bool = False,
        sequential_objectives: bool = False,
        sampler: Any = None,
        strict_resume: bool = False,
    ):
        if gradient_accumulation < 1 or max_grad_norm <= 0:
            raise ValueError("Gradient accumulation and max_grad_norm must be positive")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.gradient_accumulation, self.max_grad_norm = (
            gradient_accumulation,
            max_grad_norm,
        )
        self.enable_training = enable_training
        self.sequential_objectives = sequential_objectives
        self.global_step = 0
        self.sampler, self.strict_resume = sampler, strict_resume
        self.resume_status = {"restored": False, "exact_resume": None}
        self._checkpoint_poisoned = False

    def _forward(self, batch: dict):
        # Calling forward preserves FSDP pre/post-forward hooks.
        return self.model(batch=batch)

    def train(self, batches: Iterable[dict], *, max_updates: int) -> dict[str, float]:
        """Refuse checkpointing after any failed, potentially consumed group."""
        if self._checkpoint_poisoned:
            raise RuntimeError("Failed actor training requires restoring a valid checkpoint")
        try:
            return self._train(batches, max_updates=max_updates)
        except Exception:
            # Prefetch/collation, forward and backward can advance sampler/RNG
            # before an optimizer update. Their partial state is not resumable.
            self._checkpoint_poisoned = True
            raise

    def _train(self, batches: Iterable[dict], *, max_updates: int) -> dict[str, float]:
        """Consume full accumulation groups and report separate objectives."""
        if not self.enable_training:
            raise PermissionError(
                "Actor training is disabled; explicit enable_training=True is required"
            )
        if max_updates < 1:
            raise ValueError("max_updates must be positive")
        import torch

        self.model.train()
        iterator = iter(batches)
        totals = {
            "loss": 0.0,
            "high_ce": 0.0,
            "fast_ce": 0.0,
            "fm": 0.0,
            "kv_anchor": 0.0,
            "grad_norm": 0.0,
        }
        updates = 0
        anchor_micro_batches = 0
        for _ in range(max_updates):
            group = []
            for _ in range(self.gradient_accumulation):
                try:
                    group.append(next(iterator))
                except StopIteration:
                    break
            if not group:
                break
            self.optimizer.zero_grad(set_to_none=True)
            try:
                for index, batch in enumerate(group):
                    anchor_micro_batches += int(
                        batch.get("supervision_source") == "verified_sft_anchor"
                    )
                    branches = [batch]
                    if self.sequential_objectives:
                        from rlinf.models.embodiment.openpi.tasks.recap_hier import (
                            Pi0RecapHier,
                        )

                        weights = Pi0RecapHier.enabled_loss_weights(batch)
                        branches = [
                            {**batch, "loss_weights": {name: value}}
                            for name, value in weights.items()
                            if value
                        ]
                    for branch_index, branch in enumerate(branches):
                        synchronize = (
                            index == len(group) - 1
                            and branch_index == len(branches) - 1
                        )
                        context = (
                            contextlib.nullcontext()
                            if synchronize or not hasattr(self.model, "no_sync")
                            else self.model.no_sync()
                        )
                        with context:
                            output = self._forward(branch)
                            loss = output["loss"]
                            if loss.ndim or not torch.isfinite(loss):
                                raise FloatingPointError(
                                    "Joint loss must be a finite scalar"
                                )
                            if not loss.requires_grad:
                                raise RuntimeError(
                                    "Enabled objective has no trainable gradient path"
                                )
                            (loss / len(group)).backward()
                        for name in ("loss", "high_ce", "fast_ce", "fm", "kv_anchor"):
                            value = output.get(name)
                            if value is not None:
                                totals[name] += float(value.detach()) / len(group)
                        del output, loss
                parameters = [p for p in self.model.parameters() if p.requires_grad]
                if not any(p.grad is not None for p in parameters):
                    raise RuntimeError(
                        "No trainable actor parameter received a gradient"
                    )
                if hasattr(self.model, "clip_grad_norm_"):
                    norm = self.model.clip_grad_norm_(self.max_grad_norm)
                else:
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters, self.max_grad_norm
                    )
                if not torch.isfinite(norm):
                    raise FloatingPointError(
                        "Actor gradient norm is not finite; optimizer step refused"
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                totals["grad_norm"] += float(norm)
                updates += 1
                self.global_step += 1
            except Exception:
                self.optimizer.zero_grad(set_to_none=True)
                raise
        if not updates:
            raise ValueError("Actor dataloader produced no training batches")
        return {
            **{name: value / updates for name, value in totals.items()},
            "updates": updates,
            "global_step": self.global_step,
            "anchor_micro_batches": anchor_micro_batches,
        }

    def save_checkpoint(
        self, directory: str | Path, *, version: str, metadata: dict | None = None
    ) -> Path:
        """Write a resumable local checkpoint; never overwrite a prior version."""
        if self._checkpoint_poisoned:
            raise RuntimeError("Cannot save an exact checkpoint after failed actor training/restore")
        import torch
        from torch.distributed.fsdp import FullyShardedDataParallel

        if isinstance(self.model, FullyShardedDataParallel):
            raise ValueError(
                "Use the RLinf FSDP checkpoint manager for distributed full-state saves"
            )
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        from rlinf.models.embodiment.openpi.recap_lora import recap_training_manifest

        training_manifest = recap_training_manifest(self.model)
        torch.save(self.model.state_dict(), destination / "full_weights.pt")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": None
                if self.scheduler is None
                else self.scheduler.state_dict(),
                "global_step": self.global_step,
                "training_manifest": training_manifest,
                "rng_state": capture_training_rng(),
                "sampler_state": None if self.sampler is None else self.sampler.state_dict(),
                "resume_state_version": 1,
                "batch_contract": {
                    "gradient_accumulation": self.gradient_accumulation,
                    "max_grad_norm": self.max_grad_norm,
                    "sequential_objectives": self.sequential_objectives,
                    "scheduler_enabled": self.scheduler is not None,
                },
            },
            destination / "trainer_state.pt",
        )
        with (destination / "version.json").open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "version": version,
                    "global_step": self.global_step,
                    "metadata": metadata or {},
                    "training_manifest": training_manifest,
                },
                stream,
                indent=2,
            )
        return destination

    def restore_trainer_state(self, checkpoint: Path) -> dict:
        """Restore optimizer state after the caller strictly reloads the actor."""
        import torch

        state = torch.load(
            Path(checkpoint) / "trainer_state.pt", map_location="cpu", weights_only=True
        )
        from rlinf.models.embodiment.openpi.recap_lora import recap_training_manifest

        self._checkpoint_poisoned = True
        expected = recap_training_manifest(self.model)
        saved_manifest = state.get("training_manifest")
        if saved_manifest is None and expected["training_stage"] != "legacy":
            raise RuntimeError("MEM optimizer resume requires a training manifest.")
        if saved_manifest is not None and saved_manifest != expected:
            raise RuntimeError(
                "Checkpoint training stage, MEM configuration or trainable parameters differ."
            )
        expected_contract = {
            "gradient_accumulation": self.gradient_accumulation,
            "max_grad_norm": self.max_grad_norm,
            "sequential_objectives": self.sequential_objectives,
            "scheduler_enabled": self.scheduler is not None,
        }
        if state.get("batch_contract") is not None and state["batch_contract"] != expected_contract:
            raise RuntimeError("Checkpoint worker accumulation/clipping/objective contract differs")
        missing = []
        resume_version = state.get("resume_state_version")
        if resume_version is None:
            missing.append("resume_state_version")
        elif type(resume_version) is not int or resume_version != 1:
            raise RuntimeError("Unsupported trainer resume_state_version")
        if state.get("rng_state") is None:
            missing.append("rng_state")
        if self.sampler is None or state.get("sampler_state") is None:
            missing.append("sampler_state")
        if state.get("batch_contract") is None:
            missing.append("batch_contract")
        if missing and self.strict_resume:
            raise RuntimeError("Exact resume refused; checkpoint lacks " + ", ".join(missing))
        if missing:
            import warnings
            warnings.warn(
                "NON-EXACT RESUME: sampler/RNG continuity cannot be claimed; missing "
                + ", ".join(missing), RuntimeWarning,
            )
        if self.sampler is not None and state.get("sampler_state") is not None:
            self.sampler.load_state_dict(state["sampler_state"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state["global_step"])
        if state.get("rng_state") is not None:
            restore_training_rng(state["rng_state"])
        self.resume_status = {
            "restored": True, "exact_resume": not missing,
            "missing": missing, "global_step": self.global_step,
            "resume_state_version": resume_version,
        }
        self._checkpoint_poisoned = False
        return self.resume_status


class IndependentValueTrainer:
    """Fit the existing categorical value model with disjoint parameters."""

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        *,
        actor: Any = None,
        enable_training: bool = False,
        max_grad_norm: float = 1.0,
    ):
        if actor is not None:
            actor_parameters = {id(value) for value in actor.parameters()}
            if any(id(value) in actor_parameters for value in model.parameters()):
                raise ValueError(
                    "Value model parameters must not alias actor parameters"
                )
        self.model, self.optimizer = model, optimizer
        self.enable_training, self.max_grad_norm = enable_training, max_grad_norm

    def fit(self, batches: Iterable[dict], *, max_updates: int) -> dict[str, float]:
        if not self.enable_training:
            raise PermissionError(
                "Value training requires explicit enable_training=True"
            )
        import torch

        self.model.train()
        count, loss_sum = 0, 0.0
        for batch in batches:
            if count >= max_updates:
                break
            self.optimizer.zero_grad(set_to_none=True)
            result = self.model(
                observation=batch["observation"], target_values=batch["target_values"]
            )
            loss = result.loss
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("Categorical value loss must be finite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            if not torch.isfinite(norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Value gradients are nonfinite")
            self.optimizer.step()
            loss_sum += float(loss.detach())
            count += 1
        if not count:
            raise ValueError("Value dataloader produced no training batches")
        return {"value_ce": loss_sum / count, "updates": count}
