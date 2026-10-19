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

"""Local-only text/FAST tokenization and generated memory-operation prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.schema.history_events import serialize_history

PROMPT_VERSION = "rm65b_recap_prompt_v3_history_events"
TOKENIZER_VERSION = "paligemma_fast_reverse_v1"


def canonical_json(value: Any) -> str:
    """Serialize tool-visible structures identically in collection and training."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def quantized_state(state: Any) -> str:
    """Format normalized model-space state using native Pi0.5's 256 bins.

    The caller must first apply the checkpoint's state normalization and
    padding. Raw SDK24 state is not a valid input at this boundary.
    """
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    values = np.asarray(state)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected one finite normalized model-space state vector.")
    bins = np.digitize(values, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    return " ".join(map(str, bins))


def _native_header(task: str, state: Any) -> str:
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned}, State: {quantized_state(state)};\n"


def build_high_prefix(
    mode: str,
    *,
    task: str,
    state: Any,
    visible_memory: str = "",
    memory_index: str = "",
    history: Any = None,
    advantage: str | None = None,
) -> str:
    """Build a generated memory-operation/subtask context from normalized state."""
    mode = mode.upper()
    if mode not in {"READ", "WRITE", "SUBTASK"}:
        raise ValueError(f"Unknown memory generation mode: {mode!r}")
    if advantage not in {None, "positive", "negative"}:
        raise ValueError("Advantage must be positive, negative or omitted.")
    advantage_text = "" if advantage is None else f"Advantage: {advantage}\n"
    formats = {
        "READ": 'Return JSON {"read_ids":["slot_id",...]}.',
        "WRITE": 'Return JSON {"operations":[{"op":"WRITE|KEEP|CORRECT|REVOKE","slot_id":"id","value":"observed_fact","confirmed":false}]}. WRITE/CORRECT require value. Use an empty operations list for no write.',
        "SUBTASK": "Generate the next executable subtask as natural-language text. Do not return a skill identifier.",
    }
    return (
        _native_header(task, state)
        + f"Memory index: {memory_index}\nMemory: {visible_memory}\nHistory: {serialize_history(history)}\n"
        + f"{advantage_text}Mode: {mode}\n{formats[mode]}\nResponse: "
    )


def build_action_prefix(
    *,
    task: str,
    subtask: str,
    state: Any,
    visible_memory: str = "",
    advantage: str | None = None,
) -> str:
    """Extend the native Task/State format without any action answer tokens."""
    if not subtask.strip():
        raise ValueError("An action context requires a nonempty generated subtask.")
    if advantage not in {None, "positive", "negative"}:
        raise ValueError("Advantage must be positive, negative or omitted.")
    text = _native_header(task, state)
    text += f"Memory: {visible_memory}\nSubtask: {subtask}\n"
    return (
        text + ("" if advantage is None else f"Advantage: {advantage}\n") + "Action: "
    )


class RecapTokenizer:
    """Use an existing text tokenizer and an explicitly local FAST processor.

    Construction and text operations do not import Transformers or load FAST.
    ``actions`` passed to ``encode_fast`` are already normalized model-space
    values; raw SDK joints must never be encoded before their transforms.
    """

    def __init__(
        self,
        text_tokenizer: Any,
        *,
        fast_processor: Any = None,
        fast_processor_path: str | Path | None = None,
        max_prefix_tokens: int = 1024,
        max_target_tokens: int = 512,
        fast_skip_tokens: int = 128,
    ):
        self.text_tokenizer = text_tokenizer
        self.fast_processor = fast_processor
        self.fast_processor_path = (
            None if fast_processor_path is None else Path(fast_processor_path)
        )
        self.max_prefix_tokens = max_prefix_tokens
        self.max_target_tokens = max_target_tokens
        self.fast_skip_tokens = fast_skip_tokens
        if min(max_prefix_tokens, max_target_tokens) < 1 or fast_skip_tokens < 0:
            raise ValueError(
                "Token budgets must be positive and FAST skip nonnegative."
            )

    @classmethod
    def from_local_paths(cls, text_tokenizer_path, **kwargs):
        """Load SentencePiece only from the supplied local file."""
        path = Path(text_tokenizer_path)
        if not path.is_file():
            raise FileNotFoundError(f"Local text tokenizer not found: {path}")
        import sentencepiece

        return cls(sentencepiece.SentencePieceProcessor(model_file=str(path)), **kwargs)

    @property
    def vocab_size(self) -> int:
        value = self.text_tokenizer.vocab_size
        return int(value() if callable(value) else value)

    @property
    def eos_token_id(self) -> int:
        value = self.text_tokenizer.eos_id
        token = int(value() if callable(value) else value)
        if token < 0:
            raise ValueError("The local text tokenizer has no EOS id.")
        return token

    def encode_text(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> np.ndarray:
        tokens = np.asarray(
            self.text_tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos),
            dtype=np.int64,
        )
        if tokens.ndim != 1 or np.any(tokens < 0) or np.any(tokens >= self.vocab_size):
            raise ValueError("Text tokenizer returned invalid token IDs.")
        return tokens

    def encode_prefix(self, text: str) -> np.ndarray:
        tokens = self.encode_text(text, add_bos=True)
        if len(tokens) > self.max_prefix_tokens:
            raise ValueError(
                f"Prefix has {len(tokens)} tokens; budget is {self.max_prefix_tokens}. Truncation would hide memory or goals."
            )
        return tokens

    def encode_target(self, text: str) -> np.ndarray:
        tokens = self.encode_text(text, add_eos=True)
        if len(tokens) > self.max_target_tokens:
            raise ValueError(
                f"Target has {len(tokens)} tokens; budget is {self.max_target_tokens}."
            )
        return tokens

    def decode_text(self, tokens) -> str:
        return self.text_tokenizer.decode(np.asarray(tokens, dtype=np.int64).tolist())

    def _load_fast(self):
        if self.fast_processor is None:
            if (
                self.fast_processor_path is None
                or not self.fast_processor_path.is_dir()
            ):
                raise FileNotFoundError(
                    "FAST requires an existing local processor directory; automatic downloads are disabled."
                )
            from transformers import AutoProcessor

            self.fast_processor = AutoProcessor.from_pretrained(
                str(self.fast_processor_path),
                trust_remote_code=True,
                local_files_only=True,
            )
        return self.fast_processor

    def encode_fast(self, actions: np.ndarray) -> np.ndarray:
        """Map local FAST codes onto the official reserved PaliGemma IDs."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError(
                "FAST actions must be a finite normalized [horizon,model_action_dim] array."
            )
        processor = self._load_fast()
        raw = np.asarray(processor(actions[None])[0], dtype=np.int64)
        if raw.ndim != 1 or not len(raw) or np.any(raw < 0):
            raise ValueError("FAST returned an empty or invalid code sequence.")
        processor_vocab_size = getattr(processor, "vocab_size", None)
        if callable(processor_vocab_size):
            processor_vocab_size = processor_vocab_size()
        if processor_vocab_size is not None and np.any(
            raw >= int(processor_vocab_size)
        ):
            raise ValueError("FAST returned a code outside its vocabulary.")
        mapped = self.vocab_size - 1 - self.fast_skip_tokens - raw
        if np.any(mapped < 0) or np.any(
            mapped >= self.vocab_size - self.fast_skip_tokens
        ):
            raise ValueError(
                "FAST code cannot be mapped into the PaliGemma vocabulary."
            )
        result = np.concatenate([mapped, [self.eos_token_id]]).astype(np.int64)
        if len(result) > self.max_target_tokens:
            raise ValueError(
                f"FAST target has {len(result)} tokens; budget is {self.max_target_tokens}."
            )
        return result
