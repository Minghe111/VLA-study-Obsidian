# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""One causal history representation shared by labels, actor and controller.

Times are observed 60ms control-step boundaries, not physics ticks. An issued
subtask records its command and actual control window; it asserts no completion
or physical success. Legacy text is explicitly undated and never gains a clock.
No event is silently dropped, merged, reordered, summarized or truncated.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

HISTORY_VERSION = "rm65b_history_events_v1"
_ISSUED_FIELDS = {
    "type", "decision_id", "text", "issued_at_step", "executed_until_step"
}


def _step(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative control-step integer")
    return value


def control_decision_id(step: int) -> str:
    """Canonical model/runtime decision identity from an actual control step.

    Raw collection-log IDs remain provenance. A dataset builder may explicitly
    map a raw source window to this ID using its verified issued_at_step; this
    helper never edits raw logs or guesses a clock from an ID string.
    """
    return f"h{_step(step, 'decision step'):06d}"


def normalize_history(
    value: Any,
    *,
    observed_until_step: int | None = None,
    allow_legacy: bool = True,
) -> list[dict[str, Any]]:
    """Return a fresh, ordered event list, validating any available clock.

    None, empty text and [] all mean an empty history. Serialized JSON lists
    take the same path as lists. Nonempty old strings/list[str] become explicit
    legacy_text events; strict new labels must set allow_legacy=False.
    """
    bound = (
        None
        if observed_until_step is None
        else _step(observed_until_step, "observed_until_step")
    )
    if value is None:
        value = []
    elif isinstance(value, str):
        if not value.strip():
            value = []
        elif value.lstrip().startswith(("[", "{")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Serialized history must be one valid JSON list") from exc
        else:
            value = [value]
    if not isinstance(value, list):
        raise ValueError("History must be an ordered list of typed events")
    result = []
    ids: set[str] = set()
    previous_end = None
    for item in value:
        if isinstance(item, str):
            item = {"type": "legacy_text", "text": item}
        if not isinstance(item, Mapping):
            raise ValueError("Every history event must be an object")
        event = dict(item)
        if not isinstance(event.get("text"), str) or not event["text"].strip():
            raise ValueError("A history event needs nonempty text")
        if event.get("type") == "legacy_text":
            if not allow_legacy:
                raise ValueError("Strict history rejects undated legacy_text events")
            if set(event) != {"type", "text"}:
                raise ValueError("legacy_text cannot invent timing or outcome fields")
        elif event.get("type") == "issued_subtask":
            if set(event) != _ISSUED_FIELDS:
                raise ValueError("issued_subtask requires exactly its five causal fields")
            decision_id = event["decision_id"]
            if not isinstance(decision_id, str) or not decision_id.strip():
                raise ValueError("issued_subtask needs a nonempty decision_id")
            if decision_id in ids:
                raise ValueError("History decision_id values must not repeat")
            ids.add(decision_id)
            start = _step(event["issued_at_step"], "issued_at_step")
            end = _step(event["executed_until_step"], "executed_until_step")
            if end < start:
                raise ValueError("Executed history cannot end before it was issued")
            if previous_end is not None and start < previous_end:
                raise ValueError("History control windows must be ordered and nonoverlapping")
            if bound is not None and end > bound:
                raise ValueError("History cannot include execution after the current observation")
            previous_end = end
        else:
            raise ValueError("Unknown history event type")
        result.append(event)
    return result


def serialize_history(value: Any, **kwargs) -> str:
    """Serialize exactly the same normalized list at every model boundary."""
    return json.dumps(
        normalize_history(value, **kwargs),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def history_from_context(
    context: Mapping,
    *,
    observed_until_step: int | None = None,
    allow_legacy: bool = True,
) -> list[dict[str, Any]]:
    """Resolve old label/runtime aliases; conflicting inputs are an error."""
    if not isinstance(context, Mapping):
        raise ValueError("History context must be an object")
    if observed_until_step is None:
        observed_until_step = context.get("observed_until_step")
    options = dict(observed_until_step=observed_until_step, allow_legacy=allow_legacy)
    values = [
        normalize_history(context[key], **options)
        for key in ("visible_history", "history")
        if key in context
    ]
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError("visible_history and history disagree after normalization")
    return values[0] if values else []


def history_from_sample(record: Mapping) -> list[dict[str, Any]]:
    """Validate strict declared history at actual observation/decision bounds.

    Old undeclared labels retain legacy compatibility. Declared new records
    cannot downgrade to that path, omit their clock, or supply disagreeing
    bounds/aliases. The input record is never rewritten (sampler provenance).
    """
    metadata = record.get("metadata", {})
    context = record.get("context", {})
    if not isinstance(metadata, Mapping) or not isinstance(context, Mapping):
        raise ValueError("History metadata and context must be objects")
    versions = [scope["history_version"] for scope in (record, metadata, context) if "history_version" in scope]
    if versions and any(version != HISTORY_VERSION for version in versions):
        raise ValueError("Unknown or conflicting declared history_version")
    strict = bool(versions)
    decisions = [scope["decision_step"] for scope in (record, metadata, context) if "decision_step" in scope]
    observations = [scope["observed_until_step"] for scope in (record, metadata, context) if "observed_until_step" in scope]
    decision = decisions[0] if decisions else None
    observed = observations[0] if observations else None
    if strict:
        _step(decision, "decision_step")
        _step(observed, "observed_until_step")
        if any(type(step) is not int or step != decision for step in decisions):
            raise ValueError("Declared history decision-step bounds disagree")
        if any(type(step) is not int or step != observed for step in observations):
            raise ValueError("Declared history observation-step bounds disagree")
        if observed > decision:
            raise ValueError("Declared history observation cannot be after its decision")
        if not {"visible_history", "history"}.intersection(context):
            raise ValueError("Declared history requires an explicit event list, including []")
    bound = observed if observed is not None else decision
    history = history_from_context(context, observed_until_step=bound, allow_legacy=not strict)
    if strict and any(
        event["decision_id"] != control_decision_id(event["issued_at_step"])
        for event in history
    ):
        raise ValueError("Declared model history requires canonical control decision IDs")
    return history


def issued_subtask_event(
    *, decision_id: str, text: str, issued_at_step: int,
    executed_until_step: int, observed_until_step: int | None = None,
) -> dict[str, Any]:
    """Construct one factual issued-command event, never a completion label."""
    return normalize_history([{
        "type": "issued_subtask", "decision_id": decision_id, "text": text,
        "issued_at_step": issued_at_step, "executed_until_step": executed_until_step,
    }], observed_until_step=observed_until_step, allow_legacy=False)[0]


def history_from_decisions(
    decisions: list, *, observed_until_step: int,
) -> list[dict[str, Any]]:
    """Rebuild all prior actual decision windows; current/future windows fail."""
    def field(decision, key):
        return decision[key] if isinstance(decision, Mapping) else getattr(decision, key)
    return normalize_history([
        issued_subtask_event(
            decision_id=field(d, "decision_id"), text=field(d, "generated_subtask"),
            issued_at_step=field(d, "start_step"), executed_until_step=field(d, "end_step"),
            observed_until_step=observed_until_step,
        ) for d in decisions
    ], observed_until_step=observed_until_step, allow_legacy=False)
