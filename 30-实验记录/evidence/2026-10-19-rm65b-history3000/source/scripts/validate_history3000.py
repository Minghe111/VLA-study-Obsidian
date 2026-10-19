"""Independent CPU acceptance of dynamic-history protocol supervision.

This file is owned by the independent audit agent. It never loads model weights,
runs inference, invokes a renderer, repairs labels, or changes input/source data.
Strict schema/source failures are collected rather than silently normalized away.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

# This validator is a CPU-only process, including native image transforms.
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
for _thread_env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_env] = '2'
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "RLinf"))
from rlinf.data.schema.history_events import (HISTORY_VERSION, history_from_context, history_from_sample,
                                              normalize_history, serialize_history)
from rlinf.data.storage.episode_memory import EpisodeMemory

VERSION = "rm65b_history3000_independent_acceptance_v2_runtime_ids"
MODES = ("READ", "WRITE", "SUBTASK")
CAMERAS = ("full_image", "left_wrist_image", "right_wrist_image")
EXPECTED = {"train": {"READ": 1000, "WRITE": 1000, "SUBTASK": 1000},
            "val": {"READ": 200, "WRITE": 200, "SUBTASK": 200},
            "test": {"READ": 200, "WRITE": 200, "SUBTASK": 200}}
ACTIVE_REVISIONS = (1, 2, 4, 7, 10)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.checks = Counter()

    def check(self, condition, code, detail, sample_id=None):
        self.checks[code] += 1
        if not condition:
            self.issues.append({"code": code, "sample_id": sample_id, "detail": detail})
        return bool(condition)

    def warning(self, code, detail):
        self.warnings.append({"code": code, "detail": detail})


def validate_slot(slot: dict, audit: Audit, sample_id: str):
    if not audit.check(isinstance(slot, dict), "memory_slot_object", str(type(slot)), sample_id):
        return
    fields = {"slot_id", "status", "value", "confirmed", "revision"}
    audit.check(set(slot) == fields, "memory_public_exact_fields", sorted(slot), sample_id)
    audit.check(isinstance(slot.get("slot_id"), str) and bool(slot.get("slot_id")),
                "memory_slot_id", slot, sample_id)
    audit.check(type(slot.get("confirmed")) is bool, "memory_confirmed_bool", slot, sample_id)
    r = slot.get("revision")
    audit.check(type(r) is int and r >= 0, "memory_revision_integer", slot, sample_id)
    if slot.get("status") == "empty":
        audit.check(slot.get("value") is None and slot.get("confirmed") is False and r == 0,
                    "memory_empty_invariant", slot, sample_id)
    elif slot.get("status") == "active":
        audit.check(isinstance(slot.get("value"), str) and bool(slot.get("value", "").strip())
                    and type(r) is int and r >= 1, "memory_active_invariant", slot, sample_id)
    else:
        audit.check(False, "memory_status_supported", slot, sample_id)


def validate_visible_memory(row: dict, audit: Audit):
    sid = row.get("sample_id")
    ctx = row.get("context", {})
    mode = row.get("mode")
    catalogue = ctx.get("catalogue")
    audit.check(isinstance(catalogue, list) and bool(catalogue)
                and all(isinstance(x, dict) and set(x) == {"slot_id", "description"}
                        for x in catalogue), "catalogue_schema", catalogue, sid)
    ids = [x.get("slot_id") for x in catalogue if isinstance(x, dict)] if isinstance(catalogue, list) else []
    audit.check(len(ids) == len(set(ids)), "catalogue_unique_ids", ids, sid)
    if mode == "READ":
        audit.check(not ctx.get("selected_memory") and not ctx.get("actor_memory"),
                    "read_no_hidden_memory", ctx, sid)
        return
    if mode == "WRITE":
        memory = ctx.get("selected_memory")
        audit.check(isinstance(memory, list), "write_selected_list", memory, sid)
    else:
        memory = ctx.get("actor_memory")
        if not audit.check(isinstance(memory, dict)
                           and set(memory) == {"selected", "current_writes"},
                           "subtask_native_nested_memory", memory, sid):
            return
        audit.check(all(isinstance(memory.get(k), list) for k in memory),
                    "actor_memory_list_values", memory, sid)
        memory = memory.get("selected", []) + memory.get("current_writes", [])
    for slot in memory or []:
        validate_slot(slot, audit, sid)
        audit.check(slot.get("slot_id") in ids, "memory_known_slot", slot.get("slot_id"), sid)


def strict_history(row: dict, audit: Audit):
    sid = row.get("sample_id")
    d = row.get("observed_until_step", row.get("decision_step"))
    ctx = row.get("context", {})
    versions = [scope['history_version'] for scope in (row, row.get('metadata', {}), ctx)
                if 'history_version' in scope]
    audit.check(bool(versions) and all(x == HISTORY_VERSION for x in versions),
                'new_data_explicit_history_version', versions, sid)
    audit.check(row.get('metadata', {}).get('history_id_encoding') == 'runtime_control_step_h6',
                'new_data_explicit_runtime_history_ID_encoding', row.get('metadata', {}).get('history_id_encoding'), sid)
    audit.check(type(d) is int and d >= 0, "observed_step_integer", d, sid)
    try:
        history = history_from_context(ctx, observed_until_step=d, allow_legacy=False)
        for event in history:
            # Independent formatting from the real control step. Deliberately
            # do not use the builder/shared runtime-ID helper as the oracle.
            expected_id = f"h{event['issued_at_step']:06d}"
            audit.check(event['decision_id'] == expected_id,
                        'label_history_matches_runtime_control_step_ID',
                        {'actual':event['decision_id'],'expected':expected_id}, sid)
        validated_sample_history = history_from_sample(row)
        audit.check(history == validated_sample_history, 'actual_loader_sample_history_matches', sid, sid)
    except (ValueError, TypeError) as exc:
        audit.check(False, "strict_history", str(exc), sid)
        return []
    audit.check(ctx.get("visible_history", ctx.get("history")) == history,
                "history_stored_canonical_list", ctx.get("visible_history", ctx.get("history")), sid)
    return history


def compare_actual_source_history(row, decisions, controls, audit, *, source_snapshot_history=None):
    """Reconstruct source prefix at d without trusting stored/sample history.

    Decisions are actually issued source events. A partial current window is
    clipped to already observed controls; no complete/success statement exists.
    """
    sid = row.get("sample_id")
    d = row["observed_until_step"]
    expected = []
    expected_raw = []
    expected_id_map = []
    by_step = {x["step_index"]: x for x in controls}
    audit.check(len(by_step) == len(controls), "source_unique_control_steps", len(controls), sid)
    for dec in decisions:
        start, end = dec["start_step"], dec["end_step"]
        if start >= d:
            continue
        end = min(end, d)
        span = list(range(start, end))
        audit.check(all(k in by_step for k in span), "source_history_controls_exist", [start, end], sid)
        audit.check(all(by_step[k].get("decision_id") == dec["decision_id"]
                        for k in span if k in by_step),
                    "source_history_control_decision_match", dec["decision_id"], sid)
        expected_raw.append({"type": "issued_subtask", "decision_id": dec["decision_id"],
                         "text": dec["generated_subtask"], "issued_at_step": start,
                         "executed_until_step": end})
        canonical_id = f"h{start:06d}"
        expected.append({"type": "issued_subtask", "decision_id": canonical_id,
                         "text": dec["generated_subtask"], "issued_at_step": start,
                         "executed_until_step": end})
        expected_id_map.append({'source_event_id':dec['decision_id'],
                                'decision_id':canonical_id,'issued_at_step':start})
    if source_snapshot_history is not None:
        projected = [{'type':event['type'],'decision_id':event['event_id'],
                      'text':event['text'],'issued_at_step':event['issued_at_step'],
                      'executed_until_step':event['executed_until_step']}
                     for event in source_snapshot_history]
        audit.check(projected == expected_raw, 'label_provenance_raw_snapshot_matches_raw_issued_log',
                    {'raw_log_prefix':expected_raw,'raw_snapshot':projected}, sid)
    audit.check(row.get('evidence', {}).get('source_history_id_map') == expected_id_map,
                'label_ID_map_exact_raw_source_event_to_runtime_control_step',
                {'expected':expected_id_map,'actual':row.get('evidence', {}).get('source_history_id_map')}, sid)
    expected = normalize_history(expected, observed_until_step=d, allow_legacy=False)
    got = strict_history(row, audit)
    audit.check(got == expected, "history_exact_actual_source_prefix", {"expected": expected, "actual": got}, sid)
    return expected


def target_schema(row, audit):
    sid, mode = row.get("sample_id"), row.get("mode")
    target = row.get("target_text")
    audit.check(isinstance(target, str) and bool(target.strip()), "target_nonempty", target, sid)
    if mode == "SUBTASK":
        audit.check(isinstance(target, str) and not target.lstrip().startswith(("{", "[")),
                    "subtask_natural_text", target, sid)
        return None
    try:
        obj = json.loads(target)
    except (ValueError, TypeError) as exc:
        audit.check(False, "target_json", str(exc), sid)
        return None
    if mode == "READ":
        ids = obj.get("read_ids") if isinstance(obj, dict) else None
        known = {x["slot_id"] for x in row["context"]["catalogue"]}
        audit.check(isinstance(obj, dict) and set(obj) == {"read_ids"}
                    and isinstance(ids, list) and len(ids) == len(set(ids)) and set(ids) <= known,
                    "read_target_schema", obj, sid)
    else:
        ops = obj.get("operations") if isinstance(obj, dict) else None
        audit.check(isinstance(obj, dict) and set(obj) == {"operations"}
                    and isinstance(ops, list) and len(ops) <= 1,
                    "write_target_schema", obj, sid)
    return obj


def load_rows(paths):
    rows = []
    for path in paths:
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def source_receipt(path, expected_sha, audit, code, sid=None):
    path = Path(path)
    if not audit.check(path.is_file(), code + '_exists', str(path), sid):
        return False
    return audit.check(isinstance(expected_sha, str) and file_sha(path) == expected_sha,
                       code + '_sha256', str(path), sid)


def independent_command_spec(manifest, audit):
    """Read the active assets directly, without the collector's helpers."""
    import yaml
    identities = manifest['identities']
    cfg = Path(manifest['source_config'])
    source_receipt(cfg, identities['config_sha256'], audit, 'capture_config')
    asset_root = Path(yaml.safe_load(cfg.read_text())['environment']['task_config']['asset_root'])
    root = asset_root / 'embodiments/realman-rm65b-inspire-gen4'
    source_receipt(root / 'source_manifest.json', identities.get('command_manifest_sha256'),
                   audit, 'active_command_manifest')
    source_receipt(root / 'robot_sapien.urdf', identities.get('command_urdf_sha256'),
                   audit, 'active_command_URDF')
    meta = json.loads((root / 'source_manifest.json').read_text())
    joints = {x.attrib['name']: x for x in ET.parse(root / 'robot_sapien.urdf').getroot().findall('joint')}
    lower, upper = np.zeros(24), np.zeros(24)
    for side, offset, prefix in [('left', 0, 'l'), ('right', 12, 'r')]:
        for i in range(6):
            name = f'{prefix}_joint{i + 1}'
            lim = joints[name].find('limit').attrib
            lo, hi = float(lim['lower']), float(lim['upper'])
            audit.check(lo == meta['arm_limits'][name]['lower'] and hi == meta['arm_limits'][name]['upper'],
                        'active_arm_bounds_metadata_URDF_equal', name)
            lower[offset + i], upper[offset + i] = lo, hi
        for name in meta['hands'][side]['channels']:
            lim = joints[name].find('limit').attrib
            audit.check(all(float(lim[k]) == float(meta['hands'][side]['limits'][name][k])
                            for k in ['lower', 'upper']), 'active_hand_bounds_metadata_URDF_equal', name)
        lower[offset + 6:offset + 12], upper[offset + 6:offset + 12] = 0, 1000
    runtime_root = Path(identities['robotwin_root']) / 'integrations/realman_rm65b_gen4'
    for name, expected_sha in identities.get('runtime_source_sha256', {}).items():
        source_receipt(runtime_root / name, expected_sha, audit, 'capture_runtime_source')
    source_receipt(ROOT / 'scripts/capture_dynamic_protocol_v2.py', identities.get('script_sha256'),
                   audit, 'capture_v2_script')
    audit.check(identities.get('readback_validation') ==
                'explicit_native_hand_mapping_float32_storage_inverse_atol1e-9_rtol1e-12',
                'capture_readback_contract', identities.get('readback_validation'))
    return {'hands': meta['hands'], 'lower': lower, 'upper': upper}


def independent_drive_readback(target, command_spec):
    """Native command-space readback, not physical measured joint tracking."""
    target = np.asarray(target, dtype=np.float32)
    result = target.astype(np.float64).copy()
    for side, offset in [('left', 6), ('right', 18)]:
        hand = command_spec['hands'][side]
        for c, name in enumerate(hand['channels']):
            lo = float(hand['limits'][name]['lower'])
            span = float(hand['limits'][name]['upper']) - lo
            radians = lo + (1 - float(target[offset + c]) / 1000) * span
            stored = float(np.float32(radians))
            result[offset + c] = 1000 * (1 - (stored - lo) / span)
    return result


def audit_capture(manifest_path, audit, *, allow_partial=False):
    """Independently check the capture's raw receipts and per-control clocks."""
    from openpi.shared.image_tools import resize_with_pad
    manifest = json.loads(manifest_path.read_text())
    audit.check(manifest.get('schema') == 'rm65b_dynamic_protocol_capture_v2',
                'capture_schema', manifest.get('schema'))
    command_spec = independent_command_spec(manifest, audit)
    if allow_partial and manifest.get('completed') is not True:
        audit.warning('partial_capture_explicitly_allowed_for_source_audit_only',
                      [manifest.get('completed'), manifest.get('status')])
    else:
        audit.check(manifest.get('completed') is True and manifest.get('status') == 'completed',
                    'capture_completed', [manifest.get('completed'), manifest.get('status')])
    contract = manifest.get('contract', {})
    for key, expected in [('physical_action_dim', 24), ('physics_dt_s', .004),
                          ('physics_steps_per_action', 15), ('nominal_control_seconds', .06),
                          ('native_rgb_shape', [480, 640, 3]), ('policy_loaded', False),
                          ('training_enabled', False), ('motor_supervision_eligible', False)]:
        audit.check(contract.get(key) == expected, 'capture_contract_' + key,
                    {'actual': contract.get(key), 'expected': expected})
    audit.check(contract.get('camera_keys') == list(CAMERAS), 'capture_three_cameras', contract)
    audit.check(manifest.get('vendor_urdf_unchanged') is True
                and manifest.get('source_actions_unchanged') is True,
                'capture_original_source_unchanged',
                [manifest.get('vendor_urdf_unchanged'), manifest.get('source_actions_unchanged')])
    planned = {x['episode_id']: x for x in manifest.get('planned_episodes', [])}
    episodes = manifest.get('episodes', [])
    audit.check(manifest.get('expected_full_episodes') == len(planned)
                and (len(episodes) <= len(planned) if allow_partial else len(episodes) == len(planned)),
                'capture_expected_episode_count', [len(episodes), len(planned)])
    source = manifest.get('source_action_receipt', {})
    reference = []
    if source_receipt(source.get('path', ''), source.get('sha256'), audit, 'source_action'):
        raw = [json.loads(x) for x in Path(source['path']).read_text().splitlines() if x.strip()]
        reference = [np.asarray(x['actions'], dtype=np.float32)[0] for x in raw if x['physics_ticks'] == 15]
        audit.check(len(reference) == 244, 'source_reference244', len(reference))
    by_episode, by_observation = {}, {}
    raw_log_receipts = []
    endpoint_metrics = {'targets':0, 'max_abs_actual_minus_expected':0.0,
                        'max_abs_expected_minus_raw_arm_rad':0.0,
                        'max_abs_expected_minus_raw_hand_SDK':0.0}
    pixel_splits, model_pixel_splits, scene_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    family_splits, lineage_splits = defaultdict(set), defaultdict(set)
    for ep in episodes:
        sid = ep.get('episode_id')
        if not audit.check(sid not in by_episode and sid in planned,
                           'capture_unique_planned_episode', sid, sid):
            continue
        source_receipt(ep['episode_json_path'], ep['episode_json_sha256'], audit,
                       'capture_episode', sid)
        on_disk = json.loads(Path(ep['episode_json_path']).read_text())
        reduced = {k: v for k, v in ep.items() if k not in {'episode_json_path', 'episode_json_sha256'}}
        audit.check(on_disk == reduced, 'capture_episode_manifest_equal', str(sid), sid)
        spec = planned[sid]
        audit.check(all(ep.get(k) == spec.get(k) for k in ['episode_id', 'family_id', 'split',
                                                           'seed', 'target_side', 'layout']),
                    'capture_matches_planned_spec', spec, sid)
        audit.check(ep.get('completed') is True and ep.get('status') == 'completed'
                    and ep.get('worker_shutdown_confirmed') is True,
                    'capture_episode_completed_worker_closed', sid, sid)
        audit.check(ep.get('capture_budget_reached') is True
                    and ep.get('executed_controls') == manifest['expected_targets_per_full_episode']
                    and ep.get('physics_ticks') == ep['executed_controls'] * 15,
                    'capture_episode_full_controls_clock',
                    [ep.get('executed_controls'), ep.get('physics_ticks')], sid)
        audit.check(ep.get('policy_loaded') is False and ep.get('motor_supervision_eligible') is False
                    and ep.get('physical_skill_success_claimed') is False,
                    'capture_non_motor_label_scope', sid, sid)
        family_splits[ep['family_id']].add(ep['split'])
        lineage_splits[sid].add(ep['split'])
        physical_layout = {k: ep['layout'][k] for k in ['home_xy', 'model_ids']}
        scene_splits[digest(physical_layout)].add(ep['split'])
        events = [json.loads(x) for x in Path(ep['issued_subtasks_path']).read_text().splitlines() if x.strip()]
        log_paths = [('issued_subtasks_path', Path(ep['issued_subtasks_path'])),
                     ('transitions_path', Path(ep['transitions_path'])),
                     ('private_monitor_path', Path(ep['private_monitor_path'])),
                     ('raw_rpc_before_validation', Path(ep['episode_json_path']).parent / 'raw_rpc_before_validation.jsonl'),
                     ('drive_readback_detail', Path(ep['episode_json_path']).parent / 'drive_readback_detail.jsonl')]
        for key, log_path in log_paths:
            log_sha = file_sha(log_path)
            raw_log_receipts.append({'episode_id': sid, 'kind': key,
                                     'path': str(log_path.resolve()),
                                     'bytes': log_path.stat().st_size, 'sha256': log_sha})
            expected_log = ep.get('source_log_receipts', {}).get(key,
                             ep.get('artifact_sha256', {}).get(log_path.name))
            if expected_log is not None:
                expected_sha = expected_log['sha256'] if isinstance(expected_log, dict) else expected_log
                audit.check(log_sha == expected_sha, 'capture_frozen_raw_log_sha256',
                            [sid, key], sid)
            else:
                audit.check(False, 'raw_log_has_frozen_episode_receipt', [sid, key], sid)
        event_decisions = []
        for e in events:
            start = e['issued_at_step']
            audit.check(start % 32 == 0 and e.get('issued_at_physics_step') == start * 15,
                        'capture_issued_event_clock', e, sid)
            audit.check(e.get('source') == 'authored_subtask_for_reference_attempt'
                        and e.get('execution_semantics') == 'instruction_issued_not_physical_completion',
                        'capture_issued_event_scope', e, sid)
            event_decisions.append({'decision_id': e['event_id'], 'start_step': start,
                                    'end_step': e['planned_window_end_step'],
                                    'generated_subtask': e['text']})
        expected_starts = list(range(0, ep['executed_controls'], 32))
        audit.check([x['start_step'] for x in event_decisions] == expected_starts,
                    'capture_all_issued_events', expected_starts, sid)
        transitions = [json.loads(x) for x in Path(ep['transitions_path']).read_text().splitlines() if x.strip()]
        raw_replies = [json.loads(x) for x in (Path(ep['episode_json_path']).parent / 'raw_rpc_before_validation.jsonl').read_text().splitlines() if x.strip()]
        readback_details = [json.loads(x) for x in (Path(ep['episode_json_path']).parent / 'drive_readback_detail.jsonl').read_text().splitlines() if x.strip()]
        audit.check(len(raw_replies) == len(transitions) == len(readback_details),
                    'capture_all_actual_RPC_replies_preserved', len(transitions), sid)
        controls = []
        cursor = ticks = 0
        event_for_start = {d['start_step']: d for d in event_decisions}
        initial = np.asarray(ep['inactive_arm_initial_state24'], dtype=np.float32)
        for chunk_index, chunk in enumerate(transitions):
            start, end = chunk['control_start'], chunk['control_end']
            count = chunk['executed_action_count']
            sent = np.asarray(chunk['issued_action_chunk'], dtype=np.float32)
            receipts = chunk['sdk_target_receipts']
            raw_reply = raw_replies[chunk_index]
            audit.check(raw_reply['control_start_before_rpc'] == start
                        and raw_reply['issued_action_chunk'] == chunk['issued_action_chunk']
                        and raw_reply['actual_info']['action_trace'] == receipts
                        and raw_reply['actual_info']['step_count'] == end
                        and raw_reply['actual_info']['physics_step_count'] == chunk['physics_step_end']
                        and raw_reply['returned_actual_state24'] == chunk['measured_state24_post_chunk'],
                        'capture_actual_RPC_reply_matches_validated_transition', start, sid)
            audit.check(start == cursor and end == start + count and count == len(receipts)
                        and sent.ndim == 3 and sent.shape[0] == 1 and sent.shape[2] == 24,
                        'capture_chunk_control_sequence', [start, end, count, cursor], sid)
            audit.check(np.isfinite(sent).all()
                        and chunk['valid_action_mask'] == [i < count for i in range(sent.shape[1])],
                        'capture_chunk_sent_and_mask', start, sid)
            for offset, receipt in enumerate(receipts):
                control_step = start + offset
                target = sent[0, offset]
                actual = np.asarray(receipt['raw_model_action24'], dtype=np.float32)
                audit.check(np.array_equal(target, actual), 'capture_actual_issued_action', control_step, sid)
                n = receipt['executed_physics_steps']
                audit.check(n == 15 and receipt['requested_physics_steps'] == 15
                            and receipt['physics_step_start'] == ticks
                            and receipt['physics_step_end'] == ticks + n,
                            'capture_actual_target15tick_clock', control_step, sid)
                expected_readback = independent_drive_readback(target, command_spec)
                actual_readback = np.asarray(receipt['actual_final_drive_command24'])
                endpoint_metrics['targets'] += 1
                endpoint_metrics['max_abs_actual_minus_expected'] = max(
                    endpoint_metrics['max_abs_actual_minus_expected'],
                    float(np.max(np.abs(actual_readback - expected_readback))))
                representation_delta = expected_readback - target.astype(np.float64)
                endpoint_metrics['max_abs_expected_minus_raw_arm_rad'] = max(
                    endpoint_metrics['max_abs_expected_minus_raw_arm_rad'],
                    float(np.max(np.abs(representation_delta[np.r_[0:6,12:18]]))))
                endpoint_metrics['max_abs_expected_minus_raw_hand_SDK'] = max(
                    endpoint_metrics['max_abs_expected_minus_raw_hand_SDK'],
                    float(np.max(np.abs(representation_delta[np.r_[6:12,18:24]]))))
                audit.check(receipt.get('endpoint_command_written') is True
                            and np.allclose(actual_readback, expected_readback, atol=1e-9, rtol=1e-12),
                            'capture_actual_target_endpoint', control_step, sid)
                detail = readback_details[chunk_index]['details'][offset]
                audit.check(readback_details[chunk_index]['control_start'] == start
                            and np.array_equal(np.asarray(detail['raw_issued_sdk24'], dtype=np.float32), target)
                            and np.array_equal(np.asarray(detail['expected_final_drive_sdk24']), expected_readback)
                            and np.array_equal(np.asarray(detail['actual_final_drive_sdk24']), actual_readback),
                            'capture_readback_detail_independently_matches', control_step, sid)
                outside = (target.astype(np.float64) < command_spec['lower']) | (target.astype(np.float64) > command_spec['upper'])
                allowed_outside = np.zeros(24, dtype=bool)
                inactive_hand = range(18, 24) if ep['target_side'] == 'left' else range(6, 12)
                for dim in inactive_hand:
                    if outside[dim]:
                        boundary = target.copy()
                        boundary[dim] = 0 if target[dim] < 0 else 1000
                        explained = np.float32(independent_drive_readback(boundary, command_spec)[dim])
                        allowed_outside[dim] = target[dim] == initial[dim] == explained
                audit.check(not np.any(outside & ~allowed_outside),
                            'capture_issued_command_bounds_no_implicit_clamp',
                            {'step':control_step,'violating_dims':np.flatnonzero(outside & ~allowed_outside).tolist()}, sid)
                source_i = min(control_step, 243)
                audit.check(chunk['source_action_indices'][offset] == source_i
                            and chunk['tail_hold_mask'][offset] == (control_step >= 244),
                            'capture_reference_contiguous_no_skips', control_step, sid)
                if reference:
                    expected_action = initial.copy()
                    if ep['target_side'] == 'right':
                        expected_action[12:24] = reference[source_i][12:24]
                    else:
                        expected_action[:6] = -reference[source_i][12:18]
                        expected_action[6:12] = reference[source_i][18:24]
                    audit.check(np.array_equal(target, expected_action),
                                'capture_reference_mapping_exact', control_step, sid)
                event = event_for_start.get(control_step // 32 * 32)
                controls.append({'step_index': control_step,
                                 'decision_id': event['decision_id'] if event else None})
                ticks += n
            cursor = end
            audit.check(chunk['physics_step_end'] == ticks and chunk['observation_valid'] is True,
                        'capture_chunk_physics_observation', [cursor, ticks], sid)
        audit.check(cursor == ep['executed_controls'] and ticks == ep['physics_ticks'],
                    'capture_trace_totals', [cursor, ticks], sid)
        stride = 32 if ep['split'] == 'train' else 16
        expected_observations = list(range(0, ep['executed_controls'] + 1, stride))
        audit.check([x['step'] for x in ep['observations']] == expected_observations,
                    'capture_snapshot_boundaries', expected_observations, sid)
        for obs in ep['observations']:
            d = obs['step']
            audit.check(obs['physics_step'] == d * 15 and obs['observed_until_step'] == d,
                        'capture_observation_clock', d, sid)
            path = Path(obs['npz_path']).resolve()
            source_receipt(path, obs['sha256'], audit, 'capture_observation', sid)
            with np.load(path, allow_pickle=False) as npz:
                state = npz['state'].copy()
                audit.check(state.shape == (24,) and np.isfinite(state).all(),
                            'capture_observation_state24', str(path), sid)
                camera_shas = []
                model_camera_shas = []
                for camera in CAMERAS:
                    arr = npz[camera]
                    camera_sha = hashlib.sha256(arr.tobytes()).hexdigest()
                    audit.check(arr.shape == (480, 640, 3) and arr.dtype == np.uint8
                                and float(arr.std()) > 1
                                and camera_sha == obs['cameras'][camera]['sha256'],
                                'capture_observation_actual_rgb', [str(path), camera], sid)
                    camera_shas.append(camera_sha)
                    model_pixels = np.asarray(resize_with_pad(arr, 224, 224))
                    model_camera_shas.append(hashlib.sha256(model_pixels.tobytes()).hexdigest())
                audit.check(hashlib.sha256(state.tobytes()).hexdigest() == obs['state_sha256'],
                            'capture_observation_state_sha', str(path), sid)
                audit.check(str(npz['instruction']) == ep['goal'],
                            'capture_observation_instruction', str(path), sid)
            pixel_key = digest(camera_shas)
            model_pixel_key = digest(model_camera_shas)
            pixel_splits[pixel_key].add(ep['split'])
            model_pixel_splits[model_pixel_key].add(ep['split'])
            by_observation[str(path)] = {'episode_id': sid, 'family_id': ep['family_id'],
                                         'split': ep['split'], 'step': d, 'pixel_key': pixel_key,
                                         'model_pixel_key': model_pixel_key,
                                         'npz_sha256': obs['sha256'], 'state': state,
                                         'observation': obs}
        by_episode[sid] = {'record': ep, 'decisions': event_decisions, 'controls': controls}
    for name, groups in [('source_family', family_splits), ('source_lineage', lineage_splits),
                          ('source_physical_scene', scene_splits), ('source_exact_pixel_triplet', pixel_splits),
                          ('source_resized_model_pixel_triplet', model_pixel_splits)]:
        collisions = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
        audit.check(not collisions, name + '_split_disjoint', collisions)
    summary = {'episodes': len(by_episode), 'observations': len(by_observation),
               'unique_pixel_triplets': len(pixel_splits),
               'unique_native_resized224_model_pixel_triplets':len(model_pixel_splits),
               'families': len(family_splits), 'physical_scene_groups': len(scene_splits),
               'actual_controls': sum(len(x['controls']) for x in by_episode.values()),
               'raw_log_receipts': raw_log_receipts,
               'independent_endpoint_metrics':endpoint_metrics,
               'scope': 'actual reference-motion attempts; no physical-success or motor-demonstration labels'}
    if not allow_partial:
        expected_episodes = {'train':32, 'val':12, 'test':12}
        expected_families = {'train':16, 'val':6, 'test':6}
        expected_observations = {'train':288, 'val':204, 'test':204}
        audit.check(manifest.get('expected_targets_per_full_episode') == 256,
                    'full_capture_contract_256_controls', manifest.get('expected_targets_per_full_episode'))
        summary['episodes_by_split'] = dict(Counter(x['record']['split'] for x in by_episode.values()))
        summary['families_by_split'] = {s:sum(s in v for v in family_splits.values()) for s in EXPECTED}
        summary['observations_by_split'] = dict(Counter(x['split'] for x in by_observation.values()))
        audit.check(summary['episodes_by_split'] == expected_episodes,
                    'full_capture_fresh56_episodes', summary['episodes_by_split'])
        audit.check(summary['families_by_split'] == expected_families,
                    'full_capture_disjoint28_families', summary['families_by_split'])
        audit.check(summary['observations_by_split'] == expected_observations,
                    'full_capture696_real_observations', summary['observations_by_split'])
    return manifest, by_episode, by_observation, summary


def normalized_state(raw, stats):
    """Independent SDK hands->native checkpoint quantile->internal32 mapping."""
    state = np.asarray(raw, dtype=np.float32).copy()
    if state.shape != (24,) or not np.isfinite(state).all():
        raise ValueError('Invalid measured SDK24 state')
    state[6:12] /= 1000
    state[18:24] /= 1000
    lo, hi = (np.asarray(stats[k]) for k in ('q01', 'q99'))
    if lo.shape != (24,) or hi.shape != (24,) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
        raise ValueError('Invalid original state normalization asset')
    # Native actor converts transformed state to torch.float32 before prefix
    # quantization; float64 quantile arithmetic alone is not the final boundary.
    return np.pad((state - lo) / (hi - lo + 1e-6) * 2 - 1, (0, 8)).astype(np.float32)


def effective_inputs(rows, by_episode, by_observation, tokenizer, stats, audit, *, preview=False):
    """Use actual local token IDs and pixels, never label-builder digest claims."""
    from rlinf.models.embodiment.openpi.recap_tokenizer import build_high_prefix

    actual_groups = defaultdict(list)
    anchors = defaultdict(set)
    token_ranges = defaultdict(list)
    mem_coverage = Counter()
    item_receipts = []
    runtime_guard_rejections = []
    for row in rows:
        sid, split = row.get('sample_id'), row.get('split')
        ref = row.get('observation', {})
        path = str(Path(ref.get('npz_path', '')).resolve())
        if not audit.check(path in by_observation, 'label_actual_capture_observation', path, sid):
            continue
        source_obs = by_observation[path]
        source_ep = by_episode[source_obs['episode_id']]
        audit.check(row.get('episode_id') == source_obs['episode_id']
                    and row.get('family_id') == source_obs['family_id']
                    and split == source_obs['split']
                    and row.get('observed_until_step') == source_obs['step'],
                    'label_source_ownership_and_cutoff', path, sid)
        compare_actual_source_history(row, source_ep['decisions'], source_ep['controls'], audit,
                                      source_snapshot_history=source_obs['observation']['visible_history'])
        # Audit the actual row's prefix even when provenance/ID checks fail;
        # expected history must never silently repair the tokenizer input.
        prior_issues = len(audit.issues)
        hist = strict_history(row, audit)
        guard_failures = [x for x in audit.issues[prior_issues:] if x['code'] == 'strict_history']
        if guard_failures:
            runtime_guard_rejections.append({'sample_id':sid,'reasons':guard_failures,
                                            'prefix_token_audit':'not_applicable_actual_input_guard_rejects_before_generation'})
            continue
        visible = row['context'].get('actor_memory', row['context'].get('selected_memory', {}))
        prefix = build_high_prefix(row['mode'], task=row['context']['goal'],
                                   state=normalized_state(source_obs['state'], stats),
                                   visible_memory=canonical(visible),
                                   memory_index=canonical(row['context']['catalogue']), history=hist)
        try:
            ids = tokenizer.encode_prefix(prefix).tolist()
            targets = tokenizer.encode_target(row['target_text']).tolist()
        except ValueError as exc:
            audit.check(False, 'real_token_budgets_no_truncation', str(exc), sid)
            continue
        bos = tokenizer.text_tokenizer.bos_id()
        eos = tokenizer.eos_token_id
        unk = tokenizer.text_tokenizer.unk_id()
        audit.check(bool(ids) and ids[0] == bos and ids.count(bos) == 1,
                    'actual_prefix_one_bos', ids[:8], sid)
        audit.check(bool(targets) and targets[-1] == eos and targets.count(eos) == 1
                    and bos not in targets, 'actual_target_one_eos_no_bos', targets[-8:], sid)
        audit.check(unk not in ids and unk not in targets, 'actual_tokens_no_unk', sid, sid)
        audit.check(tokenizer.decode_text(targets[:-1]) == row['target_text'],
                    'actual_target_token_roundtrip', row['target_text'], sid)
        if not preview:
            builder_digest = digest({'mode':row['mode'], 'prefix_ids':ids,
                                     'camera_pixel_sha256':{k:source_obs['observation']['cameras'][k]['sha256'] for k in CAMERAS}})
            audit.check(row.get('actor_input_digest') == builder_digest,
                        'builder_actor_digest_matches_real_float32_token_input', builder_digest, sid)
            audit.check(row.get('token_lengths') == {'prefix':len(ids),'target':len(targets),'total':len(ids)+len(targets)},
                        'builder_token_length_receipt_matches_real_tokens', row.get('token_lengths'), sid)
        token_ranges[row['mode']].append((len(ids), len(targets)))
        mem = ref.get('mem_history')
        mem_signature = None
        if mem is not None:
            mem_coverage['rows_with_mem_history'] += 1
            frames = mem.get('frames', [])
            valid = mem.get('valid_mask', [])
            ages = mem.get('age_s', [])
            steps = mem.get('physics_steps', [])
            length_ok = isinstance(frames, list) and bool(frames) and all(
                isinstance(x, list) and len(x) == len(frames) for x in [valid, ages, steps])
            if not audit.check(length_ok, 'MEM_parallel_arrays', sid, sid):
                continue
            current = source_obs['step'] * 15
            audit.check(valid[-1] is True and ages[-1] == 0 and steps[-1] == current
                        and mem.get('current_physics_step', current) == current,
                        'MEM_current_clock', sid, sid)
            last_step, sig = -1, []
            for frame, is_valid, age, step in zip(frames, valid, ages, steps):
                audit.check(type(is_valid) is bool, 'MEM_boolean_mask', is_valid, sid)
                if not is_valid:
                    sig.append(None)
                    continue
                frame_path = str(Path(frame.get('npz_path', '')).resolve()) if isinstance(frame, dict) else ''
                if not audit.check(frame_path in by_observation, 'MEM_actual_source_frame', frame_path, sid):
                    continue
                other = by_observation[frame_path]
                audit.check(other['episode_id'] == source_obs['episode_id']
                            and type(step) is int and last_step < step <= current
                            and step == other['step'] * 15
                            and abs(float(age) - (current - step) * .004) < 1e-6,
                            'MEM_same_episode_causal_age', frame_path, sid)
                last_step = step
                sig.append({'model_pixel_key':other['model_pixel_key'],
                            'state_sha':hashlib.sha256(other['state'].tobytes()).hexdigest(),
                            'physics_step':step, 'age_s':age})
            audit.check(isinstance(frames[-1], dict)
                        and str(Path(frames[-1].get('npz_path', '')).resolve()) == path,
                        'MEM_last_frame_is_current_actual', path, sid)
            mem_signature = {'frames':sig,'valid':valid,'ages':ages,'physics_steps':steps}
        else:
            mem_coverage['rows_without_mem_history'] += 1
        actual = digest({'current_model_pixel_triplet':source_obs['model_pixel_key'],
                         'actual_prefix_token_ids':ids, 'mem_history':mem_signature})
        actual_groups[actual].append({'sample_id':sid,'split':split,'target':row['target_text']})
        anchors[split].add(digest({'pixels':source_obs['pixel_key'],
                                  'measured_state_sha':hashlib.sha256(source_obs['state'].tobytes()).hexdigest()}))
        item_receipts.append({'sample_id':sid,'split':split,'actual_input_sha256':actual,
                              'actual_prefix_token_sha256':digest(ids),'prefix_tokens':len(ids),
                              'target_including_eos_tokens':len(targets)})
    duplicate_groups = []
    for key, group in actual_groups.items():
        if len(group) > 1:
            duplicate_groups.append({'actual_input_sha256':key,'rows':group})
        audit.check(len({x['split'] for x in group}) == 1, 'actual_effective_input_split_disjoint', group)
        audit.check(len({x['target'] for x in group}) == 1, 'actual_identical_input_no_conflicting_target', group)
        audit.check(len(group) == 1, 'actual_effective_input_unique_after_dedup', group)
    return {'unique_effective_inputs':len(actual_groups),
            'unique_physical_observation_anchors_by_split':{k:len(v) for k,v in anchors.items()},
            'actual_token_lengths':{k:{'prefix_min':min(x[0] for x in v),'prefix_max':max(x[0] for x in v),
                                       'target_min':min(x[1] for x in v),'target_max':max(x[1] for x in v)}
                                    for k,v in token_ranges.items()},
            'MEM_history_coverage':dict(mem_coverage), 'duplicate_groups':duplicate_groups,
            'runtime_input_guard_rejections':runtime_guard_rejections,
            'per_sample_token_receipts':item_receipts}


def summarize_rows(rows, audit, *, preview=False):
    counts = defaultdict(Counter)
    coverage = defaultdict(Counter)
    rev_ops = defaultdict(lambda: defaultdict(Counter))
    family_splits, episode_splits = defaultdict(set), defaultdict(set)
    seen_ids = Counter()
    for row in rows:
        sid, split, mode = row.get("sample_id"), row.get("split"), row.get("mode")
        seen_ids[sid] += 1
        audit.check(split in EXPECTED and mode in MODES, "split_mode", [split, mode], sid)
        audit.check(row.get("context", {}).get("mode") == mode, "context_mode_matches", mode, sid)
        counts[split][mode] += 1
        family_splits[row.get("family_id")].add(split)
        episode_splits[row.get("episode_id")].add(split)
        validate_visible_memory(row, audit)
        hist = strict_history(row, audit)
        coverage[split + ':' + mode]["rows"] += 1
        coverage[split + ':' + mode]["empty"] += not bool(hist)
        coverage[split + ':' + mode]["nonempty_executed"] += any(
            x['executed_until_step'] > x['issued_at_step'] for x in hist)
        coverage[split + ':' + mode]["multiple_events"] += len(hist) >= 2
        coverage[split + ':' + mode]["observed_step_ge32"] += row.get('observed_until_step', -1) >= 32
        target = target_schema(row, audit)
        if mode == 'WRITE' and target:
            ops = target.get('operations', [])
            op = ops[0].get('op') if ops else 'EMPTY'
            memory = row.get('context', {}).get('selected_memory', [])
            changed_slot = ops[0].get('slot_id') if ops else None
            # Builder must declare the focused slot for EMPTY multi-slot records.
            focus = row.get('evidence', {}).get('memory_trace', {}).get('controlled_target_slot', changed_slot)
            slots = [x for x in memory if x.get('slot_id') == focus] if focus else memory[:1]
            rev = slots[0].get('revision') if slots else 'missing'
            rev_ops[split][op][str(rev)] += 1
    audit.check(all(x == 1 for x in seen_ids.values()), 'sample_ids_unique',
                {k: v for k, v in seen_ids.items() if v != 1})
    for field, groups in [('family', family_splits), ('episode', episode_splits)]:
        audit.check(None not in groups and all(len(s) == 1 for s in groups.values()),
                    field + '_split_disjoint', {str(k): sorted(v) for k, v in groups.items() if k is None or len(v) != 1})
    if not preview:
        for split, expected in EXPECTED.items():
            audit.check(dict(counts.get(split, {})) == expected, 'exact_counts_' + split,
                        {'actual': dict(counts.get(split, {})), 'expected': expected})
            for mode in MODES:
                cov = coverage[split + ':' + mode]
                audit.check(cov['nonempty_executed'] >= cov['rows'] // 2
                            and cov['multiple_events'] > 0 and cov['observed_step_ge32'] > 0,
                            'actual_dynamic_history_slice_coverage_' + split + '_' + mode, dict(cov))
    return {'counts': {k: dict(v) for k, v in counts.items()},
            'history_coverage': {k: dict(v) for k, v in coverage.items()},
            'WRITE_revision_operation_counts': {s: {o: dict(v) for o, v in ops.items()} for s, ops in rev_ops.items()},
            'physical_families': {s: sum(s in v for v in family_splits.values()) for s in EXPECTED},
            'source_episodes': {s: sum(s in v for v in episode_splits.values()) for s in EXPECTED}}


def protocol_truth(row, history, audit):
    """Independent formal truth, based on explicit goal and actual control spans."""
    sid = row['sample_id']
    sides = re.findall(r'sample initially on the (left|right)', row['context']['goal'])
    audit.check(len(sides) == 2 and set(sides) == {'left', 'right'},
                'goal_two_ordered_initial_roles', sides, sid)
    if not sides:
        return None, {}
    side = sides[0]
    audit.check(side == row['metadata']['target_side'], 'goal_metadata_role_agrees', side, sid)
    total = sum(e['executed_until_step'] - e['issued_at_step'] for e in history)
    audit.check(total == row['observed_until_step'], 'actual_history_spans_cover_observed_controls', total, sid)
    facts = {'sample_identity': f'The first requested sample is initially on the {side}.'}
    if history:
        facts['current_progress'] = (f"Last issued {history[-1]['decision_id']} for initial-{side}; "
                                     f'executed_control_steps={total}; physical completion unverified.')
    return side, facts


def formal_expected_operation(selected, focus, facts):
    entry = next(x for x in selected if x['slot_id'] == focus)
    if entry['status'] == 'empty':
        return [{'op':'WRITE', 'slot_id':focus, 'value':facts[focus], 'confirmed':True}]
    if entry['value'] != facts[focus]:
        return [{'op':'CORRECT', 'slot_id':focus, 'value':facts[focus], 'confirmed':True}]
    if not entry['confirmed']:
        return [{'op':'KEEP', 'slot_id':focus, 'confirmed':True}]
    return []


def memory_replay(row, facts, audit):
    """Re-run strict tools from source-bearing seed operations, never trust view."""
    sid, ctx, meta = row['sample_id'], row['context'], row['metadata']
    trace = row['evidence'].get('memory_trace')
    if row['mode'] == 'READ':
        audit.check(trace is None, 'READ_no_hidden_memory_trace_needed', trace, sid)
        return None
    if not audit.check(isinstance(trace, dict) and trace.get('native_strict_protocol') is True,
                       'private_trace_native_strict', trace, sid):
        return None
    bank = EpisodeMemory(ctx['catalogue'], 'history3000_bank', row['episode_id'], strict_protocol=True)
    def source_check(source, kind):
        audit.check(source.get('kind') == kind
                    and source.get('step') == row['observed_until_step']
                    and source.get('observed_until_step') == row['observed_until_step']
                    and source.get('observation_refs') == [{'npz_path':row['observation']['npz_path']}],
                    'native_memory_source_visible_and_causal', source, sid)
    for seed in trace['seed_rounds']:
        bank.read(seed['read_ids'])
        source_check(seed['source'], 'authored_counterfactual_memory_condition')
        result = bank.apply(seed['operation'], source=seed['source'])
        audit.check(result == seed['response'], 'seed_native_tool_response_exact', seed, sid)
    ids = ['sample_identity', 'current_progress'] if 'current_progress' in facts else ['sample_identity']
    audit.check(trace['read_ids'] == ids and ctx['read_ids'] == ids,
                'actual_trace_READ_ids', trace['read_ids'], sid)
    selected = bank.read(trace['read_ids'])
    audit.check(selected == trace['selected_memory'], 'native_selected_pre_write_exact', selected, sid)
    focus = trace['controlled_target_slot']
    audit.check(focus in facts, 'controlled_slot_fact_visible', focus, sid)
    expected = formal_expected_operation(selected, focus, facts)
    expected_op = expected[0]['op'] if expected else 'EMPTY'
    audit.check(expected_op == trace['condition'] == meta['memory_condition'],
                'operation_label_from_actual_visible_slot', expected_op, sid)
    focused = next(x for x in selected if x['slot_id'] == focus)
    controlled_revision = trace['controlled_active_revision']
    audit.check(controlled_revision in ACTIVE_REVISIONS and controlled_revision == meta['active_revision']
                and (focused['revision'] == 0 if expected_op == 'WRITE' else focused['revision'] == controlled_revision),
                'actual_focus_revision_matches_declared_control', focused, sid)
    for slot in selected:
        if slot['slot_id'] != focus:
            audit.check(slot['status'] == 'active' and slot['confirmed'] is True
                        and slot['value'] == facts[slot['slot_id']],
                        'nonfocused_memory_truth_unchanged', slot, sid)
    stress = meta['stress_without_writer_repair']
    audit.check(type(stress) is bool and stress == trace['stress_without_writer_repair'],
                'stress_metadata_trace_agree', stress, sid)
    if stress:
        audit.check(row['mode'] == 'SUBTASK' and expected_op in {'WRITE','KEEP','CORRECT'}
                    and trace['operations'] == [], 'stress_has_real_unrepaired_memory_fault', expected_op, sid)
    else:
        audit.check(trace['operations'] == expected, 'normal_writer_operations_follow_formal_truth', expected, sid)
    audit.check(len(trace['operations']) == len(trace['operation_responses']),
                'native_operation_response_count', len(trace['operation_responses']), sid)
    for op, response in zip(trace['operations'], trace['operation_responses']):
        source_check(response['source'], 'formal_goal_and_issued_control_history')
        actual_response = bank.apply(op, source=response['source'])
        audit.check(actual_response == response, 'actual_native_operation_response_exact', response, sid)
    actor = bank.actor_view(trace['read_ids'], [x['slot_id'] for x in trace['operations']])
    audit.check(actor == trace['actor_memory'], 'native_actor_view_latest_exact', actor, sid)
    if row['mode'] == 'WRITE':
        audit.check(ctx['selected_memory'] == selected
                    and json.loads(row['target_text']) == {'operations':expected},
                    'WRITE_label_and_visible_prebank_match_independent_rule', expected, sid)
    else:
        audit.check(ctx['actor_memory'] == actor, 'SUBTASK_visible_real_native_actor_view', actor, sid)
    return {'focus_slot':focus, 'expected_prebank_operation':expected_op,
            'controlled_revision':controlled_revision, 'stress':stress,
            'confirmed_wrong_prebank':focused['status'] == 'active' and focused['confirmed'] is True
                                     and focused['value'] != facts[focus],
            'selected':selected}


def audit_protocol_rows(rows, by_episode, by_observation, audit, *, preview=False):
    """Formal label acceptance is deliberately separate from physical grounding."""
    groups = defaultdict(lambda:defaultdict(list))
    stress_counts, stress_cells, template_counts, target_strings, side_counts = Counter(), Counter(), defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    active_counts = defaultdict(lambda:defaultdict(Counter))
    WRITE_wrong_flag_cells = Counter()
    read_ids_by_base, subtask_bases = {}, defaultdict(list)
    guard_rejected_rows = []
    for row in rows:
        sid, mode, meta, ev = row['sample_id'], row['mode'], row['metadata'], row['evidence']
        prior_issues = len(audit.issues)
        hist = strict_history(row, audit)
        guard_failures = [x for x in audit.issues[prior_issues:] if x['code'] == 'strict_history']
        if guard_failures:
            guard_rejected_rows.append({'sample_id':sid,'reasons':guard_failures})
            continue
        side, facts = protocol_truth(row, hist, audit)
        if not side:
            continue
        split = row['split']
        side_counts[split + ':' + mode][side] += 1
        template_counts[split][str(meta['goal_template'])] += 1
        target_strings[split + ':' + mode].add(row['target_text'])
        expected_training = split == 'train' and not preview
        audit.check(meta.get('training_eligible') is expected_training
                    and row.get('supervision_kind') == ('protocol_candidate' if preview else 'verified_high_sft'),
                    'training_eligibility_and_preview_scope', [split,row.get('supervision_kind')], sid)
        audit.check(meta.get('MEM_ready') is False and not row['observation'].get('mem_history')
                    and meta.get('motor_labels') is False and meta.get('goal_action_alignment_validated') is False
                    and meta.get('label_scope') == 'protocol_rules_not_motor_success'
                    and ev.get('physical_progress_claimed') is False,
                    'formal_protocol_only_no_video_or_motor_claim', meta, sid)
        audit.check(set(row['provenance'].get('not_verified', [])) >=
                    {'visual_grounding','physical_skill_success','model_performance'},
                    'provenance_explicit_capability_limits', row['provenance'], sid)
        audit.check(ev.get('truth_facts') == facts, 'facts_independently_from_explicit_goal_and_actual_history', facts, sid)
        ep = by_episode.get(row['episode_id'])
        obs = by_observation.get(str(Path(row['observation']['npz_path']).resolve()))
        if ep is not None and obs is not None:
            source = ep['record']
            audit.check(side == source['target_side'] and row['seed'] == source['seed']
                        and row['decision_physics_step'] == row['decision_step'] * 15
                        and ev['source_episode_json'] == source['episode_json_path']
                        and ev['source_episode_json_sha256'] == source['episode_json_sha256']
                        and ev['source_observation_sha256'] == obs['npz_sha256']
                        and ev['source_issued_subtasks_path'] == source['issued_subtasks_path']
                        and ev['source_transitions_path'] == source['transitions_path']
                        and ev['camera_pixel_sha256'] == obs['observation']['cameras'],
                        'label_evidence_matches_actual_source_receipts', sid, sid)
        recomputed_base = digest({'episode':row['episode_id'],'step':row['observed_until_step'],
                                  'goal':row['context']['goal'],'history':hist})
        audit.check(meta['base_context_id'] == recomputed_base
                    and meta['physical_context_id'] == f"{row['episode_id']}:s{row['observed_until_step']}",
                    'base_and_physical_context_ids_recomputed', recomputed_base, sid)
        base = meta['base_context_id']
        groups[split][base].append(row)
        if mode == 'READ':
            expected_ids = ['sample_identity','current_progress'] if hist else ['sample_identity']
            audit.check(json.loads(row['target_text']) == {'read_ids':expected_ids},
                        'READ_rule_history_requires_progress_slot', expected_ids, sid)
            read_ids_by_base[base] = sid
        else:
            result = memory_replay(row, facts, audit)
            if result is not None and mode == 'WRITE' and result['expected_prebank_operation'] != 'WRITE':
                active_counts[split][result['expected_prebank_operation']][str(result['controlled_revision'])] += 1
                if result['expected_prebank_operation'] == 'CORRECT':
                    focused = next(x for x in result['selected'] if x['slot_id'] == result['focus_slot'])
                    cell = ':'.join([split,str(focused['confirmed']).lower(),side,str(focused['revision'])])
                    WRITE_wrong_flag_cells[cell] += 1
            if mode == 'SUBTASK':
                subtask_bases[base].append(sid)
                allowed = [f'Pick up and hold the sample initially on the {side}.',
                           f'Grasp and hold the sample initially on the {side}.',
                           f'Pick up the sample initially on the {side} and hold it.',
                           f'Lift and hold the sample initially on the {side}.',
                           f'Pick up and keep holding the sample initially on the {side}.',
                           f'Grasp the sample initially on the {side} and hold it.']
                audit.check(row['target_text'] in allowed, 'SUBTASK_goal_bound_limited_action_capability', row['target_text'], sid)
                if result is not None and result['stress']:
                    stress_counts[split] += 1
                    stress_counts[split + ':' + result['expected_prebank_operation']] += 1
                    stress_counts[split + ':confirmed_wrong'] += result['confirmed_wrong_prebank']
                    focused = next(x for x in result['selected'] if x['slot_id'] == result['focus_slot'])
                    cell = ':'.join([split,result['focus_slot'],result['expected_prebank_operation'],
                                     str(focused['confirmed']).lower(),side])
                    stress_cells[cell] += 1
    if not preview:
        for split, mode_counts in EXPECTED.items():
            n = mode_counts['READ']
            split_groups = groups[split]
            audit.check(len(split_groups) == n, 'unique_base_contexts_' + split, len(split_groups))
            write_groups = [g for g in split_groups.values() if any(r['mode']=='WRITE' for r in g)]
            audit.check(len(write_groups) == n // 4, 'WRITE_matched_fourway_group_count_' + split, len(write_groups))
            for group in split_groups.values():
                base = group[0]['metadata']['base_context_id']
                modes = Counter(x['mode'] for x in group)
                audit.check(modes['READ'] == 1 and modes['SUBTASK'] == 1 and modes['WRITE'] in {0,4},
                            'one_READ_SUBTASK_and_optional_fourway_WRITE_per_base', dict(modes), base)
                writes = [x for x in group if x['mode']=='WRITE']
                if writes:
                    conditions = {x['metadata']['memory_condition'] for x in writes}
                    audit.check(conditions == {'WRITE','KEEP','CORRECT','EMPTY'}
                                and len({x['evidence']['memory_trace']['controlled_target_slot'] for x in writes}) == 1
                                and len({x['metadata']['active_revision'] for x in writes}) == 1,
                                'matched_WRITE_four_conditions_same_focus_revision', sorted(conditions), base)
                    focus = writes[0]['evidence']['memory_trace']['controlled_target_slot']
                    nontarget = [canonical([x for x in r['context']['selected_memory'] if x['slot_id'] != focus]) for r in writes]
                    audit.check(len(set(nontarget)) == 1, 'matched_WRITE_nontarget_slots_identical', nontarget, base)
            audit.check(stress_counts[split] == n // 5, 'SUBTASK_real_fault_stress_count_' + split, stress_counts[split])
            for side in ['left','right']:
                for confirmed in ['false','true']:
                    cell = ':'.join([split,'sample_identity','CORRECT',confirmed,side])
                    audit.check(stress_cells[cell] > 0,
                                'SUBTASK_wrong_identity_goal_priority_coverage_' + split,
                                {'cell':cell,'count':stress_cells[cell]})
            for op in ['EMPTY','KEEP','CORRECT']:
                audit.check(dict(active_counts[split][op]) == {str(r):n//20 for r in ACTIVE_REVISIONS},
                            'active_operation_revision_balance_' + split + '_' + op, dict(active_counts[split][op]))
            for side in ['left','right']:
                for revision in ACTIVE_REVISIONS:
                    for confirmed in ['false','true']:
                        cell = ':'.join([split,confirmed,side,str(revision)])
                        audit.check(WRITE_wrong_flag_cells[cell] >= 1,
                                    'WRITE_wrong_prebank_confirmed_goal_revision_coverage_' + split,
                                    {'cell':cell,'actual_count':WRITE_wrong_flag_cells[cell]})
            for mode in MODES:
                audit.check(dict(side_counts[split + ':' + mode]) == {'left':n//2,'right':n//2},
                            'goal_side_balance_' + split + '_' + mode, dict(side_counts[split + ':' + mode]))
        audit.check(set(subtask_bases) == set(read_ids_by_base)
                    and all(len(x) == 1 for x in subtask_bases.values()),
                    'SUBTASK_uses_exactly_one_per_distinct_READ_base', len(subtask_bases))
    return {'stress_counts':dict(stress_counts),
            'WRITE_wrong_prebank_confirmed_goal_revision_cells':dict(WRITE_wrong_flag_cells),
            'runtime_input_guard_rejections':guard_rejected_rows,
            'stress_actual_focus_operation_confirmed_goal_cells':dict(stress_cells),
            'active_WRITE_revision_operation_counts':{s:{o:dict(v) for o,v in ops.items()} for s,ops in active_counts.items()},
            'goal_template_counts':{s:dict(v) for s,v in template_counts.items()},
            'goal_side_counts':{s:dict(v) for s,v in side_counts.items()},
            'unique_target_strings':{s:len(v) for s,v in target_strings.items()},
            'distinct_protocol_base_contexts_by_split':{s:len(g) for s,g in groups.items()},
            'independence_limit':'Goal paraphrases and memory variants of the same physical observation remain correlated augmentations, even if their actual token prefixes are unique.',
            'label_scope':'Formal goal-role / actual issued-control history / native memory protocol, not physical grounding or success.'}


def preview_source_index(rows, audit):
    """Read only the preview's completed episode; never the growing manifest."""
    from openpi.shared.image_tools import resize_with_pad
    episodes, observations = {}, {}
    for row in rows:
        sid, ev = row['sample_id'], row['evidence']
        path = Path(ev['source_episode_json'])
        if row['episode_id'] not in episodes:
            source_receipt(path, ev['source_episode_json_sha256'], audit, 'preview_source_episode', sid)
            ep = json.loads(path.read_text())
            audit.check(ep.get('completed') is True and ep.get('worker_shutdown_confirmed') is True,
                        'preview_source_episode_completed', row['episode_id'], sid)
            ep['episode_json_path'] = str(path)
            ep['episode_json_sha256'] = ev['source_episode_json_sha256']
            issued = [json.loads(x) for x in Path(ep['issued_subtasks_path']).read_text().splitlines() if x.strip()]
            decisions = [{'decision_id':x['event_id'],'start_step':x['issued_at_step'],
                          'end_step':x['planned_window_end_step'],'generated_subtask':x['text']} for x in issued]
            controls = []
            for line in Path(ep['transitions_path']).read_text().splitlines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                for k in range(chunk['control_start'], chunk['control_end']):
                    event = next(x for x in decisions if x['start_step'] <= k < x['end_step'])
                    controls.append({'step_index':k, 'decision_id':event['decision_id']})
            episodes[row['episode_id']] = {'record':ep,'decisions':decisions,'controls':controls}
        obs_path = str(Path(row['observation']['npz_path']).resolve())
        if obs_path not in observations:
            ep = episodes[row['episode_id']]['record']
            stored = next(x for x in ep['observations'] if str(Path(x['npz_path']).resolve()) == obs_path)
            source_receipt(Path(obs_path), stored['sha256'], audit, 'preview_actual_NPZ', sid)
            with np.load(obs_path, allow_pickle=False) as data:
                state = data['state'].copy()
                camera_shas = [hashlib.sha256(data[k].tobytes()).hexdigest() for k in CAMERAS]
                model_shas = [hashlib.sha256(np.asarray(resize_with_pad(data[k],224,224)).tobytes()).hexdigest() for k in CAMERAS]
            observations[obs_path] = {'episode_id':row['episode_id'],'family_id':ep['family_id'],
                                      'split':ep['split'],'step':stored['step'],
                                      'pixel_key':digest(camera_shas),'model_pixel_key':digest(model_shas),
                                      'state':state,'observation':stored,'npz_sha256':stored['sha256']}
    return episodes, observations


def audit_dataset_manifest(path, labels, capture_path, tokenizer_path, norm_path, audit):
    from rlinf.models.embodiment.openpi.recap_tokenizer import PROMPT_VERSION
    info = json.loads(path.read_text())
    audit.check(info.get('schema') == 'rm65b_dynamic_history_protocol_3000_v1'
                and info.get('history_version') == HISTORY_VERSION and info.get('prompt_version') == PROMPT_VERSION,
                'dataset_manifest_schema_versions', {k:info.get(k) for k in ['schema','history_version','prompt_version']})
    source_receipt(ROOT/'scripts/build_history3000.py', info.get('builder_script_sha256'),
                   audit, 'label_builder_script')
    audit.check(str(Path(info['capture_manifest']).resolve()) == str(capture_path.resolve())
                and info.get('capture_manifest_sha256') == file_sha(capture_path),
                'dataset_frozen_complete_source_manifest', str(capture_path))
    for name, actual in [('tokenizer',tokenizer_path),('norm_stats',norm_path)]:
        audit.check(str(Path(info[name]['path']).resolve()) == str(actual.resolve())
                    and info[name]['sha256'] == file_sha(actual), 'dataset_' + name + '_asset_receipt', str(actual))
    declared = info.get('labels', {})
    audit.check(set(declared) == set(EXPECTED)
                and {str(Path(x['path']).resolve()) for x in declared.values()} == {str(x.resolve()) for x in labels},
                'dataset_exact_three_split_files', sorted(declared))
    for split, item in declared.items():
        file = Path(item['path'])
        source_receipt(file, item['sha256'], audit, 'dataset_label_file_' + split)
        audit.check(file.stat().st_size == item['bytes'], 'dataset_label_bytes_' + split, item['bytes'])
    audit.check(info.get('training_enabled') is False and info.get('optimizer_updates') == 0
                and info.get('MEM_ready') is False and info.get('video_history_rows') == 0
                and info.get('physical_skill_success_claimed') is False
                and info.get('visual_grounding_verified') is False,
                'dataset_manifest_training_and_capability_boundaries',
                {k:info.get(k) for k in ['training_enabled','optimizer_updates','MEM_ready','video_history_rows',
                                        'physical_skill_success_claimed','visual_grounding_verified']})
    audit.check(info.get('source_physical_context_counts') == {'train':288,'val':204,'test':204},
                'dataset_manifest_real_observation_counts', info.get('source_physical_context_counts'))
    return info


def actual_training_loader_gate(labels, rows, audit, *, preview=False):
    """Exercise data guards only; no model/optimizer/runtime is constructed."""
    from rlinf.runners.hierarchical_recap_runner import load_verified_high_samples
    if preview:
        rejected = []
        for path in labels:
            try:
                load_verified_high_samples([path])
            except ValueError as exc:
                rejected.append({'path':str(path),'reason':str(exc)})
            else:
                audit.check(False, 'preview_must_not_enter_training_loader', str(path))
        return {'scope':'preview loader rejection only','rejected':rejected,'training_started':False}
    by_split = {}
    for path in labels:
        file_rows = load_rows([path])
        split = file_rows[0]['split'] if file_rows else None
        audit.check(bool(file_rows) and all(x['split'] == split for x in file_rows),
                    'one_explicit_split_per_label_file', str(path))
        by_split[split] = (path,file_rows)
    eval_rows = [x for x in rows if x['split'] in {'val','test'}]
    train_path, train_rows = by_split['train']
    loaded = load_verified_high_samples([train_path],
        eval_seeds={x['seed'] for x in eval_rows}, excluded_families={x['family_id'] for x in eval_rows})
    audit.check(loaded == train_rows and len(loaded) == 3000,
                'actual_training_loader_accepts_exact_3000_unmodified', len(loaded))
    rejected = []
    for split in ['val','test']:
        try:
            load_verified_high_samples([by_split[split][0]])
        except ValueError as exc:
            rejected.append({'split':split,'reason':str(exc)})
        else:
            audit.check(False, 'actual_training_loader_rejects_' + split, str(by_split[split][0]))
    return {'training_rows_loaded':len(loaded),'records_unchanged':loaded==train_rows,
            'eval_files_rejected':rejected,'training_started':False,'model_or_optimizer_loaded':False}


def actual_controller_ID_contract():
    """Exercise real controller ID creation with CPU text-only actor stubs.

    No shared ID helper or builder is imported as the expected oracle. This
    catches a runtime/controller drift that same-row tokenizer parity misses.
    """
    from rlinf.workers.rollout.hf.hierarchical_recap_worker import (
        GeneratedText, HierarchicalReCapRolloutWorker,
    )
    class ActorStub:
        version = 'CPU_controller_ID_contract_no_model'
        def generate(self, observation, mode, context):
            text = {'READ':'{"read_ids":["sample_identity"]}',
                    'WRITE':'{"operations":[]}',
                    'SUBTASK':'Attempt the explicitly requested sample.'}[mode]
            return GeneratedText(text=text, token_ids=[1], ended_with_eos=True)
    catalogue = [{'slot_id':'sample_identity','description':'Requested role.'}]
    controller = HierarchicalReCapRolloutWorker(ActorStub(), None, catalogue,
                                                strict_memory_protocol=True)
    cases = []
    for step in [0,32,64,256]:
        bank = EpisodeMemory(catalogue, 'CPU_fixture_env', 'CPU_fixture_episode', strict_protocol=True)
        decision = controller._decision({'instruction':'Attempt the requested role.'}, bank, step, [])
        expected = f'h{step:06d}'
        if decision.decision_id != expected:
            raise AssertionError(f'Actual controller ID drift: step{step}, actual{decision.decision_id!r}, expected{expected!r}')
        cases.append({'control_step':step,'actual_decision_id':decision.decision_id,'expected':expected})
    module_path = Path(sys.modules[HierarchicalReCapRolloutWorker.__module__].__file__)
    return {'status':'passed_CPU_controller_entry_stub_no_model_no_simulator',
            'controller_source':{'path':str(module_path),'sha256':file_sha(module_path)},'cases':cases}


def self_test():
    """Adversarial contract fixtures, without model/runtime dependencies."""
    assert len({serialize_history(x) for x in [None, '', [], '[]', '  [] ']}) == 1
    event = {'type':'issued_subtask','decision_id':'h000000','text':'Attempt left grasp.',
             'issued_at_step':0,'executed_until_step':7}
    assert serialize_history([event], allow_legacy=False) == serialize_history(canonical([event]), allow_legacy=False)
    rejects = []
    for name, value, opts in [
        ('legacy', ['Attempt left grasp.'], {'allow_legacy':False}),
        ('future', [event], {'observed_until_step':6}),
        ('completed', [{**event,'completed':True}], {}),
        ('bool_clock', [{**event,'issued_at_step':False}], {}),
        ('negative_clock', [{**event,'issued_at_step':-1}], {}),
        ('float_clock', [{**event,'issued_at_step':0.0}], {}),
        ('duplicate_id', [event,{**event,'issued_at_step':7}], {}),
        ('overlap', [event,{**event,'decision_id':'d1','issued_at_step':6}], {}),
        ('end_before_start', [{**event,'issued_at_step':8}], {}),
    ]:
        try:
            normalize_history(value, **opts)
        except ValueError:
            rejects.append(name)
        else:
            raise AssertionError('Failed to reject '+name)
    assert history_from_context({'visible_history': [], 'history': '[]'}, allow_legacy=False) == []
    try:
        history_from_context({'visible_history': [],'history':[event]}, allow_legacy=False)
    except ValueError:
        rejects.append('alias_conflict')
    else:
        raise AssertionError('Conflicting aliases accepted')
    a = Audit()
    source = [{'decision_id':'issued_0000','start_step':0,'end_step':32,'generated_subtask':'Attempt left grasp.'}]
    controls = [{'step_index':k,'decision_id':'issued_0000'} for k in range(7)]
    row = {'sample_id':'selftest','observed_until_step':7,'decision_step':7,
           'metadata':{'history_version':HISTORY_VERSION,'history_id_encoding':'runtime_control_step_h6'},
           'context':{'visible_history':[event]},
           'evidence':{'source_history_id_map':[{'source_event_id':'issued_0000',
                                                'decision_id':'h000000','issued_at_step':0}]}}
    compare_actual_source_history(row, source, controls, a)
    assert not a.issues, a.issues
    future = deepcopy(row)
    future['context']['visible_history'][0]['executed_until_step'] = 32
    b = Audit()
    compare_actual_source_history(future, source, controls, b)
    assert any(x['code']=='strict_history' for x in b.issues)
    source_id_row = deepcopy(row)
    source_id_row['context']['visible_history'][0]['decision_id'] = 'issued_0000'
    # Syntax and identical-row serialization can agree while runtime IDs differ.
    assert serialize_history(source_id_row['context']['visible_history']) == serialize_history(
        deepcopy(source_id_row['context']['visible_history']))
    source_id_audit = Audit()
    compare_actual_source_history(source_id_row, source, controls, source_id_audit)
    assert any(x['code']=='label_history_matches_runtime_control_step_ID' for x in source_id_audit.issues)
    assert any(x['code']=='history_exact_actual_source_prefix' for x in source_id_audit.issues)
    forged_map = deepcopy(row)
    forged_map['evidence']['source_history_id_map'][0]['source_event_id'] = 'issued_9999'
    forged_audit = Audit()
    compare_actual_source_history(forged_map, source, controls, forged_audit)
    assert any(x['code']=='label_ID_map_exact_raw_source_event_to_runtime_control_step' for x in forged_audit.issues)
    # Formal goal facts outrank a counterfactual confirmed-but-wrong slot.
    slots = [{'slot_id':'sample_identity','description':'Explicit first initial role.'},
             {'slot_id':'current_progress','description':'Actual issued-control count; no completion assertion.'}]
    facts = {'sample_identity':'The first requested sample is initially on the left.',
             'current_progress':'Last issued h000000 for initial-left; executed_control_steps=32; physical completion unverified.'}
    fixture = {'sample_id':'synthetic_memory_contract_fixture','mode':'SUBTASK','episode_id':'synthetic_episode',
               'observed_until_step':32,'observation':{'npz_path':'/synthetic/not_dataset/obs32.npz'},
               'context':{'catalogue':slots,'read_ids':['sample_identity','current_progress']},
               'metadata':{'memory_condition':'CORRECT','active_revision':1,'stress_without_writer_repair':True},
               'evidence':{}}
    bank = EpisodeMemory(slots, 'history3000_bank', 'synthetic_episode', strict_protocol=True)
    source = {'kind':'authored_counterfactual_memory_condition','step':32,'observed_until_step':32,
              'observation_refs':[{'npz_path':fixture['observation']['npz_path']}]}
    seeds = []
    for slot, value in [('sample_identity','The first requested sample is initially on the right.'),
                         ('current_progress',facts['current_progress'])]:
        bank.read([slot])
        operation = {'op':'WRITE','slot_id':slot,'value':value,'confirmed':True}
        response = bank.apply(operation, source=source)
        seeds.append({'read_ids':[slot],'operation':operation,'source':source,'response':response})
    selected = bank.read(fixture['context']['read_ids'])
    actor = bank.actor_view(fixture['context']['read_ids'], [])
    fixture['context']['actor_memory'] = actor
    fixture['evidence']['memory_trace'] = {'seed_rounds':seeds,
        'read_ids':fixture['context']['read_ids'],'selected_memory':selected,
        'operations':[],'operation_responses':[],'actor_memory':actor,'native_strict_protocol':True,
        'condition':'CORRECT','controlled_target_slot':'sample_identity',
        'controlled_active_revision':1,'stress_without_writer_repair':True}
    c = Audit()
    result = memory_replay(fixture, facts, c)
    assert not c.issues and result['confirmed_wrong_prebank'], c.issues
    malformed = deepcopy(fixture)
    malformed['context']['actor_memory'] = actor['selected']
    d = Audit()
    validate_visible_memory(malformed, d)
    assert any(x['code']=='subtask_native_nested_memory' for x in d.issues)
    invariant = []
    for revision in [1,2,4,7,10]:
        entry = {'slot_id':'sample_identity','status':'active','value':facts['sample_identity'],
                 'confirmed':False,'revision':revision}
        invariant.append(formal_expected_operation([entry],'sample_identity',facts))
    assert all(x == invariant[0] for x in invariant)
    wrong_priority = []
    for confirmed in [False,True]:
        wrong_entry = {'slot_id':'sample_identity','status':'active',
                       'value':'The first requested sample is initially on the right.',
                       'confirmed':confirmed,'revision':4}
        wrong_priority.append(formal_expected_operation([wrong_entry],'sample_identity',facts))
    assert wrong_priority[0] == wrong_priority[1] == [
        {'op':'CORRECT','slot_id':'sample_identity','value':facts['sample_identity'],'confirmed':True}]
    return {'status':'passed_cpu_only','negative_fixtures_rejected':rejects,
            'source_clip_fixture_passed':True,'future_unclipped_fixture_rejected':True,
            'empty_and_serialized_typed_history_parity':True,
            'native_confirmed_wrong_identity_stress_replayed':True,
            'list_actor_memory_negative_fixture_rejected':True,
            'active_revision_only_rule_invariance':True,
            'wrong_value_requires_CORRECT_for_confirmed_true_and_false':True,
            'same_row_static_source_ID_negative_fixture_rejected':True,
            'forged_raw_to_model_ID_map_negative_fixture_rejected':True,
            'actual_controller_runtime_ID_contract':actual_controller_ID_contract()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', nargs='*', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--capture-manifest', type=Path)
    p.add_argument('--source-only', action='store_true')
    p.add_argument('--allow-partial-capture', action='store_true')
    p.add_argument('--schema-preview', action='store_true')
    p.add_argument('--config', type=Path, default=ROOT/'configs/rm65b_mem_protocol.yaml')
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--norm-stats', type=Path)
    p.add_argument('--dataset-manifest', type=Path)
    a = p.parse_args()
    validator_sha_at_start = file_sha(Path(__file__))
    if a.self_test:
        report = {'version':VERSION,'self_test':self_test(),'GPU_loaded':False,'training_started':False}
    elif a.source_only and a.capture_manifest:
        audit = Audit()
        manifest, episodes, observations, summary = audit_capture(
            a.capture_manifest.resolve(), audit, allow_partial=a.allow_partial_capture)
        complete = manifest.get('completed') is True
        report = {'version': VERSION,
                  'status': ('source_receipts_passed' if complete else 'partial_source_receipts_passed')
                            if not audit.issues else 'source_receipts_failed',
                  'GPU_loaded': False, 'training_started': False,
                  'capture_manifest': {'path':str(a.capture_manifest.resolve()),
                                       'sha256':file_sha(a.capture_manifest)},
                  'source_summary':summary, 'checks':dict(audit.checks),
                  'issues':audit.issues, 'warnings':audit.warnings,
                  'all_capture_completed':complete, 'dataset_ready':False,
                  'scope':'Source receipt audit only; no dataset/model capability claim.'}
    elif a.labels:
        import yaml
        from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer
        audit = Audit()
        file_paths = [*a.labels,a.config]
        input_sha_before = {str(x.resolve()):file_sha(x) for x in file_paths}
        rows = load_rows(a.labels)
        cfg = yaml.safe_load(a.config.read_text())
        tokenizer_path = a.tokenizer or Path(cfg['tokenizer']['text_tokenizer_path'])
        norm_path = a.norm_stats or Path(cfg['model']['openpi_data']['norm_stats_path'])
        for file in [tokenizer_path,norm_path]:
            input_sha_before[str(file.resolve())] = file_sha(file)
        norm_json = json.loads(norm_path.read_text())
        stats = norm_json.get('norm_stats', norm_json)['state']
        tokenizer = RecapTokenizer.from_local_paths(tokenizer_path, max_prefix_tokens=1400, max_target_tokens=96)
        if a.schema_preview:
            episodes, observations = preview_source_index(rows, audit)
            source_summary = {'scope':'Preview referenced completed source only; no growing manifest read or complete source acceptance claim.',
                              'episodes':len(episodes),'observations':len(observations)}
            complete_source = False
        else:
            if not a.capture_manifest or a.allow_partial_capture:
                raise SystemExit('Full label acceptance requires stable --capture-manifest; partial source is not dataset-ready.')
            dataset_manifest = a.dataset_manifest or a.labels[0].parent/'manifest.json'
            if not dataset_manifest.is_file():
                raise SystemExit('Full acceptance requires the builder dataset manifest receipt, not labels alone.')
            for file in [a.capture_manifest,dataset_manifest]:
                input_sha_before[str(file.resolve())] = file_sha(file)
            manifest, episodes, observations, source_summary = audit_capture(a.capture_manifest.resolve(), audit)
            complete_source = manifest.get('completed') is True
            audit_dataset_manifest(dataset_manifest, a.labels, a.capture_manifest, tokenizer_path, norm_path, audit)
        row_summary = summarize_rows(rows, audit, preview=a.schema_preview)
        controller_ID_summary = actual_controller_ID_contract()
        protocol_summary = audit_protocol_rows(rows, episodes, observations, audit, preview=a.schema_preview)
        actual_summary = effective_inputs(rows, episodes, observations, tokenizer, stats, audit, preview=a.schema_preview)
        training_loader_summary = actual_training_loader_gate(a.labels, rows, audit, preview=a.schema_preview)
        for key, old_sha in input_sha_before.items():
            audit.check(file_sha(Path(key)) == old_sha, 'immutable_input_file_SHA_before_after', key)
        report = {'version':VERSION,
                  'status':('schema_preview_passed_not_trainable' if a.schema_preview else 'protocol_dataset_acceptance_passed')
                           if not audit.issues else 'protocol_dataset_acceptance_failed',
                  'GPU_loaded':False,'training_started':False,
                  'protocol_dataset_ready':not a.schema_preview and complete_source and not audit.issues,
                  'MEM_ready':False,'video_history_rows':0,
                  'physical_source_groups_disjoint_verified':complete_source and not audit.issues,
                  'physical_grounding_verified':False,'motor_skill_retention_verified':False,
                  'source_summary':source_summary,'row_summary':row_summary,
                  'formal_protocol_summary':protocol_summary,'actual_effective_inputs':actual_summary,
                  'actual_controller_runtime_ID_contract':controller_ID_summary,
                  'actual_training_loader_gate':training_loader_summary,
                  'input_files':[{'path':str(x.resolve()),'sha256':file_sha(x),'bytes':x.stat().st_size}
                                 for x in [*a.labels,a.config,tokenizer_path,norm_path]],
                  'checks':dict(audit.checks),'issues':audit.issues,'warnings':audit.warnings,
                  'scope':'Independent CPU protocol-supervision dataset validation; no model inference, optimizer, SR or physical grounding claim.'}
        if not a.schema_preview:
            report['dataset_manifest'] = {'path':str(dataset_manifest.resolve()),'sha256':file_sha(dataset_manifest)}
    else:
        raise SystemExit('Choose --self-test, --source-only --capture-manifest, or --labels [--schema-preview].')
    report['validator_script'] = {'path':str(Path(__file__).resolve()),'sha256_at_start':validator_sha_at_start,
                                  'sha256_at_end':file_sha(Path(__file__))}
    if report['validator_script']['sha256_at_start'] != report['validator_script']['sha256_at_end']:
        raise RuntimeError('Validator source changed during execution; rerun stable script.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(canonical({'path':str(a.output),'sha256':file_sha(a.output),
                     'status':report.get('status',report.get('self_test',{}).get('status')),
                     'issues':len(report.get('issues',[]))}))
    if report.get('issues'):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
