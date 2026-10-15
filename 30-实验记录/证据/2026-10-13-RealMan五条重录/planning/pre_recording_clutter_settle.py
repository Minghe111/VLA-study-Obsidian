"""Bounded, ordinary native physics before a new recording's real t0.

This helper is additive and is not a replacement for official initialization
or for the unchanged 1 mm/no-contact execution guard.  It has no actor, robot,
drive, velocity, shape, material, SDK, wake, render or random-number writes.
The caller inserts it after its complete original prepare and before capturing
randomization/hardware/environment/expert baselines.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import importlib
import json
import math
import numbers
from pathlib import Path

import numpy as np

PHYSICS_DT_S = .004
DEFAULT_MAX_EXTRA_S = 4.0
DEFAULT_WINDOW_S = 1.0
DEFAULT_WINDOW_PAIRWISE_BUDGET_M = .0001


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def producer_pins(task=None, source_hashes=None):
    """Include the helper and exact native extraction/coverage producers."""
    paths = {str(Path(__file__).resolve()): _sha(__file__)}
    for name in ('randomized_environment_guard', 'full_mesh_path_audit'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        paths[str(path)] = _sha(path)
    historical = getattr(task, '_rgb50_initial_extra_settle_receipt', None)
    if historical is not None:
        if not isinstance(historical, dict) or not isinstance(historical.get('source_hashes'), dict):
            raise RuntimeError('Missing exact original extra-settle producer identities')
        paths.update(historical['source_hashes'])
    if source_hashes is not None:
        if not isinstance(source_hashes, dict):
            raise TypeError('Protected original source hashes must be a mapping')
        for path, digest in source_hashes.items():
            resolved = str(Path(path).resolve())
            if resolved in paths and paths[resolved] != digest:
                raise RuntimeError('Conflicting initialization source identity')
            paths[resolved] = digest
    for path, digest in paths.items():
        if not isinstance(digest, str) or not Path(path).is_file() or _sha(path) != digest:
            raise RuntimeError('Initialization source bytes differ: ' + path)
    return paths


def settle_plan(dt, max_extra_s=DEFAULT_MAX_EXTRA_S, window_s=DEFAULT_WINDOW_S,
                window_pairwise_budget_m=DEFAULT_WINDOW_PAIRWISE_BUDGET_M):
    for name, value in (('physics timestep', dt), ('max extra seconds', max_extra_s),
                        ('window seconds', window_s), ('window vertex budget', window_pairwise_budget_m)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            raise ValueError('Invalid finite ' + name)
    if abs(float(dt) - PHYSICS_DT_S) > 1e-9:
        raise ValueError('Pre-recording settling requires the actual native 4 ms clock')
    if not 0 < window_s <= max_extra_s <= 8:
        raise ValueError('Require a positive stability window and finite preparation <=8 seconds')
    if not 0 < window_pairwise_budget_m <= DEFAULT_WINDOW_PAIRWISE_BUDGET_M:
        raise ValueError('Preparation window budget must be positive and <=0.1 mm')
    maximum, window = int(round(max_extra_s / PHYSICS_DT_S)), int(round(window_s / PHYSICS_DT_S))
    if abs(maximum * PHYSICS_DT_S - max_extra_s) > 1e-9 or abs(window * PHYSICS_DT_S - window_s) > 1e-9:
        raise ValueError('Preparation durations must contain exact complete 4 ms steps')
    return {'physics_dt_s': float(dt), 'maximum_extra_physics_steps': maximum,
            'requested_maximum_extra_seconds': float(max_extra_s),
            'maximum_extra_simulated_seconds': maximum * float(dt),
            'requested_window_seconds': float(window_s),
            'window_physics_steps': window, 'window_simulated_seconds': window * float(dt),
            'window_pairwise_vertex_displacement_budget_m': float(window_pairwise_budget_m),
            'window_proof': 'For every component, sum the maximum displacement of its actual vertices over all adjacent 4 ms samples. Triangle inequality bounds every vertex displacement between every pair of samples by this total variation.'}


def _matrix(pose):
    matrix = np.asarray(pose.to_transformation_matrix(), dtype=float).copy()
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-4, rtol=0)
            or np.linalg.det(matrix[:3, :3]) < .9998):
        raise RuntimeError('Invalid actual native rigid pose')
    return matrix


def _records(task):
    guard = importlib.import_module('randomized_environment_guard')
    records = guard._snapshot_clutter(task)
    coverage = guard._coverage(records, task)
    if not coverage['passed']:
        raise RuntimeError('Incomplete official native clutter coverage')
    # Official clean draws or incomplete placements are rejected for this
    # explicitly requested ten-clutter experiment, without changing RNG.
    if coverage['expected_clutter_count'] != 10:
        raise RuntimeError('This preparation requires all ten official generated clutter actors')
    keys = [(r['clutter_index'], r['component_index']) for r in records]
    if len(keys) != len(set(keys)) or len({id(r['entity']) for r in records}) != len(records):
        raise RuntimeError('Duplicate or ambiguous native clutter component identity')
    for r in records:
        v = np.asarray(r['vertices'], dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
            raise RuntimeError('Incomplete or nonfinite native clutter vertices')
        r['vertices'] = v.copy()
        r['geometry_sha256'] = hashlib.sha256(v.tobytes()).hexdigest()
        _matrix(r['entity'].pose)
    return records, coverage


def _identity(records):
    return [(r['clutter_index'], r['component_index'], id(r['entity']), id(r['component']),
             r['entity_name'], r['shape_count'], r['geometry_sha256']) for r in records]


def _poses(records):
    return [_matrix(r['entity'].pose) for r in records]


def _world_vertices(records, poses):
    return [r['vertices'] @ m[:3, :3].T + m[:3, 3] for r, m in zip(records, poses)]


def _displacements(a, b):
    return [float(np.linalg.norm(x - y, axis=1).max()) for x, y in zip(a, b)]


def _active_contacts(task, records):
    clutter = {id(r['entity']) for r in records}
    robot = set()
    for side in ('left', 'right'):
        art = getattr(task.robot, side + '_entity', None)
        if art is not None:
            robot.update(id(link.entity) for link in art.links)
    bottle = id(task.bottle.actor)
    events = []
    for contact in task.scene.get_contacts():
        ids = {id(body.entity) for body in contact.bodies}
        if not ids.intersection(clutter) or not ids.intersection(robot | {bottle}):
            continue
        for point in contact.points:
            impulse, normal = np.asarray(point.impulse, dtype=float), np.asarray(point.normal, dtype=float)
            separation = float(point.separation)
            if (impulse.shape != (3,) or normal.shape != (3,) or not np.isfinite(impulse).all()
                    or not np.isfinite(normal).all() or not math.isfinite(separation) or np.linalg.norm(normal) < 1e-10):
                raise RuntimeError('Invalid actual native clutter contact')
            normal_impulse = float(abs(impulse @ normal / np.linalg.norm(normal)))
            if normal_impulse > 1e-10 or separation < -.00002:
                events.append({'entities': [body.entity.name for body in contact.bodies],
                               'kind': 'bottle_clutter' if bottle in ids else 'robot_clutter',
                               'normal_impulse_Ns': normal_impulse, 'separation_m': separation})
    return events


def _actual_bottle(task, spec):
    """Reapply the unchanged original native geometry/class/nearest checks."""
    from sapien.physx import PhysxRigidBaseComponent
    fma = importlib.import_module('full_mesh_path_audit')
    comps = [c for c in task.bottle.actor.get_components()
             if isinstance(c, PhysxRigidBaseComponent) and len(c.collision_shapes)]
    if len(comps) != 1:
        raise RuntimeError('Exactly one current collidable native bottle component required')
    vertices, _ = fma._link_vertices(comps[0])
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise RuntimeError('Missing finite native bottle collision vertices')
    matrix = _matrix(task.bottle.get_pose())
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    center = (world.min(0) + world.max(0)) / 2
    shoulders = {s: np.asarray(task.robot.left_entity.find_link_by_name(
        ('l_' if s == 'left' else 'r_') + 'link1').get_pose().p, dtype=float) for s in ('left', 'right')}
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in shoulders.values()):
        raise RuntimeError('Missing actual fixed shoulder origins')
    distances = {s: float(np.linalg.norm(center - v)) for s, v in shoulders.items()}
    nearest = min(distances, key=distances.get)
    axis = matrix[:3, 1]
    upright = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0, 1))))
    heading = float(np.degrees(np.arctan2(axis[1], axis[0])))
    kind, arm = spec['pose_class'], spec['arm']
    margin = abs(distances['left'] - distances['right'])
    if nearest != arm or margin < .020:
        raise RuntimeError('Prepared arm is not unambiguously nearest')
    if kind not in ('upright', 'horizontal', 'diagonal'):
        raise RuntimeError('Unsupported prepared bottle pose class')
    if (kind == 'upright' and upright > 5) or (kind != 'upright' and upright < 85):
        raise RuntimeError('Prepared orientation class differs')
    heading_distance = min(heading % 180, 180 - heading % 180)
    if kind == 'horizontal' and heading_distance > 12:
        raise RuntimeError('Prepared horizontal heading exceeds12deg')
    if kind == 'diagonal' and not 25 <= heading_distance <= 75:
        raise RuntimeError('Prepared diagonal heading outside25..75deg')
    q = np.asarray(task.robot.left_entity.get_qpos(), dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
        raise RuntimeError('Invalid actual robot qpos after preparation')
    return {'initial_pose': matrix.tolist(), 'actual_bottle_center_world': center.tolist(),
            'actual_collision_vertices': len(vertices), 'actual_collision_shape_count': len(comps[0].collision_shapes),
            'nearest_arm': nearest, 'selected_arm': arm,
            'nearest_rule': 'Euclidean distance from actual collision bbox center to fixed shoulder joint1 origins',
            'shoulder_origins_world': {s: v.tolist() for s, v in shoulders.items()}, 'arm_distances_m': distances,
            'nearest_margin_m': margin, 'actual_axis_world': axis.tolist(), 'upright_angle_deg': upright,
            'actual_axis_yaw_deg': heading, 'pose_class_rechecked': kind,
            'initial_robot_full_qpos_rad': q.tolist(), 'original_orientation_and_nearest_checks_reapplied': True}


def _row(records, poses, previous_world, initial_world, window, step, plan, task):
    world = _world_vertices(records, poses)
    delta = _displacements(world, previous_world)
    original = _displacements(world, initial_world)
    window.append((step, poses, delta))
    bounds = [0.] * len(records)
    if len(window) == plan['window_physics_steps'] + 1:
        bounds = np.sum([entry[2] for entry in list(window)[1:]], axis=0).tolist()
    else:
        bounds = None
    events = _active_contacts(task, records)
    components = []
    for i, (record, pose) in enumerate(zip(records, poses)):
        components.append({'category': record['category'], 'clutter_index': record['clutter_index'],
            'component_index': record['component_index'], 'entity_name': record['entity_name'],
            'entity_runtime_id': id(record['entity']), 'component_runtime_id': id(record['component']),
            'shape_count': record['shape_count'], 'geometry_sha256': record['geometry_sha256'],
            'pose_matrix': pose.tolist(), 'max_vertex_displacement_from_previous_m': delta[i],
            'max_vertex_displacement_from_preparation_start_m': original[i],
            'window_pairwise_vertex_displacement_upper_bound_m': None if bounds is None else bounds[i]})
    bound = None if bounds is None else max(bounds, default=0.)
    row = {'preparation_step': step, 'preparation_sim_time_before_s': (step - 1) * plan['physics_dt_s'],
           'preparation_sim_time_after_s': step * plan['physics_dt_s'],
           'components': components, 'max_vertex_displacement_from_previous_m': max(delta, default=0.),
           'max_vertex_displacement_from_preparation_start_m': max(original, default=0.),
           'worst_component_from_preparation_start': components[int(np.argmax(original))]['category'],
           'window_first_step': window[0][0], 'window_last_step': window[-1][0],
           'window_complete': bounds is not None,
           'window_pairwise_vertex_displacement_upper_bound_m': bound,
           'active_target_or_robot_clutter_contacts': events}
    return row, world


def settle_clutter(task, out, scene, spec, *, source_hashes=None,
                   max_extra_s=DEFAULT_MAX_EXTRA_S, window_s=DEFAULT_WINDOW_S,
                   window_pairwise_budget_m=DEFAULT_WINDOW_PAIRWISE_BUDGET_M):
    """Run exactly one bounded pre-t0 preparation or persist rejection evidence."""
    if getattr(task.robot, 'communication_flag', None) is not False:
        raise RuntimeError('Explicit simulated robot IO=false required')
    for name in ('_rerecord_clutter_settle_receipt', '_rerecord_environment_guard', '_rerecord_randomization_receipt'):
        if hasattr(task, name):
            raise RuntimeError('Clutter preparation must run once before all recording baselines: ' + name)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = settle_plan(task.scene.get_timestep(), max_extra_s, window_s, window_pairwise_budget_m)
    pins = producer_pins(task, source_hashes)
    historical = copy.deepcopy(getattr(task, '_rgb50_initial_extra_settle_receipt', None))
    records, coverage = _records(task)
    identities = _identity(records)
    before = _poses(records)
    initial_world = _world_vertices(records, before)
    previous_world = initial_world
    scene_before = copy.deepcopy(scene)
    bottle_before = _actual_bottle(task, spec)
    initial = [{'category': r['category'], 'entity_name': r['entity_name'],
                'entity_runtime_id': id(r['entity']), 'component_runtime_id': id(r['component']),
                'shape_count': r['shape_count'], 'geometry_sha256': r['geometry_sha256'],
                'pose_matrix': m.tolist()} for r, m in zip(records, before)]
    window = deque([(0, before, [0.] * len(records))], maxlen=plan['window_physics_steps'] + 1)
    receipt = {'version': 'pre_recording_clutter_settle_v1', 'passed': False, **plan,
        'completed_extra_physics_steps': 0, 'actual_extra_simulated_seconds': 0.,
        'coverage': coverage, 'initial_components': initial, 'final_components': None,
        'source_hashes_before': pins, 'source_hashes_after': None,
        'bottle_before': bottle_before, 'bottle_after': None,
        'historical_original_extra_settle_receipt': historical,
        'historical_original_extra_settle_receipt_preserved_exactly': False,
        'all_native_collision_component_identities_unchanged': False,
        'recording_starts_after_preparation_at_true_new_t0': True,
        'SDK24_record_every_physics_steps': 15, 'SDK24_record_interval_s': .06,
        'execution_vertex_drift_budget_m_unchanged': .001,
        'only_native_scene_step_mutates_physics': True, 'wake_calls': 0,
        'extra_random_draws': 0, 'native_actor_robot_drive_velocity_geometry_material_writes': 0,
        'hardware_io': False, 'trace_path': str((out / 'pre_recording_clutter_settle_steps.jsonl').resolve())}
    _json_write(out / 'pre_recording_clutter_settle_initial.json', {'components': initial, 'scene_sample': scene_before,
                                                               'bottle': bottle_before, 'coverage': coverage})
    try:
        with (out / 'pre_recording_clutter_settle_steps.jsonl').open('w') as trace:
            for step in range(1, plan['maximum_extra_physics_steps'] + 1):
                if abs(float(task.scene.get_timestep()) - plan['physics_dt_s']) > 1e-9:
                    raise RuntimeError('Native 4 ms clock changed during preparation')
                task.scene.step()
                receipt['completed_extra_physics_steps'] = step
                receipt['actual_extra_simulated_seconds'] = step * plan['physics_dt_s']
                poses = _poses(records)
                row, previous_world = _row(records, poses, previous_world, initial_world, window, step, plan, task)
                trace.write(json.dumps(row, ensure_ascii=False) + '\n')
                trace.flush()
                receipt['last_step'] = row
                if row['active_target_or_robot_clutter_contacts']:
                    raise RuntimeError('Prepared clutter has an active robot or target bottle contact')
                if row['window_complete'] and row['window_pairwise_vertex_displacement_upper_bound_m'] <= plan['window_pairwise_vertex_displacement_budget_m']:
                    receipt['passed'] = True
                    break
        if not receipt['passed']:
            raise RuntimeError('Native clutter did not stabilize within bounded pre-recording preparation')
        after_records, after_coverage = _records(task)
        if _identity(after_records) != identities or after_coverage != coverage:
            raise RuntimeError('Native clutter identity or geometry changed during preparation')
        receipt['all_native_collision_component_identities_unchanged'] = True
        if getattr(task, '_rgb50_initial_extra_settle_receipt', None) != historical:
            raise RuntimeError('Original wake/settle historical receipt changed')
        receipt['historical_original_extra_settle_receipt_preserved_exactly'] = True
        receipt['source_hashes_after'] = producer_pins(task, source_hashes)
        if receipt['source_hashes_after'] != pins:
            raise RuntimeError('Original/helper producer source bytes changed during preparation')
        actual = _actual_bottle(task, spec)
        receipt['bottle_after'] = actual
        receipt['final_components'] = copy.deepcopy(receipt['last_step']['components'])
        # These fields deliberately become the true final preparation pose.
        # Old metadata is retained as history, never relabeled as unchanged.
        scene['pre_clutter_settle_scene_sample'] = scene_before
        scene.update(actual)
        scene['pre_recording_clutter_settle'] = receipt
        scene['initial_pose_capture_scope'] = 'Actual native t0 after complete original prepare and bounded ordinary clutter preparation'
        # Keep the native Python wrappers alive: recreating a wrapper for the
        # same C++ component can otherwise change Python id without changing
        # the physical object. This is reference retention only.
        task._rerecord_clutter_settle_native_records = records
        task._rerecord_clutter_settle_receipt = receipt
        _json_write(out / 'scene_sample.json', scene)
        _json_write(out / 'pre_recording_clutter_settle_receipt.json', receipt)
        return receipt
    except BaseException as error:
        receipt['passed'] = False
        receipt['error'] = str(error)
        _json_write(out / 'pre_recording_clutter_settle_receipt.json', receipt)
        raise


def validate_before_expert(task, spec):
    """Call before the expert constructor registers producer/initial state."""
    receipt = getattr(task, '_rerecord_clutter_settle_receipt', None)
    if not isinstance(receipt, dict) or not receipt.get('passed'):
        raise RuntimeError('Expert requires a successful actual clutter preparation')
    pins = producer_pins(task, receipt['source_hashes_before'])
    if pins != receipt['source_hashes_before'] or pins != receipt['source_hashes_after']:
        raise RuntimeError('Clutter preparation producer source identity differs before expert')
    if getattr(task, '_rgb50_initial_extra_settle_receipt', None) != receipt['historical_original_extra_settle_receipt']:
        raise RuntimeError('Original wake/settle historical receipt changed before expert')
    actual = _actual_bottle(task, spec)
    if actual != receipt['bottle_after']:
        raise RuntimeError('Actual t0 bottle or robot state changed after preparation')
    records, coverage = _records(task)
    if coverage != receipt['coverage']:
        raise RuntimeError('Clutter coverage changed after preparation')
    for record, final in zip(records, receipt['final_components']):
        if (id(record['entity']) != final['entity_runtime_id']
                or id(record['component']) != final['component_runtime_id']
                or record['geometry_sha256'] != final['geometry_sha256']
                or not np.array_equal(_matrix(record['entity'].pose), np.asarray(final['pose_matrix']))):
            raise RuntimeError('Actual clutter t0 differs from completed preparation')
    return copy.deepcopy(receipt)
