"""Additive, zero-step reconciliation of an unloaded closure boundary.

The complete v13 audit stays on disk and remains the primary geometry record.
Only an exact selected finger's single sample can be reclassified, using its
immediately preceding REAL positive native contact, current zero normal impulse
and nonnegative native separation, and independently enumerated FCL penetration
depths within an independently measured event-state numerical budget. No controller, exemption,
physics parameter, canonical source, original sample or historical report is
changed. This is discrete recorded-state review, not a swept-volume certificate.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import inspect
import os
from pathlib import Path
import re
import sys

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')
LEGACY = ROOT.parent / 'deploy/realman_gen4/adjust_bottle_20261002'
VERSION = 'closure_boundary_v14'
FROZEN_CPU_CONSTRUCTOR_SHA256 = 'e5d13b16ef6aff9ad9624978e9c269a1bd4b86b4c526542bb03acaa9c139f490'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _finite_scalar(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def native_boundary_evidence(previous, current, link, selected_side, threshold):
    """Fail-closed native predicate; reconciled samples are never input here."""
    evidence = {'eligible': False, 'link': link, 'reason': None}
    def reject(reason):
        evidence['reason'] = reason
        return evidence
    prefix = 'l_' if selected_side == 'left' else 'r_' if selected_side == 'right' else None
    if prefix is None or not re.fullmatch(prefix + 'inspire_' + selected_side + r'_(thumb|index|middle)_[1-4]', link):
        return reject('not_selected_exact_finger_link')
    if previous is None or current.get('step') != previous.get('step', -2) + 1:
        return reject('no_immediately_previous_physical_step')
    for row in (previous, current):
        phase = row.get('phase', {})
        if not isinstance(phase, dict) or phase.get('name') != 'close' or phase.get('gripper') is not True:
            return reject('not_native_close_gripper_phase')
        if not isinstance(phase.get('dense_call'), int) or isinstance(phase.get('dense_call'), bool):
            return reject('missing_dense_call_identity')
    if previous['phase']['dense_call'] != current['phase']['dense_call']:
        return reject('different_close_dense_call')
    for key in ('model_id', 'qpose_tag'):
        if key not in previous.get('bottle', {}) or previous['bottle'][key] != current.get('bottle', {}).get(key):
            return reject('different_recorded_bottle_identity')
    def exact_records(row):
        contacts = row.get('bottle_actual_contacts')
        return [c for c in contacts if c.get('other_link') == link] if isinstance(contacts, list) else []
    old = exact_records(previous)
    now = exact_records(current)
    if not old or not now:
        return reject('missing_exact_native_contact_records')
    def points(records):
        result = []
        for record in records:
            values = record.get('points')
            if not isinstance(values, list) or not values:
                raise ValueError('missing_native_contact_points')
            declared = record.get('normal_impulse_Ns')
            if not _finite_scalar(declared) or declared < 0:
                raise ValueError('invalid_native_contact_record_impulse')
            summed = 0.
            for point in values:
                normal = point.get('normal_impulse_Ns')
                separation = point.get('separation_m')
                if not _finite_scalar(normal) or normal < 0 or not _finite_scalar(separation):
                    raise ValueError('invalid_native_point_impulse_or_separation')
                for key in ('impulse_Ns', 'normal_world', 'point_world_m'):
                    vector = np.asarray(point.get(key), dtype=float)
                    if vector.shape != (3,) or not np.isfinite(vector).all():
                        raise ValueError('invalid_native_point_vector')
                true_normal = abs(float(np.dot(point['impulse_Ns'], point['normal_world'])))
                if not math.isclose(true_normal, normal, rel_tol=1e-10, abs_tol=1e-18):
                    raise ValueError('native_point_normal_impulse_disagrees')
                summed += normal
                result.append(point)
            if not math.isclose(summed, declared, rel_tol=1e-12, abs_tol=1e-18):
                raise ValueError('native_record_normal_impulse_disagrees')
        return result
    try:
        old_points, now_points = points(old), points(now)
    except (ValueError, TypeError) as exc:
        return reject(str(exc))
    old_normal = sum(p['normal_impulse_Ns'] for p in old_points)
    if not _finite_scalar(threshold) or threshold < 0 or old_normal <= threshold:
        return reject('previous_exact_contact_not_true_positive')
    if any(p['normal_impulse_Ns'] != 0. for p in now_points):
        return reject('current_contact_has_nonzero_normal_impulse')
    if any(p['separation_m'] < 0. for p in now_points):
        return reject('current_native_contact_has_negative_separation')
    evidence.update(eligible=True, reason='adjacent_real_touch_to_zero_load_nonnegative_closure_boundary',
                    previous_step=previous['step'], current_step=current['step'],
                    dense_call=current['phase']['dense_call'], previous_normal_impulse_Ns=old_normal,
                    current_normal_impulse_Ns=0., current_native_points=len(now_points),
                    minimum_native_separation_m=min(p['separation_m'] for p in now_points),
                    maximum_native_separation_m=max(p['separation_m'] for p in now_points))
    return evidence


def depth_within_budget(certificate):
    """Unknown, deep, incomplete, or nonfinite geometry always remains failed."""
    for key in ('measured_penetration_depth_m', 'numerical_vertex_budget_m'):
        if not _finite_scalar(certificate.get(key)) or certificate[key] < 0:
            return False
    return bool(certificate.get('all_shape_pairs_evaluated') is True
                and certificate.get('contact_lists_untruncated') is True
                and certificate.get('original_boolean_collision_reproduced') is True
                and certificate.get('collision_count_positive') is True
                and certificate.get('event_specific_vertex_certificate', {}).get('native_signed_gap_within_conversion_resolution') is True
                and certificate['measured_penetration_depth_m'] <= certificate['numerical_vertex_budget_m'])


def _snapshot(task):
    from sapien.physx import PhysxRigidDynamicComponent
    art = task.robot.left_entity
    bottle_body = task.bottle.actor.find_component_by_type(PhysxRigidDynamicComponent)
    return {'qpos': np.asarray(art.qpos).copy(), 'qvel': np.asarray(art.qvel).copy(),
            'root': np.asarray(art.root_pose.to_transformation_matrix()).copy(),
            'bottle': np.asarray(task.bottle.actor.pose.to_transformation_matrix()).copy(),
            'bottle_linear': np.asarray(bottle_body.linear_velocity).copy(),
            'bottle_angular': np.asarray(bottle_body.angular_velocity).copy(),
            'actor_poses': [(a.name, np.asarray(a.pose.to_transformation_matrix()).copy()) for a in task.scene.get_all_actors()]}


def _assert_snapshot(task, snapshot):
    actual = _snapshot(task)
    for key in snapshot:
        if key == 'actor_poses':
            if len(actual[key]) != len(snapshot[key]) or any(x[0] != y[0] or not np.array_equal(x[1], y[1]) for x, y in zip(actual[key], snapshot[key])):
                raise RuntimeError('Physical actor state changed during independent review')
        elif not np.array_equal(actual[key], snapshot[key]):
            raise RuntimeError('Physical state changed during independent review: ' + key)


def _single_shape(auditor, obj, index):
    shape = auditor.fcl.CollisionObject(obj.shapes[index].get_collision_geometry())
    return auditor.fcl.FCLObject(obj.name + '_shape' + str(index), obj.pose,
                                [shape], [obj.shape_poses[index]])


def _geometry_fingerprint(component):
    """Content identity: native convex vertices/topology/scale/local transforms."""
    digest = hashlib.sha256()
    for shape in component.collision_shapes:
        digest.update(type(shape).__name__.encode())
        for key, dtype in (('vertices', '<f8'), ('triangles', '<i8'), ('scale', '<f8')):
            if not hasattr(shape, key): raise ValueError('Unsupported boundary geometry fingerprint')
            value = np.asarray(getattr(shape, key), dtype=dtype)
            digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        digest.update(np.asarray(shape.local_pose.to_transformation_matrix(), dtype='<f8').tobytes())
    return digest.hexdigest()


def _assert_cpu_constructor():
    import benchmark_geometry_cache as cache
    if sha(cache.__file__) != FROZEN_CPU_CONSTRUCTOR_SHA256:
        raise ValueError('Frozen zero-step CPU constructor source changed')
    tree = ast.parse(inspect.getsource(cache.cpu_snapshot_scene))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    if any(isinstance(n.func, ast.Attribute) and n.func.attr == 'step' for n in calls):
        raise ValueError('CPU constructor must never advance physics')
    if sum(isinstance(n.func, ast.Attribute) and n.func.attr == 'set_qpos' for n in calls) != 1:
        raise ValueError('CPU constructor must initialize qpos exactly once')
    return cache, tree


def _assert_cpu_receipt(receipt):
    expected = {'cpu_only_scene': True, 'render_system_count': 0, 'scene_steps': 0,
                'snapshot_initial_qpos_writes_before_audits': 1, 'audit_sapien_qpos_writes': 0,
                'snapshot_initial_qpos_exact': True, 'hardware_io': False, 'new_physics_episodes': 0}
    if any(receipt.get(k) != value for k, value in expected.items()):
        raise ValueError('Independent event scene was not exact, frozen, zero-step CPU initialization')


def _trial_identity(trial_path, trace, report):
    trial_path = Path(trial_path)
    scene_path = trial_path / 'scene_sample.json'
    config_path = trial_path / 'run_config.yml'
    scene = json.loads(scene_path.read_text())
    model = scene['sample_spec']['model_id']
    tag = trace[0]['bottle']['qpose_tag']
    if isinstance(tag, bool) or tag not in (0, 1): raise ValueError('Invalid recorded side tag')
    side = 'left' if tag == 0 else 'right'
    if report['active_arm'] != side or any(r['bottle']['qpose_tag'] != tag for r in trace):
        raise ValueError('Selected side differs from constant actual-trace qpose_tag')
    if report['sample_spec']['model_id'] != model or any(r['bottle']['model_id'] != model for r in trace):
        raise ValueError('Recorded model differs from frozen trial asset model')
    if scene['sample_spec'].get('arm') not in (None, side):
        raise ValueError('Scene selected side differs from actual trace')
    return {'run_config_sha256': sha(config_path), 'scene_sample_sha256': sha(scene_path),
            'selected_side': side, 'model_id': model, 'qpose_tag': tag}


def _event_snapshot_scene(trial_path, row):
    """Second independent frozen scene, initialized ONCE at the event state.

    Reuse the audited CPU constructor with exactly two initialization values
    replaced in a private compiled copy. Its source and active function remain
    untouched. No event scene exists before its initial qpos/pose assignments,
    and no state assignment or physics advance occurs during the audit.
    """
    cache, tree = _assert_cpu_constructor()
    q_sites = []
    pose_sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and len(node.body) == 1 and isinstance(node.body[0], ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == 'initial' for t in node.body[0].targets):
                q_sites.append(node)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'b.set_initial_pose':
            pose_sites.append(node)
    if len(q_sites) != 1 or len(pose_sites) != 1:
        raise ValueError('Frozen CPU constructor initialization sites changed')
    target = q_sites[0]
    replacement = ast.parse('initial = np.asarray(_event_qpos, dtype=float).copy()').body[0]
    class ReplaceInitial(ast.NodeTransformer):
        def visit_With(self, node):
            if node is target: return replacement
            return self.generic_visit(node)
    tree = ReplaceInitial().visit(tree)
    pose_sites[0].args = [ast.parse('sapien.Pose(np.asarray(_event_bottle_pose, dtype=np.float32))', mode='eval').body]
    ast.fix_missing_locations(tree)
    env = dict(cache.cpu_snapshot_scene.__globals__)
    env.update(_event_qpos=row['full_qpos_actual_rad'], _event_bottle_pose=row['bottle']['pose'])
    exec(compile(tree, str(Path(cache.__file__).resolve()) + ':v14_event_initialization_only', 'exec'), env)
    task, receipt = env['cpu_snapshot_scene'](Path(trial_path))
    _assert_cpu_receipt(receipt)
    if not np.array_equal(np.asarray(task.robot.left_entity.qpos, float), np.asarray(row['full_qpos_actual_rad'], float)):
        raise ValueError('Event native36q differs from recorded state after initialization')
    receipt.update(initialization_state='exact_recorded_event_qpos_and_bottle_pose', event_step=row['step'],
                   constructor_source_sha256=sha(cache.__file__), changed_initialization_values=2,
                   constructor_ast_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest())
    return task, receipt


def _event_vertex_certificate(trial_path, row, link, original_metadata, depth_auditor):
    """Event-specific native/FMA vertex agreement; no initial-error extrapolation."""
    from full_mesh_path_audit import FullMeshPathAuditor, _pose_matrix, _link_vertices
    task, receipt = _event_snapshot_scene(trial_path, row)
    physical = _snapshot(task)
    event_auditor = None
    try:
        event_auditor = FullMeshPathAuditor(task, check_table=True)
        keys = ['full_dof', 'collision_links', 'collision_shapes', 'geometry_types', 'bottle_shapes',
                'native_exemption_pairs', 'effective_exemption_pairs', 'checked_self_pairs',
                'urdf_kinematic_export_sha256']
        if any(event_auditor.metadata[k] != original_metadata[k] for k in keys):
            raise ValueError('Event snapshot native geometry identity differs')
        native_matrix = event_auditor.initial_scene_poses[link]
        model_matrix = _pose_matrix(event_auditor.objects[link].pose)
        vertices = event_auditor.vertices[link]
        finger_fingerprint = _geometry_fingerprint(event_auditor.links[link])
        if finger_fingerprint != _geometry_fingerprint(depth_auditor.links[link]):
            raise ValueError('Independent event and depth finger collision contents differ')
        depth_matrix = _pose_matrix(depth_auditor.objects[link].pose)
        if not np.array_equal(vertices, depth_auditor.vertices[link]):
            raise ValueError('Independent event and depth finger collision vertices differ')
        depth_world = vertices @ depth_matrix[:3, :3].T + depth_matrix[:3, 3]
        native_world = vertices @ native_matrix[:3, :3].T + native_matrix[:3, 3]
        model_world = vertices @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        event_to_depth_auxiliary_error = float(np.linalg.norm(depth_world - model_world, axis=1).max())
        if event_to_depth_auxiliary_error > 1e-12:
            raise ValueError('Event auxiliary FK differs from actual depth-measurement FK')
        measured_vertex_error = float(np.linalg.norm(native_world - model_world, axis=1).max())
        # Independent native FK object: same exact shape geometries, native link
        # global matrix, no auxiliary FK pose. It is a separate FCL object.
        source = event_auditor.objects[link]
        native_finger = event_auditor.fcl.FCLObject('native_event_' + link,
            event_auditor.Pose(native_matrix),
            [event_auditor.fcl.CollisionObject(s.get_collision_geometry()) for s in source.shapes], source.shape_poses)
        native_conversion_matrix = _pose_matrix(native_finger.pose)
        converted_world = vertices @ native_conversion_matrix[:3, :3].T + native_conversion_matrix[:3, 3]
        native_finger_conversion = float(np.linalg.norm(native_world - converted_world, axis=1).max())
        from sapien.physx import PhysxRigidBaseComponent
        event_bottle_component = task.bottle.actor.find_component_by_type(PhysxRigidBaseComponent)
        bottle_fingerprint = _geometry_fingerprint(event_bottle_component)
        if bottle_fingerprint != _geometry_fingerprint(depth_auditor.task.bottle.actor.find_component_by_type(PhysxRigidBaseComponent)):
            raise ValueError('Independent event and depth bottle collision contents differ')
        bottle_vertices, _ = _link_vertices(event_bottle_component)
        native_bottle_matrix = np.asarray(task.bottle.actor.pose.to_transformation_matrix(), float)
        bottle = event_auditor.static['bottle']
        fcl_bottle_matrix = _pose_matrix(bottle.pose)
        recorded_bottle_matrix = np.asarray(row['bottle']['pose'], float)
        recorded_world = bottle_vertices @ recorded_bottle_matrix[:3, :3].T + recorded_bottle_matrix[:3, 3]
        native_bottle_world = bottle_vertices @ native_bottle_matrix[:3, :3].T + native_bottle_matrix[:3, 3]
        fcl_bottle_world = bottle_vertices @ fcl_bottle_matrix[:3, :3].T + fcl_bottle_matrix[:3, 3]
        record_to_native_bottle = float(np.linalg.norm(recorded_world - native_bottle_world, axis=1).max())
        native_to_fcl_bottle = float(np.linalg.norm(native_bottle_world - fcl_bottle_world, axis=1).max())
        pure_conversion_budget = native_finger_conversion + record_to_native_bottle + native_to_fcl_bottle
        probes = []
        for tolerance in (1e-6, 1e-9, 1e-12):
            hit = bool(event_auditor.fcl.collide(native_finger, bottle,
                event_auditor.fcl.CollisionRequest(num_max_contacts=512, enable_contact=True, gjk_tolerance=tolerance)).is_collision())
            distance = event_auditor.fcl.distance(native_finger, bottle,
                event_auditor.fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True, distance_tolerance=tolerance))
            value = float(distance.min_distance)
            if not math.isfinite(value): raise ValueError('Nonfinite independent native FK signed gap')
            probes.append({'gjk_tolerance': tolerance, 'native_fcl_collision': hit, 'signed_distance_m': value})
        native_shape_probes = []
        for finger_index in range(len(native_finger.shapes)):
            for bottle_index in range(len(bottle.shapes)):
                first = _single_shape(event_auditor, native_finger, finger_index)
                second = _single_shape(event_auditor, bottle, bottle_index)
                for tolerance in (1e-6, 1e-9, 1e-12):
                    collision = event_auditor.fcl.collide(first, second,
                        event_auditor.fcl.CollisionRequest(num_max_contacts=512, enable_contact=True, gjk_tolerance=tolerance))
                    contacts = collision.get_contacts()
                    if len(contacts) != collision.num_contacts() or len(contacts) >= 512 or (collision.is_collision() and not contacts):
                        raise ValueError('Native-FK penetration evidence missing or truncated')
                    depths = [float(c.penetration_depth) for c in contacts]
                    if any(not math.isfinite(d) or d < 0 for d in depths):
                        raise ValueError('Invalid native-FK penetration depth')
                    value = float(event_auditor.fcl.distance(first, second,
                        event_auditor.fcl.DistanceRequest(enable_signed_distance=True, distance_tolerance=tolerance)).min_distance)
                    if not math.isfinite(value): raise ValueError('Invalid native-FK per-shape signed distance')
                    native_shape_probes.append({'finger_shape_index': finger_index, 'bottle_shape_index': bottle_index,
                        'gjk_tolerance': tolerance, 'collision': bool(collision.is_collision()),
                        'signed_distance_m': value, 'penetration_depths_m': depths})
        native_in_resolution = all(p['signed_distance_m'] >= -pure_conversion_budget
            and all(d <= pure_conversion_budget for d in p['penetration_depths_m']) for p in native_shape_probes)
        return {'event_step': row['step'], 'link': link, 'scene_receipt': receipt,
                'event_fk_audit': event_auditor.fk_audit,
                'native_to_auxiliary_finger_vertex_error_m': measured_vertex_error,
                'event_auxiliary_to_depth_auxiliary_vertex_error_m': event_to_depth_auxiliary_error,
                'finger_collision_content_sha256': finger_fingerprint,
                'bottle_collision_content_sha256': bottle_fingerprint,
                'native_finger_matrix_to_fcl_vertex_error_m': native_finger_conversion,
                'recorded_to_native_bottle_vertex_error_m': record_to_native_bottle,
                'native_bottle_matrix_to_fcl_vertex_error_m': native_to_fcl_bottle,
                'native_fk_signed_gap_probes': probes,
                'native_signed_gap_nonnegative': all(p['signed_distance_m'] >= 0. and not p['native_fcl_collision'] for p in probes),
                'native_signed_gap_within_conversion_resolution': native_in_resolution and all(p['signed_distance_m'] >= -pure_conversion_budget for p in probes),
                'native_signed_gap_conversion_budget_m': pure_conversion_budget,
                'native_all_shape_pair_probes': native_shape_probes,
                'event_physical_scene_unchanged': True,
                'numerical_vertex_budget_m': measured_vertex_error + native_finger_conversion + record_to_native_bottle + native_to_fcl_bottle}
    finally:
        if event_auditor is not None: event_auditor.model.set_qpos(event_auditor.old_model_q, full=True)
        _assert_snapshot(task, physical)


def _depth_certificate(auditor, row, link, original_metadata, trial_path):
    """Independently enumerate every shape pair; never use _distance's zero gap."""
    from full_mesh_path_audit import _pose_matrix, _link_vertices
    matrix = np.asarray(row['bottle']['pose'], dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid recorded bottle transform')
    source = auditor.static['bottle']
    bottle = auditor.fcl.FCLObject(source.name, auditor.Pose(matrix),
                [auditor.fcl.CollisionObject(s.get_collision_geometry()) for s in source.shapes], source.shape_poses)
    obj = auditor.objects[link]
    original_hit = bool(auditor.fcl.collide(obj, bottle, auditor.collision_request).is_collision())
    fk = original_metadata['fk_audit']
    if fk.get('passed') is not True or fk['checked_links'] != len(auditor.links):
        raise ValueError('Missing verified original FK evidence')
    translation = float(fk['worst_position']['position_error_m'])
    rotation = float(fk['worst_rotation']['rotation_error_rad'])
    if not np.isfinite([translation, rotation]).all() or not (0 <= translation <= 2e-5 and 0 <= rotation <= 2e-4):
        raise ValueError('Original measured FK inconsistency outside unchanged limits')
    radius = float(np.linalg.norm(auditor.vertices[link], axis=1).max())
    from sapien.physx import PhysxRigidBaseComponent
    vertices, _ = _link_vertices(auditor.task.bottle.actor.find_component_by_type(PhysxRigidBaseComponent))
    converted = _pose_matrix(bottle.pose)
    source_world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    converted_world = vertices @ converted[:3, :3].T + converted[:3, 3]
    bottle_residual = float(np.linalg.norm(source_world - converted_world, axis=1).max())
    # Initial error is reported for context only; it is NEVER the event budget.
    rotation_vertex_budget = 2. * radius * math.sin(rotation / 2.)
    event_vertices = _event_vertex_certificate(trial_path, row, link, original_metadata, auditor)
    budget = event_vertices['numerical_vertex_budget_m']
    pairs = []
    max_depth = 0.
    collisions = 0
    max_contacts = 512
    for i in range(len(obj.shapes)):
        first = _single_shape(auditor, obj, i)
        for j in range(len(bottle.shapes)):
            second = _single_shape(auditor, bottle, j)
            probes = []
            for tolerance in (1e-6, 1e-9, 1e-12):
                request = auditor.fcl.CollisionRequest(num_max_contacts=max_contacts, enable_contact=True, gjk_tolerance=tolerance)
                result = auditor.fcl.collide(first, second, request)
                contacts = result.get_contacts()
                if result.num_contacts() != len(contacts) or len(contacts) >= max_contacts:
                    raise ValueError('Penetration contact list unknown or truncated')
                hit = bool(result.is_collision())
                if hit and not contacts:
                    raise ValueError('Collision without independent depth evidence')
                depths = []
                for c in contacts:
                    depth = float(c.penetration_depth)
                    if not np.isfinite(depth) or depth < 0. or not np.isfinite(np.asarray(c.pos, float)).all() or not np.isfinite(np.asarray(c.normal, float)).all():
                        raise ValueError('Invalid independent FCL penetration evidence')
                    depths.append(depth)
                probes.append({'gjk_tolerance': tolerance, 'collision': hit, 'contact_count': len(contacts), 'penetration_depths_m': depths})
                if depths:
                    max_depth = max(max_depth, max(depths))
            if any(p['collision'] != probes[0]['collision'] for p in probes):
                raise ValueError('Independent FCL collision unstable across precision settings')
            if probes[0]['collision']:
                collisions += 1
            pairs.append({'finger_shape_index': i, 'bottle_shape_index': j, 'probes': probes})
    return {'original_boolean_collision_reproduced': original_hit,
            'all_shape_pairs_evaluated': len(pairs) == len(obj.shapes) * len(bottle.shapes),
            'contact_lists_untruncated': True, 'collision_count_positive': collisions > 0,
            'shape_pairs_evaluated': len(pairs), 'shape_pairs_colliding': collisions,
            'measured_penetration_depth_m': max_depth, 'numerical_vertex_budget_m': budget,
            'measured_initial_fk_translation_budget_m': translation,
            'measured_initial_fk_rotation_rad': rotation,
            'finger_maximum_local_vertex_radius_m': radius,
            'rotation_vertex_budget_m': rotation_vertex_budget,
            'recorded_bottle_matrix_to_fcl_vertex_residual_m': bottle_residual,
            'event_specific_vertex_certificate': event_vertices,
            'budget_scope': 'Exact recorded-event independent native CPU FK and auxiliary-FK collision vertex agreement plus measured transform conversion errors; no extrapolation of original initial FK residual.',
            'shape_pair_depths': pairs}


def _events(samples, impulse_threshold):
    merged = {}
    def add(category, pair, row):
        key = (category, *pair)
        ev = merged.setdefault(key, {'category': category, 'pair': list(pair), 'count': 0,
            'first_step': row['step'], 'last_step': row['step'], 'first_trace_index': row['trace_index']})
        ev['count'] += 1
        ev['last_step'] = row['step']
    for row in samples:
        for pair in row['self_collisions']: add('self', pair, row)
        for link in row['robot_table_collisions']: add('robot_table', (link, 'table'), row)
        for link in row['bottle_non_touch_collisions']: add('bottle_robot_non_touch', ('bottle', link), row)
        for contact in row['actual_self_contacts']:
            if contact['normal_impulse_Ns'] > impulse_threshold:
                add('historical_true_self_normal', contact.get('links', contact.get('names', [])), row)
    return merged


def _base_checks(original):
    return bool(original['bottle_table_first_release'] is not None
        and original['samples'][-1]['bottle_table_geometry']['distance_m'] > 0
        and original['max_measured_joint_limit_excess_rad'] <= 1e-4
        and original['max_post_grasp_relative_translation_drift_m'] <= .002
        and original['max_post_grasp_relative_rotation_drift_deg'] <= 2.
        and original['complete_to_reported_endpoint'] is True
        and original['full_temporal_coverage'] is True
        and original['physical_scene_unchanged'] is True)


def reconcile_original_result(task, trace, report, original, *, trace_sha256, original_audit_path=None, trial_path=None):
    """Return an additive v14 result; the supplied v13 object is not modified."""
    from full_mesh_path_audit import FullMeshPathAuditor
    if original['trace_sha256'] != trace_sha256 or original['source_hashes'] != report['source_hashes']:
        raise ValueError('Trace/source identity differs from retained v13 audit')
    sources = {p: sha(p) for p in original['source_hashes']}
    if sources != original['source_hashes'] or report.get('canonical_sources_unchanged') is not True:
        raise ValueError('Canonical source identity failed')
    if getattr(task.robot, 'communication_flag', False) or report.get('hardware_io') is not False:
        raise ValueError('Simulation-only independent review')
    samples = original['samples']
    if len(trace) != len(samples) or len(trace) != original['recorded_samples'] or not trace:
        raise ValueError('Old audit/trace sample coverage differs')
    if [r['step'] for r in trace] != list(range(1, len(trace) + 1)):
        raise ValueError('Every actual physical step is required')
    if report.get('physical_steps') != trace[-1]['step'] or report['full_qpos_joint_names'] != original['full_qpos_joint_names']:
        raise ValueError('Endpoint/joint identity differs')
    if trial_path is None: raise ValueError('Missing frozen trial identity')
    trial_identity = _trial_identity(trial_path, trace, report)
    if hasattr(task, 'model_id') and task.model_id != trial_identity['model_id']:
        raise ValueError('Production scene model differs from actual recorded asset')
    if hasattr(task, 'qpose_tag') and task.qpose_tag != trial_identity['qpose_tag']:
        raise ValueError('Production scene side tag differs from actual recorded side')
    for i, (row, audited) in enumerate(zip(trace, samples)):
        if audited['trace_index'] != i or row['step'] != audited['step'] or row['phase'] != audited['phase'] or row['actual_self_contacts'] != audited['actual_self_contacts']:
            raise ValueError('Historical frame identity changed')
    recomposed = _events(samples, original['actual_self_normal_impulse_threshold_Ns'])
    # Table events are retained explicitly; all other old events must reconstruct.
    table_events = [e for e in original['collision_events'] if e['category'] == 'bottle_table_after_release']
    non_table_events = [e for e in original['collision_events'] if e['category'] != 'bottle_table_after_release']
    if sorted(recomposed.values(), key=str) != sorted(non_table_events, key=str):
        raise ValueError('Original sample collision counts do not reconstruct')
    baseline = _base_checks(original)
    if original['passed'] != bool(baseline and not original['collision_events']):
        raise ValueError('Original v13 acceptance cannot be reconstructed unchanged')
    physical = _snapshot(task)
    auditor = None
    result = dict(original)
    result['samples'] = list(samples)
    resolved, reviewed = [], []
    try:
        auditor = FullMeshPathAuditor(task, check_table=True)
        if auditor.sim_joints != original['full_qpos_joint_names']:
            raise ValueError('Current collision model joint order differs')
        keys = ['full_dof', 'collision_links', 'collision_shapes', 'geometry_types', 'bottle_shapes',
                'native_exemption_pairs', 'engine_adjacent_pairs', 'engine_adjacent_additional_pairs',
                'effective_exemption_pairs', 'checked_self_pairs', 'urdf_kinematic_export_sha256']
        identity = {k: auditor.metadata[k] == original['native_mesh_metadata'][k] for k in keys}
        if not all(identity.values()):
            raise ValueError('Current native geometry differs from retained v13 audit')
        q = np.asarray([r['full_qpos_actual_rad'] for r in trace], float)
        if q.shape != (len(trace), 36) or not np.isfinite(q).all():
            raise ValueError('Expected all finite actual36q')
        side = trial_identity['selected_side']
        for arm in ('left', 'right'):
            idx = [auditor.sim_joints.index(n) for n in getattr(task.robot, arm + '_arm_joints_name')]
            if not np.allclose(q[:, idx], [r['arm_qpos'][arm] for r in trace], atol=1e-6, rtol=0):
                raise ValueError('Actual36q disagrees with arm actual states')
        threshold = original['grasp_touch_normal_impulse_threshold_Ns']
        unique_occurrence = sum(len(s['bottle_non_touch_collisions']) for s in samples) == 1
        for i, row in enumerate(trace):
            for link in samples[i]['bottle_non_touch_collisions']:
                native = native_boundary_evidence(trace[i-1] if i else None, row, link, side, threshold)
                event = {'step': row['step'], 'trace_index': i, 'link': link, 'native_evidence': native, 'resolved': False}
                if native['eligible'] and not unique_occurrence:
                    event['resolution_refused_reason'] = 'more_than_one_original_non_touch_pair_sample_in_trial'
                if native['eligible'] and unique_occurrence:
                    auditor.model.set_qpos(q[i], full=True)
                    try:
                        if trial_path is None: raise ValueError('Missing frozen event snapshot source trial')
                        depth = _depth_certificate(auditor, row, link, original['native_mesh_metadata'], trial_path)
                    except (ValueError, RuntimeError) as exc:
                        depth = {'certificate_error': str(exc)}
                    event['depth_certificate'] = depth
                    if depth_within_budget(depth):
                        newrow = dict(result['samples'][i])
                        newrow['bottle_non_touch_collisions'] = [n for n in newrow['bottle_non_touch_collisions'] if n != link]
                        newrow['reconciled_zero_load_closure_boundary_links'] = list(newrow.get('reconciled_zero_load_closure_boundary_links', [])) + [link]
                        result['samples'][i] = newrow
                        event['resolved'] = True
                        resolved.append(event)
                reviewed.append(event)
        remaining = list(_events(result['samples'], original['actual_self_normal_impulse_threshold_Ns']).values()) + table_events
        result.update(passed=bool(baseline and not remaining), collision_events=remaining,
                      audit_version=VERSION, original_v13_passed=original['passed'],
                      boundary_resolution_policy='Only one unique original non-touch pair/sample per trial; selected primary thumb/index/middle exact link only.',
                      original_v13_collision_events=copy.deepcopy(original['collision_events']),
                      resolved_closure_boundary_events=resolved, reviewed_original_non_touch_events=reviewed,
                      original_noncollision_checks_unchanged=True, baseline_all_other_checks_passed=baseline,
                      current_geometry_identity_checks=identity, cpu_review_fk_audit=auditor.fk_audit,
                      frozen_trial_identity=trial_identity,
                      boundary_helper_sha256=sha(__file__), boundary_helper_path=str(Path(__file__).resolve()),
                      acceptance_scope='Independent discrete geometry review only; original rejection retained. No dataset export or physical success is created by this review.',
                      physical_scene_unchanged=True)
        if original_audit_path:
            result['retained_original_v13_audit'] = {'path': str(original_audit_path), 'sha256': sha(original_audit_path)}
    finally:
        if auditor is not None:
            auditor.model.set_qpos(auditor.old_model_q, full=True)
        _assert_snapshot(task, physical)
        if sources != {p: sha(p) for p in sources}:
            raise RuntimeError('Canonical sources changed during independent boundary review')
        if trial_identity != _trial_identity(trial_path, trace, report):
            raise RuntimeError('Frozen trial configuration changed during review')
    return result


def audit_accepted_actual_trace_v14(task, trace_path, *, report_path, original_output_path, output_path=None, **kwargs):
    """Future opt-in wrapper. Retain unmodified original full audit first."""
    from observed_full_trace_audit_v13 import audit_accepted_actual_trace
    original_output_path = Path(original_output_path)
    if original_output_path.exists():
        raise ValueError('Refusing to overwrite retained original v13 audit')
    original = audit_accepted_actual_trace(task, trace_path, report_path=report_path,
                                          output_path=original_output_path, **kwargs)
    trace = json.loads(Path(trace_path).read_text())
    report = json.loads(Path(report_path).read_text())
    result = reconcile_original_result(task, trace, report, original, trace_sha256=sha(trace_path),
                                       original_audit_path=original_output_path, trial_path=Path(report_path).parent)
    if output_path:
        path = Path(output_path)
        if path.exists(): raise ValueError('Refusing to overwrite previous v14 review')
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise ValueError('Refusing to overwrite an existing certificate')
    sys.path[:0] = [str(ROOT), str(LEGACY), str(WORK)]
    os.chdir(ROOT)
    from geometry_overlay import install_native_shape_count_overlay
    install_native_shape_count_overlay(LEGACY)
    from benchmark_geometry_cache import cpu_snapshot_scene
    original_path = args.trial / 'actual36_geometry_audit.json'
    trace_path = args.trial / 'physics_trace.json'
    report_path = args.trial / 'expert_report.json'
    original = json.loads(original_path.read_text())
    trace = json.loads(trace_path.read_text())
    report = json.loads(report_path.read_text())
    # A rejected historical report is reviewable only for this already retained
    # v13 failure. It remains failed; this CLI never rewrites it or exports data.
    if report.get('accepted') is not True and (original.get('passed') is not False or
            report.get('error') != 'Independent actual36 geometry audit failed'):
        raise ValueError('Not an originally accepted or solely audit-rejected physical record')
    print('ORIGINAL_LOADED', len(trace), flush=True)
    _assert_cpu_constructor()
    task, receipt = cpu_snapshot_scene(args.trial)
    _assert_cpu_receipt(receipt)
    result = reconcile_original_result(task, trace, report, original, trace_sha256=sha(trace_path),
                                       original_audit_path=original_path, trial_path=args.trial)
    summary = {k: v for k, v in result.items() if k != 'samples'}
    summary.update(physical_snapshot_receipt=receipt, original_report_recorded_sha256=original['report_sha256'],
                   current_rejection_report_sha256=sha(report_path), current_rejection_report_unmodified=True,
                   samples_retained_by_reference=True, sample_overlay=[result['samples'][e['trace_index']] for e in result['resolved_closure_boundary_events']],
                   unchanged_sample_count=len(trace) - len({e['trace_index'] for e in result['resolved_closure_boundary_events']}),
                   original_audit_unmodified=sha(original_path) == result['retained_original_v13_audit']['sha256'])
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': result['passed'], 'original_passed': original['passed'],
                      'resolved_events': len(result['resolved_closure_boundary_events']),
                      'remaining_events': len(result['collision_events']), 'out': str(args.out), 'sha256': sha(args.out)}), flush=True)


if __name__ == '__main__':
    main()
