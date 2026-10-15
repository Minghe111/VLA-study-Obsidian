"""Restrict one numeric branch on a NEW auxiliary MPLib model only.

Never changes the original model, vendor URDF, physical articulation or returned
path. The caller keeps its original FMA model and all physical safety checks.
"""
from __future__ import annotations
import hashlib, math, xml.etree.ElementTree as ET
import numpy as np
from mplib import ArticulatedModel, Pose
from mplib.collision_detection import fcl

def _matrix(pose):
    value = pose.to_transformation_matrix() if hasattr(pose, 'to_transformation_matrix') else pose
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError('Finite 4x4 rigid pose matrix required')
    return result

def narrowed_numeric_window(current, vendor_lower, vendor_upper, *, margin_rad=1e-5):
    values = np.asarray([current, vendor_lower, vendor_upper, margin_rad], dtype=float)
    if not np.isfinite(values).all() or not 0 < margin_rad < math.pi:
        raise ValueError('Finite current/vendor bounds and positive sub-pi margin required')
    if not vendor_lower <= current <= vendor_upper:
        raise ValueError('Actual current q is outside exact vendor bounds; never clip it')
    half = math.pi - margin_rad
    lower, upper = max(vendor_lower, current-half), min(vendor_upper, current+half)
    if not lower <= current <= upper or not 0 < upper-lower < 2*math.pi:
        raise ValueError('A nonempty bounded window narrower than 2pi is required')
    return lower, upper

def build_auxiliary_window_model(source_model, kinematic_urdf, srdf, *, current_full_q,
                                 joint_name, move_group, margin_rad=1e-5):
    """Clone all actual collision geometries, narrowing only one auxiliary limit.

    For a live native auditor pass its bare ``audit.urdf`` and the SRDF encoding
    exactly ``audit.allowed``. FCL geometry and every shape-local pose are cloned
    from ``audit.model``; shared source CollisionObject poses are never mutated.
    This creates only a CPU kinematic/collision model, not a SAPIEN scene.
    """
    q = np.asarray(current_full_q, dtype=float)
    names = list(source_model.get_user_joint_names())
    links = list(source_model.get_user_link_names())
    if q.shape != (len(names),) or not np.isfinite(q).all() or names.count(joint_name) != 1:
        raise ValueError('Exact full ordered current state and one selected joint required')
    tree = ET.fromstring(kinematic_urdf)
    if any(link.findall('visual') or link.findall('collision') for link in tree.findall('link')):
        raise ValueError('Bare native kinematic export required; collision geometries supplied separately')
    matches = [joint for joint in tree.findall('joint') if joint.get('name') == joint_name]
    if len(matches) != 1 or matches[0].get('type') != 'revolute':
        raise ValueError('Exactly one explicitly bounded revolute joint required')
    joint = matches[0]; limit = joint.find('limit')
    if limit is None:
        raise ValueError('Original finite revolute limit required')
    lower, upper = float(limit.get('lower')), float(limit.get('upper'))
    index = names.index(joint_name)
    native_limits = np.concatenate(source_model.get_pinocchio_model().get_joint_limits())
    if np.max(abs(native_limits[index] - [lower, upper])) > 1e-5:
        raise ValueError('Source model bounds differ from declared original kinematic URDF')
    window = narrowed_numeric_window(float(q[index]), lower, upper, margin_rad=margin_rad)
    before_tree = ET.tostring(tree, encoding='unicode')
    limit.set('lower', format(window[0], '.17g')); limit.set('upper', format(window[1], '.17g'))
    derived = ET.tostring(tree, encoding='unicode')
    original_q = np.asarray(source_model.get_qpos()).copy()
    original_base = _matrix(source_model.get_base_pose()).copy()
    source_objects = list(source_model.get_fcl_model().get_collision_objects())
    source_object_poses = {obj.name: _matrix(obj.pose).copy() for obj in source_objects}
    colliders = []
    inventory = {}
    for obj in source_objects:
        shapes = [fcl.CollisionObject(shape.get_collision_geometry()) for shape in obj.shapes]
        local = [Pose(_matrix(p)) for p in obj.shape_poses]
        colliders.append(fcl.FCLObject(obj.name, Pose(), shapes, local))
        inventory[obj.name] = len(shapes)
    model = ArticulatedModel.create_from_urdf_string(derived, srdf, colliders,
                                                   link_names=links, joint_names=names)
    model.set_base_pose(Pose(original_base)); model.set_qpos(q, full=True)
    model.set_move_group(move_group)
    if list(model.get_user_joint_names()) != names or list(model.get_user_link_names()) != links:
        raise RuntimeError('Auxiliary model changed original ordering')
    observed = {obj.name: len(obj.shapes) for obj in model.get_fcl_model().get_collision_objects()}
    if observed != inventory:
        raise RuntimeError('Auxiliary model lost any original collision link/shape')
    new_limits = np.concatenate(model.get_pinocchio_model().get_joint_limits())
    unchanged = np.ones(len(names), dtype=bool); unchanged[index] = False
    if not np.array_equal(new_limits[unchanged], native_limits[unchanged]):
        raise RuntimeError('An unselected auxiliary joint limit changed')
    if np.max(abs(new_limits[index] - window)) > 1e-10:
        raise RuntimeError('Auxiliary selected window readback differs')
    if not np.array_equal(source_model.get_qpos(), original_q) or not np.array_equal(_matrix(source_model.get_base_pose()), original_base):
        raise RuntimeError('Cloning changed original model state')
    if any(not np.array_equal(_matrix(obj.pose), source_object_poses[obj.name]) for obj in source_objects):
        raise RuntimeError('Cloning aliased and changed original collision object poses')
    receipt = {'selected_joint': joint_name, 'selected_full_q_index': index,
        'current_q_rad': float(q[index]), 'original_vendor_limits_rad': [lower, upper],
        'actual_full_vendor_limits_rad': native_limits.tolist(),
        'actual_full_joint_names': names,
        'auxiliary_window_rad': list(window), 'auxiliary_width_rad': window[1]-window[0],
        'source_kinematic_urdf_sha256': hashlib.sha256(kinematic_urdf.encode()).hexdigest(),
        'canonical_original_urdf_sha256': hashlib.sha256(before_tree.encode()).hexdigest(),
        'auxiliary_derived_urdf_sha256': hashlib.sha256(derived.encode()).hexdigest(),
        'collision_shape_inventory': inventory, 'only_selected_auxiliary_limit_changed': True,
        'source_model_limits_state_and_collision_poses_unchanged': True,
        'source_collision_geometry_and_local_shape_poses_cloned': True,
        'path_postprocessing_wrap_or_clip_allowed': False,
        'original_FMA_and_physical_limits_must_still_be_audited': True}
    return model, receipt

def create_model(auditor, side):
    """Return (independent planning model, receipt) from a native FMA auditor."""
    if side not in ('left', 'right'):
        raise ValueError('Explicit left/right arm required')
    names = list(auditor.sim_joints)
    arm_names = list(getattr(auditor.task.robot, side + '_arm_joints_name'))
    if len(arm_names) != 6:
        raise ValueError('Exact native arm6 joint names required')
    q = np.asarray(auditor.initial_sim_q, dtype=float).copy()
    if not np.array_equal(auditor.model.get_qpos(), q):
        raise ValueError('Fresh original auditor model must still equal actual initial full q')
    srdf = ET.Element('robot', name=auditor.urdf_tree.get('name'))
    for pair in sorted(sorted(pair) for pair in auditor.allowed):
        ET.SubElement(srdf, 'disable_collisions', link1=pair[0], link2=pair[1],
                      reason='Unchanged actual native physics exemptions')
    palm = ('l_' if side == 'left' else 'r_') + 'inspire_base_link'
    model, receipt = build_auxiliary_window_model(auditor.model, auditor.urdf,
        ET.tostring(srdf, encoding='unicode'), current_full_q=q,
        joint_name=arm_names[-1], move_group=palm)
    if list(model.get_move_group_joint_indices()) != [names.index(n) for n in arm_names]:
        raise RuntimeError('Planning clone changed native selected-arm move group')
    original_pk, new_pk = auditor.pk, model.get_pinocchio_model()
    original_pk.compute_forward_kinematics(q); new_pk.compute_forward_kinematics(q)
    original_links, new_links = list(original_pk.get_link_names()), list(new_pk.get_link_names())
    if original_links != new_links or len(original_links) != len(auditor.links):
        raise RuntimeError('Planning clone does not preserve every actual native FK link')
    error = max(float(abs(_matrix(original_pk.get_link_pose(i)) -
                          _matrix(new_pk.get_link_pose(i))).max())
                for i in range(len(original_links)))
    if error > 1e-8:
        raise RuntimeError('Only-limit planning clone changed native FK')
    original_objects = list(auditor.model.get_fcl_model().get_collision_objects())
    new_objects = {obj.name: obj for obj in model.get_fcl_model().get_collision_objects()}
    pose_error = max(float(abs(_matrix(obj.pose)-_matrix(new_objects[obj.name].pose)).max())
                     for obj in original_objects)
    if pose_error > 1e-8:
        raise RuntimeError('Planning clone changed an actual initial collision-link pose')
    receipt.update(native_FK_checked_link_count=len(original_links),
        native_full_FK_initial_matrix_max_error=error,
        native_collision_link_initial_matrix_max_error=pose_error,
        original_physics_exemption_pair_count=len(auditor.allowed),
        actual_vendor_FMA_model_not_replaced=True,
        auxiliary_planning_pk_is_separate_from_auditor_pk=True)
    return model, receipt

def contains_selected_arm(model, q, indices):
    """Reject out-of-window numeric goals before OMPL can synthesize aliases."""
    full = np.asarray(q, dtype=float)
    limits = np.concatenate(model.get_pinocchio_model().get_joint_limits())
    if full.shape != (len(limits),) or not np.isfinite(full).all():
        return False
    return bool(np.all(full[indices] >= limits[indices,0]) and
                np.all(full[indices] <= limits[indices,1]))

def contains_arm_path(model, path, indices):
    """Read back every raw/TOPP numeric row against the auxiliary window."""
    rows = np.asarray(path, dtype=float)
    limits = np.concatenate(model.get_pinocchio_model().get_joint_limits())[indices]
    if rows.ndim != 2 or rows.shape[1] != len(indices) or len(rows)<2 or not np.isfinite(rows).all():
        return False
    return bool(np.all(rows >= limits[:,0]) and np.all(rows <= limits[:,1]))
