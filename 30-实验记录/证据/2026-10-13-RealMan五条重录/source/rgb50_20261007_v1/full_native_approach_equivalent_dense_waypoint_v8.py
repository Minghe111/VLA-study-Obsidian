"""CPU-only free-hand candidate via original goal palm +world-Z 80 mm.

The complete frozen v7 direct attempt precedes this fallback and is retained.
Both waypoint legs use finite vendor-limit goal declarations, all-link native
FK, the complete collision world, and unchanged numeric endpoint tolerances.
All returned raw rows are concatenated unchanged, then official TOPP and the
original full-mesh gate/strict v7 certificate refinement certify the whole path.
This module performs no physical-state writes, steps, or grasp-success tests.
"""
import time
import numpy as np
import full_native_approach_equivalent_goals_dense_v7 as original
import equivalent_arm_goals_v5 as equivalent
import dense_free_certificate_policy_v7 as dense_policy
from full_native_approach_waypoint_v4 import _physical_snapshot, _assert_unchanged

WAYPOINT_WORLD_Z_M = .08


def plan_free_approach(task, target_pose, side, *, time_step=.004, random_seed=0, ik_attempts=24):
    before = _physical_snapshot(task)
    direct = original.plan_free_approach(task, target_pose, side,
        time_step=time_step, random_seed=random_seed, ik_attempts=ik_attempts)
    _assert_unchanged(task, before)
    if direct.get('status') == 'Success':
        return direct
    if direct.get('status') != 'Fail':
        raise ValueError('Waypoint fallback requires an explicit rejected direct result')
    import full_mesh_path_audit as fma
    from full_native_carry_planner_v31 import official_topp_polyline
    from mplib import PlanningWorld
    from mplib.pymp.planning.ompl import OMPLPlanner
    audit = fma.FullMeshPathAuditor(task)
    model = audit.model
    old_pose, old_q = model.get_base_pose(), model.get_qpos().copy()
    old_group = list(model.get_move_group_end_effectors())
    old_group_indices = list(model.get_move_group_joint_indices())
    metadata = {
        'backend': 'full_native_free_hand_MPLib_explicit_bounded_equivalent_goals_dense_waypoint_v8',
        'original_direct_result_before_waypoint': direct,
        'physical_state_unchanged': False, 'no_attached_objects': True,
        'waypoint_semantics': 'Original hardware palm target +world Z .08m, identical rotation; original endpoint retained',
        'waypoint_world_z_m': WAYPOINT_WORLD_Z_M,
        'raw_path_never_wrapped_or_clipped': True,
        'CPU_path_is_not_physical_success': True,
        'historical_alias_or_waypoint_success_not_inferred': True,
        'endpoint_attempts': [], 'RRT_attempts': [],
    }
    try:
        base = audit.initial_sim_q.copy()
        names = audit.sim_joints
        indices = [names.index(n) for n in getattr(task.robot, side + '_arm_joints_name')]
        palm = ('l_' if side == 'left' else 'r_') + 'inspire_base_link'
        model.set_move_group(palm)
        if list(model.get_move_group_joint_indices()) != indices:
            raise ValueError('Native move-group indices differ from the actual selected arm')
        pk = audit.pk
        link_names = list(pk.get_link_names())
        palm_index = link_names.index(palm)
        root = task.robot.left_entity.root_pose.to_transformation_matrix()
        inverse = np.linalg.inv(root)
        goal_world = task.robot._trans_from_gripper_to_endlink(target_pose, arm_tag=side).to_transformation_matrix()
        via_world = goal_world.copy()
        via_world[2, 3] += WAYPOINT_WORLD_Z_M
        goal, via = inverse @ goal_world, inverse @ via_world
        limits = np.concatenate([np.asarray(j.limits).reshape(-1, 2) for j in audit.art.active_joints])
        exported = {j.get('name'): j.get('type') for j in audit.urdf_tree.findall('joint')}
        joint_types = [exported[n] for n in names]
        model.set_base_pose(audit.Pose())
        model.set_qpos(base, full=True)
        obstacles = []
        for name, source in audit.static.items():
            shapes = [audit.fcl.CollisionObject(s.get_collision_geometry()) for s in source.shapes]
            obstacles.append(audit.fcl.FCLObject('realman_' + name,
                audit.Pose(inverse @ source.pose.to_transformation_matrix()), shapes,
                [audit.Pose(p) for p in source.shape_poses]))
        world = PlanningWorld([model], obstacles)
        acm = world.get_allowed_collision_matrix()
        for pair in audit.allowed:
            a, b = sorted(pair)
            acm.set_entry(a, b, True)
        mask = np.ones(len(base), dtype=bool)
        mask[indices] = False
        metadata.update(initial_actual_full_qpos_rad=before['q'].tolist(),
                        initial_native_bottle_matrix=before['bottle'].tolist(),
                        initial_model_base_pose_matrix=fma._pose_matrix(old_pose).tolist(),
                        selected_arm_joint_names=[names[i] for i in indices],
                        selected_arm_indices=indices, native_FK_link_names=link_names,
                        goal_palm_world_matrix=goal_world.tolist(), via_palm_world_matrix=via_world.tolist(),
                        native_commanded_limits_rad=limits[indices].tolist(),
                        native_commanded_URDF_types=[joint_types[i] for i in indices],
                        raw_start_tolerance_rad=equivalent.RAW_START_TOLERANCE_RAD,
                        raw_end_tolerance_rad=equivalent.RAW_END_TOLERANCE_RAD)

        def reset(q):
            model.set_base_pose(audit.Pose())
            model.set_qpos(q, full=True)
            world.set_qpos_all(q[indices])

        def native_fk(q):
            pk.compute_forward_kinematics(q)
            return np.asarray([fma._pose_matrix(pk.get_link_pose(i)) for i in range(len(link_names))])

        def families(target, label, offset):
            rng = np.random.default_rng(random_seed + offset)
            result = []
            for attempt in range(ik_attempts):
                seed = base.copy()
                if attempt:
                    seed[indices] = rng.uniform(limits[indices, 0], limits[indices, 1])
                q, ok, error = pk.compute_IK_CLIK(palm_index, audit.Pose(target), seed,
                    mask.tolist(), eps=1e-7, max_iter=2000, dt=.1, damp=1e-8)
                q = np.asarray(q)
                row = {'target': label, 'attempt': attempt, 'IK_converged': bool(ok)}
                metadata['endpoint_attempts'].append(row)
                if not (ok and np.isfinite(q).all()
                        and np.all(q[indices] >= limits[indices, 0] - 1e-5)
                        and np.all(q[indices] <= limits[indices, 1] + 1e-5)
                        and np.max(abs(q[mask] - base[mask])) < 1e-8):
                    row['rejection_reason'] = 'Original IK/commanded bounds/frozen-joint predicate'
                    continue
                declared, declaration = equivalent.declare_bounded_equivalent_goals(
                    q, indices, limits, joint_types, native_fk)
                row['equivalent_goal_declaration'] = declaration
                legal = []
                row['declared_goal_collision_checks'] = []
                for declared_index, candidate in enumerate(declared):
                    reset(candidate)
                    collision_count = len(world.check_collision())
                    row['declared_goal_collision_checks'].append({
                        'declared_goal_index': declared_index,
                        'full_qpos_rad': candidate.tolist(), 'collision_count': collision_count})
                    if not collision_count:
                        legal.append(candidate.copy())
                row['native_free_declared_goal_count'] = len(legal)
                if not legal:
                    continue
                legal.sort(key=lambda candidate: np.linalg.norm(candidate[indices] - base[indices]))
                covered = [any(np.linalg.norm(candidate[indices] - old[indices]) < 1e-3
                               for family in result for old in family['goals']) for candidate in legal]
                row['predeclared_numeric_goals_already_covered'] = covered
                if all(covered):
                    row['all_predeclared_numeric_goals_already_covered'] = True
                    continue
                result.append({'IK_attempt': attempt, 'original_IK_full_qpos_rad': q.copy(), 'goals': legal})
            return sorted(result, key=lambda family: min(np.linalg.norm(candidate[indices] - base[indices]) for candidate in family['goals']))

        def leg(planner, start, family, label):
            declared = family['goals']
            numeric_goals = [candidate[indices] for candidate in declared]
            row = {'leg': label, 'IK_attempt': family['IK_attempt'],
                   'original_IK_full_qpos_rad': family['original_IK_full_qpos_rad'].tolist(),
                   'predeclared_native_free_goals_arm_rad': [candidate.tolist() for candidate in numeric_goals],
                   'predeclared_native_free_goals_full_rad': [candidate.tolist() for candidate in declared]}
            reset(start)
            stamp = time.time()
            status, raw = planner.plan(start[indices], numeric_goals, time=10.,
                range=.05, simplify=True, verbose=False)
            path = np.asarray(raw)
            row.update(RRT_status=str(status), elapsed_s=time.time() - stamp, raw_path_arm_rad=path.tolist())
            if str(status).rstrip('.') != 'Exact solution':
                row['rejection_reason'] = 'Original exact RRT status predicate'
                return None, row
            matched = equivalent.match_declared_numeric_endpoint(path, start[indices], numeric_goals)
            row['numeric_endpoint_match'] = matched
            row['raw_path_rows'] = len(path) if path.ndim else 0
            if not matched['passed']:
                row['rejection_reason'] = matched['rejection_reason']
                return None, row
            full = np.tile(base, (len(path), 1))
            full[:, indices] = path
            row.update(raw_start_arm_rad=path[0].tolist(), raw_end_arm_rad=path[-1].tolist(),
                       raw_path_full_qpos_rad=full.tolist(), raw_start_full_qpos_rad=full[0].tolist(),
                       raw_end_full_qpos_rad=full[-1].tolist(),
                       raw_start_delta_per_joint_rad=(path[0] - start[indices]).tolist(),
                       raw_min_per_commanded_joint_rad=path.min(axis=0).tolist(),
                       raw_max_per_commanded_joint_rad=path.max(axis=0).tolist(),
                       raw_end_delta_per_declared_goal_rad=[(path[-1] - candidate).tolist() for candidate in numeric_goals])
            check = fma.joint_limit_audit(np.vstack([start, full]), limits, names, indices)
            row['raw_joint_limit_audit'] = check
            if not check['passed']:
                row['rejection_reason'] = 'Native raw commanded joint limits predicate'
                return None, row
            return path, row

        reset(base)
        collisions = world.check_collision()
        if collisions:
            metadata['start_collision_count'] = len(collisions)
            return {'status': 'Fail', 'reason': 'Full native waypoint approach start collides', 'planning_metadata': metadata}
        vias = families(via, 'via_goal_world_z_plus_80mm', 13331)
        goals = families(goal, 'original_goal', 26663)
        metadata.update(native_free_via_family_count=len(vias), native_free_goal_family_count=len(goals))
        planner = OMPLPlanner(world)
        for via_branch, via_family in enumerate(vias[:2]):
            for goal_branch, goal_family in enumerate(sorted(goals, key=lambda family:
                    min(np.linalg.norm(candidate[indices] - via_candidate[indices])
                        for candidate in family['goals'] for via_candidate in via_family['goals']))[:3]):
                row = {'via_branch': via_branch, 'goal_branch': goal_branch}
                metadata['RRT_attempts'].append(row)
                first, first_result = leg(planner, base, via_family, 'start_to_raised_goal')
                row['first_leg'] = first_result
                if first is None:
                    continue
                join = base.copy()
                join[indices] = first[-1]
                second, second_result = leg(planner, join, goal_family, 'raised_goal_to_original_goal')
                row['second_leg'] = second_result
                if second is None:
                    continue
                row['raw_join_error_rad'] = float(np.max(abs(first[-1] - second[0])))
                if row['raw_join_error_rad'] > equivalent.RAW_START_TOLERANCE_RAD:
                    row['rejection_reason'] = 'Original raw start predicate at concatenation'
                    continue
                # Keep every raw RRT row, including both rows at the seam.
                path = np.vstack([first, second])
                raw_full = np.tile(base, (len(path), 1))
                raw_full[:, indices] = path
                raw_check = fma.joint_limit_audit(np.vstack([base, raw_full]), limits, names, indices)
                row.update(raw_path_rows=len(path), raw_path_arm_rad=path.tolist(),
                           raw_path_full_qpos_rad=raw_full.tolist(), raw_joint_limit_audit=raw_check)
                if not raw_check['passed']:
                    row['rejection_reason'] = 'Native concatenated raw commanded joint limits predicate'
                    continue
                position, velocity, times, retime = official_topp_polyline(task, path, side, time_step=time_step)
                full = np.tile(base, (len(position), 1))
                full[:, indices] = position
                check = fma.joint_limit_audit(np.vstack([base, full]), limits, names, indices)
                row.update(retimed_joint_limit_audit=check, retimed_rows=len(position),
                           complete_original_official_TOPP_retiming=retime,
                           actual_TOPP_audit_vertices_arm_rad=np.asarray(retime['audit_vertices']).tolist(),
                           retimed_commanded_position_arm_rad=np.asarray(position).tolist(),
                           retimed_commanded_velocity_arm_rad_s=np.asarray(velocity).tolist(),
                           retimed_official_native_times_s=np.asarray(times).tolist())
                if not check['passed']:
                    row['rejection_reason'] = 'Original retimed commanded joint limits predicate'
                    continue
                model.set_base_pose(old_pose)
                model.set_qpos(old_q, full=True)
                geometry = audit.audit(retime['audit_vertices'], side,
                    min_bottle_clearance_m=.0005, min_self_clearance_m=0., min_table_clearance_m=0.)
                geometry, refinement = dense_policy.refine_certificate(audit, retime['audit_vertices'], side, geometry)
                row['strict_dense_certificate_refinement'] = refinement
                row.update(audit_passed=geometry['passed'], audit=geometry)
                if not geometry['passed']:
                    row['rejection_reason'] = 'Original complete native geometry clearance predicate'
                    continue
                metadata.update(selected_via_branch=via_branch, selected_goal_branch=goal_branch,
                                selected_via_declared_goal_index=first_result['numeric_endpoint_match']['matched_declared_goal_index'],
                                selected_goal_declared_goal_index=second_result['numeric_endpoint_match']['matched_declared_goal_index'],
                                selected_numeric_via_arm_qpos_rad=first[-1].tolist(),
                                selected_numeric_goal_arm_qpos_rad=second[-1].tolist(),
                                selected_raw_joint_polyline=path.tolist(), retiming=retime)
                return {'status': 'Success', 'position': position, 'velocity': velocity,
                        'time': times, 'duration': float(times[-1]), 'audit': geometry,
                        'planning_metadata': metadata}
        return {'status': 'Fail', 'reason': 'No raised-goal finite exact native path passed unchanged TOPP/full mesh predicates',
                'planning_metadata': metadata}
    finally:
        model.set_move_group(old_group)
        model.set_base_pose(old_pose)
        model.set_qpos(old_q, full=True)
        if list(model.get_move_group_end_effectors()) != old_group or list(model.get_move_group_joint_indices()) != old_group_indices:
            raise RuntimeError('Auxiliary move-group restoration differs from original')
        if not np.array_equal(np.asarray(model.get_qpos()), old_q):
            raise RuntimeError('Auxiliary joint-state restoration differs from original')
        _assert_unchanged(task, before)
        metadata.update(physical_state_unchanged=True,
                        auxiliary_state_restored_including_construction_exception=True,
                        original_auxiliary_move_group_restored=True)
