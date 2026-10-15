"""Native free-hand planning to explicitly declared finite equivalent goals.

All goals precede RRT and are checked with native FK for every link and the
complete native collision world. Returned paths are never wrapped or clipped.
Original numeric tolerances, official TOPP, native mesh guards and v9 physics
remain independent gates. This is not evidence of historical failure causes.
"""
import time
import numpy as np
import equivalent_arm_goals_v5 as equivalent
import dense_free_certificate_policy_v7 as dense_policy
from full_native_approach_waypoint_v4 import _physical_snapshot, _assert_unchanged


def plan_free_approach(task, target_pose, side, *, time_step=.004, random_seed=0, ik_attempts=24):
    import full_mesh_path_audit as fma
    from full_native_carry_planner_v31 import official_topp_polyline
    from mplib import PlanningWorld
    from mplib.pymp.planning.ompl import OMPLPlanner
    before = _physical_snapshot(task)
    audit = fma.FullMeshPathAuditor(task)
    clutter_coverage = _require_clutter_coverage(task, audit)
    model = audit.model
    old_pose, old_q = model.get_base_pose(), model.get_qpos().copy()
    old_group = list(model.get_move_group_end_effectors())
    old_group_indices = list(model.get_move_group_joint_indices())
    metadata = {'backend': 'full_native_free_hand_MPLib_explicit_bounded_equivalent_goals_dense_v7',
                'physical_state_unchanged': False, 'no_attached_objects': True,
                'raw_path_never_wrapped_or_clipped': True,
                'historical_alias_failure_not_inferred': True,
                'endpoint_attempts': [], 'RRT_attempts': [],
                'motion_quality_overlay': _OVERLAY_RECEIPT,
                'randomized_obstacle_coverage': clutter_coverage}
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
        goal = inverse @ goal_world
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
        rng = np.random.default_rng(random_seed)
        metadata.update(initial_actual_full_qpos_rad=before['q'].tolist(),
                        initial_native_bottle_matrix=before['bottle'].tolist(),
                        initial_model_base_pose_matrix=fma._pose_matrix(old_pose).tolist(),
                        selected_arm_joint_names=[names[i] for i in indices],
                        selected_arm_indices=indices, native_FK_link_names=link_names,
                        goal_palm_world_matrix=goal_world.tolist(),
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

        reset(base)
        collisions = world.check_collision()
        if collisions:
            metadata['start_collision_count'] = len(collisions)
            return {'status': 'Fail', 'reason': 'Full native approach start collides', 'planning_metadata': metadata}
        families = []
        for attempt in range(ik_attempts):
            seed = base.copy()
            if attempt:
                seed[indices] = rng.uniform(limits[indices, 0], limits[indices, 1])
            q, ok, error = pk.compute_IK_CLIK(palm_index, audit.Pose(goal), seed,
                mask.tolist(), eps=1e-7, max_iter=2000, dt=.1, damp=1e-8)
            q = np.asarray(q)
            row = {'attempt': attempt, 'IK_converged': bool(ok)}
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
            legal.sort(key=lambda c: np.linalg.norm(c[indices] - base[indices]))
            covered = [any(np.linalg.norm(c[indices] - old[indices]) < 1e-3
                           for family in families for old in family['goals']) for c in legal]
            row['predeclared_numeric_goals_already_covered'] = covered
            if all(covered):
                row['all_predeclared_numeric_goals_already_covered'] = True
                continue
            families.append({'IK_attempt': attempt, 'original_IK_full_qpos_rad': q.copy(), 'goals': legal})
        families.sort(key=lambda f: min(np.linalg.norm(c[indices] - base[indices]) for c in f['goals']))
        metadata['native_free_goal_family_count'] = len(families)
        planner = OMPLPlanner(world)
        for branch, family in enumerate(families[:6]):
            declared, near_selection = choose_nearest_family(family, base, indices)
            numeric_goals = [c[indices] for c in declared]
            row = {'branch': branch, 'IK_attempt': family['IK_attempt'],
                   'original_IK_full_qpos_rad': family['original_IK_full_qpos_rad'].tolist(),
                   'predeclared_native_free_goals_arm_rad': [c.tolist() for c in numeric_goals],
                   'predeclared_native_free_goals_full_rad': [c.tolist() for c in declared]}
            row['near_numeric_goal_selection'] = near_selection
            row['chosen_goal_target_FK_proof'] = _strict_palm_goal(declared[0], base, indices, native_fk, goal, palm_index, len(link_names))
            metadata['RRT_attempts'].append(row)
            if not row['chosen_goal_target_FK_proof']['passed']:
                row['rejection_reason'] = 'Chosen near numeric goal differs from exact native task target'
                continue
            reset(base)
            stamp = time.time()
            status, raw = planner.plan(base[indices], numeric_goals, time=10.,
                range=.05, simplify=True, verbose=False)
            path = np.asarray(raw)
            row.update(RRT_status=str(status), elapsed_s=time.time() - stamp,
                       raw_path_arm_rad=path.tolist())
            if str(status).rstrip('.') != 'Exact solution':
                row['rejection_reason'] = 'Original exact RRT status predicate'
                continue
            matched = equivalent.match_declared_numeric_endpoint(path, base[indices], numeric_goals)
            row['numeric_endpoint_match'] = matched
            row['raw_path_rows'] = len(path) if path.ndim else 0
            if path.ndim == 2 and path.shape[1] == len(indices) and len(path) and np.isfinite(path).all():
                raw_full = np.tile(base, (len(path), 1))
                raw_full[:, indices] = path
                row.update(raw_start_arm_rad=path[0].tolist(), raw_end_arm_rad=path[-1].tolist(),
                           raw_start_full_qpos_rad=raw_full[0].tolist(), raw_end_full_qpos_rad=raw_full[-1].tolist(),
                           raw_path_full_qpos_rad=raw_full.tolist(),
                           raw_start_delta_per_joint_rad=(path[0] - base[indices]).tolist(),
                           raw_min_per_commanded_joint_rad=path.min(axis=0).tolist(),
                           raw_max_per_commanded_joint_rad=path.max(axis=0).tolist(),
                           raw_end_delta_per_declared_goal_rad=[(path[-1] - c).tolist() for c in numeric_goals])
            if not matched['passed']:
                row['rejection_reason'] = matched['rejection_reason']
                continue
            raw_limits = fma.joint_limit_audit(np.vstack([base, raw_full]), limits, names, indices)
            row['raw_joint_limit_audit'] = raw_limits
            if not raw_limits['passed']:
                row['rejection_reason'] = 'Native raw commanded joint limits predicate'
                continue
            raw_quality, _ = _quality_path(task, path, base, indices, declared[0], limits, pk, palm_index, root, cumulative=True)
            row['motion_quality_raw'] = raw_quality
            if not raw_quality['passed']:
                row['rejection_reason'] = 'New demonstration raw path quality predicate'
                continue
            position, velocity, times, retime = official_topp_polyline(task, path, side, time_step=time_step)
            velocity_limits = np.asarray(retime['official_joint_velocity_limits_rad_s'], dtype=float)
            physical_times = np.arange(len(position)) * float(task.scene.get_timestep())
            dense_quality, dense_xyz = _quality_path(task, position, base, indices, declared[0], limits, pk, palm_index, root, times=physical_times, velocity_limits=velocity_limits, cumulative=True)
            row['motion_quality_retimed'] = dense_quality
            row['retimed_numeric_endpoint_match'] = equivalent.match_declared_numeric_endpoint(position, base[indices], numeric_goals)
            row['audited_vertices_numeric_endpoint_match'] = equivalent.match_declared_numeric_endpoint(np.asarray(retime['audit_vertices']), base[indices], numeric_goals)
            if not (dense_quality['passed'] and row['retimed_numeric_endpoint_match']['passed'] and row['audited_vertices_numeric_endpoint_match']['passed']):
                row['rejection_reason'] = 'New demonstration dense/cumulative quality or specifically chosen TOPP endpoint predicate'
                continue
            full = np.tile(base, (len(position), 1))
            full[:, indices] = position
            limit_check = fma.joint_limit_audit(np.vstack([base, full]), limits, names, indices)
            row.update(retimed_joint_limit_audit=limit_check, retimed_rows=len(position))
            row['complete_original_official_TOPP_retiming'] = retime
            row['actual_TOPP_audit_vertices_arm_rad'] = np.asarray(retime['audit_vertices']).tolist()
            row['retimed_commanded_position_arm_rad'] = np.asarray(position).tolist()
            row['retimed_commanded_velocity_arm_rad_s'] = np.asarray(velocity).tolist()
            row['retimed_official_native_times_s'] = np.asarray(times).tolist()
            if not limit_check['passed']:
                row['rejection_reason'] = 'Original retimed commanded joint limits predicate'
                continue
            model.set_base_pose(old_pose)
            model.set_qpos(old_q, full=True)
            geometry = audit.audit(retime['audit_vertices'], side,
                min_bottle_clearance_m=.0005, min_self_clearance_m=0., min_table_clearance_m=0.)
            geometry, refinement = dense_policy.refine_certificate(
                audit, retime['audit_vertices'], side, geometry)
            row['strict_dense_certificate_refinement'] = refinement
            row.update(audit_passed=geometry['passed'], audit=geometry)
            if not geometry['passed']:
                row['rejection_reason'] = 'Original complete native geometry clearance predicate'
                continue
            selected = matched['matched_declared_goal_index']
            metadata.update(selected_branch=branch, selected_declared_goal_index=selected,
                            selected_numeric_goal_arm_qpos_rad=numeric_goals[selected].tolist(),
                            retiming=retime)
            _record_plan(task, side, position, dense_xyz)
            return {'status': 'Success', 'position': position, 'velocity': velocity,
                    'time': times, 'duration': float(times[-1]), 'audit': geometry,
                    'planning_metadata': metadata}
        return {'status': 'Fail', 'reason': 'No predeclared finite exact native path passed IK/RRT/TOPP/full mesh audit',
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
