def plan_carry_to_task_pose(task, task_pose, side='left', *, palm_to_bottle=None, time_step=0.004, max_joint_velocity=1.0, max_joint_acceleration=1.0, min_duration_s=0.0, planning_time_s=10.0, ik_attempts=24, rrt_range_rad=0.05, max_goal_branches=6, candidate_arm_q=None, position_tolerance_m=5e-05, rotation_tolerance_rad=0.0002, random_seed=7, audit_options=None, endpoint_only=False, official_TOPP=True, support_departure=False):
    """Same task pose, full-native masked IK endpoints then RRT and complete audit.
    Goal IK is never clipped or wrapped. Noncommanded actual joints stay frozen.
    Supplied candidate_arm_q entries are checked like every IK branch, not trusted.
    Return Success only after full carry continuous mesh audit passes.
    """
    if side not in ('left', 'right'):
        raise ValueError('side must be left/right')
    if not isinstance(support_departure, bool):
        raise ValueError('support_departure must be explicit bool')
    if support_departure and (endpoint_only or not official_TOPP):
        raise ValueError('Initial support departure requires full trajectory and actual official TOPP')
    if not isinstance(ik_attempts, int) or ik_attempts < 1 or (not isinstance(max_goal_branches, int)) or (max_goal_branches < 1):
        raise ValueError('Invalid branch budget')
    if not math.isfinite(planning_time_s) or planning_time_s <= 0 or (not math.isfinite(rrt_range_rad)) or (rrt_range_rad <= 0):
        raise ValueError('Invalid RRT budget')
    if position_tolerance_m <= 0 or rotation_tolerance_rad <= 0:
        raise ValueError('Invalid endpoint tolerances')
    for value in (time_step, max_joint_velocity, max_joint_acceleration):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid retiming options')
    if not math.isfinite(min_duration_s) or min_duration_s < 0:
        raise ValueError('Invalid minimum duration')
    from mplib import PlanningWorld
    from mplib.pymp.planning.ompl import OMPLPlanner
    import preflight_place_goals as ppg
    before = ppg._snapshot(task)
    started = time.monotonic()
    auditor = cma.CarryMeshPathAuditor(task, side, palm_to_bottle=palm_to_bottle)
    (model, _numeric_window_receipt) = _window_create_model(auditor, side)
    _window_planning_pk = model.get_pinocchio_model()
    oldq = cma.fma._array(model.get_qpos()).copy()
    old_base_pose = auditor.Pose(model.get_base_pose())
    world = None
    meta = {'scope': 'Independent full-native CPU IK/RRT, planning-only attached collision object, physical scene unchanged.', 'task_goal_pose_world': ppg._vector(task_pose), 'ik_attempts_requested': ik_attempts, 'endpoint_position_tolerance_m': position_tolerance_m, 'endpoint_rotation_tolerance_rad': rotation_tolerance_rad, 'auxiliary_numeric_joint_window': _numeric_window_receipt}
    try:
        palm_goal = cma._matrix(task.robot._trans_from_gripper_to_endlink(list(map(float, ppg._vector(task_pose))), arm_tag=side))
        root = cma._matrix(auditor.art.root_pose)
        goal_base = np.linalg.inv(root) @ palm_goal
        base = auditor.initial_sim_q.copy()
        names = auditor.sim_joints
        arm_names = list(getattr(task.robot, side + '_arm_joints_name'))
        indices = [names.index(n) for n in arm_names]
        frozen = [i for i in range(len(base)) if i not in indices]
        model.set_move_group(auditor.palm_name)
        if list(model.get_move_group_joint_indices()) != indices or list(model.get_move_group_joint_names()) != arm_names:
            raise ValueError('Full-native move-group ordering does not match commanded arm joints')
        inverse_root = np.linalg.inv(root)
        model.set_base_pose(auditor.Pose())
        model.set_qpos(base, full=True)
        source_table = auditor.static['table']
        source_bottle = auditor.static['bottle']

        def clone_obstacle(source, name, pose):
            shapes = [auditor.fcl.CollisionObject(shape.get_collision_geometry()) for shape in source.shapes]
            locals = [auditor.Pose(local) for local in source.shape_poses]
            return auditor.fcl.FCLObject(name, auditor.Pose(pose), shapes, locals)
        table_object = clone_obstacle(source_table, 'realman_table', inverse_root @ cma._matrix(source_table.pose))
        bottle_object = clone_obstacle(source_bottle, 'held_bottle', inverse_root @ cma._matrix(source_bottle.pose))
        if table_object.name in auditor.objects or bottle_object.name in auditor.objects:
            raise ValueError('Obstacle name conflicts with robot link')
        table_before = cma._matrix(table_object.pose)
        table_shape_before = [cma._matrix(shape.get_pose()) for shape in table_object.shapes]
        source_table_shape_before = [cma._matrix(shape.get_pose()) for shape in source_table.shapes]

        def assert_table_unchanged():
            if not np.allclose(cma._matrix(table_object.pose), table_before, atol=1e-10, rtol=0):
                raise RuntimeError('Planning table global pose changed')
            for (shape, old) in zip(table_object.shapes, table_shape_before):
                if not np.allclose(cma._matrix(shape.get_pose()), old, atol=1e-10, rtol=0):
                    raise RuntimeError('Planning table shape moved')
            for (shape, old) in zip(source_table.shapes, source_table_shape_before):
                if not np.allclose(cma._matrix(shape.get_pose()), old, atol=1e-10, rtol=0):
                    raise RuntimeError('FMA world table changed through shared CollisionObject')
        world = PlanningWorld([model], [table_object, bottle_object] + [clone_obstacle(source, 'realman_' + name, inverse_root @ cma._matrix(source.pose)) for (name, source) in auditor.static.items() if name not in ('bottle', 'table')])
        acm = world.get_allowed_collision_matrix()
        registry = list(world.get_object_names())
        if set(registry) != {'realman_table', 'held_bottle'} | {'realman_' + name for name in auditor.static if name not in ('bottle', 'table')}:
            raise RuntimeError('Invalid distinct world registry names')
        for pair in auditor.allowed:
            (a, b) = sorted(pair)
            acm.set_entry(a, b, True)
        palm_index = list(_window_planning_pk.get_link_names()).index(auditor.palm_name)
        user_palm_index = list(model.get_user_link_names()).index(auditor.palm_name)
        if user_palm_index != palm_index:
            raise RuntimeError('PK/user attachment indices differ; explicit diagnostic needed')
        world.attach_object('held_bottle', model.get_name(), user_palm_index, auditor.Pose(auditor.reference), sorted(auditor.allowed_bottle_hand))
        world.set_qpos_all(base[indices])
        attached = world.get_attached_object('held_bottle')
        attached.update_pose()
        anchor_base = cma._matrix(attached.get_attached_link_global_pose())
        bottle_base = cma._matrix(attached.get_global_pose())
        actual_palm = auditor.initial_scene_poses[auditor.palm_name]
        actual_bottle = cma._matrix(task.bottle.get_pose())
        anchor_error = _pose_error(actual_palm, root @ anchor_base)
        bottle_error = _pose_error(actual_bottle, root @ bottle_base)
        predicted_error = _pose_error(actual_palm @ auditor.reference, root @ bottle_base)
        if anchor_error[0] > 2e-05 or anchor_error[1] > 0.0002:
            raise RuntimeError('Attachment anchor does not match actual palm after BASE->WORLD transform: ' + str(anchor_error))
        if bottle_error[0] > 2e-05 or bottle_error[1] > 0.0002:
            raise RuntimeError('Attachment bottle does not match actual actor after BASE->WORLD transform: ' + str(bottle_error))
        shape_errors = []
        attached_object = attached.get_object()
        for (shape, local) in zip(attached_object.shapes, attached_object.shape_poses):
            shape_errors.append(_pose_error(bottle_base @ cma._matrix(local), cma._matrix(shape.get_pose())))
        if any((pe > 2e-05 or re > 0.0002 for (pe, re) in shape_errors)):
            raise RuntimeError('Attached collision shape frame mismatch')
        assert_table_unchanged()
        if not world.is_object_attached('held_bottle') or world.is_object_attached('realman_table'):
            raise RuntimeError('Wrong auxiliary attachment')
        meta.update(planning_coordinate_frame='Robot BASE, ALL geometry consistently transformed from unchanged scene WORLD', planning_model_base_pose_identity=True, world_from_planning_base=root.tolist(), planning_object_names=registry, only_planning_attached_object='held_bottle', independent_obstacle_collision_objects=True, table_world_pose_unchanged=True, attachment_frame_validation={'anchor_world_error_m_rad': list(anchor_error), 'bottle_actor_world_error_m_rad': list(bottle_error), 'bottle_predicted_world_error_m_rad': list(predicted_error), 'collision_shape_pose_errors_m_rad': [list(x) for x in shape_errors], 'anchor_matrix_base': anchor_base.tolist(), 'bottle_matrix_base': bottle_base.tolist()}, actual_start_arm_joint_rad=base[indices].tolist(), goal_palm_matrix_world=palm_goal.tolist(), goal_palm_matrix_base=goal_base.tolist(), move_group_joint_names=arm_names, frozen_joint_names=[names[i] for i in frozen], native_self_exception_count=len(auditor.allowed), actual_bottle_touch_links=sorted(auditor.allowed_bottle_hand), start_collisions=_collision_summary(world), full_mesh_metadata=auditor.metadata)
        if support_departure:
            if not _support_start_pair_only(meta['start_collisions']):
                return {'status': 'Fail', 'reason': 'Support departure permits ONLY the initial held_bottle/realman_table pair', 'planning_metadata': meta}
            expected = actual_palm.copy()
            expected[:3, 3] += [0, 0, 0.1]
            (pe, re) = _pose_error(expected, palm_goal)
            if pe > position_tolerance_m or re > rotation_tolerance_rad:
                return {'status': 'Fail', 'reason': 'Support departure target must be the first official WORLD z+.1m fixed-orientation lift', 'planning_metadata': meta}
            vertices = auditor.bottle_vertices @ actual_bottle[:3, :3].T + actual_bottle[:3, 3]
            signed_baseline = float(vertices[:, 2].min() - auditor.static_boxes['table'][1, 2])
            if signed_baseline < -0.0005 or signed_baseline >= 0.001:
                return {'status': 'Fail', 'reason': 'Initial actual signed support depth outside [-.5mm,1mm)', 'planning_metadata': meta}
            meta['support_departure'] = {'enabled': True, 'planning_only_allowed_pair': ['held_bottle', 'realman_table'], 'actual_start_signed_gap_m': signed_baseline, 'max_initial_penetration_m': 0.0005, 'max_extra_depth_m': 2e-05, 'release_gap_m': 0.001, 'physical_table_collisions_unchanged': True, 'qualification': 'Every candidate requires unchanged full CMA raw evidence plus independent continuous signed support-exit certificate'}
            acm.set_entry('held_bottle', 'realman_table', True)
            if _collision_summary(world):
                raise RuntimeError('Other start collisions remained after the sole planning support pair')
        elif meta['start_collisions']:
            return {'status': 'Fail', 'reason': 'Full-native start state collides', 'planning_metadata': meta}
        limits = np.concatenate([cma.fma._array(j.limits).reshape(-1, 2) for j in auditor.art.active_joints])
        mask = np.ones(len(base), dtype=bool)
        mask[indices] = False
        rng = np.random.default_rng(random_seed)
        candidates = []
        endpoint_reports = []

        def inspect_candidate(q, label, ik_success=True, ik_error=None):
            q = np.asarray(q, dtype=float).copy()
            entry = {'label': label, 'ik_success': bool(ik_success)}
            if ik_error is not None:
                entry['ik_error_vector'] = np.asarray(ik_error).tolist()
            if q.shape != base.shape or not np.isfinite(q).all():
                entry['reason'] = 'Invalid joint vector'
                endpoint_reports.append(entry)
                return
            entry['arm_joint_rad'] = q[indices].tolist()
            if not ik_success:
                entry['reason'] = 'CLIK did not converge'
                endpoint_reports.append(entry)
                return
            if frozen and np.max(np.abs(q[frozen] - base[frozen])) > 1e-08:
                entry['reason'] = 'IK changed frozen joints'
                endpoint_reports.append(entry)
                return
            check = cma.fma.joint_limit_audit(np.vstack([base, q]), limits, names, indices)
            entry['joint_limit_audit'] = check
            if not check['passed']:
                entry['reason'] = 'Strict joint limit rejection'
                endpoint_reports.append(entry)
                return
            if not _window_contains_selected_arm(model, q, indices):
                entry['reason'] = 'Outside independent near numeric planning window; physical vendor goal is not modified'
                endpoint_reports.append(entry)
                return
            world.set_qpos_all(q[indices])
            assert_table_unchanged()
            _window_planning_pk.compute_forward_kinematics(q)
            endpoint = root @ cma._matrix(_window_planning_pk.get_link_pose(palm_index))
            (pe, re) = _pose_error(palm_goal, endpoint)
            entry.update(position_error_m=pe, rotation_error_rad=re, collisions=_collision_summary(world))
            if pe > position_tolerance_m or re > rotation_tolerance_rad:
                entry['reason'] = 'Endpoint FK mismatch'
            elif entry['collisions']:
                entry['reason'] = 'Full-native endpoint collision'
            elif any((np.linalg.norm(q[indices] - old[indices]) < 0.001 for old in candidates)):
                entry['reason'] = 'Duplicate endpoint branch'
            else:
                candidates.append(q)
                entry['reason'] = 'Accepted free endpoint'
                entry['joint_distance_from_start_rad'] = float(np.linalg.norm(q[indices] - base[indices]))
                entry['wrist6_delta_rad'] = float(q[indices[-1]] - base[indices[-1]])
            endpoint_reports.append(entry)
        support_path = None
        if support_departure:
            try:
                cartesian_knots = cartesian_world_lift_knots(actual_palm, palm_goal, position_tolerance_m=position_tolerance_m, rotation_tolerance_rad=rotation_tolerance_rad)
                meta['cartesian_world_knots'] = cartesian_knots.tolist()
                meta['cartesian_ik_scope'] = '21 WORLD vertical poses, fixed orientation/XY; previous full native IK seed; all 30 other actual DOFs frozen'
                meta['frozen_actual_qpos_rad'] = base.tolist()
                meta['prior_actual_palm_world'] = actual_palm.tolist()
                meta['prior_native_FK_palm_world'] = (root @ anchor_base).tolist()
                solved = [base.copy()]
                seed = base.copy()
                knot_reports = []
                for (knot_index, knot_world) in enumerate(cartesian_knots[1:], 1):
                    knot_base = inverse_root @ knot_world
                    (q, ok, error) = _window_planning_pk.compute_IK_CLIK(palm_index, auditor.Pose(knot_base), seed, mask.tolist(), eps=1e-07, max_iter=2000, dt=0.1, damp=1e-08)
                    q = np.asarray(q, dtype=float)
                    entry = {'knot_index': knot_index, 'IK_success': bool(ok), 'IK_error': np.asarray(error).tolist()}
                    knot_reports.append(entry)
                    meta['cartesian_ik_knots'] = knot_reports
                    if q.shape != base.shape or not np.isfinite(q).all() or (not ok):
                        raise ValueError('Full native Cartesian CLIK failed at knot ' + str(knot_index))
                    if frozen and np.max(np.abs(q[frozen] - base[frozen])) > 1e-08:
                        raise ValueError('Cartesian CLIK changed frozen actual joints')
                    check = cma.fma.joint_limit_audit(np.vstack([base, q]), limits, names, indices)
                    entry['joint_limit_audit'] = check
                    if not check['passed']:
                        raise ValueError('Cartesian knot joint limits rejected')
                    if not _window_contains_selected_arm(model, q, indices):
                        raise ValueError('Cartesian support knot exits independent numeric planning window; never wrap')
                    world.set_qpos_all(q[indices])
                    assert_table_unchanged()
                    _window_planning_pk.compute_forward_kinematics(q)
                    actual_knot = root @ cma._matrix(_window_planning_pk.get_link_pose(palm_index))
                    (pe, re) = _pose_error(knot_world, actual_knot)
                    entry.update(position_error_m=pe, rotation_error_rad=re, arm_joint_rad=q[indices].tolist(), collisions=_collision_summary(world))
                    if pe > position_tolerance_m or re > rotation_tolerance_rad:
                        raise ValueError('Cartesian knot actual native FK rejected')
                    if entry['collisions']:
                        raise ValueError('Cartesian knot non-support full-native collision')
                    solved.append(q.copy())
                    seed = q.copy()
                support_path = np.asarray(solved)[:, indices]
                candidates.append(solved[-1])
                endpoint_reports = knot_reports
                meta['cartesian_full_native_knots_rad'] = np.asarray(solved).tolist()
            except (ValueError, RuntimeError, AssertionError) as error:
                return {'status': 'Fail', 'reason': 'Cartesian support lift construction rejected: ' + str(error), 'planning_metadata': meta}
        else:
            if candidate_arm_q is not None:
                supplied = np.asarray(candidate_arm_q, dtype=float)
                if supplied.ndim == 1:
                    supplied = supplied[None, :]
                if supplied.ndim != 2 or supplied.shape[1] != len(indices) or (not np.isfinite(supplied).all()):
                    raise ValueError('Invalid supplied candidate arm joints')
                for (i, qarm) in enumerate(supplied):
                    q = base.copy()
                    q[indices] = qarm
                    inspect_candidate(q, 'supplied_' + str(i))
            for attempt in range(ik_attempts):
                seed = base.copy()
                if attempt:
                    if attempt < ik_attempts // 2:
                        seed[indices] += rng.normal(0, 0.15 + 0.07 * attempt, len(indices))
                        for i in indices:
                            if seed[i] < limits[i, 0] or seed[i] > limits[i, 1]:
                                seed[i] = rng.uniform(limits[i, 0], limits[i, 1])
                    else:
                        seed[indices] = rng.uniform(limits[indices, 0], limits[indices, 1])
                (q, ok, error) = _window_planning_pk.compute_IK_CLIK(palm_index, auditor.Pose(goal_base), seed, mask.tolist(), eps=1e-07, max_iter=2000, dt=0.1, damp=1e-08)
                inspect_candidate(q, 'CLIK_' + str(attempt), ok, error)
        meta.update(endpoint_candidates=endpoint_reports, accepted_free_endpoints=len(candidates))
        if not candidates:
            return {'status': 'Fail', 'reason': 'No free exact endpoint among finite IK/candidate search; this is not a proof of global unreachability.', 'planning_metadata': meta}
        if endpoint_only:
            return {'status': 'EndpointSuccess', 'planning_metadata': meta, 'free_goal_arm_joint_rad': [q[indices].tolist() for q in candidates]}
        candidates.sort(key=lambda q: float(np.linalg.norm(q[indices] - base[indices])) + 0.15 * abs(float(q[indices[-1]] - base[indices[-1]])))
        planner = OMPLPlanner(world)
        attempts = []
        for (branch, goal) in enumerate(candidates[:max_goal_branches]):
            model.set_base_pose(auditor.Pose())
            model.set_qpos(base, full=True)
            world.set_qpos_all(base[indices])
            stamp = time.monotonic()
            if support_departure:
                status = 'Exact solution'
                path = support_path.copy()
            else:
                (status, path) = planner.plan(base[indices], [goal[indices]], time=30.0, range=0.01, simplify=False, verbose=False)
            assert_table_unchanged()
            detail = {'branch': branch, 'goal_arm_joint_rad': goal[indices].tolist(), 'RRT_status': None if support_departure else str(status), 'RRT_elapsed_s': None if support_departure else time.monotonic() - stamp, 'planning_algorithm': 'Full native 21-knot Cartesian CLIK' if support_departure else 'Original strict full native OMPL RRT'}
            attempts.append(detail)
            if str(status).rstrip('.') != 'Exact solution':
                continue
            path = np.asarray(path, dtype=float)
            if path.ndim != 2 or path.shape[1] != len(indices) or len(path) < 2 or (not np.isfinite(path).all()):
                detail['reason'] = 'Invalid OMPL path'
                continue
            if np.max(np.abs(path[0] - base[indices])) > 1e-07 or np.max(np.abs(path[-1] - goal[indices])) > 1e-06:
                detail['reason'] = 'OMPL endpoint differs from exact checked q'
                continue
            if not _window_contains_arm_path(model, path, indices):
                detail['reason'] = 'Raw or official TOPP carry path exits independent numeric planning window'
                continue
            command_full = np.tile(base, (len(path), 1))
            command_full[:, indices] = path
            check = cma.fma.joint_limit_audit(np.vstack([base, command_full]), limits, names, indices)
            if not check['passed']:
                detail['reason'] = 'RRT strict commanded joint limits rejected'
                detail['joint_limit_audit'] = check
                continue
            try:
                if official_TOPP:
                    if min_duration_s != 0:
                        raise ValueError('Official TOPP mode forbids artificial minimum duration')
                    (position, velocity, timing, retime) = (official_topp_command_polyline if support_departure else official_topp_polyline)(task, path, side, time_step=time_step)
                    geometry = np.asarray(retime['audit_vertices'], dtype=float)
                    retime['unused_external_velocity_acceleration_arguments'] = [max_joint_velocity, max_joint_acceleration]
                else:
                    (position, velocity, timing, retime) = retime_polyline(path, time_step=time_step, max_joint_velocity=max_joint_velocity, max_joint_acceleration=max_joint_acceleration, min_duration_s=min_duration_s)
                    geometry = path
            except (ValueError, RuntimeError, AssertionError) as error:
                detail['reason'] = 'Retiming rejected: ' + str(error)
                continue
            if not _window_contains_arm_path(model, position, indices):
                detail['reason'] = 'Raw or official TOPP carry path exits independent numeric planning window'
                continue
            retimed_full = np.tile(base, (len(position), 1))
            retimed_full[:, indices] = position
            returned_limits = cma.fma.joint_limit_audit(np.vstack([base, retimed_full]), limits, names, indices)
            if not returned_limits['passed']:
                detail.update(reason='Actual TOPP command joint limits rejected', joint_limit_audit=returned_limits)
                continue
            opts = dict(audit_options or {})
            model.set_base_pose(old_base_pose)
            model.set_qpos(oldq, full=True)
            audit = auditor.audit(geometry, side, **opts)
            if support_departure:
                from support_departure_certificate_v20 import certify_support_departure
                support = certify_support_departure(task, geometry, side, palm_to_bottle=auditor.reference, audit_options=opts, endpoint_translation_tolerance_m=position_tolerance_m, endpoint_rotation_tolerance_rad=rotation_tolerance_rad)
                detail['support_departure_combined_audit'] = support
                audit = {**audit, 'passed': bool(support['passed']), 'raw_ordinary_audit_passed': audit['passed'], 'support_departure_combined_audit': support, 'support_departure_certificate': support['signed_support_departure_certificate'], 'physical_signed_height_guard': support['physical_signed_height_guard'], 'scope': 'Initial first WORLD upward support phase ONLY. Ordinary bottle_table evidence retained; support certificate replaces only that category. All other full meshes unchanged.'}
            else:
                audit = resolve_bottle_table_bound(task, geometry, side, auditor.reference, audit, opts)
            detail.update(polyline_waypoints=len(path), audit_passed=audit['passed'], retiming=retime, audit_collision_pairs=audit['collisions_first_64'], audit_continuous_lower_bound=audit['continuous_gap_lower_bound'])
            if not audit['passed']:
                continue
            meta.update(RRT_attempts=attempts, selected_branch=branch, elapsed_s=time.monotonic() - started, audit_curve_relation=retime.get('curve', 'Actual native TOPP curve proof'), geometric_polyline=path.tolist(), retiming=retime, physical_state_unchanged=True)
            return {'status': 'Success', 'position': position, 'velocity': velocity, 'time': timing, 'duration': float(timing[-1]), 'audit': audit, 'planning_metadata': meta}
        meta.update(RRT_attempts=attempts, elapsed_s=time.monotonic() - started)
        return {'status': 'Fail', 'reason': 'Cartesian first-lift returned command polyline failed complete audits' if support_departure else 'No RRT branch passed complete continuous carry audit.', 'planning_metadata': meta}
    finally:
        model.set_base_pose(old_base_pose)
        model.set_qpos(oldq, full=True)
        source = auditor.static['bottle']
        auditor.static['bottle'] = auditor.fcl.FCLObject(source.name, auditor.original_bottle_pose, source.shapes, source.shape_poses)
        ppg._assert_snapshot(task, before)
        meta['physical_state_unchanged'] = True
