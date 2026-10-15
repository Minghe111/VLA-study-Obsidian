"""Task-only planning/recording quality overlay on the SHA-pinned old collector.

Original IK/FK, collision, TOPP, force control and final actual-geometry gates
stay in their original functions. This adds explicit near-goal selection and
independent quality rejections, without rewriting any returned trajectory.
"""
from __future__ import annotations
import ast, copy, hashlib, importlib, json, math, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np

DIRECTORY = Path(__file__).resolve().parent
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))
from near_branch_quality import PrecontactQualityPolicy, path_quality

POLICY = PrecontactQualityPolicy()
SOURCE_PINS = {
    'full_native_approach_equivalent_goals_v6': 'f73514b97ab9795e1bbf7866126aa45ff0f271b91eb08459d423b8fcf9aa0fa2',
    'full_native_approach_equivalent_goals_dense_v7': 'e87bedb3adb98ae5358bc695eed284c067f8c020a1eed742be5de493d84d31cb',
}

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _goal_key(q, base, indices):
    d = np.asarray(q)[indices] - np.asarray(base)[indices]
    return (float(np.linalg.norm(d)), float(abs(d[-1])), tuple(np.asarray(q)[indices]))

def choose_nearest_family(family, base, indices):
    """Only one fully declared/FK-and-collision-checked legal goal reaches RRT."""
    goals = [np.asarray(q, dtype=float).copy() for q in family['goals']]
    if not goals or any(q.shape != np.asarray(base).shape or not np.isfinite(q).all() for q in goals):
        raise ValueError('Missing finite full-q IK family goals')
    chosen = min(range(len(goals)), key=lambda i: _goal_key(goals[i], base, indices))
    return [goals[chosen]], {
        'policy': 'Nearest legal finite numeric endpoint to current actual arm q, per original IK family',
        'original_legal_goal_count': len(goals),
        'original_family_selected_legal_index': chosen,
        'original_legal_goals_full_qpos_rad': [q.tolist() for q in goals],
        'original_legal_goal_distances_L2_rad': [float(np.linalg.norm(q[indices] - np.asarray(base)[indices])) for q in goals],
        'chosen_full_qpos_rad': goals[chosen].tolist(),
        'chosen_arm_qpos_rad': goals[chosen][indices].tolist(),
        'RRT_goal_count': 1, 'path_modified': False,
        'far_equivalent_goal_fallback_allowed': False,
        'original_all_link_FK_and_collision_declaration_retained': True,
    }

def _strict_palm_goal(q, base, indices, native_fk, goal, palm_index, required_link_count):
    """Supplement original IK convergence with explicit all-link/target proof."""
    frozen = np.ones(len(base), dtype=bool); frozen[indices] = False
    matrices = np.asarray(native_fk(q), dtype=float)
    if matrices.shape != (required_link_count, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError('Original complete native FK link inventory was not retained')
    current = matrices[palm_index]
    pe = float(np.linalg.norm(current[:3, 3] - np.asarray(goal)[:3, 3]))
    relative = np.asarray(goal)[:3, :3] @ current[:3, :3].T
    re = float(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    fe = float(np.max(abs(np.asarray(q)[frozen] - np.asarray(base)[frozen])))
    row = {'native_FK_checked_links': required_link_count,
           'task_palm_target_position_error_m': pe,
           'task_palm_target_rotation_error_rad': re,
           'frozen_joint_max_error_rad': fe,
           'new_quality_position_tolerance_m': 5e-5,
           'new_quality_rotation_tolerance_rad': 2e-4,
           'passed': bool(pe <= 5e-5 and re <= 2e-4 and fe < 1e-8)}
    return row

def _require_clutter_coverage(task, audit):
    """Current native clutter may not disappear from the planning collision world."""
    items = list(getattr(task, '_rerecord_clutter_entities', []))
    actors = [getattr(item, 'actor', item) for item in items]
    expected = sorted(id(actor) for actor in actors)
    receipt = audit.metadata.get('randomized_environment_coverage', {})
    actual = sorted(receipt.get('covered_clutter_entity_ids', []))
    clean_draw = getattr(task, '_rerecord_clutter_sampling_outcome', None) == 'native_clean_background_draw'
    if getattr(task, 'cluttered_table', False) and not expected and not clean_draw:
        raise RuntimeError('Randomized collector lost actual clutter actor references')
    if expected and (receipt.get('expected_clutter_count') != len(expected)
            or receipt.get('covered_clutter_count') != len(expected)
            or receipt.get('all_collidable_clutter_covered') is not True
            or receipt.get('passed') is not True):
        raise RuntimeError('Randomized clutter is missing from current full native planning/FMA obstacle coverage')
    if actual and actual != expected:
        raise RuntimeError('Clutter receipt actor identity differs from actual randomized scene')
    categories = list(receipt.get('static_categories', []))
    if expected and (len(categories) != len(expected) or any(c not in audit.static or c not in audit.static_boxes for c in categories)):
        raise RuntimeError('Randomized obstacle receipt does not cover actual independent collision categories')
    return {'required_current_clutter_count': len(expected),
            'matched_actor_identity_count': len(actual), 'coverage_passed': not expected or receipt.get('passed') is True,
            'native_random_clean_background_draw': bool(clean_draw),
            'native_geometry_coverage_receipt': copy.deepcopy(receipt)}

def _fk_xyz(path, base, indices, pk, palm_index, root):
    xyz = []
    for arm in np.asarray(path):
        full = np.asarray(base).copy(); full[indices] = arm
        pk.compute_forward_kinematics(full)
        pose = pk.get_link_pose(palm_index).to_transformation_matrix()
        xyz.append((np.asarray(root) @ pose)[:3, 3])
    return np.asarray(xyz)

def _quality_path(task, path, base, indices, chosen, limits, pk, palm_index, root, *, times=None, velocity_limits=None, cumulative=False):
    p = np.asarray(path, dtype=float)
    xyz = _fk_xyz(p, base, indices, pk, palm_index, root)
    row = path_quality(p, base[indices], chosen[indices], policy=POLICY,
        hardware_limits=limits[indices], EE_positions_world_m=xyz,
        times_s=times, velocity_limits_rad_s=velocity_limits)
    row['EE_frame'] = 'World origin of native inspire_base_link; same exact FK frame at every path row'
    if cumulative:
        history = getattr(task, '_rerecord_precontact_plans', None)
        if history:
            all_p = np.vstack([history['position'], p]); all_xyz = np.vstack([history['xyz'], xyz])
            combined = path_quality(all_p, all_p[0], chosen[indices], policy=POLICY,
                                    hardware_limits=limits[indices], EE_positions_world_m=all_xyz)
            combined['inter_stage_current_actual_start_included_without_wrapping'] = True
            row['cumulative_precontact_command_quality'] = combined
            if not combined['passed']:
                row['passed'] = False
                row['rejection_reasons'] += ['Combined approach stages: ' + s for s in combined['rejection_reasons']]
    return row, xyz

def _record_plan(task, side, path, xyz):
    history = getattr(task, '_rerecord_precontact_plans', None)
    if history and history['side'] != side:
        raise RuntimeError('Precontact active arm changed')
    p = np.asarray(path, dtype=float).copy(); e = np.asarray(xyz, dtype=float).copy()
    if history:
        p = np.vstack([history['position'], p]); e = np.vstack([history['xyz'], e])
    task._rerecord_precontact_plans = {'side': side, 'position': p, 'xyz': e}

def _replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise RuntimeError('Exactly one reviewed source site required: ' + label)
    return source.replace(old, new, 1)

def derive_planner_source(source, module_name):
    """Local derived function, with reviewed exact source sites and no repo writes."""
    source = _replace_once(source, "    before = _physical_snapshot(task)",
        "    before = _physical_snapshot(task)", 'snapshot preserved')
    source = _replace_once(source, "    model = audit.model", "    clutter_coverage = _require_clutter_coverage(task, audit)\n    model = audit.model", 'clutter coverage')
    source = _replace_once(source, "                'endpoint_attempts': [], 'RRT_attempts': []}",
        "                'endpoint_attempts': [], 'RRT_attempts': [],\n                'motion_quality_overlay': _OVERLAY_RECEIPT,\n                'randomized_obstacle_coverage': clutter_coverage}", 'overlay identity')
    source = _replace_once(source, "            declared = family['goals']", "            declared, near_selection = choose_nearest_family(family, base, indices)", 'singleton near family')
    source = _replace_once(source, "            metadata['RRT_attempts'].append(row)\n            reset(base)",
        "            row['near_numeric_goal_selection'] = near_selection\n            row['chosen_goal_target_FK_proof'] = _strict_palm_goal(declared[0], base, indices, native_fk, goal, palm_index, len(link_names))\n            metadata['RRT_attempts'].append(row)\n            if not row['chosen_goal_target_FK_proof']['passed']:\n                row['rejection_reason'] = 'Chosen near numeric goal differs from exact native task target'\n                continue\n            reset(base)", 'chosen goal target proof')
    source = _replace_once(source, "            position, velocity, times, retime = official_topp_polyline(task, path, side, time_step=time_step)",
        "            raw_quality, _ = _quality_path(task, path, base, indices, declared[0], limits, pk, palm_index, root, cumulative=True)\n            row['motion_quality_raw'] = raw_quality\n            if not raw_quality['passed']:\n                row['rejection_reason'] = 'New demonstration raw path quality predicate'\n                continue\n            position, velocity, times, retime = official_topp_polyline(task, path, side, time_step=time_step)\n            velocity_limits = np.asarray(retime['official_joint_velocity_limits_rad_s'], dtype=float)\n            physical_times = np.arange(len(position)) * float(task.scene.get_timestep())\n            dense_quality, dense_xyz = _quality_path(task, position, base, indices, declared[0], limits, pk, palm_index, root, times=physical_times, velocity_limits=velocity_limits, cumulative=True)\n            row['motion_quality_retimed'] = dense_quality\n            row['retimed_numeric_endpoint_match'] = equivalent.match_declared_numeric_endpoint(position, base[indices], numeric_goals)\n            row['audited_vertices_numeric_endpoint_match'] = equivalent.match_declared_numeric_endpoint(np.asarray(retime['audit_vertices']), base[indices], numeric_goals)\n            if not (dense_quality['passed'] and row['retimed_numeric_endpoint_match']['passed'] and row['audited_vertices_numeric_endpoint_match']['passed']):\n                row['rejection_reason'] = 'New demonstration dense/cumulative quality or specifically chosen TOPP endpoint predicate'\n                continue", 'raw and dense quality')
    source = _replace_once(source, "            return {'status': 'Success', 'position': position, 'velocity': velocity,",
        "            _record_plan(task, side, position, dense_xyz)\n            return {'status': 'Success', 'position': position, 'velocity': velocity,", 'accepted plan history')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plan_free_approach']
    if len(functions) != 1:
        raise RuntimeError('Expected one original free approach function')
    receipt = {'source_module': module_name,
               'nearest_single_numeric_goal_before_RRT': True,
               'chosen_raw_retimed_audited_endpoint_identity_required': True,
               'original_full_native_IK_FK_collision_declarations_retained': True,
               'original_official_TOPP_call_retained': True,
               'original_full_mesh_and_dense_refinement_calls_retained': True,
               'original_auxiliary_and_physical_state_finally_checks_retained': True,
               'paths_never_wrapped_clipped_or_modified': True,
               'quality_policy': asdict(POLICY),
               'CPU_overlay_is_not_physical_demonstration_success': True}
    return source, functions[0], receipt

def _actual_pose(session):
    palm = ('l_' if session.side == 'left' else 'r_') + 'inspire_base_link'
    return np.asarray(session.audit.links[palm].entity.pose.to_transformation_matrix())[:3, 3].copy()

def _actual_q(session):
    robot = session.task.robot
    q = np.asarray(robot.left_entity.get_qpos(), dtype=float)
    return q[[robot._joint_index[name] for name in getattr(robot, session.side + '_arm_joints_name')]].copy()

def _init_actual(session):
    q = _actual_q(session); xyz = _actual_pose(session)
    vmax = np.asarray(getattr(session.task.robot, session.side + '_mplib_planner').planner.joint_vel_limits)
    names = session.audit.sim_joints
    indices = [names.index(n) for n in getattr(session.task.robot, session.side + '_arm_joints_name')]
    limits = np.concatenate([np.asarray(j.limits).reshape(-1, 2) for j in session.audit.art.active_joints])[indices]
    session._rerecord_actual_motion = {'q': [q], 'xyz': [xyz], 'time': [float(session.steps * session.task.scene.get_timestep())],
        'velocity_limits': vmax.copy(), 'hardware_limits': limits.copy(), 'finalized': False,
        'min_q': q.copy(), 'max_q': q.copy(), 'total_variation_rad': np.zeros_like(q)}
    return session._rerecord_actual_motion

def _finish_actual(session):
    state = getattr(session, '_rerecord_actual_motion', None)
    if not state or state['finalized']:
        return
    p = np.asarray(state['q']); xyz = np.asarray(state['xyz']); t = np.asarray(state['time'])
    if len(p) < 2:
        raise RuntimeError('New actual precontact quality has no complete measured trajectory')
    receipt = path_quality(p, p[0], p[-1], policy=POLICY, hardware_limits=state['hardware_limits'],
                          EE_positions_world_m=xyz, times_s=t, velocity_limits_rad_s=state['velocity_limits'])
    receipt['measurement_source'] = 'Current native SAPIEN actual arm qpos and inspire_base_link world pose, each original physical step'
    receipt['includes_all_precontact_approach_stages'] = True
    commanded = getattr(session.task, '_rerecord_precontact_plans', None)
    if commanded:
        receipt['actual_last_vs_chosen_command_goal_delta_per_joint_rad'] = (p[-1] - commanded['position'][-1]).tolist()
        receipt['actual_last_vs_chosen_command_goal_max_error_rad'] = float(np.max(abs(p[-1] - commanded['position'][-1])))
    receipt['raw_RRT_TopP_exact_goal_tolerance_not_applied_to_physical_servo_tracking'] = True
    state['finalized'] = True
    session.report['motion_quality_actual_precontact'] = receipt
    if not receipt['passed']:
        raise RuntimeError('New actual precontact demonstration quality failed: ' + '; '.join(receipt['rejection_reasons']))

def install_planning_quality(*, source_package, legacy_directory):
    """Install task-process overlays before importing the pinned original worker.

    Environment/scene/randomized-clutter coverage is installed independently by
    the scene addon. This installer never claims to bind scene restoration.
    """
    source_package = Path(source_package).resolve(); legacy_directory = Path(legacy_directory).resolve()
    if not source_package.is_dir() or not legacy_directory.is_dir():
        raise FileNotFoundError('Mirrored original collector directories required')
    for p in (DIRECTORY, source_package, legacy_directory):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    receipt = {'installer': str(Path(__file__).resolve()), 'installer_sha256': _sha(__file__),
        'quality_library_sha256': _sha(DIRECTORY / 'near_branch_quality.py'), 'modules': {},
        'quality_policy': asdict(POLICY), 'original_source_files_modified': False,
        'future_original_commands_used': False, 'original_physical_success_not_replaced': True}
    for module_name, digest in SOURCE_PINS.items():
        file = source_package / (module_name + '.py')
        if _sha(file) != digest:
            raise RuntimeError('Exact original planner source pin differs: ' + module_name)
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != file:
            raise RuntimeError('Wrong module source selected: ' + module_name)
        generated, function, details = derive_planner_source(file.read_text(), module_name)
        generated_file = DIRECTORY / (module_name + '_near_quality_generated.py')
        generated_file.write_text(generated)
        details.update(original_source_path=str(file), original_source_sha256=digest,
                       generated_source_path=str(generated_file), generated_source_sha256=_sha(generated_file))
        namespace = module.__dict__.copy()
        namespace.update({name: value for name, value in globals().items() if name.startswith('_') or name == 'choose_nearest_family'})
        namespace['_OVERLAY_RECEIPT'] = details
        tree = ast.Module(body=[function], type_ignores=[]); ast.fix_missing_locations(tree)
        exec(compile(tree, str(generated_file), 'exec'), namespace)
        module.plan_free_approach = namespace['plan_free_approach']
        module.plan_free_approach.__rerecord_derivation_ast__ = copy.deepcopy(tree)
        receipt['modules'][module_name] = details
    base = importlib.import_module('expert_generation').ExpertSession
    if getattr(base, '_rerecord_quality_installed', False):
        raise RuntimeError('Quality installer must run once per original worker process')
    original_init, original_step, original_write = base.__init__, base.step, base.write
    def init(session, *args, **kwargs):
        original_init(session, *args, **kwargs)
        session.task._rerecord_precontact_plans = None
        session._rerecord_actual_motion = None
        for p in [Path(__file__).resolve(), DIRECTORY / 'near_branch_quality.py'] + [Path(v['generated_source_path']) for v in receipt['modules'].values()]:
            session.sources[str(p)] = _sha(p)
        session.report['motion_quality_overlay'] = copy.deepcopy(receipt)
        session.report['source_hashes'] = session.sources
    def step(session):
        approach = session.stage.get('name') == 'approach'
        if not approach:
            _finish_actual(session)
        elif getattr(session, '_rerecord_actual_motion', None) is None:
            _init_actual(session)
        result = original_step(session)
        if approach:
            state = session._rerecord_actual_motion
            measured = _actual_q(session)
            state['total_variation_rad'] += abs(measured - state['q'][-1])
            state['min_q'] = np.minimum(state['min_q'], measured); state['max_q'] = np.maximum(state['max_q'], measured)
            state['q'].append(measured); state['xyz'].append(_actual_pose(session))
            state['time'].append(float(session.steps * session.task.scene.get_timestep()))
            # Independent absolute gates are valid at every prefix. Detour
            # ratios are measured after all approach stages, before close.
            span = state['max_q'] - state['min_q']
            if span[-1] > POLICY.max_J6_span_rad + 1e-9 or state['total_variation_rad'][-1] > POLICY.max_J6_total_variation_rad + 1e-9 or span.max() > POLICY.max_each_joint_span_rad + 1e-9:
                session.report['motion_quality_actual_prefix_rejected'] = {'steps': len(state['q'])-1,
                    'J6_range_rad': float(span[-1]), 'J6_total_variation_rad': float(state['total_variation_rad'][-1]),
                    'each_joint_span_rad': span.tolist(),
                    'quality_policy': asdict(POLICY)}
                raise RuntimeError('New actual precontact J6 travel/range quality exceeded')
        return result
    def write(session):
        if hasattr(session, '_rerecord_actual_motion') and session._rerecord_actual_motion:
            state = session._rerecord_actual_motion
            session.report['motion_quality_actual_trace_rows_so_far'] = len(state['q'])
        return original_write(session)
    base.__init__, base.step, base.write = init, step, write
    base._rerecord_quality_installed = True
    receipt['actual_precontact_measured_quality_before_close_enabled'] = True
    receipt['independent_quality_gates_not_hardware_safety_replacements'] = True
    (DIRECTORY / 'installed_quality_overlay_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
