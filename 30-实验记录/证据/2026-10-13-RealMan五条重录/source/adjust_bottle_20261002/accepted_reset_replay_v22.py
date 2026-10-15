"""Reviewable accepted-only reset/replay command for the already-warm worker.

Execute with the worker's accepted v22 globals ONLY after a strict candidate is accepted.
Set accepted_replay_version='v22' explicitly; no version default is allowed.
This file does not launch a worker, queue commands, install hooks, or modify any
canonical source/model. It preserves the installed coupling initialization hook,
creates fresh feedback/guards, and reuses the native saved arm paths/exporter.
24D command ordering: L arm6 rad, L primary SDK6, R arm6 rad, R primary SDK6.
SDK6 is model-derived and uncalibrated; raw physical hand24 rad is also recorded.
"""
from pathlib import Path
import ast
import copy
import datetime
import hashlib
import json
import traceback
from types import MethodType, CodeType


def _accepted_gate(context):
    """Reject failure/stale acceptance before any IO, reset or exporter call."""
    if context.get('accepted') is not True:
        raise RuntimeError('Accepted replay requires current accepted is True')
    if context.get('accepted_replay_version') != 'v22':
        raise RuntimeError('Set accepted_replay_version=v22 explicitly; other versions require review')
    source_report = context.get('report', {})
    if source_report.get('runtime_sources_unchanged') is not True:
        raise RuntimeError('V22 candidate runtime-source begin/end audit missing or failed')
    if source_report.get('accepted') is not True or source_report.get('result') != 'physics_success':
        raise RuntimeError('Current report is not a strict physics_success candidate')
    if source_report.get('canonical_sources_unchanged') is not True:
        raise RuntimeError('Candidate canonical-source audit is missing or failed')
    required = ('task', 'args', 'BASE', 'OUT', 'protected', 'sha',
                'accepted_grasp_setting', 'accepted_seed', 'RealManRobot', 'Base_Task')
    missing = [name for name in required if name not in context]
    if missing:
        raise RuntimeError('Missing accepted worker globals: ' + ', '.join(missing))
    task = context['task']
    if getattr(task.robot, 'communication_flag', False):
        raise RuntimeError('Replay is simulation-only; communication_flag must be False')
    legal, excess = task.planned_joints_legal()
    if not task.plan_success or not task.check_success() or not legal:
        raise RuntimeError('Accepted candidate state/planned-joint legality no longer passes')
    return source_report


def _validate_runtime_sources(report, base, sha):
    """Candidate begin == candidate end == current explicit V22 source bytes."""
    base=Path(base).resolve();initial=report.get('runtime_source_hashes');final=report.get('runtime_source_hashes_final')
    if not isinstance(initial,dict) or not initial or not isinstance(final,dict):
        raise RuntimeError('Missing initial/final runtime source hash maps')
    if initial!=final or report.get('runtime_sources_unchanged') is not True:
        raise RuntimeError('Candidate runtime files changed during its execution')
    required={str(base/n) for n in ('try_coupled_side_entry_v22.py','contact_capture_control_v22.py','full_native_carry_planner_v21.py','support_departure_certificate_v20.py')}
    if not required.issubset(initial):raise RuntimeError('Explicit V22 candidate/controller/planner/support-certificate hashes required')
    actual={}
    for file,expected in initial.items():
        path=Path(file)
        if not path.is_absolute() or str(path.resolve())!=file or path.parent.resolve()!=base:
            raise RuntimeError('Runtime hash must use absolute package-local path: '+str(file))
        if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise RuntimeError('Invalid runtime SHA256: '+file)
        actual[file]=sha(path)
    if actual!=initial:raise RuntimeError('Current source differs from accepted V22 begin/end hashes')
    return actual


def _closing_window_gate(samples,feedback):
    """Only read LAST50 existing close samples; no controller/physics advance."""
    import numpy as np
    samples=samples[-50:]
    if len(samples)!=50:raise RuntimeError('Missing last50 fresh closing samples')
    ref=np.asarray(samples[0]['relative_bottle_palm'],dtype=float)
    if ref.shape!=(4,4) or not np.isfinite(ref).all():raise RuntimeError('Invalid closing-window reference')
    displacement=0.;rotation=0.;minimum={n:float('inf') for n in ('middle','index','thumb')};paired=True
    for sample in samples:
        matrix=np.asarray(sample['relative_bottle_palm'],dtype=float)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise RuntimeError('Invalid closing-window relative pose')
        displacement=max(displacement,float(np.linalg.norm(matrix[:3,3]-ref[:3,3])))
        rotation=max(rotation,float(np.degrees(np.arccos(np.clip((np.trace(ref[:3,:3].T@matrix[:3,:3])-1)/2,-1,1)))))
        forces={n:0. for n in minimum}
        for contact in sample['contacts']:
            value=float(contact['normal_force_N'])
            if not np.isfinite(value) or value<0:raise RuntimeError('Invalid actual contact normal force')
            for n in forces:
                if n in contact['link']:forces[n]+=value
        for n,value in forces.items():minimum[n]=min(minimum[n],value)
        paired=paired and all(v>.05 for v in forces.values()) and sample['thumb_normal_N']>.05 and sample['fingers_normal_N']>.05 and sample['opposing_normals_dot']<-.3
    allocation=feedback['lp_history'][-1]['allocation'] if feedback['lp_history'] else None
    usable=allocation is not None and allocation.get('usable') is True
    passed=bool(paired and feedback['actual_pair_streak']>=50 and displacement<.002 and rotation<2 and usable and feedback['lp_failure_steps']==0 and feedback['last_lp_phase']=='close')
    return {'passed':passed,'validation_window_existing_steps':50,'validation_physics_steps_added':0,
        'actual_pair_streak_steps':feedback['actual_pair_streak'],'relative_drift_m':displacement,'relative_rotation_deg':rotation,
        'minimum_main3_normal_N_over_window':minimum,'all_50_samples_main3_paired':bool(paired),
        'last_allocation_usable':usable,'lp_failure_steps':feedback['lp_failure_steps'],'last_allocation':allocation,'last_lp_phase':feedback['last_lp_phase']}


def _extract_function_defs(path, names, context):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    selected = [copy.deepcopy(n) for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in set(names)]
    if {node.name for node in selected} != set(names):
        raise RuntimeError('Expected source helper functions missing: ' + str(path))
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path) + ':selected_helpers', 'exec'), context)


def _extract_native_exporter(path, context):
    """Reuse the original collector's actual export statements via AST."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    blocks = [node for node in tree.body if isinstance(node, ast.Try)]
    if len(blocks) != 1:
        raise RuntimeError('Collector top-level try block changed; review extraction')
    body = blocks[0].body
    starts = [i for i, node in enumerate(body) if isinstance(node, ast.ImportFrom)
              and node.module == 'scripts.collect_data'
              and any(alias.name == 'build_episode_instructions' for alias in node.names)]
    if len(starts) != 1:
        raise RuntimeError('Native exporter AST anchor changed; review extraction')
    selected = copy.deepcopy(body[starts[0]:])
    selected.append(ast.Return(value=ast.Subscript(
        value=ast.Name(id='report', ctx=ast.Load()),
        slice=ast.Constant(value='dataset'), ctx=ast.Load())))
    function = ast.FunctionDef(
        name='_export_native_collector_block',
        args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                           kw_defaults=[], kwarg=None, defaults=[]),
        body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path) + ':native_export_block', 'exec'), context)
    return ast.unparse(module)


def _function_code_signature(function):
    """Compare live/source control code without filename/line-table differences."""
    def describe(code):
        constants = tuple(describe(c) if isinstance(c, CodeType) else c for c in code.co_consts)
        return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
                code.co_cellvars, code.co_argcount, code.co_posonlyargcount,
                code.co_kwonlyargcount, constants)
    return hashlib.sha256(repr(describe(function.__code__)).encode('utf-8')).hexdigest()


def _json_default(value):
    return value.tolist() if hasattr(value, 'tolist') else str(value)


def _validate_lift_command_polyline(planning,native,support):
    """Bind V22 first-lift proof to exact RETURNED targets, not Cartesian IK knots."""
    import numpy as np
    positions=np.asarray(native['position'],dtype=float)
    retime=planning.get('retiming',{})
    vertices=np.asarray(retime.get('audit_vertices'),dtype=float)
    if planning.get('physical_state_unchanged') is not True or not np.array_equal(vertices,positions):
        raise RuntimeError('First lift audit vertices differ from exact saved native command positions')
    digest=hashlib.sha256(positions.tobytes()).hexdigest()
    if retime.get('returned_commands_sha256')!=digest or support.get('raw_command_path_sha256')!=digest:
        raise RuntimeError('First lift TOPP/signed-support proof hashes differ from exact saved command targets')
    if (retime.get('input_knot_count')!=21 or retime.get('dense_rows')!=len(positions)
            or retime.get('internal_knot_rest_stops_forced') is not False or retime.get('additional_hold_seconds')!=0.):
        raise RuntimeError('First lift must use one 21-knot TOPP call without internal stops/additional hold')
    expected_scope='Fresh audit of exact returned command-target joint polyline; does NOT certify internal TOPP continuous cubic or physical tracking'
    if retime.get('command_curve_scope')!=expected_scope:
        raise RuntimeError('First lift command-curve audit scope is missing or changed')
    return {'returned_commands_sha256':digest,'audited_command_rows':len(positions),'scope':expected_scope}


def _validate_contact_lp_options(setting,report_setting):
    """Bind V22 force reserve to the accepted candidate setting, never a default."""
    import math
    value=setting.get('lp_min_main_force_N')
    recorded=report_setting.get('lp_min_main_force_N')
    if (isinstance(value,bool) or isinstance(recorded,bool) or not isinstance(value,(float,int))
            or not isinstance(recorded,(float,int)) or not math.isfinite(value) or not math.isfinite(recorded)
            or value!=1.0 or recorded!=value):
        raise RuntimeError('V22 accepted worker/report force reserve must explicitly match min_main_force_N=1.0')
    return {'min_main_force_N':float(value)}


def _load_candidate_wrenches(source_out, saved_paths, sha, segment_references, *, lp_options):
    """Verify native saved paths against original candidate carry/wrench artifacts.

    Official adjust_bottle left-arm ordering: approach2, then retreat/lift/place3.
    Candidate planner filenames use the PREVIOUS dense_call (3/4/5); action uses
    dense_call4/5/6 (no additional initial hold). This explicit mapping is checked against native consumo index.
    No static mg fallback is permitted during a scheduled carry action.
    """
    import numpy as np
    left=saved_paths['left_joint_path'];right=saved_paths['right_joint_path']
    if len(left)!=5 or len(right)!=0:
        raise RuntimeError('v22 replay expects official left-arm approach2 plus retreat/lift/place3, no right paths')
    schedules={};manifest=[]
    for path_index in (2,3,4):
        planner_dense=path_index+1;action_dense=planner_dense+1
        path_file=source_out/f'planned_carry_path_{planner_dense}.npz'
        wrench_file=source_out/f'planned_wrench_{planner_dense}.npz'
        mesh_file=source_out/f'carry_mesh_path_{planner_dense}.json'
        preflight_file=source_out/f'carry_wrench_preflight_{planner_dense}.json'
        planning_file=source_out/f'carry_planning_{planner_dense}.json'
        files=(path_file,wrench_file,mesh_file,preflight_file,planning_file)
        if not all(p.is_file() for p in files):
            raise RuntimeError('Accepted v22 carry evidence missing: '+str(planner_dense))
        file_hashes={str(p):sha(p) for p in files}
        native=left[path_index]
        with np.load(path_file,allow_pickle=False) as planned:
            planning_reference=np.asarray(planned['planning_reference'],dtype=float).copy()
            if planning_reference.shape!=(4,4) or not np.isfinite(planning_reference).all() or not np.allclose(planning_reference[3],[0,0,0,1],atol=1e-8):
                raise RuntimeError('Invalid saved per-segment planning reference')
            if not np.allclose(planning_reference[:3,:3].T@planning_reference[:3,:3],np.eye(3),atol=2e-5):
                raise RuntimeError('Saved per-segment planning reference rotation is invalid')
            if str(planner_dense) not in segment_references or not np.array_equal(planning_reference,np.asarray(segment_references[str(planner_dense)],dtype=float)):
                raise RuntimeError('Saved NPZ per-segment reference differs from accepted source report')
            for key in ('position','velocity'):
                if not np.array_equal(np.asarray(planned[key]),np.asarray(native[key])):
                    raise RuntimeError('Original carry path differs from native saved trajectory: '+str(path_index))
        mesh=json.loads(mesh_file.read_text());preflight=json.loads(preflight_file.read_text())
        if preflight.get('lp_options')!=lp_options:
            raise RuntimeError('Accepted carry preflight LP options differ from exact V22 force reserve setting')
        if not np.array_equal(planning_reference,np.asarray(mesh.get('metadata',{}).get('rigid_carry_reference'),dtype=float)):
            raise RuntimeError('Per-segment reference differs between NPZ, continuous mesh audit and source report')
        if mesh.get('passed') is not True or preflight.get('passed') is not True:
            raise RuntimeError('Candidate carry mesh/inertial wrench preflight was not accepted')
        if planner_dense==3:
            support=mesh.get('support_departure_combined_audit',{})
            signed=mesh.get('support_departure_certificate',{})
            guard=mesh.get('physical_signed_height_guard',{})
            if (support.get('passed') is not True or support.get('other_full_mesh_categories_passed') is not True
                    or support.get('physical_state_unchanged') is not True or signed.get('passed') is not True
                    or guard.get('physical_table_collisions_enabled') is not True):
                raise RuntimeError('First carry lacks accepted signed support departure/full other meshes/unchanged physical collision proof')
            _validate_lift_command_polyline(json.loads(planning_file.read_text()),native,support)
        schedule={}
        with np.load(wrench_file,allow_pickle=False) as saved:
            if set(saved.files)!={'force_target_world_N','torque_target_world_Nm'}:
                raise RuntimeError('Unexpected candidate wrench NPZ schema')
            for key in ('force_target_world_N','torque_target_world_Nm'):
                values=np.asarray(saved[key],dtype=float).copy()
                if values.shape!=(len(native['position']),3) or not np.isfinite(values).all():
                    raise RuntimeError('Candidate wrench length/shape/finite mismatch: '+str(wrench_file))
                schedule[key]=values
        if any(sha(p)!=file_hashes[str(p)] for p in files):
            raise RuntimeError('Accepted source carry/wrench files changed during read')
        schedules[action_dense]={'path_index':path_index,'planner_dense_call':planner_dense,
                                 'action_dense_call':action_dense,'schedule':schedule,
                                 'native_position':np.asarray(native['position']).copy(),
                                 'native_velocity':np.asarray(native['velocity']).copy(),
                                 'source_hashes':file_hashes,'planning_reference':planning_reference}
        manifest.append({'saved_left_path_index':path_index,'planner_dense_call':planner_dense,
                         'action_dense_call':action_dense,'schedule_rows':len(native['position']),
                         'source_hashes':file_hashes,'planning_reference':planning_reference.tolist(),
                         'reference_source':'fresh actual grip_measure at each accepted planner call; NPZ == mesh metadata == source report'})
    return schedules,manifest


def _install_saved_wrench_for_action(feedback, sequence, *, dense_call,
                                    native_left_consumed, schedules, actual_relative=None):
    """Select exact candidate schedule before the first action physics step."""
    import numpy as np
    if dense_call not in schedules:
        raise RuntimeError('Carry action has no saved candidate inertial wrench: '+str(dense_call))
    entry=schedules[dense_call]
    if native_left_consumed!=entry['path_index']+1:
        raise RuntimeError('Saved carry consumption index does not match candidate wrench')
    arm=sequence.get('left_arm')
    if arm is None:raise RuntimeError('Scheduled candidate carry action has no left arm path')
    for key,target in (('position','native_position'),('velocity','native_velocity')):
        if not np.array_equal(np.asarray(arm[key]),entry[target]):
            raise RuntimeError('Actual replay control sequence differs from accepted saved carry path')
    from observed_trace_audit_v13 import _matrix,_angle_deg
    if actual_relative is None:raise RuntimeError('Replay segment requires current actual grasp-relative measurement')
    current=_matrix(actual_relative);reference=entry['planning_reference']
    position_error=float(np.linalg.norm(current[:3,3]-reference[:3,3]));rotation_error=_angle_deg(current,reference)
    if position_error>.002 or rotation_error>2:
        raise RuntimeError('Actual replay segment grip differs from accepted planning reference by >2mm/2deg')
    feedback['wrench_schedule']={key:array.copy() for key,array in entry['schedule'].items()}
    feedback['wrench_index']=0
    feedback['last_lp_phase']=None
    metadata={key:value for key,value in entry.items() if key not in ('schedule','native_position','native_velocity')}
    metadata.update(replay_actual_segment_start_relative=current.tolist(),reference_error_m=position_error,reference_error_deg=rotation_error)
    return metadata

def _support_monitor_summary(monitor):
    """Small actual physical guard summary, never repeat native vertices/bounds."""
    return {key:value for key,value in monitor.items() if key not in ('vertices','table_bounds')}


def _run_accepted_reset_replay(worker):
    source_report = _accepted_gate(worker)
    # All simulation classes/objects come from the current already-warm worker.
    import numpy as np
    import h5py
    import pickle
    task = worker['task']
    base = Path(worker['BASE']).resolve()
    source_out = Path(worker['OUT']).resolve()
    runtime_initial=_validate_runtime_sources(source_report,base,worker['sha'])
    trajectory = source_out / '_traj_data' / 'episode0.pkl'
    if Path(task.save_dir).resolve() != source_out or not trajectory.is_file():
        raise RuntimeError('Accepted OUT saved trajectory is missing or differs from task.save_dir')
    if not all(worker['sha'](p) == source_report['source_hashes'][str(p)] for p in worker['protected']):
        raise RuntimeError('Canonical sources drifted since accepted candidate')
    # Call the native loader before changing save_dir; preserve the original file.
    saved_paths = copy.deepcopy(task.load_tran_data(0))
    for side in ('left', 'right'):
        for entry in saved_paths[side + '_joint_path']:
            q = np.asarray(entry['position'], dtype=float)
            v = np.asarray(entry['velocity'], dtype=float)
            if entry.get('status') != 'Success' or q.ndim != 2 or q.shape[1] != 6 or v.shape != q.shape:
                raise RuntimeError('Saved path is not a complete successful native arm6 trajectory')
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise RuntimeError('Saved path has nonfinite commands')
    expected_lp_options=_validate_contact_lp_options(worker['accepted_grasp_setting'],source_report.get('grasp_setting',{}))
    candidate_wrenches, wrench_manifest = _load_candidate_wrenches(source_out, saved_paths, worker['sha'],source_report.get('segment_planning_references',{}),lp_options=expected_lp_options)
    candidate_before = copy.deepcopy(source_report['bottle_attempts'][-1]['before'])
    label = worker.get('accepted_replay_label') or ('accepted_replay_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if Path(label).name != label:
        raise RuntimeError('Replay label must be a single directory name')
    replay_out = source_out / label
    replay_out.mkdir(exist_ok=False)
    (replay_out / 'accepted_candidate_report.json').write_text(json.dumps(source_report, indent=2, default=_json_default))
    setting = copy.deepcopy(worker['accepted_grasp_setting'])
    replay_args = copy.deepcopy(worker['args'])
    replay_args.update(seed=int(worker['accepted_seed']), need_plan=False, save_data=True,
                       save_freq=15, now_ep_num=0, save_path=str(replay_out))
    for side in ('left', 'right'):
        replay_args[side + '_embodiment_config'].update(
            hand_synergy={'open': setting['open_sdk'], 'close': setting['close_sdk']},
            hand_joint_stiffness=5, hand_joint_damping=.1)
    replay_args.update(copy.deepcopy(saved_paths))
    # Functions must resolve fresh globals. A shallow function reuse would retain
    # the previous v9 stage/feedback closure and is intentionally avoided.
    context = dict(worker)
    context.update(OUT=replay_out, args=replay_args, grasp_setting=setting,
                   strict_hold=[False], hold_samples=[], pre_retreat_samples=[], support_monitor={},
                   carry_reference=[None], carry_audit={'max_translation_drift_m': 0., 'max_rotation_drift_deg': 0.},
                   steps=[0], dense_calls=[0], traces=[],
                   stage={'name': 'ready', 'gripper': False},
                   report={'result': 'accepted_replay_running', 'accepted': False,
                           'task': 'adjust_bottle', 'seed': int(worker['accepted_seed']),
                           'source_candidate_out': str(source_out),
                           'source_trajectory': str(trajectory),
                           'source_trajectory_sha256': worker['sha'](trajectory),
                           'source_hashes': copy.deepcopy(source_report['source_hashes']),
                           'runtime_source_hashes': copy.deepcopy(runtime_initial),
                           'runtime_source_hashes_final': copy.deepcopy(runtime_initial),
                           'runtime_sources_unchanged': True,
                           'candidate_runtime_source_hashes': copy.deepcopy(source_report['runtime_source_hashes']),
                           'candidate_runtime_source_hashes_final': copy.deepcopy(source_report['runtime_source_hashes_final']),
                           'grasp_setting': setting, 'hardware_io': False, 'attempts': [],
                           'planner_calls_allowed': False,
                           'candidate_wrench_manifest': wrench_manifest,
                           'contact_lp_options': copy.deepcopy(expected_lp_options),
                           'replay_wrench_actions': [],
                           'sdk_mapping_calibrated': False})
    collector = base / 'run_adjust_bottle.py'
    version = worker['accepted_replay_version']
    candidate = (base / f'try_coupled_side_entry_{version}.py').resolve()
    # Explicit V22 source bytes are checked against its accepted begin/end maps.
    capture=(base/'contact_capture_control_v22.py').resolve()
    if _validate_runtime_sources(source_report,base,worker['sha'])!=runtime_initial:
        raise RuntimeError('V22 runtime source changed during replay preparation')
    context['report']['accepted_replay_version'] = version
    if candidate.parent != base or capture.parent != base:
        raise RuntimeError('Replay source files must remain package-local')
    _extract_function_defs(collector, {'write', 'snapshot', 'bottle_state', 'hold_current_targets'}, context)
    _extract_function_defs(candidate, {'side_entry_grasp', 'realman_place_pose', 'realman_lift_displacement', 'grip_measure', 'strict_official_episode', 'configure_support_departure', 'check_support_departure'}, context)
    # Capture source only defines imports/function; this does not step or plan.
    exec(compile(capture.read_text(), str(capture), 'exec'), context)
    exporter_ast = _extract_native_exporter(collector, context)
    (replay_out / 'native_exporter_extracted.py').write_text(exporter_ast)
    context['report']['replay_helper_source_hashes'] = {
        str(p): worker['sha'](p) for p in (collector, candidate, capture, base / 'accepted_reset_replay_v22.py')}
    coupling_hook = worker['RealManRobot']._initialize_hands
    old_entity = task.robot.left_entity
    telemetry = []
    external_movie = None
    original_step = None
    original_obs = None
    stage_out = replay_out / '_validated_export_staging'
    hand_names = []
    try:
        signatures = {}
        for name in ('side_entry_grasp', 'realman_place_pose', 'realman_lift_displacement', 'grip_measure', 'strict_official_episode', 'install_contact_capture', 'configure_support_departure', 'check_support_departure'):
            if name not in worker or _function_code_signature(worker[name]) != _function_code_signature(context[name]):
                raise RuntimeError('Live accepted controller differs from explicit version source: ' + name)
            signatures[name] = _function_code_signature(context[name])
        context['report']['live_and_source_control_code_signatures'] = signatures
        for name in ('set_gripper', 'set_arm_joints', 'left_plan_grippers', 'right_plan_grippers',
                     'left_plan_path', 'right_plan_path', 'left_plan_multi_path', 'right_plan_multi_path'):
            task.robot.__dict__.pop(name, None)
        for name in ('move', 'take_dense_action', 'choose_grasp_pose', 'get_place_pose', 'move_by_displacement', 'get_obs'):
            task.__dict__.pop(name, None)
        task.close_env()
        task.setup_demo(**replay_args)
        if worker['RealManRobot']._initialize_hands is not coupling_hook:
            raise RuntimeError('Existing coupling class initialization hook changed')
        if task.robot.left_entity is old_entity or getattr(task.robot, 'communication_flag', False):
            raise RuntimeError('Replay did not create a fresh simulation articulation')
        if int(task.qpose_tag) != int(candidate_before['qpose_tag']) or int(task.model_id) != int(candidate_before['model_id']):
            raise RuntimeError('Reset seed did not reproduce the accepted object/arm selection')
        if task.qpose_tag != 0:
            raise RuntimeError('Current contact capture controller supports accepted left-arm seed only')
        if len(task.robot.hand_coupling_audit) != 12:
            raise RuntimeError('Fresh articulation lacks the 12 expected physical mimic tendons')
        task.set_path_lst(replay_args)
        context['report']['reset_audit'] = {
            'fresh_articulation': True, 'coupling_class_hook_preserved': True,
            'tendons': task.robot.hand_coupling_audit,
            'hand_friction': task.robot.hand_friction_audit,
            'hand_material': task.robot.hand_material_audit,
            'initial_qpos': task.robot.left_entity.get_qpos().tolist(),
            'frame_idx_before_replay': int(task.FRAME_IDX)}
        def forbidden_planner(self, *args, **kwargs):
            raise RuntimeError('Accepted replay attempted replanning rather than saved native paths')
        for side in ('left', 'right'):
            for suffix in ('plan_path', 'plan_multi_path'):
                setattr(task.robot, side + '_' + suffix, MethodType(forbidden_planner, task.robot))
        task.choose_grasp_pose = MethodType(context['side_entry_grasp'], task)
        task.get_place_pose = MethodType(context['realman_place_pose'], task)
        task.move_by_displacement = MethodType(context['realman_lift_displacement'], task)
        feedback, contact_update = context['install_contact_capture'](task, setting)
        context.update(feedback=feedback, contact_update=contact_update)
        if feedback.get('lp_options')!=expected_lp_options or feedback.get('lp_min_main_force_N')!=expected_lp_options['min_main_force_N']:
            raise RuntimeError('Fresh V22 controller did not preserve accepted force reserve LP options')
        context['report']['force_feedback_simulation_parameters']=copy.deepcopy(feedback.get('feedback_simulation_parameters',{}))
        task.robot.left_plan_grippers = MethodType(feedback['plan'], task.robot)
        from full_mesh_path_audit import FullMeshPathAuditor
        from observed_full_trace_audit_v13 import collect_actual_self_contacts,collect_actual_bottle_contacts
        native_auditor=FullMeshPathAuditor(task)
        actual_self_allowed=native_auditor.allowed
        actual_robot_link_names=set(native_auditor.links)
        context['report']['full_qpos_joint_names']=list(native_auditor.sim_joints)
        context['report']['actual_self_contact_audit']={'normal_impulse_threshold_Ns':1e-7,
            'allowed_native_pairs':[sorted(pair) for pair in native_auditor.allowed],
            'source':'FullMeshPathAuditor native configured fixed components/exemptions and direct physical adjacent joints',
            'active_nonexempt_contacts':0}
        context['report']['replay_helper_source_hashes'][str(base/'observed_full_trace_audit_v13.py')]=worker['sha'](base/'observed_full_trace_audit_v13.py')
        context['report']['replay_helper_source_hashes'][str(base/'observed_trace_audit_v13.py')]=worker['sha'](base/'observed_trace_audit_v13.py')
        dt = float(task.scene.get_timestep())
        hand_names = [n for side in ('left', 'right') for n in task.robot._manifest['hands'][side]['limits']]
        hand_ids = [task.robot._joint_index[n] for n in hand_names]
        hand_limits = {n: lim for side in ('left', 'right') for n, lim in task.robot._manifest['hands'][side]['limits'].items()}
        stage = context['stage']; steps = context['steps']
        initial_fp = np.asarray(task.bottle.get_functional_point(0)[:3])
        initial_R = task.bottle.get_pose().to_transformation_matrix()[:3, :3]
        dynamic = {'max_actual_fd_rad_s': 0., 'max_actual_fd_source_ratio': 0.,
                   'max_engine_qvel_rad_s': 0., 'max_mimic_residual_rad': 0.,
                   'max_hand_joint_limit_excess_rad': 0.}
        def state24_actual():
            return np.asarray(task.robot.get_state24(measured=True), dtype=float)
        def command24_actual():
            values = []
            for side in ('left', 'right'):
                values.extend(float(j.get_drive_target()[0]) for j in getattr(task.robot, side + '_arm_joints'))
                h = task.robot._hand_maps[side]
                targets = {n: float(task.robot._hand_joints[n].get_drive_target()[0]) for n in h.limits}
                values.extend(h.to_sdk(targets))
            result = np.asarray(values, dtype=float)
            if result.shape != (24,) or not np.isfinite(result).all():
                raise RuntimeError('Actual drive command24 is not finite')
            return result
        from external_realtime_video import ExternalRealtimeVideo
        external_movie=ExternalRealtimeVideo(task,replay_out/'external_realtime_diagnostic.mp4',
            pose_factory=worker['sapien'].Pose,quat_converter=worker['mat2quat'])
        external_movie.capture(0)
        context['report']['replay_helper_source_hashes'][str(base/'external_realtime_video.py')]=worker['sha'](base/'external_realtime_video.py')
        original_step = task.scene.step
        def trace_step():
            q_before = np.asarray(task.robot.left_entity.get_qpos(), dtype=float).copy()
            before24 = state24_actual()
            contact_update()
            applied24 = command24_actual()
            hand_targets = np.array([task.robot._hand_joints[n].get_drive_target()[0] for n in hand_names], dtype=float)
            original_step(); steps[0] += 1
            q_after = np.asarray(task.robot.left_entity.get_qpos(), dtype=float).copy()
            engine_v = np.asarray(task.robot.left_entity.get_qvel(), dtype=float).copy()
            actual_self_contacts=collect_actual_self_contacts(task,actual_robot_link_names,actual_self_allowed)
            bottle_actual_contacts=collect_actual_bottle_contacts(task)
            active_self=[c for c in actual_self_contacts if c['active']]
            context['report']['actual_self_contact_audit']['active_nonexempt_contacts']+=len(active_self)
            fd = (q_after - q_before) / dt
            if not np.isfinite(q_after).all() or not np.isfinite(engine_v).all():
                raise RuntimeError('Nonfinite actual articulation during accepted replay')
            q = task.robot._measured_positions()
            residual = max(abs(q[n] - float(m['multiplier']) * q[m['joint']] - float(m.get('offset', 0)))
                           for h in task.robot._manifest['hands'].values() for n, m in h['mimic'].items())
            excess = max(max(lim['lower'] - q[n], q[n] - lim['upper'], 0) for n, lim in hand_limits.items())
            fd_ratio = max(abs(fd[task.robot._joint_index[n]]) / lim['velocity'] for n, lim in hand_limits.items())
            dynamic['max_actual_fd_rad_s'] = max(dynamic['max_actual_fd_rad_s'], float(np.max(np.abs(fd[hand_ids]))))
            dynamic['max_actual_fd_source_ratio'] = max(dynamic['max_actual_fd_source_ratio'], float(fd_ratio))
            dynamic['max_engine_qvel_rad_s'] = max(dynamic['max_engine_qvel_rad_s'], float(np.max(np.abs(engine_v[hand_ids]))))
            dynamic['max_mimic_residual_rad'] = max(dynamic['max_mimic_residual_rad'], float(residual))
            dynamic['max_hand_joint_limit_excess_rad'] = max(dynamic['max_hand_joint_limit_excess_rad'], float(excess))
            telemetry.append((steps[0], (steps[0]-1)*dt, steps[0]*dt, before24, applied24,
                              state24_actual(), q_before[hand_ids], q_after[hand_ids],
                              hand_targets, fd[hand_ids], engine_v[hand_ids], q_before, q_after))
            state = context['bottle_state'](task)
            grip = context['grip_measure'](task)
            if stage['name'] == 'after_grasp':
                ref = context['carry_reference'][0]
                if ref is None: raise RuntimeError('Carry began without a fresh accepted relative grasp reference')
                actual = np.asarray(grip['relative_bottle_palm'])
                slip = float(np.linalg.norm(actual[:3, 3]-ref[:3, 3]))
                tilt = float(np.degrees(np.arccos(np.clip((np.trace(ref[:3, :3].T @ actual[:3, :3])-1)/2, -1, 1))))
                context['carry_audit']['max_translation_drift_m'] = max(context['carry_audit']['max_translation_drift_m'], slip)
                context['carry_audit']['max_rotation_drift_deg'] = max(context['carry_audit']['max_rotation_drift_deg'], tilt)
                if slip > .002 or tilt > 2:
                    raise RuntimeError(f'Carry replay slip guard: translation={slip}, rotation_deg={tilt}')
            if context['strict_hold'][0]: context['hold_samples'].append(grip)
            if stage['name'] in ('close','close_hold'): context['pre_retreat_samples'].append(grip)
            fp = np.asarray(state['functional_point0'][:3]); rotation = np.asarray(state['pose'])[:3, :3]
            delta = float(np.linalg.norm(fp - initial_fp))
            angle = float(np.degrees(np.arccos(np.clip((np.trace(initial_R.T @ rotation)-1)/2, -1, 1))))
            contacts = [c for c in state['hand_contacts'] if c['active_point_count'] > 0]
            # Record every physical step, including the offending state before
            # a true nonexempt self-contact guard aborts the attempted replay.
            context['traces'].append({'step': steps[0], 'sim_time_s': steps[0]*dt,
                'phase': dict(stage), 'fp_delta_m': delta, 'rotation_delta_deg': angle,
                'bottle': state, 'grip': grip,'full_qpos_actual_rad':q_after.tolist(),
                'arm_qpos':{side:[q[n] for n in getattr(task.robot,side+'_arm_joints_name')] for side in ('left','right')},
                'actual_self_contacts':actual_self_contacts,'bottle_actual_contacts':bottle_actual_contacts,
                'capture_latched': list(feedback['latched']), 'capture_filtered_N': feedback['filtered_N'].tolist(),
                'capture_actual_pair_streak': feedback['actual_pair_streak'],
                'mimic_residual_rad': residual, 'hand_actual_fd_max_rad_s': float(np.max(np.abs(fd[hand_ids]))),
                'command24_actual': applied24.tolist()})
            # Evaluate exactly once AFTER the real step and its evidence row.
            # A failing state remains available in the trace; no validation step is added.
            if context['support_monitor'].get('active'):
                try:
                    support_state=context['check_support_departure'](task)
                except BaseException as error:
                    context['traces'][-1]['support_departure']={
                        **_support_monitor_summary(context['support_monitor']),
                        'guard_passed':False,'guard_error':str(error)}
                    raise
                if support_state is None:raise RuntimeError('Active support monitor returned no actual physical evidence')
                context['traces'][-1]['support_departure']=dict(support_state,guard_passed=True)
            external_movie.capture(steps[0])
            if active_self:
                raise RuntimeError('Actual native nonexempt self contact during replay: '+str(active_self))
            if fd_ratio > 1.05 or residual > .01 or excess > .0001:
                raise RuntimeError('Actual hand FD/mimic/joint-limit guard failed')
            if (stage['name'] == 'approach' and (delta > .001 or angle > 1 or contacts)) or (stage['name'] in ('close', 'close_hold') and (delta > .002 or angle > 2)):
                raise RuntimeError(f'No-push replay guard: delta={delta}, angle={angle}, contacts={contacts}')
        task.scene.step = trace_step
        original_obs = task.get_obs
        def timed_obs():
            obs = original_obs()
            obs['recording_clock'] = {'physics_step': steps[0], 'sim_time_s': steps[0]*dt}
            obs['object_audit'] = context['bottle_state'](task)
            dex = dict(obs.get('dexterous_state', {}))
            dex.update(state24=state24_actual().tolist(), command24=command24_actual().tolist(),
                       command_source='actual articulation drive targets, not previous cached SDK commands')
            obs['dexterous_state'] = dex
            return obs
        task.get_obs = timed_obs
        original_dense = worker['Base_Task'].take_dense_action
        def dense_trace(self, sequence, **kwargs):
            context['dense_calls'][0] += 1
            stage.update(gripper=sequence.get('left_gripper') is not None,
                         dense_call=context['dense_calls'][0])
            stage['name'] = 'close' if stage['gripper'] else ('approach' if context['dense_calls'][0] <= 2 else 'after_grasp')
            dense_call=context['dense_calls'][0]
            if stage['gripper']:context['pre_retreat_samples'].clear()
            physical_step_start=steps[0]
            scheduled=dense_call in candidate_wrenches
            if stage['name']=='after_grasp':
                if scheduled:
                    entry=candidate_wrenches[dense_call]
                    for file,expected in entry['source_hashes'].items():
                        if worker['sha'](Path(file))!=expected:
                            raise RuntimeError('Source path/wrench changed before accepted replay action')
                    applied=_install_saved_wrench_for_action(feedback,sequence,dense_call=dense_call,
                        native_left_consumed=self.left_cnt,schedules=candidate_wrenches,
                        actual_relative=context['grip_measure'](self)['relative_bottle_palm'])
                    context['report']['replay_wrench_actions'].append(dict(applied,physical_step_start=steps[0]))
                elif context['strict_hold'][0] and dense_call==max(candidate_wrenches)+1:
                    last=feedback.get('wrench_schedule')
                    if last is None or feedback['wrench_index']!=len(last['force_target_world_N']) or self.left_cnt!=len(saved_paths['left_joint_path']):
                        raise RuntimeError('Final stationary hold requires fully consumed accepted carry schedule')
                else:
                    raise RuntimeError('Unscheduled post-grasp action would silently replace accepted inertial wrench with static mg')
            if dense_call==4:
                if (dense_call!=min(candidate_wrenches) or context['report'].get('pre_retreat_grip',{}).get('passed') is not True
                        or context['carry_reference'][0] is None or context['support_monitor']):
                    raise RuntimeError('First carry requires fresh accepted close and exactly one support configuration')
                context['report']['support_departure_initial']=context['configure_support_departure'](self)
                if context['support_monitor'].get('active') is not True or context['support_monitor'].get('configured_at_step')!=steps[0]:
                    raise RuntimeError('Fresh support monitor was not anchored before the first carry step')
            result = original_dense(self, sequence, **kwargs)
            if scheduled:
                expected_rows=len(candidate_wrenches[dense_call]['schedule']['force_target_world_N'])
                if feedback['wrench_index']!=expected_rows:
                    raise RuntimeError('Candidate inertia schedule was not consumed once per actual path step')
                context['report']['replay_wrench_actions'][-1].update(
                    physical_step_end=steps[0],actual_schedule_rows_consumed=feedback['wrench_index'])
            context['report'].setdefault('action_timing',[]).append({'dense_call':dense_call,'phase':stage['name'],
                'start_step':physical_step_start,'end_step':steps[0],'duration_s':(steps[0]-physical_step_start)*dt})
            if stage['gripper']:
                stage.update(name='close_hold',gripper=False)
                # Gate existing close samples; no extra controller/scene/dense action.
                checks=_closing_window_gate(context['pre_retreat_samples'],feedback)
                actual=context['grip_measure'](self);checks['actual']=actual
                checks['reference_step']=steps[0]
                context['report']['pre_retreat_grip']=checks
                if checks['passed'] is not True:raise RuntimeError('Last50 actual closing samples/current LP do not support retreat')
                context['carry_reference'][0]=np.asarray(actual['relative_bottle_palm']).copy()
            return result
        task.take_dense_action = MethodType(dense_trace, task)
        context['snapshot'](task, 'replay_initial')
        info, replay_accepted = context['strict_official_episode'](task, 'replay')
        legal, excess = task.planned_joints_legal()
        consumed = all(getattr(task, side + '_cnt') == len(saved_paths[side + '_joint_path']) for side in ('left', 'right'))
        canonical = all(worker['sha'](p) == source_report['source_hashes'][str(p)] for p in worker['protected'])
        support_summary=_support_monitor_summary(context['support_monitor'])
        support_complete=bool(support_summary.get('active') is True and support_summary.get('release_step') is not None
            and support_summary.get('configured_at_step')==context['report']['pre_retreat_grip']['reference_step']
            and support_summary.get('checked_steps')==steps[0]-support_summary['configured_at_step'])
        context['report'].update(support_departure_actual=support_summary,support_departure_guard_complete=support_complete)
        if not support_complete:raise RuntimeError('Physical support departure guard was not complete/released on every carry step')
        runtime_final=_validate_runtime_sources(source_report,base,worker['sha'])
        runtime_unchanged=runtime_final==runtime_initial
        context['report'].update(runtime_source_hashes_final=runtime_final,runtime_sources_unchanged=runtime_unchanged)
        context['report'].update(accepted=bool(replay_accepted and legal and consumed and canonical and runtime_unchanged),
            replay_strict_accepted=bool(replay_accepted), planned_joints_legal=bool(legal), joint_limit_excess_rad=float(excess),
            saved_paths_consumed=consumed, dynamic_hand_audit=dynamic, plan_success=bool(task.plan_success),
            final_bottle=context['bottle_state'](task), continuous_hand_profile=feedback['profile_segments'],
            carry_audit=context['carry_audit'], adaptive_force_allocations=feedback['lp_history'])
        wrench_hashes_unchanged=all(worker['sha'](Path(file))==expected
            for entry in candidate_wrenches.values() for file,expected in entry['source_hashes'].items())
        wrench_actions_complete=[entry['action_dense_call'] for entry in context['report']['replay_wrench_actions']]==sorted(candidate_wrenches)
        context['report'].update(candidate_wrench_sources_unchanged=wrench_hashes_unchanged,
            candidate_wrench_actions_complete=wrench_actions_complete)
        context['report']['accepted']=bool(context['report']['accepted'] and wrench_hashes_unchanged and wrench_actions_complete)
        if context['report']['accepted'] is not True:
            context['report']['result'] = 'accepted_replay_failed_no_export'
            raise RuntimeError('Strict reset/replay failed; HDF exporter was not called')
        context['report']['external_realtime_video']=external_movie.close(steps[0])
        # Mandatory full actual36 geometry audit BEFORE any HDF exporter call.
        # Save evidence-only JSONs so the read-only audit can validate exact source
        # hashes/order and every physical frame from the completed accepted replay.
        context['report'].update(result='physics_success',canonical_sources_unchanged=canonical,physical_steps=steps[0])
        trace_file=replay_out/'physics_trace.json';report_file=replay_out/'replay_report.json'
        trace_file.write_text(json.dumps(context['traces'],default=_json_default))
        report_file.write_text(json.dumps(context['report'],indent=2,default=_json_default))
        from observed_full_trace_audit_v13 import audit_accepted_actual_trace
        full_actual_audit=audit_accepted_actual_trace(task,trace_file,report_path=report_file,
            output_path=replay_out/'actual36_geometry_audit.json')
        context['report']['actual36_geometry_audit']={key:value for key,value in full_actual_audit.items() if key!='samples'}
        if full_actual_audit.get('passed') is not True or full_actual_audit.get('physical_scene_unchanged') is not True:
            context['report'].update(accepted=False,result='accepted_replay_actual36_failed_no_export')
            raise RuntimeError('Full actual36 geometry audit failed; HDF exporter was not called')
        context['info'] = info
        context['snapshot'](task, 'replay_final')
        # Export only after complete physical acceptance. Keep exporter staging
        # private until all native14D and true-command24D checks pass.
        stage_out.mkdir(exist_ok=False)
        (stage_out / 'data').mkdir(); (stage_out / 'video').mkdir()
        context['OUT'] = stage_out; task.save_dir = str(stage_out)
        if _validate_runtime_sources(source_report,base,worker['sha'])!=runtime_initial:
            raise RuntimeError('V22 runtime source changed before HDF export')
        dataset = context['_export_native_collector_block']()
        staged_hdf = Path(dataset['hdf5'])
        with h5py.File(staged_hdf, 'a') as f:
            n = int(dataset['pairs'])
            for group in ('state', 'action'):
                for field in ('left_arm_joint_states', 'right_arm_joint_states'):
                    if f[group + '/' + field].shape != (n, 6): raise RuntimeError('Native arm6 export mismatch')
                for field in ('left_ee_joint_states', 'right_ee_joint_states'):
                    if f[group + '/' + field].shape != (n, 1): raise RuntimeError('Native gripper scalar export mismatch')
            sample_times = np.asarray(f['realman_dexterous/sample_time_s'])
            next_times = np.asarray(f['realman_dexterous/next_sample_time_s'])
            if not np.isfinite(sample_times).all() or np.any(np.diff(sample_times) < 0) or np.any(next_times < sample_times):
                raise RuntimeError('Native frame timestamps are not finite/nondecreasing')
            g = f.create_group('realman_dexterous_physics')
            g.attrs['sdk_mapping_calibrated'] = False
            g.attrs['ordering'] = 'Larm6 rad, Lprimary SDK6, Rarm6 rad, Rprimary SDK6'
            g.attrs['hand24_joint_names'] = json.dumps(hand_names)
            g.attrs['full36_joint_names'] = json.dumps(native_auditor.sim_joints)
            g.attrs['command_source'] = 'read actual drive targets after fresh feedback update and before each physics step'
            fields = ('physics_step', 'sim_time_before_s', 'sim_time_after_s', 'measured_state24_before', 'applied_command24', 'measured_state24_after', 'hand24_qpos_before_rad', 'hand24_qpos_after_rad', 'hand24_drive_target_rad', 'hand24_actual_fd_rad_s', 'hand24_engine_qvel_rad_s', 'full36_qpos_before_rad', 'full36_qpos_after_rad')
            for i, field in enumerate(fields):
                data = np.asarray([row[i] for row in telemetry])
                if not np.isfinite(data).all(): raise RuntimeError('Nonfinite full physics telemetry')
                g.create_dataset(field, data=data, compression='gzip')
            if g['applied_command24'].shape != (steps[0], 24): raise RuntimeError('Full physics command24 shape mismatch')
            f['realman_dexterous'].attrs['command_source'] = 'actual drive targets at native frame timestamps'
        for directory in ('data', 'video'):
            (replay_out / directory).mkdir(exist_ok=False)
        publish_files = ((staged_hdf, replay_out / 'data' / staged_hdf.name),
                         (Path(dataset['video']), replay_out / 'video' / Path(dataset['video']).name),
                         (stage_out / 'scene_info.json', replay_out / 'scene_info.json'),
                         (external_movie.path,replay_out/'video'/'external_realtime.mp4'))
        for staged, final in publish_files:
            if final.exists(): raise RuntimeError('Refusing to overwrite replay output')
            if not staged.is_file(): raise RuntimeError('Native export output missing: ' + str(staged))
        for staged, final in publish_files:
            staged.replace(final)
        context['OUT'] = replay_out; task.save_dir = str(replay_out)
        dataset.update(hdf5=str(replay_out / 'data' / staged_hdf.name), video=str(replay_out / 'video' / Path(dataset['video']).name))
        dataset['external_realtime_video']=str(replay_out/'video'/'external_realtime.mp4')
        context['report']['external_realtime_video']['path']=dataset['external_realtime_video']
        dataset['external_realtime_video_sha256']=worker['sha'](Path(dataset['external_realtime_video']))
        dataset['sha256'] = worker['sha'](Path(dataset['hdf5']))
        context['report'].update(result='one_strictly_validated_episode_collected', dataset=dataset,
                                 physical_steps=steps[0], native14D_preserved=True, actual24D_preserved=True)
        print('ACCEPTED_REPLAY_DATASET', json.dumps(dataset), flush=True)
        worker['accepted_replay_out'] = replay_out
        worker['accepted_replay_report'] = context['report']
    except BaseException:
        context['report'].update(result='accepted_replay_exception_no_validated_publish', traceback=traceback.format_exc())
        raise
    finally:
        if external_movie is not None:
            context['report'].setdefault('external_realtime_video',external_movie.close(context['steps'][0],validate=False))
        context['OUT'] = replay_out
        context['report']['canonical_sources_unchanged'] = all(worker['sha'](p) == source_report['source_hashes'][str(p)] for p in worker['protected'])
        try:
            observed_runtime={file:worker['sha'](Path(file)) for file in runtime_initial}
            context['report'].update(runtime_source_hashes_final=observed_runtime,runtime_sources_unchanged=(observed_runtime==runtime_initial))
        except BaseException:
            context['report'].update(runtime_sources_unchanged=False,runtime_hash_check_error=traceback.format_exc())
        context['report']['source_trajectory_unchanged'] = worker['sha'](trajectory) == context['report']['source_trajectory_sha256']
        context['report']['coupling_class_hook_preserved'] = worker['RealManRobot']._initialize_hands is coupling_hook
        context['report']['carry_audit'] = context['carry_audit']
        context['report']['support_departure_actual']=_support_monitor_summary(context['support_monitor'])
        if 'feedback' in context: context['report']['adaptive_force_allocations'] = context['feedback'].get('lp_history', [])
        (replay_out / 'replay_report.json').write_text(json.dumps(context['report'], indent=2, default=_json_default))
        (replay_out / 'physics_trace.json').write_text(json.dumps(context['traces'], default=_json_default))
        (replay_out / 'final_hold_contact_trace.json').write_text(json.dumps(context['hold_samples'], default=_json_default))
        if telemetry:
            np.savez_compressed(replay_out / 'physics_command24_trace.npz',
                physics_step=np.asarray([r[0] for r in telemetry]),
                sim_time_before_s=np.asarray([r[1] for r in telemetry]),
                sim_time_after_s=np.asarray([r[2] for r in telemetry]),
                measured_state24_before=np.asarray([r[3] for r in telemetry]),
                applied_command24=np.asarray([r[4] for r in telemetry]),
                measured_state24_after=np.asarray([r[5] for r in telemetry]),
                hand24_qpos_before_rad=np.asarray([r[6] for r in telemetry]),
                hand24_qpos_after_rad=np.asarray([r[7] for r in telemetry]),
                hand24_actual_fd_rad_s=np.asarray([r[9] for r in telemetry]),
                full36_qpos_before_rad=np.asarray([r[11] for r in telemetry]),
                full36_qpos_after_rad=np.asarray([r[12] for r in telemetry]))
        if original_step is not None: task.scene.step = original_step
        if original_obs is not None: task.get_obs = original_obs
        for name in ('take_dense_action', 'choose_grasp_pose', 'get_place_pose', 'move_by_displacement'):
            task.__dict__.pop(name, None)
        for name in ('set_gripper', 'left_plan_grippers', 'left_plan_path', 'right_plan_path', 'left_plan_multi_path', 'right_plan_multi_path'):
            task.robot.__dict__.pop(name, None)
        print('ACCEPTED_REPLAY_FINISHED', context['report']['result'], flush=True)


_run_accepted_reset_replay(globals())
