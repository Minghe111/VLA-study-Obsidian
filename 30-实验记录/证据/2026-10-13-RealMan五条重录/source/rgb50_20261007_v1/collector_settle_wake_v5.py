"""Extra natural initialization steps only; frozen delay(4) and exporter retained.

No task episode is run by importing this module. Extra duration defaults to 0;
Optional 8/12 seconds, or two explicit 8-second wake rounds, follow original setup;
actual geometry is measured afterward, before expert creation and RGB recording.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import math
import numbers
import numpy as np
from pathlib import Path
import generation_runner_v2 as original

WORK = Path(__file__).resolve().parent
ORIGINAL_RUNNER_SHA256 = 'aa466900867a899654bc7d6f1a5b1d8b643fd22436a1654820849fbce330955c'
ORIGINAL_WORKER_SHA256 = '258833e23cb43e6fc1d55d0b300ded27414e548e12f328dfb2e7efad7fe2c6bd'
ROOT, LEGACY, OLD20, CONFIG = original.ROOT, original.LEGACY, original.OLD20, original.CONFIG
sha, json_write, export_recording = original.sha, original.json_write, original.export_recording


def require_source(path, digest):
    path = Path(path)
    if not path.is_file() or sha(path) != digest:
        raise RuntimeError('Frozen initialization producer source changed: ' + str(path))
    return str(path.resolve())


def producer_pins():
    require_source(WORK / 'generation_runner_v2.py', ORIGINAL_RUNNER_SHA256)
    require_source(WORK / 'generation_worker_v2.py', ORIGINAL_WORKER_SHA256)
    paths = [WORK / name for name in ('generation_runner_v2.py', 'generation_worker_v2.py',
             'collector_settle_wake_v5.py', 'worker_settle_wake_v5.py', 'expert_v9_settle_wake_v5.py')]
    if any(not p.is_file() for p in paths):
        raise RuntimeError('Missing complete initialization producer sources')
    return {str(p.resolve()): sha(p) for p in paths}


def settle_plan(spec, dt):
    value = spec.get('initial_extra_settle_s', 0)
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value not in (0, 8, 12):
        raise ValueError('initial_extra_settle_s must be exactly 0, 8 or 12 seconds')
    if isinstance(dt, bool) or not isinstance(dt, numbers.Real) or not math.isfinite(float(dt)) or dt <= 0 or abs(float(dt) - .004) > 1e-8:
        raise ValueError('Extra initialization requires the actual finite RoboTwin 4ms physics clock')
    wake = spec.get('initial_wake_before_extra_settle', False)
    if type(wake) is not bool:
        raise ValueError('initial_wake_before_extra_settle must be an explicit boolean')
    if wake and value not in (8, 12):
        raise ValueError('A requested native wake requires exactly 8 or 12 extra seconds')
    rounds = spec.get('initial_wake_rounds', 1)
    if type(rounds) is not int or rounds not in (1, 2):
        raise ValueError('initial_wake_rounds must be the explicit integer 1 or 2')
    if rounds == 2 and (not wake or value != 8):
        raise ValueError('Two native wake rounds require explicit wake=true and 8 seconds per round')
    round_steps = int(round(float(value) / float(dt)))
    steps = round_steps * rounds
    return {'requested_extra_seconds': int(value), 'initial_wake_before_extra_settle': wake, 'actual_physics_dt_s': float(dt),
            'initial_wake_rounds': rounds, 'requested_seconds_per_round': int(value),
            'extra_physics_steps_per_round': round_steps,
            'extra_physics_steps': steps, 'actual_extra_simulated_seconds': steps * float(dt),
            'step_count_rounding': 'round(requested_seconds / actual_scene_timestep)'}


def _native_bottle_component(task):
    candidates = [c for c in task.bottle.actor.get_components()
                  if hasattr(c, 'collision_shapes') and callable(getattr(c, 'wake_up', None))]
    if len(candidates) != 1:
        raise RuntimeError('Exactly one current native dynamic bottle component required')
    component = candidates[0]
    if component.entity is not task.bottle.actor:
        raise RuntimeError('Native wake component is not the current bottle actor')
    for name in ('is_sleeping', 'get_sleep_threshold', 'get_linear_velocity', 'get_angular_velocity'):
        if not hasattr(component, name):
            raise RuntimeError('Missing installed native bottle read API: ' + name)
    return component


def _read_native_bottle_state(task, component):
    sleeping = component.is_sleeping
    sleeping = sleeping() if callable(sleeping) else sleeping
    if not isinstance(sleeping, (bool, np.bool_)):
        raise RuntimeError('Native sleeping state is not boolean')
    threshold = float(component.get_sleep_threshold())
    matrix = np.asarray(task.bottle.get_pose().to_transformation_matrix(), dtype=float).copy()
    linear = np.asarray(component.get_linear_velocity(), dtype=float).copy()
    angular = np.asarray(component.get_angular_velocity(), dtype=float).copy()
    q = np.asarray(task.robot.left_entity.get_qpos(), dtype=float).copy()
    if (not math.isfinite(threshold) or threshold < 0 or matrix.shape != (4, 4)
            or linear.shape != (3,) or angular.shape != (3,) or q.ndim != 1 or not len(q)
            or not all(np.isfinite(v).all() for v in (matrix, linear, angular, q))):
        raise RuntimeError('Invalid finite native bottle/robot initialization state')
    return {'native_is_sleeping': bool(sleeping), 'native_sleep_threshold': threshold,
            'bottle_pose_matrix': matrix.tolist(), 'linear_velocity_m_s': linear.tolist(),
            'angular_velocity_rad_s': angular.tolist(), 'robot_full_qpos_rad': q.tolist()}


def _verify_immediate_native_wake(before, after):
    fields = ['native_sleep_threshold', 'bottle_pose_matrix', 'linear_velocity_m_s',
              'angular_velocity_rad_s', 'robot_full_qpos_rad']
    for name in fields:
        if not np.array_equal(np.asarray(before[name]), np.asarray(after[name])):
            raise RuntimeError('Native wake changed numeric initialization state: ' + name)
    if after['native_is_sleeping'] is not False:
        raise RuntimeError('Native bottle remains sleeping immediately after wake')
    return {'numeric_fields_exactly_equal_before_after': fields,
            'pose_velocity_threshold_q_before_after_equal': True,
            'native_after_is_sleeping_false': True,
            'only_native_sleep_state_allowed_to_change': True}


def _extra_initial_settle(task, spec):
    """Explicit one/two native wake rounds; ordinary pre-recording physics only."""
    if hasattr(task, '_rgb50_initial_extra_settle_receipt'):
        raise RuntimeError('Extra initialization must run exactly once')
    if getattr(task.robot, 'communication_flag', False):
        raise RuntimeError('Physical robot IO forbidden')
    pins = producer_pins()
    plan = settle_plan(spec, task.scene.get_timestep())
    wake = {'requested': plan['initial_wake_before_extra_settle'], 'wake_calls': 0,
            'native_state_before': None, 'native_state_immediately_after': None,
            'immediate_verification': None, 'native_state_after_extra': None,
            'rounds': []}
    completed = 0
    if plan['initial_wake_before_extra_settle']:
        component = _native_bottle_component(task)
        for round_index in range(plan['initial_wake_rounds']):
            before = _read_native_bottle_state(task, component)
            if wake['rounds'] and before != wake['rounds'][-1]['native_state_after_extra']:
                raise RuntimeError('Native state changed between initialization rounds without a physics step')
            component.wake_up()
            wake['wake_calls'] += 1
            after = _read_native_bottle_state(task, component)
            verification = _verify_immediate_native_wake(before, after)
            if round_index == 0:
                wake.update(native_state_before=before, native_state_immediately_after=after,
                            immediate_verification=verification)
            round_completed = 0
            for _ in range(plan['extra_physics_steps_per_round']):
                task.scene.step()
                completed += 1
                round_completed += 1
            final = _read_native_bottle_state(task, component)
            wake['native_state_after_extra'] = final
            wake['rounds'].append({'round_index': round_index + 1, 'wake_calls': 1,
                'native_state_before': before, 'native_state_immediately_after': after,
                'immediate_verification': verification, 'native_state_after_extra': final,
                'completed_extra_physics_steps': round_completed,
                'actual_physics_dt_s': plan['actual_physics_dt_s'],
                'actual_extra_simulated_seconds': round_completed * plan['actual_physics_dt_s']})
    else:
        for _ in range(plan['extra_physics_steps']):
            task.scene.step()
            completed += 1
    if producer_pins() != pins:
        raise RuntimeError('Producer source changed during initialization')
    receipt = {**plan, 'completed_extra_physics_steps': completed,
        'native_wake': wake, 'source_hashes': pins, 'hardware_io': False,
        'scope': 'Initialized candidate: explicitly requested one/two native wake rounds and ordinary extra physics after complete original setup_demo including unchanged delay(4) and official check_stable, before original actual geometry measurements and expert/recording. This is not an official stability certification or a demonstrated fix.',
        'original_delay_4_gripper_cycles_preserved': True, 'official_check_stable_unmodified': True,
        'original_setup_demo_completed_before_extra': True, 'official_check_stable_not_repeated': True,
        'recording_or_episode_pause': False, 'stability_not_certified': True,
        'no_fix_or_grasp_success_claim': True,
        'object_pose_velocity_sleep_threshold_and_robot_q_not_written_by_helper': True,
        'object_pose_velocity_and_robot_q_not_written_by_extra_settle': True,
        'renderer_observation_calls_by_extra_settle': 0}
    task._rgb50_initial_extra_settle_receipt = receipt
    return receipt

def private_prepare_tree(payload):
    tree = ast.parse(payload)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_task']
    if len(functions) != 1:
        raise RuntimeError('Exactly one frozen prepare_task is required')
    function = copy.deepcopy(functions[0])
    # Preserve the entire original load_actors delay/stability chain.  Its delay
    # remains four gripper-control cycles, never reinterpreted as seconds.
    delays = []
    for cls in ast.walk(function):
        if not isinstance(cls, ast.ClassDef) or cls.name != 'SampleScene':
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name != 'load_actors':
                continue
            for statement in method.body:
                call = statement.value if isinstance(statement, ast.Expr) else None
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'self' and call.func.attr == 'delay':
                    if len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Constant) or type(call.args[0].value) is not int or call.args[0].value != 4:
                        raise RuntimeError('Unexpected original initialization delay site')
                    delays.append(call)
    if len(delays) != 1:
        raise RuntimeError('Exactly one unchanged original delay(4) is required')
    sites = 0
    body = []
    for statement in function.body:
        body.append(statement)
        call = statement.value if isinstance(statement, ast.Expr) else None
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'task' and call.func.attr == 'setup_demo':
            if call.args or len(call.keywords) != 1 or call.keywords[0].arg is not None or not isinstance(call.keywords[0].value, ast.Name) or call.keywords[0].value.id != 'args':
                raise RuntimeError('Unexpected original task.setup_demo(**args) signature')
            body.append(ast.parse('_extra_initial_settle(task, spec)').body[0])
            sites += 1
    all_setup_calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == 'setup_demo']
    if sites != 1 or len(all_setup_calls) != 1:
        raise RuntimeError('Exactly one post-complete-original-setup insertion is required')
    function.body = body
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    return module, {'inserted_extra_settle_sites': sites,
        'original_delay_statement_changed': False,
        'insertion_scope': 'Immediately after complete original task.setup_demo(**args), before original actual mesh/bbox/shoulder/pose measurements',
        'original_official_stability_chain_preserved_and_not_repeated': True,
        'private_prepare_ast_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()}


require_source(Path(original.__file__), ORIGINAL_RUNNER_SHA256)
_TREE, PRIVATE_BINDING = private_prepare_tree(Path(original.__file__).read_text(encoding='utf-8-sig'))
_ENV = dict(original.__dict__)
_ENV['_extra_initial_settle'] = _extra_initial_settle
exec(compile(_TREE, str(WORK / 'collector_settle_wake_v5.py') + ':private_prepare', 'exec'), _ENV)
_PRIVATE_PREPARE = _ENV['prepare_task']


def prepare_task(spec, out, **kwargs):
    """Original setup with one extra initialization call; unchanged export API."""
    pins = producer_pins()
    settle_plan(spec, .004)  # Reject unknown requests before constructing a scene.
    task, scene, initializer = _PRIVATE_PREPARE(spec, out, **kwargs)
    receipt = getattr(task, '_rgb50_initial_extra_settle_receipt', None)
    if receipt is None or receipt['source_hashes'] != pins or producer_pins() != pins:
        raise RuntimeError('Missing or changed pre-recording initialization receipt')
    receipt = {**receipt, 'private_prepare_binding': PRIVATE_BINDING,
               'three_RGB_exporter_original_function_reused': export_recording is original.export_recording}
    task._rgb50_initial_extra_settle_receipt = receipt
    scene['initial_extra_settle'] = copy.deepcopy(receipt)
    json_write(Path(out) / 'scene_sample.json', scene)
    return task, scene, initializer
