"""Extra natural initialization steps only; frozen delay(4) and exporter retained.

No task episode is run by importing this module. Extra duration defaults to 0;
8/12 seconds precede official check_stable, expert creation and RGB recording.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import math
import numbers
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
             'collector_settle_v3.py', 'worker_settle_v3.py', 'expert_v9_settle_v3.py')]
    if any(not p.is_file() for p in paths):
        raise RuntimeError('Missing complete initialization producer sources')
    return {str(p.resolve()): sha(p) for p in paths}


def settle_plan(spec, dt):
    value = spec.get('initial_extra_settle_s', 0)
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value not in (0, 8, 12):
        raise ValueError('initial_extra_settle_s must be exactly 0, 8 or 12 seconds')
    if isinstance(dt, bool) or not isinstance(dt, numbers.Real) or not math.isfinite(float(dt)) or dt <= 0 or abs(float(dt) - .004) > 1e-8:
        raise ValueError('Extra initialization requires the actual finite RoboTwin 4ms physics clock')
    steps = int(round(float(value) / float(dt)))
    return {'requested_extra_seconds': int(value), 'actual_physics_dt_s': float(dt),
            'extra_physics_steps': steps, 'actual_extra_simulated_seconds': steps * float(dt),
            'step_count_rounding': 'round(requested_seconds / actual_scene_timestep)'}


def _extra_initial_settle(task, spec):
    """Only ordinary pre-recording scene.step; no delay or state setters."""
    if hasattr(task, '_rgb50_initial_extra_settle_receipt'):
        raise RuntimeError('Extra initialization must run exactly once')
    if getattr(task.robot, 'communication_flag', False):
        raise RuntimeError('Physical robot IO forbidden')
    pins = producer_pins()
    plan = settle_plan(spec, task.scene.get_timestep())
    completed = 0
    for _ in range(plan['extra_physics_steps']):
        task.scene.step()
        completed += 1
    if producer_pins() != pins:
        raise RuntimeError('Producer source changed during initialization')
    receipt = {**plan, 'completed_extra_physics_steps': completed,
        'source_hashes': pins, 'hardware_io': False,
        'scope': 'Extra ordinary physics after original delay(4), before unchanged official check_stable and before expert/recording',
        'original_delay_4_gripper_cycles_preserved': True, 'official_check_stable_unmodified': True,
        'recording_or_episode_pause': False, 'stability_not_certified': True,
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
    sites = 0
    for cls in ast.walk(function):
        if not isinstance(cls, ast.ClassDef) or cls.name != 'SampleScene':
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name != 'load_actors':
                continue
            body = []
            for statement in method.body:
                body.append(statement)
                call = statement.value if isinstance(statement, ast.Expr) else None
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'self' and call.func.attr == 'delay':
                    if len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Constant) or type(call.args[0].value) is not int or call.args[0].value != 4:
                        raise RuntimeError('Unexpected initialization delay site')
                    body.append(ast.parse('_extra_initial_settle(self, spec)').body[0])
                    sites += 1
            method.body = body
    if sites != 1:
        raise RuntimeError('Exactly one insertion after preserved delay(4) is required')
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    return module, {'inserted_extra_settle_sites': sites,
        'original_delay_statement_changed': False,
        'private_prepare_ast_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()}


require_source(Path(original.__file__), ORIGINAL_RUNNER_SHA256)
_TREE, PRIVATE_BINDING = private_prepare_tree(Path(original.__file__).read_text(encoding='utf-8-sig'))
_ENV = dict(original.__dict__)
_ENV['_extra_initial_settle'] = _extra_initial_settle
exec(compile(_TREE, str(WORK / 'collector_settle_v3.py') + ':private_prepare', 'exec'), _ENV)
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