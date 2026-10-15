"""Private official TOPP request only; original hardware gates and outputs retained.

No scene/model/GPU is created. Explicit 0.99 request scale is independently
reviewable; default 1 returns the original carry module without a replacement.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import inspect
import math
import numbers
import types
from pathlib import Path
import numpy as np

ORIGINAL_CARRY_SHA256 = '7a0d4cb0cedc40ee25daed1226a073705fc71bf1dc9a2a34064a11cd588b21d7'
NATIVE_TOPP_FUNCTION_SHA256 = '3aeae04bf979a573315010c6580b51ecaad714e99b41ea6f5df039d2af0e4b8f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_margin(spec):
    value = spec.get('native_topp_request_margin', 1)
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value not in (1, .99):
        raise ValueError('native_topp_request_margin must be exactly 1 or 0.99')
    return float(value)


def _native_request(task, planner, native, path, *, step, margin):
    code = getattr(native, '__func__', None)
    if (code is None or code.__module__ != 'mplib.planner'
            or getattr(native, '__self__', None) is not planner
            or getattr(planner.TOPP, '__func__', None) is not code
            or hashlib.sha256(inspect.getsource(code).encode()).hexdigest() != NATIVE_TOPP_FUNCTION_SHA256):
        raise ValueError('Expected unchanged installed official TOPP bound to current planner')
    if margin != .99:
        raise ValueError('Private request hook only permits the explicit reviewed 0.99 margin')
    if getattr(task, '_rgb50_native_topp_request_active', False):
        raise RuntimeError('Nested native TOPP limit requests forbidden')
    original_v = planner.joint_vel_limits
    original_a = planner.joint_acc_limits
    hardware_v = np.asarray(original_v, dtype=float).copy()
    hardware_a = np.asarray(original_a, dtype=float).copy()
    if (hardware_v.ndim != 1 or hardware_a.shape != hardware_v.shape
            or not np.isfinite(hardware_v).all() or not np.isfinite(hardware_a).all()
            or np.any(hardware_v <= 0) or np.any(hardware_a <= 0)):
        raise ValueError('Invalid actual hardware limits for native request')
    requested_v = hardware_v * margin
    requested_a = hardware_a * margin
    receipt = {'request_margin': margin, 'hardware_velocity_limits_rad_s': hardware_v.tolist(),
        'hardware_acceleration_limits_rad_s2': hardware_a.tolist(),
        'native_requested_velocity_limits_rad_s': requested_v.tolist(),
        'native_requested_acceleration_limits_rad_s2': requested_a.tolist(),
        'input_knots_sha256': hashlib.sha256(np.asarray(path,dtype=float).tobytes()).hexdigest(),
        'actual_native_function_sha256': NATIVE_TOPP_FUNCTION_SHA256,
        'original_hardware_arrays_not_modified_in_place': True,
        'status': 'requested', 'restoration_verified': False,
        'scope': 'Official TOPP conservative request only; no hardware limit, physical state, returned trajectory or original acceptance threshold changes'}
    receipts = getattr(task, '_rgb50_native_topp_request_receipts', None)
    if receipts is None:
        receipts = []
        task._rgb50_native_topp_request_receipts = receipts
    receipts.append(receipt)
    task._rgb50_native_topp_request_active = True
    try:
        planner.joint_vel_limits = requested_v
        planner.joint_acc_limits = requested_a
        result = native(path, step=step)
        receipt['status'] = 'native_returned'
    except BaseException as exc:
        receipt.update(status='native_failed', exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        planner.joint_vel_limits = original_v
        planner.joint_acc_limits = original_a
        task._rgb50_native_topp_request_active = False
        if (planner.joint_vel_limits is not original_v or planner.joint_acc_limits is not original_a
                or not np.array_equal(np.asarray(original_v), hardware_v)
                or not np.array_equal(np.asarray(original_a), hardware_a)):
            receipt['status'] = 'restore_failed'
            raise RuntimeError('Official TOPP original limit objects/numeric values not exactly restored')
        receipt['restoration_verified'] = True
    # Extra strict clock audit only rejects; the original gates still run next
    # against their captured original hardware arrays (not the 0.99 requests).
    ts,pos,vel,acc,duration = result
    pos=np.asarray(pos,dtype=float);vel=np.asarray(vel,dtype=float)
    dt=float(task.scene.get_timestep())
    if not math.isfinite(dt) or dt <= 0 or not math.isclose(dt,step,abs_tol=1e-8,rel_tol=0):
        raise ValueError('Native request clock differs from actual scene timestep')
    if pos.ndim != 2 or vel.shape != pos.shape or len(pos)<2 or pos.shape[1]!=len(hardware_v) or not np.isfinite(pos).all() or not np.isfinite(vel).all():
        raise ValueError('Invalid native return for strict hardware clock audit')
    fd=np.max(np.abs(np.diff(pos,axis=0)/dt),axis=0)
    va=np.max(np.abs(np.diff(vel,axis=0)/dt),axis=0)
    strict=bool(np.all(fd<=hardware_v+1e-5) and np.all(va<=hardware_a+1e-5))
    receipt.update(actual_physics_dt_s=dt, physical_clock_fd_peak_rad_s=fd.tolist(),
        physical_clock_velocity_target_acc_peak_rad_s2=va.tolist(),
        additional_strict_hardware_clock_audit_passed=strict,
        returned_commands_sha256=hashlib.sha256(pos.tobytes()).hexdigest(),
        status='native_returned_strict_clock_passed' if strict else 'strict_clock_rejected')
    if not strict:
        raise ValueError('Conservative official TOPP request still exceeds original strict hardware 4ms FD limits')
    return result


def private_official_tree(payload):
    tree=ast.parse(payload)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='official_topp_command_polyline']
    if len(functions)!=1:
        raise RuntimeError('Exactly one original official retimer required')
    fn=copy.deepcopy(functions[0]);sites=[]
    for n in ast.walk(fn):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='native':sites.append(n)
    if len(sites)!=1 or ast.unparse(sites[0])!='native(p, step=time_step)':
        raise RuntimeError('Unexpected original official TOPP invocation')
    call=sites[0]
    replacement=ast.parse('_safe_native_request(task, planner, native, p, step=time_step)').body[0].value
    call.func=replacement.func;call.args=replacement.args;call.keywords=replacement.keywords
    module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
    return module


def private_carry_module(original, margin):
    source=Path(original.__file__).resolve()
    if sha(source)!=ORIGINAL_CARRY_SHA256:
        raise RuntimeError('Frozen original carry planner source changed')
    if margin==1:
        return original, {'request_margin':1.,'original_module_and_function_returned_unchanged':True}
    if margin!=.99:
        raise ValueError('Unknown private TOPP margin')
    tree=private_official_tree(source.read_text(encoding='utf-8-sig'))
    namespace=dict(original.__dict__)
    namespace['_safe_native_request']=lambda task,planner,native,path,step: _native_request(task,planner,native,path,step=step,margin=margin)
    exec(compile(tree,str(Path(__file__).resolve())+':private_original_retimer','exec'),namespace)
    compiled=namespace['official_topp_command_polyline']
    def with_receipt(task,*args,**kwargs):
        pos,vel,ts,meta=compiled(task,*args,**kwargs)
        return pos,vel,ts,{**meta,'conservative_native_TOPP_request':copy.deepcopy(task._rgb50_native_topp_request_receipts[-1])}
    namespace['official_topp_command_polyline']=with_receipt
    function=original.plan_carry_to_task_pose
    rebound=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    rebound.__kwdefaults__=function.__kwdefaults__
    private=types.ModuleType('private_original_carry_native_topp_request_v1')
    private.__dict__.update(namespace)
    private.plan_carry_to_task_pose=rebound
    return private, {'request_margin':margin,'original_module_globals_unmodified':True,
        'private_original_retimer_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'original_carry_sha256':ORIGINAL_CARRY_SHA256,
        'only_original_retimer_AST_change':'native(p,step=time_step) replaced by explicit conservative request hook',
        'all_original_acceptance_gates_and_hardware_limit_metadata_preserved':True,
        'additional_strict_4ms_hardware_clock_gate':True}
