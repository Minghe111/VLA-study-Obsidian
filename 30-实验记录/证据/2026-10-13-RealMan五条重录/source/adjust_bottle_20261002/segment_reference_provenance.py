"""Offline carry-segment reference provenance; no SAPIEN, worker, or GPU APIs.

Checks source data flow, NPZ/audit/report reference agreement, the measured
pre-execution trace row, full-DOF ordering, and wrench/preflight array binding.
The existing wrench files omit inertial inputs and the frozen reference:
source-level binding is explicitly distinguished from numerical recomputation.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix(value, label):
    a = np.asarray(value, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError(label + ' must be a finite 4x4 matrix')
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError(label + ' is not homogeneous')
    if np.max(np.abs(a[:3,:3].T @ a[:3,:3] - np.eye(3))) > 1e-5:
        raise ValueError(label + ' rotation is not orthonormal')
    return a


def _pose_error(a, b):
    a, b = _matrix(a, 'pose_a'), _matrix(b, 'pose_b')
    # Polar projection only in the numeric angle metric, never in input/output.
    def rotation(r):
        u, _, vt = np.linalg.svd(r)
        return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    r = rotation(a[:3,:3]).T @ rotation(b[:3,:3])
    return (float(np.linalg.norm(a[:3,3] - b[:3,3])),
            float(math.acos(np.clip((np.trace(r)-1)/2, -1., 1.))))


def review_source_dataflow(source_path):
    """Strictly check the v12 recorded reference reaches both helper calls.

    This is evidence about the supplied source file, not proof that a process
    executed its byte-for-byte contents; the episode-source SHA is recorded.
    """
    source_path = Path(source_path)
    tree = ast.parse(source_path.read_text(encoding='utf-8-sig'))
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'precise_path']
    if len(funcs) != 1:
        return {'passed': False, 'reason': 'Expected one precise_path function'}
    fn = funcs[0]
    assignments = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'planning_reference' for t in n.targets)]
    measured = len(assignments) == 1 and 'grip_measure(task)' in ast.unparse(assignments[0].value)
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    def named_value(v, name):
        return isinstance(v, ast.Name) and v.id == name
    planner_calls = [n for n in calls if isinstance(n.func, ast.Attribute)
                     and n.func.attr == 'plan_carry_to_task_pose']
    planner = len(planner_calls) == 1 and any(k.arg == 'palm_to_bottle'
        and named_value(k.value, 'planning_reference') for k in planner_calls[0].keywords)
    wrench_calls = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'planned_carry_wrench']
    wrench = len(wrench_calls) == 1 and len(wrench_calls[0].args) >= 2 and named_value(wrench_calls[0].args[1], 'planning_reference')
    savez_calls = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'savez']
    recorded = any(any(k.arg == 'planning_reference' and named_value(k.value, 'planning_reference') for k in n.keywords) for n in savez_calls)
    def schedule_key(v, key):
        return (isinstance(v, ast.Subscript) and named_value(v.value, 'schedule')
                and isinstance(v.slice, ast.Constant) and v.slice.value == key)
    arrays_saved = any(all(any(k.arg == key and schedule_key(k.value, key) for k in n.keywords)
        for key in ['force_target_world_N','torque_target_world_Nm']) for n in savez_calls)
    hold = len(wrench_calls) == 1 and any(k.arg == 'endpoint_mode' and isinstance(k.value, ast.Constant)
        and k.value.value == 'hold' for k in wrench_calls[0].keywords)
    checks = {'fresh_measured_reference_assignment': measured, 'planner_uses_same_reference': planner,
              'wrench_uses_same_reference': wrench, 'path_npz_records_same_reference': recorded,
              'wrench_npz_records_schedule_arrays': arrays_saved, 'wrench_endpoint_mode_hold': hold}
    return {'passed': bool(all(checks.values())), 'checks': checks,
            'source_path': str(source_path.resolve()), 'source_sha256': _sha(source_path),
            'binding_level': 'AST/source dataflow; not independent numeric wrench recomputation',
            'executed_source_identity_verified': False}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _finite_array(value, shape_tail, label):
    x = np.asarray(value, dtype=float)
    if x.ndim != len(shape_tail)+1 or x.shape[1:] != shape_tail or not len(x) or not np.isfinite(x).all():
        raise ValueError(label + ' has invalid shape or nonfinite values')
    return x


def _reference_check(a, b):
    p, r = _pose_error(a, b)
    return {'passed': bool(p <= 2e-6 and r <= 2e-5), 'position_error_m': p,
            'rotation_error_rad': r, 'matrix_max_abs_error': float(np.max(np.abs(a-b))),
            'position_tolerance_m': 2e-6, 'rotation_tolerance_rad': 2e-5}


def _trace_start(trace, call_id, start_q, side, q_tolerance=1e-8):
    # Planning happens BEFORE dense_trace increments the counter. Filename N
    # therefore normally executes in trace dense_call N+1, not N.
    first_exec = next((i for i, row in enumerate(trace)
        if row.get('phase', {}).get('name') == 'after_grasp'
        and row.get('phase', {}).get('dense_call') == call_id+1), None)
    if first_exec is None:
        matches = [i for i, row in enumerate(trace)
            if row.get('phase', {}).get('dense_call', -1) <= call_id
            and np.max(np.abs(np.asarray(row.get('arm_qpos', {}).get(side, [np.inf]*6))-start_q)) <= q_tolerance]
        index = matches[-1] if matches else None
        rule = 'Last matching measured row before planning call; no execution boundary found'
    else:
        index = first_exec - 1 if first_exec > 0 else None
        rule = 'Trace row immediately before first execution dense_call N+1'
    if index is None:
        raise ValueError('No measured pre-planning state found')
    row = trace[index]
    error = float(np.max(np.abs(np.asarray(row['arm_qpos'][side], dtype=float)-start_q)))
    if error > q_tolerance:
        raise ValueError('Pre-execution arm q differs from recorded planning start: '+str(error))
    return index, row, first_exec, rule, error


def audit_episode_provenance(episode_dir, *, executed_source_path=None, dt=.004, side='left'):
    """Read completed episode files and return a provenance report, without writes.

    Simulation timestamps are step*dt. No wall-clock timestamps are inferred.
    All errors fail provenance; passing does not certify physical task success.
    """
    directory = Path(episode_dir)
    if side not in ['left', 'right'] or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid side or timestep')
    files = sorted(directory.glob('planned_carry_path_*.npz'),
                   key=lambda p: int(re.search(r'_(\d+)\.npz$', p.name).group(1)))
    report = _load_json(directory/'attempt_report.json')
    if report.get('result') == 'running':
        raise ValueError('Episode is still running; provenance must use final files')
    trace = _load_json(directory/'physics_trace.json')
    if not isinstance(trace, list) or not trace:
        raise ValueError('Expected nonempty full physics trace list')
    steps = [row['step'] for row in trace]
    if any(b <= a for a,b in zip(steps, steps[1:])):
        raise ValueError('Trace steps must increase strictly')
    names = report.get('full_qpos_joint_names', [])
    if len(names) != 36 or len(set(names)) != 36:
        raise ValueError('Expected recorded full 36-DOF unique joint order')
    source = review_source_dataflow(executed_source_path) if executed_source_path else {'passed': False, 'reason': 'No execution source supplied'}
    initial_raw = report.get('pre_retreat_grip', {}).get('actual', {}).get('relative_bottle_palm')
    initial = _matrix(initial_raw, 'initial_grasp_reference') if initial_raw is not None else None
    out = {'schema_version': 1, 'scope': 'Independent offline artifact/source reference provenance; no scene or task-success certification',
           'episode_dir': str(directory.resolve()), 'episode_result': report.get('result'), 'dt_s': dt,
           'source_dataflow': source, 'trace_sha256': _sha(directory/'physics_trace.json'),
           'attempt_report_sha256': _sha(directory/'attempt_report.json'), 'full_qpos_joint_names': names,
           'initial_grasp_reference': initial.tolist() if initial is not None else None,
           'timestamp_basis': 'Simulation step*dt; no wall-clock timestamp was recorded',
           'wrench_numerical_recomputation_performed': False,
           'wrench_missing_recomputation_inputs': ['palm FK schedule','cmass_local_pose','mass','principal inertia','reference in wrench NPZ'],
           'segments': [], 'errors': []}
    for path in files:
        n = int(re.search(r'_(\d+)\.npz$', path.name).group(1))
        segment = {'planning_call_id': n, 'expected_execution_dense_call': n+1, 'errors': []}
        out['segments'].append(segment)
        try:
            audit_path = directory/f'carry_mesh_path_{n}.json'
            planning_path = directory/f'carry_planning_{n}.json'
            wrench_path = directory/f'planned_wrench_{n}.npz'
            lp_path = directory/f'carry_wrench_preflight_{n}.json'
            audit, planning, lp = _load_json(audit_path), _load_json(planning_path), _load_json(lp_path)
            with np.load(path, allow_pickle=False) as z:
                position = _finite_array(z['position'], (6,), 'planned position')
                velocity = _finite_array(z['velocity'], (6,), 'planned velocity')
                reference = _matrix(z['planning_reference'], 'path planning_reference')
            if velocity.shape != position.shape: raise ValueError('Velocity shape does not match position')
            checks = {'audit': _reference_check(reference, _matrix(audit['metadata']['rigid_carry_reference'], 'audit ref')),
                      'planning': _reference_check(reference, _matrix(planning['full_mesh_metadata']['rigid_carry_reference'], 'planning ref')),
                      'report': _reference_check(reference, _matrix(report['segment_planning_references'][str(n)], 'report ref'))}
            if not all(x['passed'] for x in checks.values()): raise ValueError('Artifact frozen references disagree: '+str(checks))
            start_q = np.asarray(planning['actual_start_arm_joint_rad'], dtype=float)
            index, row, execution, rule, q_error = _trace_start(trace, n, start_q, side)
            full_q = np.asarray(row['full_qpos_actual_rad'], dtype=float)
            if full_q.shape != (36,) or not np.isfinite(full_q).all(): raise ValueError('Invalid measured full qpos')
            indices = [names.index(name) for name in planning['move_group_joint_names']]
            if not np.allclose(full_q[indices], start_q, atol=1e-8, rtol=0): raise ValueError('Full DOF order differs from planning arm order')
            if not np.allclose(position[0], start_q, atol=1e-8, rtol=0): raise ValueError('Command path does not begin at planning start')
            root = _matrix(planning['world_from_planning_base'], 'root world transform')
            anchor = _matrix(planning['attachment_frame_validation']['anchor_matrix_base'], 'attached anchor')
            predicted_bottle = _matrix(planning['attachment_frame_validation']['bottle_matrix_base'], 'attached bottle')
            palm_world, attached_bottle_world = root @ anchor, root @ predicted_bottle
            measured_bottle = _matrix(row['bottle']['pose'], 'trace bottle')
            bottle_error = _pose_error(measured_bottle, attached_bottle_world)
            reconstructed_reference = np.linalg.inv(palm_world) @ measured_bottle
            reconstructed_error = _pose_error(reference, reconstructed_reference)
            if bottle_error[0] > 2e-5 or bottle_error[1] > 2e-4: raise ValueError('Trace bottle does not match recorded planning attachment')
            if reconstructed_error[0] > 2e-5 or reconstructed_error[1] > 2e-4: raise ValueError('Trace and recorded palm FK do not reconstruct saved reference')
            with np.load(wrench_path, allow_pickle=False) as z:
                force = _finite_array(z['force_target_world_N'], (3,), 'saved force')
                torque = _finite_array(z['torque_target_world_Nm'], (3,), 'saved torque')
            if len(force) != len(position) or len(torque) != len(position): raise ValueError('Saved wrench length differs from path')
            if not lp['passed'] or lp['sample_stride'] != 1 or lp['checked_count'] != len(position) or lp['checked_indices'] != list(range(len(position))):
                raise ValueError('LP does not cover every saved wrench/path row')
            peak_checks = {}
            for key, array in [('peak_force_target_N',force),('peak_torque_target_Nm',torque)]:
                peak = lp[key]; i = int(peak['sample_index'])
                if not 0 <= i < len(array): raise ValueError('LP peak index out of range')
                e = float(np.max(np.abs(array[i]-np.asarray(peak['vector'],dtype=float))))
                te = abs(float(peak['timestamp_s'])-i*dt)
                ne = abs(float(peak['norm'])-float(np.linalg.norm(array[i])))
                if e > 1e-10 or te > 1e-9 or ne > 1e-10: raise ValueError('Saved wrench does not match recorded LP peak '+key)
                peak_checks[key] = {'vector_max_error': e, 'norm_error': ne, 'timestamp_error_s': te}
            if not audit['passed'] or not planning.get('physical_state_unchanged') or not audit.get('physical_state_unchanged'):
                raise ValueError('Geometry preflight did not pass with unchanged physical state')
            end_indices = [i for i,r in enumerate(trace) if r.get('phase',{}).get('name') == 'after_grasp'
                and r.get('phase',{}).get('dense_call') == n+1]
            end_step = trace[end_indices[-1]]['step'] if end_indices else None
            segment.update(segment_reference=reference.tolist(), artifact_reference_checks=checks,
                start_trace_index=index, start_step=row['step'], start_sim_timestamp_s=row['step']*dt,
                start_phase=row['phase'], measured_start_full_qpos_rad=full_q.tolist(), measured_start_arm_max_error_rad=q_error,
                start_palm_world_from_independent_planning_FK=palm_world.tolist(), measured_start_bottle_world=measured_bottle.tolist(),
                attachment_bottle_vs_trace_error_m_rad=list(bottle_error), reference_reconstruction_error_m_rad=list(reconstructed_error),
                trace_start_rule=rule, execution_first_step=trace[execution]['step'] if execution is not None else None,
                execution_last_step=end_step, commanded_path_rows=len(position), commanded_duration_s=(len(position)-1)*dt,
                initial_reference_delta_m_rad=list(_pose_error(initial,reference)) if initial is not None else None,
                wrench_lp_peak_array_checks=peak_checks, wrench_reference_binding='AST/source dataflow only; numerical recomputation inputs missing',
                files={p.name:_sha(p) for p in [path,audit_path,planning_path,wrench_path,lp_path]}, passed=True)
        except (ValueError, KeyError, IndexError, OSError) as exc:
            segment['errors'].append(str(exc)); segment['passed'] = False
    if not files: out['errors'].append('No complete planned carry path artifacts')
    if not source['passed']: out['errors'].append('Recorded source reference dataflow was not verified')
    out['passed'] = bool(files and source['passed'] and not out['errors'] and all(s['passed'] for s in out['segments']))
    return out


def write_provenance(episode_dir, *, output_path=None, **kwargs):
    """Explicitly create one new independent JSON file; never overwrite evidence."""
    result = audit_episode_provenance(episode_dir, **kwargs)
    path = Path(output_path) if output_path is not None else Path(episode_dir)/'segment_reference_provenance.json'
    with path.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return result


def _cpu_tests():
    import tempfile
    checks = []
    trace = [dict(step=1,phase={'name':'close_hold','dense_call':4},arm_qpos={'left':[0.]*6}),
             dict(step=2,phase={'name':'after_grasp','dense_call':5},arm_qpos={'left':[.1]*6})]
    i,row,execution,rule,e = _trace_start(trace,4,np.zeros(6),'left')
    assert i == 0 and execution == 1 and e == 0; checks.append('Planning filename N uses measured trace BEFORE execution dense_call N+1')
    try:_trace_start(trace,4,np.ones(6),'left')
    except ValueError:checks.append('Mismatched measured start rejected')
    else:raise AssertionError('Bad start accepted')
    a=np.eye(4);b=a.copy();b[0,3]=.001
    assert not _reference_check(a,b)['passed'];checks.append('Stale segment reference rejected')
    valid='''def precise_path(self):
 planning_reference=np.asarray(grip_measure(task)['relative_bottle_palm'])
 res=ncp.plan_carry_to_task_pose(task,pose,palm_to_bottle=planning_reference)
 np.savez(path,planning_reference=planning_reference)
 schedule=planned_carry_wrench(palms,planning_reference,endpoint_mode='hold')
 np.savez(wrench,force_target_world_N=schedule['force_target_world_N'],torque_target_world_Nm=schedule['torque_target_world_Nm'])
'''
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'source.py';p.write_text(valid)
        assert review_source_dataflow(p)['passed'];checks.append('Same reference source dataflow passes')
        p.write_text(valid.replace('palms,planning_reference','palms,old_reference'))
        assert not review_source_dataflow(p)['passed'];checks.append('Wrench using old reference rejected')
    return {'passed':True,'checks':checks}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',nargs='?');parser.add_argument('--source');parser.add_argument('--output')
    parser.add_argument('--dt',type=float,default=.004);parser.add_argument('--tests',action='store_true')
    parser.add_argument('--write',action='store_true',help='Create independent output once; no overwrite')
    args=parser.parse_args()
    if args.tests: result=_cpu_tests()
    elif args.episode:
        fn=write_provenance if args.write else audit_episode_provenance
        kwargs={'executed_source_path':args.source,'dt':args.dt}
        if args.write:kwargs['output_path']=args.output
        result=fn(args.episode,**kwargs)
    else:parser.error('Provide an episode directory or --tests')
    print(json.dumps(result if args.tests else {'passed':result['passed'],'episode_result':result['episode_result'],
        'segments':[{k:s.get(k) for k in ['planning_call_id','passed','start_step','execution_first_step','execution_last_step','errors']} for s in result['segments']],
        'errors':result['errors'],'wrench_binding':'source dataflow only; no independent numerical recomputation'},indent=2))
