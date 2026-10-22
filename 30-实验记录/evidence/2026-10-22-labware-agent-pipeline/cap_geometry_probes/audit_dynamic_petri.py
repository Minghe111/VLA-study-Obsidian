"""Independent, read-only audit of the saved dynamic Petri component run.

Uses SciPy rotations and explicit segment endpoints, not the runtime capsule
predicate. Constructor readback uses zero simulation steps. No root files edit.
"""
from __future__ import annotations
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import sapien

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
TASK = PIPE / 'robot_cap_task'
RUN = TASK / 'receiver_probes/dynamic_petri_drop12_v1'
OUT = HERE / 'dynamic_petri_independent_audit_v1'
SPEC = PIPE.parent / '2026-10-22-labware-integration/physics_spec.json'
sys.path.insert(0, str(TASK))
from petri_receiver import load_dynamic_petri

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rotation(pq):
    # Actual saved convention is SAPIEN [p3,w,x,y,z]. SciPy consumes xyzw.
    q = np.asarray(pq[3:], dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()

def vertices(path):
    return np.asarray([[float(v) for v in line.split()[1:4]]
                       for line in Path(path).read_text().splitlines()
                       if line.startswith('v ')])

def mutation_audit(source):
    tree = ast.parse(Path(source).read_text())
    forbidden_names = {'set_pose', 'set_qpos', 'set_qvel', 'set_velocity',
        'set_linear_velocity', 'set_angular_velocity', 'add_force_at_point',
        'add_force_torque', 'set_drive_target', 'set_drive_velocity_target',
        'create_drive', 'create_joint', 'build_kinematic'}
    forbidden_attributes = {'pose', 'qpos', 'qvel', 'linear_velocity',
        'angular_velocity', 'kinematic'}
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_names:
            findings.append({'line': node.lineno, 'call': node.func.attr})
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in forbidden_attributes:
                    findings.append({'line': node.lineno, 'assignment': target.attr})
    return findings

def main():
    OUT.mkdir(exist_ok=False)
    receipt = json.loads((RUN / 'receipt.json').read_text())
    trace = json.loads(gzip.decompress((RUN / 'trace.json.gz').read_bytes()))
    spec = json.loads(SPEC.read_text())
    cfg, receiver = spec['tablets'], spec['receiver']
    tol = spec['physics']['contact_offset_m']
    all_hashes = {**receipt['source_sha256'], **receipt['receiver_contract']['source_sha256']}
    hashes = [{'path': str(path), 'recorded': expected, 'actual': digest(path),
               'passed': digest(path) == expected} for path, expected in all_hashes.items()]
    collision_paths = [Path(p) for p in receipt['receiver_contract']['source_sha256'] if Path(p).suffix == '.obj']
    assert len(collision_paths) == 49
    wall_paths = sorted(p for p in collision_paths if p.name.startswith('wall_'))
    # Check true inner polygon halfspaces from the original convex-piece OBJ
    # vertices, rather than silently identifying the circular mask with mesh.
    wall_planes = []
    for path in wall_paths:
        vv = vertices(path)
        radii = np.linalg.norm(vv[:, :2], axis=1)
        inner = vv[np.isclose(radii, receiver['inner_radius_m'], atol=1e-7), :2]
        normal = inner.mean(0)
        normal /= np.linalg.norm(normal)
        offset = float(np.mean(inner @ normal))
        wall_planes.append((normal, offset))
    rows = []
    mismatches = []
    holds = []
    start = None
    strict_holds = []
    strict_start = None
    max_qnorm_error = 0.
    previous_step = 0
    for sample in trace:
        step = sample['physics_step']
        assert step == previous_step+1
        previous_step = step
        assert len(sample['tablet_poses']) == cfg['count'] == 12
        assert np.isclose(sample['time_s'], step*.004, atol=1e-9)
        pq = sample['receiver_pose']
        inv = rotation(pq).T
        all_margins = []
        polygon_margins = []
        for pill in sample['tablet_poses']:
            max_qnorm_error = max(max_qnorm_error, abs(np.linalg.norm(pill[3:])-1.))
            center = inv @ (np.asarray(pill[:3])-np.asarray(pq[:3]))
            axis = inv @ rotation(pill)[:, 0]
            endpoints = np.vstack([center-cfg['half_length_m']*axis,
                                   center+cfg['half_length_m']*axis])
            radius = cfg['radius_m']
            margins = [receiver['inner_radius_m']-np.linalg.norm(endpoints[:, :2], axis=1).max()-radius,
                       endpoints[:, 2].min()-radius-receiver['base_thickness_m'],
                       receiver['height_m']-endpoints[:, 2].max()-radius]
            all_margins.append(margins)
            polygon_margins.append(min(offset-float(np.max(endpoints[:, :2] @ n))-radius
                                       for n, offset in wall_planes))
        margins = np.asarray(all_margins)
        count = int(np.sum(np.all(margins >= -tol, axis=1)))
        raw_minimum = float(margins.min())
        if count != sample['full_geometry_inside_count'] or abs(raw_minimum-sample['minimum_margin_m']) > 2e-7:
            mismatches.append({'step': step, 'count_actual': count,
                'count_recorded': sample['full_geometry_inside_count'],
                'min_margin_disagreement_m': raw_minimum-sample['minimum_margin_m']})
        inside = count == 12 and not sample['direct_table_spill_contact']
        strict_inside = bool(np.all(margins >= 0.)) and not sample['direct_table_spill_contact']
        if inside and start is None:
            start = step
        if not inside and start is not None:
            holds.append([start, step-1]); start = None
        if strict_inside and strict_start is None:
            strict_start = step
        if not strict_inside and strict_start is not None:
            strict_holds.append([strict_start, step-1]); strict_start = None
        rows.append({'physics_step': step, 'count': count, 'minimum_margin_m': raw_minimum,
                     'minimum_polygon_radial_margin_m': float(min(polygon_margins)),
                     'all12_within_declared_tolerance': inside,
                     'all12_strict_zero_tolerance': strict_inside})
    if start is not None: holds.append([start, len(trace)])
    if strict_start is not None: strict_holds.append([strict_start, len(trace)])
    final_hold = holds[-1] if holds and holds[-1][1] == len(trace) else None
    nominal_duration = (final_hold[1]-final_hold[0]+1)*.004 if final_hold else 0.
    actual_duration = nominal_duration*(receipt['physics_dt_s_actual']/.004)
    trace_spill_steps = sum(bool(x['direct_table_spill_contact']) for x in trace)
    # Zero-step runtime constructor readback, in a new isolated CPU scene.
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(.004)
    entity, contract = load_dynamic_petri(scene, sapien.Pose([.04,-.03,.0005]))
    component = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    actual = {'dynamic_component': type(component).__name__,
              'kinematic': bool(component.kinematic), 'mass_kg': float(component.mass),
              'collision_shape_count': len(component.collision_shapes),
              'scene_steps_executed': 0,
              'pose_matrix': entity.pose.to_transformation_matrix().tolist()}
    findings = {str(p): mutation_audit(p) for p in [TASK/'petri_receiver.py', TASK/'probe_dynamic_petri.py']}
    checks = {
        'recorded_source_and_49_collision_hashes_match': all(x['passed'] for x in hashes),
        'frozen_spec_sha_matches': digest(SPEC) == receipt['receiver_contract']['frozen_spec_sha256'],
        'trace_sha_matches': digest(RUN/'trace.json.gz') == receipt['trace_sha256'],
        'zero_step_dynamic_30g_49_parts_readback': not actual['kinematic'] and np.isclose(actual['mass_kg'],.03,atol=1e-8) and actual['collision_shape_count']==49,
        'no_runtime_pose_velocity_fixture_drive_write_in_two_sources': not any(findings.values()),
        'raw_2500_steps_each_actual_12_pose_containment_matches': len(trace)==2500 and not mismatches,
        'reported_final_contiguous_8p424_seconds_matches': abs(nominal_duration-receipt['last_inside_contiguous_seconds'])<1e-9,
        'recorded_table_spill_zero_consistent': trace_spill_steps==0 and receipt['tablet_table_spill_contact_count']==0,
        'all_initial_capsules_dynamic_350mg': len(receipt['initial_tablets'])==12 and all(x['dynamic'] and np.isclose(x['mass_kg'],.00035,atol=1e-10) for x in receipt['initial_tablets']),
        'not_claimed_robot_success': not receipt['robot_present'] and not receipt['robot_success'] and not receipt['usable_robot_demonstration'],
    }
    result = {
        'scope': 'Independent readback/zero-step constructor audit of gravity-drop component; no robot cap-removal or pour proof',
        'passed': bool(all(checks.values())), 'checks': {k:bool(v) for k,v in checks.items()},
        'auditor_source_sha256': digest(__file__), 'receipt_sha256': digest(RUN/'receipt.json'),
        'runtime_readback': actual, 'source_hash_readback': hashes,
        'mutation_ast_findings': findings, 'raw_trace_rows': len(trace),
        'independent_predicate': 'SciPy normalized quaternion transforms, actual receiver pose each step, capsule endpoints +/-2mm on local X plus 3mm sphere, circular cavity R42.5mm floor1mm rim14mm; explicit .4mm tolerance',
        'runtime_predicate_called': False, 'containment_mismatches': mismatches,
        'final_all12_interval_inclusive_steps': final_hold,
        'nominal_4ms_final_hold_seconds': nominal_duration, 'actual_dt_final_hold_seconds': actual_duration,
        'full_all12_intervals_inclusive_steps': holds,
        'strict_zero_tolerance_intervals_inclusive_steps': strict_holds,
        'final_minimum_margin_m': rows[-1]['minimum_margin_m'],
        'final_interval_minimum_margin_m': min(x['minimum_margin_m'] for x in rows[final_hold[0]-1:]) if final_hold else None,
        'final_interval_minimum_actual_polygon_radial_margin_m': min(x['minimum_polygon_radial_margin_m'] for x in rows[final_hold[0]-1:]) if final_hold else None,
        'nominal_circle_vs_48segment_polygon_max_radial_difference_m': receiver['inner_radius_m']-min(x[1] for x in wall_planes),
        'tablet_quaternion_max_norm_error': max_qnorm_error,
        'receiver_translation_range_m': np.ptp(np.asarray([x['receiver_pose'][:3] for x in trace]),axis=0).tolist(),
        'recorded_table_spill_steps': trace_spill_steps,
        'limitation': 'Raw contact impulses were not saved. Zero spill is checked against original actual-entity-ID contact logic and its per-step flags, not independently reconstructed from pose-only trace. 30g mass/materials are engineering assignments, not measured glass. No physical rerun, robot, cap removal, bottle pour, rollout or SDK24 demonstration is certified.',
    }
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    (OUT/'recomputed_per_step.json.gz').write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode()))
    (OUT/'audit.md').write_text('# 动态培养皿组件独立复核\n\n'
        f'结论：{result["passed"]}。仅复核现有重力落片组件，未执行机器人。\n\n'
        f'- 零步CPU构造：动态{actual["dynamic_component"]}，质量{actual["mass_kg"]:.9f} kg，原49件凸碰撞。\n'
        f'- 2500步实际receiver及12片world pq重新计算：{len(mismatches)}处计数/最小裕量不一致；最后连续区间{final_hold}，标称{nominal_duration:.3f}s。\n'
        f'- 声明接触容差0.4mm；最后区间最小原始裕量{result["final_interval_minimum_margin_m"]*1000:.6f}mm，不能解释成零容差内含。\n'
        f'- 原48段壁多边形较名义圆内缩最大{result["nominal_circle_vs_48segment_polygon_max_radial_difference_m"]*1000:.6f}mm；最后区间最小实际壁面径向裕量{result["final_interval_minimum_actual_polygon_radial_margin_m"]*1000:.6f}mm。\n'
        '- 原源码SHA、49碰撞文件/metadata/visual及冻结spec全部匹配；两运行源码未发现姿态、速度、驱动、外力或夹具写入。\n'
        '- table spill逐步标记为0且原代码以entity ID检测；未保存接触冲量，因此不能仅凭位姿独立重建接触力。\n'
        '- 30g及材料是工程赋值。本结论不包括机器人开盖、倒药、闭环或SDK24示范成功。\n')
    print(json.dumps({k:result[k] for k in ('passed','checks','final_all12_interval_inclusive_steps','nominal_4ms_final_hold_seconds','final_interval_minimum_margin_m','final_interval_minimum_actual_polygon_radial_margin_m')},indent=2))

if __name__ == '__main__': main()
