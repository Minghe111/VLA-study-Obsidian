"""Worker-ready read-only official place-alignment endpoint probes.
Only existing CuRobo IK is queried. No scene/drive/qpos/solver configuration writes.
The official target, FP0 and upright requirement remain unchanged.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import time
from pathlib import Path
import numpy as np
import preflight_place_goals as ppg

ALIGN_AXES=(('x_plus',[1.,0.,0.]),('x_minus',[-1.,0.,0.]),
            ('y_plus',[0.,1.,0.]),('y_minus',[0.,-1.,0.]))

def _angle(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    return math.degrees(math.acos(float(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1.,1.))))

def alignment_cases(task,g,history,*,side='left',place_function=None):
    base=ppg.build_place_goals(task,g,bottle_at_grasp_pose=history,side=side,place_function=place_function)
    carry=ppg._matrix(base['hypothetical_retreat_task_pose_world'])
    bottle=ppg.ReadOnlyActorProxy(task.bottle,base['hypothetical_retreat_bottle_matrix_world'])
    fp=ppg._matrix(bottle.get_functional_point(0,'matrix'))
    actors=(('carry_ee_camera_z',carry[:3,2]),('fp0_minus_carry_ee',fp[:3,3]-carry[:3,3]))
    cases=[]
    for label,axis in actors:
        if not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9 or np.linalg.norm(axis[:2])<1e-9:
            raise ValueError('Official world-axis XY projection is degenerate: '+label)
        for name,align in ALIGN_AXES:
            cases.append({'label':label+'__'+name,'place_kwargs':{'constrain':'align',
                'actor_axis':axis.tolist(),'actor_axis_type':'world','align_axis':align.copy()}})
    return cases

def _frames(robot,side):
    planner=getattr(robot,side+'_planner');art=getattr(robot,side+'_entity')
    result={'curobo_robot_origin_pose_world':ppg._vector(planner.robot_origion_pose),
            'curobo_frame_bias_base_m':np.asarray(planner.frame_bias).tolist(),
            'transform':'p_base=R_origin.T@(p_world-p_origin)+frame_bias; R_base=R_origin.T@R_world'}
    try:result['actual_articulation_root_world']=ppg._vector(art.root_pose)
    except Exception as e:result['actual_root_unavailable']=str(e)
    joint=getattr(robot,side+'_arm_joints')[0]
    try:
        parent=ppg._matrix(joint.parent_link.entity.pose)
        result['first_arm_joint_parent_link']=joint.parent_link.name
        result['first_arm_joint_parent_pose_world']=ppg._vector(parent)
        result['distance_reference_world_m']=parent[:3,3].tolist()
        result['distance_reference_kind']='first_arm_joint_parent_link_origin'
        for attr in ('pose_in_parent','get_pose_in_parent'):
            if hasattr(joint,attr):
                v=getattr(joint,attr);v=v() if callable(v) else v
                origin=parent@ppg._matrix(v)
                result['first_arm_joint_frame_world']=ppg._vector(origin)
                result['distance_reference_world_m']=origin[:3,3].tolist()
                result['distance_reference_kind']='first_arm_joint_origin_from_actual_parent_and_joint_pose'
                break
    except Exception as e:result['first_joint_reference_unavailable']=str(e)
    return result

def _geometry(report,frames):
    target=ppg._matrix(report['target_fp0_pose_world']);checks={}
    for label,goal in report['goals'].items():
        fp=ppg._matrix(goal['hypothetical_fp0_goal_world'])
        bm=ppg._matrix(goal['hypothetical_bottle_goal_matrix_world'])
        angle=_angle(fp[:3,2],target[:3,2]);err=goal['fp0_target_position_error_m']
        expected=goal['pre_dis_m']
        entry={'fp0_z_axis_error_deg':angle,'bottle_local_y_to_world_up_deg':_angle(bm[:3,1],[0,0,1]),
               'fp0_position_offset_m':err,'expected_pre_dis_offset_m':expected,
               'fp0_geometry_passed':bool(angle<.001 and abs(err-expected)<1e-5 and goal['official_position_success_if_rigid_hold'])}
        ref=frames.get('distance_reference_world_m')
        if ref is not None:entry['palm_distance_from_first_joint_reference_m']=float(np.linalg.norm(np.asarray(goal['palm_goal_world'][:3])-ref))
        origin=np.asarray(frames['curobo_robot_origin_pose_world'][:3])
        entry['palm_distance_from_curobo_origin_m']=float(np.linalg.norm(np.asarray(goal['palm_goal_world'][:3])-origin))
        checks[label]=entry
    return checks

def probe_place_alignments(task,g,*,bottle_at_grasp_pose,side='left',run_ik=False,place_function=None):
    before=ppg._snapshot(task);started=time.monotonic();frames=_frames(task.robot,side)
    _,names,limits,arms,ids=ppg._limits(task.robot,side)
    full_start=ppg._array(getattr(task.robot,side+'_entity').get_qpos()).copy()
    reports=[]
    try:
        cases=alignment_cases(task,g,bottle_at_grasp_pose,side=side,place_function=place_function)
        for case in cases:
            kwargs=case['place_kwargs']
            report=ppg.preflight_place_goals(task,g,bottle_at_grasp_pose=bottle_at_grasp_pose,side=side,
                place_kwargs=kwargs,run_planning=False,start_qpos=full_start,place_function=place_function)
            checks=_geometry(report,frames);eligible=all(x['fp0_geometry_passed'] for x in checks.values())
            if run_ik and eligible:
                report=ppg.preflight_place_goals(task,g,bottle_at_grasp_pose=bottle_at_grasp_pose,side=side,
                    place_kwargs=kwargs,run_planning=True,planner_modes=('curobo_ik',),start_qpos=full_start,place_function=place_function)
            feasible=eligible and run_ik and all(
                goal.get('planning_probes',{}).get('curobo_ik',{}).get('status')=='Success' and
                goal.get('planning_probes',{}).get('curobo_ik',{}).get('command_limits_passed',False)
                for goal in report['goals'].values())
            report.update(case_label=case['label'],place_kwargs=kwargs,geometry_checks=checks,
                          eligible_for_endpoint_ik=eligible,both_endpoints_ik_and_command_limits_passed=bool(feasible))
            reports.append(report)
            compact={k:{'palm_world':v['palm_goal_world'],
                'fp0_z_error_deg':checks[k]['fp0_z_axis_error_deg'],
                'status':v.get('planning_probes',{}).get('curobo_ik',{}).get('status','GeometryOnly'),
                'position_error_m':v.get('planning_probes',{}).get('curobo_ik',{}).get('position_error_m'),
                'command_limits_passed':v.get('planning_probes',{}).get('curobo_ik',{}).get('command_limits_passed')}
                for k,v in report['goals'].items()}
            print('PLACE_ALIGNMENT',case['label'],json.dumps(compact),flush=True)
            ppg._assert_snapshot(task,before)
        return {'scope':'Read-only official alignment kwargs. Target FP0 pose fixed, no action/path execution; endpoint reachability alone does not prove carry safety.',
            'frame_audit':frames,'arm_limits_rad':{n:limits[i].tolist() for n,i in zip(arms,ids)},
            'command_limit_tolerance_rad':1e-5,'actual_start_arm_joint_rad':full_start[ids].tolist(),
            'target_fp0_pose_world':list(ppg.OFFICIAL_LEFT_TARGET if side=='left' else ppg.OFFICIAL_RIGHT_TARGET),
            'cases':reports,'feasible_case_labels':[r['case_label'] for r in reports if r['both_endpoints_ik_and_command_limits_passed']],
            'endpoint_ik_calls':sum(2 for r in reports if run_ik and r['eligible_for_endpoint_ik']),
            'physical_state_unchanged':True,'elapsed_s':time.monotonic()-started}
    finally:ppg._assert_snapshot(task,before)

def run_worker_probe(task,base,grasp_function,grasp_setting):
    importlib.reload(ppg)
    base=Path(base);tracefile=base/'coupled_side_entry_v8_B085_20261003_seed7/physics_trace.json'
    trace=json.loads(tracefile.read_text());rows=trace if isinstance(trace,list) else trace['samples']
    first=next(x for x in rows if x.get('phase',{}).get('name')=='close')
    settings=dict(grasp_setting)
    for key,expected in (('pitch_deg',10.),('hand_y',.085),('hand_z',.165)):
        if abs(float(settings[key])-expected)>1e-9:raise ValueError('Expected historical B085 grasp geometry '+key)
    history=first['bottle']['pose'];actor=ppg.ReadOnlyActorProxy(task.bottle,history)
    fp=actor.get_functional_point(0,'list')
    point_error=float(np.linalg.norm(np.asarray(fp[:3])-first['bottle']['functional_point0'][:3]))
    if point_error>2e-6:raise ValueError('Historical/current official FP0 scaled geometry mismatch '+str(point_error))
    _,g=grasp_function(task,actor,'left',pre_dis=.1,target_dis=0)
    report=probe_place_alignments(task,g,bottle_at_grasp_pose=history,side='left',run_ik=True)
    report.update(reference_trace=str(tracefile),reference_trace_sha256=hashlib.sha256(tracefile.read_bytes()).hexdigest(),
        reference_trace_step=first['step'],reference_bottle_model_id=first['bottle']['model_id'],
        reference_bottle_qpose_tag=first['bottle']['qpose_tag'],grasp_setting=settings,
        historical_fp0_geometry_error_m=point_error,
        helper_sha256=hashlib.sha256(Path(ppg.__file__).read_bytes()).hexdigest())
    output=base/'b085_place_alignment_preflight_20261003.json'
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print('PLACE_ALIGNMENT_REPORT',str(output),'feasible',report['feasible_case_labels'],'calls',report['endpoint_ik_calls'],flush=True)
    return report

if 'task' in globals() and 'side_entry_grasp' in globals():
    place_alignment_report=run_worker_probe(task,BASE,side_entry_grasp,grasp_setting)
elif __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Pure syntax/import check; worker entry requires existing task/solver')
    parser.add_argument('--official-repo');args=parser.parse_args()
    function,source=ppg.load_official_cpu_function(args.official_repo) if args.official_repo else (None,{})
    result=ppg.run_cpu_tests(function);result.update(source);print(json.dumps(result,indent=2))
