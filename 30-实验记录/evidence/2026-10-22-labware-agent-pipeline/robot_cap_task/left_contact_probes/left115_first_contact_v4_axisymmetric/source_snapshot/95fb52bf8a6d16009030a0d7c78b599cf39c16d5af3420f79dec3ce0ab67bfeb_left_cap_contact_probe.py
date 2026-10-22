"""One bounded physical left-cap first-contact diagnostic, not collection.

Replay the reviewed body-hold prefix exactly. Every NEW action is assembled
from current drive targets plus the original right grip feedback. Metadata
supplies LEFT geometry only; it never supplies right actions or physical poses.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, gzip, hashlib, inspect, json, math, shutil, sys, textwrap, time, traceback, types
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from left_cap_contact_task import left_cap_contact_task, FirstContactStop
from labware_cap115_task import PROFILE
from held_left_planner import install_held_left_planner
from cap_contact_controller import CapContactController
from camera_framing import apply_camera_framing
from capsule_geometry import read_capsule_entities
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def matrix(p):return np.asarray(p.to_transformation_matrix(),dtype=float)
def yaw_minimal_cap_goal(reference_cap,current_cap,reference_cap_from_palm):
    """Keep world grasp azimuth for an axisymmetric cap; follow centre/axis.

    This is a target calculation only, not an object or robot pose write.
    """
    z0=np.asarray(reference_cap[:3,2],float);z0=z0/np.linalg.norm(z0)
    z1=np.asarray(current_cap[:3,2],float);z1=z1/np.linalg.norm(z1)
    axis=np.cross(z0,z1);cosine=float(z0@z1)
    if cosine<-.99:raise RuntimeError('Antiparallel cap axes outside this screened binding')
    skew=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    rotation=np.eye(3)+skew+(skew@skew)/(1.+cosine)
    reference_palm=reference_cap@reference_cap_from_palm;goal=np.eye(4)
    goal[:3,:3]=rotation@reference_palm[:3,:3]
    goal[:3,3]=current_cap[:3,3]+rotation@(reference_palm[:3,3]-reference_cap[:3,3])
    return goal
def snapshot(task,native):
    art=task.robot.left_entity
    return {'captured_at_utc':datetime.now(timezone.utc).isoformat(),
      'physics_tick':task.physics_step_count,'assembly_physics_step':task.assembly.physics_steps,
      'sim_time_s':task.assembly.physics_steps*.004,
      'joint_names':[j.name for j in art.active_joints], 'q36':np.asarray(art.qpos).tolist(),
      'qvel36':np.asarray(art.qvel).tolist(),'qlimits':np.asarray(art.get_qlimits()).tolist(),
      'state24':task.get_state24().tolist(),'actual_drive_target24':native.command24(task.robot).tolist(),
      'root_matrix':matrix(art.root_pose).tolist(),
      'left_palm_matrix':matrix(art.find_link_by_name('l_inspire_base_link').entity.pose).tolist(),
      'right_palm_matrix':matrix(art.find_link_by_name('r_inspire_base_link').entity.pose).tolist(),
      'body_matrix':matrix(task.assembly.body.pose).tolist(),'cap_matrix':matrix(task.assembly.cap.pose).tolist(),
      'tablet_world_pose_pq':[np.r_[e.pose.p,e.pose.q].tolist() for e in task.assembly.tablets],
      'metrics':copy.deepcopy(task.metrics)}

def moving_path_audit(audit,path,native):
    """All moving-left/static and moving-left/all-robot pairs, no new ACM.

    Static right grasps are reported by the parent tagged baseline, but frozen
    pairs not affected by this LEFT move are outside this scoped path check.
    This is a <=1-mm sampled geometric check, not a continuous certificate.
    """
    # The held-pair method is generated privately and has no source file;
    # read the retained on-disk base method, then apply the explicit scope.
    source=textwrap.dedent(inspect.getsource(type(audit).__mro__[1].audit))
    loop='for link, obj in self.objects.items():'
    if source.count(loop)!=1:raise RuntimeError('Auditor static loop changed')
    source=source.replace(loop,loop+'\n                    if not moving[link]:\n                        continue')
    loop='for first, second in self.self_pairs:'
    if source.count(loop)!=1:raise RuntimeError('Auditor self loop changed')
    source=source.replace(loop,loop+'\n                if not moving[first] and not moving[second]:\n                    continue')
    old='thresholds = {"bottle": min_bottle_clearance_m, "self": min_self_clearance_m, "table": min_table_clearance_m}'
    if source.count(old)!=1:raise RuntimeError('Auditor threshold layout changed')
    source=source.replace(old,'thresholds = {key: (min_self_clearance_m if key == "self" else min_table_clearance_m if key == "table" else min_bottle_clearance_m) for key in categories}')
    namespace=dict(native.fma.__dict__)
    exec(compile(source,'<left-moving-fullmesh-sampled>','exec'),namespace)
    report=namespace['audit'](audit,path,'left',min_bottle_clearance_m=0.,min_self_clearance_m=0.,
        min_table_clearance_m=0.,max_samples=12000,stop_on_first_collision=True,
        deadline_monotonic=time.monotonic()+90.)
    no_collision=report.get('unique_collision_pairs')==0 and not report.get('incomplete_failed',False)
    report['scoped_left_sampled_path_passed']=bool(no_collision and report['joint_limit_audit']['passed'])
    report['scope']='All moving LEFT links against actual body, cap,12pill,table and all unexempt robot pairs; frozen-only pair scope excluded; <=1mm samples, not continuous clearance proof'
    report['continuous_certificate_required_for_this_reversible_diagnostic']=False
    if not report['scoped_left_sampled_path_passed']:raise PathRejected(report)
    return report

class PathRejected(RuntimeError):
    def __init__(self,report):self.report=report;super().__init__('Actual-scene moving-left path check rejected')

def solve_rebound(audit,qreference,goal_world,task,native):
    art=task.robot.left_entity;start=np.asarray(art.qpos,dtype=float).copy()
    indices=[audit.sim_joints.index(n) for n in task.robot.left_arm_joints_name]
    mask=np.ones(len(start),bool);mask[indices]=False
    seed=start.copy();seed[indices]=qreference
    link=list(audit.pk.get_link_names()).index('l_inspire_base_link')
    target=np.linalg.inv(matrix(art.root_pose))@goal_world
    q,ok,error=audit.pk.compute_IK_CLIK(link,audit.Pose(target),seed,mask.tolist(),eps=1e-7,max_iter=2000,dt=.1,damp=1e-8)
    q=np.asarray(q,dtype=float)
    if not ok or not np.isfinite(q).all():raise RuntimeError('Current-cap rebound IK failed')
    if np.max(np.abs(q[mask]-start[mask]))>1e-8:raise RuntimeError('Rebound IK changed other joints')
    limits=np.concatenate([np.asarray(j.limits).reshape(-1,2) for j in art.active_joints])
    if np.any(q[indices]<limits[indices,0]-1e-8) or np.any(q[indices]>limits[indices,1]+1e-8):raise RuntimeError('Rebound IK arm limits')
    audit.pk.compute_forward_kinematics(q)
    actual=matrix(art.root_pose)@np.asarray(audit.pk.get_link_pose(link).to_transformation_matrix())
    position=float(np.linalg.norm(actual[:3,3]-goal_world[:3,3]));_,rotation=native.rotation_error_so3(actual[:3,:3],goal_world[:3,:3])
    if position>5e-5 or rotation>2e-4:raise RuntimeError('Rebound IK FK residual exceeds retained tolerance')
    return q[indices],{'actual_q_start36':start.tolist(),'left_seed6':np.asarray(qreference).tolist(),
      'goal_palm_world':goal_world.tolist(),'actual_cap_world':matrix(task.assembly.cap.pose).tolist(),
      'left_result6':q[indices].tolist(),'position_error_m':position,'rotation_error_rad':rotation}

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--rebind',type=Path,required=True)
    p.add_argument('--resume-final',action='store_true',help='Replay retained505 targets, then the single final approach')
    p.add_argument('--axisymmetric-binding',action='store_true',help='Follow actualcap centre/axis, retaining world grasp azimuth')
    args=p.parse_args()
    out=ROOT/'left_contact_probes'/args.name;out.mkdir(parents=True,exist_ok=False)
    prefix=json.loads((args.source/'transitions.json').read_text())
    expected_prefix=505 if args.resume_final else 294
    if len(prefix)!=expected_prefix:raise RuntimeError(f'Expect reviewed{expected_prefix}-target prefix')
    plan=json.loads(args.plan.read_text());waypoints=np.asarray(plan['raw_arm6_waypoints'],float)
    reference=json.loads(Path(plan['snapshot']).read_text());refcap=np.asarray(reference['cap_matrix'])
    rebind=json.loads(args.rebind.read_text())
    if waypoints.ndim!=2 or waypoints.shape[1]!=6 or not 2<=len(waypoints)<=10:raise RuntimeError('Expected bounded shortcut LEFT6 path')
    sources=[Path(__file__),ROOT/'left_cap_contact_task.py',ROOT/'held_left_planner.py',ROOT/'cap_contact_controller.py',
      ROOT/'labware_cap_workflow.py',ROOT/'labware_cap115_task.py',ROOT/'labware_cap_task.py',ROOT/'capsule_geometry.py',
      ROOT.parent/'labware_task.py',ROOT.parent/'robot_expert.py',ROOT.parent/'robot_cap_assembly.py',
      args.plan,args.rebind,Path(plan['snapshot']),args.source/'transitions.json',*[x for x in PROFILE.iterdir() if x.is_file()]]
    fingerprints={str(x.resolve()):sha(x) for x in sources};dump(out/'source_manifest.json',fingerprints)
    snapdir=out/'source_snapshot';snapdir.mkdir();snaps={}
    for path,digest in fingerprints.items():
        dest=snapdir/(digest+'_'+Path(path).name);shutil.copyfile(path,dest);snaps[path]={'snapshot':str(dest),'sha256':digest}
    dump(out/'snapshot_manifest.json',snaps)
    native=install_held_left_planner();task=left_cap_contact_task();error=None;partial=None;rejected=None
    shadow=[];prefix_pass=[];planning=[];proposals=[];rebindings=[];fixed_right=None;event=False
    current_command=None;current_before=None;normalized=[]
    try:
        task.setup_demo(seed=0,runtime_path=out,capture_rgb=False,center_xy=[.10,-.11]);apply_camera_framing(task)
        setting={'open_sdk':[0.,0.,800.,800.,800.,0.],'close_sdk':[0.,0.,608.,610.,610.,0.],
            'pitch_deg':10.,'hand_x':-.015,'hand_y':.076,'hand_z':.177,'palm_roll_deg':5.,'approach_azimuth_deg':0.}
        feedback=native.BoundaryContactController(task,'right',setting)
        for i,row in enumerate(prefix):
            if not np.array_equal(task.get_state24(),np.asarray(row['state'],np.float32)):raise RuntimeError(f'Prefix state differs{i}')
            if row['phase'] in ('close','lift_plan','lift','hold','right_fixed_arm_same_grip_feedback') or row['phase'].startswith('left_'):
                hand=feedback.target(carry=row['phase'] not in ('close','lift_plan'))
                diff=float(np.max(np.abs(hand-np.asarray(row['action'][18:24]))));shadow.append(diff)
                if diff>1e-8:raise RuntimeError(f'Feedback shadow differs{i}: {diff}')
            task.execute_command24(row['action'],row['phase'])
            if not np.array_equal(task.get_state24(),np.asarray(row['next_state'],np.float32)):raise RuntimeError(f'Prefix nextstate differs{i}')
            prefix_pass.append(True)
        dump(out/'actual_replay_terminal_snapshot.json',snapshot(task,native));print(f'PREFIX{expected_prefix}_EXACT',flush=True)
        fixed_right=native.command24(task.robot)[12:18].copy();task.activate_left_probe()
        def execute(left_arm=None,left_sdk=None,phase='left_approach'):
            nonlocal current_command,current_before
            target=native.command24(task.robot)
            target[12:18]=fixed_right;target[18:24]=feedback.target(carry=True)
            if left_arm is not None:target[:6]=left_arm
            if left_sdk is not None:target[6:12]=left_sdk
            current_before=snapshot(task,native);current_command={'phase':phase,'requested_target24':target.tolist(),
                'start_tick':task.physics_step_count,'before':current_before}
            task.execute_command24(target,phase)
            current_command=None;current_before=None
        mapping=native.ContinuousHandMapping(task.robot._manifest['hands']['left'])
        raw=native.command24(task.robot)[6:12];start=raw.copy()
        for i,v in enumerate(start):
            if v<0. or v>1000.:
                if v<-.001 or v>1000.001:raise RuntimeError('Actual left drive outside SDK interval')
                start[i]=float(np.clip(v,0,1000));normalized.append({'channel':i,'actual_drive_sdk':float(v),'profile_endpoint':float(start[i]),'scope':'Only endpoint float representation <=.001 SDK; actual state not written'})
        opened=np.array([0.,0.,0.,1000.,1000.,400.])
        profile=native.SDKWaypointProfile(mapping,[start,opened],dt=.06,velocity_fraction=1.)
        dump(out/'left_preparation_profile.json',profile.metadata());dump(out/'sdk_endpoint_representation.json',normalized)
        if profile.num_steps>100:raise RuntimeError('Finite home preparation budget exceeded')
        if not args.resume_final:
            for index in range(1,profile.num_steps):execute(left_sdk=profile.sample(index)['sdk6'],phase='left_prepare_open400')
        dump(out/'actual_preapproach_snapshot.json',snapshot(task,native))
        # Transit nodes belong to the table/world obstacle corridor. Do not
        # rotate that distant corridor with the moving cap. Only the final
        # cap-approach endpoint follows the currently measured cap frame.
        base_audit=native.fma.FullMeshPathAuditor(task)
        indices=[base_audit.sim_joints.index(n) for n in task.robot.left_arm_joints_name]
        palm_index=list(base_audit.pk.get_link_names()).index('l_inspire_base_link')
        relatives=[];world_palms=[]
        for qleft in waypoints[1:]:
            q=base_audit.initial_sim_q.copy();q[indices]=qleft;base_audit.pk.compute_forward_kinematics(q)
            palm=matrix(task.robot.left_entity.root_pose)@np.asarray(base_audit.pk.get_link_pose(palm_index).to_transformation_matrix())
            world_palms.append(palm);relatives.append(np.linalg.inv(refcap)@palm)
        base_audit.model.set_qpos(base_audit.old_model_q,full=True)
        # The screened endpoint is exactly its measured cap-relative frame.
        key='T_cap_from_palm'
        if key not in rebind:
            matching=[k for k in rebind if 'cap' in k and 'palm' in k]
            if len(matching)!=1:raise RuntimeError(f'No unique cap-relative palm matrix: {matching}')
            key=matching[0]
        relatives[-1]=np.asarray(rebind[key],float)
        if relatives[-1].shape!=(4,4):raise RuntimeError('Cap-relative endpoint must be4x4')
        for index,(qreference,relative) in enumerate(zip(waypoints[1:],relatives)):
            if args.resume_final and index<len(relatives)-1:continue
            print(json.dumps({'stage':'rebind_and_audit','via':index,'tick':task.physics_step_count}),flush=True)
            actual_before=snapshot(task,native);audit=native.fma.FullMeshPathAuditor(task)
            if index<len(relatives)-1:
                endpoint=np.asarray(qreference,float).copy()
                receipt={'frame':'World/table transit corridor retained; no cap-follow rotation',
                    'actual_q_start36':np.asarray(task.robot.left_entity.qpos).tolist(),
                    'left_result6':endpoint.tolist(),'world_palm_reference':world_palms[index].tolist(),
                    'actual_cap_world_observed':matrix(task.assembly.cap.pose).tolist()}
            else:
                if args.axisymmetric_binding:
                    goal=yaw_minimal_cap_goal(refcap,matrix(task.assembly.cap.pose),relative)
                    seed=np.asarray(task.robot.left_entity.qpos)[indices]
                    endpoint,receipt=solve_rebound(audit,seed,goal,task,native)
                    receipt['frame']='Axisymmetriccap centre/axis follows actualpose; world grasp azimuth retained'
                else:
                    goal=matrix(task.assembly.cap.pose)@relative
                    endpoint,receipt=solve_rebound(audit,qreference,goal,task,native)
                    receipt['frame']='Final cap approach rebound to actual cap pose'
            receipt['via_index']=index;rebindings.append(receipt)
            actual_drive_left=native.command24(task.robot)[:6].copy()
            # The auditor already prepends measured q. Explicitly include the
            # current drive endpoint: the executed quintic starts THERE.
            report=moving_path_audit(audit,np.vstack([actual_drive_left,endpoint]),native);planning.append(report)
            if not np.array_equal(task.get_state24(),np.asarray(actual_before['state24'],np.float32)):raise RuntimeError('CPU planner changed physical state')
            rows=native.quintic_targets(actual_drive_left,endpoint)
            if len(rows)>120:raise RuntimeError('Per-via120 target budget exceeded')
            for row in rows:execute(left_arm=row,phase=f'left_approach_via{index}')
            dump(out/'planning_reports.json',planning);dump(out/'actual_cap_rebindings.json',rebindings)
            print(json.dumps({'stage':'via_complete','via':index,'tick':task.physics_step_count,'rows':len(rows)}),flush=True)
        dump(out/'actual_preclose_snapshot.json',snapshot(task,native))
        controller=CapContactController(task,mapping,side='left',actor=task.cap,
            open_sdk=opened,close_sdk=[0.,0.,0.,560.09765625,1000.,348.443359375],
            force_targets_N=(.1,.1),rate_cap_rad_s=.08,feedback_gain_rad_s_per_N=.05,
            pair_actuator_indices=(3,5))
        for index in range(160):
            proposal=controller.propose(task.physics_step_count);proposals.append(proposal)
            execute(left_sdk=proposal['sdk6'],phase='left_first_contact_lowforce')
            if index%20==0:print(json.dumps({'stage':'limited_close','index':index,'sdk6':proposal['sdk6']}),flush=True)
        raise RuntimeError('Finite160-boundary close budget ended without actual left-cap impulse')
    except FirstContactStop:
        event=True
        if current_command:
            partial={**current_command,'executed_ticks':task.physics_step_count-current_command['start_tick'],
                'end_tick':task.physics_step_count,'actual_drive_target24_at_stop':native.command24(task.robot).tolist(),
                'after':snapshot(task,native),'scope':'Partial60ms command; NOT exported as15tick transition'}
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};traceback.print_exc()
        if isinstance(exc,PathRejected):rejected=exc.report
        if current_command:
            partial={**current_command,'executed_ticks':task.physics_step_count-current_command['start_tick'],
                'end_tick':task.physics_step_count,'actual_drive_target24_at_stop':native.command24(task.robot).tolist(),
                'after':snapshot(task,native),'scope':'Aborted partial command; no filled transition'}
    finally:
        if getattr(task,'assembly',None) is not None:
            terminal=snapshot(task,native)
            terminal['capsule_geometry']=read_capsule_entities(task.assembly.tablets,task.assembly.body.pose,
                half_length_m=.002,capsule_radius_m=.003,inner_radius_m=.021,floor_z_m=.003,rim_z_m=.115,tolerance_m=.0004)
            dump(out/'actual_terminal_snapshot.json',terminal);dump(out/'assembly_contract.json',task.assembly.contract())
            rgb=task.observe_rgb()
            for name in ('full_image','left_wrist_image','right_wrist_image'):Image.fromarray(rgb[name]).save(out/('final_'+name+'.png'))
            for name,rows in [('assembly_trace.json.gz',task.assembly.trace),('world_kinematics_pre_post.json.gz',task.assembly.kinematics_trace)]:
                with gzip.open(out/name,'wt') as stream:json.dump(rows,stream)
        dump(out/'partial_command.json',partial);dump(out/'rejected_path.json',rejected)
        dump(out/'planning_reports.json',planning);dump(out/'actual_cap_rebindings.json',rebindings)
        dump(out/'controller_proposals.json',proposals);dump(out/'transitions.json',task.transition_log)
        dump(out/'measured_full_state_trace.json',task.full_state_trace);dump(out/'body_physics_trace.json',task.physics_trace)
        dump(out/'assembly_observations.json',task.assembly_observation_trace)
        result={'scope':'One bounded actual first-touch diagnostic; no cap grasp, unscrew or pour claim',
          'error':error,'first_contact_observed':event,'first_contact':task.first_left_cap_contact,
          'guard_failure':task.left_contact_guard_failure,'full_task_success':False,
          'prefix294_action_state_exact':len(prefix_pass)>=294,'prefix_target_count':expected_prefix,
          'prefix_all_action_state_exact':len(prefix_pass)==expected_prefix,'axisymmetric_binding':args.axisymmetric_binding,
          'resume_only_final_approach':args.resume_final,'shadow_feedback_max_abs_diff':max(shadow,default=None),
          'complete60ms_targets':len(task.transition_log),'partial_ticks':partial['executed_ticks'] if partial else 0,
          'right_arm_drive_held_constant':fixed_right is not None and all(np.array_equal(np.asarray(row['action'][12:18]),fixed_right) for row in task.transition_log[expected_prefix:]),
          'sources_unchanged':all(sha(path)==digest for path,digest in fingerprints.items()),'final_metrics':task.metrics}
        dump(out/'result.json',result);task.close()
    print(json.dumps({'output':str(out),'first_contact_observed':event,'error':error and error['message']}),flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':main()
