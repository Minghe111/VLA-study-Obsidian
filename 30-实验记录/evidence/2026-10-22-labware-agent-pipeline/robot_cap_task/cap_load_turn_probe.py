"""Contact-only finite pressure and small-turn continuation from actual v6.

The old one-tick diagnostic is reproduced inside a newly executed COMPLETE
60-ms window. No saved physical state is restored and no object is actuated.
"""
from pathlib import Path
import argparse,copy,gzip,json,shutil,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
from left_cap_contact_probe import sha,dump,matrix,snapshot,actual_collision_geometry,solve_rebound
from left_cap_contact_task import left_cap_contact_task
from labware_cap115_task import PROFILE
from robot_expert import adapted_native
from cap_contact_controller import CapContactController,read_hand_contacts
from contact_turn_geometry import small_contact_turn_palm_target
from contact_wrench_observer import read_contact_wrench
from camera_framing import apply_camera_framing
from capsule_geometry import read_capsule_entities
from PIL import Image

class LoadTurnTask(left_cap_contact_task):
    def setup_demo(self,**kwargs):
        self.wrench_trace=[];self.reference_tick=None;self.reference_snapshot=None;self.reference_comparison=None
        answer=super().setup_demo(**kwargs);self.stop_at_first_contact=False;return answer
    def _update_metrics(self):
        super()._update_metrics()
        if not getattr(self,'_active',False) or not self.left_probe_active:return
        body=matrix(self.assembly.body.pose)
        self.wrench_trace.append({'physics_tick':self.physics_step_count,'phase':self.current_phase,
            'cap_contact_wrench':read_contact_wrench(self.scene,self.cap,
                origin_world_m=self.assembly.cap.pose.p,axis_world=body[:3,2])})
        if self.reference_tick==self.physics_step_count:
            actual=snapshot(self,adapted_native());fields=('q36','qvel36','state24','actual_drive_target24',
                'root_matrix','left_palm_matrix','right_palm_matrix','body_matrix','cap_matrix','tablet_world_pose_pq')
            self.reference_comparison={key:bool(np.array_equal(np.asarray(actual[key]),np.asarray(self.reference_snapshot[key]))) for key in fields}
            if not all(self.reference_comparison.values()):raise RuntimeError('Reexecuted first-pressure physical reference differs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--angle',type=float,default=.05);args=p.parse_args()
    if not 0<args.angle<=.05:raise ValueError('First turn is bounded to0.05rad')
    out=ROOT/'load_turn_probes'/args.name;out.mkdir(parents=True,exist_ok=False)
    original=json.loads((args.source/'transitions.json').read_text());partial=json.loads((args.source/'partial_command.json').read_text())
    if len(original)!=732 or partial['executed_ticks']!=1:raise RuntimeError('Expected exact actual v6 reference732+1tick')
    files=[Path(__file__),ROOT/'left_cap_contact_probe.py',ROOT/'left_cap_contact_task.py',ROOT/'cap_contact_controller.py',
      ROOT/'contact_turn_geometry.py',ROOT/'contact_wrench_observer.py',ROOT/'labware_cap_workflow.py',ROOT/'labware_cap115_task.py',
      ROOT/'labware_cap_task.py',ROOT/'capsule_geometry.py',ROOT.parent/'labware_task.py',ROOT.parent/'robot_expert.py',
      ROOT.parent/'robot_cap_assembly.py',args.source/'transitions.json',args.source/'partial_command.json',
      *[path for path in PROFILE.iterdir() if path.is_file()]]
    fingerprints={str(path.resolve()):sha(path)for path in files};dump(out/'source_manifest.json',fingerprints)
    folder=out/'source_snapshot';folder.mkdir();snaps={}
    for path,digest in fingerprints.items():
        dest=folder/(digest+'_'+Path(path).name);shutil.copyfile(path,dest);snaps[path]={'snapshot':str(dest),'sha256':digest}
    dump(out/'snapshot_manifest.json',snaps)
    task=LoadTurnTask();native=adapted_native();error=None;prefix=[];shadow=[];load=[];turn=[];pending=None
    fixed=None;reference_completed=False;opposed_ready=False;before_turn=None;commanded_angle=0.
    try:
        task.setup_demo(seed=0,runtime_path=out,capture_rgb=False,center_xy=[.10,-.11]);apply_camera_framing(task)
        dump(out/'actual_setup_initial_snapshot.json',snapshot(task,native))
        setting={'open_sdk':[0.,0.,800.,800.,800.,0.],'close_sdk':[0.,0.,608.,610.,610.,0.],
            'pitch_deg':10.,'hand_x':-.015,'hand_y':.076,'hand_z':.177,'palm_roll_deg':5.,'approach_azimuth_deg':0.}
        right=native.BoundaryContactController(task,'right',setting)
        for index,row in enumerate(original):
            if not np.array_equal(task.get_state24(),np.asarray(row['state'],np.float32)):raise RuntimeError(f'Prefix state differs{index}')
            if row['phase'] in ('close','lift_plan','lift','hold','right_fixed_arm_same_grip_feedback') or row['phase'].startswith('left_'):
                sdk=right.target(carry=row['phase']not in('close','lift_plan'));diff=float(np.max(np.abs(sdk-np.asarray(row['action'][18:24]))));shadow.append(diff)
                if diff>1e-8:raise RuntimeError('Right feedback shadow differs')
            task.execute_command24(row['action'],row['phase'])
            if not np.array_equal(task.get_state24(),np.asarray(row['next_state'],np.float32)):raise RuntimeError(f'Prefix nextstate differs{index}')
            prefix.append(True)
        print('PREFIX732_EXACT',flush=True);task.activate_left_probe();task.stop_at_first_contact=False
        task.reference_tick=partial['end_tick'];task.reference_snapshot=partial['after']
        replay_right=right.target(carry=True)
        if np.max(np.abs(replay_right-np.asarray(partial['requested_target24'][18:24])))>1e-8:raise RuntimeError('Partial right shadow differs')
        # Actually execute the original pending target for all15 physics steps.
        pending={'target24':partial['requested_target24'],'start_tick':task.physics_step_count,'phase':partial['phase']}
        task.execute_command24(partial['requested_target24'],partial['phase']);pending=None
        reference_completed=True;dump(out/'reexecuted_first_pressure_reference.json',task.reference_comparison)
        dump(out/'actual_complete_boundary_after_first_pressure.json',snapshot(task,native))
        fixed=native.command24(task.robot)[12:18].copy()
        def execute(left_sdk,phase,left_arm=None):
            nonlocal pending
            target=native.command24(task.robot);target[12:18]=fixed;target[18:24]=right.target(carry=True)
            target[6:12]=left_sdk
            if left_arm is not None:target[:6]=left_arm
            pending={'target24':target.tolist(),'start_tick':task.physics_step_count,'phase':phase}
            task.execute_command24(target,phase);pending=None
        mapping=native.ContinuousHandMapping(task.robot._manifest['hands']['left'])
        # Existing free-side range permits pressure relief. The new closure
        # bound is only8index SDK units beyond observed0.855mm zero-load gap.
        opened=[0.,0.,0.,560.09765625,1000.,348.443359375]
        close=[0.,0.,0.,528.0999393463135,1000.,180.54951238632202]
        controller=CapContactController(task,mapping,side='left',actor=task.cap,open_sdk=opened,close_sdk=close,
            force_targets_N=(.1,.1),rate_cap_rad_s=.08,feedback_gain_rad_s_per_N=.05,pair_actuator_indices=(3,5))
        dump(out/'pressure_interval.json',{'open_sdk6':opened,'close_sdk6':close,'target_force_N':[.1,.1],
            'index_extra_closure_sdk':8.,'scope':'Finite pressure feedback from actual v6; actualguards decide outcome, no rigid-grasp assumption'})
        ready=0
        for i in range(60):
            proposal=controller.propose(task.physics_step_count);execute(proposal['sdk6'],'cap_pair_loading')
            contact=read_hand_contacts(task.scene,task.cap,'left');ready=ready+1 if contact['opposed']else 0
            load.append({'boundary_index':i,'physics_tick':task.physics_step_count,'proposal':proposal,'actual_contact':contact,
                'cap_state':copy.deepcopy(task.metrics['cap_state'])})
            print(json.dumps({'load_boundary':i,'index_N':contact['forces_N'][3],'thumb_N':contact['forces_N'][4],
                'normals_dot':contact['index_thumb_normals_dot'],'opposed_windows':ready}),flush=True)
            if ready>=3:opposed_ready=True;break
        if not opposed_ready:raise RuntimeError('Finite60-boundary pressure budget ended without actual opposing two-finger load')
        before_turn=snapshot(task,native);dump(out/'actual_before_small_turn.json',before_turn)
        indices=[task.robot.left_entity.active_joints.index(j)for j in task.robot.left_arm_joints]
        for index in range(10):
            body=matrix(task.assembly.body.pose);cap=matrix(task.assembly.cap.pose)
            palm=matrix(task.robot.left_entity.find_link_by_name('l_inspire_base_link').entity.pose)
            goal,geometry=small_contact_turn_palm_target(body,cap,palm,delta_angle_rad=args.angle/10.)
            audit=native.fma.FullMeshPathAuditor(task)
            seed=np.asarray(task.robot.left_entity.qpos)[indices]
            arm,ik=solve_rebound(audit,seed,goal,task,native)
            rows=native.quintic_targets(native.command24(task.robot)[:6],arm)
            if len(rows)>20:raise RuntimeError('Small-turn increment requires >20bounded targets')
            for row in rows:
                proposal=controller.propose(task.physics_step_count);execute(proposal['sdk6'],'cap_small_contact_turn',row)
            commanded_angle+=args.angle/10.
            measured=task.metrics['cap_state']['rotation_measured_rad']-before_turn['metrics']['cap_state']['rotation_measured_rad']
            turn.append({'increment':index,'commanded_angle_sum_rad':commanded_angle,'actual_cap_relative_rotation_delta_rad':measured,
                'geometry':geometry,'ik':ik,'actual_snapshot':snapshot(task,native)})
            print(json.dumps({'turn_increment':index,'requested_rad':commanded_angle,'actual_cap_delta_rad':measured}),flush=True)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};traceback.print_exc()
    finally:
        if getattr(task,'assembly',None)is not None:
            terminal=snapshot(task,native);terminal['capsule_geometry']=read_capsule_entities(task.assembly.tablets,task.assembly.body.pose,
                half_length_m=.002,capsule_radius_m=.003,inner_radius_m=.021,floor_z_m=.003,rim_z_m=.115,tolerance_m=.0004)
            dump(out/'actual_terminal_snapshot.json',terminal);dump(out/'actual_collision_geometry.json',actual_collision_geometry(task))
            dump(out/'assembly_contract.json',task.assembly.contract())
            rgb=task.observe_rgb()
            for name in('full_image','left_wrist_image','right_wrist_image'):Image.fromarray(rgb[name]).save(out/('final_'+name+'.png'))
            for filename,rows in [('assembly_trace.json.gz',task.assembly.trace),('world_kinematics_pre_post.json.gz',task.assembly.kinematics_trace),
                ('left_cap_separation_trace.json.gz',task.left_cap_separation_trace),('cap_contact_wrench_trace.json.gz',task.wrench_trace)]:
                with gzip.open(out/filename,'wt')as f:json.dump(rows,f)
        if pending:pending.update(actual_end_tick=task.physics_step_count,executed_ticks=task.physics_step_count-pending['start_tick'],actual_terminal=snapshot(task,native))
        result={'scope':'Finite physical pressure and <=.05rad contactturn only; not full unscrew/pour/dataset',
            'error':error,'prefix732_exact':len(prefix)==732,'shadow_feedback_max_abs_diff':max(shadow,default=None),
            'original_first_pressure_tick_reproduced_exact':task.reference_comparison,
            'new_complete_window_reexecuted':reference_completed,'opposing_pressure_ready':opposed_ready,
            'small_turn_increments':len(turn),'commanded_angle_rad':commanded_angle,'full_task_success':False,
            'targets':len(task.transition_log),'sources_unchanged':all(sha(path)==digest for path,digest in fingerprints.items()),
            'final_metrics':task.metrics}
        dump(out/'result.json',result);dump(out/'partial_command.json',pending);dump(out/'load_observations.json',load)
        dump(out/'turn_observations.json',turn);dump(out/'transitions.json',task.transition_log)
        dump(out/'measured_full_state_trace.json',task.full_state_trace);dump(out/'body_physics_trace.json',task.physics_trace)
        dump(out/'assembly_observations.json',task.assembly_observation_trace);task.close()
    print(json.dumps({'output':str(out),'opposing_pressure_ready':opposed_ready,'turn_increments':len(turn),'error':error and error['message']}),flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':main()
