from pathlib import Path
import sys,os,json,argparse,time,traceback,types
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parent.parent
OLD=Path('/tmp/realman-eval60k-20261011/local_eval')
REPO=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
sys.path[:0]=[str(ROOT/'scripts'),str(OLD/'scripts'),str(REPO),str(REPO.parent/'deploy/realman_gen4/adjust_bottle_20261002')]
os.chdir(REPO)
import envs
base=types.ModuleType('envs._base_task');base.__package__='envs'
base.__file__=str(OLD/'scripts/remote_base_task_snapshot.py');sys.modules['envs._base_task']=base
exec(compile(Path(base.__file__).read_text(),base.__file__,'exec'),base.__dict__)
from scene_replay import build_replay_scene,read_physical_contract,sha
from observations40k import Observations,write
from integrations.realman_rm65b_gen4.action_clock import ClockedTask24Adapter

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--mode',default='current60',choices=['current60','source_offset60','source_offset_dense']);a=ap.parse_args()
    e=next(e for e in json.loads((ROOT/'source_manifest.json').read_text())['episodes'] if e['index']==a.index)
    proof=ROOT/'source'/e['episode_proof_dir'];sample=json.loads((proof/'scene_sample.json').read_text());spec=json.loads((proof/'sample_spec.json').read_text())
    hdf=ROOT/'source/data'/f'episode_{a.index:07d}.hdf5'
    with h5py.File(hdf) as f:
        targets=f['realman_dexterous/next_command24'][:];times=f['realman_dexterous/sample_time_s'][:]
        source_q=f['realman_dexterous_physics/full36_qpos_before_rad'][:]
        source_after=f['realman_dexterous_physics/full36_qpos_after_rad'][:]
        dense=f['realman_dexterous_physics/applied_command24'][:]
        source_state=f['realman_dexterous/measured_state24'][0]
        camera_extrinsics={k:f['vision/'+k+'/extrinsics_matrix'][0].tolist() for k in ['cam_head','cam_left_wrist','cam_right_wrist']}
    assert np.max(abs(np.diff(times)-.06))<1e-7 and len(dense)==len(targets)*15
    out=ROOT/'episodes'/f'ep{a.index:02d}_{a.mode}';out.mkdir(parents=True,exist_ok=False)
    task=observer=adapter=None;error=None;start=time.monotonic()
    try:
        s={'model_id':e['model_id'],'arm':e['arm'],'pose_class':'upright' if e['pose_class']=='upright' else 'horizontal','center_xy':sample['actual_bottle_center_world'][:2],'axis_yaw_deg':sample['actual_axis_yaw_deg'],'source_initial_pose':sample['initial_pose']}
        task,args,setup=build_replay_scene(e['seed'],out/'bootstrap',requested_model_id=e['model_id'],scene_spec=s)
        import sapien
        from transforms3d.quaternions import mat2quat
        from sapien.physx import PhysxRigidDynamicComponent
        p=np.asarray(sample['initial_pose'],float)
        task.bottle.actor.set_pose(sapien.Pose(p[:3,3],mat2quat(p[:3,:3])))
        body=task.bottle.actor.find_component_by_type(PhysxRigidDynamicComponent)
        body.set_linear_velocity(np.zeros(3));body.set_angular_velocity(np.zeros(3));body.put_to_sleep()
        entity=task.robot.left_entity;entity.set_qpos(source_q[0]);entity.set_qvel(np.zeros(36))
        observer=Observations(task,out);adapter=ClockedTask24Adapter(task,observer,out,stop_on_task_success=False)
        adapter._apply(dense[0],np.zeros(24))
        changed=[];contact=spec.get('grasp_setting',{}).get('hand_contact_offset_m',.01)
        if a.mode!='current60':
            for link in entity.get_links():
                if 'inspire' in link.name:
                    for shape in link.get_collision_shapes():
                        changed.append({'link':link.name,'before':float(shape.contact_offset),'after':contact});shape.set_contact_offset(contact)
        obs=task.get_obs();initial={'source_bottle_pose':p.tolist(),'actual_bottle_pose':task.bottle.get_pose().to_transformation_matrix().tolist(),
            'qpos_max_abs_error_rad':float(np.max(abs(entity.get_qpos()-source_q[0]))),'state24_max_abs_error':float(np.max(abs(task.robot.get_state24(True)-source_state))),
            'source_initial_camera_extrinsics':camera_extrinsics,'actual_initial_camera_extrinsics':{out:task.cameras.camera_by_name[name].get_extrinsic_matrix().tolist() for out,name in [('cam_head','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]},
            'initial_bottle_sleeping':bool(body.is_sleeping),'bottle_velocity_assumption':'zero; source initial velocities absent except episode7 explicit settled receipt',
            'robot_initial_qvel_assumption':'zero; original initial engine qvel/contact cache not recorded in HDF',
            'source_initial_pose_matrix_max_abs_error':float(np.max(abs(task.bottle.get_pose().to_transformation_matrix()-p))),
            'initialization_only_pose_writes':True,'no_object_pose_writes_after_start':True,'source_hand_contact_offset_m':contact,'contact_offset_changes':changed,
            'actual_physical_contract':read_physical_contract(task),'replay_builder_sha256':sha(ROOT/'scripts/scene_replay.py'),'replay_runner_sha256':sha(Path(__file__)),'source_hdf_sha256':sha(hdf),'source_scene_sha256':sha(proof/'scene_sample.json'),'current_clock_sha256':sha(REPO/'integrations/realman_rm65b_gen4/action_clock.py')}
        write(out/'initial_state.json',initial)
        if a.mode=='source_offset_dense':
            # Separate diagnostic: recorded 4ms drive targets, no learned policy.
            for j,cmd in enumerate(dense):
                vel=(dense[min(j+1,len(dense)-1)]-cmd)/observer.dt
                # Original contact-controller writes zero hand drive velocity.
                vel[6:12]=0.;vel[18:24]=0.
                adapter._apply(cmd,vel);observer.advance()
                if task.check_success():task.eval_success=True
                if (j+1)%750==0:print('PROGRESS',a.index,a.mode,j+1,flush=True)
        else:
            for j,target in enumerate(targets):
                adapter.take_action(target)
                if (j+1)%40==0:print('PROGRESS',a.index,a.mode,j+1,'gap',observer.max_gap,flush=True)
            assert observer.step==len(dense) and len(adapter.actions)==len(targets)
    except Exception as ex:
        error={'error':repr(ex),'traceback':traceback.format_exc()};print(error['traceback'],flush=True)
    finally:
        rec=observer.finish() if observer else None;summary=observer.summary() if observer else None
        report={'scope':'Original train50 expert labels replayed through current 24D inference actuator; real physics and newly rendered RGB','index':a.index,'mode':a.mode,'source_episode':{k:e[k] for k in ['index','seed','model_id','brand','arm','pose_class','physical_success','state_action_pairs','physical_steps']},'source_hdf':str(hdf),'source_hdf_sha256':sha(hdf),'model_evaluated':False,'model_query_count':0,'expert_actions_used':True,'passed':bool(summary and summary['user_air_grasp_observed_1cm_1s']),'native_task_success':bool(task and task.eval_success),'actual_physics_steps':0 if observer is None else observer.step,'physical_duration_s':0 if observer is None else observer.step*observer.dt,'wall_duration_s':time.monotonic()-start,'own_actual_rgb_recording':rec,'observation':summary,'interface_or_infrastructure_error':error,'executed_actions':[] if adapter is None else adapter.actions,'hardware_io':False,'nominal_action_period_s':.06,'original_contact_feedback_not_recomputed':True,'current_physics_configuration_retained':a.mode=='current60','stop_on_success':False}
        write(out/'report.json',report)
        print('EPISODE_RESULT',json.dumps({k:report[k] for k in ['index','mode','passed','native_task_success','physical_duration_s','observation','interface_or_infrastructure_error']}),flush=True)
        if task:
            if observer:task.scene.step=observer.raw_step
            task.close_env()
    return 1 if error else 0
if __name__=='__main__':sys.exit(main())
