"""Actual local80000 model on original six expert starts; official fullH16 loop."""
from pathlib import Path
import sys,os,json,argparse,time,traceback,types,importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
REPO=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
sys.path[:0]=[str(ROOT/'scripts'),str(REPO),str(REPO.parent/'deploy/realman_gen4/adjust_bottle_20261002'),str(ROOT/'viewer')]
os.chdir(REPO)
import envs
base=types.ModuleType('envs._base_task');base.__package__='envs'
base.__file__=str(ROOT/'scripts/remote_base_task_snapshot.py');sys.modules['envs._base_task']=base
exec(compile(Path(base.__file__).read_text(),base.__file__,'exec'),base.__dict__)
from scene_exact import build_exact_expert_scene,read_physical_contract,assert_input_guard,sha
from initial_source import load_initial_record,restore_initial_only,audit_initial
from observations80k import Observations80k
from observations40k import write
from client80k import LocalPolicyClient,PolicyClientAdapter
from integrations.realman_rm65b_gen4.action_clock import ClockedTask24Adapter
from publish_live import LivePublisher

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True)
    ap.add_argument('--base-url',default='http://127.0.0.1:8769');ap.add_argument('--identity',required=True)
    ap.add_argument('--identity-sha256',required=True);ap.add_argument('--episode-position',type=int,default=1)
    ap.add_argument('--output');ap.add_argument('--live-root',default=str(ROOT/'live'));ap.add_argument('--no-external-camera',action='store_true')
    ap.add_argument('--pace',type=float,default=0.,help='Display pacing in wall seconds per60ms physical frame; default no added delay')
    a=ap.parse_args();out=Path(a.output) if a.output else ROOT/'episodes'/f'ep{a.index:02d}_model80000'
    out.mkdir(parents=True,exist_ok=False);record=load_initial_record(a.index);e=record['episode']
    # Identity/health is checked before allocating a simulation renderer.
    client=LocalPolicyClient(a.base_url,expected_model_identity_sha256=a.identity_sha256,identity_path=a.identity)
    health=client.health();write(out/'initial_model_health.json',health)
    live=LivePublisher(a.live_root,checkpoint=80000)
    label=f"{e['brand']} / {e['arm']} / {e['pose_class']} / source {a.index}"
    live.begin_episode(a.index,label,episode_position=a.episode_position,episode_total=6)
    current_phase='initializing_scene'
    def metadata(observer=None,task=None):
        return {'episode_index':a.index,'episode_label':label,'episode_position':a.episode_position,'episode_total':6,
                'model_query_count':len(client.records),'physics_step':0 if observer is None else int(observer.step),
                'sim_time_s':0. if observer is None else float(observer.step*observer.dt),
                'phase':current_phase,'native_success':False if task is None else bool(task.eval_success),
                'joint_state':None if task is None else list(task.robot.get_state24(True))}
    task=observer=adapter=None;error=None;initial={};start=time.monotonic()
    def capture_hook(obs,obsr,t):
        live.update(images=obs['display_images'],metadata=metadata(obsr,t))
        if a.pace>0:time.sleep(a.pace)
    def stage_hook(phase):
        nonlocal current_phase
        current_phase=phase;live.update(metadata=metadata(observer,task))
        if phase=='waiting_for_model':print('MODEL_QUERY',a.index,len(client.records)+1,'simtime',observer.step*observer.dt,flush=True)
    try:
        stage_hook('initializing_scene')
        task,args,setup=build_exact_expert_scene(e['seed'],out/'bootstrap',requested_model_id=e['model_id'],scene_spec=record['spec'])
        restoration=restore_initial_only(task,record)
        current_phase='scene_ready'
        observer=Observations80k(task,out,capture_hook=capture_hook,external_camera=not a.no_external_camera)
        adapter=ClockedTask24Adapter(task,observer,out,stop_on_task_success=False)
        # Readable source first drive setting only. All later actions come from
        # this checkpoint's inverse-transformed model predictions.
        adapter._apply(record['initial_command24'],np.zeros(24))
        initial={**restoration,**audit_initial(task,record),'source_hdf':record['source_hdf'],'source_hdf_sha256':record['source_hdf_sha256'],
                 'source_initial_scene_sha256':record['source_initial_scene_sha256'],'read_boundaries':record['read_boundaries'],
                 'initial_command24':record['initial_command24'].tolist(),'actual_physical_contract':read_physical_contract(task),
                 'clock_contract':adapter.clock_contract,'source_scene':record['spec'],'bootstrap_evidence':setup,
                 'original_expert_replay_reference':str(Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')/f'episode_{a.index:07d}'),
                 'display_camera_model_input':False,'hardware_io':False,'prompt':record['prompt'],
                 'current_clock_sha256':sha(REPO/'integrations/realman_rm65b_gen4/action_clock.py')}
        write(out/'initial_state.json',initial)
        deployment=REPO/'XPolicyLab/policy/Pi_05/deploy.py'
        spec=importlib.util.spec_from_file_location('current_official_pi05_deploy',deployment);deploy=importlib.util.module_from_spec(spec);spec.loader.exec_module(deploy)
        model=PolicyClientAdapter(client,record['prompt'],out,stage_hook=stage_hook)
        stage_hook('executing_actions');deploy.eval_one_episode(adapter,model)
        stage_hook('episode_complete')
    except Exception as ex:
        error={'error':repr(ex),'traceback':traceback.format_exc()};print(error['traceback'],flush=True)
        stage_hook('episode_error')
    finally:
        rec=observer.finish() if observer else None;summary=observer.summary() if observer else None
        report={'scope':'Local checkpoint80000 model inference from original train50 initial scenes through current60ms24D actuator and new physical RGB',
                'index':a.index,'source_episode':{k:e[k] for k in ['index','seed','model_id','brand','arm','pose_class']},
                'checkpoint_step':80000,'model_identity_sha256':a.identity_sha256,'model_evaluated':True,
                'model_query_count':len(client.records),'model_queries':client.records,'expert_future_actions_used':False,
                'source_initial_HDF_only':record['read_boundaries'],'native_task_success':bool(task and task.eval_success),
                'actual_physics_steps':0 if observer is None else observer.step,'actual_action_count':0 if adapter is None else len(adapter.actions),
                'physical_duration_s':0 if observer is None else observer.step*observer.dt,'wall_duration_s':time.monotonic()-start,
                'actual_rgb_recording':rec,'observation':summary,'interface_or_infrastructure_error':error,
                'executed_actions':[] if adapter is None else adapter.actions,'hardware_io':False,'stop_on_success':False,
                'termination_protocol':'Diagnostic whole400-target rollout; native success retained as an event, continued to observe learned turn/hold',
                'not_directly_comparable_to_previous8_stop_on_success_cases':True,
                'nominal_action_period_s':.06,'policy_full_prediction_horizon':16,'policy_full_chunk_execution':True,
                'native_action_limit':None if task is None else task.step_lim,'display_pace_wall_s_per_frame':a.pace,
                'source_hdf_sha256':record['source_hdf_sha256'],'simulation_builder_sha256':sha(ROOT/'scripts/scene_exact.py'),
                'runner_sha256':sha(Path(__file__)),'official_deployment_sha256':sha(REPO/'XPolicyLab/policy/Pi_05/deploy.py'),
                'hardware_guard_reverified':assert_input_guard()['archived_guard_mapping']}
        write(out/'report.json',report)
        print('EPISODE_RESULT',json.dumps({k:report[k] for k in ['index','checkpoint_step','model_query_count','native_task_success','physical_duration_s','observation','interface_or_infrastructure_error']},allow_nan=False),flush=True)
        if task:
            if observer:task.scene.step=observer.raw_step
            task.close_env()
    return 1 if error else 0

if __name__=='__main__':sys.exit(main())
