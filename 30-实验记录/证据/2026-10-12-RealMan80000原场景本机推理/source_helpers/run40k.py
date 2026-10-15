from pathlib import Path
import argparse,json,sys,os,importlib.util,traceback,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
ROBOTWIN=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
GEN4=ROBOTWIN.parent/'deploy/realman_gen4'
sys.path[:0]=[str(ROOT/'scripts'),str(ROBOTWIN),str(GEN4/'adjust_bottle_20261002')]
os.chdir(ROBOTWIN)
# Bind the exact161 Base_Task snapshot in this one process for comparable mechanics.
import types
import envs
base_module=types.ModuleType('envs._base_task');base_module.__package__='envs'
base_module.__file__=str(ROOT/'scripts/remote_base_task_snapshot.py')
sys.modules['envs._base_task']=base_module
exec(compile(Path(base_module.__file__).read_text(),base_module.__file__,'exec'),base_module.__dict__)
from scene40k import build_fresh_scene,sha
from observations40k import Observations,PolicyClientAdapter,write
from client40k import LocalPolicyClient
from integrations.realman_rm65b_gen4.action_clock import ClockedTask24Adapter

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['left','right'],required=True);ap.add_argument('--label',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--model',type=int,required=True);ap.add_argument('--pose',choices=['upright','horizontal'],required=True);ap.add_argument('--x',type=float,default=-.12);ap.add_argument('--y',type=float,default=-.10);ap.add_argument('--yaw',type=float,default=0.);a=ap.parse_args()
    out=ROOT/'episodes'/a.label;out.mkdir(parents=True,exist_ok=False)
    identity=ROOT/'model_service/model_identity.json'
    client=LocalPolicyClient(expected_model_identity_sha256=sha(identity),identity_path=identity,timeout_s=600)
    client.health();task=None;observer=None;adapter=None;error=None;started=time.monotonic();stage='scene'
    spec={'model_id':a.model,'arm':a.arm,'pose_class':a.pose,'center_xy':[a.x,a.y],'axis_yaw_deg':a.yaw,'seed':a.seed}
    prompt='Pick up the soda bottle with the nearer arm, turn it upright, and hold it steadily in the air.'
    try:
        task,args,evidence=build_fresh_scene(a.seed,out/'bootstrap',requested_model_id=a.model,scene_spec=spec)
        write(out/'fresh_setup.json',evidence)
        observer=Observations(task,out);adapter=ClockedTask24Adapter(task,observer,out)
        specmod=importlib.util.spec_from_file_location('official_pi05_deploy',ROBOTWIN/'XPolicyLab/policy/Pi_05/deploy.py');module=importlib.util.module_from_spec(specmod);specmod.loader.exec_module(module)
        policy_adapter=PolicyClientAdapter(client,prompt,out)
        take=adapter.take_action
        def progress(action):
            r=take(action)
            if task.take_action_cnt%16==0 or task.eval_success:
                print(json.dumps({'actions':task.take_action_cnt,'queries':len(client.records),'sim_s':observer.step*observer.dt,'native_success':bool(task.eval_success),'max_bottom_gap_m':observer.max_gap}),flush=True)
            return r
        adapter.take_action=progress;stage='actual_closed_loop'
        module.eval_one_episode(adapter,policy_adapter)
    except Exception as exc:
        error={'stage':stage,'error':repr(exc),'traceback':traceback.format_exc()};print(error['traceback'],flush=True)
    finally:
        manifest=observer.finish() if observer is not None else None
        summary=observer.summary() if observer is not None else None
        report={'scope':'Checkpoint60000 full50 qualitative fresh-scene closed-loop; official Pi_05 eval loop + fixed60ms rawSDK24 embodiment adapter','checkpoint_identity':json.loads(identity.read_text()),'model_identity_sha256':sha(identity),'sample_spec':spec,'prompt':prompt,'finished':True,'model_evaluated':bool(client.records),'native_task_success':bool(task is not None and task.eval_success),'actual_task_success':bool(task is not None and task.eval_success),'passed':bool(task is not None and task.eval_success),'interface_or_infrastructure_error':error,'stop_reason':'infrastructure_error' if error else 'native_task_success' if task.eval_success else 'native_action_step_limit','native_action_step_limit':None if task is None else task.step_lim,'native_action_calls':0 if task is None else task.take_action_cnt,'actual_physics_steps':0 if observer is None else observer.step,'physical_duration_s':0 if observer is None else observer.step*observer.dt,'wall_duration_s':time.monotonic()-started,'model_query_count':len(client.records),'own_actual_rgb_recording':manifest,'observation':summary,'raw_queries':client.records,'executed_actions':[] if adapter is None else adapter.actions,'expert_actions_used':False,'SDK_projection':False,'nominal_action_period_s':.06,'execute_complete_returned_action_chunk':bool(adapter is not None and len(adapter.actions)==16*len(client.records)),'hardware_io':False}
        write(out/'report.json',report)
        print('EPISODE_RESULT '+json.dumps({k:report[k] for k in ('model_evaluated','native_task_success','stop_reason','native_action_calls','physical_duration_s','model_query_count','observation')}),flush=True)
        if task is not None:
            if observer is not None:task.scene.step=observer.raw_step
            task.close_env()
    return 0 if error is None else 1
if __name__=='__main__':sys.exit(main())
