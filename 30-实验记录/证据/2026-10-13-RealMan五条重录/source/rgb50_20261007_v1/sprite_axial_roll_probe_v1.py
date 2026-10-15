"""Independent Sprite axial-roll initialization audit. No grasp actions or pose writes."""
from pathlib import Path
import argparse, copy, hashlib, inspect, json, os, sys, traceback
import numpy as np
WORK=Path(__file__).resolve().parent
ROOT=Path("/bigdata2/liminghe/VLA-benchmark/RoboTwin")
LEGACY=ROOT.parent/"deploy/realman_gen4/adjust_bottle_20261002"
sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)]
os.chdir(ROOT)
os.environ.setdefault("ROBOTWIN_RT_DENOISER","none")
os.environ.setdefault("RGB50_EXPERT_MODULE","expert_generation_v3")
ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
opt=ap.parse_args();opt.out.mkdir(parents=True,exist_ok=False)
import generation_runner as runner
from envs._base_task import Base_Task, cal_quat_dis
from integrations.realman_rm65b_gen4.native_robot import RealManRobot
source=inspect.getsource(runner.prepare_task)
old="R=axangle2mat([0,0,1],yaw-np.pi/2)"
assert source.count(old)==1
source=source.replace(old,old+"@axangle2mat([0,1,0],np.deg2rad(float(spec.get('axial_roll_deg',0.))))")
actor_line="self.bottle=create_actor(self,sapien.Pose(origin,mat2quat(R)),modelname='001_bottle',convex=True,model_id=model)"
assert source.count(actor_line)==1
source=source.replace(actor_line,actor_line+"\n            self._axial_probe_creation_matrix=self.bottle.get_pose().to_transformation_matrix().copy()")
namespace=vars(runner).copy()
exec(compile(source,str(Path(__file__).resolve())+":creation_axial_roll_overlay","exec"),namespace)
prepare=namespace["prepare_task"]
original_check=Base_Task.check_stable
lines,startline=inspect.getsourcelines(original_check)
step_lines={startline+i for i,l in enumerate(lines) if l.strip()=="self.scene.step()"}
sample_line=next(startline+i for i,l in enumerate(lines) if l.strip()=="for idx, actor in enumerate(actors_list):")
# This source line occurs once inside check(times), then once after settling.
# The first occurrence follows scene.step and yields one sample for each of500 inspections.
summaries=[]
plan=json.loads((WORK/"sample_plan.json").read_text())
for index in [36,4]:
    for roll in [45,90,180,270]:
        spec=copy.deepcopy(next(s for s in plan["specs"] if s["index"]==index))
        spec["axial_roll_deg"]=roll
        out=opt.out/f"spec_{index:07d}_roll_{roll:03d}";out.mkdir()
        records=[];scene_steps=[0];observed_task=[None];check_result=[None]
        initializer_before=RealManRobot._initialize_hands
        def observed(self):
            observed_task[0]=self
            entry=self.bottle.get_pose().to_transformation_matrix().copy()
            prior=sys.gettrace()
            def trace(frame,event,arg):
                if frame.f_code is original_check.__code__:
                    if event=="line" and frame.f_lineno in step_lines:
                        scene_steps[0]+=1
                    return trace
                # Observe the original nested check(times) function; numerical code remains identical.
                if frame.f_code.co_name=="check" and frame.f_code.co_filename==original_check.__code__.co_filename:
                    if event=="line" and frame.f_lineno in step_lines:
                        scene_steps[0]+=1
                    if event=="line" and frame.f_lineno==sample_line:
                        pose=self.bottle.get_pose()
                        records.append({"step":scene_steps[0],"q":np.asarray(pose.q).tolist(),"p":np.asarray(pose.p).tolist()})
                    return trace
                return None
            sys.settrace(trace)
            try:
                result=original_check(self)
                check_result[0]={"passed":bool(result[0]),"unstable_objects":result[1],
                    "entry_matrix":entry.tolist(),"exit_matrix":self.bottle.get_pose().to_transformation_matrix().tolist(),
                    "dt_s":float(self.scene.get_timestep())}
                return result
            finally:sys.settrace(prior)
        Base_Task.check_stable=observed
        task=None;receipt=None;error=None
        print("START",index,spec["arm"],"roll",roll,flush=True)
        try:task,receipt,unused=prepare(spec,out/"scene",need_planner=False)
        except BaseException as e:error={"message":str(e),"traceback":traceback.format_exc()}
        finally:
            Base_Task.check_stable=original_check
            RealManRobot._initialize_hands=initializer_before
            live=task or observed_task[0]
            if live is not None:
                creation=getattr(live,"_axial_probe_creation_matrix",None)
                post=live.bottle.get_pose().to_transformation_matrix()
            else:creation=post=None
            if records:
                final=np.asarray(records[-1]["q"])
                max_last200=float(max(abs(cal_quat_dis(final,np.asarray(r["q"]))*180.) for r in records[-200:]))
                max500=float(max(abs(cal_quat_dis(final,np.asarray(r["q"]))*180.) for r in records))
            else:max_last200=max500=None
            o={"spec":spec,"scope":"Creation orientation varies only by local bottle Y-axis roll; untouched native stable check; no grasp/motion or post-creation bottle pose writes",
               "check":check_result[0],"actual_native_check_scene_steps":scene_steps[0],
               "actual_native_check_duration_s":scene_steps[0]*(check_result[0]["dt_s"] if check_result[0] else .004),
               "inspection_pose_count":len(records),"max_official_angle_to_final_last200_deg":max_last200,
               "max_official_angle_to_final500_deg":max500,
               "creation_matrix":None if creation is None else creation.tolist(),"post_setup_matrix":None if post is None else post.tolist(),
               "scene_receipt":receipt,"error":error,"hardware_io":False,"grasp_actions":0,
               "source_hashes":{"generation_runner.py":hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
                   "_base_task.py":hashlib.sha256(Path(inspect.getsourcefile(original_check)).read_bytes()).hexdigest(),
                   "private_creation_overlay":hashlib.sha256(source.encode()).hexdigest()},
               "pose_observations":records}
            runner.json_write(out/"axial_roll_stability.json",o)
            short={k:o[k] for k in ["spec","check","actual_native_check_scene_steps","actual_native_check_duration_s",
                      "inspection_pose_count","max_official_angle_to_final_last200_deg","max_official_angle_to_final500_deg","error"]}
            short["report"]=str(out/"axial_roll_stability.json")
            summaries.append(short);runner.json_write(opt.out/"summary.json",summaries)
            print("DONE",index,"roll",roll,"check",check_result[0] and check_result[0]["passed"],
                "steps",scene_steps[0],"poses",len(records),"last200deg",max_last200,
                "prepare_ok",error is None,flush=True)
            if live is not None:live.close_env()
