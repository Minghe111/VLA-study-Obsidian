from pathlib import Path
import json,hashlib,importlib.util
from types import SimpleNamespace as S
import numpy as np
P=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task')
R=P/'left_contact_probes/left115_first_contact_v3_world_transit';snapshot=R/'actual_terminal_snapshot.json';manifest=json.loads((R/'snapshot_manifest.json').read_text());frozen=Path(manifest[str(P/'cap_contact_controller.py')]['snapshot'])
mappingfile=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin/integrations/realman_rm65b_gen4/teacher_legacy/continuous_servo.py');mf=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/source_manifest.json')
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ctrl=load(frozen,'frozen_v3_cap_controller');mp=load(mappingfile,'readonly_v3_actual_mapping');s=json.loads(snapshot.read_text());meta=json.loads(mf.read_text())['hands']['left'];m=mp.ContinuousHandMapping(meta);qdict=dict(zip(s['joint_names'],s['q36']));sdkdrive=np.array(s['actual_drive_target24'][6:12]);driveq=m.lower+(1-sdkdrive/1000)*m.width;actual=np.array([qdict[n] for n in m.channels]);original=qdict.copy()
class Joint:
 def __init__(self,x):self.x=x
 def get_drive_target(self):return [self.x]
 def set_drive_target(self,*a):raise AssertionError('readonly')
 def set_drive_velocity_target(self,*a):raise AssertionError('readonly')
robot=S(communication_flag=False,_manifest={'hands':{'left':meta}},_measured_positions=lambda:qdict.copy(),_hand_joints={n:Joint(x) for n,x in zip(m.channels,driveq)})
task=S(robot=robot,clock=S(dt=.004,DEFAULT_STEPS=15),physics_step_count=s['physics_tick'],scene=S(get_timestep=lambda:.004,get_contacts=lambda:[]))
exc=None;constructed=False;proposal=None
try:
 c=ctrl.CapContactController(task,m,side='left',actor=S(per_scene_id=104,name='actualcap'),open_sdk=[0,0,0,1000,1000,400],close_sdk=[0,0,0,560.09765625,1000,348.443359375],force_targets_N=(.1,.1),pair_actuator_indices=(3,5));constructed=True;proposal=c.propose(s['physics_tick'])
except (ValueError,RuntimeError) as e:exc={'type':type(e).__name__,'message':str(e)}
ex=np.maximum(np.maximum(m.lower-actual,actual-m.lower-m.width),0);d=np.maximum(np.maximum(m.lower-driveq,driveq-m.lower-m.width),0)
o={'scope':'One CPU readback of archived actual prepared v3 primary q/drive through actual frozen controller and native mapping; no physics/drive writes; left-cap contact array empty as recorded terminal leftcap0','physics_tick':s['physics_tick'],'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [frozen,mappingfile,mf,snapshot]},'channels':m.channels,'actual_primary_joint_positions_rad':actual.tolist(),'actual_primary_lower_rad':m.lower.tolist(),'actual_primary_upper_rad':(m.lower+m.width).tolist(),'actual_joint_limit_excess_rad':ex.tolist(),'actual_max_joint_limit_excess_rad':float(ex.max()),'actual_drive_sdk6_unclipped':sdkdrive.tolist(),'actual_drive_joint_positions_rad':driveq.tolist(),'actual_drive_joint_limit_excess_rad':d.tolist(),'actual_drive_max_joint_limit_excess_rad':float(d.max()),'existing_observation_tolerance_rad':1e-6,'constructor_passed':constructed,'proposal_returned':proposal is not None,'exception':exc,'actual_state_unchanged':original==qdict,'new_physical_trials':0,'live_source_modified':False,'interpretation':'A measured soft-limit excursion rejection is an observer-interface result; it does not establish collision, grip failure or illegal commanded target.'}
(R/'independent_prepared_controller_readback.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
