from pathlib import Path
import json,hashlib,importlib.util
from types import SimpleNamespace as S
import numpy as np
P=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task');snapfile=P/'static_hold_probes/held115_static_reference_v1/actual_terminal_snapshot.json';proposal=P.parent/'cap_geometry_probes/left115_static_rebind400/rebind.json';mf=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/source_manifest.json');mappingfile=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin/integrations/realman_rm65b_gen4/teacher_legacy/continuous_servo.py')
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ctrl=load(P/'cap_contact_controller.py','actual_readback_controller');mp=load(mappingfile,'actual_readback_mapping');s=json.loads(snapfile.read_text());candidate=json.loads(proposal.read_text());meta=json.loads(mf.read_text())['hands']['left'];m=mp.ContinuousHandMapping(meta);qdict=dict(zip(s['joint_names'],s['q36']));sdkdrive=np.array(s['actual_drive_target24'][6:12]);driveq=m.lower+(1-sdkdrive/1000)*m.width
class Joint:
 def __init__(self,x):self.x=x
 def get_drive_target(self):return [self.x]
 def set_drive_target(self,*a):raise AssertionError('no drive writes allowed')
 def set_drive_velocity_target(self,*a):raise AssertionError('no drive writes allowed')
robot=S(communication_flag=False,_manifest={'hands':{'left':meta}},_measured_positions=lambda:qdict.copy(),_hand_joints={n:Joint(x) for n,x in zip(m.channels,driveq)})
task=S(robot=robot,clock=S(dt=.004,DEFAULT_STEPS=15),physics_step_count=s['physics_tick'],scene=S(get_timestep=lambda:.004,get_contacts=lambda:[]));probe=object.__new__(ctrl.CapContactController);probe.task=task;probe.mapping=m
actual=probe._measured_q();drive=probe._drive_q();actual_before=qdict.copy();drive_before=driveq.copy();exception=None;constructed=False;output=None
try:
 c=ctrl.CapContactController(task,m,side='left',actor=S(per_scene_id=104,name='actualcap'),open_sdk=candidate['open_sdk6'],close_sdk=candidate['first_contact_sdk6'],pair_actuator_indices=(3,5));constructed=True;output=c.propose(s['physics_tick'])
except (ValueError,RuntimeError) as exc:exception={'type':type(exc).__name__,'message':str(exc)}
actual_excess=np.maximum(np.maximum(m.lower-actual,actual-m.lower-m.width),0);drive_excess=np.maximum(np.maximum(m.lower-drive,drive-m.lower-m.width),0)
out={'date':'2026-10-22','scope':'CPU actualarchivedreadback and actualContinuousHandMapping; no newphysicsscene, no driveorstatewrite','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'cap_contact_controller.py',mappingfile,mf,snapfile,proposal]},'physics_tick':s['physics_tick'],'actual_primary_joint_positions_rad':actual.tolist(),'actual_primary_sdk_unclipped':(1000*(1-(actual-m.lower)/m.width)).tolist(),'actual_drive_sdk6_unclipped':sdkdrive.tolist(),'actual_drive_joint_positions_rad':drive.tolist(),'actual_primary_max_joint_limit_excess_rad':float(actual_excess.max()),'actual_drive_max_joint_limit_excess_rad':float(drive_excess.max()),'existing_actual_readback_tolerance_rad':1e-6,'actual_readback_primary_validation_passed':True,'actual_readback_drive_validation_passed':True,'actual_mimic_joint_observation_not_forced_to_command_coupling':True,'screened_open_sdk6':candidate['open_sdk6'],'screened_close_sdk6':candidate['first_contact_sdk6'],'pair_actuator_indices':[3,5],'actual_candidate_endpoint_mapping_passed':False,'constructor_passed':constructed,'proposal_returned':output is not None,'constructor_or_proposal_exception':exception,'actual_q_not_clipped':qdict==actual_before,'actual_drive_not_changed':np.array_equal(drive,drive_before),'reason':'Currentactualstaticreferencelefthandisfactoryopen; candidateopenrequiresactualpreparationoflittle/ring/middleandopposition. Refuseoutside-screenedstartinstead ofpretendingreadbackisprepared.','no_control_parameters_changed':True,'new_physical_trials':0,'actual_grasp_or_turn_success':False}
m.state(candidate['open_sdk6']);m.state(candidate['first_contact_sdk6']);out['actual_candidate_endpoint_mapping_passed']=True
assert not constructed and 'screened posture' in exception['message'];assert out['actual_q_not_clipped'] and out['actual_drive_not_changed']
(P/'cap_contact_controller_actual_readback_smoke.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
