from pathlib import Path
import json,hashlib,importlib.util,copy
from types import SimpleNamespace as S
import numpy as np
P=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task');snapfile=P/'left_contact_probes/left115_first_contact_v5_true_surface/actual_terminal_snapshot.json';candidatefile=P.parent/'cap_geometry_probes/true_surface_closure_v4_v2/candidate.json';mappingfile=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin/integrations/realman_rm65b_gen4/teacher_legacy/continuous_servo.py');mf=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/source_manifest.json')
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
mod=load(P/'cap_contact_controller.py','ordered_independent_controller');mp=load(mappingfile,'ordered_independent_mapping');meta=json.loads(mf.read_text())['hands']['left'];mapping=mp.ContinuousHandMapping(meta);snap=json.loads(snapfile.read_text());candidate=json.loads(candidatefile.read_text());checks=[]
class Joint:
 def __init__(self,x):self.x=float(x)
 def get_drive_target(self):return [self.x]
 def set_drive_target(self,*a):raise AssertionError('No write API in CPU readback')
 def set_drive_velocity_target(self,*a):raise AssertionError('No write API in CPU readback')
def fixture(sdk=None,delta=None,force_fixed=None):
 sdk=np.array(snap['actual_drive_target24'][6:12] if sdk is None else sdk,float);drive=mapping.lower+(1-sdk/1000)*mapping.width
 if delta is not None:drive[delta[0]]+=delta[1]
 qdict=dict(zip(snap['joint_names'],snap['q36']));joints={n:Joint(q) for n,q in zip(mapping.channels,drive)};contacts=[]
 if force_fixed is not None:
  i,f=force_fixed;name='l_inspire_left_index_2' if i==3 else 'l_inspire_left_thumb_4';contacts=[S(bodies=[S(entity=S(name='cap',per_scene_id=104)),S(entity=S(name=name,per_scene_id=77))],points=[S(normal=np.array([1.,0,0]),impulse=np.array([f*.004,0,0]),position=np.zeros(3))])]
 robot=S(communication_flag=False,_manifest={'hands':{'left':meta}},_measured_positions=lambda:qdict.copy(),_hand_joints=joints)
 task=S(robot=robot,clock=S(dt=.004,DEFAULT_STEPS=15),physics_step_count=snap['physics_tick'],scene=S(get_timestep=lambda:.004,get_contacts=lambda:contacts))
 return task,joints,qdict,drive

def ctor(task,stage,**kw):return mod.CapContactController(task,mapping,side='left',actor=S(per_scene_id=104,name='cap'),open_sdk=stage['start_sdk6'],close_sdk=stage['end_sdk6'],force_targets_N=(.1,.1),pair_actuator_indices=(3,5),**kw)
def rejects(fn):
 try:fn()
 except (ValueError,RuntimeError) as e:return str(e)
 raise AssertionError('Expected rejection')
stages=candidate['ordered_actuator_continuations']
# Stage 1 starts from the real v5 actual drive; only stage 2 uses a labelled endpoint transport fixture.
for si,stage in enumerate(stages):
 fixed=5 if si==0 else 3;active=3 if si==0 else 5;sdk=None if si==0 else stage['start_sdk6'];task,joints,qdict,drive=fixture(sdk);rawq=qdict.copy();strict=rejects(lambda:ctor(task,stage));c=ctor(task,stage,allow_fixed_feedback_actuator=True);proposal=c.propose(task.physics_step_count);assert proposal['fixed_feedback_actuator_indices']==[fixed];assert c.target_q[fixed]==drive[fixed];assert qdict==rawq
 prev=c.target_q.copy();maxspeed=0.;count=1
 while abs(c.target_q[active]-c.close_q[active])>1e-12 and count<100:
  for name,q in zip(mapping.channels,c.target_q):joints[name].x=float(q)
  task.physics_step_count+=15;proposal=c.propose(task.physics_step_count);diff=(c.target_q-prev)/.06;assert abs(diff[fixed])==0 and np.all(abs(diff)<=c.caps+1e-10);sdknew=np.array(proposal['sdk6']);positions,velocities=mapping.state(sdknew,-1000*diff/mapping.width);assert all(abs(velocities[n])<=.08+1e-10 for n in mapping.joint_names);maxspeed=max(maxspeed,max(abs(v) for v in velocities.values()));prev=c.target_q.copy();count+=1
 assert count<100 and abs(c.target_q[active]-c.close_q[active])<1e-12
 checks.append({'case':stage['group']+'_actual_mapping_ordered_stage','passed':True,'actual_v5_drive_first_stage':si==0,'second_stage_endpoint_fixture_only':si==1,'strict_default_rejection':strict,'fixed_index':fixed,'fixed_drive_preserved_exact':c.target_q[fixed]==drive[fixed],'commands_to_endpoint_in_transport_fixture':count,'max_command_primary_or_mimic_speed_rad_s':maxspeed,'measured_q_not_clamped':qdict==rawq})
 # Fixed feedback force must stay observable without changing that drive, even with canonical point rounding mismatch.
 task,joints,qdict,drive=fixture(sdk,delta=(fixed,3e-7),force_fixed=(fixed,.2));c=ctor(task,stage,allow_fixed_feedback_actuator=True);pr=c.propose(task.physics_step_count);assert c.target_q[fixed]==drive[fixed] and not pr['latched_index_thumb'][0 if fixed==3 else 1];assert pr['contact']['forces_N'][3 if fixed==3 else 4]>.19
 checks.append({'case':stage['group']+'_fixed_actual_drive_offset_with_contact','passed':True,'offset_rad':3e-7,'fixed_force_observed_N':pr['contact']['forces_N'][3 if fixed==3 else 4],'fixed_drive_preserved_exact':True})
 task,*_=fixture(sdk,delta=(fixed,2e-6));reject=rejects(lambda:ctor(task,stage,allow_fixed_feedback_actuator=True));checks.append({'case':stage['group']+'_fixed_outside_screened_start_rejected','passed':True,'message':reject})
bothfixed={'start_sdk6':stages[0]['start_sdk6'],'end_sdk6':stages[0]['start_sdk6']};task,*_=fixture();message=rejects(lambda:ctor(task,bothfixed,allow_fixed_feedback_actuator=True));checks.append({'case':'both_feedback_actuators_fixed_rejected','passed':True,'message':message})
message=rejects(lambda:ctor(task,stages[0],allow_fixed_feedback_actuator=1));checks.append({'case':'non_boolean_optin_rejected','passed':True,'message':message})
for invalid in [[0,0,0,-.01,1000,348],[0,0,0,560,1000,1000.01]]:
 bad={'start_sdk6':stages[0]['start_sdk6'],'end_sdk6':invalid};message=rejects(lambda:ctor(task,bad,allow_fixed_feedback_actuator=True));checks.append({'case':'strict_sdk_interval_rejected','passed':True,'endpoint':invalid,'message':message})
out={'scope':'Finite actual native mapping / archived real v5 state-drive CPU API smoke. Transport/contact doubles are interface tests, not simulator stepping, physical contact, grip or task success.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'cap_contact_controller.py',snapfile,candidatefile,mappingfile,mf]},'all_passed':all(x['passed'] for x in checks),'check_count':len(checks),'checks':checks,'auditor_source_modified':False,'new_physical_trials':0,'original_v5_failure_preserved':True};(P/'ordered_controller_cpu_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n');print(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:x.item()))
