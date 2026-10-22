from pathlib import Path
import importlib.util,json,copy,hashlib,ast
from types import SimpleNamespace as NS
import numpy as np
P=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task')
N=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin/integrations/realman_rm65b_gen4')
MAN=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/source_manifest.json')
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
mod=load(P/'cap_contact_controller.py','cap_controller_audit');servo=load(N/'teacher_legacy/continuous_servo.py','cap_audit_real_mapping')
checks=[]
def check(name,fn):
 try:
  detail=fn();checks.append({'name':name,'passed':True,'detail':detail})
 except Exception as e:
  checks.append({'name':name,'passed':False,'error':repr(e)});raise

def reject(fn,typ=(ValueError,RuntimeError)):
 try:fn()
 except typ:return
 raise AssertionError('Invalid input accepted')

def entity(id,name):return NS(per_scene_id=id,name=name)
CAP=entity(50,'cap');BODY=entity(51,'body')
def contact(obj,link,force,normal=(1,0,0),reverse=False,tangent=0):
 n=np.array(normal,float);im=n*force*.004+np.array([0,tangent,0]);pt=NS(normal=n,impulse=im,position=np.zeros(3));parts=[NS(entity=obj),NS(entity=link)]
 return NS(bodies=parts[::-1] if reverse else parts,points=[pt])
class Scene:
 def __init__(self):self.dt=.004;self.contacts=[];self.reads=0
 def get_timestep(self):return self.dt
 def get_contacts(self):self.reads+=1;return self.contacts
 def step(self):raise AssertionError('scene write/step forbidden')
class Joint:
 def __init__(self,q):self.q=q
 def get_drive_target(self):return [self.q]
 def set_drive_target(self,*a):raise AssertionError('drive write forbidden')
 def set_drive_velocity_target(self,*a):raise AssertionError('drive write forbidden')
class Robot:
 def __init__(self,m,q):
  self.communication_flag=False;self._manifest={'hands':{'left':{'channels':list(m.channels)},'right':{'channels':['wrong']*6}}};self.measured=dict(zip(m.channels,q));self._hand_joints={n:Joint(v) for n,v in zip(m.channels,q)}
 def _measured_positions(self):return self.measured.copy()
 def set_qpos(self,*a):raise AssertionError('state write forbidden')
 def set_gripper(self,*a):raise AssertionError('actuator write forbidden')
 def set_arm_joints(self,*a):raise AssertionError('actuator write forbidden')

def mapping(nested=False):
 data={'channels':['p'+str(i) for i in range(6)],'limits':{'p'+str(i):{'lower':0.,'upper':1.,'velocity':1.} for i in range(6)},'mimic':{}}
 if nested:
  data['mimic']={'f1':{'joint':'p3','multiplier':2.,'offset':.1},'f2':{'joint':'f1','multiplier':-3.,'offset':6.5}}
  data['limits'].update({'f1':{'lower':.1,'upper':2.1,'velocity':.07},'f2':{'lower':.2,'upper':6.2,'velocity':.03}})
 return servo.ContinuousHandMapping(data)
def fixture(*,m=None,q=.5,openq=0.,closeq=1.,pair=(3,4),**kw):
 m=m or mapping();q=np.broadcast_to(q,(6,)).astype(float).copy();op=1000*(1-np.broadcast_to(openq,(6,)));cl=1000*(1-np.broadcast_to(closeq,(6,)))
 task=NS(scene=Scene(),clock=NS(dt=.004,DEFAULT_STEPS=15),physics_step_count=0,robot=Robot(m,q))
 ctrl=mod.CapContactController(task,m,side='left',actor=CAP,open_sdk=op,close_sdk=cl,pair_actuator_indices=pair,**kw)
 return task,m,ctrl

def forces(task,index=0.,thumb=0.,reverse=False):
 task.scene.contacts=[contact(CAP,entity(80,'l_inspire_left_index_2'),index,(1,0,0),reverse),contact(CAP,entity(81,'l_inspire_left_thumb_4'),thumb,(-1,0,0),reverse)]
def state(c):return (c.target_q.tolist(),c.filtered.tolist(),c.latched.tolist(),c.last_tick,copy.deepcopy(c.cached),c.ready_streak)
def propose(task,c):
 before=copy.deepcopy(task.robot.measured);drive={n:j.q for n,j in task.robot._hand_joints.items()};out=c.propose(task.physics_step_count)
 assert task.robot.measured==before and drive=={n:j.q for n,j in task.robot._hand_joints.items()};return out

def execute(task,m,out):
 # CPU harness changes mock readback only; this is not simulator execution.
 q=m.lower+(1-np.array(out['sdk6'])/1000)*m.width
 for n,v in zip(m.channels,q):task.robot._hand_joints[n].q=v;task.robot.measured[n]=v
 task.physics_step_count+=15

def contacts_test():
 s=Scene();idx=entity(3,'l_inspire_left_index_2');thumb=entity(4,'l_inspire_left_thumb_4')
 s.contacts=[contact(CAP,idx,.7,tangent=100),contact(CAP,thumb,.8,(-1,0,0)),contact(entity(999,'cap'),idx,500),contact(BODY,idx,500),contact(CAP,entity(7,'r_inspire_right_index_2'),500),contact(CAP,entity(8,'l_other_index'),500)]
 x=mod.read_hand_contacts(s,NS(actor=CAP),'left');assert np.allclose(x['forces_N'],[0,0,0,.7,.8]) and x['opposed'];assert len(x['contacts'])==2
 s.contacts=[contact(CAP,idx,.7,(-1,0,0),True),contact(CAP,thumb,.8,(1,0,0),True)]
 y=mod.read_hand_contacts(s,CAP,'left');assert y['opposed'] and np.allclose(y['forces_N'],x['forces_N'])
 reject(lambda:mod.read_hand_contacts(s,CAP,'bogus'))
 s.dt=0;reject(lambda:mod.read_hand_contacts(s,CAP,'left'))
 s.dt=np.nan;reject(lambda:mod.read_hand_contacts(s,CAP,'left'))
 return {'normal_only':True,'entity_ID_exact':True,'side_and_actual_inspire_prefix':True,'order_reversal':True}
check('contact_entity_id_normal_impulse_side_and_orientation',contacts_test)

def no_write_idempotent():
 t,m,c=fixture();forces(t,.8,.8);a=propose(t,c);s=state(c);reads=t.scene.reads;a['sdk6'][0]=-99;b=propose(t,c);assert state(c)==s and t.scene.reads==reads and b['sdk6'][0]>=0 and not b['grasp_or_turn_success_inferred']
 return {'no_mock_state_or_drive_write':True,'cache_defensive_copy':True,'same_tick_no_double_filter_streak':True}
check('no_write_and_same_tick_idempotence',no_write_idempotent)

def wrong_clock():
 for bad in [True,-15,1,0.5,np.nan,'0']:
  t,m,c=fixture();s=state(c);reject(lambda:c.propose(bad));assert state(c)==s
 for dt in [.001,.008,np.nan,np.inf]:
  t,m,c=fixture();t.scene.dt=dt;s=state(c);reject(lambda:c.propose(0));assert state(c)==s
 t,m,c=fixture();t.clock.DEFAULT_STEPS=14;reject(lambda:c.propose(0))
 t,m,c=fixture();t.clock.dt=.001;reject(lambda:c.propose(0))
 t,m,c=fixture();t.physics_step_count=15;reject(lambda:c.propose(0))
 t,m,c=fixture();out=propose(t,c);execute(t,m,out);t.physics_step_count=30;reject(lambda:c.propose(30))
 t,m,c=fixture();out=propose(t,c);t.physics_step_count=15;reject(lambda:c.propose(15))
 return {'bad_type_fraction_negative_stale_and_skipped_ticks_rejected':True,'runtime_dt_and_driver_revalidated':True,'unexecuted_prior_proposal_rejected':True}
check('wrong_clock_and_invalid_tick_rejection',wrong_clock)

def latch_pdlag():
 t,m,c=fixture(q=.1);t.robot.measured['p3']=.8;forces(t,.1,0);out=propose(t,c);q=m.lower+(1-np.array(out['sdk6'])/1000)*m.width
 assert abs(q[3]-.1048)<1e-12 and out['latched_index_thumb']==[True,False]
 # Constructor must use drive rather than a measured PD error as the first segment origin.
 q=np.full(6,.1);task=NS(scene=Scene(),clock=NS(dt=.004,DEFAULT_STEPS=15),physics_step_count=0,robot=Robot(m,q));task.robot.measured.update(dict(zip(m.channels,np.full(6,.8))))
 cc=mod.CapContactController(task,m,side='left',actor=CAP,open_sdk=np.full(6,1000.),close_sdk=np.zeros(6));a=propose(task,cc);qq=1-np.array(a['sdk6'])/1000;assert np.allclose(qq,.1048)
 return {'first_contact_latch_actual_q_slew_rad':float(q[3]-.1) if False else .0048,'first_segment_from_actual_drive_despite_pd_error':True}
check('first_contact_latch_and_drive_origin',latch_pdlag)

def independent_feedback():
 t,m,c=fixture();forces(t,1.,.1);a=propose(t,c);execute(t,m,a);prior=c.target_q.copy();b=propose(t,c);delta=c.target_q-prior;assert delta[3]<0 and delta[4]>0
 t,m,c=fixture(pair=(3,5));forces(t,1.,.1);a=propose(t,c);execute(t,m,a);prior=c.target_q.copy();b=propose(t,c);delta=c.target_q-prior;assert delta[3]<0 and delta[5]>0 and delta[4]>0;assert b['pair_actuator_indices']==[3,5]
 t,m,c=fixture(q=.6,openq=.8,closeq=.2,pair=(3,5));forces(t,1.,.1);a=propose(t,c);execute(t,m,a);prior=c.target_q.copy();b=propose(t,c);delta=c.target_q-prior;assert delta[3]>0 and delta[5]<0
 return {'independent_index_thumb_force_signs':True,'opposition_channel5_supported':True,'reverse_closure_sign_supported':True}
check('independent_force_feedback_pair_mapping_and_closure_sign',independent_feedback)

def speed_and_screen():
 t,m,c=fixture(m=mapping(True));forces(t,0,0);prev=c.target_q.copy();last=None;maxprimary=maxfollower=0.
 for _ in range(60):
  a=propose(t,c);now=c.target_q.copy();sdk_rate=-1000*(now-prev)/(m.width*.06);pos,vel=m.state(a['sdk6'],sdk_rate)
  assert max(abs(x) for x in vel.values())<=.08+1e-10 and all(abs(vel[n])<=m.limits[n]['velocity']+1e-10 for n in vel)
  maxprimary=max(maxprimary,max(abs(vel[n]) for n in m.channels));maxfollower=max(maxfollower,max(abs(vel[n]) for n in m.followers));prev=now;execute(t,m,a)
 # At screenedclose, low force must not cause blind exploration past endpoint.
 t,m,c=fixture(q=.6,openq=.4,closeq=.6,pair=(3,5));forces(t,.1,.1)
 for _ in range(20):
  a=propose(t,c);assert np.all(c.target_q<=.6+1e-12) and np.all(c.target_q>=.4-1e-12);execute(t,m,a)
 reject(lambda:fixture(rate_cap_rad_s=.081));reject(lambda:fixture(rate_cap_rad_s=.16))
 return {'max_primary_rad_s':maxprimary,'max_nested_follower_rad_s':maxfollower,'nested_negative_mimic_limits_obeyed':True,'screened_close_not_exceeded':True}
check('sdk_primary_nested_mimic_speed_and_screened_bounds',speed_and_screen)

def invalid_actual_atomic():
 for q in [np.nan,np.inf,-100.,100.]:
  t,m,c=fixture();t.robot.measured['p3']=q;forces(t,.2,.2);s=state(c);reject(lambda:c.propose(0));assert state(c)==s
 for q in [np.nan,np.inf,-10.,10.]:
  t,m,c=fixture();t.robot._hand_joints['p3'].q=q;s=state(c);reject(lambda:c.propose(0));assert state(c)==s
 t,m,c=fixture();forces(t,.2,.2);t.scene.contacts[0].points[0].normal=np.array([np.nan,0,0]);s=state(c);reject(lambda:c.propose(0));assert state(c)==s
 for pair in [(3,3),(3,6),(True,4),(3.5,4),(3,)]:reject(lambda:fixture(pair=pair))
 return {'nonfinite_extreme_actual_and_drive_rejected':True,'invalid_contacts_rejected':'no partial filter/latch/target state commit','pair_validation':True}
check('invalid_actual_readback_contact_and_atomic_failure',invalid_actual_atomic)

def readiness():
 t,m,c=fixture();forces(t,.8,.8)
 for i in range(3):
  a=propose(t,c);assert a['ready_boundaries']==i+1 and a['ready']==(i==2);s=state(c);propose(t,c);assert state(c)==s;execute(t,m,a)
 forces(t,.8,0);a=propose(t,c);assert not a['ready'] and a['ready_boundaries']==0
 return {'requires_three_distinct_opposed_boundaries':True,'contact_loss_resets':True,'never_grasp_turn_success':True}
check('three_boundary_diagnostic_readiness_only',readiness)

def actual_mapping():
 meta=json.loads(MAN.read_text())['hands']['left'];m=servo.ContinuousHandMapping(meta);op=np.array([1000,1000,1000,1000,1000,500.]);cl=np.array([1000,1000,1000,990,1000,490.]);q=m.lower+(1-op/1000)*m.width
 # Explicit real manifest, not a constructedsix-joint map.
 task=NS(scene=Scene(),clock=NS(dt=.004,DEFAULT_STEPS=15),physics_step_count=4035,robot=Robot(m,q));task.robot._manifest={'hands':{'left':meta}}
 c=mod.CapContactController(task,m,side='left',actor=CAP,open_sdk=op,close_sdk=cl,pair_actuator_indices=(3,5));a=propose(task,c);pos,vel=m.state(a['sdk6']);assert len(pos)>6 and all(np.isfinite(list(pos.values())))
 return {'actual_manifest_mapping_loaded':True,'primary6':list(m.channels),'mapped_joint_count':len(pos),'actual_metadata_constructor_and_proposal_passed':True}
check('actual_retained_manifest_and_continuous_mapping_interface',actual_mapping)

def no_write_ast():
 tree=ast.parse((P/'cap_contact_controller.py').read_text());forbidden={'set_pose','set_qpos','set_qvel','set_velocity','set_angular_velocity','set_drive_target','set_drive_velocity_target','add_force_torque','step','set_kinematic_target'}
 calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden];assert not calls
 return {'forbidden_simulator_actuator_call_sites':calls}
check('static_no_simulator_or_drive_writer_calls',no_write_ast)
def sdk_endpoint_types_and_mimic_positions():
 m=mapping();q=np.full(6,.5)
 def call(op,cl):
  task=NS(scene=Scene(),clock=NS(dt=.004,DEFAULT_STEPS=15),physics_step_count=0,robot=Robot(m,q))
  return mod.CapContactController(task,m,side='left',actor=CAP,open_sdk=op,close_sdk=cl)
 for bad in [[True]*6,['1']*6,[1001.]*6,[-1.]*6,[np.nan]*6,[0.]*5]:reject(lambda:call(bad,[0.]*6))
 meta={'channels':['p'+str(i) for i in range(6)],'limits':{'p'+str(i):{'lower':0.,'upper':1.,'velocity':1.} for i in range(6)},'mimic':{'f1':{'joint':'p3','multiplier':2.,'offset':0.}}}
 meta['limits']['f1']={'lower':0.,'upper':1.,'velocity':1.}
 mm=servo.ContinuousHandMapping(meta)
 reject(lambda:fixture(m=mm,q=.2,openq=0,closeq=1))
 return {'raw_bool_string_nonfinite_range_shape_rejected':True,'mimic_endpoint_position_limits_checked':True}
check('raw_sdk_types_and_complete_mimic_position_endpoints',sdk_endpoint_types_and_mimic_positions)

out={'date':'2026-10-22','scope':'meaningful CPU/mocks and actualContinuousHandMapping interface only; no SAPIEN scene, actual contact or grasp/turn validation','all_passed':all(x['passed'] for x in checks),'checks':checks,'source_sha256':{'controller':hashlib.sha256((P/'cap_contact_controller.py').read_bytes()).hexdigest(),'actual_continuous_mapping':hashlib.sha256((N/'teacher_legacy/continuous_servo.py').read_bytes()).hexdigest(),'retained_source_manifest':hashlib.sha256(MAN.read_bytes()).hexdigest()},'controller_source_changed_by_audit':True,'external_mapping_or_manifest_modified':False,'actual_physical_trials':0,'actual_cap_contact_validated':False,'grasp_or_turn_success':False,'original_empty_task_or_frozen_assembly_modified':False}
(P/'cap_contact_controller_cpu_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
