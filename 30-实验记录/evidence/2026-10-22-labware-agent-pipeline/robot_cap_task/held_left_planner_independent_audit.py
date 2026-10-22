from pathlib import Path
from types import SimpleNamespace as S
import sys,copy,json,hashlib,inspect,ast,time
import numpy as np
ROOT=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task')
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
import held_left_planner as h
SHA=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
N=h.install_held_left_planner();Held=N.fma.FullMeshPathAuditor;checks=[]
def check(name,fn):
 try:r=fn();checks.append({'name':name,'passed':True,'detail':r})
 except Exception as exc:checks.append({'name':name,'passed':False,'error':repr(exc)});raise

def reject(fn):
 try:fn()
 except (ValueError,RuntimeError):return
 raise AssertionError('Input was not refused')
class Pose:
 def __init__(self,m=None):self.m=np.eye(4) if m is None else (m.m.copy() if isinstance(m,Pose) else np.asarray(m).copy())
 def to_transformation_matrix(self):return self.m.copy()
class Obj:
 def __init__(self,name,pose=None,shapes=None,shape_poses=None):self.name=name;self.pose=Pose(pose);self.shapes=[] if shapes is None else shapes;self.shape_poses=[] if shape_poses is None else shape_poses
class Joint:
 def __init__(self,n,lim):self.name=n;self.limits=np.array([lim])
class AuxModel:
 def __init__(self,q,names,idx):self.q=np.array(q).copy();self.names=names;self.idx=idx
 def set_qpos(self,q,full=True):self.q=np.array(q).copy()
 def set_move_group(self,x):self.group=x
 def get_move_group_joint_indices(self):return self.idx
 def get_move_group_joint_names(self):return [self.names[i] for i in self.idx]

snap=json.loads((ROOT/'static_hold_probes/held115_static_reference_v1/actual_terminal_snapshot.json').read_text());JN=snap['joint_names'];Q=np.array(snap['q36']);L=[JN.index('l_joint'+str(i)) for i in range(1,7)];RR=[JN.index('r_joint'+str(i)) for i in range(1,7)]
RI='r_inspire_right_index_2';RM='r_inspire_right_middle_2';RN='r_inspire_right_ring_2';LI='l_inspire_left_index_2'
CATEGORIES=['bottle','cap',*[f'pill_{i:02d}' for i in range(12)],'table']
def make_auditor(*,collision=None,allow=None,mid_only=False):
 a=object.__new__(Held);a.sim_joints=list(JN);a.initial_sim_q=Q.copy();a.old_model_q=Q.copy();a.art=S(qpos=Q.copy(),active_joints=[Joint(n,l) for n,l in zip(JN,snap['qlimits'])]);a.task=S(robot=S(left_arm_joints_name=['l_joint'+str(i) for i in range(1,7)],right_arm_joints_name=['r_joint'+str(i) for i in range(1,7)]));a.model=AuxModel(Q,JN,L);a.objects={n:Obj(n) for n in [RI,RM,RN,LI]};a.vertices={n:np.array([[-.01,-.01,-.01],[.01,.01,.01]]) for n in a.objects};a.static={n:Obj(n) for n in CATEGORIES};a.static_boxes={n:np.array([[-1.,-1.,-1.],[1.,1.,1.]]) for n in CATEGORIES};a.motion_coefficients={n:np.zeros(36) for n in a.objects}
 for n in [RI,RM,RN]:a.motion_coefficients[n][RR[0]]=.1
 a.motion_coefficients[LI][L[0]]=.1
 a.self_pairs=[(LI,RI)];a.allowed=set();a.metadata={'held_contact_baseline':{'pairs':[]}};a.measured_fixed_contact_allowlist=set(allow or [('bottle',RI)]);a.links={n:S(entity=S(pose=Pose())) for n in a.objects};a.initial_scene_poses={n:np.eye(4) for n in a.objects};a.distance_calls=[]
 def distance(first,second):
  a.distance_calls.append((first.name,second.name));hit=(first.name,second.name)==collision
  if mid_only:hit=hit and abs(a.model.q[L[0]]-(Q[L[0]]+.02))<.006
  return (0.,True) if hit else (.1,False)
 a._distance=distance
 return a

def baseline(scene,parts,ids):return h.measured_contact_baseline(scene,parts,ids)
def pt(force):return S(impulse=np.array([force*.004,0,0]),normal=np.array([1.,0,0]),position=np.zeros(3))
def con(a,b,force=.2,points=True):return S(bodies=[S(entity=a),S(entity=b)],points=[pt(force)] if points else [])
def selection():
 body=S(name='actualbody',per_scene_id=1);cap=S(name='actualcap',per_scene_id=2);pill=S(name='actualpill',per_scene_id=3);ri=S(name=RI,per_scene_id=101);rm=S(name=RM,per_scene_id=102);rn=S(name=RN,per_scene_id=103);li=S(name=LI,per_scene_id=104);imposter=S(name=RI,per_scene_id=999);fakebody=S(name='actualbody',per_scene_id=9999)
 parts={1:('bottle','body',body),2:('cap','cap',cap),3:('pill_00','pill',pill)};ids={RI:101,RM:102,RN:103,LI:104}
 contacts=[con(body,ri,0),con(cap,ri,.5),con(body,rm,.3),con(cap,rm,0),con(cap,imposter,.9),con(body,li,.2),con(body,rn,.2),con(pill,ri,.2),con(fakebody,ri,.2),con(body,ri,points=False)]
 scene=S(get_contacts=lambda:contacts,get_timestep=lambda:.004);allowed,rows=baseline(scene,parts,ids);assert allowed=={('bottle',RI),('cap',RI),('bottle',RM)};assert all(x['sourcepart'] in ['body','cap'] for x in rows)
 scene.get_timestep=lambda:np.nan;reject(lambda:baseline(scene,parts,ids));scene.get_timestep=lambda:0;reject(lambda:baseline(scene,parts,ids));scene.get_timestep=lambda:.004
 bad=con(cap,ri,.2);bad.points[0].impulse[0]=np.nan;scene.get_contacts=lambda:[bad];reject(lambda:baseline(scene,parts,ids))
 return {'allowed_exact':sorted(allowed),'right_same_name_wrong_ID_blocked':True,'left_pill_ring_empty_unloadedcap_and_wrongobject_blocked':True,'invalid_clock_contact_refused':True}
check('actual_contact_ID_and_sourcepart_only_allowlist',selection)

def audit_case(pair,allow=None,moving_right=False,mid=False):
 a=make_auditor(collision=pair,allow=allow,mid_only=mid);old=a.art.qpos.copy();goal=Q[L].copy();goal[0]+=.04
 path=goal[None,:]
 if moving_right:
  row=Q.copy();row[RR[0]]+=.02;path=row[None,:]
 out=a.audit(path,'left',include_samples=True);assert np.array_equal(a.art.qpos,old);return a,out

def real_kernel():
 a,p=audit_case((RI,'bottle'));assert p['passed'] and (RI,'bottle') not in a.distance_calls
 cases=[(RN,'bottle'),(RM,'cap'),(LI,'cap'),(LI,'bottle'),(LI,'table'),(RI,'pill_11'),(LI,RI)]
 res=[]
 for pair in cases:
  a,o=audit_case(pair);assert not o['passed'] and o['collisions_first_64'];res.append({'pair':pair,'blocked':True})
 a,o=audit_case((RI,'bottle'),moving_right=True);assert not o['passed'] and (RI,'bottle') in a.distance_calls
 a,o=audit_case((LI,'cap'),mid=True);assert not o['passed'] and o['audited_samples']>2;assert o['collisions_first_64'][0]['interval_fraction'] not in [0.,1.]
 a,o=audit_case(None);assert o['passed'];assert {b for _,b in a.distance_calls if b in CATEGORIES}==set(CATEGORIES);assert set(o['thresholds_m'])==set(CATEGORIES+['self'])
 return {'negative_pairs':res,'moving_baseline_pair_not_exempt':True,'interior_path_collision_blocked_despite_clear_endpoints':True,'all15_static_categories_and_self_visited':True,'patched_method_code':Held.audit.__code__.co_filename}
check('actual_patched_mesh_audit_kernel_negative_controls',real_kernel)

class ACM:
 def __init__(self):self.allowed=set()
 def set_entry(self,a,b,v):
  if v:self.allowed.add(frozenset([a,b]))
class World:
 latest=None;collisions=[]
 def __init__(self,models,obstacles):self.models=models;self.obstacles=obstacles;self.acm=ACM();World.latest=self
 def get_allowed_collision_matrix(self):return self.acm
 def get_object_names(self):return [o.name for o in self.obstacles]
 def set_qpos_all(self,q):self.models[0].q[self.models[0].idx]=q
 def check_collision(self):return [p for p in World.collisions if frozenset(p) not in self.acm.allowed]
class OMPL:
 def __init__(self,w):self.world=w
 def plan(self,start,goals,**kw):return 'Exact solution',np.array([start,goals[0]])
class FCL:
 FCLObject=Obj
 @staticmethod
 def CollisionObject(x):return x

def rrt_logic():
 import mplib
 import mplib.pymp.planning.ompl as ompl
 g=N.strict_rrt_polyline.__globals__;saved={k:g[k] for k in ['exact_branch_planning_model','collision_summary','clearance_shortcut']};oldworld=mplib.PlanningWorld;oldompl=ompl.OMPLPlanner
 try:
  mplib.PlanningWorld=World;ompl.OMPLPlanner=OMPL
  g['exact_branch_planning_model']=lambda a,q,ix,goal:(AuxModel(q,JN,list(ix)),{'mock_search_model':True});g['collision_summary']=lambda w:w.check_collision();g['clearance_shortcut']=lambda path,valid,coeff,**kw:(path,{'test_double_shortcut':True})
  def call(pair=None,moving=False):
   a=make_auditor();a.Pose=Pose;a.fcl=FCL;World.collisions=[] if pair is None else [pair];indices=RR if moving else L;goal=Q[indices].copy();goal[0]+=.01
   return N.strict_rrt_polyline(a,Q.copy(),indices,goal,'right' if moving else 'left',deadline=time.monotonic()+5,checkpoint=lambda *a,**k:None)
  p,d=call((RI,'teacher_bottle'));assert p is not None and (frozenset([RI,'teacher_bottle']) in World.latest.acm.allowed)
  assert set(World.latest.get_object_names())=={'teacher_'+s for s in CATEGORIES}
  for pair in [(RN,'teacher_bottle'),(RM,'teacher_cap'),(LI,'teacher_cap'),(LI,'teacher_bottle'),(LI,RI),(LI,'teacher_table'),(RI,'teacher_pill_11')]:
   p,d=call(pair);assert p is None and d['reason']=='Actual full-native OMPL start collision; no relaxation'
  reject(lambda:call((RI,'teacher_bottle'),moving=True))
  return {'teacher_category_names_equal_ACM_obstacles':True,'baseline_only_frozen_link':True,'unseen_right_leftcap_leftbody_leftright_table_pill_rejected':True,'actual_OMPL_search_executed':False,'scope':'actual wrapper with controlledPlanningWorld/OMPL test doubles, not a path feasibility proof'}
 finally:
  mplib.PlanningWorld=oldworld;ompl.OMPLPlanner=oldompl
  for k,v in saved.items():g[k]=v
check('actual_RRT_wrapper_ACM_names_and_unseen_pair_rejection',rrt_logic)

def source_readonly():
 tree=ast.parse(Path(h.__file__).read_text());bad={'set_pose','set_qpos','set_qvel','set_velocity','set_angular_velocity','set_collision_groups','set_collision_group','create_drive','set_kinematic_target','attach_object','step'};calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in bad];assert not calls
 auditnames=set(Held.audit.__code__.co_names);assert 'FullMeshPathAuditor' not in auditnames
 assert N.strict_rrt_polyline.__globals__['fma'] is N.fma
 assert h.install_held_left_planner() is N
 return {'root_planner_physical_writer_calls':calls,'compiled_audit_does_not_resolve_old_class_name':True,'RRT_globals_use_current_private_fma_module':True,'installation_idempotent':True}
check('private_namespaces_no_physical_writes',source_readonly)
def constructor_part_isolation():
 base=Held.__mro__[1];original_init=base.__init__;original_rigid=h.rigid;oldconv=N.fma._convert_component_scaled;oldverts=N.fma._link_vertices
 FCL.Convex=staticmethod(lambda vertices,faces:S(vertices=vertices,faces=faces))
 def ent(i,name):
  e=S(per_scene_id=i,name=name,pose=Pose());e.component=S(name=name);return e
 body=ent(1,'actualbody_only');cap=ent(2,'actualcap_only');pills=[ent(10+i,'pillentity'+str(i)) for i in range(12)];ri=ent(101,RI)
 task=S(assembly=S(body=body,cap=cap,tablets=pills),scene=S(get_timestep=lambda:.004,get_contacts=lambda:[con(body,ri,.5),con(cap,ri,.2)]),physics_step_count=4410)
 def fake_init(self,task,*args,**kwargs):
  self.fcl=FCL;self.Pose=Pose;self.static={'bottle':Obj('old_compound_with_cap_and_pills',shapes=[object()]*99),'table':Obj('table')};self.static_boxes={'table':np.array([[-1,-1,-1],[1,1,1]])};self.metadata={'closed_assembly_obstacle':{'old_union':True}};self.links={RI:S(entity=ri)}
 try:
  base.__init__=fake_init;h.rigid=lambda e:e.component;N.fma._convert_component_scaled=lambda component,*args:Obj(component.name,shapes=[object()]);N.fma._link_vertices=lambda component:(np.array([[-.003,-.003,-.003],[.003,.003,.003]]),[])
  a=Held(task);assert a.static['bottle'].name=='actualbody_only' and len(a.static['bottle'].shapes)==1;assert set(a.static)==set(CATEGORIES);assert 'closed_assembly_obstacle' not in a.metadata
  tags=a.metadata['held_contact_baseline']['sourcepart_tags'];assert tags['bottle']['sourcepart']=='body' and tags['cap']['sourcepart']=='cap' and tags['pill_11']['sourcepart']=='pill';assert a.measured_fixed_contact_allowlist=={('bottle',RI),('cap',RI)}
  return {'preexisting_body_cap_pill_union_replaced_by_body_only':True,'all15_static_categories_retained':True,'actualpart_identity_tags_retained':True,'scope':'actual constructor with read-onlygeometry conversion test doubles; no actual mesh/physics claim'}
 finally:
  base.__init__=original_init;h.rigid=original_rigid;N.fma._convert_component_scaled=oldconv;N.fma._link_vertices=oldverts
check('constructor_rebuilds_independent_body_and_keeps_cap_pills',constructor_part_isolation)
def archived_actual_contact_readback():
 import gzip
 path=ROOT/'static_hold_probes/held115_static_reference_v1/world_kinematics_pre_post.json.gz'
 with gzip.open(path,'rt') as f:trace=json.load(f)
 last=trace[-1];raw=last['contacts_post'];entities={};names={}
 for c in raw:
  for name,ident in zip(c['body_names'],c['body_actor_ids']):
   assert name not in names or names[name]==ident;names[name]=ident;entities[ident]=S(name=name,per_scene_id=ident)
 parts={}
 for name,ident in names.items():
  if name=='derived115_closed_tablet_bottle_body':parts[ident]=('bottle','body',entities[ident])
  elif name=='derived115_closed_tablet_bottle_cap':parts[ident]=('cap','cap',entities[ident])
  elif name.startswith('derived115_closed_tablet_bottle_tablet_'):parts[ident]=('pill_'+name[-2:],'pill',entities[ident])
 ids={name:ident for name,ident in names.items() if name.startswith(('l_inspire_','r_inspire_'))}
 contacts=[S(bodies=[S(entity=entities[i]) for i in c['body_actor_ids']],points=[S(impulse=p['impulse_raw_world_Ns'],normal=p['normal_raw_world'],position=p['position_world_m']) for p in c['points']]) for c in raw]
 a,rows=baseline(S(get_timestep=lambda:last['dt_s'],get_contacts=lambda:contacts),parts,ids)
 assert ('cap',RI) in a and ('bottle',RI) in a and all(part in ['bottle','cap'] and link.startswith('r_inspire_') for part,link in a)
 return {'archived_actual_PhysX_contact_source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'physics_step':last['physics_step'],'baseline_pairs':rows,'selection_derived_from_actual_contacts_not_FCL':True,'scope':'offline reconstruction of recorded contact fields, no new scene/probe'}
check('archived_actual_static_terminal_contact_selection',archived_actual_contact_readback)


assert hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()==SHA
out={'date':'2026-10-22','scope':'CPU independent source and negative-control audit; actual patchedaudit and RRTwrapper with testdouble collision/FK/world oracles; no physical or OMPL pathsuccess','all_passed':all(x['passed'] for x in checks),'checks':checks,'source_sha256':{'held_left_planner':SHA,'native_bounded_fullmesh':hashlib.sha256(Path(N.fma.__file__).read_bytes()).hexdigest(),'native_acob':N.labware_source_receipt['native_expert_sha256']},'native_fma_path':N.fma.__file__,'actual_snapshot_q36_used':str(ROOT/'static_hold_probes/held115_static_reference_v1/actual_terminal_snapshot.json'),'actual_current_physics_model_built_here':False,'fullactualpath_verified':False,'new_physical_trials':0,'source_files_modified_by_auditor':False}
(ROOT/'held_left_planner_independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
