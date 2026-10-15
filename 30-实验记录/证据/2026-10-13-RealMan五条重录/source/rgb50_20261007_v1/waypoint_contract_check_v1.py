"""Zero-step CPU fixtures for control-flow contracts; not physical validation."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
import ast,hashlib,json,sys
import numpy as np
import full_native_approach_waypoint_v3 as helper

work=Path(__file__).resolve().parent
class Pose:
    def __init__(self,m=None):self.m=np.eye(4) if m is None else np.asarray(m).copy()
    def to_transformation_matrix(self):return self.m.copy()
class Model:
    def __init__(self):self.q=np.zeros(36);self.pose=Pose()
    def get_base_pose(self):return self.pose
    def set_base_pose(self,p):self.pose=p
    def get_qpos(self):return self.q.copy()
    def set_qpos(self,q,full=True):self.q=np.asarray(q).copy()
    def set_move_group(self,name):self.group=name
class PK:
    def get_link_names(self):return ['l_inspire_base_link','r_inspire_base_link']
    def compute_IK_CLIK(self,index,target,seed,mask,**kwargs):
        calls['IK_targets'].append(target.m.copy());q=np.zeros(36);q[:3]=target.m[:3,3]
        return q,True,None
class Auditor:
    def __init__(self,task):
        self.model=Model();self.initial_sim_q=np.zeros(36)
        self.sim_joints=['arm'+str(i) for i in range(6)]+['frozen'+str(i) for i in range(30)]
        self.pk=PK();self.Pose=Pose;self.allowed={frozenset(('a','b'))}
        shape=SimpleNamespace(get_collision_geometry=lambda:'full_native_shape')
        self.static={name:SimpleNamespace(shapes=[shape],pose=Pose(),shape_poses=[np.eye(4)])
                     for name in ('bottle','table')}
        self.fcl=SimpleNamespace(CollisionObject=lambda geom:geom,
            FCLObject=lambda *args:args)
        self.art=SimpleNamespace(active_joints=[SimpleNamespace(limits=np.array([[-3.,3.]])) for _ in range(36)])
    def audit(self,vertices,side,**kwargs):
        calls['audit'].append({'vertices':vertices,'side':side,**kwargs})
        return {'passed':mode!='audit_reject'}
class World:
    def __init__(self,models,obstacles):
        calls['obstacles'].append(obstacles)
    def get_allowed_collision_matrix(self):return SimpleNamespace(set_entry=lambda *args:calls['ACM'].append(args))
    def set_qpos_all(self,q):pass
    def check_collision(self):return ['real_collision'] if mode=='start_collision' else []
class Planner:
    def __init__(self,world):pass
    def plan(self,start,goals,**kwargs):
        calls['RRT'].append(kwargs);path=np.vstack([start,goals[0]])
        if mode=='bad_endpoint':path[-1,0]+=2*np.pi
        return 'Exact solution',path

fma=ModuleType('full_mesh_path_audit');fma.FullMeshPathAuditor=Auditor
fma.joint_limit_audit=lambda *args: {'passed':True}
mplib=ModuleType('mplib');mplib.PlanningWorld=World
ompl=ModuleType('mplib.pymp.planning.ompl');ompl.OMPLPlanner=Planner
carry=ModuleType('full_native_carry_planner_v31')
def topp(task,path,side,*,time_step):
    calls['TOPP'].append(np.asarray(path).copy())
    return np.asarray(path).copy(),np.zeros_like(path),np.arange(len(path))*time_step,{'audit_vertices':np.asarray(path).tolist()}
carry.official_topp_polyline=topp
for name,module in [('full_mesh_path_audit',fma),('mplib',mplib),
    ('mplib.pymp.planning.ompl',ompl),('full_native_carry_planner_v31',carry)]:
    sys.modules[name]=module

physical=np.zeros(36)
entity=SimpleNamespace(get_qpos=lambda:physical.copy(),get_qvel=lambda:np.zeros(36),root_pose=Pose())
goal=np.eye(4);goal[:3,3]=[.1,.2,.3]
joint=SimpleNamespace(get_drive_target=lambda:np.array([.2]))
robot=SimpleNamespace(left_entity=entity,_hand_joints={'hand':joint},
    left_arm_joints_name=['arm'+str(i) for i in range(6)],
    right_arm_joints_name=['arm'+str(i) for i in range(6)],
    _trans_from_gripper_to_endlink=lambda target_pose,arm_tag:Pose(goal))
task=SimpleNamespace(robot=robot,bottle=SimpleNamespace(get_pose=lambda:Pose()),
    scene=SimpleNamespace(step=lambda:(_ for _ in ()).throw(AssertionError('No scene step allowed'))))
direct_success={'status':'Success','position':np.zeros((2,6)),
    'planning_metadata':{'physical_state_unchanged':True,'no_attached_objects':True}}
def direct(*args,**kwargs):
    calls['direct'].append(kwargs)
    if mode=='direct_success':return direct_success
    if mode=='physical_write':physical[0]=.1
    return {'status':'Fail','reason':'Retained original rejection fixture',
        'planning_metadata':{'physical_state_unchanged':True,'no_attached_objects':True}}
helper.original.plan_free_approach=direct
results=[]
for mode in ('direct_success','fallback_success','audit_reject','bad_endpoint','start_collision','physical_write'):
    physical[:]=0
    calls={name:[] for name in ('IK_targets','audit','obstacles','ACM','RRT','TOPP','direct')}
    try:
        result=helper.plan_free_approach(task,[0]*7,'right',random_seed=100025,ik_attempts=2)
        assert len(calls['direct'])==1
        if mode=='direct_success':
            assert result is direct_success and not calls['RRT'] and not calls['audit']
        elif mode=='fallback_success':
            assert result['status']=='Success' and result['planning_metadata']['physical_state_unchanged']
            high=np.asarray(result['planning_metadata']['via_palm_world_matrix'])
            assert np.array_equal(high[:3,:3],goal[:3,:3])
            assert np.max(abs((high[:3,3]-goal[:3,3])-[0,0,.08]))<1e-15
            assert len(calls['RRT'])==2 and len(calls['TOPP'])==1 and len(calls['audit'])==1
            assert calls['audit'][0]['min_bottle_clearance_m']==.0005
            assert calls['audit'][0]['min_table_clearance_m']==0 and calls['audit'][0]['min_self_clearance_m']==0
            assert calls['ACM']==[('a','b',True)] or calls['ACM']==[('b','a',True)]
            assert len(calls['obstacles'][0])==2
        elif mode in ('audit_reject','bad_endpoint','start_collision'):
            assert result['status']=='Fail' and result['planning_metadata']['physical_state_unchanged']
        elif mode=='physical_write':raise AssertionError('Physical write fixture was not rejected')
        results.append({'case':mode,'passed':True,'RRT_calls':len(calls['RRT']),
                        'TOPP_calls':len(calls['TOPP']),'full_mesh_gate_calls':len(calls['audit'])})
    except RuntimeError as error:
        if mode!='physical_write':raise
        assert 'changed physical' in str(error)
        results.append({'case':mode,'passed':True,'unauthorized_state_change_rejected':True})

source=work/'full_native_approach_waypoint_v3.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
        expression=ast.unparse(node.func)
        assert node.func.attr not in ('step','set_pose','set_drive_target','set_drive_velocity_target','set_qvel')
        if node.func.attr in ('set_qpos','set_base_pose','set_move_group'):
            assert ast.unparse(node.func.value)=='model',expression
overlay=work/'expert_generation_v9_waypoint.py'
overlay_tree=ast.parse(overlay.read_text(encoding='utf-8'))
assert not any('v11' in ast.unparse(node) for node in ast.walk(overlay_tree) if isinstance(node,(ast.Import,ast.ImportFrom)))
methods=[node.name for node in ast.walk(overlay_tree) if isinstance(node,ast.FunctionDef)]
assert not any(name in methods for name in ('step','dense','hold','plan_path','save_evidence'))
receipt={'passed':True,'scope':'Zero-step mocked CPU contracts only; not native IK/RRT/FCL/physics success',
    'physical_validation_completed':False,'GPU_initialized_by_fixture':False,
    'control_methods_not_overridden':True,'v11_not_imported':True,
    'physical_setters_or_scene_steps_not_present':True,'original_direct_success_returned_unchanged':True,
    'negative_endpoint_and_collision_fixtures_rejected':True,'cases':results,
    'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source,overlay,Path(__file__).resolve())}}
with (work/'waypoint_contract_receipt_v1.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
