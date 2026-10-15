"""Read-only rigid carry/place geometry and opt-in existing-planner preflight.
No scene.step, SAPIEN qpos/drive writes, attachments, hardware IO or new workers.
Historical bottle_at_grasp_pose is required: the current slipped pose is not a
valid grip reference. Planning is disabled unless run_planning=True. Reachability
under the configured single-arm model does not prove full-body carry clearance.
"""
from __future__ import annotations
import math
import time
from types import SimpleNamespace
import numpy as np
try:
    from transforms3d.quaternions import mat2quat, quat2mat
except ModuleNotFoundError:
    # NumPy-only wxyz fallback for offline CPU checks; the server uses its
    # already installed official transforms3d implementation when available.
    def quat2mat(quaternion):
        q=np.asarray(quaternion,dtype=np.float64);n=float(q@q)
        if n<1e-24:raise ValueError('Zero quaternion')
        w,x,y,z=q/np.sqrt(n)
        return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                         [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                         [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    def mat2quat(matrix):
        m=np.asarray(matrix,dtype=np.float64);trace=float(np.trace(m))
        if trace>0:
            s=2*math.sqrt(trace+1);q=np.array([s/4,(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s])
        else:
            i=int(np.argmax(np.diag(m)))
            if i==0:
                s=2*math.sqrt(max(0,1+m[0,0]-m[1,1]-m[2,2]));q=np.array([(m[2,1]-m[1,2])/s,s/4,(m[0,1]+m[1,0])/s,(m[0,2]+m[2,0])/s])
            elif i==1:
                s=2*math.sqrt(max(0,1+m[1,1]-m[0,0]-m[2,2]));q=np.array([(m[0,2]-m[2,0])/s,(m[0,1]+m[1,0])/s,s/4,(m[1,2]+m[2,1])/s])
            else:
                s=2*math.sqrt(max(0,1+m[2,2]-m[0,0]-m[1,1]));q=np.array([(m[1,0]-m[0,1])/s,(m[0,2]+m[2,0])/s,(m[1,2]+m[2,1])/s,s/4])
        q/=np.linalg.norm(q)
        return -q if q[0]<0 else q

OFFICIAL_LEFT_TARGET=(-.25,-.12,.95,0.,1.,0.,0.)
OFFICIAL_RIGHT_TARGET=(.25,-.12,.95,0.,1.,0.,0.)

def _array(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value,dtype=np.float64)

def _matrix(pose):
    if hasattr(pose,'to_transformation_matrix'):m=_array(pose.to_transformation_matrix()).copy()
    else:
        raw=_array(pose)
        if raw.shape==(4,4):m=raw.copy()
        elif raw.shape==(7,):
            if np.linalg.norm(raw[3:])<1e-12:raise ValueError('Zero quaternion')
            m=np.eye(4);m[:3,3]=raw[:3];m[:3,:3]=quat2mat(raw[3:])
        else:raise ValueError('Expected Pose, 7-vector or 4x4 transform')
    if not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-8):raise ValueError('Invalid transform')
    if not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=2e-5) or np.linalg.det(m[:3,:3])<.9999:raise ValueError('Rotation is not SO(3)')
    return m

def _vector(pose):
    m=_matrix(pose);return [*m[:3,3].tolist(),*mat2quat(m[:3,:3]).tolist()]

def _new_pose(matrix,factory):return factory(matrix[:3,3].copy(),mat2quat(matrix[:3,:3]))

class ReadOnlyActorProxy:
    """Only pose/point getters; official scaled local point transforms copied once."""
    def __init__(self,source_actor,hypothetical_pose,*,functional_ids=(0,),contact_ids=(0,)):
        self._matrix=_matrix(hypothetical_pose);source=source_actor.get_pose();self._factory=type(source)
        reference=_matrix(source);self._points={}
        for kind,ids in (('functional',functional_ids),('contact',contact_ids)):
            getter=getattr(source_actor,'get_'+kind+'_point')
            for index in ids:
                point=getter(index,'matrix')
                if point is not None:self._points[kind,index]=np.linalg.inv(reference)@_matrix(point)
        self._name=source_actor.get_name() if hasattr(source_actor,'get_name') else 'bottle_geometry_proxy'
    def get_pose(self):return _new_pose(self._matrix,self._factory)
    def get_name(self):return self._name
    def _point(self,kind,index,ret):
        world=self._matrix@self._points[kind,index]
        if ret=='matrix':return world.copy()
        if ret=='list':return _vector(world)
        if ret=='pose':return _new_pose(world,self._factory)
        raise ValueError('Point return type must be matrix/list/pose')
    def get_functional_point(self,index,ret='list'):return self._point('functional',index,ret)
    def get_contact_point(self,index,ret='list'):return self._point('contact',index,ret)

class _RobotReadProxy:
    def __init__(self,robot,side,task_pose):self._robot,self._side,self._pose=robot,side,_vector(task_pose)
    def get_left_ee_pose(self):return self._pose.copy() if self._side=='left' else self._robot.get_left_ee_pose()
    def get_right_ee_pose(self):return self._pose.copy() if self._side=='right' else self._robot.get_right_ee_pose()
    def __getattr__(self,name):return getattr(self._robot,name)

class _TaskReadProxy:
    def __init__(self,task,robot_proxy):self._task,self.robot,self.plan_success=task,robot_proxy,True
    def __getattr__(self,name):return getattr(self._task,name)

def build_place_goals(task,grasp_task_pose,*,bottle_at_grasp_pose,side='left',target_pose=None,
                      carry_task_pose=None,retreat_distance_m=.1,functional_point_id=0,
                      place_kwargs=None,place_function=None):
    """Call unchanged Base_Task.get_place_pose on read-only actor/robot/task proxies.
    With no carry_task_pose, official retreat is -distance*R_task[:,0]. Explicit
    pose may instead describe a measured arm retreat while assuming zero slip.
    place_function injection permits isolated official-source CPU tests.
    """
    if side not in ('left','right'):raise ValueError('side must be left/right')
    if not math.isfinite(retreat_distance_m) or retreat_distance_m<0:raise ValueError('Invalid retreat distance')
    grasp=_matrix(grasp_task_pose);reference_bottle=_matrix(bottle_at_grasp_pose)
    delta=-retreat_distance_m*grasp[:3,0]
    if carry_task_pose is None:carry=grasp.copy();carry[:3,3]+=delta
    else:carry=_matrix(carry_task_pose)
    robot=task.robot
    palm_grasp=_matrix(robot._trans_from_gripper_to_endlink(_vector(grasp),arm_tag=side))
    palm_carry=_matrix(robot._trans_from_gripper_to_endlink(_vector(carry),arm_tag=side))
    relative=np.linalg.inv(palm_grasp)@reference_bottle;bottle_carry=palm_carry@relative
    actor=ReadOnlyActorProxy(task.bottle,bottle_carry,functional_ids=(functional_point_id,))
    read_task=_TaskReadProxy(task,_RobotReadProxy(robot,side,carry))
    target=_vector(target_pose if target_pose is not None else (OFFICIAL_LEFT_TARGET if side=='left' else OFFICIAL_RIGHT_TARGET))
    kwargs=dict(place_kwargs or {})
    if any(k in kwargs for k in ('actor','arm_tag','target_pose','functional_point_id','pre_dis')):raise ValueError('Reserved place_kwargs key')
    if place_function is None:
        from envs._base_task import Base_Task
        place_function=Base_Task.get_place_pose
    local_fp=np.linalg.inv(bottle_carry)@_matrix(actor.get_functional_point(functional_point_id,'matrix'))
    goals={}
    for label,distance in (('place_pre0',0.),('place_dis02',.02)):
        goal=_matrix(place_function(read_task,actor,side,target,functional_point_id=functional_point_id,pre_dis=distance,**kwargs))
        palm=_matrix(robot._trans_from_gripper_to_endlink(_vector(goal),arm_tag=side))
        bottle_goal=palm@relative;fp=bottle_goal@local_fp
        success=(fp[0,3]<-.15 if side=='left' else fp[0,3]>.15) and fp[2,3]>.9
        goals[label]={'pre_dis_m':distance,'task_goal_world':_vector(goal),'palm_goal_world':_vector(palm),
                      'palm_goal_matrix_world':palm.tolist(),'hypothetical_bottle_goal_matrix_world':bottle_goal.tolist(),
                      'hypothetical_fp0_goal_world':_vector(fp),
                      'fp0_target_position_error_m':float(np.linalg.norm(fp[:3,3]-np.asarray(target[:3]))),
                      'official_position_success_if_rigid_hold':bool(success)}
    actual=_matrix(getattr(robot,'get_'+side+'_ee_pose')())
    return {'side':side,'target_fp0_pose_world':target,'grasp_task_pose_world':_vector(grasp),
            'grasp_palm_pose_world':_vector(palm_grasp),'bottle_at_grasp_matrix_world':reference_bottle.tolist(),
            'palm_to_bottle_rigid_hypothesis':relative.tolist(),'official_retreat_delta_world_m':delta.tolist(),
            'hypothetical_retreat_task_pose_world':_vector(carry),'hypothetical_retreat_palm_pose_world':_vector(palm_carry),
            'hypothetical_retreat_bottle_matrix_world':bottle_carry.tolist(),'actual_current_task_pose_world':_vector(actual),
            'actual_vs_hypothetical_carry_position_m':float(np.linalg.norm(actual[:3,3]-carry[:3,3])),
            'actual_physical_bottle_matrix_world':_matrix(task.bottle.get_pose()).tolist(),
            'functional_point_id':functional_point_id,'goals':goals,
            'scope':'Unchanged official FP/place formula, rigid hold geometry only, no attachment or execution.'}

def _limits(robot,side):
    art=getattr(robot,side+'_entity');names=[j.name for j in art.active_joints]
    limits=np.concatenate([_array(j.limits).reshape(-1,2) for j in art.active_joints])
    arms=list(getattr(robot,side+'_arm_joints_name'));ids=[names.index(n) for n in arms]
    return art,names,limits,arms,ids

def _snapshot(task):
    art=task.robot.left_entity;drives={}
    for joint in art.active_joints:
        velocity=None
        if hasattr(joint,'get_drive_velocity_target'):velocity=_array(joint.get_drive_velocity_target()).copy()
        elif hasattr(joint,'drive_velocity_target'):velocity=_array(joint.drive_velocity_target).copy()
        drives[joint.name]=(_array(joint.get_drive_target()).copy(),velocity)
    return {'qpos':_array(art.get_qpos()).copy(),'qvel':_array(art.get_qvel()).copy(),
            'bottle':_matrix(task.bottle.get_pose()),'links':{l.name:_matrix(l.entity.pose) for l in art.links},
            'drives':drives,'plan_success':task.plan_success}

def _assert_snapshot(task,before):
    after=_snapshot(task)
    for key in ('qpos','qvel','bottle'):
        if not np.array_equal(before[key],after[key]):raise RuntimeError('Preflight changed physical '+key)
    if before['plan_success']!=after['plan_success']:raise RuntimeError('Preflight changed actual task state')
    for name,old in before['links'].items():
        if not np.array_equal(old,after['links'][name]):raise RuntimeError('Preflight changed link '+name)
    for name,pair in before['drives'].items():
        for old,value in zip(pair,after['drives'][name]):
            if old is None and value is None:continue
            if old is None or value is None or not np.array_equal(old,value):raise RuntimeError('Preflight changed drive '+name)

def _path_summary(result,arms,limits,include_paths=False):
    summary={'status':str(result.get('status','unknown')),'returned_keys':list(result)}
    for key in ('status_detail','message','error','planning_time','solve_time'):
        if key in result:summary[key]=str(result[key])
    if 'position' in result:
        q=_array(result['position'])
        if q.ndim==3 and q.shape[0]==1:q=q[0]
        summary['position_shape']=list(q.shape)
        if q.ndim==2 and q.shape[1]==len(arms) and len(q):
            summary['command_finite']=bool(np.isfinite(q).all())
            if np.isfinite(q).all():
                excess=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0.)
                row,j=np.unravel_index(int(excess.argmax()),excess.shape)
                summary.update(goal_arm_joint_rad=q[-1].tolist(),max_command_limit_excess_rad=float(excess[row,j]),
                               command_limits_passed=bool(excess.max()<=1e-5),worst_limit_joint=arms[j],
                               worst_limit_path_index=int(row),minimum_upper_limit_margin_rad=float(np.min(limits[:,1]-q)),
                               minimum_lower_limit_margin_rad=float(np.min(q-limits[:,0])))
            else:summary['command_limits_passed']=False
        if include_paths:summary['position']=q.tolist()
    if 'velocity' in result:
        v=_array(result['velocity']);summary['velocity_shape']=list(v.shape)
        summary['max_command_velocity_rad_s']=float(np.max(np.abs(v))) if v.size else 0.
        if include_paths:summary['velocity']=v.tolist()
    if summary['status']!='Success' and 'status_detail' not in summary:
        summary['status_detail']='Official CuroboPlanner adapter returns only Fail, discarding native MotionGen failure reason. Endpoint IK is a separate probe.'
    return summary

def _existing_curobo_ik(robot,side,palm_goal,full_start,names,limits):
    """Only opt-in worker calls invoke its existing local IK solver; never create one."""
    if getattr(robot,'communication_flag',False):raise ValueError('Direct IK requires an existing local planner')
    planner=getattr(robot,side+'_planner');mg=planner.motion_gen;solver=mg.ik_solver
    if 'aloha-agilex' in planner.yml_path:raise ValueError('RealMan helper excludes ALOHA frame special case')
    p,q=planner._trans_from_world_to_base(np.asarray(_vector(planner.robot_origion_pose)),np.asarray(_vector(palm_goal)))
    p=np.asarray(p)+np.asarray(planner.frame_bias)
    from curobo.types.math import Pose as CuroboPose
    goal=CuroboPose.from_list([*p.tolist(),*q.tolist()]);joint_names=list(solver.joint_names)
    ids=[names.index(n) for n in joint_names];current=mg.tensor_args.to_device(full_start[ids]).reshape(1,-1)
    # Keep cached solver batch size/num_seeds unchanged; use its existing single solve path.
    result=solver.solve_single(goal,retract_config=current,seed_config=current.reshape(1,1,-1),return_seeds=1)
    solutions=_array(result.solution).reshape(-1,len(joint_names))
    success=np.asarray(result.success.detach().cpu().numpy(),dtype=bool).reshape(-1)
    excess=np.maximum(np.maximum(limits[ids,0]-solutions,solutions-limits[ids,1]),0.)
    return {'status':'Success' if success.any() else 'IK_Fail','success_mask':success.tolist(),
            'solution_joint_names':joint_names,'solution_joint_rad':solutions.tolist(),
            'position_error_m':_array(result.position_error).tolist(),
            'rotation_error_solver_units':_array(result.rotation_error).tolist(),'solve_time_s':float(result.solve_time),
            'max_command_limit_excess_rad':float(excess.max()) if excess.size else None,
            'command_limits_passed':bool(excess.size and np.isfinite(solutions).all() and excess.max()<=1e-5),
            'goal_palm_in_curobo_base':[*p.tolist(),*q.tolist()],
            'scope':'Configured single-arm collision-aware endpoint IK only; no full-body or held-bottle clearance guarantee.'}

def preflight_place_goals(task,grasp_task_pose,*,bottle_at_grasp_pose,side='left',target_pose=None,
                          carry_task_pose=None,retreat_distance_m=.1,functional_point_id=0,
                          place_kwargs=None,run_planning=False,planner_modes=('robot_path',),
                          start_qpos=None,include_paths=False,place_function=None):
    """Build official goals; optionally call original Robot planner or current CuRobo IK.
    Both targets use the same current/supplied full start, not a sequential path.
    Original class methods bypass task/robot instance wrappers. No path executes.
    Snapshot checks assert qpos/qvel/poses/drives/task-state are unchanged.
    """
    before=_snapshot(task);started=time.monotonic()
    try:
        report=build_place_goals(task,grasp_task_pose,bottle_at_grasp_pose=bottle_at_grasp_pose,side=side,
                                 target_pose=target_pose,carry_task_pose=carry_task_pose,
                                 retreat_distance_m=retreat_distance_m,functional_point_id=functional_point_id,
                                 place_kwargs=place_kwargs,place_function=place_function)
        art,names,limits,arms,ids=_limits(task.robot,side);actual=_array(art.get_qpos()).copy()
        full_start=actual.copy() if start_qpos is None else _array(start_qpos).copy()
        if full_start.shape!=actual.shape or not np.isfinite(full_start).all():raise ValueError('start_qpos must contain all active joints')
        report.update(actual_arm_joint_rad=actual[ids].tolist(),planning_start_arm_joint_rad=full_start[ids].tolist(),
                      arm_limits_rad={n:limits[i].tolist() for n,i in zip(arms,ids)},
                      actual_arm_upper_margins_rad={n:float(limits[i,1]-actual[i]) for n,i in zip(arms,ids)},
                      planning_run=bool(run_planning))
        modes=tuple(planner_modes)
        if any(m not in ('robot_path','curobo_ik') for m in modes):raise ValueError('Unknown planner mode')
        if run_planning:
            from envs.robot.robot import Robot
            original=getattr(Robot,side+'_plan_path')
            for label,goal in report['goals'].items():
                probes={};goal['planning_probes']=probes
                for mode in modes:
                    stamp=time.monotonic()
                    try:
                        if mode=='robot_path':
                            result=original(task.robot,list(map(float,goal['task_goal_world'])),last_qpos=list(map(float,full_start)))
                            probes[mode]=_path_summary(result,arms,limits[ids],include_paths)
                        else:probes[mode]=_existing_curobo_ik(task.robot,side,goal['palm_goal_world'],full_start,names,limits)
                    except Exception as error:probes[mode]={'status':'Exception','exception_type':type(error).__name__,'message':str(error)}
                    probes[mode]['elapsed_s']=time.monotonic()-stamp
                _assert_snapshot(task,before)
        report.update(elapsed_s=time.monotonic()-started,physical_state_unchanged=True)
        return report
    finally:_assert_snapshot(task,before)

def preflight_candidates(task,candidates,*,bottle_at_grasp_pose,**kwargs):
    """Each candidate supplies label, grasp_task_pose and optional geometry_metadata."""
    results=[]
    for candidate in candidates:
        report=preflight_place_goals(task,candidate['grasp_task_pose'],bottle_at_grasp_pose=bottle_at_grasp_pose,**kwargs)
        report.update(candidate_label=candidate.get('label','unnamed'),geometry_metadata=dict(candidate.get('geometry_metadata',{})))
        results.append(report)
    return results

class _CpuPose:
    def __init__(self,p,q):self.p=np.asarray(p,dtype=float);self.q=np.asarray(q,dtype=float)
    def to_transformation_matrix(self):
        m=np.eye(4);m[:3,3]=self.p;m[:3,:3]=quat2mat(self.q);return m


def load_official_cpu_function(repo_path):
    """Compile ONLY current pure transform functions and Base_Task.get_place_pose.
    No project imports, SAPIEN constructor, environment, planner or GPU is loaded.
    Fake Pose implements the same read-only p/q/transformation matrix interface.
    """
    import ast,copy,hashlib
    from pathlib import Path
    repo=Path(repo_path)
    utility=repo/'envs/utils/transforms.py';task_source=repo/'envs/_base_task.py'
    util_tree=ast.parse(utility.read_text(encoding='utf-8'))
    wanted={'_toPose','get_align_matrix','get_product_vector','get_place_pose'}
    nodes=[copy.deepcopy(n) for n in util_tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if len(nodes)!=4:raise ValueError('Missing official pure transform functions')
    task_tree=ast.parse(task_source.read_text(encoding='utf-8'))
    cls=next(n for n in task_tree.body if isinstance(n,ast.ClassDef) and n.name=='Base_Task')
    function=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_place_pose'))
    function.name='official_task_get_place_pose';nodes.append(function)
    def axangle2mat(axis,angle):
        axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
        x,y,z=axis;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
    ns={'np':np,'sapien':SimpleNamespace(Pose=_CpuPose),
        't3d':SimpleNamespace(quaternions=SimpleNamespace(mat2quat=mat2quat,quat2mat=quat2mat),
                              axangles=SimpleNamespace(axangle2mat=axangle2mat))}
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    ast.fix_missing_locations(module);exec(compile(module,'official_pure_place_cpu','exec'),ns)
    ns['transforms']=SimpleNamespace(_toPose=ns['_toPose'])
    return ns['official_task_get_place_pose'],{'task_source_sha256':hashlib.sha256(task_source.read_bytes()).hexdigest(),
                                              'transform_source_sha256':hashlib.sha256(utility.read_bytes()).hexdigest()}


def run_cpu_tests(place_function=None):
    """Read-only proxy/rigid math checks; optionally use isolated official AST."""
    class Actor:
        def __init__(self):
            self.pose=_CpuPose([.6,.2,.1],[1,0,0,0]);self.local=np.eye(4);self.local[:3,3]=[0,.12,0]
        def get_pose(self):return self.pose
        def get_name(self):return '001_bottle'
        def get_functional_point(self,i,ret='list'):
            m=self.pose.to_transformation_matrix()@self.local
            return m if ret=='matrix' else (_new_pose(m,_CpuPose) if ret=='pose' else _vector(m))
        def get_contact_point(self,i,ret='list'):return self.get_functional_point(i,ret)
    class Robot:
        def _trans_from_gripper_to_endlink(self,p,arm_tag=None):
            m=_matrix(p);m[:3,3]+=m[:3,:3]@np.array([.06,0,0]);return _new_pose(m,_CpuPose)
        def get_left_ee_pose(self):return [.3,.4,.5,1,0,0,0]
        def get_right_ee_pose(self):return [-.3,.4,.5,1,0,0,0]
    def mock_formula(task,actor,side,target_pose,functional_point_id=None,pre_dis=0.,**kwargs):
        assert task.plan_success is True
        b=_matrix(actor.get_pose());fp=_matrix(actor.get_functional_point(functional_point_id,'pose'))
        ee=_matrix(task.robot.get_left_ee_pose() if side=='left' else task.robot.get_right_ee_pose())
        target=_matrix(target_pose);rotate=target[:3,:3]@fp[:3,:3].T
        goal=ee.copy();goal[:3,:3]=rotate@ee[:3,:3];goal[:3,3]=target[:3,3]+rotate@(ee[:3,3]-fp[:3,3])
        direction=rotate@(b[:3,3]-ee[:3,3]);direction/=np.linalg.norm(direction);goal[:3,3]-=pre_dis*direction
        return _vector(goal)
    task=SimpleNamespace(robot=Robot(),bottle=Actor(),plan_success=False)
    original=_matrix(task.bottle.get_pose()).copy();history=np.eye(4);history[:3,3]=[-.1,-.1,.774]
    grasp=[.03,-.2,.87,1,0,0,0];callback=place_function or mock_formula
    report=build_place_goals(task,grasp,bottle_at_grasp_pose=history,place_function=callback)
    assert task.plan_success is False and np.array_equal(original,_matrix(task.bottle.get_pose()))
    assert np.allclose(report['official_retreat_delta_world_m'],[-.1,0,0])
    assert np.allclose(np.asarray(report['hypothetical_retreat_bottle_matrix_world'])[:3,3],history[:3,3]+[-.1,0,0])
    p0=report['goals']['place_pre0'];p2=report['goals']['place_dis02']
    assert p0['fp0_target_position_error_m']<2e-6
    assert abs(p2['fp0_target_position_error_m']-.02)<2e-6
    assert abs(np.linalg.norm(np.array(p0['task_goal_world'][:3])-np.array(p2['task_goal_world'][:3]))-.02)<2e-6
    carry=[.2,-.1,1.,0,1,0,0]
    other=build_place_goals(task,grasp,bottle_at_grasp_pose=history,carry_task_pose=carry,place_function=callback)
    pg=_matrix(task.robot._trans_from_gripper_to_endlink(grasp,arm_tag='left'))
    pc=_matrix(task.robot._trans_from_gripper_to_endlink(carry,arm_tag='left'))
    assert np.allclose(other['hypothetical_retreat_bottle_matrix_world'],pc@np.linalg.inv(pg)@history,atol=2e-6)
    rng=np.random.default_rng(7)
    for q in rng.normal(size=(1000,4)):
        assert np.max(np.abs(quat2mat(mat2quat(quat2mat(q)))-quat2mat(q)))<2e-12
    return {'passed':True,'checks':['historical_grip_not_current_slipped_pose','official_scaled_local_fp_proxy',
                                   'official_negative_task_X_retreat','pre0_and_dis02_target_geometry',
                                   'actual_task_and_bottle_unchanged','explicit_rigid_carry_transform',
                                   'quaternion_1000_random_roundtrips'],
            'used_official_place_function':place_function is not None}


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description='Pure CPU read-only geometry tests; never runs planning')
    parser.add_argument('--official-repo')
    args=parser.parse_args();function=None;source={}
    if args.official_repo:function,source=load_official_cpu_function(args.official_repo)
    result=run_cpu_tests(function);result.update(source)
    print(json.dumps(result,indent=2))
