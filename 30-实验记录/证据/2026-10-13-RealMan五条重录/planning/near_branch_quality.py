"""Pure CPU checks for NEW demonstration candidates; never mutate a trajectory.

Finite revolute IK endpoints may be enumerated before planning. Hardware limits,
full-model FK equivalence and exact selected-endpoint matching remain mandatory.
This module does not use modulo/unwrap on a planned or recorded path.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import itertools,math
from typing import Callable
import numpy as np

TAU=2*math.pi

def _vector(value,length=None):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=1 or (length is not None and len(a)!=length) or not np.isfinite(a).all():
        raise ValueError('Expected finite one-dimensional vector')
    return a.copy()

def enumerate_equivalent_goals(raw_arm_q,current_arm_q,limits,*,periodic_indices,maximum_candidates=256):
    """Enumerate only predeclared finite endpoint branches, closest actual q first.

    periodic_indices must explicitly name verified revolute joints. Prismatic or
    unknown joints are never treated as periodic. A raw IK result outside a finite
    limit may have a legal equivalent endpoint, but must still pass external FK.
    """
    raw=_vector(raw_arm_q,6);current=_vector(current_arm_q,6)
    lim=np.asarray(limits,dtype=np.float64)
    if lim.shape!=(len(raw),2) or not np.isfinite(lim).all() or np.any(lim[:,0]>lim[:,1]):
        raise ValueError('Exact finite ordered hardware limits required')
    periodic=set(periodic_indices)
    if not all(type(i) is int and 0<=i<len(raw) for i in periodic):raise ValueError('Invalid periodic joint indices')
    if np.any(current<lim[:,0]-1e-4) or np.any(current>lim[:,1]+1e-4):raise ValueError('Actual start exceeds measured hardware limit tolerance')
    if type(maximum_candidates) is not int or maximum_candidates<1:raise ValueError('Positive explicit finite branch budget required')
    choices=[]
    for i,q in enumerate(raw):
        if i in periodic:
            low=int(math.ceil((lim[i,0]-q)/TAU-1e-12))
            high=int(math.floor((lim[i,1]-q)/TAU+1e-12))
            pairs=[(q+TAU*k,k) for k in range(low,high+1)]
        else:pairs=[(q,0)] if lim[i,0]-1e-10<=q<=lim[i,1]+1e-10 else []
        if not pairs:return []
        # Values are never snapped/clipped to a limit.
        pairs=[(v,k) for v,k in pairs if lim[i,0]-1e-10<=v<=lim[i,1]+1e-10]
        choices.append(pairs)
    count=math.prod(map(len,choices))
    if count>maximum_candidates:raise ValueError('Finite branch enumeration budget exceeded; do not truncate silently')
    result=[]
    for combination in itertools.product(*choices):
        q=np.array([v for v,k in combination]);delta=q-current
        result.append({'goal_arm_q_rad':q.tolist(),'raw_IK_arm_q_rad':raw.tolist(),
                       'equivalent_integer_turns':[int(k) for v,k in combination],
                       'raw_delta_from_actual_start_rad':delta.tolist(),
                       'distance_from_actual_start_L2_rad':float(np.linalg.norm(delta)),
                       'J6_delta_from_actual_start_rad':float(delta[-1]),
                       'within_finite_hardware_limits':True,'endpoint_only_not_path_transformation':True,
                       'full_model_FK_validated':False})
    result.sort(key=lambda r:(r['distance_from_actual_start_L2_rad'],abs(r['J6_delta_from_actual_start_rad']),
                              sum(abs(v) for v in r['raw_delta_from_actual_start_rad']),tuple(r['goal_arm_q_rad'])))
    return result

def merge_equivalent_goals(raw_IK_arm_solutions,current_arm_q,limits,*,periodic_indices,maximum_candidates=256):
    """Merge independent IK attempts and globally order all legal equivalents."""
    current=_vector(current_arm_q,6);result=[]
    for attempt,raw in enumerate(raw_IK_arm_solutions):
        for row in enumerate_equivalent_goals(raw,current,limits,periodic_indices=periodic_indices,maximum_candidates=maximum_candidates):
            if any(np.max(abs(np.asarray(row['goal_arm_q_rad'])-np.asarray(old['goal_arm_q_rad'])))<1e-8 for old in result):continue
            row['raw_IK_attempt_index']=attempt;result.append(row)
            if len(result)>maximum_candidates:raise ValueError('Global IK branch budget exceeded; do not silently discard branches')
    result.sort(key=lambda r:(r['distance_from_actual_start_L2_rad'],abs(r['J6_delta_from_actual_start_rad']),tuple(r['goal_arm_q_rad'])))
    return result

def validate_equivalent_goals(candidates,*,fk_validator:Callable,required_link_count=93,
                             full_FK_position_tolerance_m=2e-5,full_FK_rotation_tolerance_rad=2e-4,
                             endpoint_position_tolerance_m=5e-5,endpoint_rotation_tolerance_rad=2e-4,
                             max_J6_goal_delta_rad=math.pi):
    """Fail closed on missing FK/source proof; preserve raw nearest-first ordering.

    Caller evaluates each endpoint on a CPU planning model with unchanged frozen
    joints, against the exact task pose and the original raw IK endpoint. All
    hardware links—not only the palm—must be covered by that callback.
    """
    if not callable(fk_validator):raise ValueError('Explicit full-model FK validator required')
    if not math.isfinite(max_J6_goal_delta_rad) or max_J6_goal_delta_rad<0:raise ValueError('Explicit finite nonnegative J6 travel policy required')
    accepted=[];audited=[]
    for number,source in enumerate(candidates):
        row=dict(source);row['candidate_order']=number;reasons=[]
        if abs(row['J6_delta_from_actual_start_rad'])>max_J6_goal_delta_rad+1e-9:
            reasons.append('J6 nearest finite goal still exceeds new pre-contact travel policy')
        proof=fk_validator(np.asarray(row['goal_arm_q_rad'],float))
        row['FK_proof']=proof
        required=('passed','link_count','frozen_joint_max_change_rad','full_link_max_position_error_m',
                  'full_link_max_rotation_error_rad','task_endpoint_position_error_m','task_endpoint_rotation_error_rad')
        if not isinstance(proof,dict) or any(k not in proof for k in required):
            reasons.append('Missing complete full-model FK proof')
        else:
            metrics=np.asarray([proof[k] for k in required[2:]],dtype=float)
            if not np.isfinite(metrics).all() or np.any(metrics<0):reasons.append('Invalid FK proof metrics')
            if proof['passed'] is not True or proof['link_count']!=required_link_count:reasons.append('Full model/link coverage FK proof failed')
            if proof['frozen_joint_max_change_rad']>1e-8:reasons.append('Frozen joints changed')
            if proof['full_link_max_position_error_m']>full_FK_position_tolerance_m or proof['full_link_max_rotation_error_rad']>full_FK_rotation_tolerance_rad:reasons.append('Equivalent goal differs in full-link FK')
            if proof['task_endpoint_position_error_m']>endpoint_position_tolerance_m or proof['task_endpoint_rotation_error_rad']>endpoint_rotation_tolerance_rad:reasons.append('Declared endpoint differs from exact task target')
        row['accepted']=not reasons;row['rejection_reasons']=reasons;row['full_model_FK_validated']=not reasons
        audited.append(row)
        if row['accepted']:accepted.append(row)
    return {'accepted_candidates':accepted,'candidate_audit':audited,'closest_actual_raw_joint_order_preserved':True,
            'planned_path_wrapped_or_clipped':False,'FK_callback_is_external_not_physical_success':True}

@dataclass(frozen=True)
class PrecontactQualityPolicy:
    """New recording quality policy; does not redefine hardware safety limits."""
    max_J6_span_rad:float=math.pi
    max_J6_total_variation_rad:float=math.pi
    max_each_joint_span_rad:float=math.pi
    joint_length_factor:float=3.0
    joint_length_slack_rad:float=0.15
    EE_length_factor:float=3.0
    EE_length_slack_m:float=0.03
    velocity_ratio_tolerance:float=1.02
    joint_step_slack_rad:float=1e-5
    start_endpoint_tolerance_rad:float=1e-6
    selected_goal_endpoint_tolerance_rad:float=1e-6

    def __post_init__(self):
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in asdict(self).values()):
            raise ValueError('Quality policy fields must be finite nonnegative scalars')
        if self.velocity_ratio_tolerance==0:raise ValueError('Velocity step ratio must be positive')

def path_quality(arm_path,actual_start_arm_q,selected_goal_arm_q,*,policy=None,times_s=None,
                 velocity_limits_rad_s=None,hardware_limits=None,EE_positions_world_m=None):
    """Audit raw RRT or exact retimed path, never return modified positions.

    Exact selected goal is required: matching another legal ±2π endpoint is a
    rejection, even if all endpoint Cartesian poses are equivalent.
    Timestamped dense commands additionally obey original speed-based step caps.
    Sparse RRT node step size is reported, not confused with a physical timestep.
    """
    p=np.asarray(arm_path,dtype=float)
    if p.ndim!=2 or p.shape[1]!=6 or len(p)<2 or not np.isfinite(p).all():raise ValueError('Finite >=2-row arm6 path required')
    start=_vector(actual_start_arm_q,p.shape[1]);goal=_vector(selected_goal_arm_q,p.shape[1])
    policy=policy or PrecontactQualityPolicy();reasons=[]
    start_error=float(np.max(abs(p[0]-start)));goal_error=float(np.max(abs(p[-1]-goal)))
    if start_error>policy.start_endpoint_tolerance_rad:reasons.append('Path start differs from actual current q')
    if goal_error>policy.selected_goal_endpoint_tolerance_rad:reasons.append('Path endpoint differs from the specifically selected near raw goal')
    step=np.diff(p,axis=0);variation=np.sum(abs(step),axis=0);span=np.ptp(p,axis=0)
    length=float(np.linalg.norm(step,axis=1).sum());direct=float(np.linalg.norm(goal-start))
    if span[-1]>policy.max_J6_span_rad+1e-9:reasons.append('J6 range exceeds new pre-contact quality policy')
    if variation[-1]>policy.max_J6_total_variation_rad+1e-9:reasons.append('J6 cumulative travel exceeds new pre-contact quality policy')
    if np.max(span)>policy.max_each_joint_span_rad+1e-9:reasons.append('A joint range exceeds new pre-contact quality policy')
    if length>policy.joint_length_factor*direct+policy.joint_length_slack_rad+1e-9:reasons.append('Joint path detour exceeds declared quality budget')
    metrics={'start_max_error_rad':start_error,'selected_goal_max_error_rad':goal_error,
             'each_joint_span_rad':span.tolist(),'each_joint_total_variation_rad':variation.tolist(),
             'each_joint_net_delta_rad':(p[-1]-p[0]).tolist(),'joint_path_L2_length_rad':length,
             'joint_endpoint_L2_distance_rad':direct,'joint_detour_ratio':None if direct<1e-9 else length/direct,
             'max_sparse_or_dense_step_rad':float(abs(step).max()),'trajectory_rows':len(p)}
    if hardware_limits is not None:
        lim=np.asarray(hardware_limits,float)
        if lim.shape!=(p.shape[1],2) or not np.isfinite(lim).all() or np.any(lim[:,0]>lim[:,1]):raise ValueError('Finite ordered hardware limits required')
        excess=np.maximum(np.maximum(lim[:,0]-p,p-lim[:,1]),0.)
        metrics['max_command_hardware_limit_excess_rad']=float(excess.max())
        if excess[1:].max()>1e-5 or excess[0].max()>1e-4:reasons.append('Original commanded/measured hardware limit gate failed')
    if times_s is not None:
        t=_vector(times_s,len(p));dt=np.diff(t)
        if np.any(dt<=0):raise ValueError('Strictly increasing real command times required')
        vmax=_vector(velocity_limits_rad_s,p.shape[1])
        if np.any(vmax<=0):raise ValueError('Original positive velocity limits required')
        cap=dt[:,None]*vmax[None,:]*policy.velocity_ratio_tolerance+policy.joint_step_slack_rad
        metrics['max_step_to_original_velocity_dt_budget_ratio']=float(np.max(abs(step)/cap))
        metrics['max_finite_difference_command_velocity_rad_s']=float(np.max(abs(step)/dt[:,None]))
        if np.any(abs(step)>cap):reasons.append('Dense command step exceeds original velocity-times-dt budget')
    if EE_positions_world_m is not None:
        xyz=np.asarray(EE_positions_world_m,float)
        if xyz.shape!=(len(p),3) or not np.isfinite(xyz).all():raise ValueError('Same-row finite world EE positions required')
        EE_length=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum());EE_direct=float(np.linalg.norm(xyz[-1]-xyz[0]))
        metrics.update(EE_path_length_m=EE_length,EE_endpoint_distance_m=EE_direct,
                       EE_detour_ratio=None if EE_direct<1e-9 else EE_length/EE_direct,
                       EE_max_distance_from_start_m=float(np.linalg.norm(xyz-xyz[0],axis=1).max()))
        if EE_length>policy.EE_length_factor*EE_direct+policy.EE_length_slack_m+1e-9:reasons.append('World EE detour exceeds declared quality budget')
    return {'passed':not reasons,'rejection_reasons':reasons,'quality_policy':asdict(policy),'metrics':metrics,
            'trajectory_modified':False,'hardware_safety_checks_not_replaced':True,
            'scope':'NEW pre-contact demonstration quality audit, independent of original physical success/LP/geometry'}
