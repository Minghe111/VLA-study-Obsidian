"""Strict CPU recursive bottle/table clearance certificate for joint polylines.
Refines only intervals whose conservative continuous distance bound is unresolved.
No collision exemptions, gap clipping, SAPIEN writes or GPU/physical execution.
"""
from __future__ import annotations
import hashlib
import math
import time
import numpy as np
import carry_mesh_path_audit as cma


def certify_lipschitz_polyline(full_path,value_function,coefficients,*,clearance_m=0.,vertex_step_m=.001,max_evaluations=50000,max_depth=26):
    p=np.asarray(full_path,dtype=float);coeff=np.asarray(coefficients,dtype=float)
    if p.ndim!=2 or len(p)<1 or not np.isfinite(p).all() or coeff.shape!=(p.shape[1],) or not np.isfinite(coeff).all() or np.any(coeff<0):raise ValueError('Invalid finite full path/coefficients')
    if not math.isfinite(clearance_m) or clearance_m<0 or not 0<vertex_step_m<=.001:raise ValueError('Invalid clearance/vertex step')
    if not isinstance(max_evaluations,int) or max_evaluations<2 or not isinstance(max_depth,int) or max_depth<0:raise ValueError('Invalid refinement budget')
    cache={};best={'distance_m':math.inf};cert={'lower_bound_m':math.inf};accepted=0;maxbound=0.;deepest=0;collisions=[];unresolved=[]
    def evaluate(q,stamp):
        key=q.tobytes()
        if key not in cache:
            if len(cache)>=max_evaluations:raise RuntimeError('Refinement evaluation budget exhausted')
            gap,collision=value_function(q);gap=float(gap)
            if not math.isfinite(gap) or gap<0:raise ValueError('Distance function returned invalid unsigned gap')
            cache[key]=(gap,bool(collision))
        gap,collision=cache[key]
        if gap<best['distance_m']:best.update(**stamp,distance_m=gap)
        return gap,collision
    try:
        if len(p)==1:
            gap,collision=evaluate(p[0],{'path_interval':-1,'fraction':0.})
            return {'passed':not collision and gap>=clearance_m,'evaluations':len(cache),'accepted_intervals':0,
                'sampled_minimum_gap':best,'continuous_lower_bound':{'lower_bound_m':gap},'collisions':[best] if collision else [],'unresolved_intervals':[]}
        for row,(a,b) in enumerate(zip(p[:-1],p[1:])):
            stack=[(a.copy(),b.copy(),0.,1.,0)]
            while stack:
                q0,q1,s0,s1,depth=stack.pop();deepest=max(deepest,depth)
                g0,c0=evaluate(q0,{'path_interval':row-1,'fraction':s0});g1,c1=evaluate(q1,{'path_interval':row-1,'fraction':s1})
                B=float(coeff@np.abs(q1-q0));lower=min(g0,g1)-B/2
                entry={'path_interval':row-1,'fraction_start':s0,'fraction_end':s1,'depth':depth,
                    'endpoint_gaps_m':[g0,g1],'bottle_vertex_displacement_bound_m':B,'lower_bound_m':lower}
                if c0 or c1:
                    collisions.append(entry);return {'passed':False,'evaluations':len(cache),'accepted_intervals':accepted,
                        'sampled_minimum_gap':best,'continuous_lower_bound':cert,'collisions':collisions,'unresolved_intervals':unresolved}
                if B<=vertex_step_m and lower>=clearance_m:
                    accepted+=1;maxbound=max(maxbound,B)
                    if lower<cert['lower_bound_m']:cert=entry
                    continue
                mid=(q0+q1)/2;sm=(s0+s1)/2
                if depth>=max_depth or np.array_equal(mid,q0) or np.array_equal(mid,q1):
                    unresolved.append(entry);return {'passed':False,'evaluations':len(cache),'accepted_intervals':accepted,
                        'sampled_minimum_gap':best,'continuous_lower_bound':cert,'collisions':collisions,'unresolved_intervals':unresolved}
                stack.append((mid,q1,sm,s1,depth+1));stack.append((q0,mid,s0,sm,depth+1))
    except RuntimeError as e:
        unresolved.append({'reason':str(e)})
    return {'passed':not collisions and not unresolved,'evaluations':len(cache),'accepted_intervals':accepted,
        'maximum_certified_bottle_vertex_step_m':maxbound,'deepest_refinement_level':deepest,
        'sampled_minimum_gap':best,'continuous_lower_bound':cert,'collisions':collisions,'unresolved_intervals':unresolved,
        'proof':'Distance is 1-Lipschitz in rigid shape point displacement. Each point on a linear joint segment is at most B/2 from its nearest endpoint; lower=min(g0,g1)-B/2.'}


def certify_bottle_table_path(task,path,side='left',*,palm_to_bottle=None,clearance_m=0.,vertex_step_m=.001,max_evaluations=50000,max_depth=26):
    import preflight_place_goals as ppg
    before=ppg._snapshot(task);started=time.monotonic()
    a=cma.CarryMeshPathAuditor(task,side,palm_to_bottle=palm_to_bottle)
    original=a.static['bottle'];base=a.initial_sim_q.copy();raw=np.asarray(path['position'] if isinstance(path,dict) else path,dtype=float)
    indices=[a.sim_joints.index(n) for n in getattr(task.robot,side+'_arm_joints_name')]
    try:
        if raw.ndim!=2 or not len(raw) or not np.isfinite(raw).all():raise ValueError('Invalid command path')
        if raw.shape[1]==len(indices):full=np.tile(base,(len(raw),1));full[:,indices]=raw;command=indices
        elif raw.shape[1]==len(base):full=raw.copy();command=list(range(len(base)))
        else:raise ValueError('Path width must be arm or full DOF')
        full=np.vstack([base,full]);limits=np.concatenate([cma.fma._array(j.limits).reshape(-1,2) for j in a.art.active_joints])
        limit=cma.fma.joint_limit_audit(full,limits,a.sim_joints,command)
        def value(q):
            a.model.set_qpos(q,full=True)
            pose=cma.rigid_bottle_pose(cma._matrix(a.objects[a.palm_name].pose),a.reference)
            bottle=a.fcl.FCLObject(original.name,a.Pose(pose),original.shapes,original.shape_poses)
            return a._distance(bottle,a.static['table'])
        result=certify_lipschitz_polyline(full,value,a.bottle_coefficients,clearance_m=clearance_m,
            vertex_step_m=vertex_step_m,max_evaluations=max_evaluations,max_depth=max_depth)
        result['passed']=bool(result['passed'] and limit['passed'])
        result.update(joint_limit_audit=limit,metadata=a.metadata,command_path_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            full_path_with_measured_start_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
            clearance_m=clearance_m,path_rows=len(raw),physical_state_unchanged=True,elapsed_s=time.monotonic()-started,
            scope='ONLY strict continuous bottle/table clearance. Full robot/table/self/other-arm checks must separately pass for this identical path/reference; no force/slip/servo guarantee.')
        return result
    finally:
        a.model.set_qpos(a.old_model_q,full=True)
        a.static['bottle']=a.fcl.FCLObject(original.name,a.original_bottle_pose,original.shapes,original.shape_poses)
        ppg._assert_snapshot(task,before)


def run_cpu_tests():
    r=certify_lipschitz_polyline([[0.],[1.]],lambda q:(6e-6+.01*float(q[0]),False),[.1])
    assert r['passed'] and r['continuous_lower_bound']['lower_bound_m']>=0 and r['maximum_certified_bottle_vertex_step_m']<=.001
    bad=certify_lipschitz_polyline([[0.],[1.]],lambda q:(max(.01-.03*math.sin(math.pi*float(q[0])),0),.01-.03*math.sin(math.pi*float(q[0]))<0),[.1])
    assert not bad['passed'] and bad['collisions']
    zero=certify_lipschitz_polyline([[0.],[1.]],lambda q:(0.,False),[1.],max_depth=4)
    assert not zero['passed'] and zero['unresolved_intervals']
    budget=certify_lipschitz_polyline([[0.],[1.]],lambda q:(.000001,False),[1.],max_evaluations=3)
    assert not budget['passed'] and budget['unresolved_intervals']
    return {'passed':True,'checks':['6micrometre_start_clearance_certified_by_local_refinement','midinterval_true_collision_rejected',
        'zero_gap_not_falsely_certified','budget_exhaustion_rejected'],'clear_case_evaluations':r['evaluations'],'GPU':False,'SAPIEN_scene_created':False}


if __name__=='__main__':
    import json
    print(json.dumps(run_cpu_tests(),indent=2))
