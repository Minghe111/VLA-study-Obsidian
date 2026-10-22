"""Finite native masked CLIK, one unchanged latest actual cap-frame target.

Two reference seeds plus sixteen deterministic small perturbations, no
alternative pose family, no collision relaxation and no physical scene steps.
"""
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent;PIPE=ROOT.parent;TASK=PIPE/'robot_cap_task'
sys.path.insert(0,str(TASK))
from cap_grasp_probe import Geometry,matrix,sha

SNAPSHOT=TASK/'left_contact_probes/left115_first_contact_v3_world_transit/actual_terminal_snapshot.json'
REBIND=ROOT/'left115_static_rebind400/rebind.json'
PLAN=ROOT/'left115_static_shortcut_v1/candidate_plan.json'
OUT=ROOT/'terminal_cap_ik_v3_finite18'

def main():
    OUT.mkdir(exist_ok=False);t0=time.monotonic()
    g=Geometry(OUT,snapshot=SNAPSHOT,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    actual=g.q0.copy();snap=json.loads(SNAPSHOT.read_text())
    T_cap_palm=np.asarray(json.loads(REBIND.read_text())['T_cap_from_palm'])
    target=np.asarray(snap['cap_matrix'])@T_cap_palm
    root_inv=np.linalg.inv(matrix(g.art.root_pose))
    mask=np.ones(len(g.names),dtype=bool);arm=np.array([g.index[f'l_joint{i}']for i in range(1,7)]);mask[arm]=False
    frozen=np.flatnonzero(mask);limits=np.asarray(g.art.get_qlimits())
    palm_index=list(g.pk.get_link_names()).index('l_inspire_base_link')
    final=actual.copy();final[arm]=np.asarray(json.loads(PLAN.read_text())['raw_arm6_waypoints'])[-1]
    seeds=[('actual_current_q36',actual.copy()),('original_final_left_arm_only',final.copy())]
    rng=np.random.default_rng(20261022)
    for k in range(16):
        base=actual if k%2==0 else final
        radius=.04 if k<8 else .12
        # Retained seeds always preserve every frozen measured joint. No qpos
        # clipping: reject the sample instead of changing physical state.
        while True:
            q=base.copy();q[arm]+=rng.uniform(-radius,radius,6)
            if np.all(q[arm]>=limits[arm,0])and np.all(q[arm]<=limits[arm,1]):break
        seeds.append((f'small_perturb_{k:02d}_{"actual"if k%2==0 else"original_final"}_radius{radius}',q))
    rows=[]
    for name,seed in seeds:
        before=time.monotonic()
        q,success,error=g.pk.compute_IK_CLIK(palm_index,g.Pose(root_inv@target),seed,mask.tolist(),eps=1e-6,max_iter=1500,dt=.1,damp=1e-6)
        q=np.asarray(q,dtype=float);error=np.asarray(error,dtype=float)
        g.pk.compute_forward_kinematics(q);fk=g.world_matrix('l_inspire_base_link')
        dp=fk[:3,3]-target[:3,3]
        drot=Rotation.from_matrix(target[:3,:3].T@fk[:3,:3]).as_rotvec()
        violation=np.maximum(limits[arm,0]-q[arm],q[arm]-limits[arm,1])
        limit_excess=max(0.,float(np.max(violation)))
        frozen_change=float(np.max(np.abs(q[frozen]-actual[frozen])))
        accepted=bool(success and limit_excess<=1e-6 and frozen_change<=1e-10
                      and np.linalg.norm(dp)<2e-5 and np.linalg.norm(drot)<2e-4)
        row={'seed_name':name,'seed_left_arm6':seed[arm].tolist(),'solver_success':bool(success),
             'raw_error':error.tolist(),'raw_error_norm':float(np.linalg.norm(error)),
             'returned_q36':q.tolist(),'left_arm6':q[arm].tolist(),
             'jointlimit_excess_rad':limit_excess,'frozen_joint_max_change_rad':frozen_change,
             'world_fk_matrix':fk.tolist(),'world_position_error_vector_m':dp.tolist(),
             'world_position_error_norm_m':float(np.linalg.norm(dp)),
             'world_rotation_error_vector_rad':drot.tolist(),'world_rotation_error_norm_rad':float(np.linalg.norm(drot)),
             'accepted_ik_endpoint_only':accepted,'elapsed_s':time.monotonic()-before}
        if accepted:
            # Endpoint screen is supplementary; target is unchanged. Keep raw
            # contact/clearance evidence, not a physical force-closure claim.
            row['endpoint_collision_readback']=g.screen(q)
        rows.append(row)
        (OUT/'partial_results.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps({k:row[k]for k in['seed_name','solver_success','raw_error_norm','jointlimit_excess_rad','world_position_error_norm_m','world_rotation_error_norm_rad','accepted_ik_endpoint_only']}),flush=True)
    accepted=[r for r in rows if r['accepted_ik_endpoint_only']]
    # Prefer a legal clear endpoint nearest the physically measured left arm.
    clear=[r for r in accepted if r['endpoint_collision_readback']['collision_free_except_intended_cap_contact']]
    selected=min(clear or accepted,key=lambda r:np.linalg.norm(np.asarray(r['left_arm6'])-actual[arm]))if accepted else None
    result={'scope':'Finite native masked CLIK of unchanged latest actual cap target, no new poses or planner search',
        'source_sha256':sha(__file__),'snapshot_path':str(SNAPSHOT),'snapshot_sha256':sha(SNAPSHOT),
        'physics_tick':snap['physics_tick'],'rebind_sha256':sha(REBIND),'original_plan_sha256':sha(PLAN),
        'target_definition':'actual_terminal.cap_matrix @ original rebind400.T_cap_from_palm',
        'T_cap_from_palm':T_cap_palm.tolist(),'target_palm_world_matrix':target.tolist(),
        'actual_q36':actual.tolist(),'joint_names':g.names,
        'moving_joint_names':[g.names[i]for i in arm],'frozen_joint_names':[g.names[i]for i in frozen],
        'jointlimits':limits.tolist(),'solver_config':{'eps':1e-6,'max_iter':1500,'dt':.1,'damp':1e-6,'true_mask_frozen':True},
        'seed_count':len(rows),'retained_small_perturbations':16,'accepted_count':len(accepted),
        'accepted_endpoint_geometry_clear_count':len(clear),'preferred':selected,
        'raw_results':rows,'scene_steps':0,'physical_pose_or_drive_writes_after_initialization':False,
        'elapsed_s':time.monotonic()-t0,
        'limitation':'Finite numerical IK probe, not global reachability proof. Endpoint collision screen uses declared CPU helper geometry, not renewed actual-scene full path or first-contact validation. No physical robot execution/closure/force success.',
    }
    (OUT/'ik_diagnosis.json').write_text(json.dumps(result,indent=2))
    (OUT/'ik_diagnosis.md').write_text('# 实际terminal盖位的有限IK初值复核\n\n'
        f'快照physics_tick={snap["physics_tick"]}；相同目标18个初值（实测current、原final、16个±0.04/0.12rad有限小扰动），合法收敛{len(accepted)}，端点CPU几何筛查通过{len(clear)}。\n\n'
        '仅左臂6关节可变，右臂及全部手指保持actual q36；原始solver error、FK位置/角度残差、冻结关节变化和jointlimits逐次保存，未clip。未扩展姿态族、未搜索path、未物理执行。\n\n'
        '若存在preferred，left_arm6和returned_q36仅是IK候选，仍需actualscene规划与碰撞审计；这里不代表末段接近或真实首触成功。\n')
    print(json.dumps({'accepted_count':len(accepted),'clear_count':len(clear),'preferred_left6':selected['left_arm6']if selected else None,'elapsed_s':result['elapsed_s']},indent=2))

if __name__=='__main__':main()
