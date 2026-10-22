"""One allowed binding contrast, same 18 seeds and unchanged hand gesture.

Preserve reference palm yaw; update cap center and minimally align its Z axis.
This relies on explicitly hypothetical axial cap symmetry, not a new grasp.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent;PIPE=ROOT.parent;TASK=PIPE/'robot_cap_task'
sys.path.insert(0,str(TASK))
from cap_grasp_probe import Geometry,matrix,sha

ORIGINAL=ROOT/'terminal_cap_ik_v3_finite18/ik_diagnosis.json'
REBIND=ROOT/'left115_static_rebind400/rebind.json'
PLAN=ROOT/'left115_static_shortcut_v1/candidate_plan.json'
OUT=ROOT/'terminal_cap_ik_v3_yaw_minimal_binding'

def main():
    OUT.mkdir(exist_ok=False);t0=time.monotonic()
    original=json.loads(ORIGINAL.read_text());snapshot=Path(original['snapshot_path']);actual=json.loads(snapshot.read_text())
    refsnapshot=Path(json.loads(PLAN.read_text())['snapshot']);reference=json.loads(refsnapshot.read_text())
    refcap=np.asarray(reference['cap_matrix']);newcap=np.asarray(actual['cap_matrix'])
    refpalm=np.asarray(json.loads(REBIND.read_text())['palm_target_world_matrix'])
    z0=refcap[:3,2];z1=newcap[:3,2];z0/=np.linalg.norm(z0);z1/=np.linalg.norm(z1)
    axis=np.cross(z0,z1);sin=float(np.linalg.norm(axis));cos=float(np.clip(z0@z1,-1.,1.))
    if sin<1e-10:
        assert cos>0.;align=np.eye(3)
    else:align=Rotation.from_rotvec(axis/sin*np.arctan2(sin,cos)).as_matrix()
    target=np.eye(4);target[:3,:3]=align@refpalm[:3,:3]
    target[:3,3]=newcap[:3,3]+align@(refpalm[:3,3]-refcap[:3,3])
    g=Geometry(OUT,snapshot=snapshot,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    base=g.q0.copy();arm=np.array([g.index[f'l_joint{i}']for i in range(1,7)])
    mask=np.ones(len(base),dtype=bool);mask[arm]=False;frozen=np.flatnonzero(mask)
    limits=np.asarray(g.art.get_qlimits());link=list(g.pk.get_link_names()).index('l_inspire_base_link')
    inverse=np.linalg.inv(matrix(g.art.root_pose));rows=[]
    for oldrow in original['raw_results']:
        seed=base.copy();seed[arm]=oldrow['seed_left_arm6']
        q,ok,err=g.pk.compute_IK_CLIK(link,g.Pose(inverse@target),seed,mask.tolist(),eps=1e-6,max_iter=1500,dt=.1,damp=1e-6)
        q=np.asarray(q);err=np.asarray(err);g.pk.compute_forward_kinematics(q);fk=g.world_matrix('l_inspire_base_link')
        dp=fk[:3,3]-target[:3,3];dr=Rotation.from_matrix(target[:3,:3].T@fk[:3,:3]).as_rotvec()
        excess=max(0.,float(np.max(np.maximum(limits[arm,0]-q[arm],q[arm]-limits[arm,1]))))
        frozenchange=float(np.max(np.abs(q[frozen]-base[frozen])))
        accepted=bool(ok and excess<=1e-6 and frozenchange<=1e-10 and np.linalg.norm(dp)<2e-5 and np.linalg.norm(dr)<2e-4)
        row={'seed_name':oldrow['seed_name'],'seed_left_arm6':seed[arm].tolist(),
            'solver_success':bool(ok),'raw_error':err.tolist(),'raw_error_norm':float(np.linalg.norm(err)),
            'left_arm6':q[arm].tolist(),'returned_q36':q.tolist(),'jointlimit_excess_rad':excess,
            'jointlimit_min_margin_left_rad':float(np.min(np.c_[q[arm]-limits[arm,0],limits[arm,1]-q[arm]])),
            'frozen_joint_max_change_rad':frozenchange,'world_fk_matrix':fk.tolist(),
            'world_position_error_vector_m':dp.tolist(),'world_position_error_norm_m':float(np.linalg.norm(dp)),
            'world_rotation_error_vector_rad':dr.tolist(),'world_rotation_error_norm_rad':float(np.linalg.norm(dr)),
            'accepted_ik_endpoint_only':accepted}
        if accepted:
            row['endpoint_geometry']=g.screen(q)
            row['cap_contact_geometry']=g.cap_contact_depths()
        rows.append(row)
    accepted=[r for r in rows if r['accepted_ik_endpoint_only']]
    clear=[r for r in accepted if r['endpoint_geometry']['collision_free_except_intended_cap_contact']]
    selected=min(clear or accepted,key=lambda r:np.linalg.norm(np.asarray(r['left_arm6'])-base[arm]))if accepted else None
    yaw0=float(np.arctan2(refcap[1,0],refcap[0,0]));yaw1=float(np.arctan2(newcap[1,0],newcap[0,0]))
    yawdelta=float(np.arctan2(np.sin(yaw1-yaw0),np.cos(yaw1-yaw0)))
    fullrot=float(np.linalg.norm(Rotation.from_matrix(refcap[:3,:3].T@newcap[:3,:3]).as_rotvec()))
    result={'scope':'One binding contrast on the same cap and hand posture; actual cap center/minimum Z alignment with reference world grasp yaw preserved',
        'source_sha256':sha(__file__),'whole_cap_ik_result_sha256':sha(ORIGINAL),
        'actual_snapshot_path':str(snapshot),'actual_snapshot_sha256':sha(snapshot),'reference_snapshot_sha256':sha(refsnapshot),
        'physics_tick':actual['physics_tick'],'cap_center_displacement_world_m':(newcap[:3,3]-refcap[:3,3]).tolist(),
        'cap_center_displacement_norm_m':float(np.linalg.norm(newcap[:3,3]-refcap[:3,3])),
        'cap_axis_reference_world':z0.tolist(),'cap_axis_actual_world':z1.tolist(),
        'cap_axis_change_rad':float(np.arctan2(sin,cos)),'cap_yaw_reference_rad':yaw0,'cap_yaw_actual_rad':yaw1,
        'cap_world_yaw_change_rad':yawdelta,'cap_full_rotation_change_rad':fullrot,
        'minimum_axis_alignment_matrix':align.tolist(),'reference_palm_world_matrix':refpalm.tolist(),
        'target_palm_world_matrix':target.tolist(),
        'binding_equations':'Rmin:zref->zactual; Rtarget=Rmin@Rrefpalm; ptarget=pactualcap+Rmin@(prefpalm-prefcap)',
        'new_pose_family_count':0,'binding_contrast_count':1,'same_retained_18_seeds':True,
        'accepted_count':len(accepted),'endpoint_cpu_geometry_clear_count':len(clear),
        'preferred':selected,'raw_results':rows,'joint_names':g.names,'jointlimits':limits.tolist(),
        'frozen_joint_names':[g.names[i]for i in frozen],
        'physical_steps':0,'physical_pose_or_drive_write_after_initialization':False,
        'elapsed_s':time.monotonic()-t0,
        'limitation':'Axial cap symmetry makes yaw-invariant binding a reasonable hypothesis, not an executed contact proof. Geometry screen remains endpoint-only with declared helper tabletop proxy. Actual scene full path/closing guards required, no first touch/force closure/unscrewing success.',
    }
    (OUT/'binding_comparison.json').write_text(json.dumps(result,indent=2))
    (OUT/'binding_comparison.md').write_text('# 盖轴最小对齐的唯一绑定对照\n\n'
        f'实际中心位移{result["cap_center_displacement_norm_m"]*1000:.3f}mm，盖轴变化{np.degrees(result["cap_axis_change_rad"]):.3f}°，世界yaw变化{np.degrees(yawdelta):.3f}°，完整旋转变化{np.degrees(fullrot):.3f}°。\n\n'
        f'同18初值：整个cap frame跟随0个合法收敛；保持原world grasp yaw、只跟随中心和最小Z对齐得到{len(accepted)}个合法IK解，CPU端点几何通过{len(clear)}。\n\n'
        '没有新增姿态族，只比较一种轴对称盖的绑定规则。preferred left_arm6不是命令：仍需actual scene完整路径与接触检查；没有物理执行或首触成功。\n')
    print(json.dumps({k:result[k]for k in['cap_center_displacement_norm_m','cap_axis_change_rad','cap_world_yaw_change_rad','accepted_count','endpoint_cpu_geometry_clear_count','elapsed_s']},indent=2))
    if selected:print(json.dumps({'preferred_seed':selected['seed_name'],'left6':selected['left_arm6'],'position_error_m':selected['world_position_error_norm_m'],'rotation_error_rad':selected['world_rotation_error_norm_rad'],'endpoint_geometry':selected.get('endpoint_geometry')},indent=2))

if __name__=='__main__':main()
