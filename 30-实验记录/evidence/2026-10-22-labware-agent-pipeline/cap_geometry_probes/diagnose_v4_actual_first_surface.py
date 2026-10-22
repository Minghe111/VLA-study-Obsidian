"""Read actual preclose/terminal geometry, finite counterfactual readback only.

No new grasp search, target extension, physical simulation or state write.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent;PIPE=ROOT.parent;TASK=PIPE/'robot_cap_task'
sys.path.insert(0,str(TASK))
from cap_grasp_probe import Geometry,sha
from left_cap_held_probe import set_hand

RUN=TASK/'left_contact_probes/left115_first_contact_v4_axisymmetric'
PRE=RUN/'actual_preclose_snapshot.json';END=RUN/'actual_terminal_snapshot.json'
BINDING=ROOT/'terminal_cap_ik_v3_yaw_minimal_binding/binding_comparison.json'
REF=ROOT/'left115_static_rebind400/rebind.json'
OUT=ROOT/'v4_actual_first_surface_readback'

def difference(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return {'translation_vector_m':(b[:3,3]-a[:3,3]).tolist(),
        'translation_norm_m':float(np.linalg.norm(b[:3,3]-a[:3,3])),
        'rotation_norm_rad':float(np.linalg.norm(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).as_rotvec()))}

def read_geometry(g,q):
    g.model.set_qpos(q,full=True);groups={}
    for group,names in [('thumb',g.thumb_names),('index',g.index_names)]:
        values=[]
        for name in names:
            gap,hit=g.distance(g.objects[name],g.cap_obj)
            values.append({'link':name,'gap_m':gap,'collision':hit})
        groups[group]={'nearest':min(values,key=lambda v:v['gap_m']),'all_actual_parent_and_sensor_links':values}
    return {'cap_groups':groups,'forbidden_geometry':g.screen(q)}

def main():
    OUT.mkdir(exist_ok=False);pre=json.loads(PRE.read_text());end=json.loads(END.read_text());binding=json.loads(BINDING.read_text())
    g=Geometry(OUT,snapshot=END,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    actual=g.q0.copy();arm=[g.index[f'l_joint{i}']for i in range(1,7)]
    drive=np.asarray(end['actual_drive_target24']);measured=np.asarray(end['state24'])
    cases={}
    cases['actual_terminal']=read_geometry(g,actual)
    nominal_hand=set_hand(g,actual,drive[6:12]);cases['actual_arm_nominal_drive_hand']=read_geometry(g,nominal_hand)
    nominal_arm=actual.copy();nominal_arm[arm]=drive[:6];cases['drive_arm_actual_hand']=read_geometry(g,nominal_arm)
    nominal_all=nominal_hand.copy();nominal_all[arm]=drive[:6];cases['drive_arm_drive_hand']=read_geometry(g,nominal_all)
    gp=Geometry(OUT/'preclose_geometry',snapshot=PRE,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    cases['actual_preclose']=read_geometry(gp,gp.q0)
    # Original static first-touch is re-read only at its original actual cap.
    ref=json.loads(REF.read_text());old_snapshot=Path(json.loads((ROOT/'left115_static_shortcut_v1/candidate_plan.json').read_text())['snapshot'])
    gr=Geometry(OUT/'original_first_touch_geometry',snapshot=old_snapshot,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    cases['original_static_first_touch']=read_geometry(gr,np.asarray(ref['first_contact_q36']))
    # Same close SDK at the already selected yaw-minimal IK target, before its
    # physical execution. This is a readback of an existing target, not search.
    gv3=Geometry(OUT/'planned_v3_geometry',snapshot=Path(binding['actual_snapshot_path']),geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    qplan=np.asarray(binding['preferred']['returned_q36'])
    qplan=set_hand(gv3,qplan,np.asarray(ref['first_contact_sdk6']))
    cases['existing_yaw_minimal_IK_with_original_close_sdk']=read_geometry(gv3,qplan)
    actual_relation_pre=np.linalg.inv(np.asarray(pre['cap_matrix']))@np.asarray(pre['left_palm_matrix'])
    actual_relation_end=np.linalg.inv(np.asarray(end['cap_matrix']))@np.asarray(end['left_palm_matrix'])
    original_relative=np.linalg.inv(gr.cap_matrix)@np.asarray(ref['palm_target_world_matrix'])
    original_body_right=np.linalg.inv(np.asarray(pre['body_matrix']))@np.asarray(pre['right_palm_matrix'])
    terminal_body_right=np.linalg.inv(np.asarray(end['body_matrix']))@np.asarray(end['right_palm_matrix'])
    result={'scope':'Read-only actual q36 geometry and existing target readback after v4 zero-left-force closure; no search or physical step',
        'source_sha256':sha(__file__),'preclose_sha256':sha(PRE),'terminal_sha256':sha(END),'binding_sha256':sha(BINDING),'original_reference_sha256':sha(REF),
        'actual_physics_ticks':[pre['physics_tick'],end['physics_tick']],
        'actual_left_cap_force_pre_N':pre['metrics']['left_cap_force_N'],'actual_left_cap_force_terminal_N':end['metrics']['left_cap_force_N'],
        'cap_preclose_to_terminal':difference(pre['cap_matrix'],end['cap_matrix']),
        'left_palm_preclose_to_terminal':difference(pre['left_palm_matrix'],end['left_palm_matrix']),
        'cap_from_left_palm_preclose_to_terminal':difference(actual_relation_pre,actual_relation_end),
        'body_from_right_palm_preclose_to_terminal':difference(original_body_right,terminal_body_right),
        'original_grasp_relative_to_actual_terminal':difference(original_relative,actual_relation_end),
        'existing_target_to_actual_terminal_palm':difference(binding['target_palm_world_matrix'],end['left_palm_matrix']),
        'actual_arm_minus_drive_rad':(measured[:6]-drive[:6]).tolist(),
        'actual_hand_minus_drive_sdk':(measured[6:12]-drive[6:12]).tolist(),
        'cases':cases,'cap_collision_parts_actual':len(g.cap_obj.shapes),
        'geometry_basis':'Original same source cap 32 skirt convex pieces plus top; actual profile14mm cap. Original robot convex link collision meshes, full actual q36/root FK. Finger sensor surfaces are not required; all parent/sensor links queried.',
        'physical_steps':0,'physical_pose_drive_writes_after_initialization':False,
        'limitation':'MPlib/FCL distances are CPU geometric evidence, not PhysX force. No raw live collision-mesh export is contained in these snapshots; source-equivalent collision construction is used. Root must choose any future finite correction; this file does not extend close budget or command deeper targets.',
    }
    (OUT/'actual_first_surface.json').write_text(json.dumps(result,indent=2))
    (OUT/'actual_first_surface.md').write_text('# v4无首触后的实际几何读回\n\n'
        f'实际left cap冲量对应力前后均0；末态真实q36下最近thumb gap {cases["actual_terminal"]["cap_groups"]["thumb"]["nearest"]["gap_m"]*1000:.4f}mm，index gap {cases["actual_terminal"]["cap_groups"]["index"]["nearest"]["gap_m"]*1000:.4f}mm。\n\n'
        '各实际父指节及sensor均检查，不把sensor表面当必要条件。对比实际关节与已有drive、原first-surface和新yaw-minimal既有目标；没有新姿态族、没有增加close budget或物理执行。原始距离/FK/来源见JSON。\n')
    print(json.dumps({k:result[k]for k in['cap_preclose_to_terminal','cap_from_left_palm_preclose_to_terminal','body_from_right_palm_preclose_to_terminal','existing_target_to_actual_terminal_palm','actual_arm_minus_drive_rad','actual_hand_minus_drive_sdk']},indent=2))
    print(json.dumps({name:{group:case['cap_groups'][group]['nearest']for group in['thumb','index']}for name,case in cases.items()},indent=2))

if __name__=='__main__':main()
