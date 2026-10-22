"""Independent CPU construction audit of taskagent's 115mm profile.

Read task/spec/mesh/metadata only; writes only the local audit receipt.
No renderer and no scene.step. A pass here is not a grasp or physics pass.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import sapien
import trimesh

PIPE=Path(__file__).resolve().parent.parent
TASK=PIPE/'robot_cap_task'
PROFILE=TASK/'profiles/bottle115'
FROZEN=PIPE.parent/'2026-10-22-labware-integration'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[PROFILE/p for p in ['physics_spec115.json','body115.glb','cap14.glb','body_model_data0.json','cap_model_data0.json','profile_manifest.json']]
paths += [TASK/'labware_cap115_task.py',TASK/'build_profile115.py',PIPE/'robot_cap_assembly.py']
before={str(p):sha(p)for p in paths}
old=json.loads((FROZEN/'physics_spec.json').read_text());new=json.loads((PROFILE/'physics_spec115.json').read_text())
checks={}
checks['exact_dimension_delta']=new['bottle']['height_m']==.115 and new['cap']['closed_center_z_m']==.118
checks['remaining_bottle_values_unchanged']=all(v==new['bottle'][k]for k,v in old['bottle'].items()if k!='height_m')
checks['remaining_cap_values_unchanged']=all(v==new['cap'][k]for k,v in old['cap'].items()if k!='closed_center_z_m')
checks['physics_pills_unchanged']=new['physics']==old['physics'] and new['tablets']==old['tablets']
checks['thread_numeric_unchanged']=all(v==new['thread_guide'][k]for k,v in old['thread_guide'].items()if isinstance(v,(int,float)))
checks['frozen_component_unchanged']=sha(PIPE/'robot_cap_assembly.py')=='0785eb4e2a487f4ec4a7c637011e163ceabcce18bbc990fc17980e86b95708d5'
meshes={k:trimesh.load(PROFILE/p,force='scene').dump(concatenate=True)for k,p in [('body','body115.glb'),('cap','cap14.glb')]}
checks['visual_body_bounds']=bool(np.allclose(meshes['body'].bounds,[[-.025,-.025,0],[.025,.025,.115]],atol=2e-8))
checks['visual_cap_bounds']=bool(np.allclose(meshes['cap'].bounds,[[-.03,-.03,-.012],[.03,.03,.002]],atol=2e-8))
sys.path[:0]=[str(TASK),str(PIPE)]
from labware_cap115_task import Bottle115Assembly
from robot_cap_assembly import rigid
scene=sapien.Scene([sapien.physx.PhysxCpuSystem()]);scene.set_timestep(.004)
assembly=Bottle115Assembly(scene,pose=sapien.Pose([0,0,0]),seed=0,visual=False)
wrappers=assembly.make_actor_wrappers()
module_path=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin/integrations/realman_rm65b_gen4/teacher_legacy/full_mesh_path_audit_bounded.py')
spec=importlib.util.spec_from_file_location('profile115_original_fma',module_path);fma=importlib.util.module_from_spec(spec);spec.loader.exec_module(fma)
bounds={}
for key,entity in [('body',assembly.body),('cap',assembly.cap)]:
    vertices,_=fma._link_vertices(rigid(entity));bounds[key]=np.stack([vertices.min(0),vertices.max(0)]).tolist()
checks['actual_body_collision_height']=bool(abs(bounds['body'][1][2]-.115)<1e-7 and abs(bounds['body'][0][2])<1e-7)
checks['actual_cap_collision_height']=bool(np.allclose(np.array(bounds['cap'])[:,2],[-.012,.002],atol=1e-7))
checks['actual_closed_cap_relative_z']=abs(float((assembly.body.pose.inv()*assembly.cap.pose).p[2])-.118)<1e-7
checks['body_mass_80g']=abs(float(rigid(assembly.body).mass)-.08)<1e-8
checks['cap_mass_12g']=abs(float(rigid(assembly.cap).mass)-.012)<1e-8
checks['12_independent_tablets']=len(assembly.tablets)==12
functional={}
for key,actor in wrappers.items():
    data=json.loads((PROFILE/f'{key}_model_data0.json').read_text())
    functional[key]=[]
    for i,matrix in enumerate(data['functional_matrix']):
        actual=np.asarray(actor.get_functional_point(i,'matrix'))
        expected=np.asarray(actor.get_pose().to_transformation_matrix())@np.asarray(matrix)
        functional[key].append(float(np.max(np.abs(actual-expected))))
checks['actual_Actor_functional_transform']=max(v for rows in functional.values()for v in rows)<1e-6
body_data=json.loads((PROFILE/'body_model_data0.json').read_text())
checks['body_grasp45mm_preserved']=all(abs(float(m[2][3])-.045)<1e-12 for m in body_data['contact_points_pose'])
issues=[]
for key in ('body','cap'):
    data=json.loads((PROFILE/f'{key}_model_data0.json').read_text());pro=data['annotation_provenance'];source=pro['source']
    if source.get('mesh_path')!=str(PROFILE/('body115.glb'if key=='body'else'cap14.glb')):issues.append(f'{key}: source mesh is old geometry')
    if pro.get('hollow_collision_verified')is True:issues.append(f'{key}: inherited old hollow validation claimed as current')
    if 'surface_anchor_candidates'in pro:issues.append(f'{key}: inherited surface triangle references need new mesh provenance')
    if key=='body'and pro.get('pour_rim_surface_check',{}).get('point_m',[0,0,0])[2]==.09:issues.append('body: inherited90mm rim surface evidence')
after={str(p):sha(p)for p in paths}
result={'scope':'CPU actual component construction, zero physics steps, no renderer; not physical grasp','checks':checks,'all_geometry_checks_passed':all(checks.values()),
    'actual_collision_local_bounds_m':bounds,'functional_transform_max_errors':functional,'metadata_provenance_issues':issues,
    'input_sha256':before,'inputs_changed_during_audit':before!=after,'physics_steps':assembly.physics_steps,
    'remaining':'Actual rendered115 visual binding, closed stability, contact grasp/lift, torque/pill containment not established here.'}
out=Path(__file__).with_name('profile115_construction_audit.json');out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
