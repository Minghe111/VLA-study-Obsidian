"""Only metadata and a CPU MPLib/FCL/Pinocchio model; no GPU or scene."""
from pathlib import Path
import json,hashlib,importlib.metadata as md,importlib.util as util,xml.etree.ElementTree as ET,os,re
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
ARCHIVE=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')
LEGACY=REPO.parent/'deploy/realman_gen4/adjust_bottle_20261002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
packages={}
for name,pkg in [('sapien','sapien'),('mplib','mplib'),('torch','torch'),('numpy','numpy'),('scipy','scipy'),('h5py','h5py'),('trimesh','trimesh'),('toppra','toppra'),('cv2','opencv-python'),('yaml','PyYAML'),('transforms3d','transforms3d')]:
    spec=util.find_spec(name)
    try:ver=md.version(pkg)
    except md.PackageNotFoundError:ver=None
    packages[name]={'version':ver,'import_spec_origin':None if spec is None else spec.origin}
needs=['try_coupled_side_entry_v29.py','contact_capture_control_v26.py','continuous_servo.py','purecontact_lp_v26.py','planned_carry_wrench_v26.py','full_mesh_path_audit.py','carry_mesh_path_audit.py','full_native_carry_planner_v31.py','support_departure_certificate_v20.py','observed_full_trace_audit_v13.py','preflight_place_goals.py','bottle_table_clearance_certificate.py']
files={str(LEGACY/n):{'exists':(LEGACY/n).is_file(),'sha256':sha(LEGACY/n) if (LEGACY/n).is_file() else None} for n in needs}
original=ARCHIVE/'shared_frozen_runtime/robot_head_down_fixed.urdf';srdf=REPO/'assets/embodiments/realman-rm65b-inspire-gen4/robot.srdf'
cpu=ROOT/'robot_head_down_cpu_relative_meshes.urdf';mapping={}
def replace(match):
    name=match.group(1);p=Path(name)
    if not p.is_file():raise ValueError('Original vendor mesh missing'+name)
    relative=os.path.relpath(p,ROOT);mapping[name]={'relative':relative,'file_sha256':sha(p)}
    return 'filename="'+relative+'"'
payload=re.sub(r'filename="(/[^\"]+)"',replace,original.read_text());cpu.write_text(payload)
a=ET.fromstring(original.read_text());b=ET.fromstring(payload)
for x in a.findall('.//mesh'):x.attrib.pop('filename',None)
for x in b.findall('.//mesh'):x.attrib.pop('filename',None)
assert ET.tostring(a)==ET.tostring(b),'CPU copy alters more than filename relocation'
import mplib
initial=json.loads((ARCHIVE/'episode_0000017/initial_state.json').read_text());names=initial['actual_physical_contract']['full_qpos_joint_names'];links=[n.attrib['name'] for n in ET.parse(cpu).getroot().findall('link')]
planners={};errors={}
for side,group in [('left','l_inspire_base_link'),('right','r_inspire_base_link')]:
    try:
        planner=mplib.Planner(urdf=str(cpu),srdf=str(srdf),move_group=group,user_joint_names=names,user_link_names=links,use_convex=False)
        ids=planner.move_group_joint_indices
        planners[side]={'model_constructed':True,'move_group':group,'move_group_joint_indices':list(ids),'move_group_joint_names':[names[j] for j in ids],'joint_limits_shape':list(planner.joint_limits.shape),'planner_class':str(type(planner))}
        assert len(ids)==6
    except Exception as ex:errors[side]=repr(ex)
receipt={'scope':'Static dependencies and real CPU MPLib/FCL/Pinocchio construction only; no SAPIEN scene/GPU/robot/episode started','packages':packages,'legacy_primary_dependencies':files,
         'original_SAPIEN_URDF':str(original),'original_SAPIEN_URDF_sha256':sha(original),'cpu_URDF':str(cpu),'cpu_URDF_sha256':sha(cpu),'CPU_URDF_only_mesh_filename_relocation':True,'mesh_path_mapping':mapping,
         'prior_absolute_filename_MPLib_failure':'MPLib appended URDF directory before absolute mesh filenames; strict CPU-only relative-filename relocation, same resolved vendor mesh bytes',
         'srdf':str(srdf),'srdf_sha256':sha(srdf),'cpu_planner_probe':planners,'cpu_planner_errors':errors,'passed_cpu_probe':len(planners)==2 and not errors,'gpu_model_or_simulator_started':False}
(ROOT/'local_dependency_cpu_probe.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'cpu_planners':planners,'errors':errors,'missing_legacy_primary_dependencies':[p for p,v in files.items() if not v['exists']]},indent=2))
if errors:raise SystemExit(1)
