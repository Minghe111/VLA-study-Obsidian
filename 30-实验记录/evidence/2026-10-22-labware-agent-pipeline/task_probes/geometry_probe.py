from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from labware_task import labware_pick_place
from robot_expert import adapted_native
from transforms3d.quaternions import mat2quat
T=labware_pick_place();T.setup_demo(runtime_path=ROOT/'task_probes/geometry_probe_v1',center_xy=[.10,-.11])
native=adapted_native();audit=native.fma.FullMeshPathAuditor(T)
q0=np.asarray(T.robot.left_entity.get_qpos(),dtype=float)
idx=[audit.sim_joints.index(n) for n in T.robot.right_arm_joints_name]
mask=np.ones(len(q0),dtype=bool);mask[idx]=False
palm=list(audit.pk.get_link_names()).index('r_inspire_base_link')
root=np.asarray(T.robot.left_entity.root_pose.to_transformation_matrix())
rows=[]
for hand_x in [-.015,-.025,-.035]:
 pre,grasp=native.retained_grasp_matrices(T.bottle.get_pose().to_transformation_matrix(),T.bottle.get_contact_point(0,'matrix')[:3,3],'right',dict(native.UPRIGHT_SETTING,hand_x=hand_x))
 m=T.robot._trans_from_gripper_to_endlink([*grasp[:3,3],*mat2quat(grasp[:3,:3])],arm_tag='right').to_transformation_matrix()
 q,ok,err=audit.pk.compute_IK_CLIK(palm,audit.Pose(np.linalg.inv(root)@m),q0,mask.tolist(),eps=1e-7,max_iter=2000,dt=.1,damp=1e-8)
 if not ok:continue
 for fold in [0,100,200,300,500,700,900,1000]:
  values=np.asarray(q).copy()
  measured=dict(zip(audit.sim_joints,values))
  mapping=T.robot._hand_maps['right'].from_sdk([fold,fold,1000,1000,1000,0],measured)
  for name,value in mapping.items():values[audit.sim_joints.index(name)]=value
  audit.model.set_qpos(values,full=True)
  collisions=[];nearest={'bottle':1.,'table':1.}
  for category,obstacle in audit.static.items():
   for name,obj in audit.objects.items():
    if not name.startswith('r_inspire_'):continue
    dist,col=audit._distance(obj,obstacle)
    nearest[category]=min(nearest[category],dist)
    if col:collisions.append([name,category])
  rows.append(dict(hand_x=hand_x,fold_sdk=fold,collisions=collisions,nearest=nearest,qarm=values[idx].tolist()))
print(json.dumps(rows,indent=2))
(ROOT/'task_probes/grasp_geometry_probe.json').write_text(json.dumps(rows,indent=2))
T.close()
