"""Native public API restore on a branched model and non-URDF joint order."""
from pathlib import Path
import hashlib,json
import mplib
urdf="""<robot name="tiny_dual"><link name="base"/><link name="left_tip"/><link name="right_tip"/>
<joint name="left_joint" type="revolute"><parent link="base"/><child link="left_tip"/>
<origin xyz="0 .1 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
<joint name="right_joint" type="revolute"><parent link="base"/><child link="right_tip"/>
<origin xyz="0 -.1 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>"""
model=mplib.ArticulatedModel.create_from_urdf_string(urdf,'<robot name="tiny_dual"/>',[],
    link_names=['base','right_tip','left_tip'],joint_names=['right_joint','left_joint'],verbose=False)
def read():return {'end_effectors':list(model.get_move_group_end_effectors()),
    'joint_indices':list(model.get_move_group_joint_indices()),'joint_names':list(model.get_move_group_joint_names())}
before=read();rows=[]
for target in ('right_tip','left_tip'):
    model.set_move_group(target);selected=read();model.set_move_group(before['end_effectors']);after=read()
    assert before==after
    rows.append({'target':target,'selected':selected,'restored':after})
receipt={'passed':True,'scope':'Installed native auxiliary group API on branched reversed-joint-order fixture, not real robot physics',
    'before':before,'cases':rows,'scene_steps':0,'SAPIEN_not_imported':True,'GPU_not_initialized':True,
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (Path(__file__).resolve().parent/'native_branched_group_restore_receipt_v2.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
