"""Installed native public auxiliary API probe, no SAPIEN/scene/GPU."""
from pathlib import Path
import hashlib,json
import mplib
urdf="""<robot name="tiny"><link name="base"/><link name="tip"/>
<joint name="j" type="revolute"><parent link="base"/><child link="tip"/>
<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>"""
model=mplib.ArticulatedModel.create_from_urdf_string(urdf,'<robot name="tiny"/>',[],
    link_names=['base','tip'],joint_names=['j'],verbose=False)
before={'end_effectors':list(model.get_move_group_end_effectors()),
        'joint_indices':list(model.get_move_group_joint_indices())}
model.set_move_group('tip')
selected={'end_effectors':list(model.get_move_group_end_effectors()),
          'joint_indices':list(model.get_move_group_joint_indices())}
model.set_move_group(before['end_effectors'])
after={'end_effectors':list(model.get_move_group_end_effectors()),
       'joint_indices':list(model.get_move_group_joint_indices())}
assert before==after
receipt={'passed':True,'scope':'Installed native MPlib auxiliary move-group public restoration; no robot physics',
    'before':before,'temporary_selected':selected,'after':after,'scene_steps':0,
    'SAPIEN_not_imported':True,'GPU_not_initialized':True,
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (Path(__file__).resolve().parent/'native_move_group_restore_receipt_v1.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
