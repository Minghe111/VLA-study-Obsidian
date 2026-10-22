"""Private simulation optical-frame preset; preserves original task/profile.

Changing cameras is not robot motion or physical D435 calibration. The head
local-pose dictionary is used on every MountedD435RGBCamera.update_mount call,
so a new render cannot silently restore the old cropped optical orientation.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DEFAULT_CONFIG=ROOT/'camera_framing_probes/selected_head_framing.json'


def apply_camera_framing(task,config=None):
    """Apply one explicit head-only preset and return its runtime receipt.

    config may be a JSON path or dict. No physics step, joint target, qpos,
    object pose, original YAML, or wrist camera local transform is changed.
    """
    from integrations.realman_rm65b_gen4.native_camera import MountedHeadCamera
    if config is None:config=json.loads(DEFAULT_CONFIG.read_text())
    elif isinstance(config,(str,Path)):config=json.loads(Path(config).read_text())
    else:config=copy.deepcopy(config)
    require_keys={'head_extra_pitch_down_degrees','head_vertical_fov_degrees'}
    if not require_keys.issubset(config):raise ValueError('Explicit head pitch and vertical FOV are required')
    pitch=float(config['head_extra_pitch_down_degrees']);fovy=float(config['head_vertical_fov_degrees'])
    if not np.isfinite([pitch,fovy]).all() or not 0<=pitch<=35 or not 25<=fovy<=65:
        raise ValueError('Unexpected simulation camera framing range')
    cameras=task.cameras
    if set(cameras.local)!={'head_camera','left_camera','right_camera'}:
        raise TypeError('Expected native three-mounted-camera interface')
    if not hasattr(cameras,'_labware_original_head_mount'):
        cameras._labware_original_head_mount=copy.deepcopy(cameras.mounts['head_camera'])
        cameras._labware_original_head_pose=np.asarray(cameras.local['head_camera'].to_transformation_matrix()).copy()
        cameras._labware_original_wrist_poses={n:np.asarray(cameras.local[n].to_transformation_matrix()).copy()
            for n in ('left_camera','right_camera')}
    original=cameras._labware_original_head_pose
    angle=np.deg2rad(pitch)
    # In SAPIEN camera axes (+X forward,+Y left,+Z up), positive rotation
    # about the camera-left axis tips its forward vector toward camera-down.
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    new=original.copy();new[:3,:3]=original[:3,:3]@rotation
    mount=copy.deepcopy(cameras._labware_original_head_mount)
    mount.update(position=new[:3,3].tolist(),forward=new[:3,0].tolist(),left=new[:3,1].tolist(),
        calibration_status='runtime simulation optical preset; not measured physical D435 calibration',
        extra_optical_pitch_down_degrees=pitch)
    cameras.mounts['head_camera']=mount
    cameras.local['head_camera']=MountedHeadCamera._local_pose(mount)
    camera=cameras.camera_by_name['head_camera']
    # Perspective parameters preserve the same output resolution and clipping.
    # SAPIEN exposes camera setters for FOV without rebuilding the scene.
    camera.set_fovy(np.deg2rad(fovy),compute_x=True)
    cameras.update_mount(task.robot)
    for name,expected in cameras._labware_original_wrist_poses.items():
        if not np.array_equal(np.asarray(cameras.local[name].to_transformation_matrix()),expected):
            raise RuntimeError('Wrist local pose changed during head framing')
    receipt={'preset_id':config.get('preset_id','head_down_runtime_preset'),
        'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config':config,'head_local_pose_before':original.tolist(),
        'head_local_pose_after':cameras.local['head_camera'].to_transformation_matrix().tolist(),
        'runtime_head_world_pose':camera.entity.get_pose().to_transformation_matrix().tolist(),
        'runtime_intrinsic_matrix':camera.get_intrinsic_matrix().tolist(),
        'parent_link':mount['link'],'wrist_local_transforms_unchanged':True,
        'persistent_on_update_mount':True,'physical_d435_calibration':False,
        'robot_joint_or_object_pose_modified':False,'physics_steps_taken':0}
    task.camera_framing_receipt=receipt
    return receipt


def project_target(camera,vertices,object_matrix):
    """Project actual object geometry into the actual SAPIEN pinhole camera."""
    matrix=np.asarray(object_matrix,dtype=float)
    world=np.asarray(vertices,dtype=float)@matrix[:3,:3].T+matrix[:3,3]
    pose=np.asarray(camera.entity.get_pose().to_transformation_matrix(),dtype=float)
    local=(world-pose[:3,3])@pose[:3,:3]
    depth=local[:,0]
    if np.any(depth<=float(camera.get_near())):
        return {'in_front':False,'fully_inside':False,'reason':'geometry behind/near camera'}
    k=np.asarray(camera.get_intrinsic_matrix())
    pixels=np.column_stack([k[0,2]-k[0,0]*local[:,1]/depth,k[1,2]-k[1,1]*local[:,2]/depth])
    bounds=np.stack([pixels.min(0),pixels.max(0)])
    width,height=int(camera.get_width()),int(camera.get_height())
    margins=np.array([bounds[0,0],bounds[0,1],width-1-bounds[1,0],height-1-bounds[1,1]])
    return {'in_front':True,'bbox_xyxy_pixels':[float(x) for x in [*bounds[0],*bounds[1]]],
        'margins_left_top_right_bottom_pixels':margins.tolist(),'minimum_margin_pixels':float(margins.min()),
        'fully_inside':bool(np.all(margins>=0)),'image_size':[width,height],
        'depth_range_m':[float(depth.min()),float(depth.max())]}
