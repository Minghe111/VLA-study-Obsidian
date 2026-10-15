"""Read only a source episode's initial state and static scene metadata.

No future expert command, trajectory, outcome or contact feedback is returned to
the learned policy. Hashing the raw source file is independent provenance work.
"""
from pathlib import Path
import hashlib,json
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ARCHIVE=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')
SELECTED=(0,38,30,31,17,7)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()

def load_initial_record(index):
    if index not in SELECTED:raise ValueError('Only the six authorized source scenes')
    e=next(e for e in json.loads((ROOT/'source_manifest.json').read_text())['episodes'] if e['index']==index)
    folder=ARCHIVE/f'episode_{index:07d}'
    sample=json.loads((folder/'source_scene_sample.json').read_text())
    reference=json.loads((folder/'initial_state.json').read_text())
    hdf=ROOT/'source/data'/f'episode_{index:07d}.hdf5'
    expected=reference['source_hdf_sha256'];actual=sha(hdf)
    if actual!=expected:raise RuntimeError('Original source HDF bytes differ')
    fields=[]
    with h5py.File(hdf,'r') as f:
        def initial(key):
            fields.append({'dataset':key,'row':0})
            return np.asarray(f[key][0],dtype=float)
        q=initial('realman_dexterous_physics/full36_qpos_before_rad')
        state=initial('realman_dexterous/measured_state24')
        command=initial('realman_dexterous_physics/applied_command24')
        extrinsics={c:initial('vision/'+c+'/extrinsics_matrix') for c in ('cam_head','cam_left_wrist','cam_right_wrist')}
        intrinsics={c:initial('vision/'+c+'/intrinsic_matrix') for c in ('cam_head','cam_left_wrist','cam_right_wrist')}
        prompt=f['instructions'][()]
        if isinstance(prompt,bytes):prompt=prompt.decode('utf-8')
        if isinstance(prompt,np.ndarray):
            prompt=prompt.reshape(-1)[0]
            if isinstance(prompt,bytes):prompt=prompt.decode('utf-8')
        if not isinstance(prompt,str):raise ValueError('Expected source static language string')
        # Native exporter occasionally stores the static language as JSON.
        try:
            decoded=json.loads(prompt)
            if isinstance(decoded,list) and decoded:prompt=decoded[0]
            elif isinstance(decoded,dict):prompt=decoded.get('instruction',decoded.get('instructions',prompt))
        except json.JSONDecodeError:pass
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Nonempty static language needed')
    if q.shape!=(36,) or state.shape!=(24,) or command.shape!=(24,):raise ValueError('Exact source state shapes differ')
    spec={'model_id':int(e['model_id']),'arm':e['arm'],'pose_class':'upright' if e['pose_class']=='upright' else 'horizontal',
          'center_xy':sample['actual_bottle_center_world'][:2],'axis_yaw_deg':sample['actual_axis_yaw_deg'],'source_initial_pose':sample['initial_pose']}
    return {'episode':e,'spec':spec,'initial36':q,'initial24':state,'initial_command24':command,
            'camera_extrinsics':extrinsics,'camera_intrinsics':intrinsics,'prompt':prompt,
            'original_robot_root':np.asarray(reference['actual_physical_contract']['robot_root_matrix']),
            'source_hdf':str(hdf),'source_hdf_sha256':actual,'source_initial_scene':str(folder/'source_scene_sample.json'),
            'source_initial_scene_sha256':sha(folder/'source_scene_sample.json'),
            'read_boundaries':{'numeric_HDF_reads':fields,'instructions_static_string_read':True,
                               'future_commands_or_states_read':False,'source_outcomes_used_to_control_policy':False}}

def restore_initial_only(task,record):
    import sapien
    from transforms3d.quaternions import mat2quat
    from sapien.physx import PhysxRigidDynamicComponent
    p=np.asarray(record['spec']['source_initial_pose'],float)
    task.bottle.actor.set_pose(sapien.Pose(p[:3,3],mat2quat(p[:3,:3])))
    body=task.bottle.actor.find_component_by_type(PhysxRigidDynamicComponent)
    body.set_linear_velocity(np.zeros(3));body.set_angular_velocity(np.zeros(3));body.put_to_sleep()
    entity=task.robot.left_entity
    root_error=float(np.max(abs(entity.get_root_pose().to_transformation_matrix()-record['original_robot_root'])))
    if root_error>1e-6:raise RuntimeError('Same source root pose not reproduced: '+str(root_error))
    entity.set_qpos(record['initial36']);entity.set_qvel(np.zeros(36))
    return {'initial_bottle_sleeping':bool(body.is_sleeping),'bottle_velocity_assumption':'zero initial velocity; native sleep once at restart, never forced during policy motion',
            'robot_initial_qvel_assumption':'zero; original engine qvel/contact cache not available in source HDF',
            'initialization_only_pose_writes':True,'no_object_pose_writes_after_start':True,'robot_root_matrix_max_abs_error':root_error}

def audit_initial(task,record):
    task.get_obs()
    cams={};intrinsics={}
    for key,name in [('cam_head','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]:
        camera=task.cameras.camera_by_name[name]
        pose=camera.entity.get_pose().to_transformation_matrix()
        # Native exporter stores its OpenGL world model matrix. SAPIEN entity
        # pose uses X forward / Y left / Z up, so convert explicit column axes.
        model=np.stack([-pose[:,1],pose[:,2],-pose[:,0],pose[:,3]],axis=1)
        cams[key]=float(np.max(abs(model-record['camera_extrinsics'][key])))
        intrinsics[key]=float(np.max(abs(camera.get_intrinsic_matrix()-record['camera_intrinsics'][key])))
    qerr=float(np.max(abs(task.robot.left_entity.get_qpos()-record['initial36'])))
    serr=float(np.max(abs(np.asarray(task.robot.get_state24(True))-record['initial24'])))
    perr=float(np.max(abs(task.bottle.get_pose().to_transformation_matrix()-np.asarray(record['spec']['source_initial_pose']))))
    if qerr!=0 or serr!=0 or perr>1e-6 or max(cams.values())>1e-6 or max(intrinsics.values())>1e-6:
        raise RuntimeError('Original first-state restoration differs: '+str((qerr,serr,perr,cams,intrinsics)))
    return {'qpos_max_abs_error_rad':qerr,'state24_max_abs_error':serr,'source_initial_pose_matrix_max_abs_error':perr,
            'source_initial_camera_world_model_matrix_error':cams,'source_initial_intrinsic_matrix_error':intrinsics,
            'source_bottle_pose':record['spec']['source_initial_pose'],
            'actual_bottle_pose':task.bottle.get_pose().to_transformation_matrix().tolist()}
