"""Task-local SDK24 export; preserves actual targets instead of shifting states.

The unchanged official collect_data.run invokes this adapter's task hooks.
Camera bytes use RoboTwin's original RGB codec. This is an explicit RealMan
export extension, not a claim that the generic 14-D exporter supports SDK24.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np

CAMERAS = {'full_image': 'cam_head', 'left_wrist_image': 'cam_left_wrist',
           'right_wrist_image': 'cam_right_wrist'}
SLICES = {'left_arm_joint_states': slice(0, 6),
          'left_ee_joint_states': slice(6, 12),
          'right_arm_joint_states': slice(12, 18),
          'right_ee_joint_states': slice(18, 24)}


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def save_transition(folder, index, before, command24, after, row):
    """One measured transition, timestamped on the simulator clock."""
    from data.decode_image_bit import encode_image_bit
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{index:07d}.npz'
    if path.exists(): raise FileExistsError(path)
    assert int(after['physics_tick']) - int(before['physics_tick']) == 15
    values = {'state': np.asarray(before['state'], dtype=np.float32),
              'action': np.asarray(command24, dtype=np.float32),
              'next_state': np.asarray(after['state'], dtype=np.float32),
              'tick_start': np.int64(before['physics_tick']),
              'tick_end': np.int64(after['physics_tick']),
              'row_json': np.asarray(json.dumps(row, ensure_ascii=False))}
    for source in CAMERAS:
        image = np.asarray(before[source])
        assert image.dtype == np.uint8 and image.shape == (480, 640, 3), (source, image.shape)
        encoded = encode_image_bit(image)
        if isinstance(encoded, np.ndarray): encoded = encoded.tobytes()
        values[source] = np.frombuffer(encoded, dtype=np.uint8)
    for key in ('state', 'action', 'next_state'):
        assert values[key].shape == (24,) and np.isfinite(values[key]).all(), key
    np.savez_compressed(path, **values)


def export_episode(cache, hdf5_path, instructions, receipt, video_path=None):
    import h5py
    from data.decode_image_bit import decode_image_bit
    files = sorted(Path(cache).glob('*.npz'))
    if not files: raise RuntimeError('No actual captured transitions')
    hdf5_path = Path(hdf5_path); hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    if hdf5_path.exists(): raise FileExistsError(hdf5_path)
    states=[]; actions=[]; next_states=[]; ticks=[]; phases=[]; metrics=[]; drive_records=[]
    encoded={key:[] for key in CAMERAS}
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            states.append(d['state']); actions.append(d['action']); next_states.append(d['next_state'])
            ticks.append([int(d['tick_start']), int(d['tick_end'])])
            row=json.loads(str(d['row_json'])); phases.append(row['phase']); metrics.append(row['metrics'])
            drive_records.append(row.get('actual_drive_record'))
            for key in CAMERAS: encoded[key].append(d[key].tobytes())
    states=np.asarray(states,dtype=np.float32); actions=np.asarray(actions,dtype=np.float32)
    next_states=np.asarray(next_states,dtype=np.float32); ticks=np.asarray(ticks,dtype=np.int64)
    assert states.shape == actions.shape == next_states.shape == (len(files),24)
    assert np.all(ticks[:,1]-ticks[:,0]==15)
    assert np.all(ticks[1:,0]==ticks[:-1,1])
    assert np.array_equal(states[1:],next_states[:-1]), 'Discontinuous measured-state chain'
    assert receipt['success'] is True, 'Do not export a failed task as a demonstration'
    assert receipt.get('error') is None, 'Do not export an interrupted rollout'
    string=h5py.string_dtype('utf-8')
    with h5py.File(hdf5_path,'w') as f:
        f.attrs['schema']='robotwin_xpolicylab_realman_sdk24_v1'
        f.attrs['source']='actual robot-driven simulator rollout; no object replay or teleports'
        f.attrs['state_source']='measured get_state24(True) before each actual 60ms control action'
        f.attrs['action_source']='actual issued absolute SDK24 command; not next measured state'
        f.attrs['order']='left_arm6,left_hand6,right_arm6,right_hand6'
        f.attrs['arm_unit']='radian'
        f.attrs['hand_unit']='model-derived Inspire SDK coordinate 0..1000'
        f.attrs['sdk_mapping_hardware_calibrated']=False
        f.attrs['camera_mount_hardware_calibrated']=False
        f.attrs['success']=True
        f.create_dataset('language_instructions',data=json.dumps(list(instructions),ensure_ascii=False),dtype=string)
        f.create_dataset('additional_info/frequency',data=np.float64(1/.06))
        f.create_dataset('additional_info/control_dt_seconds',data=np.float64(.06))
        f.create_dataset('additional_info/physics_dt_seconds',data=np.float64(.004))
        f.create_dataset('additional_info/ticks',data=ticks)
        f.create_dataset('timestamp',data=ticks[:,0]*.004)
        f.create_dataset('state/vector',data=states)
        f.create_dataset('action/vector',data=actions)
        f.create_dataset('next_state/vector',data=next_states)
        for key,sl in SLICES.items():
            f.create_dataset('state/'+key,data=states[:,sl])
            f.create_dataset('action/'+key,data=actions[:,sl])
        f.create_dataset('diagnostics/phase',data=phases,dtype=string)
        f.create_dataset('diagnostics/physical_metrics',data=[json.dumps(v) for v in metrics],dtype=string)
        f.create_dataset('diagnostics/actual_drive_records',data=[json.dumps(v) for v in drive_records],dtype=string)
        f.create_dataset('diagnostics/task_receipt',data=json.dumps(receipt,ensure_ascii=False),dtype=string)
        for key,target in CAMERAS.items():
            width=max(map(len,encoded[key]))
            ds=f.create_dataset('vision/'+target+'/colors',data=np.asarray(encoded[key],dtype=f'S{width}'))
            ds.attrs['encoding']='RoboTwin encode_image_bit marked standard RGB JPEG'
    # A video is diagnostic; its 50fps timing is exact (3 frames per control tick).
    # Real transition timestamps and 60ms action cadence are authoritative in HDF5.
    if video_path:
        import imageio.v2 as imageio
        video_path=Path(video_path); video_path.parent.mkdir(parents=True,exist_ok=True)
        writer=imageio.get_writer(str(video_path),format='FFMPEG',mode='I',fps=50,
                                  codec='libx264',quality=7,macro_block_size=1)
        try:
            for i in range(len(files)):
                images=[decode_image_bit(encoded[k][i]) for k in CAMERAS]
                frame=np.concatenate(images,axis=1)
                for _ in range(3): writer.append_data(frame)
        finally: writer.close()
    return validate_episode(hdf5_path)


def validate_episode(path):
    """Read every stored camera buffer and check timing, targets and state chain."""
    import h5py
    from data.decode_image_bit import decode_image_bit
    path=Path(path)
    with h5py.File(path,'r') as f:
        state=f['state/vector'][:]; action=f['action/vector'][:]; nxt=f['next_state/vector'][:]
        n=len(state); ticks=f['additional_info/ticks'][:]; times=f['timestamp'][:]
        assert n>1 and state.shape==action.shape==nxt.shape==(n,24)
        assert all(np.isfinite(a).all() for a in (state,action,nxt))
        assert np.array_equal(state[1:],nxt[:-1])
        assert np.all(ticks[:,1]-ticks[:,0]==15) and np.all(ticks[1:,0]==ticks[:-1,1])
        assert np.allclose(np.diff(times),.06,rtol=0,atol=1e-10)
        assert bool(f.attrs['success'])
        instruction=json.loads(f['language_instructions'][()].decode())
        assert instruction and all(isinstance(v,str) and v and '{' not in v for v in instruction)
        images={}
        for key,target in CAMERAS.items():
            ds=f['vision/'+target+'/colors']; assert len(ds)==n
            means=[]; digests=[]; last=None
            for b in ds:
                rgb=decode_image_bit(b)
                assert rgb.dtype==np.uint8 and rgb.shape==(480,640,3)
                means.append(float(rgb.mean())); digests.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                assert float(rgb.std())>1., 'Blank simulated camera'
                last=rgb
            images[target]={'decoded_frames':n,'shape':list(last.shape),'unique_frames':len(set(digests)),
                            'mean_range':[min(means),max(means)]}
        metrics=[json.loads(s) for s in f['diagnostics/physical_metrics'].asstr()[:]]
        assert metrics[-1]['success'] is True
        drives=[json.loads(s) for s in f['diagnostics/actual_drive_records'].asstr()[:]]
        assert len(drives)==n and all(v is not None for v in drives)
        for i,v in enumerate(drives):
            assert v['executed_physics_steps']==15 and v['endpoint_command_written']
            assert v['physics_step_start']==ticks[i,0] and v['physics_step_end']==ticks[i,1]
            assert np.array_equal(np.asarray(v['actual_final_drive_command24'],dtype=np.float32),action[i])
        result={'passed':True,'hdf5':str(path.resolve()),'frames':n,'duration_seconds':n*.06,
                'state_shape':list(state.shape),'action_shape':list(action.shape),
                'measured_state_chain_exact':True,'control_dt_seconds':.06,'physics_ticks_per_action':15,
                'actual_drive_readback_matches_action_exact':True,
                'actual_target_differs_from_next_state_max':float(np.abs(action-nxt).max()),
                'cameras':images,'language_count':len(instruction),'physical_final':metrics[-1],
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    dump(path.with_suffix('.validation.json'),result)
    return result


if __name__=='__main__':
    import argparse
    import sys
    sys.path.insert(0,'/home/admin123/liminghe/vla-benchmark/RoboTwin')
    parser=argparse.ArgumentParser(description='Read back every real SDK24 transition and camera frame from an exported episode.')
    parser.add_argument('hdf5',type=Path)
    cli=parser.parse_args()
    print(json.dumps(validate_episode(cli.hdf5),ensure_ascii=False,indent=2))
