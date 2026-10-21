from pathlib import Path
import hashlib,json,math,datetime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
vault=Path('/home/admin123/liminghe/VLA-WAM-RL')
output=vault/'30-实验记录/附件/2026-10-10-π05-40000推理/头部相机核查';output.mkdir(exist_ok=False)
roots=[Path('/tmp/realman-eval40k-20261010/local_eval'),Path('/tmp/realman-eval40k-more-20261010/local_eval')]
rows=[];all_pitch=[];all_positions=[];first_pose=None
for root in roots:
 for manifest in sorted((root/'episodes').glob('*/actual_rgb/observations.json')):
  episode=manifest.parent.parent;frames=json.loads(manifest.read_text())['frames'];pitches=[];positions=[]
  for frame in frames:
   with np.load(frame['path']) as z:
    pose=z['head_camera_world_GL_pose'];forward=pose[:3,0]
    pitch=float(np.degrees(np.arctan2(-forward[2],np.linalg.norm(forward[:2]))));pitches.append(pitch);positions.append(pose[:3,3]);all_pitch.append(pitch);all_positions.append(pose[:3,3])
    if first_pose is None:first_pose=pose.copy()
  setup=json.loads((episode/'fresh_setup.json').read_text());profile=setup['camera_profile'];fovy=profile['rgb_profile']['simulation_fovy_deg'];h=profile['rgb_profile']['height'];w=profile['rgb_profile']['width']
  table_z=setup['physical_contract'].get('table_top_z',.7400000095367432)
  contact=np.array([*setup['sample_spec']['center_xy'],table_z]);camera=first_pose[:3,3];local=first_pose[:3,:3].T@(contact-camera);fy=h/(2*np.tan(np.radians(fovy/2)));pixel_v=h/2-fy*local[2]/local[0]
  rows.append({'label':episode.name,'head_pose_samples_checked':len(frames),'pitch_down_deg_min':min(pitches),'pitch_down_deg_max':max(pitches),'head_position_world_m':positions[0].tolist(),'position_span_world_m':np.ptp(np.array(positions),axis=0).tolist(),'nominal_camera_fovy_deg':fovy,'initial_bottle_center_XY':setup['sample_spec']['center_xy'],'table_plane_target_pixel_v_nominal':float(pixel_v),'image_height_pixels':h,'table_plane_target_below_image':bool(pixel_v>=h),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'URDF_head_origin_rpy':setup['fixed_head_configuration']['joints'][1]['origin_rpy']})
p=first_pose[:3,3];f=first_pose[:3,0];pitch=all_pitch[0];a=np.radians(pitch);table_z=.7400000095367432;dh=p[2]-table_z
axis_hit=p+f*(-dh/f[2]);lowest_y=float(p[1]+dh/np.tan(a+np.radians(fovy/2)))
chosen=roots[1]/'episodes/ep08_right_sprite_upright';manifest=json.loads((chosen/'actual_rgb/observations.json').read_text());raw=manifest['frames'][0];r=json.loads((chosen/'report.json').read_text())
with np.load(raw['path']) as z:Image.fromarray(z['cam_head']).save(output/'头部相机原始首帧.png')
image_sha=hashlib.sha256((output/'头部相机原始首帧.png').read_bytes()).hexdigest();assert image_sha==r['raw_queries'][0]['input_png_sha256']['cam_head']
info={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Post-hoc actual recorded camera poses and nominal pinhole geometry; no new rollout or training changes','pose_convention':'Saved *_world_GL_pose key holds SAPIEN camera.entity.get_pose(), not OpenGL model matrix: +X forward,+Y left,+Z up; basis verified against MountedHeadCamera._local_pose','recorded_pose_count':len(all_pitch),'episode_count':len(rows),'actual_pitch_down_deg_min':min(all_pitch),'actual_pitch_down_deg_max':max(all_pitch),'head_world_position_m':p.tolist(),'head_world_forward_vector':f.tolist(),'head_world_position_span_m':np.ptp(np.array(all_positions),axis=0).tolist(),'head_is_fixed_in_all_recorded_frames':True,'nominal_vertical_fov_deg':fovy,'nominal_ray_pitch_range_down_deg':[pitch-fovy/2,pitch+fovy/2],'table_world_z_m':table_z,'table_y_bounds_m':[-.35,.35],'optical_axis_table_intersection_world_m':axis_hit.tolist(),'lowest_fov_ray_table_intersection_y_m':lowest_y,'target_table_y_minus_0_10_required_optical_pitch_down_deg':float(np.degrees(np.arctan2(dh,-.10-p[1]))),'rendered_input_first_frame_sha_matches_policy_service':True,'actual_head_first_frame_sha256':image_sha,'first_frame_episode':chosen.name,'episodes':rows,'limitations':['37deg is simulation nominal FOV, not verified per-device real D435 calibration','Projection ignores occlusion and lens distortion; table-plane point is not a scored grasp target','The required optical angle is a geometric direction, not a verified real head joint command','Training image camera extrinsics not reverified in this audit']}
evidence=vault/'30-实验记录/证据/2026-10-10-头部相机俯角与视野核查.json';evidence.write_text(json.dumps(info,indent=2,ensure_ascii=False))
fig,ax=plt.subplots(figsize=(9.2,4.6));cy,cz=p[1],p[2]
ax.plot([-.35,.35],[table_z,table_z],color='#555555',lw=7,label='Table surface')
ax.plot([lowest_y,.35],[table_z+.006,table_z+.006],color='#22965b',lw=4,label='Visible table strip (nominal FOV)')
ax.scatter([cy],[cz],s=120,c='#244e76',marker='s',zorder=4);ax.text(cy-.04,cz+.037,'Recorded head camera',fontsize=11)
ax.plot([cy,axis_hit[1]],[cz,table_z],'--',color='#2867a5',lw=2,label=f'Optical axis: {pitch:.1f} deg down')
ax.scatter([axis_hit[1]],[table_z],color='#2867a5',s=35);ax.text(axis_hit[1]-.065,table_z-.075,'Axis hits past far edge',color='#2867a5',fontsize=10)
ax.plot([cy,lowest_y],[cz,table_z],color='#68a5ca',lw=2,label=f'Lowest image ray: {pitch+fovy/2:.1f} deg down')
ax.plot([cy,-.10],[cz,table_z],':',color='#b83a30',lw=2,label='Line to bottle base: 44.3 deg down')
ax.scatter([-.10],[table_z],color='#b83a30',s=60,zorder=5);ax.text(-.31,table_z-.10,'Bottle base below image edge',fontsize=10,color='#b83a30')
ax.set(xlim=(-.79,.79),ylim=(.59,1.38),xlabel='World Y (m), camera looks to the right',ylabel='World Z (m)',title='Recorded head view: tilted down, but misses nearby table contact area')
ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=9);fig.tight_layout();fig.savefig(output/'头部相机视野侧视图.png',dpi=160);plt.close(fig)
print(json.dumps({k:info[k] for k in ['recorded_pose_count','actual_pitch_down_deg_min','nominal_ray_pitch_range_down_deg','optical_axis_table_intersection_world_m','lowest_fov_ray_table_intersection_y_m','rendered_input_first_frame_sha_matches_policy_service']},ensure_ascii=False))
