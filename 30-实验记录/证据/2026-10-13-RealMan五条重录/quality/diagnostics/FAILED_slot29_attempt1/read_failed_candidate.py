from pathlib import Path
import json,pickle,hashlib,csv,math,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
SOURCE=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v1/new_physical_episodes/slot29_trials/slot_0000029_randomized_attempt_001')
OUT=Path(__file__).resolve().parent
FFMPEG='/home/admin123/liminghe/miniconda3/envs/robotwin/bin/ffmpeg'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,v): (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def normrot(q1,q2):
 q1=np.asarray(q1,dtype=float);q2=np.asarray(q2,dtype=float)
 return float(2*np.arccos(np.clip(abs(np.dot(q1,q2)/(np.linalg.norm(q1)*np.linalg.norm(q2))),-1,1))*180/np.pi)
def stamp(row):
 return {'physics_step':row['step'],'time_s':row['sim_time_s'],'phase':row['phase'],
 'thumb_normal_N':row['grip']['thumb_normal_N'],'fingers_normal_N':row['grip']['fingers_normal_N'],
 'capture_latched':row['capture_latched'],'capture_actual_pair_streak':row['capture_actual_pair_streak'],
 'contacts':row['grip']['contacts'],'capture_filtered_N':row['capture_filtered_N'],
 'actual_hand_sdk':telemetry['measured_state24_after'][row['step']-1,6:12].tolist(),
 'command_hand_sdk':telemetry['applied_command24'][row['step']-1,6:12].tolist()}
protected={str(SOURCE/n):sha(SOURCE/n) for n in ['expert_report.json','physics_trace.json','physics_telemetry.npz','recording_samples.json','sample_spec.json','runner_error.json']}
report=json.loads((SOURCE/'expert_report.json').read_text());assert report['accepted'] is False and report['result']=='failed'
trace=json.loads((SOURCE/'physics_trace.json').read_text());telemetry=np.load(SOURCE/'physics_telemetry.npz',allow_pickle=False)
samples=json.loads((SOURCE/'recording_samples.json').read_text());sample_time=np.asarray([r['sim_time_s'] for r in samples])
assert len(trace)==len(telemetry['physics_step'])==1475
assert np.array_equal(telemetry['physics_step'],np.arange(1,1476))
assert np.allclose(np.diff(sample_time),.060,atol=1e-7,rtol=0)
actual=telemetry['measured_state24_after'];cmd=telemetry['applied_command24'];atime=telemetry['sim_time_after_s']
closure=np.flatnonzero(np.any(cmd[:,6:11]<950,axis=1))[0]
phase=[stamp(trace[0])]; latch=[];lastphase=trace[0]['phase'];lastlatched=[False]*5
for row in trace:
 if row['phase'] != lastphase:phase.append(stamp(row));lastphase=row['phase']
 if row['capture_latched']!=lastlatched:latch.append(stamp(row));lastlatched=row['capture_latched']
contacts=[stamp(row) for row in trace if row['grip']['contacts']]
force=np.array([[r['grip']['thumb_normal_N'],r['grip']['fingers_normal_N']] for r in trace])
closing=np.array([r['phase']['name']=='close' for r in trace]);closeidx=np.flatnonzero(closing)
armerror=(actual[:,:6]-cmd[:,:6])*180/np.pi
setting=report['grasp_setting']; handcols=[report['full_qpos_joint_names'].index(x) for x in ['l_inspire_left_little_1_joint','l_inspire_left_ring_1_joint','l_inspire_left_middle_1_joint','l_inspire_left_index_1_joint','l_inspire_left_thumb_2_joint','l_inspire_left_thumb_1_joint']]
cachefiles=[]; selected={}; cacheproof=[]
selected_times=[4.2,4.98,5.04,5.1,5.28,5.88]
selected_indices={int(np.argmin(abs(sample_time-t))) for t in selected_times}
for i,s in enumerate(samples):
 p=SOURCE/f'.cache/episode0/{i}.pkl';data=p.read_bytes();cachefiles.append((str(p),hashlib.sha256(data).hexdigest()))
 obs=pickle.loads(data)
 assert obs['recording_clock']['physics_step']==s['physics_step'] and abs(obs['recording_clock']['sim_time_s']-s['sim_time_s'])<1e-9
 for name in ['head_camera','left_camera','right_camera']:
  image=np.asarray(obs['observation'][name]['rgb']);assert image.shape==(480,640,3) and image.dtype==np.uint8
 if i in selected_indices:selected[i]=obs
 cacheproof.append({'index':i,'physics_step':s['physics_step'],'time_s':s['sim_time_s'],'path':str(p),'sha256':cachefiles[-1][1]})
target=np.asarray(report['grasp_geometry']['grasp_pose'])
poses=[]
for i,obs in selected.items():
 pose=np.asarray(obs['endpose']['left_endpose'])
 poses.append({'observation':i,'time_s':float(sample_time[i]),'actual_legacy_ee_pose':pose.tolist(),
 'grasp_target_legacy_ee_pose':target.tolist(),'position_error_m':float(np.linalg.norm(pose[:3]-target[:3])),
 'rotation_error_deg':normrot(pose[3:],target[3:]),'hand_sdk':samples[i]['measured_state24'][6:12],
 'not_physical_calibrated_TCP':True})
summary={'schema':'FAILED-new-DR-expert-candidate-readonly-diagnostic-v1','candidate':str(SOURCE),'slot':29,'domain_seed':10100029,
 'accepted':False,'diagnostic_only_not_a_training_demonstration':True,'simulator_or_policy_started':False,'new_HDF_exported':False,
 'original_failure':report['error'],'physical_steps':len(trace),'actual_duration_s':float(atime[-1]),
 'closure_start_step':trace[closeidx[0]]['step'],'closure_start_time_s':trace[closeidx[0]]['sim_time_s'],
 'sdk950_boundary_time_s_not_actual_contact':float(telemetry['sim_time_before_s'][closure]),
 'pre_sdk950_J6_range_deg':float(np.ptp(np.r_[telemetry['measured_state24_before'][0,5],actual[:closure+1,5]])*180/np.pi),
 'pre_sdk950_J6_travel_deg':float(np.abs(np.diff(np.r_[telemetry['measured_state24_before'][0,5],actual[:closure+1,5]])).sum()*180/np.pi),
 'thumb_contact_force_max_N':float(force[:,0].max()),'finger_contact_force_max_N':float(force[:,1].max()),
 'actual_hand_bottle_contact_steps':len(contacts),'actual_contact_events':contacts,'latch_events':latch,'phase_changes':phase,
 'actual_pair_streak_max_steps':max(r['capture_actual_pair_streak'] for r in trace),
 'last_actual_hand_bottle_contact_step':contacts[-1]['physics_step'],'last_actual_hand_bottle_contact_time_s':contacts[-1]['time_s'],
 'duration_without_hand_bottle_contact_before_rejection_s':float(atime[-1]-contacts[-1]['time_s']),
 'controller_final':report['capture_controller_final'],'LP_allocations':report['adaptive_force_allocations'],
 'applied_final_hand_sdk':cmd[-1,6:12].tolist(),'actual_final_hand_sdk':actual[-1,6:12].tolist(),
 'nominal_close_hand_sdk':setting['close_sdk'],'actual_final_primary_hand_q_rad':telemetry['full36_qpos_after_rad'][-1,handcols].tolist(),
 'close_start_arm_servo_max_error_deg':float(np.abs(armerror[closeidx[0]]).max()),
 'final_arm_servo_max_error_deg':float(np.abs(armerror[-1]).max()),
 'final_hand_sdk_tracking_max_error':float(abs(actual[-1,6:12]-cmd[-1,6:12]).max()),
 'selected_RGB_actual_ee_pose_tracking':poses,'maximum_bottle_translation_m':max(r['fp_delta_m'] for r in trace),
 'max_bottle_rotation_deg':max(r['rotation_delta_deg'] for r in trace),'final_table_force_N':trace[-1]['grip']['table_normal_N'],
 'original_cached_RGB_samples':len(samples),'source_files_sha256':protected,'cache_readback':cacheproof,
 'contact_latch_time_is_one_physics_step_after_observed_impulse':True,
 'conclusion_evidence':'Momentary middle/index impulses latch those channels, thumb never contacts, paired feedback and LP never start; after latch targets remain frozen above nominal closure, actual hand tracks these targets and never forms an opposing grasp. Guard rejects before any lift physics step.',
 'remaining_uncertainty':'Images and actual force prove failed opposing grip, but do not independently identify the exact calibrated geometric palm/thumb offset that would fix this scene.'}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
largefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
# Independent contact sheets from saved actual captures, clearly FAILED.
for names,dest,chosen,scale in [(['head_camera','left_camera','right_camera'],'FAILED_slot29_three_camera_contactsheet.jpg',[70,84,88,98],(480,360)),(['left_camera'],'FAILED_slot29_left_wrist_contactsheet.jpg',sorted(selected),(640,480))]:
 columns=3 if len(names)==3 else 2
 rows=len(chosen) if len(names)==3 else math.ceil(len(chosen)/2)
 w,h=scale;sheet=Image.new('RGB',(columns*w,rows*(h+56)+44),'#111111');draw=ImageDraw.Draw(sheet)
 draw.text((10,10),'FAILED slot29 DR seed10100029 - diagnostic only - NOT accepted data',font=font,fill='#ff6060')
 for n,i in enumerate(chosen):
  obs=selected[i]
  tidx=max(int(np.searchsorted(atime,sample_time[i],side='right')-1),0); row=trace[tidx]
  for j,name in enumerate(names):
   x=(j if len(names)==3 else n%2)*w;y=(n if len(names)==3 else n//2)*(h+56)+44
   sheet.paste(Image.fromarray(obs['observation'][name]['rgb']).resize(scale),(x,y+56))
   draw.text((x+8,y+5),f'{name} | obs{i} at {sample_time[i]:.3f}s',font=font,fill='white')
   draw.text((x+8,y+29),f'Thumb {row["grip"]["thumb_normal_N"]:.3f}N | fingers {row["grip"]["fingers_normal_N"]:.3f}N',font=font,fill='#ffaa66')
 sheet.save(OUT/dest,quality=92)
# Full candidate clip: captured native60ms RGB -> 30fps zero-order hold.
count=math.ceil(float(atime[-1])*30);dest=OUT/'FAILED_slot29_actual_three_camera.mp4'
cmdline=[FFMPEG,'-nostdin','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1440x436','-framerate','30','-i','pipe:0','-c:v','libx264','-pix_fmt','yuv420p','-crf','18','-movflags','+faststart',str(dest)]
proc=subprocess.Popen(cmdline,stdin=subprocess.PIPE,stderr=subprocess.PIPE);last=-1;obs=None;maxage=0.
for tick in range(count):
 t=tick/30;i=max(int(np.searchsorted(sample_time,t,side='right')-1),0)
 if i!=last:
  obs=pickle.loads((SOURCE/f'.cache/episode0/{i}.pkl').read_bytes());last=i
 maxage=max(maxage,t-sample_time[i])
 frame=Image.new('RGB',(1440,436),'#111111');d=ImageDraw.Draw(frame)
 tidx=int(np.searchsorted(atime,t,side='right')-1)
 force_label='physical force n/a before step1' if tidx<0 else f'actual step{trace[tidx]["step"]} | thumb {force[tidx,0]:.4f}N | fingers {force[tidx,1]:.4f}N'
 d.text((10,5),f'FAILED expert slot29 | DR seed10100029 | video {t:.3f}s | held RGB obs{i} at {sample_time[i]:.3f}s',font=font,fill='#ff6666')
 d.text((10,30),force_label+' | no accepted demonstration, no policy inference',font=font,fill='white')
 for j,name in enumerate(['head_camera','left_camera','right_camera']):
  frame.paste(Image.fromarray(obs['observation'][name]['rgb']).resize((480,360)),(j*480,76));d.text((j*480+6,53),name,font=font,fill='white')
 proc.stdin.write(frame.tobytes())
proc.stdin.close();error=proc.stderr.read().decode();ret=proc.wait();assert ret==0,error
(OUT/'failed_video_encode.log').write_text(error)
decode=subprocess.run([FFMPEG,'-nostdin','-hide_banner','-loglevel','error','-i',str(dest),'-map','0:v:0','-an','-f','null','-','-nostats','-progress','pipe:1'],capture_output=True,text=True)
assert decode.returncode==0 and not decode.stderr.strip(),decode.stderr
progress=dict(line.split('=',1) for line in decode.stdout.splitlines() if '=' in line);assert progress.get('progress')=='end' and int(progress['frame'])==count
(OUT/'failed_video_decode.log').write_text(decode.stdout+decode.stderr)
import cv2
cap=cv2.VideoCapture(str(dest));fps=cap.get(cv2.CAP_PROP_FPS);total=cap.get(cv2.CAP_PROP_FRAME_COUNT);cap.release();assert abs(fps-30)<1e-6 and int(total)==count
summary['diagnostic_video']={'path':str(dest),'sha256':sha(dest),'frames':count,'fps':fps,'duration_s':count/fps,'actual_failure_clock_s':float(atime[-1]),'duration_quantization_s':count/fps-float(atime[-1]),'maximum_native_RGB_observation_age_s':maxage,'full_ffmpeg_decode_passed':True,'all_frames_are_saved_actual_RGB':True,'synthetic_scene_rendering_or_HDF_generated':False}
summary['input_bytes_unchanged']={path:sha(Path(path))==digest for path,digest in protected.items()};assert all(summary['input_bytes_unchanged'].values())
summary['cache_source_bytes_unchanged']=all(sha(Path(path))==digest for path,digest in cachefiles);assert summary['cache_source_bytes_unchanged']
dump('FAILED_slot29_readonly_diagnosis.json',summary)
with (OUT/'FAILED_slot29_force_sdk_arm_timeline.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['step','time_s','phase','thumbN','fingerN','middle_latched','index_latched','thumb_latched','actual_pair_streak','actual_middle_sdk','actual_index_sdk','actual_thumb_sdk','cmd_middle_sdk','cmd_index_sdk','cmd_thumb_sdk','max_arm_tracking_deg'])
 for j,row in enumerate(trace):writer.writerow([row['step'],row['sim_time_s'],row['phase']['name'],*force[j],*np.array(row['capture_latched'])[[2,3,4]].tolist(),row['capture_actual_pair_streak'],*actual[j,[8,9,10]],*cmd[j,[8,9,10]],float(abs(armerror[j]).max())])
print(json.dumps({k:summary[k] for k in ['accepted','actual_hand_bottle_contact_steps','thumb_contact_force_max_N','last_actual_hand_bottle_contact_time_s','duration_without_hand_bottle_contact_before_rejection_s','final_arm_servo_max_error_deg','final_hand_sdk_tracking_max_error','diagnostic_video']},ensure_ascii=False))
