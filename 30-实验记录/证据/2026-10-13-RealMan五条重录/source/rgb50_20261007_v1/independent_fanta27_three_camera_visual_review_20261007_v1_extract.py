from pathlib import Path
import json,hashlib,subprocess,sys
import cv2,numpy as np
w=Path('/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/rgb50_20261007_v1')
p=w/'batch_candidates/v7_post_setup_native_wake_v4_fanta_horizontal_attempt220/slot0000027_candidate_attempt220_v7_post_setup_wake8s'
out=w/'independent_fanta27_three_camera_visual_review_20261007_v1'
assert not out.exists(), 'Refuse to overwrite review directory'
out.mkdir()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()
source=[p/n for n in ['recording_report.json','recording_samples.json','dataset_validation.json','expert_report.json','scene_sample.json','sample_spec.json','worker_receipt.json']]+[w/'generation_runner_v2.py']+[p/'video'/f'{c}_realtime.mp4' for c in ('head','left_wrist','right_wrist')]
before={str(f):sha(f) for f in source}
r=json.loads((p/'recording_report.json').read_text());samples=json.loads((p/'recording_samples.json').read_text());times=np.asarray([x['sim_time_s'] for x in samples]); records={}
assert len(samples)==210 and times[0]==0 and np.all(np.diff(times)>0)
for c in ('head','left_wrist','right_wrist'):
    v=p/'video'/f'{c}_realtime.mp4'
    probe=json.loads(subprocess.run(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration','-show_entries','format=duration','-of','json',str(v)],check=True,capture_output=True,text=True).stdout)
    cap=cv2.VideoCapture(str(v));assert cap.isOpened();count=0;first=last=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(480,640,3)
        if count==0:first=frame.copy()
        last=frame;count+=1
    cap.release();assert count==r['realtime_videos'][c]['frames']==377
    frames=[]
    for label,index,img in [('first',0,first),('last',count-1,last)]:
        dest=out/f'{c}_{label}.png';assert not dest.exists();assert cv2.imwrite(str(dest),img)
        tick=index/30.0;obs_idx=max(int(np.searchsorted(times,tick,side='right')-1),0)
        frames.append({'label':label,'encoded_frame_index':index,'video_tick_actual_s':tick,'source_observation_index':obs_idx,'source_physics_step':samples[obs_idx]['physics_step'],'source_observation_actual_s':float(times[obs_idx]),'observation_age_at_tick_s':float(tick-times[obs_idx]),'png':str(dest),'png_sha256':sha(dest),'decoded_bgr_sha256':hashlib.sha256(img.tobytes()).hexdigest(),'pixel_min':int(img.min()),'pixel_max':int(img.max())})
    records[c]={'ffprobe':probe,'sequential_decoded_frame_count':count,'frames':frames}
after={str(f):sha(f) for f in source};assert before==after
receipt={'scope':'Existing candidate three actual-clock RGB videos decoded CPU-only; no new rendering, physical step, model/camera/control changes, or independent formal dataset acceptance','trial':str(p),'cpu_only':True,'sapien_imported':'sapien' in sys.modules,'video_sources_sha256_before':before,'video_sources_sha256_after':after,'inputs_unchanged':True,'camera_frames':records,'actual_physical_duration_s':float(times[-1]),'realtime_zero_order_hold_exporter':'generation_runner_v2.py:171-189','raw_native_episode_30fps_video_not_used':True,'latest_terminal_observation_index':len(samples)-1,'terminal_physics_step':samples[-1]['physics_step'],'video_final_rgb_is_prior_observation_not_terminal':True,'visual_assessment_pending':True}
dest=out/'frame_extraction_receipt_v1.json';dest.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'directory':str(out),'receipt_sha256':sha(dest),'frames':{c:x['frames'] for c,x in records.items()},'sources_unchanged':before==after},ensure_ascii=False))