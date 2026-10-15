from pathlib import Path
import json,math,bisect,hashlib,subprocess
import numpy as np,h5py,cv2
ROOT=Path(__file__).resolve().parent
OUT=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/附件/2026-10-11-RealMan专家回放')
OUT.mkdir(parents=True,exist_ok=True)
INDICES=[0,38,30,31,17,7]
KEYS=['cam_head','cam_left_wrist','cam_right_wrist']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encoder(path,w,h):
    assert not path.exists()
    return subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-crf','20','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
def label(img,text,height=38):
    out=np.zeros((img.shape[0]+height,img.shape[1],3),np.uint8);out[height:]=img
    cv2.putText(out,text,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.64,(235,235,235),1,cv2.LINE_AA)
    return out
def verify(p,w,h,n):
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(p)]))['streams'][0]
    assert (meta['width'],meta['height'],meta['avg_frame_rate'],int(meta['nb_read_frames']))==(w,h,'30/1',n)
    cap=cv2.VideoCapture(str(p));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(h,w);count+=1
    cap.release();assert count==n
    return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'frames':n,'decoded_frames':count,'fps':30,'width':w,'height':h,'duration_s':n/30}
class ReplayFrames:
    def __init__(self,i):
        self.folder=ROOT/f'episodes/ep{i:02d}_current60';self.report=json.load(open(self.folder/'report.json'))
        receipt=self.report['own_actual_rgb_recording'];assert sha(receipt['path'])==receipt['sha256']
        self.rows=json.load(open(receipt['path']))['frames'];self.times=[x['sim_time_s'] for x in self.rows];self.last=-1;self.images=None
    def get(self,t):
        k=max(0,bisect.bisect_right(self.times,t+1e-8)-1)
        if k!=self.last:
            row=self.rows[k];assert sha(row['path'])==row['sha256']
            with np.load(row['path']) as z:self.images={key:z[key] for key in KEYS}
            self.last=k
        return self.images
receipts=[]
for i in INDICES:
    frames=ReplayFrames(i);report=frames.report;e=report['source_episode'];T=report['physical_duration_s'];N=math.ceil(T*30)
    triple=OUT/f'episode_{i:07d}_current_executor_three_views.mp4'
    compare=OUT/f'episode_{i:07d}_source_vs_current_executor.mp4'
    proc=encoder(triple,1920,518);pc=encoder(compare,1280,1036)
    active='cam_left_wrist' if e['arm']=='left' else 'cam_right_wrist';last=-1;src=None
    with h5py.File(report['source_hdf']) as h:
        times=h['realman_dexterous/sample_time_s'][:]
        for j in range(N):
            t=j/30;now=frames.get(t);k=max(0,bisect.bisect_right(times,t+1e-8)-1)
            if k!=last:
                src={key:cv2.imdecode(np.frombuffer(bytes(h['vision/'+key+'/colors'][k]),np.uint8),cv2.IMREAD_COLOR)[...,::-1] for key in ['cam_head',active]};last=k
            title=f"Episode {i:02d} | {e['brand']} {e['arm']} {e['pose_class']} | NEW PHYSICS REPLAY | t={t:.2f}s"
            proc.stdin.write(label(np.concatenate([now[key] for key in KEYS],axis=1),title).tobytes())
            original=label(np.concatenate([src['cam_head'],src[active]],axis=1),f'ORIGINAL DATASET RGB | head / {e["arm"]} wrist | t={t:.2f}s')
            current=label(np.concatenate([now['cam_head'],now[active]],axis=1),'NEW REPLAY | original expert labels through CURRENT 60ms executor | no model')
            pc.stdin.write(np.concatenate([original,current],axis=0).tobytes())
    proc.stdin.close();pc.stdin.close();assert proc.wait()==0 and pc.wait()==0
    receipts.append({'index':i,'three_views':verify(triple,1920,518,N),'source_comparison':verify(compare,1280,1036,N)})
    terminal=frames.get(T)
    cv2.imwrite(str(OUT/f'episode_{i:07d}_terminal.png'),np.concatenate([terminal[key] for key in KEYS],axis=1)[...,::-1])
    print('VIDEOS_READY',i,flush=True)
frames=[ReplayFrames(i) for i in INDICES];T=max(f.report['physical_duration_s'] for f in frames);N=math.ceil(T*30)
overview=OUT/'six_expert_replays_overview.mp4';p=encoder(overview,1920,1036)
for j in range(N):
    t=j/30;panels=[]
    for i,f in zip(INDICES,frames):
        e=f.report['source_episode'];active='cam_left_wrist' if e['arm']=='left' else 'cam_right_wrist'
        image=f.get(min(t,f.report['physical_duration_s']))[active]
        text=f"Ep{i:02d} {e['brand']} {e['arm']} {e['pose_class']} | t={min(t,f.report['physical_duration_s']):.2f}s"
        panels.append(label(image,text))
    canvas=np.concatenate([np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)],axis=0)
    p.stdin.write(canvas.tobytes())
p.stdin.close();assert p.wait()==0
receipts.append({'overview':verify(overview,1920,1036,N),'terminal_hold_for_shorter_episodes':True})
(OUT/'video_export.json').write_text(json.dumps({'scope':'Videos of newly simulated expert-label replay; comparison top row is original HDF RGB, bottom is new simulation','fps':30,'sample_period_s':.06,'playback_resampling':'zero-order hold only, no interpolation of physical observations','source_last_pair_image_at_T_minus_60ms':'Comparison source terminal repeats last available paired HDF frame; raw final unpaired source frame not transferred','files':receipts},indent=2)+'\n')
print('ALL_VIDEOS_READY',str(overview),flush=True)
