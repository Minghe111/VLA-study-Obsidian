"""Actual six-case observer grid and full video decode evidence."""
from pathlib import Path
import subprocess,json,hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'rendered_videos';INDICES=(0,38,30,31,17,7)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cv2.setNumThreads(2);dest=DEST;dest.mkdir(exist_ok=True);rows=[];reports=[];sources=[];times=[]
    for i in INDICES:
        p=ROOT/'episodes'/f'ep{i:02d}_model80000';r=json.loads((p/'report.json').read_text());reports.append(r)
        frames=json.loads((p/'observer_rgb/observations.json').read_text())['frames'];sources.append(frames);times.append(np.asarray([f['sim_time_s'] for f in frames]))
        assert len(frames)==401 and abs(r['physical_duration_s']-24)<1e-5
    # Exact final physical snapshots are separate from the encoded video's last
    #30fps frame (23.966s), so terminal geometry can be visually reviewed at24s.
    terminal_tiles=[];terminal_pngs=[]
    for j,i in enumerate(INDICES):
        p=ROOT/'episodes'/f'ep{i:02d}_model80000';policy=json.loads((p/'actual_rgb/observations.json').read_text())['frames'][-1]
        with np.load(policy['path']) as z:ims={k:z[k] for k in ('cam_head','cam_left_wrist','cam_right_wrist')}
        with np.load(sources[j][-1]['path']) as z:ims['external']=z['external']
        terminal=np.zeros((1004,1280,3),np.uint8);title=f'Checkpoint80000 | source{i:07d} | exact terminal physical step6000 /24.000001s'
        cv2.putText(terminal,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA)
        ts=[]
        for key,label in [('external','External observer'),('cam_head','Head camera'),('cam_left_wrist','Left wrist'),('cam_right_wrist','Right wrist')]:
            im=ims[key].copy();cv2.rectangle(im,(0,0),(640,28),(24,24,24),-1);cv2.putText(im,label,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);ts.append(im)
        terminal[44:524]=np.concatenate(ts[:2],axis=1);terminal[524:]=np.concatenate(ts[2:],axis=1)
        q=DEST/f'episode_{i:07d}_exact_terminal_physics24s.png';cv2.imwrite(str(q),cv2.cvtColor(terminal,cv2.COLOR_RGB2BGR));terminal_pngs.append(str(q))
        tile=cv2.resize(ims['external'],(480,360),interpolation=cv2.INTER_AREA);ep=reports[j]['source_episode'];cv2.rectangle(tile,(0,0),(480,28),(24,24,24),-1)
        cv2.putText(tile,f"source{i:07d} | {ep['brand']} {ep['arm']} {ep['pose_class']}",(9,20),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA);terminal_tiles.append(tile)
    contact=np.zeros((764,1440,3),np.uint8);contact[44:404]=np.concatenate(terminal_tiles[:3],axis=1);contact[404:]=np.concatenate(terminal_tiles[3:],axis=1)
    cv2.putText(contact,'Checkpoint80000 | Exact terminal physical snapshots | step6000 /24.000001s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.67,(255,255,255),1,cv2.LINE_AA)
    contact_path=DEST/'six_training_scenes_exact_terminal_physics24s.png';cv2.imwrite(str(contact_path),cv2.cvtColor(contact,cv2.COLOR_RGB2BGR))
    overview=DEST/'six_model80000_original_scenes_overview.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1440x764','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(overview)]
    if overview.exists():raise FileExistsError(overview)
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);latest=[-1]*6;tiles=[None]*6
    try:
        for n in range(720):
            sim=n/30
            for j,i in enumerate(INDICES):
                frame=max(0,int(np.searchsorted(times[j],sim,side='right')-1))
                if frame!=latest[j]:
                    with np.load(sources[j][frame]['path']) as z:im=cv2.resize(z['external'],(480,360),interpolation=cv2.INTER_AREA)
                    ep=reports[j]['source_episode'];cv2.rectangle(im,(0,0),(480,28),(24,24,24),-1)
                    text=f"source{i:07d} | {ep['brand']} {ep['arm']} {ep['pose_class']}"
                    cv2.putText(im,text,(9,20),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA);tiles[j]=im;latest[j]=frame
            canvas=np.zeros((764,1440,3),np.uint8);canvas[44:404]=np.concatenate(tiles[:3],axis=1);canvas[404:764]=np.concatenate(tiles[3:],axis=1)
            cv2.putText(canvas,f'Checkpoint80000 | Six original training scenes | actual physics RGB | sim{sim:05.2f}s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError('overview encoder failed')
    except BaseException:
        if proc.poll() is None:proc.kill();proc.wait()
        raise
    # Decode every encoded frame, inspect file metadata and preserve actual frame
    # hashes plus10s/terminal images. No source demo media is decoded or inserted.
    allvideos=[(i,DEST/f'episode_{i:07d}_model80000_four_views.mp4') for i in INDICES]+[(None,overview)]
    for i,p in allvideos:
        cap=cv2.VideoCapture(str(p));count=0;frame_sha=hashlib.sha256();examples=[];size=None
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if bgr.dtype!=np.uint8 or bgr.ndim!=3 or bgr.shape[2]!=3:raise RuntimeError('Invalid decoded RGB')
            size=[int(bgr.shape[1]),int(bgr.shape[0])];frame_sha.update(bgr.tobytes())
            if count in (300,719):
                label='10s' if count==300 else 'terminal';q=DEST/(p.stem+'_'+label+'.png');cv2.imwrite(str(q),bgr);examples.append(str(q))
            count+=1
        cap.release()
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)],text=True))
        stream=next(s for s in probe['streams'] if s['codec_type']=='video');fps=stream['r_frame_rate'];duration=float(probe['format']['duration'])
        if count!=720 or int(stream['nb_frames'])!=720 or abs(duration-24)>.001 or fps!='30/1':raise RuntimeError('Incomplete/wrong-time encodedvideo')
        row={'source_index':i,'path':str(p),'file_sha256':sha(p),'bytes':p.stat().st_size,'decoded_frames':count,'declared_frames':int(stream['nb_frames']),
             'duration_s':duration,'fps':30,'dimensions':size,'decoded_pixels_sha256':frame_sha.hexdigest(),'inspection_images':examples,
             'visual_source':'Only this new physical rollout actual NPZ RGB; no expert-source RGB/video','sim_time_resampling':'Zero order hold at30fps; no synthetic motion/image interpolation'}
        rows.append(row);print('FULL_DECODE_VERIFIED',i,count,str(p),flush=True)
    (ROOT/'render_validation.json').write_text(json.dumps({'scope':'All7actual-inference videos fully decoded; each720frames/24s/30fps','passed':True,'videos':rows,
         'exact_terminal_physics_snapshots':terminal_pngs,'exact_terminal_physics_contactsheet':str(contact_path),
         'encoded_last_video_frame_sim_time_s':719/30,'exact_terminal_snapshot_sim_time_s':reports[0]['physical_duration_s']},indent=2)+'\n')
    print('ALL_RENDER_VALID',flush=True)
if __name__=='__main__':main()
