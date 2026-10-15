"""Render only newly captured RGB, on the actual 30fps simulation-time grid."""
from pathlib import Path
import argparse,json,subprocess,hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--indices',nargs='+',type=int,default=[0,38,30,31,17,7]);a=ap.parse_args()
    dest=ROOT/'rendered_videos';dest.mkdir(exist_ok=True);rows=[]
    cv2.setNumThreads(2)
    for i in a.indices:
        folder=ROOT/'episodes'/f'ep{i:02d}_model80000';report=json.loads((folder/'report.json').read_text())
        source=json.loads((folder/'actual_rgb/observations.json').read_text())['frames']
        external=json.loads((folder/'observer_rgb/observations.json').read_text())['frames']
        assert len(source)==len(external) and all(s['physics_step']==e['physics_step'] for s,e in zip(source,external))
        times=np.asarray([s['sim_time_s'] for s in source]);duration=float(report['physical_duration_s']);frame_count=round(duration*30)
        assert abs(frame_count/30-duration)<1e-5 and len(source)==401
        target=dest/f'episode_{i:07d}_model80000_four_views.mp4'
        if target.exists():raise FileExistsError(target)
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x1004','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(target)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);latest=-1;tiles=None
        try:
            for n in range(frame_count):
                sim=n/30;frame=max(0,int(np.searchsorted(times,sim,side='right')-1))
                if frame!=latest:
                    with np.load(source[frame]['path']) as z:ims={k:z[k] for k in ('cam_head','cam_left_wrist','cam_right_wrist')}
                    with np.load(external[frame]['path']) as z:ims['external']=z['external']
                    tiles=[]
                    for key,label in [('external','External observer (display only)'),('cam_head','Head camera (policy input)'),('cam_left_wrist','Left wrist (policy input)'),('cam_right_wrist','Right wrist (policy input)')]:
                        im=ims[key].copy();cv2.rectangle(im,(0,0),(640,29),(24,24,24),-1);cv2.putText(im,label,(12,21),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1,cv2.LINE_AA);tiles.append(im)
                    latest=frame
                canvas=np.zeros((1004,1280,3),np.uint8);canvas[44:524]=np.concatenate(tiles[:2],axis=1);canvas[524:1004]=np.concatenate(tiles[2:],axis=1)
                ep=report['source_episode'];title=f"Checkpoint80000 | source episode{i:07d} | {ep['brand']} {ep['arm']} {ep['pose_class']} | sim {sim:05.2f}s"
                cv2.putText(canvas,title,(14,29),cv2.FONT_HERSHEY_SIMPLEX,.66,(255,255,255),1,cv2.LINE_AA)
                proc.stdin.write(canvas.tobytes())
            proc.stdin.close();code=proc.wait()
            if code:raise RuntimeError('ffmpeg failed'+str(code))
        except BaseException:
            if proc.poll() is None:proc.kill();proc.wait()
            raise
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)],text=True))
        v=next(s for s in probe['streams'] if s['codec_type']=='video')
        if int(v['nb_frames'])!=frame_count or abs(float(probe['format']['duration'])-24.)>.01:raise RuntimeError('Rendered time/framecount differs')
        row={'index':i,'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,'fps':30,'frames':frame_count,'duration_s':float(probe['format']['duration']),
             'source_actual_observations':len(source),'source_sim_time_grid_s':times.tolist(),
             'method':'Zero order hold of actual new physical RGB on30fps simulation-time grid; no generated/interpolated image pixels',
             'expert_source_RGB_or_video_used':False,'fourth_observer_in_policy_inputs':False}
        rows.append(row);print('VIDEO_READY',i,str(target),flush=True)
    (dest/('render_receipt_'+'_'.join(map(str,a.indices))+'.json')).write_text(json.dumps({'videos':rows},indent=2)+'\n')
if __name__=='__main__':main()
