"""Export a physically timed preview from saved RGB and per-frame simulation clocks."""
from pathlib import Path
import pickle,sys,json,math
import numpy as np
import cv2
out=Path(sys.argv[1])
files=sorted((out/'.cache/episode0').glob('*.pkl'),key=lambda p:int(p.stem))
samples=[pickle.loads(p.read_bytes()) for p in files]
times=np.array([s['recording_clock']['sim_time_s'] for s in samples])
times-=times[0]
fps=30.; frame_count=math.ceil(times[-1]*fps)
p=out/'video/episode_0000000_realtime.mp4'
writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),fps,(320,240))
assert writer.isOpened()
for t in np.arange(frame_count)/fps:
 i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(samples)-1))
 rgb=samples[i]['observation']['head_camera']['rgb']
 writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
writer.release()
cap=cv2.VideoCapture(str(p)); decoded=0
while True:
 ok,frame=cap.read()
 if not ok:break
 assert frame.shape==(240,320,3)
 decoded+=1
cap.release();assert decoded==frame_count
report={'path':str(p),'source':'recorded raw head RGB; no rerendering or synthetic editing',
        'method':'last available frame sampled at 30 FPS using measured simulation timestamps',
        'decoded_frames':decoded,'video_duration_s':frame_count/fps,'simulation_duration_s':float(times[-1]),
        'native_export_video_retained':True}
(out/'realtime_preview.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))