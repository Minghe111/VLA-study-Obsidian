"""Original HDF RGB visualization only; no model/simulator/hardware calls."""
from pathlib import Path
import hashlib,json,math,subprocess,io,concurrent.futures
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/tmp/realman-data-motion-audit-20261013')
MEDIA=ROOT/'media';MEDIA.mkdir(exist_ok=True)
BIN=Path('/home/admin123/liminghe/miniconda3/envs/robotwin-official-data/bin')
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')
CAM_NAMES=('原头部 RGB','原左腕 RGB','原右腕 RGB')
W,H=1920,632
FPS=30
SOURCE_ROWS={}
for receipt_name in ('rgb31_receipt.json','rgb_selected_receipt.json'):
    for row in json.loads((ROOT/receipt_name).read_text())['episodes']:SOURCE_ROWS[row['index']]=row
META={row['index']:row for row in json.loads((ROOT/'numeric_receipt.json').read_text())['episodes']}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def render_one(index):
    source=SOURCE_ROWS[index];meta=META[index]
    folder=ROOT/'rgb'/f'episode_{index:07d}'
    with np.load(ROOT/'numeric'/f'episode_{index:07d}.npz',allow_pickle=False) as a:
        times=a['realman_dexterous/sample_time_s'].copy()
        T=float(a['realman_dexterous/next_sample_time_s'][-1])
        state=a['realman_dexterous/measured_state24'].copy()
    assert len(times)==source['frames'] and np.array_equal(times,np.asarray(source['times']))
    assert meta['source_hdf_sha256']==source['source_sha256']
    joint_column=5 if meta['arm']=='left' else 17
    j6=np.rad2deg(state[:,joint_column])
    assert state.shape==(len(times),24) and np.isfinite(j6).all()
    decoded=0
    # Verify every original encoded JPEG before visualization; never overwrite it.
    for row in source['files']:
        p=ROOT/row['path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        with Image.open(p) as im:
            im.load();assert im.format=='JPEG' and im.mode=='RGB' and im.size==(640,480)
        decoded+=1
    assert decoded==3*len(times)
    font=ImageFont.truetype(FONT,26);small=ImageFont.truetype(FONT,22)
    cache={}
    def images_at(row):
        if row not in cache:
            ims=[]
            for camera in CAMERAS:
                with Image.open(folder/camera/f'{row:04d}.jpg') as im:ims.append(im.copy())
            cache.clear();cache[row]=ims
        return cache[row]
    def canvas(row,requested):
        image=Image.new('RGB',(W,H),'#0c1420');d=ImageDraw.Draw(image)
        d.text((18,8),f'原训练 HDF RGB | source {index} | {meta["brand"]} / {meta["arm"]} / {meta["pose_class"]} | 非当前仿真／模型',font=font,fill='white')
        d.text((18,44),f'播放 t={requested:.3f}s  原图 t={times[row]:.3f}s  实际 {meta["arm"]} J6={j6[row]:+.2f}°  原始角度未 wrap',font=small,fill='#7ce3d4')
        for k,(im,label) in enumerate(zip(images_at(row),CAM_NAMES)):
            x=k*640;d.text((x+14,79),label,font=small,fill='#e5ebf5');image.paste(im,(x,108))
        footer='30 fps 零阶保持原始图像；末尾以最后 t=T−0.06 原图补齐时长，无原始 T 时刻新图像'
        d.text((18,598),footer,font=small,fill='#d0d9e8')
        return image
    output=MEDIA/f'episode_{index:07d}_original_hdf_rgb.mp4'
    count=int(math.ceil(T*FPS-1e-8))
    log=MEDIA/f'episode_{index:07d}_encode.log'
    with log.open('wb') as error:
        proc=subprocess.Popen([str(BIN/'ffmpeg'),'-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)],stdin=subprocess.PIPE,stderr=error)
        mapping=[]
        try:
            for n in range(count):
                t=n/FPS;row=max(0,min(len(times)-1,int(np.searchsorted(times,t,side='right')-1)))
                proc.stdin.write(canvas(row,t).tobytes())
                mapping.append({'output_frame':n,'output_time_s':t,'source_row':row,'source_time_s':float(times[row]),'source_actual_J6_deg':float(j6[row])})
            proc.stdin.close();assert proc.wait()==0
        except BaseException:
            proc.kill();proc.wait();raise
    # Five requested source stages; three original views per stage, presentation resize only.
    sheet=Image.new('RGB',(1200,5*340+80),'#0c1420');d=ImageDraw.Draw(sheet)
    d.text((15,8),f'原HDF RGB（非模型／新仿真） | source {index} | {meta["arm"]} {meta["brand"]}',font=font,fill='white')
    sheet_rows=[]
    for k,t in enumerate((0.,3.,6.,9.,12.)):
        row=max(0,min(len(times)-1,int(np.searchsorted(times,t,side='right')-1)))
        y=80+k*340;d.text((15,y),f'要求 {t:g}s → 原图 {times[row]:.3f}s / row {row} / 实际 J6 {j6[row]:+.2f}°',font=small,fill='#7ce3d4')
        for camera_index,im in enumerate(images_at(row)):
            sheet.paste(im.resize((400,300),Image.Resampling.LANCZOS),(camera_index*400,y+36))
        sheet_rows.append({'requested_time_s':t,'source_row':row,'source_time_s':float(times[row]),'source_actual_J6_deg':float(j6[row])})
    contact=MEDIA/f'episode_{index:07d}_source_contactsheet.png';sheet.save(contact)
    probe=json.loads(subprocess.check_output([str(BIN/'ffprobe'),'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration','-show_entries','format=duration','-of','json',str(output)]))
    duration=float(probe['format']['duration']);stream=probe['streams'][0]
    assert int(stream['nb_read_frames'])==count and stream['codec_name']=='h264' and (stream['width'],stream['height'])==(W,H)
    assert T<=duration+1e-6 and duration-T<1/FPS+1e-6
    # Full video decode, independent of frame-count/probe metadata.
    subprocess.run([str(BIN/'ffmpeg'),'-v','error','-i',str(output),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    out={'index':index,'scope':'Original immutable HDF RGB source visualization, not new simulation/model inference',
         'source_hdf_sha256':source['source_sha256'],'numeric_npz_sha256':sha(ROOT/'numeric'/f'episode_{index:07d}.npz'),
         'active_arm':meta['arm'],'J6_field':'realman_dexterous/measured_state24','J6_column':joint_column,'J6_unit':'raw actual rad converted to degree, no unwrap/wrap',
         'source_frames':len(times),'all_original_jpeg_sha_and_decode_verified':True,'decoded_JPEG_count':decoded,
         'source_last_RGB_time_s':float(times[-1]),'physical_duration_T_s':T,'source_T_RGB_exists':False,
         'tail':'last original RGB frame held to physical T; 30fps duration rounded upward within one output frame',
         'output_video_duration_s':duration,'output_frame_count':count,'fps':FPS,'zero_order_hold':True,
         'video':str(output),'video_sha256':sha(output),'contactsheet':str(contact),'contactsheet_sha256':sha(contact),
         'ffprobe':probe,'video_full_decode_passed':True,'frame_mapping':mapping,'contactsheet_rows':sheet_rows,'original_files_modified':False}
    (MEDIA/f'episode_{index:07d}_render_receipt.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print('RENDER_DONE',index,len(times),count,duration,flush=True)
    return {k:v for k,v in out.items() if k not in ('frame_mapping','ffprobe')}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(render_one,(17,1,31,16)))
    final={'scope':'Four original training HDF RGB source videos; no simulator/model execution','passed':True,'episodes':results,'render_script_sha256':sha(Path(__file__))}
    (MEDIA/'render_validation.json').write_text(json.dumps(final,indent=2,ensure_ascii=False)+'\n')
    print('ALL_FOUR_ORIGINAL_SOURCE_VIDEOS_VALIDATED',flush=True)
