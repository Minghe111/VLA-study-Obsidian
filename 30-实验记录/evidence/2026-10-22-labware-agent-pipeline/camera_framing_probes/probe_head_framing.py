from pathlib import Path
import json
import sys
import hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from labware_task import labware_pick_place
from camera_framing import apply_camera_framing,project_target
OUT=ROOT/'camera_framing_probes/reset_candidates_v1'
if OUT.exists():raise FileExistsError(OUT)
task=labware_pick_place();task.setup_demo(runtime_path=OUT,seed=0,capture_rgb=True)
q0=np.asarray(task.robot.left_entity.get_qpos()).copy();v0=np.asarray(task.robot.left_entity.get_qvel()).copy()
p0=np.asarray(task.bottle.get_pose().to_transformation_matrix()).copy();tick0=task.physics_step_count
poses=json.loads((ROOT/'collection_runs/official_session_pick_place_v1/runs/capture_ep0000000_seed00000/physics_trace.json').read_text())
rows=[]
for down in [0,15,20,25]:
 config={'preset_id':f'head_extra_down_{down}_vfov37','head_extra_pitch_down_degrees':down,'head_vertical_fov_degrees':37.0,
     'scope':'runtime simulation optical frame only; nominal, not physical calibration'}
 receipt=apply_camera_framing(task,config)
 obs=task.observe_rgb()
 camera=task.cameras.camera_by_name['head_camera']
 projection=project_target(camera,task.vertices,p0)
 swept=[project_target(camera,task.vertices,np.asarray(row['pose'])) for row in poses]
 image=Image.fromarray(obs['full_image']);draw=ImageDraw.Draw(image)
 if projection.get('bbox_xyxy_pixels'):draw.rectangle(projection['bbox_xyxy_pixels'],outline=(230,35,35),width=2)
 image.save(OUT/f'head_extra_down_{down}.png')
 Image.fromarray(obs['left_wrist_image']).save(OUT/f'left_wrist_down_{down}.png')
 Image.fromarray(obs['right_wrist_image']).save(OUT/f'right_wrist_down_{down}.png')
 # Repeated get_obs executes update_mount again; the chosen optical transform
 # must remain active and the render calls must not take a physics step.
 expected=np.asarray(camera.entity.get_pose().to_transformation_matrix()).copy()
 task.get_obs();persisted=np.array_equal(expected,np.asarray(camera.entity.get_pose().to_transformation_matrix()))
 if not persisted:raise RuntimeError('update_mount erased the framing preset')
 if not np.array_equal(q0,np.asarray(task.robot.left_entity.get_qpos())) or not np.array_equal(v0,np.asarray(task.robot.left_entity.get_qvel())):
  raise RuntimeError('Camera probe changed robot state')
 if not np.array_equal(p0,np.asarray(task.bottle.get_pose().to_transformation_matrix())) or tick0!=task.physics_step_count:
  raise RuntimeError('Camera probe changed object/physics state')
 rows.append({'config':config,'runtime_receipt':receipt,'reset_projection':projection,
     'existing_trace_projection_minimum_margin_px':min(r.get('minimum_margin_pixels',-10000) for r in swept),
     'existing_trace_all_geometry_inside':all(r['fully_inside'] for r in swept),
     'persistent_after_repeated_update_mount':persisted,
     'wrist_frame_sha256':{n:hashlib.sha256(obs[n].tobytes()).hexdigest() for n in ('left_wrist_image','right_wrist_image')}})
 print(json.dumps({k:v for k,v in rows[-1].items() if k!='runtime_receipt'}),flush=True)
(OUT/'candidate_receipts.json').write_text(json.dumps(rows,indent=2)+'\n')
# Choose maximum worst-case trace margin; this is a local rendering preset.
valid=[r for r in rows if r['existing_trace_all_geometry_inside'] and r['existing_trace_projection_minimum_margin_px']>=40]
if not valid:raise RuntimeError('No camera candidate gives a 40px full-geometry margin')
best=max(valid,key=lambda r:r['existing_trace_projection_minimum_margin_px'])
selection=dict(best['config'],selection_rule='largest worst-case projected margin over actual v1 trajectory; minimum40px',
    old_trace_projection_min_margin_pixels=best['existing_trace_projection_minimum_margin_px'],
    hardware_calibrated=False,probe_receipt=str(OUT/'candidate_receipts.json'))
(ROOT/'camera_framing_probes/selected_head_framing.json').write_text(json.dumps(selection,indent=2)+'\n')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
canvas=Image.new('RGB',(1280,2*520),(245,247,250));d=ImageDraw.Draw(canvas)
for i,r in enumerate(rows):
 down=r['config']['head_extra_pitch_down_degrees'];x=(i%2)*640;y=(i//2)*520
 canvas.paste(Image.open(OUT/f'head_extra_down_{down}.png'),(x,y+40))
 d.text((x+10,y+8),f'Extra down {down} deg | min trajectory margin {r["existing_trace_projection_minimum_margin_px"]:.1f}px',font=font,fill=(20,35,50))
canvas.save(OUT/'head_framing_candidates.png')
task.close()
print(json.dumps({'selected':selection,'physics_steps':0,'robot_and_object_state_unchanged':True}),flush=True)
