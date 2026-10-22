"""Original RoboTwin bottle transparency preview; no material overrides."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
import sapien.core as sapien
from transforms3d.quaternions import mat2quat,quat2mat

ROOT=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/objects')
OUT=Path(__file__).resolve().parent
engine=sapien.Engine()
renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
sapien.render.set_camera_shader_dir('rt')
sapien.render.set_ray_tracing_samples_per_pixel(128)
sapien.render.set_ray_tracing_path_depth(12)
sapien.render.set_ray_tracing_denoiser('none')
scene=engine.create_scene(sapien.SceneConfig())
scene.set_ambient_light([.4,.4,.4])
scene.add_directional_light([-.4,-.4,-1],[1.6,1.6,1.6],shadow=True)
scene.add_directional_light([.3,.5,-1],[.6,.6,.6],shadow=False)
def material(rgb):
    m=renderer.create_material();m.base_color=[*rgb,1];m.roughness=.8
    return m
def box(pos,half,color,name):
    b=scene.create_actor_builder();b.add_box_visual(half_size=half,material=material(color))
    b.set_initial_pose(sapien.Pose(pos));return b.build_static(name=name)
box([0,0,-.012],[.5,.22,.012],[.55,.59,.62],'floor')
# Black and white board behind the bottle bodies demonstrates visibility through them.
for r in range(9):
    for c in range(27):
        color=[.13,.17,.21] if (r+c)%2 else [.92,.94,.96]
        box([-.39+c*.03,-.067,.015+r*.03],[.015,.002,.015],color,f'checker_{r}_{c}')
rotation=quat2mat([math.sqrt(.5),math.sqrt(.5),0,0])
records=[]
for index,model_id in enumerate([1,2,4]):
    d=ROOT/'114_bottle'
    source=d/'visual'/f'base{model_id}.glb'
    scale=json.loads((d/f'model_data{model_id}.json').read_text())['scale']
    mesh=trimesh.load(str(source),force='scene').dump(concatenate=True)
    verts=(mesh.vertices*np.asarray(scale))@rotation.T
    lo,hi=verts.min(axis=0),verts.max(axis=0)
    center_x=[-.25,0,.25][index]
    pos=[center_x-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,.001-lo[2]]
    b=scene.create_actor_builder();b.add_visual_from_file(str(source),scale=scale)
    b.set_initial_pose(sapien.Pose(pos,[math.sqrt(.5),math.sqrt(.5),0,0]))
    b.build_static(name=f'114_bottle_{model_id}')
    records.append(dict(model_id=model_id,file=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scale=scale,position=pos,size=(hi-lo).tolist()))
    print('LOADED',model_id,flush=True)

camera=scene.add_camera('preview',1800,1000,math.radians(29),.005,5)
eye=np.array([0,1.20,.40]);target=np.array([0,0,.12])
f=target-eye;f/=np.linalg.norm(f)
l=np.cross([0,0,1],f);l/=np.linalg.norm(l);u=np.cross(f,l)
camera.entity.set_pose(sapien.Pose(eye,mat2quat(np.stack([f,l,u],axis=1))))
scene.update_render();camera.take_picture()
rgb=np.clip(camera.get_picture('Color')[...,:3],0,1)
raw=Image.fromarray((rgb*255).astype(np.uint8));raw.save(OUT/'bottles_raw.png')
sheet=Image.new('RGB',(1800,1080),'#eef2f5');sheet.paste(raw,(0,0))
draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
for x,title in [(170,'114_bottle / 4 | alpha 0.50'),(700,'114_bottle / 2 | alpha 0.41'),(1250,'114_bottle / 1 | alpha 0.11')]:
    draw.text((x,1025),title,fill='#1c3445',font=font)
sheet.save(OUT/'transparent_candidates.png')
(OUT/'render_receipt.json').write_text(json.dumps(dict(
    date='2026-10-21',scope='Actual original-asset SAPIEN RT preview against a checkered board',
    material_overrides=False,physics_steps=0,hardware_io=False,
    source_objects=records,image_size=[1800,1000],renderer='SAPIEN 3 RT, 128 spp, no denoiser'),indent=2))
print('DONE',flush=True)
