"""Render installed RoboTwin assets with the real RM65-B URDF, without motion.

Static visualization only: no physics stepping, policy, dataset, or hardware I/O.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
import sapien.core as sapien
from transforms3d.quaternions import mat2quat, quat2mat

ROOT = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
OUT = Path(__file__).resolve().parent
MODEL = ROOT / 'assets/embodiments/realman-rm65b-inspire-gen4'
OUT.mkdir(parents=True, exist_ok=True)
sapien.render.set_global_config(max_num_materials=50000, max_num_textures=50000)
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
sapien.render.set_camera_shader_dir('rt')
sapien.render.set_ray_tracing_samples_per_pixel(96)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser('none')
scene = engine.create_scene(sapien.SceneConfig())
scene.set_ambient_light([.4, .4, .4])
scene.add_directional_light([-.4, -.6, -1], [1.6, 1.55, 1.45], shadow=True)
scene.add_directional_light([.5, .2, -1], [.5, .6, .75], shadow=False)
scene.add_point_light([0, .3, 3], [2.5, 2.5, 2.5])

def material(color, roughness=.65, metallic=0):
    m = renderer.create_material()
    m.base_color = [*color, 1]
    m.roughness = roughness
    m.metallic = metallic
    return m

def box(name, pos, half, mat):
    b = scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mat)
    b.set_initial_pose(sapien.Pose(pos))
    return b.build_static(name=name)

floor = material([.73, .77, .8], .85)
scene.add_ground(-.002, render_material=floor)
box('studio_backdrop', [0, -1.80, 5.0], [20, .025, 5.0],
    material([.80, .83, .85], .9))
table_top = .74
table_center_y = .08
box('workbench_top', [0, table_center_y, table_top-.025], [.90, .47, .025],
    material([.77, .81, .83], .45))
box('table_front_trim', [0, .542, table_top-.049], [.90, .008, .018],
    material([.25, .31, .36], .4, .35))
leg_mat = material([.23, .28, .32], .45, .6)
for x in [-.78, .78]:
    for y in [-.29, .45]:
        box('bench_leg', [x, y, .345], [.027, .027, .345], leg_mat)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(str(MODEL / 'robot_sapien.urdf'))
assert robot is not None
robot.set_root_pose(sapien.Pose([0, -.65, .09539821919202804], [0, 0, 0, 1]))
# Static tabletop preparation pose from the archived endpoint IK preview.
# This assigns joint positions for visualization; it does not execute a trajectory.
previous = json.loads((ROOT.parent / 'deploy/realman_gen4/render_scene_20261002/render_report.json').read_text())
q = np.asarray(previous['qpos_preview'], dtype=np.float32)
joints = robot.get_active_joints()
assert len(q) == len(joints)
robot.set_qpos(q)

specs = [
    # Numbering proceeds left-to-right as seen by the robot, back to front.
    ('080_pillbottle', 1, 0, -.19, 'Pill bottle', None),
    ('021_cup', 0, -.52, .08, 'Cup', None),
    ('022_cup-with-liquid', 0, 0, .08, 'Cup with liquid appearance', .06),
    ('040_rack', 0, .52, .08, 'Mug rack', None),
    ('008_tray', 0, -.52, .35, 'Tray', None),
    ('072_electronicscale', 0, 0, .35, 'Electronic scale', None),
    ('062_plasticbox', 3, .52, .35, 'Plastic box', None),
]
orientation = [.7071067811865476, .7071067811865476, 0, 0]
rotation = quat2mat(orientation)
assets = []
for number, (name, model_id, x, y, label, scale_override) in enumerate(specs, 1):
    folder = ROOT / 'assets/objects' / name
    visual = folder / 'visual' / f'base{model_id}.glb'
    data = json.loads((folder / f'model_data{model_id}.json').read_text())
    scale = np.asarray([scale_override]*3 if scale_override else data['scale'])
    mesh_scene = trimesh.load(str(visual), force='scene')
    mesh = mesh_scene.dump(concatenate=True)
    verts = (mesh.vertices * scale) @ rotation.T
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    # Center actual transformed geometry in each grid cell, align its visible base.
    pos = np.asarray([x-(lo[0]+hi[0])/2, y-(lo[1]+hi[1])/2,
                      table_top + .001 - lo[2]])
    b = scene.create_actor_builder()
    b.add_visual_from_file(str(visual), scale=scale)
    b.set_initial_pose(sapien.Pose(pos, orientation))
    b.build_static(name=f'asset_{number:02d}_{name}')
    assets.append(dict(number=number, name=name, model_id=model_id, label=label,
                       visual=str(visual), sha256=hashlib.sha256(visual.read_bytes()).hexdigest(),
                       scale=scale.tolist(), scale_override_for_visualization=bool(scale_override),
                       position=pos.tolist(), quaternion_wxyz=orientation,
                       world_aabb_min=(lo+pos).tolist(), world_aabb_max=(hi+pos).tolist()))
    print('LOADED', number, name, 'size', (hi-lo).round(4), flush=True)

views = [
    ('01_front', 'FRONT', [0, 2.25, 1.65], [0, -.10, .91], 48),
    ('02_front_left', 'FRONT LEFT', [-1.7, 1.80, 1.85], [0, -.11, .88], 47),
    ('03_front_right', 'FRONT RIGHT', [1.7, 1.80, 1.85], [0, -.11, .88], 47),
    ('04_top', 'TOP', [0, .04, 2.8], [0, .04, .72], 53),
    ('05_robot_overhead', 'ROBOT SIDE / OVERHEAD', [0, -1.25, 2.15], [0, .12, .78], 61),
    ('06_table_detail', 'TABLE DETAIL', [1.0, 1.2, 1.55], [0, .06, .84], 55),
]
view_records=[]
w,h=1920,1440
for name, title, eye, target, fov in views:
    cam = scene.add_camera(name, w, h, math.radians(fov), .01, 20)
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target-eye
    forward /= np.linalg.norm(forward)
    up_hint = np.asarray([0,0,1.] if abs(forward[2])<.99 else [0,1.,0])
    left = np.cross(up_hint, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    cam.entity.set_pose(sapien.Pose(eye, mat2quat(np.stack([forward,left,up],axis=1))))
    scene.update_render()
    cam.take_picture()
    color = np.clip(cam.get_picture('Color')[...,:3],0,1)
    img=Image.fromarray((color*255).astype(np.uint8))
    img.save(OUT/f'{name}.png')
    view_records.append(dict(file=f'{name}.png', title=title, eye=eye.tolist(),
                             target=target.tolist(), fovy_degrees=fov,
                             width=w,height=h,pixel_std=float(np.asarray(img).std())))
    print('RENDERED',name,flush=True)

report = dict(date='2026-10-21', renderer='SAPIEN 3 / Vulkan RT, 96 spp, no denoiser',
              scope='Static rendering of installed vendor RealMan URDF and RoboTwin GLB assets',
              physics_steps=0, hardware_io=False, policy_inference=False,
              grasp_or_collection_validated=False, representative_per_category=True,
              beverage_assets_excluded=['001_bottle', '114_bottle'],
              revision='Remove beverage bottles for the laboratory scene, per user request',
              table_top_m=table_top, robot_root_pose=[0,-.65,.09539821919202804,0,0,0,1],
              robot_urdf_sha256=hashlib.sha256((MODEL/'robot_sapien.urdf').read_bytes()).hexdigest(),
              robot_qpos=q.tolist(), robot_joint_names=[j.name for j in joints],
              assets=assets, views=view_records)
(OUT/'render_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

# A contact sheet is assembled from the actual camera renders for navigation.
thumb_w,thumb_h=960,720
sheet=Image.new('RGB',(1968,2380),'#eef2f5')
draw=ImageDraw.Draw(sheet)
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(font_path,25)
small=ImageFont.truetype(font_path,20)
draw.text((24,20),'REALMAN RM65-B | ROBOTWIN TABLETOP ASSETS',fill='#1c3445',font=font)
for i,v in enumerate(view_records):
    x=16+(i%2)*976;y=70+(i//2)*764
    sheet.paste(Image.open(OUT/v['file']).resize((thumb_w,thumb_h),Image.Resampling.LANCZOS),(x,y))
    draw.text((x+8,y+724),v['title'],fill='#20384c',font=small)
sheet.save(OUT/'multiview_overview.png')
print('DONE',OUT,flush=True)
