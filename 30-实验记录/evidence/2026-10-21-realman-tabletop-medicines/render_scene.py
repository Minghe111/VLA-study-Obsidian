"""Render the saved RealMan tabletop preset using installed official assets.

Run with the existing robotwin Python environment. The scene assigns a static
joint configuration; it does not step physics or communicate with hardware.
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sapien.core as sapien
import trimesh
from transforms3d.quaternions import mat2quat, quat2mat

ROOT = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
OUT = Path(__file__).resolve().parent
PRESET_PATH = OUT / 'scene_preset.json'
preset = json.loads(PRESET_PATH.read_text())
urdf_path = ROOT / preset['robot_model']
urdf_sha = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
assert urdf_sha == preset['robot_urdf_sha256'], 'Robot model changed; review preset.'

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
    mat = renderer.create_material()
    mat.base_color = [*color, 1]
    mat.roughness = roughness
    mat.metallic = metallic
    return mat


def box(name, position, half_size, mat):
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=half_size, material=mat)
    builder.set_initial_pose(sapien.Pose(position))
    return builder.build_static(name=name)


scene.add_ground(-.002, render_material=material([.73, .77, .8], .85))
box('studio_backdrop', [0, -1.80, 5], [20, .025, 5],
    material([.80, .83, .85], .9))
table = preset['table']
table_top = table['top_m']
table_y = table['center_y_m']
half_x, half_y = table['half_size_xy_m']
box('workbench_top', [0, table_y, table_top-.025], [half_x, half_y, .025],
    material([.45, .51, .54], .55))
box('table_front_trim', [0, .542, table_top-.049], [.90, .008, .018],
    material([.25, .31, .36], .4, .35))
leg_mat = material([.23, .28, .32], .45, .6)
for x, y in itertools.product([-.78, .78], [-.29, .45]):
    box('bench_leg', [x, y, .345], [.027, .027, .345], leg_mat)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(str(urdf_path))
assert robot is not None
root_pose = preset['root_pose_xyz_quaternion_wxyz']
robot.set_root_pose(sapien.Pose(root_pose[:3], root_pose[3:]))
joints = robot.get_active_joints()
joint_names = [joint.name for joint in joints]
assert set(joint_names) == set(preset['joint_positions_rad'])
q = np.asarray([preset['joint_positions_rad'][name] for name in joint_names],
               dtype=np.float32)
joint_xml = {element.attrib['name']: element
             for element in ET.parse(urdf_path).getroot().findall('joint')}
limit_checks = []
for name, value in zip(joint_names, q):
    limit = joint_xml[name].find('limit')
    if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
        lower, upper = float(limit.attrib['lower']), float(limit.attrib['upper'])
        valid = lower-1e-6 <= value <= upper+1e-6
        assert valid, (name, value, lower, upper)
        limit_checks.append(dict(joint=name, lower=lower, upper=upper,
                                 value=float(value), within_limits=bool(valid)))
links = {link.name: link for link in robot.get_links()}


def palm_frames():
    frames = {}
    for side in ['l', 'r']:
        pose = links[f'{side}_inspire_base_link'].get_pose()
        rot = quat2mat(pose.q)
        frames[side] = dict(position_m=pose.p.tolist(),
                            palm_normal_world=rot[:, 1].tolist(),
                            finger_long_axis_world=rot[:, 2].tolist())
    return frames


correction = preset['pose_correction']
old_q = q.copy()
old_q[joint_names.index(correction['joint'])] = correction['before_rad']
robot.set_qpos(old_q)
before = palm_frames()
robot.set_qpos(q)
after = palm_frames()
assert np.allclose(before['l']['position_m'], after['l']['position_m'], atol=1e-5)
assert np.allclose(before['l']['finger_long_axis_world'],
                   after['l']['finger_long_axis_world'], atol=1e-5)
assert np.dot(before['l']['palm_normal_world'],
              after['l']['palm_normal_world']) < -.9999
assert np.allclose(before['r']['palm_normal_world'],
                   after['r']['palm_normal_world'], atol=1e-6)

assets = []
for number, spec in enumerate(preset['objects'], 1):
    category, model_id = spec['category'], spec['model_id']
    if category == 'objaverse/plate':
        folder = ROOT / 'assets/objects/objaverse/plate' / f'{model_id:03d}'
        visual = folder / 'textured.obj'
        # Match this plate's original URDF, including Z-up identity orientation.
        scale = np.asarray([.001, .001, .001])
        orientation = [1, 0, 0, 0]
    else:
        folder = ROOT / 'assets/objects' / category
        visual = folder / 'visual' / f'base{model_id}.glb'
        data = json.loads((folder / f'model_data{model_id}.json').read_text())
        scale = np.asarray(data['scale'])
        orientation = [.7071067811865476, .7071067811865476, 0, 0]
    visual.resolve().relative_to((ROOT/'assets/objects').resolve())
    mesh = trimesh.load(str(visual), force='scene').dump(concatenate=True)
    vertices = (mesh.vertices*scale) @ quat2mat(orientation).T
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    x, y = spec['center_xy_m']
    position = np.asarray([x-(lo[0]+hi[0])/2, y-(lo[1]+hi[1])/2,
                           table_top+table['visual_clearance_m']-lo[2]])
    world_lo, world_hi = lo+position, hi+position
    assert world_lo[0] >= -half_x and world_hi[0] <= half_x
    assert world_lo[1] >= table_y-half_y and world_hi[1] <= table_y+half_y
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(visual), scale=scale)
    builder.set_initial_pose(sapien.Pose(position, orientation))
    builder.build_static(name=f'asset_{number:02d}_{category}')
    assets.append(dict(number=number, **spec, visual=str(visual),
                       sha256=hashlib.sha256(visual.read_bytes()).hexdigest(),
                       scale=scale.tolist(), scale_override_for_visualization=False,
                       position=position.tolist(), quaternion_wxyz=orientation,
                       world_aabb_min=world_lo.tolist(), world_aabb_max=world_hi.tolist(),
                       size_m=(hi-lo).tolist()))
    print('LOADED', category, model_id, 'size_m', (hi-lo).round(4), flush=True)

for first, second in itertools.combinations(assets, 2):
    lo = np.maximum(first['world_aabb_min'][:2], second['world_aabb_min'][:2])
    hi = np.minimum(first['world_aabb_max'][:2], second['world_aabb_max'][:2])
    assert not np.all(hi > lo), ('Object footprints overlap', first['number'], second['number'])

views = [
    ('01_front', 'FRONT | CORRECTED DEFAULT POSE', [0, 2.25, 1.65], [0, -.10, .91], 48),
    ('02_front_left', 'FRONT LEFT', [-1.7, 1.8, 1.85], [0, -.11, .88], 47),
    ('03_front_right', 'FRONT RIGHT', [1.7, 1.8, 1.85], [0, -.11, .88], 47),
    ('04_top', 'TOP | EIGHT OFFICIAL OBJECTS', [0, .04, 2.6], [0, .04, .72], 53),
    ('05_robot_overhead', 'ROBOT SIDE OVERHEAD', [0, -1.25, 2.15], [0, .12, .78], 61),
    ('06_table_detail', 'TABLETOP DETAIL', [.85, 1.15, 1.4], [0, .10, .82], 52),
    ('07_hands_detail', 'HANDS | PALMS FACING INWARD', [0, .85, 1.05], [0, -.08, .86], 42),
]
view_records = []
width, height = 1920, 1440


def render_camera(name, eye, target, fov):
    camera = scene.add_camera(name, width, height, math.radians(fov), .01, 20)
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target-eye
    forward /= np.linalg.norm(forward)
    up_hint = np.asarray([0, 0, 1.] if abs(forward[2]) < .99 else [0, 1., 0])
    left = np.cross(up_hint, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    camera.entity.set_pose(sapien.Pose(eye, mat2quat(rotation)))
    scene.update_render()
    camera.take_picture()
    color = np.clip(camera.get_picture('Color')[..., :3], 0, 1)
    image = Image.fromarray((color*255).astype(np.uint8))
    image.save(OUT/f'{name}.png')
    return dict(file=f'{name}.png', eye=eye.tolist(), target=target.tolist(),
                fovy_degrees=fov, width=width, height=height,
                pixel_std=float(np.asarray(image).std()))


for name, title, eye, target, fov in views:
    view_records.append(dict(title=title, **render_camera(name, eye, target, fov)))
    print('RENDERED', name, flush=True)

# The old posture is rendered separately as evidence, never saved as the default.
robot.set_qpos(old_q)
render_camera('left_wrist_before', [0, .85, 1.05], [0, -.08, .86], 42)
robot.set_qpos(q)

font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_file, 25)
small = ImageFont.truetype(font_file, 20)
sheet = Image.new('RGB', (1968, 2380), '#eef2f5')
draw = ImageDraw.Draw(sheet)
draw.text((24, 20), 'REALMAN | MEDICINAL CONTAINERS RESTORED + CORRECTED LEFT WRIST',
          fill='#1c3445', font=font)
for index, view in enumerate(view_records[:6]):
    x, y = 16+(index % 2)*976, 70+(index//2)*764
    with Image.open(OUT/view['file']) as im:
        sheet.paste(im.resize((960, 720), Image.Resampling.LANCZOS), (x, y))
    draw.text((x+8, y+724), view['title'], fill='#20384c', font=small)
sheet.save(OUT/'multiview_overview.png')

comparison = Image.new('RGB', (1968, 820), '#eef2f5')
draw = ImageDraw.Draw(comparison)
for index, (filename, title) in enumerate([
        ('left_wrist_before.png', 'BEFORE | LEFT PALM OUTWARD'),
        ('07_hands_detail.png', 'SAVED DEFAULT | LEFT PALM INWARD')]):
    x = 16+index*976
    with Image.open(OUT/filename) as im:
        comparison.paste(im.resize((960, 720), Image.Resampling.LANCZOS), (x, 16))
    draw.text((x+8, 752), title, fill='#20384c', font=font)
comparison.save(OUT/'left_wrist_comparison.png')

report = dict(date=preset['date'], renderer='SAPIEN 3 / Vulkan RT, 96 spp, no denoiser',
              scope='Static visualization using saved scene-specific default pose',
              scene_preset=str(PRESET_PATH),
              scene_preset_sha256=hashlib.sha256(PRESET_PATH.read_bytes()).hexdigest(),
              robot_urdf_sha256=urdf_sha, robot_joint_names=joint_names,
              robot_qpos=q.tolist(), robot_root_pose=root_pose,
              pose_correction=correction, palm_frames_before=before,
              palm_frames_after=after, joint_limit_checks=limit_checks,
              physics_steps=0, hardware_io=False, policy_inference=False,
              collision_validated=False, grasp_or_collection_validated=False,
              external_object_assets=False, beverage_assets_excluded=True,
              medicinal_container_count=4,
              table=table, assets=assets, views=view_records,
              source_robotwin_head='931ce21f4f0e63feab661f349634d5a25bd92681')
(OUT/'render_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print('PALM_FRAMES_AFTER', json.dumps(after), flush=True)
print('DONE', OUT, flush=True)
