"""RealMan labware and annotation views; no physical robot or physics stepping.

Imported actors and annotations use the original RoboTwin loader. A separate,
coincident visual-only layer supplies transparent SAPIEN materials, because
the generic GLB import does not reproduce those transmission parameters.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sapien.core as sapien
from transforms3d.quaternions import mat2quat, quat2mat

from annotation_contract import load_with_official_create_actor

HERE = Path(__file__).resolve().parent
OUT = HERE / 'scene_visualization'
OUT.mkdir(exist_ok=True)
ROOT = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
layout = json.loads((HERE/'scene_layout.json').read_text())
interface = json.loads((HERE/'asset_interface.json').read_text())
items = {item['id']: item for item in interface['assets']}
interface_sha = hashlib.sha256((HERE/'asset_interface.json').read_bytes()).hexdigest()

sapien.render.set_global_config(max_num_materials=50000, max_num_textures=50000)
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
sapien.render.set_camera_shader_dir('rt')
sapien.render.set_ray_tracing_samples_per_pixel(96)
sapien.render.set_ray_tracing_path_depth(12)
sapien.render.set_ray_tracing_denoiser('none')
scene = engine.create_scene(sapien.SceneConfig())
scene.set_ambient_light([.4, .4, .4])
scene.add_directional_light([-.4, -.6, -1], [1.6, 1.55, 1.45], shadow=True)
scene.add_directional_light([.5, .2, -1], [.5, .6, .75], shadow=False)
scene.add_point_light([0, .3, 3], [2.5, 2.5, 2.5])


def material(spec):
    mat = renderer.create_material()
    for key in ['base_color', 'roughness', 'metallic', 'transmission',
                'ior', 'transmission_roughness']:
        if key in spec:
            setattr(mat, key, spec[key])
    return mat


def box(name, position, half, color):
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=half, material=material({'base_color':[*color,1], 'roughness':.6}))
    builder.set_initial_pose(sapien.Pose(position))
    return builder.build_static(name=name)


scene.add_ground(-.002, render_material=material({'base_color':[.73,.77,.8,1]}))
box('backdrop', [0,-1.8,5], [20,.025,5], [.80,.83,.85])
top = layout['table']['top_m']
box('table', [0,.08,top-.025], [.90,.47,.025], [.37,.44,.48])
for x in [-.78,.78]:
    for y in [-.29,.45]:
        box('leg', [x,y,.345], [.027,.027,.345], [.23,.28,.32])

urdf = ROOT/layout['robot_model']
assert hashlib.sha256(urdf.read_bytes()).hexdigest() == layout['robot_urdf_sha256']
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(str(urdf))
pose = layout['root_pose_xyz_quaternion_wxyz']
robot.set_root_pose(sapien.Pose(pose[:3], pose[3:]))
names = [joint.name for joint in robot.get_active_joints()]
robot.set_qpos([layout['joint_positions_rad'][name] for name in names])

scale = layout['official_scale']
assert hashlib.sha256(Path(scale['visual']).read_bytes()).hexdigest() == scale['sha256']
builder = scene.create_actor_builder()
builder.add_visual_from_file(scale['visual'], scale=scale['scale'])
builder.set_initial_pose(sapien.Pose(scale['position'],scale['quaternion_wxyz']))
builder.build_static(name='original_electronic_scale')

actors = {}
actor_records = []
for entry in layout['assets']:
    ident = entry['id']
    item = items[ident]
    yaw = entry['yaw_rad']
    pose = sapien.Pose(entry['position_m'], [math.cos(yaw/2),0,0,math.sin(yaw/2)])
    wrapped = load_with_official_create_actor(scene, HERE/'prepared'/ident, pose, convex=True)
    for component in wrapped.actor.components:
        if isinstance(component, sapien.render.RenderBodyComponent):
            component.visibility = 0.0
    # No additional collision or state: this layer follows the exact static pose.
    builder = scene.create_actor_builder()
    for group in item['visual_groups']:
        builder.add_visual_from_file(group['filename'], material=material(group['material']))
    builder.set_initial_pose(pose)
    builder.build_static(name=f'{ident}_transparent_visual_overlay')
    actors[ident] = wrapped
    record = {'id':ident, 'position_m':pose.p.tolist(), 'quaternion_wxyz':pose.q.tolist(),
              'official_collision_shapes':sum(len(component.collision_shapes)
                    for component in wrapped.actor.components
                    if hasattr(component, 'collision_shapes')),
              'world_keypoints':{}}
    for semantic,key in [('contact','contact_points_pose'),('target','target_pose'),
                         ('functional','functional_matrix'),('orientation','orientation_point')]:
        record['world_keypoints'][semantic] = [
            np.asarray(wrapped.get_point(semantic,i,'matrix')).tolist()
            for i in range(len(wrapped.config[key]))]
    actor_records.append(record)
    print('OFFICIAL_LOADED',ident,record['official_collision_shapes'],flush=True)

# Populate the closed bottle with the previously simulated upright contents.
trace = json.loads((HERE/layout['physical_pill_snapshot']['trace']).read_text())
frame = min(trace,key=lambda row:abs(row['time_s']-.8))
body_record = sapien.Pose(frame['body_pose'][:3],frame['body_pose'][3:])
body_in_scene = actors['reagent_bottle_body'].get_pose()
white = material({'base_color':[.97,.97,.89,1], 'roughness':.3})
for index,tablet in enumerate(frame['tablets']):
    old_pose = sapien.Pose(tablet['pose'][:3],tablet['pose'][3:])
    pose = body_in_scene * body_record.inv() * old_pose
    builder = scene.create_actor_builder()
    builder.add_capsule_visual(radius=.003,half_length=.002,material=white)
    builder.set_initial_pose(pose)
    builder.build_static(name=f'pill_recorded_visual_{index:02d}')


def camera(name, eye, target, fov, width=1920, height=1440):
    cam = scene.add_camera(name,width,height,math.radians(fov),.005,20)
    eye,target = np.asarray(eye,float),np.asarray(target,float)
    forward = target-eye
    forward /= np.linalg.norm(forward)
    hint = np.array([0,0,1.]) if abs(forward[2])<.99 else np.array([0,1.,0])
    left = np.cross(hint,forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward,left)
    cam.entity.set_pose(sapien.Pose(eye,mat2quat(np.stack([forward,left,up],axis=1))))
    return cam


records = []
def save_view(name,title,eye,target,fov):
    cam = camera(name,eye,target,fov)
    scene.update_render()
    cam.take_picture()
    rgb = (np.clip(cam.get_picture('Color')[...,:3],0,1)*255).astype(np.uint8)
    Image.fromarray(rgb).save(OUT/f'{name}.png')
    records.append({'file':f'{name}.png','title':title,'eye':eye,'target':target,'fovy_degrees':fov})
    print('RENDERED',name,flush=True)


save_view('01_front','REALMAN | IMPORTED LABWARE',[0,2.25,1.65],[0,-.10,.91],48)
save_view('02_operation','NEAR OPERATION ZONE',[.70,.95,1.33],[.05,-.01,.82],43)
save_view('03_top','TOP VIEW',[0,.04,2.4],[0,.04,.74],53)
save_view('04_glassware','BEAKER | FLASK | GRADUATED CYLINDER',[.72,.92,1.24],[0,.14,.85],43)

# Draw annotation markers only after capturing the clean layout images.
colors = {'contact':[.06,.38,.95,1], 'opening':[1,.43,.04,1],
          'support':[.08,.78,.35,1], 'rim':[.93,.12,.20,1]}
def marker(name,xyz,rgba,radius=.0025):
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=radius, material=material({'base_color':rgba,'roughness':.25}))
    builder.set_initial_pose(sapien.Pose(xyz))
    builder.build_static(name=name)


def line(name,start,end,color,radius=.0007):
    start,end = np.asarray(start),np.asarray(end)
    direction = end-start
    length = np.linalg.norm(direction)
    direction /= length
    hint = np.array([0,0,1.]) if abs(direction[2])<.99 else np.array([0,1.,0])
    left = np.cross(hint,direction)
    left /= np.linalg.norm(left)
    up = np.cross(direction,left)
    builder = scene.create_actor_builder()
    builder.add_capsule_visual(radius=radius,half_length=length/2,
                               material=material({'base_color':color}))
    builder.set_initial_pose(sapien.Pose((start+end)/2,mat2quat(np.stack([direction,left,up],axis=1))))
    builder.build_static(name=name)


for ident,wrapped in actors.items():
    contact = np.asarray(wrapped.get_point('contact',0,'matrix'))
    marker(ident+'_grasp',contact[:3,3],colors['contact'])
    for i in range(4):
        frame = np.asarray(wrapped.get_point('contact',i,'matrix'))
        start = frame[:3,3]
        # +Y is outward/pregrasp; the actual approach is back along this line.
        end = start+.034*frame[:3,1]
        line(ident+f'_pregrasp_{i}',start,end,colors['contact'])
        marker(ident+f'_pregrasp_end_{i}',end,colors['contact'],.0015)
    opening = np.asarray(wrapped.get_point('functional',0,'matrix'))
    marker(ident+'_opening',opening[:3,3],colors['opening'])
    line(ident+'_opening_axis',opening[:3,3],opening[:3,3]+.025*opening[:3,2],colors['opening'])
    support = np.asarray(wrapped.get_point('target',0,'matrix'))
    marker(ident+'_support',support[:3,3],colors['support'])
    if ident!='reagent_bottle_cap':
        rim = np.asarray(wrapped.get_point('functional',2,'matrix'))
        marker(ident+'_rim',rim[:3,3],colors['rim'])

save_view('05_annotations','FUNCTION POINTS | LOCAL ANNOTATIONS',[.64,.83,1.29],[.04,.02,.83],43)
save_view('06_annotations_top','PREGRASP CENTRES + OPENINGS',[0,.045,1.62],[0,.045,.79],48)

font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
sheet = Image.new('RGB',(1968,2406),'#edf1f4')
draw = ImageDraw.Draw(sheet)
draw.text((20,18),'REALMAN LABWARE | IMPORTS + CANDIDATE FUNCTION POINTS',fill='#20384c',font=font)
for i,record in enumerate(records):
    x,y = 16+(i%2)*976,64+(i//2)*764
    with Image.open(OUT/record['file']) as im:
        sheet.paste(im.resize((960,720),Image.Resampling.LANCZOS),(x,y))
    draw.text((x+6,y+724),record['title'],fill='#20384c',font=small)
draw.text((20,2360),'BLUE grasp/pregrasp | ORANGE opening/axis | GREEN support | RED candidate rim',fill='#20384c',font=small)
sheet.save(OUT/'multiview_overview.png')

receipt={'date':'2026-10-22','renderer':'SAPIEN 3 Vulkan RT, 96 spp',
         'physics_steps':0,'hardware_io':False,'policy_inference':False,
         'robot_urdf_sha256':layout['robot_urdf_sha256'],
         'asset_interface_sha256_at_load':interface_sha,
         'scene_layout_sha256':hashlib.sha256((HERE/'scene_layout.json').read_bytes()).hexdigest(),
         'official_loader_actors':actor_records,'views':records,
         'visual_materials':'Coincident visual-only OBJ groups with declared SAPIEN transmission/ior; original official collision actors retained.',
         'tablet_visuals':'12 poses from upright physics record, transformed rigidly inside the static bottle.',
         'point_scope':'Geometry-derived candidate annotations, not executed RealMan grasps.'}
(HERE/'scene_render_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('DONE',OUT,flush=True)
