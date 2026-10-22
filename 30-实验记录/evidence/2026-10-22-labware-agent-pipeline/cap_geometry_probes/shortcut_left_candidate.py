"""One bounded CPU shortcut of the already saved left-arm RRT candidate.

No new search, physical execution, collision filtering, or endpoint snapping.
Greedy shortcut follows native acob_expert.clearance_shortcut's strategy.
"""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from mplib import PlanningWorld
from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parent
sys.path.insert(0, str(PIPE / 'robot_cap_task'))
from cap_grasp_probe import Geometry, sha
from left_cap_held_probe import set_hand

folder = ROOT / 'left115_static_shortcut_v1'
folder.mkdir(exist_ok=False)
rawfile = ROOT / 'left115_static_rrt_fast_v2/candidate_plan.json'
snapshot = PIPE / 'robot_cap_task/static_hold_probes/held115_static_reference_v1/actual_terminal_snapshot.json'
rebind = ROOT / 'left115_static_rebind400/rebind.json'
d = json.loads(rebind.read_text())
raw = json.loads(rawfile.read_text())
path = np.asarray(raw['raw_arm6_waypoints'])
g = Geometry(folder, snapshot=snapshot, geometry_spec_path=PIPE / 'robot_cap_task/profiles/bottle115/physics_spec115.json')
start = set_hand(g, g.q0, d['open_sdk6'])
arm = [g.index[f'l_joint{i}'] for i in range(1, 7)]
g.model.set_qpos(start, full=True)
g.model.set_move_group('l_inspire_base_link')
obstacles = []
for obj in (g.body_obj, g.cap_obj, g.table_obj):
    shapes = [g.fcl.CollisionObject(x.get_collision_geometry()) for x in obj.shapes]
    obstacles.append(g.fcl.FCLObject(obj.name, g.Pose(obj.pose), shapes, [g.Pose(x) for x in obj.shape_poses]))
world = PlanningWorld([g.model], obstacles)
acm = world.get_allowed_collision_matrix()
for pair in g.allowed:
    a, b = sorted(pair)
    acm.set_entry(a, b, True)
roles = []
for name in g.right_names:
    for label, obj in [('body', g.body_obj), ('cap', g.cap_obj)]:
        if g.distance(g.objects[name], obj)[1]:
            acm.set_entry(name, obj.name, True)
            roles.append([name, label, obj.name])
proxy = SimpleNamespace(urdf_tree=ET.fromstring(export_kinematic_chain_urdf(g.art)), vertices=g.vertices, sim_joints=g.names)
coeff = np.asarray([c[arm] for c in g.fma.FullMeshPathAuditor._motion_coefficients(proxy).values()])
t0 = time.monotonic()
deadline = t0 + 25.
queries = 0
attempts = 0
cache = {}

def edge(a, b):
    global queries, attempts
    if time.monotonic() > deadline:
        raise TimeoutError('One bounded 25s shortcut budget exhausted')
    key = (a.tobytes(), b.tobytes())
    if key in cache:
        return cache[key]
    attempts += 1
    count = max(1, int(np.ceil(float(np.max(coeff @ np.abs(b-a))) / .001)))
    for k in range(count+1):
        if time.monotonic() > deadline:
            raise TimeoutError('One bounded 25s shortcut budget exhausted')
        q = b if k == count else a+(b-a)*(k/count)
        world.set_qpos_all(q)
        queries += 1
        if world.check_collision():
            cache[key] = False
            return False
    cache[key] = True
    return True

selected = [0]
failure = None
try:
    while selected[-1] < len(path)-1:
        begin = selected[-1]
        end = len(path)-1
        while not edge(path[begin], path[end]):
            if end == begin+1:
                raise RuntimeError(f'Original adjacent edge {begin}:{end} rejected')
            end = begin+(end-begin)//2
        selected.append(end)
    # Same bounded deterministic corner-removal pass as the native helper.
    i = 0
    while i+2 < len(selected):
        if edge(path[selected[i]], path[selected[i+2]]):
            del selected[i+1]
        else:
            i += 1
except (TimeoutError, RuntimeError) as exc:
    failure = str(exc)

result = {
    'scope': 'CPU geometric shortcut candidate only; actual scene retarget, full gap audit, retiming and robot execution still required',
    'source_sha256': sha(__file__), 'raw_plan': str(rawfile), 'raw_plan_sha256': sha(rawfile),
    'snapshot': str(snapshot), 'snapshot_sha256': sha(snapshot), 'rebind_sha256': sha(rebind),
    'input_waypoints': len(path), 'selected_original_indices': selected,
    'shortcut_completed': failure is None, 'failure': failure,
    'edge_attempts': attempts, 'point_queries': queries, 'elapsed_s': time.monotonic()-t0,
    'sample_vertex_displacement_bound_m': .001,
    'same_native_world_predicate': True, 'separate_continuous_gap_certificate': False,
    'inflated_world': False, 'physical_execution': False, 'physical_collision_filters_changed': False,
    'planning_only_frozen_right_overlaps': roles,
    'fixed_left_hand_sdk6': d['open_sdk6'],
    'exact_start_end_preserved': bool(failure is None and np.array_equal(path[selected[0]], path[0]) and np.array_equal(path[selected[-1]], path[-1])),
    'raw_arm6_waypoints': path[selected].tolist() if failure is None else None,
    'fallback_original_raw_plan': str(rawfile),
}
(folder / 'candidate_plan.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ('shortcut_completed','failure','input_waypoints','selected_original_indices','point_queries','elapsed_s')}, indent=2))
