"""Zero-step mocked contracts for the new planner, not native physics evidence."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ast
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import full_native_approach_equivalent_goals_v6 as helper

work = Path(__file__).resolve().parent
physical = np.linspace(-.03, .03, 36)
physical[:6] = 0.
limits = np.tile([-3.11, 3.11], (36, 1))
limits[5] = [-6.28, 6.28]
names = ['arm' + str(i) for i in range(6)] + ['frozen' + str(i) for i in range(30)]
original_aux_q = np.linspace(.1, .2, 36)
original_base = np.eye(4)
original_base[:3, 3] = [.7, -.4, .9]


class Pose:
    def __init__(self, matrix=None):
        self.m = np.eye(4) if matrix is None else np.asarray(matrix).copy()
    def to_transformation_matrix(self):
        return self.m.copy()


class Model:
    def __init__(self):
        self.q = original_aux_q.copy()
        self.pose = Pose(original_base)
        self.group = ['original_group']
    def get_qpos(self): return self.q.copy()
    def set_qpos(self, q, full=True): self.q = np.asarray(q).copy()
    def get_base_pose(self): return self.pose
    def set_base_pose(self, p): self.pose = p
    def get_move_group_end_effectors(self): return list(self.group)
    def get_move_group_joint_indices(self): return list(range(6)) if self.group != ['original_group'] else [7, 8]
    def set_move_group(self, g): self.group = [g] if isinstance(g, str) else list(g)


class PK:
    def get_link_names(self): return ['r_inspire_base_link'] + ['native_link' + str(i) for i in range(92)]
    def compute_IK_CLIK(self, index, target, seed, mask, **kwargs):
        q = physical.copy()
        q[:3] = target.m[:3, 3]
        q[5] = 5.8
        if mode == 'finite_goal_boundary': q[5] = .0030 if calls['IK_count'] == 0 else .0034
        calls['IK_count'] += 1
        if mode == 'frozen_IK_changed': q[8] += .01
        return q, True, None
    def compute_forward_kinematics(self, q): self.q = np.asarray(q).copy()
    def get_link_pose(self, index):
        matrix = np.eye(4)
        angle = float(self.q[5])
        matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        matrix[:3, 3] = self.q[:3] + index * .001
        if mode == 'FK_not_equivalent' and index == 45:
            matrix[2, 3] += self.q[5] * .01
        return Pose(matrix)


class Auditor:
    def __init__(self, task):
        self.model = Model()
        created.append(self.model)
        self.pk, self.Pose, self.initial_sim_q, self.sim_joints = PK(), Pose, physical.copy(), names
        self.allowed = {frozenset(('nativeA', 'nativeB'))}
        self.art = SimpleNamespace(active_joints=[SimpleNamespace(limits=row.reshape(1, 2)) for row in limits])
        self.urdf_tree = ET.Element('robot')
        for name in names: ET.SubElement(self.urdf_tree, 'joint', name=name, type='revolute')
        shape = SimpleNamespace(get_collision_geometry=lambda: 'actual_native_shape')
        self.static = {n: SimpleNamespace(shapes=[shape], pose=Pose(), shape_poses=[np.eye(4)]) for n in ('table', 'bottle')}
        self.fcl = SimpleNamespace(CollisionObject=lambda g: g, FCLObject=lambda *args: args)
    def audit(self, vertices, side, **kwargs):
        calls['FMA'].append(kwargs)
        return {'passed': mode != 'audit_reject'}


class World:
    def __init__(self, models, obstacles):
        calls['obstacles'].append(obstacles)
        if mode == 'world_exception': raise ValueError('World constructor exception fixture')
    def get_allowed_collision_matrix(self): return SimpleNamespace(set_entry=lambda *args: calls['ACM'].append(args))
    def set_qpos_all(self, q): self.q = np.asarray(q).copy()
    def check_collision(self):
        if mode == 'start_collision': return ['native_collision']
        if mode == 'endpoint_collision' and np.max(abs(self.q)) > .001: return ['native_collision']
        return []


class Planner:
    def __init__(self, world): pass
    def plan(self, start, goals, **kwargs):
        calls['RRT'].append({'goals': np.asarray(goals).copy(), **kwargs})
        endpoint = np.asarray(goals[-1] if mode == 'finite_goal_boundary' else goals[0]).copy()
        path = np.vstack([start, (start + endpoint) / 2., endpoint])
        if mode in ('unmatched_endpoint', 'FK_not_equivalent'): path[-1, 5] -= helper.equivalent.TWO_PI
        if mode == 'wrong_start': path[0, 0] += .001
        if mode == 'raw_limit_violation': path[1, 0] = 4.
        if mode == 'nonfinite_path': path[1, 0] = np.nan
        if mode == 'approximate_status' or (mode == 'finite_goal_boundary' and len(goals) == 1): return 'Approximate solution', path
        return 'Exact solution', path


def limit_gate(full, native_limits, joint_names, indices):
    commanded = np.asarray(full)[1:, indices]
    return {'passed': bool(np.all(commanded >= native_limits[indices, 0] - 1e-5)
                           and np.all(commanded <= native_limits[indices, 1] + 1e-5))}


def topp(task, path, side, *, time_step):
    calls['TOPP'].append(np.asarray(path).copy())
    position = np.asarray(path).copy()
    if mode == 'retimed_limit_violation': position[1, 0] = 4.
    return position, np.zeros_like(position), np.arange(len(position)) * time_step, {'audit_vertices': position.tolist()}


fma = ModuleType('full_mesh_path_audit')
fma.FullMeshPathAuditor, fma.joint_limit_audit = Auditor, limit_gate
fma._pose_matrix = lambda p: p.to_transformation_matrix()
mplib = ModuleType('mplib'); mplib.PlanningWorld = World
ompl = ModuleType('mplib.pymp.planning.ompl'); ompl.OMPLPlanner = Planner
carry = ModuleType('full_native_carry_planner_v31'); carry.official_topp_polyline = topp
for name, module in [('full_mesh_path_audit', fma), ('mplib', mplib), ('mplib.pymp.planning.ompl', ompl), ('full_native_carry_planner_v31', carry)]:
    sys.modules[name] = module
entity = SimpleNamespace(get_qpos=lambda: physical.copy(), get_qvel=lambda: np.zeros(36), root_pose=Pose())
goal = np.eye(4); goal[:3, 3] = [.1, .2, .3]
robot = SimpleNamespace(left_entity=entity, _hand_joints={'hand': SimpleNamespace(get_drive_target=lambda: np.array([.2]))},
                        right_arm_joints_name=names[:6], _trans_from_gripper_to_endlink=lambda pose, arm_tag: Pose(goal))
task = SimpleNamespace(robot=robot, bottle=SimpleNamespace(get_pose=lambda: Pose()),
                       scene=SimpleNamespace(step=lambda: (_ for _ in ()).throw(AssertionError('No physics step allowed'))))
results = []
for mode in ('valid_predeclared_alternate', 'unmatched_endpoint', 'wrong_start', 'raw_limit_violation',
             'retimed_limit_violation', 'audit_reject', 'nonfinite_path', 'approximate_status',
             'endpoint_collision', 'start_collision', 'frozen_IK_changed', 'FK_not_equivalent', 'world_exception', 'finite_goal_boundary'):
    created = []
    calls = {name: [] for name in ('RRT', 'TOPP', 'FMA', 'ACM', 'obstacles')}
    calls['IK_count'] = 0
    snapshot = physical.copy()
    try:
        result = helper.plan_free_approach(task, [0] * 7, 'right', random_seed=100025, ik_attempts=2)
        assert mode != 'world_exception'
        if mode in ('valid_predeclared_alternate', 'finite_goal_boundary'):
            assert result['status'] == 'Success' and len(calls['TOPP']) == len(calls['FMA']) == 1
            assert len(calls['RRT'][-1]['goals']) == 2
            assert calls['FMA'][0] == {'min_bottle_clearance_m': .0005, 'min_self_clearance_m': 0., 'min_table_clearance_m': 0.}
            row = result['planning_metadata']['RRT_attempts'][-1]
            if mode == 'finite_goal_boundary':
                assert result['planning_metadata']['native_free_goal_family_count'] == 2
                assert len(calls['RRT']) == 2
                assert result['planning_metadata']['endpoint_attempts'][1]['predeclared_numeric_goals_already_covered'] == [True, False]
                assert row['numeric_endpoint_match']['matched_declared_goal_index'] == 1
            assert row['numeric_endpoint_match']['passed']
            assert np.array_equal(np.asarray(row['raw_path_arm_rad']), calls['TOPP'][0])
            assert len(row['raw_path_full_qpos_rad'][0]) == 36
        else:
            assert result['status'] == 'Fail'
        if mode in ('endpoint_collision', 'start_collision', 'frozen_IK_changed'): assert not calls['RRT']
        if mode in ('unmatched_endpoint', 'wrong_start', 'raw_limit_violation', 'nonfinite_path', 'approximate_status', 'FK_not_equivalent'): assert not calls['TOPP']
        if mode == 'retimed_limit_violation': assert not calls['FMA']
        assert result['planning_metadata']['physical_state_unchanged']
        results.append({'case': mode, 'passed': True, 'RRT_calls': len(calls['RRT']), 'TOPP_calls': len(calls['TOPP']), 'FMA_calls': len(calls['FMA'])})
    except ValueError as error:
        assert mode == 'world_exception' and 'World constructor exception fixture' in str(error)
        results.append({'case': mode, 'passed': True, 'auxiliary_restored_despite_construction_exception': True})
    model = created[-1]
    assert np.array_equal(model.q, original_aux_q) and np.array_equal(model.pose.m, original_base)
    assert model.group == ['original_group'] and np.array_equal(physical, snapshot)
source = Path(helper.__file__)
tree = ast.parse(source.read_text(encoding='utf-8'))
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        assert node.func.attr not in ('step', 'set_pose', 'set_qvel', 'set_drive_target', 'set_drive_velocity_target')
        if node.func.attr in ('set_qpos', 'set_base_pose', 'set_move_group'):
            assert ast.unparse(node.func.value) == 'model'
overlay = work / 'expert_generation_v9_equivalent_goals_v6.py'
overlay_tree = ast.parse(overlay.read_text(encoding='utf-8'))
assert not any('v11' in ast.unparse(n) for n in ast.walk(overlay_tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
assert [n.name for n in ast.walk(overlay_tree) if isinstance(n, ast.ClassDef)] == ['ExpertSession']
session_class = next(n for n in overlay_tree.body if isinstance(n, ast.ClassDef))
assert [n.name for n in session_class.body if isinstance(n, ast.FunctionDef)] == ['__init__']
output = {'passed': True, 'scope': 'Zero-step mocked planner contracts only; not actual native model, TOPP, FCL or physics success',
          'new_physics_steps': 0, 'physical_validation_completed': False, 'cases': results,
          'all_auxiliary_q_base_group_restored': True, 'returned_paths_never_wrapped_or_clipped': True,
          'v9_physics_methods_inherited_no_v11_or_waypoint_planning': True,
          'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source, Path(helper.equivalent.__file__), overlay, Path(__file__))}}
with (work / 'equivalent_native_planner_contract_receipt_v6.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps(output, indent=2), flush=True)
