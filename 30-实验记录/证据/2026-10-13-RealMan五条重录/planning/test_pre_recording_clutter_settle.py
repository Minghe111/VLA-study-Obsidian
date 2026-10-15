"""CPU counterexamples using real guard/extraction code and mock native APIs."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
FMA_PATH = HERE.parent / 'source/adjust_bottle_20261002/full_mesh_path_audit.py'
sys.path.insert(0, str(HERE))
import pre_recording_clutter_settle as settle
import randomized_environment_guard as guard


def extracted_native_geometry():
    """Unmodified original geometry functions, with their real source identity."""
    tree = ast.parse(FMA_PATH.read_text())
    names = {'_array', '_pose_matrix', '_link_vertices'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    module = types.ModuleType('full_mesh_path_audit')
    module.__file__ = str(FMA_PATH)
    module.np = np
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FMA_PATH), 'exec'), module.__dict__)
    return module


class Pose:
    def __init__(self, matrix=None):
        self.matrix = np.eye(4) if matrix is None else np.asarray(matrix).copy()
        self.p = self.matrix[:3, 3]

    def to_transformation_matrix(self):
        return self.matrix.copy()


class NativeBox:
    def __init__(self, size=.1):
        self.half_size = np.asarray([size] * 3)
        self.local_pose = Pose()


class Rigid:
    def __init__(self, entity, size=.1):
        self.entity = entity
        self.collision_shapes = [NativeBox(size)]


class Entity:
    def __init__(self, name, matrix=None, size=.1):
        self.name = name
        self.matrix = np.eye(4) if matrix is None else np.asarray(matrix).copy()
        self.component = Rigid(self, size)

    @property
    def pose(self):
        return Pose(self.matrix)

    def get_pose(self):
        return self.pose

    def get_components(self):
        return [self.component]


class Art:
    def __init__(self, links):
        self.links = [types.SimpleNamespace(entity=entity) for entity in links]

    def get_links(self):
        return self.links

    def get_qpos(self):
        return np.zeros(36)

    def find_link_by_name(self, name):
        m = np.eye(4)
        m[:3, 3] = [-.4 if name.startswith('l_') else .4, 0, .8]
        return Entity(name, m)


class Scene:
    def __init__(self, callback=None):
        self.steps = 0
        self.callback = callback
        self.dt = .004
        self.contacts = []

    def get_timestep(self):
        return self.dt

    def step(self):
        self.steps += 1
        if self.callback:
            self.callback(self.steps)

    def get_contacts(self):
        return self.contacts


def fixture(callback=None, articulated=False):
    clutter = [Entity('clutter_%d' % n) for n in range(10)]
    child = Entity('moving_child')
    references = clutter.copy()
    if articulated:
        references[0] = Art([clutter[0], child])
    bottle_matrix = np.eye(4)
    bottle_matrix[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    bottle_matrix[:3, 3] = [-.2, 0, .8]
    bottle = Entity('bottle', bottle_matrix, size=.03)
    robot = types.SimpleNamespace(communication_flag=False,
                                  left_entity=Art([Entity('robot')]), right_entity=None)
    task = types.SimpleNamespace(_rerecord_clutter_entities=references,
        record_cluttered_objects=[{'object_type': e.name, 'object_index': n} for n, e in enumerate(clutter)],
        bottle=types.SimpleNamespace(actor=bottle, get_pose=bottle.get_pose), robot=robot, scene=Scene(callback))
    return task, clutter, child


class Counterexamples(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.modules = patch.dict(sys.modules, {
            'sapien.physx': types.SimpleNamespace(PhysxRigidBaseComponent=Rigid),
            'full_mesh_path_audit': extracted_native_geometry()})
        self.modules.start()
        self.spec = {'arm': 'left', 'pose_class': 'upright'}

    def tearDown(self):
        self.modules.stop()
        self.tmp.cleanup()

    def run_prepare(self, task, scene=None, **kwargs):
        return settle.settle_clutter(task, self.out, {} if scene is None else scene,
                                    self.spec, max_extra_s=.08, window_s=.02, **kwargs)

    def test_translation_that_official_rotation_check_misses_is_rejected(self):
        task, clutter, _ = fixture()
        task.scene.callback = lambda step: clutter[3].matrix.__setitem__((0, 3), .00006 * step)
        with self.assertRaisesRegex(RuntimeError, 'did not stabilize'):
            self.run_prepare(task)
        rows = [json.loads(s) for s in (self.out / 'pre_recording_clutter_settle_steps.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 20)
        self.assertGreater(rows[-1]['max_vertex_displacement_from_preparation_start_m'], .001)
        self.assertEqual(rows[-1]['worst_component_from_preparation_start'], 'clutter_03_00')
        self.assertTrue(all(np.array_equal(clutter[3].matrix[:3, :3], np.eye(3)) for _ in rows))

    def test_gravity_preparation_changes_real_t0_and_preserves_prior_scene(self):
        task, clutter, _ = fixture()
        old_scene = {'initial_pose': task.bottle.get_pose().matrix.tolist(), 'old_metadata': 'keep'}
        old_pose = np.asarray(old_scene['initial_pose'])
        historical = {'source_hashes': {}, 'native_wake': {'requested': False, 'wake_calls': 0}}
        task._rgb50_initial_extra_settle_receipt = historical.copy()
        def motion(step):
            clutter[3].matrix[0, 3] = .0004 * min(step, 5)
            task.bottle.actor.matrix[0, 3] = -.2 + .00002 * step
        task.scene.callback = motion
        receipt = self.run_prepare(task, old_scene)
        self.assertEqual(task.scene.steps, 10)
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['completed_extra_physics_steps'], 10)
        self.assertEqual(receipt['wake_calls'], 0)
        self.assertTrue(np.array_equal(old_scene['pre_clutter_settle_scene_sample']['initial_pose'], old_pose))
        self.assertFalse(np.array_equal(old_scene['initial_pose'], old_pose))
        self.assertEqual(task._rgb50_initial_extra_settle_receipt, historical)
        self.assertEqual(settle.validate_before_expert(task, self.spec), receipt)

    def test_rotation_with_static_center_checks_collision_vertices(self):
        task, clutter, _ = fixture()
        clutter[2].component.collision_shapes[0].half_size = np.asarray([1., .1, .1])
        def motion(step):
            angle = step * .001
            c, s = np.cos(angle), np.sin(angle)
            clutter[2].matrix[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        task.scene.callback = motion
        with self.assertRaisesRegex(RuntimeError, 'did not stabilize'):
            self.run_prepare(task)
        rows = [json.loads(s) for s in (self.out / 'pre_recording_clutter_settle_steps.jsonl').read_text().splitlines()]
        self.assertGreater(rows[0]['max_vertex_displacement_from_previous_m'], .001)
        self.assertTrue(np.array_equal(clutter[2].matrix[:3, 3], np.zeros(3)))
        self.assertLess(np.degrees(.001 * task.scene.steps), 3)

    def test_articulation_child_motion_is_covered_with_static_root(self):
        task, clutter, child = fixture(articulated=True)
        task.scene.callback = lambda step: child.matrix.__setitem__((2, 3), .0001 * step)
        with self.assertRaisesRegex(RuntimeError, 'did not stabilize'):
            self.run_prepare(task)
        receipt = json.loads((self.out / 'pre_recording_clutter_settle_receipt.json').read_text())
        self.assertEqual(receipt['coverage']['covered_collision_component_count'], 11)
        self.assertEqual(receipt['last_step']['worst_component_from_preparation_start'], 'clutter_00_01')
        self.assertTrue(np.array_equal(clutter[0].matrix, np.eye(4)))

    def test_active_contact_is_rejected_on_its_exact_native_step(self):
        task, clutter, _ = fixture()
        contact = types.SimpleNamespace(bodies=[clutter[1].component, task.bottle.actor.component],
            points=[types.SimpleNamespace(impulse=[0, 0, 1e-4], normal=[0, 0, 1], separation=0.)])
        task.scene.contacts = [contact]
        with self.assertRaisesRegex(RuntimeError, 'active robot or target'):
            self.run_prepare(task)
        self.assertEqual(task.scene.steps, 1)

    def test_back_and_forth_motion_cannot_alias_to_stable_endpoints(self):
        task, clutter, _ = fixture()
        task.scene.callback = lambda step: clutter[1].matrix.__setitem__((0, 3), .00005 * (-1) ** step)
        with self.assertRaisesRegex(RuntimeError, 'did not stabilize'):
            self.run_prepare(task)
        receipt = json.loads((self.out / 'pre_recording_clutter_settle_receipt.json').read_text())
        self.assertGreater(receipt['last_step']['window_pairwise_vertex_displacement_upper_bound_m'], .0001)

    def test_geometry_change_cannot_be_relabelled_as_stable(self):
        task, clutter, _ = fixture()
        task.scene.callback = lambda step: clutter[0].component.collision_shapes.append(NativeBox()) if step == 5 else None
        with self.assertRaisesRegex(RuntimeError, 'identity or geometry changed'):
            self.run_prepare(task)

    def test_protected_source_change_during_preparation_is_rejected(self):
        task, _, _ = fixture()
        source = self.out / 'protected.py'
        source.write_text('original bytes')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        task.scene.callback = lambda step: source.write_text('changed bytes') if step == 5 else None
        with self.assertRaisesRegex(RuntimeError, 'source bytes differ'):
            self.run_prepare(task, source_hashes={str(source): digest})

    def test_final_bottle_class_rechecked_without_pose_restore(self):
        task, _, _ = fixture()
        task.scene.callback = lambda step: task.bottle.actor.matrix.__setitem__((slice(0, 3), slice(0, 3)), np.eye(3)) if step == 5 else None
        with self.assertRaisesRegex(RuntimeError, 'orientation class differs'):
            self.run_prepare(task)
        self.assertTrue(np.array_equal(task.bottle.actor.matrix[:3, :3], np.eye(3)))

    def test_new_t0_motion_before_expert_is_rejected(self):
        task, clutter, _ = fixture()
        self.run_prepare(task)
        clutter[0].matrix[0, 3] += 1e-6
        with self.assertRaisesRegex(RuntimeError, 'clutter t0 differs'):
            settle.validate_before_expert(task, self.spec)

    def test_unrequested_clean_or_incomplete_clutter_is_rejected(self):
        task, _, _ = fixture()
        task._rerecord_clutter_entities.pop()
        task.record_cluttered_objects.pop()
        with self.assertRaisesRegex(RuntimeError, 'all ten'):
            self.run_prepare(task)
        self.assertEqual(task.scene.steps, 0)

    def test_late_baseline_wrong_clock_and_real_io_fail_before_steps(self):
        for mode in ('clock', 'io', 'late'):
            with self.subTest(mode=mode):
                task, _, _ = fixture()
                if mode == 'clock': task.scene.dt = .005
                if mode == 'io': task.robot.communication_flag = True
                if mode == 'late': task._rerecord_environment_guard = {}
                with self.assertRaises((RuntimeError, ValueError)):
                    self.run_prepare(task)
                self.assertEqual(task.scene.steps, 0)


if __name__ == '__main__':
    unittest.main()
