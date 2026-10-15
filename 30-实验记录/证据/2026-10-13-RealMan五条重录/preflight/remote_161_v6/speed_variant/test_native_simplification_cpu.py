"""CPU component checks; no SAPIEN scene, renderer, GPU or physical acceptance."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE.parents[2]
PLANNING = E / 'planning'
PARENTS = {
    'full_native_approach_equivalent_goals_v6': 'plan_free_approach',
    'full_native_approach_equivalent_goals_dense_v7': 'plan_free_approach',
    'full_native_carry_planner_v31': 'plan_carry_to_task_pose',
}

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

server = load('server_simplification_candidate', HERE / 'conservative_ompl_overlay.py')
frozen = load('frozen_conservative_candidate', PLANNING / 'conservative_ompl_overlay.py')
quality = load('frozen_motion_policy', PLANNING / 'near_branch_quality.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def parent(name, function):
    path = PLANNING / (name + '_' + function + '_numeric_window_generated.py')
    return ast.parse(path.read_text()), path

def native_call(tree):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == 'planner.plan')

MEASUREMENT = {}

class SpeedVariantTests(unittest.TestCase):
    def test_only_simplify_differs_from_current_local_full_AST(self):
        for name, function in PARENTS.items():
            original, path = parent(name, function)
            local, _ = frozen.derive_conservative_ast(original)
            remote, receipt = server.derive_conservative_ast(original)
            kwargs = {k.arg: k for k in native_call(remote).keywords}
            self.assertIs(ast.literal_eval(kwargs['simplify'].value), True)
            self.assertEqual(ast.literal_eval(kwargs['time'].value), 30.)
            self.assertEqual(ast.literal_eval(kwargs['range'].value), .01)
            kwargs['simplify'].value = ast.Constant(value=False)
            self.assertEqual(ast.dump(remote, include_attributes=False),
                             ast.dump(local, include_attributes=False))
            self.assertTrue(receipt['original_TOPP_full_FMA_physical_and_quality_acceptance_not_changed'])

    def test_original_parent_not_modified(self):
        for name, function in PARENTS.items():
            original, path = parent(name, function)
            before = ast.dump(original, include_attributes=False)
            source_sha = sha(path)
            server.derive_conservative_ast(original)
            self.assertEqual(ast.dump(original, include_attributes=False), before)
            self.assertEqual(sha(path), source_sha)

    def test_invalid_native_policy_rejected(self):
        original, _ = parent('full_native_carry_planner_v31', 'plan_carry_to_task_pose')
        for key, value in [('time', float('nan')), ('time', 0.), ('range', .05),
                           ('time', True), ('simplify', False), ('simplify', 1)]:
            policy = dict(server.DEFAULT_POLICY)
            policy[key] = value
            with self.assertRaises(ValueError):
                server.derive_conservative_ast(original, policy=policy)

    def test_unreviewed_dynamic_or_multiple_call_rejected(self):
        original, _ = parent('full_native_carry_planner_v31', 'plan_carry_to_task_pose')
        dynamic = copy.deepcopy(original)
        native_call(dynamic).keywords.append(ast.keyword(arg=None, value=ast.Name(id='extra', ctx=ast.Load())))
        with self.assertRaises(RuntimeError):
            server.derive_conservative_ast(dynamic)
        ambiguous = copy.deepcopy(original)
        ambiguous.body[0].body.append(ast.Expr(value=copy.deepcopy(native_call(ambiguous))))
        with self.assertRaises(RuntimeError):
            server.derive_conservative_ast(ambiguous)

    def test_install_retains_full_numeric_window_proofs_and_does_not_start_scene(self):
        modules = {}
        flags = {'init': 0}
        for name, function in PARENTS.items():
            tree, path = parent(name, function)
            namespace = {}
            exec(compile(tree, str(path), 'exec'), namespace)
            previous = namespace[function]
            previous.__rerecord_derivation_ast__ = tree
            previous.__rerecord_window_overlay__ = {'window_identity': name}
            module = types.ModuleType(name)
            setattr(module, function, previous)
            modules[name] = module
        class Expert:
            def __init__(self):
                flags['init'] += 1
                self.sources = {}
                self.report = {}
        expert = types.ModuleType('expert_generation')
        expert.ExpertSession = Expert
        modules['expert_generation'] = expert
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.object(server, 'DIRECTORY', Path(tmp)), patch.object(server, '_INSTALLED', None):
            receipt = server.install_conservative_ompl_overlay()
            self.assertEqual(flags['init'], 0)
            self.assertEqual(len(receipt['functions']), 3)
            self.assertTrue(receipt['native_simplification_is_not_a_collision_certificate'])
            self.assertTrue(receipt['physical_speedup_pending_remote_validation'])
            for name, function in PARENTS.items():
                installed = getattr(modules[name], function)
                self.assertEqual(installed.__rerecord_window_overlay__, {'window_identity': name})
                self.assertIs(installed.__rerecord_conservative_ompl_overlay__['effective_native_policy']['simplify'], True)
            session = Expert()
            self.assertEqual(flags['init'], 1)
            # Initializer records the producer only; no hardware is constructed.

    def test_missing_window_proof_rejected_before_plan(self):
        module = types.ModuleType('full_native_approach_equivalent_goals_v6')
        module.plan_free_approach = lambda: None
        with patch.dict(sys.modules, {module.__name__: module}), patch.object(server, '_INSTALLED', None):
            with self.assertRaises(RuntimeError):
                server.install_conservative_ompl_overlay()

    def test_unchanged_gate_rejects_wrong_periodic_endpoint(self):
        start = np.zeros(6)
        goal = np.zeros(6); goal[-1] = .2
        other = goal.copy(); other[-1] += 2 * math.pi
        path = np.stack([start, other])
        before = path.copy()
        result = quality.path_quality(path, start, goal)
        self.assertFalse(result['passed'])
        self.assertIn('Path endpoint differs from the specifically selected near raw goal', result['rejection_reasons'])
        np.testing.assert_array_equal(path, before)

    def test_unchanged_gate_rejects_excessive_dense_speed_and_hardware_excursion(self):
        start = np.zeros(6); goal = np.full(6, .5)
        result = quality.path_quality(np.stack([start, goal]), start, goal,
                                     times_s=[0., .004], velocity_limits_rad_s=np.ones(6),
                                     hardware_limits=np.tile([-.4, .4], (6, 1)))
        self.assertFalse(result['passed'])
        self.assertIn('Dense command step exceeds original velocity-times-dt budget', result['rejection_reasons'])
        self.assertIn('Original commanded/measured hardware limit gate failed', result['rejection_reasons'])

    def test_actual_native_free_space_simplification_and_frozen_segmented_TOPP(self):
        import mplib
        urdf = '''<robot name="tiny"><link name="base"/><link name="tip"/>
        <joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
        <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>'''
        srdf = '<robot name="tiny"><disable_collisions link1="base" link2="tip" reason="Adjacent"/></robot>'
        with tempfile.TemporaryDirectory() as tmp:
            u = Path(tmp) / 'tiny.urdf'; s = u.with_suffix('.srdf')
            u.write_text(urdf); s.write_text(srdf)
            native = mplib.Planner(str(u), srdf=str(s), move_group='tip',
                                   joint_vel_limits=np.array([1.]), joint_acc_limits=np.array([1.]))
            status, raw = native.planner.plan(np.array([0.]), [np.array([.7])],
                                              time=1., range=.01, simplify=False)
            self.assertEqual(str(status), 'Exact solution')
            raw = np.asarray(raw).copy(); saved = raw.copy()
            simple = np.asarray(native.planner.simplify_path(raw.copy()))
            status_new, newly_planned = native.planner.plan(np.array([0.]), [np.array([.7])],
                                                           time=1., range=.01, simplify=True)
            self.assertEqual(str(status_new), 'Exact solution')
            self.assertLess(len(simple), len(raw))
            np.testing.assert_allclose(simple[[0, -1]], raw[[0, -1]], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(raw, saved)
            source = E / 'source/adjust_bottle_20261002/full_native_carry_planner_v31.py'
            tree = ast.parse(source.read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'official_topp_polyline')
            only = ast.Module(body=[function], type_ignores=[])
            namespace = {'np': np, 'math': math}
            exec(compile(only, str(source), 'exec'), namespace)
            clock = types.SimpleNamespace(get_timestep=lambda: .004)
            wrapper = types.SimpleNamespace(planner=native, TOPP=native.TOPP)
            task = types.SimpleNamespace(scene=clock, robot=types.SimpleNamespace(left_mplib_planner=wrapper))
            original_topp = namespace['official_topp_polyline']
            raw_pos, _, _, old = original_topp(task, raw)
            simple_pos, _, _, new = original_topp(task, simple)
            self.assertLess(new['dense_duration_s'], old['dense_duration_s'])
            self.assertLess(new['dense_rows'], old['dense_rows'])
            np.testing.assert_allclose(simple_pos[[0, -1]], raw_pos[[0, -1]], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(raw, saved)
            MEASUREMENT.update(raw_nodes=len(raw), simplified_same_path_nodes=len(simple),
                               newly_planned_simplify_true_nodes=len(newly_planned),
                               raw_segment_TOPP_physical_duration_s=old['dense_duration_s'],
                               simplified_segment_TOPP_physical_duration_s=new['dense_duration_s'],
                               exact_same_path_endpoints_preserved=True,
                               only_synthetic_one_prismatic_free_world=True,
                               actual_robot_or_randomized_scene_test=False,
                               mplib_version=getattr(mplib, '__version__', None),
                               official_TOPP_source_sha256=sha(Path(mplib.__file__).parent / 'planner.py'))

if __name__ == '__main__':
    inputs = [PLANNING / 'conservative_ompl_overlay.py', PLANNING / 'near_branch_quality.py',
              E / 'source/adjust_bottle_20261002/full_native_carry_planner_v31.py']
    inputs += [parent(name, fn)[1] for name, fn in PARENTS.items()]
    before = {str(p): sha(p) for p in inputs}
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SpeedVariantTests))
    unchanged = before == {str(p): sha(p) for p in inputs}
    receipt = {'schema': 'native-simplification-CPU-component-v1', 'date': '2026-10-14',
               'passed': result.wasSuccessful() and unchanged, 'tests_run': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors),
               'source_sha256': {str(p): sha(p) for p in [Path(__file__), Path(server.__file__)]},
               'frozen_input_sha256': before, 'frozen_inputs_unchanged': unchanged,
               'measurement': MEASUREMENT, 'scene_created': False, 'GPU_started': False,
               'physical_speedup_verified': False, 'component_only': True,
               'scope': 'Full parent AST equality except native simplify, rejection counterexamples, actual native CPU free-space simplification and frozen original TOPP; no physical robot, FCL final audit or success claim'}
    (HERE / 'cpu_validation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(0 if receipt['passed'] else 1)
