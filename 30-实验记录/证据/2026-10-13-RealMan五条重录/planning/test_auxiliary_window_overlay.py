"""CPU counterexamples and source composition checks; no physical success claim."""
from pathlib import Path
import ast, copy, importlib, json, math, types, unittest
import numpy as np
import auxiliary_joint_window as library
import auxiliary_window_overlay as overlay
import collector_quality_overlay as quality
import randomized_environment_guard as environment

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'source/rgb50_20261007_v1'; LEGACY=ROOT/'source/adjust_bottle_20261002'

def calls(tree,name):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(tree)
            if isinstance(n,ast.Call) and ast.unparse(n.func)==name]

class WindowCounterexamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment_receipt=environment.install_environment_overlay(SOURCE,LEGACY)
        quality.install_planning_quality(source_package=SOURCE,legacy_directory=LEGACY)
        cls.carry_parent=copy.deepcopy(importlib.import_module('full_native_carry_planner_v31').plan_carry_to_task_pose.__rerecord_derivation_ast__)
        cls.limits=np.tile([-3.1,3.1],(6,1));cls.limits[-1]=library.narrowed_numeric_window(.93,-6.28,6.28)
        cls.model=types.SimpleNamespace(get_pinocchio_model=lambda:types.SimpleNamespace(get_joint_limits=lambda:[x[None,:] for x in cls.limits]))
    def test_half_turn_window_is_strict_and_within_vendor(self):
        lo,hi=library.narrowed_numeric_window(.93,-6.28,6.28)
        self.assertLess(hi-lo,2*math.pi);self.assertLessEqual(-6.28,lo);self.assertLessEqual(hi,6.28)
        self.assertLess(lo,.93);self.assertGreater(hi,.93)
    def test_vendor_boundary_uses_intersection_without_clipping_actual(self):
        lo,hi=library.narrowed_numeric_window(6.28,-6.28,6.28)
        self.assertEqual(hi,6.28);self.assertGreater(lo,3.)
        with self.assertRaises(ValueError):library.narrowed_numeric_window(6.3,-6.28,6.28)
    def test_near_goal_passes_and_far_equivalent_goal_is_rejected(self):
        q=np.zeros(6);q[-1]=.6867480564
        self.assertTrue(library.contains_selected_arm(self.model,q,list(range(6))))
        q[-1]-=2*math.pi
        self.assertFalse(library.contains_selected_arm(self.model,q,list(range(6))))
    def test_intermediate_window_excursion_rejected_despite_valid_endpoints(self):
        path=np.zeros((3,6));path[:,5]=[.93,4.4,.68]
        self.assertFalse(library.contains_arm_path(self.model,path,list(range(6))))
        path[1,5]=.8
        self.assertTrue(library.contains_arm_path(self.model,path,list(range(6))))
    def test_nonfinite_goal_and_raw_or_TOPP_rows_fail_closed(self):
        q=np.zeros(6);q[-1]=math.nan
        self.assertFalse(library.contains_selected_arm(self.model,q,list(range(6))))
        self.assertFalse(library.contains_arm_path(self.model,np.array([q,q]),list(range(6))))
        self.assertFalse(library.contains_arm_path(self.model,np.zeros((2,5)),list(range(6))))
    def test_free_keeps_quality_exact_endpoint_and_every_original_safety_call(self):
        for name in quality.SOURCE_PINS:
            parent=importlib.import_module(name).plan_free_approach.__rerecord_derivation_ast__
            child,receipt=overlay.derive_window_ast(parent,'free')
            for target in ['choose_nearest_family','_strict_palm_goal','_quality_path','official_topp_polyline','audit.audit','dense_policy.refine_certificate','_assert_unchanged','equivalent.match_declared_numeric_endpoint']:
                self.assertEqual(calls(parent,target),calls(child,target),target)
            self.assertEqual(receipt['raw_path_window_checks'],1)
            self.assertEqual(receipt['official_TOPP_window_checks'],1)
            code=ast.unparse(child)
            self.assertNotIn('audit.pk =',code);self.assertNotIn('audit.model =',code)
    def test_carry_keeps_environment_registry_attachment_and_vendor_FMA_gates(self):
        parent=self.carry_parent;child,receipt=overlay.derive_window_ast(parent,'carry')
        for target in ['PlanningWorld','world.attach_object','auditor.audit','cma.fma.joint_limit_audit','assert_table_unchanged']:
            self.assertEqual(calls(parent,target),calls(child,target),target)
        registry=[n for n in ast.walk(parent) if isinstance(n,ast.Compare) and 'set(registry)' in ast.unparse(n)]
        self.assertEqual(len(registry),1)
        self.assertIn(ast.unparse(registry[0]),ast.unparse(child))
        code=ast.unparse(child)
        self.assertNotIn('auditor.pk =',code);self.assertNotIn('auditor.model =',code)
        self.assertEqual(receipt['each_cartesian_support_knot_window_checks'],1)
        self.assertEqual(receipt['raw_path_window_checks'],1);self.assertEqual(receipt['official_TOPP_window_checks'],1)
    def test_cartesian_knot_checks_window_before_actual_planning_world_query(self):
        tree,_=overlay.derive_window_ast(self.carry_parent,'carry')
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and isinstance(n.target.elts[0],ast.Name) and n.target.elts[0].id=='knot_index')
        world_index=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='world.set_qpos_all')
        self.assertIn('_window_contains_selected_arm(model, q, indices)',ast.unparse(loop.body[world_index-1]))
        self.assertIn('raise ValueError',ast.unparse(loop.body[world_index-1]))
    def test_stale_or_ambiguous_source_is_rejected(self):
        parent=copy.deepcopy(self.carry_parent)
        parent.body[0].body.insert(0,ast.parse('model=auditor.model').body[0])
        with self.assertRaisesRegex(RuntimeError,'Exactly one original model'):
            overlay.derive_window_ast(parent,'carry')

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WindowCounterexamples))
    receipt={'date':'2026-10-14','passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'scope':'CPU invalid path/goal counterexamples and full original ENV/quality AST composition; not actual physical collection success',
        'source_sha256':{str(p):overlay._digest(p) for p in [Path(__file__),Path(library.__file__),Path(overlay.__file__),Path(quality.__file__),Path(environment.__file__)]},
        'GPU_started':False,'SAPIEN_scene_created':False}
    (ROOT/'planning/auxiliary_window_counterexample_receipt_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
