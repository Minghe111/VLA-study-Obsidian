"""CPU AST/controlled installation tests for the independent RRT variant."""
from pathlib import Path
import ast, copy, hashlib, json, sys, types, unittest
from unittest.mock import patch
import conservative_ompl_overlay as overlay

ROOT=Path(__file__).resolve().parent.parent
PARENTS={
    'full_native_approach_equivalent_goals_v6':'plan_free_approach',
    'full_native_approach_equivalent_goals_dense_v7':'plan_free_approach',
    'full_native_carry_planner_v31':'plan_carry_to_task_pose',
}
def tree(name,function):
    path=Path(__file__).parent/(name+'_'+function+'_numeric_window_generated.py')
    return ast.parse(path.read_text()),path
def call(tree):
    return next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='planner.plan')

class ConservativeCounterexamples(unittest.TestCase):
    def test_only_three_native_kwargs_change_in_all_three_full_parent_ASTs(self):
        for name,function in PARENTS.items():
            parent,path=tree(name,function);child,receipt=overlay.derive_conservative_ast(parent)
            before={k.arg:copy.deepcopy(k.value) for k in call(parent).keywords}
            after={k.arg:k for k in call(child).keywords}
            for key,value in overlay.DEFAULT_POLICY.items():self.assertEqual(ast.literal_eval(after[key].value),value)
            normalized=copy.deepcopy(child)
            for item in call(normalized).keywords:
                if item.arg in overlay.DEFAULT_POLICY:item.value=copy.deepcopy(before[item.arg])
            self.assertEqual(ast.dump(normalized,include_attributes=False),ast.dump(parent,include_attributes=False))
            self.assertEqual(receipt['changed_original_OMPL_calls'],1)
    def test_parent_AST_never_modified(self):
        parent,_=tree('full_native_carry_planner_v31','plan_carry_to_task_pose')
        before=ast.dump(parent,include_attributes=False);overlay.derive_conservative_ast(parent)
        self.assertEqual(ast.dump(parent,include_attributes=False),before)
    def test_nan_or_nonpositive_budget_and_extension_rejected(self):
        parent,_=tree('full_native_carry_planner_v31','plan_carry_to_task_pose')
        for key,value in [('time',float('nan')),('time',0.),('range',-.01),('time',True)]:
            policy=dict(overlay.DEFAULT_POLICY);policy[key]=value
            with self.assertRaises(ValueError):overlay.derive_conservative_ast(parent,policy=policy)
    def test_larger_extensions_or_reenabled_simplification_rejected(self):
        parent,_=tree('full_native_carry_planner_v31','plan_carry_to_task_pose')
        for key,value in [('range',.05),('simplify',True)]:
            policy=dict(overlay.DEFAULT_POLICY);policy[key]=value
            with self.assertRaises(ValueError):overlay.derive_conservative_ast(parent,policy=policy)
    def test_unreviewed_native_call_ambiguity_rejected(self):
        parent,_=tree('full_native_carry_planner_v31','plan_carry_to_task_pose')
        parent.body[0].body.append(ast.Expr(value=copy.deepcopy(call(parent))))
        with self.assertRaises(RuntimeError):overlay.derive_conservative_ast(parent)
    def test_dynamic_native_kwargs_rejected(self):
        parent,_=tree('full_native_carry_planner_v31','plan_carry_to_task_pose')
        call(parent).keywords.append(ast.keyword(arg=None,value=ast.Name(id='extra',ctx=ast.Load())))
        with self.assertRaises(RuntimeError):overlay.derive_conservative_ast(parent)
    def test_installer_composes_window_parent_without_running_planner_or_scene(self):
        modules={};source_hashes={};flags={'initializers_called':0}
        for name,function in PARENTS.items():
            parent,path=tree(name,function);source_hashes[str(path)]=overlay._sha(path)
            namespace={};exec(compile(parent,str(path),'exec'),namespace)
            previous=namespace[function]
            previous.__rerecord_derivation_ast__=parent
            previous.__rerecord_window_overlay__={'retained_actual_window_proof_for_controlled_binding':True}
            module=types.ModuleType(name);setattr(module,function,previous);modules[name]=module
        class Expert:
            def __init__(self,*args,**kwargs):flags['initializers_called']+=1
        expert=types.ModuleType('expert_generation');expert.ExpertSession=Expert;modules['expert_generation']=expert
        destination=ROOT/'planning/conservative_CPU_component_20261014';destination.mkdir(exist_ok=True)
        with patch.dict(sys.modules,modules),patch.object(overlay,'DIRECTORY',destination),patch.object(overlay,'_INSTALLED',None):
            receipt=overlay.install_conservative_ompl_overlay()
            self.assertEqual(len(receipt['functions']),3)
            self.assertEqual(flags['initializers_called'],0)
            for name,function in PARENTS.items():
                installed=getattr(modules[name],function)
                self.assertTrue(installed.__rerecord_window_overlay__)
                self.assertEqual(installed.__rerecord_conservative_ompl_overlay__['effective_native_policy'],overlay.DEFAULT_POLICY)
        self.assertTrue(all(overlay._sha(path)==sha for path,sha in source_hashes.items()))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ConservativeCounterexamples))
    receipt={'date':'2026-10-14','passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),
        'scope':'CPU AST equality/invalid parameter counterexamples and controlled install binding; no native robot planner or physical success claimed',
        'policy':overlay.DEFAULT_POLICY,'no_frozen_parent_or_production_source_overwritten':True,
        'source_sha256':{str(p):overlay._sha(p) for p in [Path(__file__),Path(overlay.__file__)]},
        'parent_sha256':{str(tree(n,f)[1]):overlay._sha(tree(n,f)[1]) for n,f in PARENTS.items()},
        'GPU_started':False,'scene_created':False,'physics_steps':0}
    (ROOT/'planning/conservative_ompl_counterexamples_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
