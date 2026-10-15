"""Independent variant: smaller original RRT extensions, no path simplification.

Install after the existing window overlay and before observer wrappers. Changes
only the three kwargs of each original planner.plan call. It does not resample,
wrap, rewrite, repair or accept a returned path. The original TOPP, complete
FMA, physical and cumulative quality gates remain mandatory.
"""
from __future__ import annotations
import ast, copy, hashlib, importlib, json, math
from pathlib import Path

DIRECTORY=Path(__file__).resolve().parent
DEFAULT_POLICY={'time':30.0,'range':0.01,'simplify':False}
_INSTALLED=None

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _validate(policy):
    if set(policy)!=set(DEFAULT_POLICY):
        raise ValueError('Only native OMPL time/range/simplify policy may change')
    for key in ('time','range'):
        value=policy[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Positive finite native time/range required')
    if policy['range']>0.01 or policy['simplify'] is not False:
        raise ValueError('This variant requires at most .01 rad extension and no simplification')

def derive_conservative_ast(parent,*,policy=None):
    policy=dict(DEFAULT_POLICY if policy is None else policy);_validate(policy)
    tree=copy.deepcopy(parent)
    targets=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='planner.plan']
    if len(targets)!=1:
        raise RuntimeError('Exactly one reviewed original native OMPL call required')
    node=targets[0]
    if any(k.arg is None for k in node.keywords):
        raise RuntimeError('Unreviewed dynamic OMPL kwargs are unsupported')
    keywords={k.arg:k for k in node.keywords}
    if not set(policy).issubset(keywords) or len(keywords)!=len(node.keywords):
        raise RuntimeError('Exact explicit original time/range/simplify kwargs required')
    previous={key:ast.unparse(keywords[key].value) for key in policy}
    for key,value in policy.items():
        keywords[key].value=ast.copy_location(ast.Constant(value=value),keywords[key].value)
    ast.fix_missing_locations(tree)
    return tree,{'parent_AST_sha256':hashlib.sha256(ast.dump(parent,include_attributes=False).encode()).hexdigest(),
        'derived_AST_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'changed_original_OMPL_calls':1,'changed_native_kwargs':list(policy),'original_kwarg_expressions':previous,
        'effective_native_policy':policy,'returned_paths_postprocessed':False,
        'OMPL_validity_resolution_exposed_or_modified':False,
        'original_TOPP_full_FMA_physical_and_quality_acceptance_not_changed':True,
        'smaller_RRT_extensions_are_not_a_continuous_collision_certificate':True}

def install_conservative_ompl_overlay(*,policy=None):
    """Install only in a fresh variant worker, after window and before observers."""
    global _INSTALLED
    if _INSTALLED is not None:
        raise RuntimeError('Conservative variant installs once per fresh worker')
    policy=dict(DEFAULT_POLICY if policy is None else policy);_validate(policy)
    receipt={'overlay_source':str(Path(__file__).resolve()),'overlay_sha256':_sha(__file__),
        'effective_native_policy':policy,'source_original_bytes_modified':False,
        'GPU_scene_or_physics_started_during_install':False,'functions':{}}
    names=[('full_native_approach_equivalent_goals_v6','plan_free_approach'),
           ('full_native_approach_equivalent_goals_dense_v7','plan_free_approach'),
           ('full_native_carry_planner_v31','plan_carry_to_task_pose')]
    for module_name,function_name in names:
        module=importlib.import_module(module_name);previous=getattr(module,function_name)
        if not getattr(previous,'__rerecord_window_overlay__',None):
            raise RuntimeError('A proven independent numeric window must precede this variant')
        parent=getattr(previous,'__rerecord_derivation_ast__',None)
        if parent is None:
            raise RuntimeError('Install before metadata observer wrappers, from the proven parent AST')
        tree,details=derive_conservative_ast(parent,policy=policy)
        file=DIRECTORY/(module_name+'_'+function_name+'_conservative_native_ompl_generated.py')
        file.write_text(ast.unparse(tree)+'\n')
        namespace=dict(previous.__globals__)
        exec(compile(tree,str(file),'exec'),namespace)
        derived=namespace[function_name]
        derived.__rerecord_derivation_ast__=copy.deepcopy(tree)
        derived.__rerecord_window_overlay__=copy.deepcopy(previous.__rerecord_window_overlay__)
        derived.__rerecord_conservative_ompl_overlay__=copy.deepcopy(details)
        setattr(module,function_name,derived)
        details.update(generated_source=str(file),generated_source_sha256=_sha(file),
                       retained_window_proof=copy.deepcopy(previous.__rerecord_window_overlay__))
        receipt['functions'][module_name]=details
    expert=importlib.import_module('expert_generation').ExpertSession
    previous_init=expert.__init__
    def init(session,*args,**kwargs):
        previous_init(session,*args,**kwargs)
        session.sources[str(Path(__file__).resolve())]=receipt['overlay_sha256']
        for item in receipt['functions'].values():
            session.sources[item['generated_source']]=item['generated_source_sha256']
        session.report['conservative_native_ompl_overlay']=copy.deepcopy(receipt)
        session.report['source_hashes']=session.sources
    expert.__init__=init
    _INSTALLED=copy.deepcopy(receipt)
    (DIRECTORY/'conservative_ompl_install_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return copy.deepcopy(receipt)
