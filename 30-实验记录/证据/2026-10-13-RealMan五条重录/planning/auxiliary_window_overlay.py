"""Compose finite numeric-window planning with existing quality/environment ASTs.

Install after environment and quality installers, before observer wrappers.
Only independent planning models change; original auditor models stay intact.
"""
from __future__ import annotations
import ast, copy, hashlib, importlib, json
from pathlib import Path
from auxiliary_joint_window import create_model, contains_selected_arm, contains_arm_path

DIRECTORY=Path(__file__).resolve().parent
_INSTALLED=None

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def derive_window_ast(tree, kind):
    tree=copy.deepcopy(tree)
    if kind not in ('free','carry'):
        raise ValueError('Explicit free/carry source scope required')
    audit_name='audit' if kind=='free' else 'auditor'
    metadata_name='metadata' if kind=='free' else 'meta'
    models=0; pks=0; metadata_sites=0
    class Window(ast.NodeTransformer):
        def visit_Attribute(self,node):
            nonlocal pks
            self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id==audit_name and node.attr=='pk':
                pks+=1
                return ast.copy_location(ast.Name(id='_window_planning_pk',ctx=node.ctx),node)
            return node
        def visit_Assign(self,node):
            nonlocal models,metadata_sites
            self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='model' and isinstance(node.value,ast.Attribute) and isinstance(node.value.value,ast.Name) and node.value.value.id==audit_name and node.value.attr=='model':
                models+=1
                return ast.parse(f'model, _numeric_window_receipt = _window_create_model({audit_name}, side)\n_window_planning_pk = model.get_pinocchio_model()').body
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==metadata_name and isinstance(node.value,ast.Dict):
                metadata_sites+=1
                node.value.keys.append(ast.Constant('auxiliary_numeric_joint_window'))
                node.value.values.append(ast.Name(id='_numeric_window_receipt',ctx=ast.Load()))
            return node
    Window().visit(tree)
    if models!=1 or pks<1 or metadata_sites!=1:
        raise RuntimeError(f'Exactly one original model/meta and original planning PK sites required: {models}/{metadata_sites}/{pks}')
    goal_sites=0
    if kind=='free':
        for node in ast.walk(tree):
            if isinstance(node,ast.For) and ast.unparse(node.target)=='(declared_index, candidate)' and ast.unparse(node.iter)=='enumerate(declared)':
                goal_sites+=1
                node.body[:0]=ast.parse("if not _window_contains_selected_arm(model, candidate, indices):\n    row['declared_goal_collision_checks'].append({'declared_goal_index': declared_index, 'full_qpos_rad': candidate.tolist(), 'rejection_reason': 'Outside independent near numeric planning window; physical vendor goal is not modified'})\n    continue").body
    else:
        candidates=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='inspect_candidate']
        if len(candidates)!=1:
            raise RuntimeError('One original exact candidate validator required')
        function=candidates[0];body=[]
        for node in function.body:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='world.set_qpos_all':
                goal_sites+=1
                body.extend(ast.parse("if not _window_contains_selected_arm(model, q, indices):\n    entry['reason'] = 'Outside independent near numeric planning window; physical vendor goal is not modified'\n    endpoint_reports.append(entry)\n    return").body)
            body.append(node)
        function.body=body
    if goal_sites!=1:
        raise RuntimeError('One reviewed numeric candidate filter insertion required')
    raw_sites=0; retimed_sites=0; cartesian_sites=0
    class Paths(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal raw_sites,retimed_sites
            self.generic_visit(node)
            if kind=='free' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_quality_path' and isinstance(node.targets[0],ast.Tuple) and isinstance(node.targets[0].elts[0],ast.Name) and node.targets[0].elts[0].id=='raw_quality':
                raw_sites+=1
                return ast.parse("if not _window_contains_arm_path(model, path, indices):\n    row['rejection_reason'] = 'Raw path exits independent numeric planning window'\n    continue").body+[node]
            if kind=='free' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='official_topp_polyline':
                retimed_sites+=1
                return [node]+ast.parse("if not _window_contains_arm_path(model, position, indices):\n    row['rejection_reason'] = 'Official TOPP path exits independent numeric planning window'\n    continue").body
            if kind=='carry' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('command_full','retimed_full'):
                target=node.targets[0].id; row='path' if target=='command_full' else 'position'
                if target=='command_full':raw_sites+=1
                else:retimed_sites+=1
                return ast.parse(f"if not _window_contains_arm_path(model, {row}, indices):\n    detail['reason'] = 'Raw or official TOPP carry path exits independent numeric planning window'\n    continue").body+[node]
            return node
        def visit_For(self,node):
            nonlocal cartesian_sites
            self.generic_visit(node)
            if kind=='carry' and isinstance(node.target,ast.Tuple) and len(node.target.elts)==2 and isinstance(node.target.elts[0],ast.Name) and node.target.elts[0].id=='knot_index':
                body=[]
                for statement in node.body:
                    if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func)=='world.set_qpos_all':
                        cartesian_sites+=1
                        body.extend(ast.parse("if not _window_contains_selected_arm(model, q, indices):\n    raise ValueError('Cartesian support knot exits independent numeric planning window; never wrap')").body)
                    body.append(statement)
                node.body=body
            return node
    Paths().visit(tree)
    if raw_sites!=1 or retimed_sites!=1 or cartesian_sites!=(1 if kind=='carry' else 0):
        raise RuntimeError(f'One raw/TOPP readback and carry Cartesian knot window check required: {raw_sites}/{retimed_sites}/{cartesian_sites}')
    ast.fix_missing_locations(tree)
    return tree,{'scope':kind,'model_clone_sites':models,'independent_planning_PK_sites':pks,
        'numeric_goal_window_filters':goal_sites,'metadata_receipt_sites':metadata_sites,
        'raw_path_window_checks':raw_sites,'official_TOPP_window_checks':retimed_sites,
        'each_cartesian_support_knot_window_checks':cartesian_sites,
        'FMA_auditor_model_and_PK_not_reassigned':True,
        'original_environment_and_quality_AST_retained':True,
        'returned_paths_not_wrapped_clipped_or_rewritten':True,
        'compiled_AST_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()}

def _install_function(module,name,kind):
    previous=getattr(module,name)
    parent=getattr(previous,'__rerecord_derivation_ast__',None)
    if parent is None:
        raise RuntimeError('Install after AST-proven quality/environment and before observer wrappers')
    tree,receipt=derive_window_ast(parent,kind)
    receipt['parent_AST_sha256']=hashlib.sha256(ast.dump(parent,include_attributes=False).encode()).hexdigest()
    file=DIRECTORY/(module.__name__+'_'+name+'_numeric_window_generated.py')
    file.write_text(ast.unparse(tree)+'\n')
    namespace=dict(previous.__globals__)
    namespace['_window_create_model']=create_model
    namespace['_window_contains_selected_arm']=contains_selected_arm
    namespace['_window_contains_arm_path']=contains_arm_path
    exec(compile(tree,str(file),'exec'),namespace)
    function=namespace[name]
    function.__rerecord_derivation_ast__=copy.deepcopy(tree)
    function.__rerecord_window_overlay__=copy.deepcopy(receipt)
    setattr(module,name,function)
    receipt.update(generated_source=str(file),generated_source_sha256=_digest(file))
    return receipt

def install_auxiliary_window_overlay():
    """Call once after environment+quality installation, before metadata wrappers."""
    global _INSTALLED
    if _INSTALLED is not None:
        return copy.deepcopy(_INSTALLED)
    receipt={'overlay_source':str(Path(__file__).resolve()),'overlay_sha256':_digest(__file__),
        'library_source':str(DIRECTORY/'auxiliary_joint_window.py'),
        'library_sha256':_digest(DIRECTORY/'auxiliary_joint_window.py'),
        'source_files_or_actual_vendor_limits_modified':False,
        'GPU_or_scene_started_during_install':False,'functions':{}}
    for module in ('full_native_approach_equivalent_goals_v6','full_native_approach_equivalent_goals_dense_v7'):
        target=importlib.import_module(module)
        receipt['functions'][module]=_install_function(target,'plan_free_approach','free')
    target=importlib.import_module('full_native_carry_planner_v31')
    receipt['functions'][target.__name__]=_install_function(target,'plan_carry_to_task_pose','carry')
    expert=importlib.import_module('expert_generation').ExpertSession
    previous_init=expert.__init__
    def init(session,*args,**kwargs):
        previous_init(session,*args,**kwargs)
        session.sources[str(Path(__file__).resolve())]=receipt['overlay_sha256']
        session.sources[receipt['library_source']]=receipt['library_sha256']
        for item in receipt['functions'].values():
            session.sources[item['generated_source']]=item['generated_source_sha256']
        session.report['auxiliary_numeric_window_overlay']=copy.deepcopy(receipt)
        session.report['source_hashes']=session.sources
    expert.__init__=init
    _INSTALLED=copy.deepcopy(receipt)
    (DIRECTORY/'auxiliary_window_install_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return copy.deepcopy(receipt)
