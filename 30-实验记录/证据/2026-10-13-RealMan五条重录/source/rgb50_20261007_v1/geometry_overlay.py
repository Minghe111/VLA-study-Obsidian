"""Additive native-shape-count adaptation for multiple official bottle assets.

Only the old Coke/Sprite-specific literal seven-shape assertion is changed.
All native shapes must remain present; collision pairs, scale correction,
clearance bounds, exact contacts and trajectory checks are untouched.
Install once before importing the frozen audit/planner dependency modules.
"""
from pathlib import Path
import ast
import hashlib
import sys
import types


def install_native_shape_count_overlay(legacy_dir):
    path=Path(legacy_dir)/'full_mesh_path_audit.py'
    source=path.read_text(encoding='utf-8')
    tree=ast.parse(source)
    sites=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.If):continue
        test=ast.unparse(node.test)
        if test == "self.static['bottle'] is None or len(self.static['bottle'].shapes) != 7":
            sites.append(node)
    if len(sites)!=1:raise RuntimeError('Expected exactly one frozen seven-shape bottle assertion')
    sites[0].test=ast.parse("self.static['bottle'] is None or len(bottle_component.collision_shapes) < 1 or len(self.static['bottle'].shapes) != len(bottle_component.collision_shapes)",mode='eval').body
    for node in ast.walk(sites[0]):
        if isinstance(node,ast.Constant) and node.value == "Expected the official bottle's seven real convex collision shapes":
            node.value='Converted bottle geometry must preserve every positive-count actual native collision shape'
    ast.fix_missing_locations(tree)
    receipt={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'overlay_source':str(Path(__file__).resolve()),'overlay_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'overlay_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
             'changed_assertions':1,'actual_native_count_must_be_positive':True,
             'converted_count_must_equal_native_count':True,'collision_pairs_or_clearance_changed':False,
             'physical_scene_changed':False}
    existing=sys.modules.get('full_mesh_path_audit')
    if existing is not None and getattr(existing,'rgb50_shape_count_overlay_receipt',None)==receipt:
        return receipt
    stale=[name for name in ('carry_mesh_path_audit','full_native_carry_planner_v31','observed_full_trace_audit_v13') if name in sys.modules]
    if stale:raise RuntimeError('Install shape-count overlay before importing geometry dependencies: '+str(stale))
    module=types.ModuleType('full_mesh_path_audit')
    module.__file__=str(path);module.__package__=''
    sys.modules['full_mesh_path_audit']=module
    try:exec(compile(tree,str(path)+':rgb50_native_shape_count','exec'),module.__dict__)
    except BaseException:
        if existing is None:sys.modules.pop('full_mesh_path_audit',None)
        else:sys.modules['full_mesh_path_audit']=existing
        raise
    module.rgb50_shape_count_overlay_receipt=receipt
    return receipt
