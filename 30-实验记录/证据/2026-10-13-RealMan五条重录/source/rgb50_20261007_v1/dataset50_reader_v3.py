"""Private v3 readback bindings: original float32 pose reporting and fixed path root.

Frozen v1/v2 module globals and files remain unchanged. Physics thresholds,
full data audits and the independent boundary CPU verifier are retained.
"""
from __future__ import annotations
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from dataset50_audit import require,sha,close,BRANDS

ROOT=Path(__file__).resolve().parent
V1_SHA='f631ef652497b9c6797cb47ea6dd53a2a9ac22491ed05fca04e35ba34f7fed5b'
V2_BOUNDARY_SHA='a661e4a3ecdece0b7c7bbfcf3935bfc8e81ac4e5a72708bc527b80ae847d3ffd'


def native_pose_receipt(scene):
    spec=scene['sample_spec'];arm=spec['arm'];kind=spec['pose_class']
    require(arm in ['left','right'] and kind in ['upright','horizontal','diagonal'],'Unknown arm/pose class')
    require(int(spec['model_id']) in BRANDS,'Unverified soda brand')
    center=np.asarray(scene['actual_bottle_center_world'],float);shoulders={s:np.asarray(scene['shoulder_origins_world'][s],float) for s in ['left','right']}
    require(center.shape==(3,) and all(v.shape==(3,) for v in shoulders.values()),'Near-arm geometry dimensions')
    require(np.isfinite(center).all() and all(np.isfinite(v).all() for v in shoulders.values()),'Nonfinite near geometry')
    distances={s:float(np.linalg.norm(center-p)) for s,p in shoulders.items()};actual=min(distances,key=distances.get);margin=abs(distances['left']-distances['right'])
    require(actual==arm==scene['selected_arm']==scene['nearest_arm'] and margin>=.020,'Not actual nearest arm with20mm margin')
    for s in shoulders:close(distances[s],scene['arm_distances_m'][s],'near distance '+s)
    close(margin,scene['nearest_margin_m'],'near margin')
    matrix=np.asarray(scene['initial_pose'],float);require(matrix.shape==(4,4) and np.isfinite(matrix).all(),'Initial pose shape/values')
    close(matrix[3],[0,0,0,1],'Initial pose bottom row');close(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),'Initial rotation orthonormal',atol=2e-5)
    require(abs(np.linalg.det(matrix[:3,:3])-1)<2e-5,'Initial improper rotation')
    axis=matrix[:3,1];close(axis,scene['actual_axis_world'],'Actual local+Y axis',atol=2e-6)
    axis32=np.asarray(scene['initial_pose'],dtype=np.float32)[:3,1]
    # Exact producer arithmetic: SAPIEN matrix float32 arctan2/degrees.
    producer_heading=float(np.degrees(np.arctan2(axis32[1],axis32[0])))
    declared=float(scene['actual_axis_yaw_deg']);require(math.isfinite(declared) and -180<=declared<=180,'Invalid stored heading convention')
    circular_delta=(declared-producer_heading+180)%360-180
    require(abs(circular_delta)<=1e-8,'Stored heading differs from exact native float32 producer arithmetic')
    # Independent class checks keep the existing physical thresholds in float64.
    vertical=float(np.degrees(np.arccos(np.clip(abs(axis[2]),0,1))));heading=float(np.degrees(np.arctan2(axis[1],axis[0])));offset=abs((heading+90)%180-90)
    close(vertical,scene['upright_angle_deg'],'Initial vertical angle',atol=1e-5)
    if kind=='upright':require(vertical<=5,'Upright sample outside5deg')
    else:
        require(vertical>=85,'Lying sample outside5deg from table')
        require(offset<=20 if kind=='horizontal' else 35<=offset<=65,'Horizontal/diagonal heading class differs')
    require(scene['sampled_pose_settled_by_gravity'] is True,'Missing gravity-settle evidence')
    return {'model_id':int(spec['model_id']),'brand':BRANDS[int(spec['model_id'])],'arm':arm,'pose_class':kind,
        'seed':int(spec['seed']),'spec_index':int(spec['index']),'nearest_margin_m':margin,'upright_angle_deg':vertical,'heading_offset_deg':offset,
        'native_float32_heading_deg':producer_heading,'independent_float64_heading_deg':heading,'stored_heading_circular_error_deg':circular_delta}


def resolve_recorded_path(raw,directory):
    p=Path(raw)
    # v9 producer cwd is the fixed server RoboTwin root. Never guess basename.
    root=ROOT.parents[2]/'RoboTwin'
    resolved=(p if p.is_absolute() else root/p).resolve()
    require(resolved.is_relative_to(Path(directory).resolve()),'Retained proof escapes its actual trial under fixed producer cwd')
    return resolved


def private_module(path,name,transform=None):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'));tree=transform(tree) if transform else tree
    ast.fix_missing_locations(tree);env={'__file__':str(path),'__name__':name,'resolve_recorded_path':resolve_recorded_path}
    exec(compile(tree,str(path)+':v3_private_binding','exec'),env)
    env['private_ast_sha256']=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    return SimpleNamespace(**env)


def path_transform(tree):
    class ResolverOnly(ast.NodeTransformer):
        active=False
        replaced=0;removed=0
        def visit_FunctionDef(self,node):
            previous=self.active;self.active=node.name=='audit_geometry_chain';node=self.generic_visit(node);self.active=previous;return node
        def visit_Assign(self,node):
            if not self.active or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name) or node.targets[0].id not in ['p','prepath']:return self.generic_visit(node)
            name=node.targets[0].id
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='Path':
                require(len(node.value.args)==1,'Frozen retained path site changed')
                node.value=ast.Call(func=ast.Name(id='resolve_recorded_path',ctx=ast.Load()),args=[node.value.args[0],ast.Name(id='directory',ctx=ast.Load())],keywords=[]);self.replaced+=1;return node
            if isinstance(node.value,ast.IfExp):self.removed+=1;return None
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)==name+'.resolve':self.removed+=1;return None
            return self.generic_visit(node)
    t=ResolverOnly();tree=t.visit(tree);require(t.replaced==2 and t.removed==4,'Private boundary transform must change exactly two path-resolution sites')
    return tree


require(sha(ROOT/'dataset50_audit.py')==V1_SHA,'Frozen v1 readback changed')
require(sha(ROOT/'dataset50_boundary_v2.py')==V2_BOUNDARY_SHA,'Frozen v2 boundary readback changed')
original_readback=private_module(ROOT/'dataset50_audit.py','_dataset50_private_v1_for_v3')
original_readback.audit_trial.__globals__['pose_receipt']=native_pose_receipt
boundary_readback=private_module(ROOT/'dataset50_boundary_v2.py','_dataset50_private_v2_for_v3',path_transform)
audit_geometry_chain=boundary_readback.audit_geometry_chain