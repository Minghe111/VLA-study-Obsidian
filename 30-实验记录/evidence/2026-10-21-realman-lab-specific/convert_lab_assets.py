"""Extract passive meshes from pinned LabUtopia USD assets for SAPIEN viewing.

Geometry is preserved; Omni MDL is approximated by simple renderer materials.
No task behavior, collision, fluid, articulation, or device readout is imported.
"""
from pathlib import Path
import sys, json, hashlib
sys.path.insert(0, '/tmp/realman_lab_usd_runtime')
from pxr import Usd, UsdGeom, UsdShade
import numpy as np
import trimesh

OUT = Path(__file__).resolve().parent
SOURCE = OUT / 'source'
DEST = OUT / 'converted'
DEST.mkdir(exist_ok=True)
models = [
    ('beaker', 'Instruments/Beaker_01/Beaker_01.usd'),
    ('erlenmeyer', 'Instruments/ErlenmeyerFlask/ErlenmeyerFlask_01/ErlenmeyerFlask_01.usd'),
    ('test_tube', 'Instruments/TestTube/TestTube_01/TestTube_01.usd'),
    ('tube_rack', 'Instruments/TubeRack/TubeRack.usd'),
    ('pipette', 'Instruments/Pipette/Pipette.usd'),
    ('hotplate', 'Instruments/HeatingPlate/HeatingPlate.usd'),
    ('balance', 'Instruments/ElectronicScale/ElectronicScale_01/ElectronicScale_01.usd'),
    ('volumetric_flask', 'Instruments/VolumetricBottle/VolumetricBottle.usd'),
]

def material_info(prim):
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    inputs = {}
    if mat:
        for p in Usd.PrimRange(mat.GetPrim()):
            if p.IsA(UsdShade.Shader):
                for x in UsdShade.Shader(p).GetInputs():
                    v=x.Get()
                    if isinstance(v, (float, int, bool)):
                        inputs[x.GetBaseName()] = v
                    elif v is not None:
                        try:
                            if len(v)==3: inputs[x.GetBaseName()]=[float(a) for a in v]
                        except (TypeError, ValueError): pass
    glass='glass_ior' in inputs or 'glass_color' in inputs
    color=[.76,.78,.8]
    for key in ['diffuse_color_constant','diffuseColor','base_color_constant','glass_color']:
        if isinstance(inputs.get(key),list):
            color=inputs[key]; break
    if glass: color=[.84,.94,.98]
    return dict(source_material=str(mat.GetPath()) if mat else None,
                source_numeric_inputs=inputs, color=color,
                transmission=.88 if glass else 0,
                roughness=.07 if glass else .45,
                metallic=float(inputs.get('metallic_constant',0)),
                ior=float(inputs.get('glass_ior',1.49)))

records=[]
for name,path in models:
    stage=Usd.Stage.Open(str(SOURCE/path))
    assert stage
    cache=UsdGeom.XformCache()
    unit=UsdGeom.GetStageMetersPerUnit(stage)
    up=str(UsdGeom.GetStageUpAxis(stage))
    parts=[];bounds=[]
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):continue
        img=UsdGeom.Imageable(prim)
        if img.ComputeVisibility()==UsdGeom.Tokens.invisible:continue
        if str(img.ComputePurpose()) in ['guide','proxy']:continue
        mesh=UsdGeom.Mesh(prim)
        verts=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        if len(verts)==0:continue
        transform=np.asarray(cache.GetLocalToWorldTransform(prim))
        verts=(np.c_[verts,np.ones(len(verts))]@transform)[:,:3]*unit
        if up=='Y':verts=verts[:,[0,2,1]]*np.array([1,-1,1])
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=int)
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int)
        polygons=[];cursor=0
        for c in counts:
            polygons.append(indices[cursor:cursor+c]);cursor+=c
        subsets=UsdGeom.Subset.GetAllGeomSubsets(mesh)
        face_groups=[];covered=set()
        for subset in subsets:
            ids=[int(i) for i in subset.GetIndicesAttr().Get()]
            covered.update(ids)
            face_groups.append((ids,subset.GetPrim()))
        rest=[i for i in range(len(polygons)) if i not in covered]
        if rest:face_groups.append((rest,prim))
        for face_ids,material_prim in face_groups:
            triangles=[]
            for fid in face_ids:
                poly=polygons[fid]
                triangles.extend([poly[0],poly[j],poly[j+1]] for j in range(1,len(poly)-1))
            if not triangles:continue
            obj=trimesh.Trimesh(verts,np.asarray(triangles),process=False)
            if str(mesh.GetOrientationAttr().Get())=='leftHanded':obj.invert()
            output=DEST/f'{name}_{len(parts):03d}.obj'
            obj.export(output)
            bounds.append(obj.bounds)
            parts.append(dict(file=output.name,source_prim=str(prim.GetPath()),
                              vertices=len(obj.vertices),faces=len(obj.faces),
                              **material_info(material_prim)))
    assert parts,name
    bounds=np.asarray(bounds)
    lo=bounds[:,0].min(axis=0);hi=bounds[:,1].max(axis=0)
    rec=dict(name=name,source_file=path,source_up_axis=up,meters_per_unit=unit,
             bounds_m=[lo.tolist(),hi.tolist()],parts=parts)
    records.append(rec)
    print(name,'parts',len(parts),'bounds',(hi-lo).round(5).tolist(),
          'glass_parts',sum(p['transmission']>0 for p in parts),flush=True)

(DEST/'models.json').write_text(json.dumps(dict(
    source='LabUtopia',license='CC-BY-NC-4.0',usd_runtime='usd-core 26.8',
    scope='mesh extraction and approximate material translation for visualization only',
    models=records),ensure_ascii=False,indent=2))
