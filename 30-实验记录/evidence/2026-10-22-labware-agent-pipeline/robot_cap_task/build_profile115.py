"""Generate an independent 115-mm vessel profile, without physics or rendering."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parent
PIPE=ROOT.parent
FROZEN=PIPE.parent/'2026-10-22-labware-integration'
OUT=ROOT/'profiles/bottle115'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def export(mesh,path,color):
    mesh.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=color,metallicFactor=0.,roughnessFactor=.25,alphaMode='BLEND'))
    mesh.export(path)
    loaded=trimesh.load(path,force='scene').dump(concatenate=True)
    return {'path':str(path),'sha256':sha(path),'bounds_m':np.asarray(loaded.bounds).tolist(),
        'source':'new parametric visual geometry matching the collision dimensions; not scaled old 90-mm mesh'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-untested-draft',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=args.refresh_untested_draft)
    source=FROZEN/'physics_spec.json'
    spec=copy.deepcopy(json.loads(source.read_text()))
    spec['version']='independent-contact-bottle115-v1'
    spec['bottle']['height_m']=.115
    spec['cap']['closed_center_z_m']=.118
    spec['thread_guide']['model']='passive spring/friction helical surrogate inherited from frozen RobotCapAssembly'
    spec['thread_guide']['axial_target']='closed_z + pitch * clipped measured relative spin/(2*pi); paired passive reaction, no angular actuator'
    spec.pop('fixture',None)
    spec['boundary']=['Independent derived 115-mm bottle; original90-mm inputs unchanged',
        'No actuator, world fixture, cap holder or object pose command after initialization',
        'Body80g, cap12g and tablet0.35g are engineering mass assumptions, not measurements',
        'No hardware thread/friction calibration; robot grasp/unscrew/pour require separate actual validation']
    spec_path=OUT/'physics_spec115.json';spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    wall=trimesh.creation.annulus(r_min=.021,r_max=.025,height=.115,sections=96)
    wall.apply_translation([0,0,.115/2])
    floor=trimesh.creation.cylinder(radius=.025,height=.003,sections=96)
    floor.apply_translation([0,0,.003/2])
    body=trimesh.util.concatenate([wall,floor])
    ring=trimesh.creation.annulus(r_min=.0258,r_max=.03,height=.012,sections=96)
    ring.apply_translation([0,0,-.006])
    top=trimesh.creation.cylinder(radius=.03,height=.004,sections=96)
    cap=trimesh.util.concatenate([ring,top])
    visuals={'body':export(body,OUT/'body115.glb',[.63,.36,.08,.72]),
             'cap':export(cap,OUT/'cap14.glb',[.88,.88,.87,1.])}
    for key in ['body','cap']:
        original=FROZEN/f'prepared/reagent_bottle_{key}/model_data0.json'
        data=copy.deepcopy(json.loads(original.read_text()))
        if key=='body':
            data['center']=[0.,0.,.0575];data['extents']=[.05,.05,.115]
            data['target_pose'][1][2][3]=.115
            data['functional_matrix'][0][2][3]=.115
            data['functional_matrix'][2][2][3]=.115
            data['annotation_provenance']['geometry']['opening_z_m']=.115
            data['annotation_provenance']['bounds_m']=[[-.025,-.025,0],[.025,.025,.115]]
            data['orientation_point'][0][2][3]=.0575
            # Keep the actual grasp anchor at45mm: the tested hand pose remains unchanged.
        data['annotation_provenance']['profile']='independent115mm_body_same45mm_grasp_anchor'
        provenance=data['annotation_provenance']
        inherited_keys=['source','surface_anchor_candidates','surface_anchor_method',
            'surface_query_excluded_degenerate_faces','pour_rim_surface_check',
            'hollow_collision_validation','hollow_collision_verified','grasp_execution_verified']
        provenance['inherited90_reference_not_validated_for_profile']={k:provenance.pop(k) for k in inherited_keys if k in provenance}
        provenance['source']={'project':'Independent115mm parametric vessel profile',
            'source_file':str(spec_path),'source_sha256':sha(spec_path),
            'mesh_path':visuals[key]['path'],'mesh_sha256':visuals[key]['sha256'],
            'source_status':'new local derived mesh/spec; no inherited90mm physical validation'}
        provenance['hollow_collision_verified']=False
        provenance['grasp_execution_verified']=False
        provenance['physical_validation_status']='pending actual115mm profile tests'
        # Preserve the support-plane API added to the pipeline metadata, without
        # importing any old physical-success claim.
        registry=json.loads((PIPE/'metadata/registry.json').read_text())
        expanded=json.loads(Path(registry['assets'][f'reagent_bottle_{key}']['model_data']).read_text())
        if len(expanded['functional_matrix'])>len(data['functional_matrix']):
            data['functional_matrix'].append(expanded['functional_matrix'][-1])
            data['functional_point_description'].append(expanded['functional_point_description'][-1])
        data['annotation_provenance']['source_profile']=str(spec_path)
        data['annotation_provenance']['source_profile_sha256']=sha(spec_path)
        data['annotation_provenance']['original_metadata_sha256']=sha(original)
        data['annotation_provenance']['grasp_execution_verified']=False
        (OUT/f'{key}_model_data0.json').write_text(json.dumps(data,indent=2)+'\n')
    evidence=PIPE/'cap_geometry_probes/q36_height_extra_25mm/right_sweep.json'
    manifest={'profile':'bottle115-v1','date':'2026-10-22','source_spec':str(source),'source_spec_sha256':sha(source),
        'new_spec':str(spec_path),'new_spec_sha256':sha(spec_path),'body_height_m':.115,'cap_closed_center_z_m':.118,
        'body_grasp_anchor_z_m':.045,'physical_dimension_changes':{'body_height_m':[.09,.115],'cap_closed_center_z_m':[.093,.118]},
        'selection_basis':'First clear height in finite5mm CPU grid with actual preclose q36; not global minimum or physical success',
        'cpu_evidence':str(evidence),'cpu_evidence_sha256':sha(evidence),'mass_kg':{'body':.08,'cap':.012,'each_tablet':.00035},
        'mass_status':'Retained engineering assumption; not a density/mass measurement',
        'visuals':visuals,'frozen_assembly_sha256':sha(PIPE/'robot_cap_assembly.py'),
        'new_metadata_sha256':{k:sha(OUT/f'{k}_model_data0.json') for k in ['body','cap']},
        'generator_sha256':sha(__file__),'physics_run':False,'robot_grasp_verified':False}
    (OUT/'profile_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
