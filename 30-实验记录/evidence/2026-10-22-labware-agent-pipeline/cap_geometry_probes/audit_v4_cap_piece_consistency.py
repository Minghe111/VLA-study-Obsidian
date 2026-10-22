"""Finite per-piece readback and independent hull separation of existing q.

Audits the contradictory FCL collide/distance at existing first-touch labels.
No SDK extension, search, pose correction or simulation step is performed.
"""
import json
from pathlib import Path
import sys
import numpy as np
import sapien
from scipy.spatial import ConvexHull
import trimesh

ROOT=Path(__file__).resolve().parent;PIPE=ROOT.parent;TASK=PIPE/'robot_cap_task'
sys.path.insert(0,str(TASK))
from cap_grasp_probe import Geometry,matrix,sha
from left_cap_held_probe import set_hand

OUT=ROOT/'v4_actual_first_surface_readback'
END=TASK/'left_contact_probes/left115_first_contact_v4_axisymmetric/actual_terminal_snapshot.json'
REF=ROOT/'left115_static_rebind400/rebind.json'

def shape_world_vertices(shape,entity_matrix):
    if hasattr(shape,'vertices'):
        local=np.asarray(shape.vertices,dtype=float)*np.asarray(shape.scale,dtype=float)
    elif hasattr(shape,'half_size'):
        local=np.asarray([[x,y,z]for x in(-1,1)for y in(-1,1)for z in(-1,1)])*np.asarray(shape.half_size)
    else:raise TypeError(type(shape).__name__)
    T=entity_matrix@matrix(shape.local_pose)
    return local@T[:3,:3].T+T[:3,3]

def convex_mesh(vertices):
    hull=ConvexHull(vertices)
    return hull,trimesh.Trimesh(vertices=vertices,faces=hull.simplices,process=False)

def examine(label,snapshot,q_override=None,close_sdk=None):
    g=Geometry(OUT/('piece_'+label),snapshot=snapshot,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    q=g.q0.copy()if q_override is None else np.asarray(q_override)
    if close_sdk is not None:q=set_hand(g,q,close_sdk)
    g.model.set_qpos(q,full=True);g.pk.compute_forward_kinematics(q)
    actual_cap=matrix(g.cap.pose);expected=np.asarray(json.loads(Path(snapshot).read_text())['cap_matrix'])
    cap_component=g.cap.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
    cap_vertices=[shape_world_vertices(shape,actual_cap)for shape in cap_component.collision_shapes]
    cap_hulls=[convex_mesh(v)[0]for v in cap_vertices]
    cap_mesh=trimesh.util.concatenate([convex_mesh(v)[1]for v in cap_vertices])
    result={'label':label,'height_extra':g.height_extra,'cap_entity_pose_matrix':actual_cap.tolist(),
        'cap_entity_vs_actual_snapshot_matrix_max_error':float(np.max(np.abs(actual_cap-expected))),
        'cap_field_vs_entity_matrix_max_error':float(np.max(np.abs(g.cap_matrix-actual_cap))),
        'cap_FCL_vs_entity_matrix_max_error':float(np.max(np.abs(np.asarray(g.cap_obj.pose.to_transformation_matrix())-actual_cap))),
        'cap_collision_parts':len(cap_component.collision_shapes),
        'cap_collision_group_unique':list({tuple(s.collision_groups)for s in cap_component.collision_shapes}),
        'parents':{}}
    for name in ['l_inspire_left_thumb_4','l_inspire_left_index_2']:
        obj=g.objects[name]
        # All queried parents have one actual PhysX convex collision piece.
        robot_shapes=g.links[name].collision_shapes
        assert len(robot_shapes)==1
        world=g.world_matrix(name)
        vv=shape_world_vertices(robot_shapes[0],world)
        hull,mesh=convex_mesh(vv)
        perpiece=[]
        for i,(capv,caph)in enumerate(zip(cap_vertices,cap_hulls)):
            sa=obj.shapes[0]
            a=g.fcl.CollisionObject(sa.get_collision_geometry(),g.Pose(obj.pose)*g.Pose(obj.shape_poses[0]))
            sb=g.cap_obj.shapes[i]
            b=g.fcl.CollisionObject(sb.get_collision_geometry(),g.Pose(g.cap_obj.pose)*g.Pose(g.cap_obj.shape_poses[i]))
            hit=bool(g.fcl.collide(a,b,g.cr).is_collision());distance=float(g.fcl.distance(a,b,g.dr).min_distance)
            # Independent sufficient convex separation: use real vertex hull
            # face normals; positive separation of any unit axis proves the
            # actual cooked convex pieces disjoint. No FCL face encoding used.
            axes=np.vstack([hull.equations[:,:3],caph.equations[:,:3]])
            va=vv@axes.T;vb=capv@axes.T
            separations=np.maximum(va.min(0)-vb.max(0),vb.min(0)-va.max(0))
            separation=float(np.max(separations))
            perpiece.append({'piece':i,'fcl_collision':hit,'fcl_distance_m':distance,
                'independent_hull_face_axis_separation_m':separation,
                'independently_proven_disjoint':separation>1e-8,
                'fcl_hit_with_positive_distance':hit and distance>1e-6,
                'fcl_hit_despite_independent_separating_plane':hit and separation>1e-6})
        # Vertex-to-triangle in both directions gives a certified gap upper
        # bound for these surface point candidates. It is not called exact
        # triangle/edge distance; hull plane separation is the lower bound.
        _,da,_=trimesh.proximity.closest_point(cap_mesh,mesh.vertices)
        _,db,_=trimesh.proximity.closest_point(mesh,cap_mesh.vertices)
        upper=float(min(da.min(),db.min()))
        result['parents'][name]={'physx_collision_groups':[list(s.collision_groups)for s in robot_shapes],
            'actual_collision_vertex_count':len(vv),
            'compound_result':{'gap_collision_gated':g.distance(obj,g.cap_obj)},
            'minimum_fcl_unsigned_piece_distance_m':min(x['fcl_distance_m']for x in perpiece),
            'minimum_independent_convex_hull_separation_lower_bound_m':min(x['independent_hull_face_axis_separation_m']for x in perpiece),
            'surface_candidate_distance_upper_bound_m':upper,
            'all_actual_cap_pieces_independently_proven_disjoint':all(x['independently_proven_disjoint']for x in perpiece),
            'contradictory_fcl_hit_piece_ids':[x['piece']for x in perpiece if x['fcl_hit_despite_independent_separating_plane']],
            'per_cap_piece':perpiece}
    return result

def main():
    ref=json.loads(REF.read_text());plan=json.loads((ROOT/'left115_static_shortcut_v1/candidate_plan.json').read_text())
    rows=[examine('terminal_actual',END),examine('original_first_touch',Path(plan['snapshot']),ref['first_contact_q36'])]
    result={'scope':'Finite true-vertex convex separation and per-FCL-piece contradiction audit, no robot operation',
        'source_sha256':sha(__file__),'terminal_snapshot_sha256':sha(END),'original_first_touch_sha256':sha(REF),
        'physical_steps':0,'physical_pose_drive_writes_after_initialization':False,
        'cases':rows,
        'filter_readback_scope':'CPU original robot configure_collisions readback; cap original builder default [1,1,0,0]. Not a new live GPU filter export. Group0/1 overlap=1 and cap group2=0 supplies no self-exclusion with left group3=2001.',
        'finding':'Positive independent separating planes invalidate these FCL collide-based first-touch labels. A static binary crossing of unreliable collision predicate is not physical contact and must not terminate contact seeking.',
        'limitation':'Actual runtime contact impulses remain authoritative. No new closure endpoint is proposed here. Independent face-axis plane lower bounds and vertex/triangle upper bounds bracket physical convex mesh separation, not a full continuous robot path certificate.',
    }
    (OUT/'piece_consistency.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([{ 'case':row['label'],'cap_entity_pose_error':row['cap_entity_vs_actual_snapshot_matrix_max_error'],
        'parents':{n:{k:v[k]for k in['minimum_fcl_unsigned_piece_distance_m','minimum_independent_convex_hull_separation_lower_bound_m','surface_candidate_distance_upper_bound_m','all_actual_cap_pieces_independently_proven_disjoint','contradictory_fcl_hit_piece_ids']}for n,v in row['parents'].items()}}for row in rows],indent=2))

if __name__=='__main__':main()
