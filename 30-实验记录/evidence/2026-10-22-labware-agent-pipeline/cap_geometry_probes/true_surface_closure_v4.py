"""One finite 1-D index/opposition continuation from actual v4 terminal.

Full convex SAT uses actual PhysX vertices, never FCL collide for cap touch.
This is a CPU candidate; actual contact/force guards remain authoritative.
"""
import ast
import itertools
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
import sapien
from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf

ROOT=Path(__file__).resolve().parent;PIPE=ROOT.parent;TASK=PIPE/'robot_cap_task'
sys.path[:0]=[str(TASK),str(ROOT)]
from cap_grasp_probe import Geometry,matrix,sha
from left_cap_held_probe import set_hand
from audit_pour_full_table import SOURCE,SceneProxy

SNAP=TASK/'left_contact_probes/left115_first_contact_v4_axisymmetric/actual_terminal_snapshot.json'
OUT=ROOT/'true_surface_closure_v4_v2'
START=np.array([0.,0.,0.,560.09765625,1000.,348.443359375])

def unique_axes(axes):
    axes=np.asarray(axes);norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1e-9]/norm[norm>1e-9,None]
    # +/- are the same separating axis.
    for i in range(len(axes)):
        nonzero=np.flatnonzero(np.abs(axes[i])>1e-8)
        if len(nonzero)and axes[i,nonzero[0]]<0:axes[i]*=-1
    _,idx=np.unique(np.round(axes,7),axis=0,return_index=True)
    return axes[np.sort(idx)]

class HullPart:
    def __init__(self,shape):
        if hasattr(shape,'vertices'):v=np.asarray(shape.vertices)*np.asarray(shape.scale)
        elif hasattr(shape,'half_size'):v=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*np.asarray(shape.half_size)
        else:raise TypeError(type(shape).__name__)
        self.local=matrix(shape.local_pose);self.vertices=np.asarray(v,dtype=float)
        h=ConvexHull(v);self.normals=unique_axes(h.equations[:,:3])
        edges={tuple(sorted(pair))for f in h.simplices for pair in itertools.combinations(f,2)}
        self.edges=unique_axes([v[j]-v[i]for i,j in edges])
    def world(self,T):
        W=T@self.local
        return WorldHull(self.vertices@W[:3,:3].T+W[:3,3],self.normals@W[:3,:3].T,self.edges@W[:3,:3].T)

class WorldHull:
    def __init__(self,v,n,e):
        self.v,self.n,self.e=v,n,e;self.box=np.stack([v.min(0),v.max(0)])

def signed_separation(a,b):
    # A positive AABB plane is already a valid independent separation proof.
    boxsep=float(np.max(np.maximum(a.box[0]-b.box[1],b.box[0]-a.box[1])))
    if boxsep>.002:return boxsep
    axes=unique_axes(np.vstack([a.n,b.n,np.cross(a.e[:,None,:],b.e[None,:,:]).reshape(-1,3)]))
    av=a.v@axes.T;bv=b.v@axes.T
    return float(np.max(np.maximum(av.min(0)-bv.max(0),bv.min(0)-av.max(0))))

def main():
    OUT.mkdir(exist_ok=False);t0=time.monotonic()
    g=Geometry(OUT,snapshot=SNAP,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    base=g.q0.copy();snapshot=json.loads(SNAP.read_text());cap_entity_matrix=matrix(g.cap.pose)
    cap_parts=[HullPart(s).world(cap_entity_matrix)for s in g.cap.find_component_by_type(sapien.physx.PhysxRigidStaticComponent).collision_shapes]
    local={name:[HullPart(s)for s in link.collision_shapes]for name,link in g.links.items()if name in g.objects}
    intended={'index':g.index_names,'thumb':g.thumb_names}
    def parts(q):
        g.pk.compute_forward_kinematics(q)
        return {name:[part.world(g.world_matrix(name))for part in entries]for name,entries in local.items()}
    def contact(sdk,group):
        q=set_hand(g,base,sdk);world=parts(q);values=[]
        for name in intended[group]:
            for ai,a in enumerate(world[name]):
                for bi,b in enumerate(cap_parts):values.append((signed_separation(a,b),name,ai,bi))
        value=min(values)
        return q,{'signed_SAT_separating_projection_m':value[0],'robot_link':value[1],'robot_piece':value[2],'cap_piece':value[3],
                  'full_SAT_overlap':value[0]<0.}
    current=START.copy();search=[];phases=[];failed=None
    for group,channel in [('index',3),('thumb',5)]:
        before=current.copy();q,c0=contact(before,group);samples=[{'sdk_value':float(before[channel]),**c0}]
        free=before.copy();hit=None
        for value in np.arange(before[channel]-5.,-5.,-5.):
            trial=before.copy();trial[channel]=max(0.,value);q,c=contact(trial,group)
            samples.append({'sdk_value':float(trial[channel]),**c})
            if c['signed_SAT_separating_projection_m']<=0:
                hit=trial;break
            free=trial
            if trial[channel]==0:break
        if hit is None:
            failed='No finite legal actual-surface crossing for '+group;search.append({'group':group,'samples':samples});break
        for _ in range(22):
            mid=(free+hit)/2;_,c=contact(mid,group)
            if c['signed_SAT_separating_projection_m']<=0:hit=mid
            else:free=mid
        _,free_c=contact(free,group);_,touch_c=contact(hit,group)
        # One explicit microscopic overlap projection, not an unbounded force
        # target. Search stays within one SDK unit past the true surface.
        lower=hit.copy();upper=hit.copy();upper[channel]=max(0.,hit[channel]-1.)
        _,margin_c=contact(upper,group)
        if margin_c['signed_SAT_separating_projection_m']>-.00005:
            # One unit may be <50um in this local geometry. Retain this smaller
            # overlap, never extend the micro allowance farther without cause.
            final=upper
        else:
            for _ in range(20):
                mid=(lower+upper)/2;_,c=contact(mid,group)
                if c['signed_SAT_separating_projection_m']<=-.00005:upper=mid
                else:lower=mid
            final=upper
        q,final_c=contact(final,group)
        search.append({'group':group,'channel':channel,'samples':samples,'free_surface_bracket_sdk6':free.tolist(),
            'touch_surface_bracket_sdk6':hit.tolist(),'free_surface':free_c,'touch_surface':touch_c,
            'candidate_sdk6':final.tolist(),'candidate_surface':final_c,
            'maximum_extra_sdk_past_zero_surface':1.,'target_negative_SAT_margin_m':-.00005})
        phases.append((group,before.copy(),final.copy()));current=final
    # Source-faithful complete native tabletop + four legs.
    fn=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='create_table')
    namespace={'sapien':sapien,'preprocess':lambda scene,p:(scene,p)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),namespace)
    table=namespace['create_table'](SceneProxy(g.scene),sapien.Pose([0,0,.74]),length=1.2,width=.7,height=.74,thickness=.05,is_static=True,texture_id=None)
    tableparts=[HullPart(s).world(matrix(table.pose))for s in table.find_component_by_type(sapien.physx.PhysxRigidStaticComponent).collision_shapes]
    bodyparts=[HullPart(s).world(matrix(g.body.pose))for s in g.body.find_component_by_type(sapien.physx.PhysxRigidStaticComponent).collision_shapes]
    proxy=SimpleNamespace(urdf_tree=ET.fromstring(export_kinematic_chain_urdf(g.art)),vertices=g.vertices,sim_joints=g.names)
    coeff=g.fma.FullMeshPathAuditor._motion_coefficients(proxy)
    audit=[];minimum={key:10. for key in['body','table','right','self','noncontact_cap']};collision=None;mininterval=10.
    moving=set(g.thumb_names+g.index_names)
    if failed is None:
        for group,a,b in phases:
            qa=set_hand(g,base,a);qb=set_hand(g,base,b);delta=np.abs(qb-qa)
            bound=max(float(c@delta)for c in coeff.values());count=max(1,int(np.ceil(bound/.00025)))
            # Relative self-link vertex displacement cannot exceed twice this
            # per-link bound. Subtract it for every interval's plane proof.
            interval_bound=max(float(c@delta)for c in coeff.values())/count
            for k in range(count+1):
                sdk=a+(b-a)*(k/count);q=set_hand(g,base,sdk);world=parts(q);mins={key:10. for key in minimum};hits=[]
                def compare(name,a,other,b,category):
                    gap=signed_separation(a,b);mins[category]=min(mins[category],gap)
                    if gap<=0:hits.append({'link':name,'other':other,'category':category,'signed_projection_m':gap})
                for name in moving:
                    for part in world[name]:
                        for bi,body in enumerate(bodyparts):compare(name,part,f'body{bi}',body,'body')
                        for ti,tablep in enumerate(tableparts):compare(name,part,f'table{ti}',tablep,'table')
                        for other,others in world.items():
                            if name==other or frozenset((name,other))in g.allowed:continue
                            if other in moving and other<name:continue
                            for otherpart in others:compare(name,part,other,otherpart,'right'if other.startswith('r_')else'self')
                # Other cap fingers are frozen but their actual geometry is
                # explicitly queried, not automatically exempted.
                for name in g.left_names:
                    if name in moving:continue
                    for part in world[name]:
                        for ci,cappart in enumerate(cap_parts):compare(name,part,f'cap{ci}',cappart,'noncontact_cap')
                for key,value in mins.items():minimum[key]=min(minimum[key],value)
                lower={key:value-interval_bound*(2. if key=='self'else 1.)for key,value in mins.items()}
                mininterval=min(mininterval,min(lower.values()))
                audit.append({'phase':group,'sample':k,'sample_count':count,'sdk6':sdk.tolist(),
                    'minimum_SAT_separation_m':mins,'interval_vertex_bound_m':interval_bound,
                    'conservative_interval_separation_lower_m':lower,'forbidden_overlap':hits})
                if hits:collision={'phase':group,'sample':k,'hits':hits};break
            if collision:break
    qfinal=set_hand(g,base,current)
    finalcontacts={name:contact(current,name)[1]for name in('index','thumb')}
    result={'scope':'Unique finite one-dimensional true-surface closure continuation from v4 measured pose; CPU candidate only',
        'source_sha256':sha(__file__),'snapshot_path':str(SNAP),'snapshot_sha256':sha(SNAP),
        'input_shas':{},
        'geometry_profile_sha256':sha(TASK/'profiles/bottle115/physics_spec115.json'),
        'geometry_helper_sha256':sha(TASK/'cap_grasp_probe.py'),'native_table_source_sha256':sha(SOURCE),
        'physical_tick':snapshot['physics_tick'],'start_sdk6':START.tolist(),'newclose_sdk6':current.tolist(),
        'ordered_actuator_continuations':[{'group':name,'start_sdk6':a.tolist(),'end_sdk6':b.tolist()}for name,a,b in phases],
        'actual_start_q36':base.tolist(),'candidate_q36':qfinal.tolist(),'joint_names':g.names,
        'palm_world_matrix':snapshot['left_palm_matrix'],'cap_world_matrix':snapshot['cap_matrix'],
        'cap_entity_vs_snapshot_max_matrix_error':float(np.max(np.abs(cap_entity_matrix-np.asarray(snapshot['cap_matrix'])))),
        'height_extra_m':g.height_extra,'surface_search':search,'surface_search_failure':failed,
        'final_true_surface_readback':finalcontacts,'forbidden_sample_count':len(audit),
        'forbidden_geometry_minimum_SAT_separation_m':minimum,'forbidden_overlap':collision,
        'minimum_conservative_forbidden_interval_lower_bound_m':mininterval,
        'forbidden_separation_positive_over_declared_intervals':failed is None and collision is None and mininterval>0,
        'candidate_passed_cpu':failed is None and collision is None and mininterval>0 and all(c['signed_SAT_separating_projection_m']>=-.0001001 and c['signed_SAT_separating_projection_m']<=0 for c in finalcontacts.values()),
        'physical_steps':0,'physical_pose_drive_writes_after_initialization':False,
        'intended_contact_algorithm':'Actual PhysX convex vertices, all hull face normals and all edge-cross-edge SAT axes; cap union33pieces. Positive separating projection proves disjoint; negative projection is finite geometric overlap margin, not actual force.',
        'fcl_collide_used_for_touch':False,'maximum_negative_projection_m':.0001,
        'source_boundary':'Original collision shape construction bound by profile/helper SHA; actual v4 q36/root and body/cap world pose. Five native table boxes reconstructed without visuals. Full arm pose/physics/assets unchanged.',
        'elapsed_s':time.monotonic()-t0,
        'limitation':'Candidate is only at the frozen actual terminal geometry. Physical replay may drift; keep slow real drive, per-4ms normal-force stop/force limits and current full forbidden guards. No new physical first touch, force closure, cap rotation, or demonstration success is claimed.',
    }
    result['input_shas']={str(path):sha(path)for path in[SNAP,TASK/'cap_grasp_probe.py',TASK/'profiles/bottle115/physics_spec115.json',PIPE.parent/'2026-10-22-labware-integration/lab_physics.py',SOURCE]}
    (OUT/'candidate.json').write_text(json.dumps(result,indent=2));(OUT/'forbidden_interval_samples.json').write_text(json.dumps(audit,indent=2))
    (OUT/'candidate.md').write_text('# v4实际末态的唯一真实表面续闭候选\n\n'
        f'CPU candidate_passed={result["candidate_passed_cpu"]}；SDK6从{START.tolist()}到{current.tolist()}，依次仅index(channel3)与thumbopp(channel5)。\n\n'
        '未改变palm、物理参数、抓握姿态族或实际状态。使用真实PhysX碰撞顶点的完整SAT而非FCL hit；保留零表面bracket与至多一SDK unit/50μm负投影余量。全部forbidden body/right/table/self与其它手指cap在≤0.25mm顶点移动界采样，并记录保守区间分离下界。\n\n'
        '该候选只供下一次精确回放后的低速限力续闭；每4ms真实法向力与actual geometry守卫优先，不表示机器人接触或开盖已通过。\n')
    print(json.dumps({k:result[k]for k in['candidate_passed_cpu','newclose_sdk6','final_true_surface_readback','forbidden_sample_count','forbidden_overlap','forbidden_geometry_minimum_SAT_separation_m','minimum_conservative_forbidden_interval_lower_bound_m','elapsed_s']},indent=2))

if __name__=='__main__':main()
