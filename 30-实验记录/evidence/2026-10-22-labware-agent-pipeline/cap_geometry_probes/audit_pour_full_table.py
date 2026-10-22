"""Recheck the SAME 18 candidate endpoints against original five-piece table.

Loads native create_table function verbatim via AST, suppressing visuals only.
No new candidate generation, IK, scene stepping or physical pose writes.
"""
import ast
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import sapien

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parent
TASK = PIPE/'robot_cap_task'
sys.path.insert(0,str(TASK));sys.path.insert(0,str(ROOT))
from cap_grasp_probe import Geometry,sha
from prepare_pour_geometry import moved_fcl

SOURCE = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin/envs/utils/create_actor.py')
BASE = SOURCE.parents[1]/'_base_task.py'
PRIOR = ROOT/'pour_geometry18_v1/pour_geometry.json'
OUT = ROOT/'pour_geometry18_fulltable_readback_v1'

class BuilderProxy:
    def __init__(self,builder):self.builder=builder
    def add_box_visual(self,*args,**kwargs):pass
    def __getattr__(self,name):return getattr(self.builder,name)

class SceneProxy:
    def __init__(self,scene):
        self.scene=scene
        self.default_physical_material=sapien.physx.PhysxMaterial(.5,.5,0.)
    def create_actor_builder(self):return BuilderProxy(self.scene.create_actor_builder())

def main():
    OUT.mkdir(exist_ok=False)
    prior=json.loads(PRIOR.read_text())
    snapshot=Path(prior['snapshot_path'])
    g=Geometry(OUT,snapshot=snapshot,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    tree=ast.parse(SOURCE.read_text());function=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='create_table')
    namespace={'sapien':sapien,'preprocess':lambda scene,pose:(scene,pose)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),'exec'),namespace)
    table=namespace['create_table'](SceneProxy(g.scene),sapien.Pose([0,0,.74]),length=1.2,width=.7,height=.74,thickness=.05,is_static=True,texture_id=None)
    component=table.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
    table_query=g.convert(component)
    shapes=[]
    for shape in component.collision_shapes:
        assert type(shape).__name__=='PhysxCollisionShapeBox'
        half=np.asarray(shape.half_size);T=np.asarray((table.pose*shape.local_pose).to_transformation_matrix())
        vertices=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*half
        world=vertices@T[:3,:3].T+T[:3,3]
        shapes.append({'type':type(shape).__name__,'half_size_m':half.tolist(),'world_matrix':T.tolist(),
                       'world_bounds_m':np.stack([world.min(0),world.max(0)]).tolist()})
    assert len(shapes)==len(table_query.shapes)==5
    rows=[]
    for candidate in prior['candidates']:
        if not candidate['ik_solved']:
            rows.append({'id':candidate['id'],'ik_solved':False,'all_declared_geometry_clear':False});continue
        g.model.set_qpos(np.asarray(candidate['q36']),full=True)
        body=moved_fcl(g,g.body_obj,np.asarray(candidate['body_target_world_matrix']))
        hits=[];pairs=[]
        for name,obj in [('body',body),*[(n,o)for n,o in g.objects.items()if n.startswith('r_')]]:
            gap,hit=g.distance(obj,table_query)
            pairs.append({'name':name,'gap_m':gap,'collision':hit})
            if hit:hits.append(name)
        # Geometry hand/body/dish/self checks retain their original exact poses
        # and results; only the proxy tabletop is replaced by all native parts.
        other_collisions=[x for x in candidate['collisions']if x['category']not in('right_table','body_table')]
        rows.append({'id':candidate['id'],'ik_solved':True,
            'native_table_clear':not hits,'native_table_colliding_links':hits,
            'native_table_minimum_body_gap_m':next(x['gap_m']for x in pairs if x['name']=='body'),
            'native_table_minimum_right_gap_m':min(x['gap_m']for x in pairs if x['name']!='body'),
            'native_table_pairs':pairs,'other_collision_count':len(other_collisions),
            'all_declared_geometry_clear':not hits and not other_collisions})
    clear=[x['id']for x in rows if x['all_declared_geometry_clear']]
    preferred=prior['preferred_candidate_id']if prior['preferred_candidate_id']in clear else(clear[0]if clear else None)
    result={'scope':'Same eighteen static post-cap-removal pour endpoint hypotheses, source-faithful full native table reconstruction',
        'source_sha256':sha(__file__),'prior_plan_sha256':sha(PRIOR),'native_create_actor_sha256':sha(SOURCE),'native_base_task_sha256':sha(BASE),
        'original_native_create_table_called':True,'visual_only_suppressed':True,
        'native_table_parameters':{'pose':[0,0,.74,1,0,0,0],'length':1.2,'width':.7,'height':.74,'thickness':.05},
        'native_table_shape_count':len(shapes),'native_table_shapes':shapes,
        'scope_of_table_identity':'Original current task Base_Task with random_table_height0/table_bias0 source parameters; reconstructed on CPU, not a new readback of the live GPU entity.',
        'physical_steps':0,'physical_pose_or_drive_writes_after_initialization':False,
        'new_candidate_count':0,'original_candidate_count':18,'clear_candidate_ids':clear,'preferred_candidate_id':preferred,
        'rechecks':rows,
        'preferred_full_record':next((x for x in prior['candidates']if x['id']==preferred),None),
        'limitation':'Endpoint geometry only. Right grasp relation is instantaneous actual snapshot geometry, not an attachment or anti-slip proof. Cap removal and left return are assumptions; actual cap entity is retained. No robot execution, pour dynamics, pills reception, path certification or dataset success.',
    }
    (OUT/'fulltable_audit.json').write_text(json.dumps(result,indent=2))
    record=result['preferred_full_record']
    (OUT/'fulltable_audit.md').write_text('# 倒药端点的完整原生桌复核\n\n'
        f'未新增候选；对原18项中的7个IK解，将桌面proxy替换为原始create_table函数实际构造的5个碰撞box，最终通过ID：{clear}，首选{preferred}。\n\n'
        '原函数逐字AST加载，仅屏蔽visual，桌参数取当前Base_Task的1.2×0.7m、高0.74m、厚0.05m。含桌板与四腿；是源码一致的CPU重构，未冒称live GPU entity新快照。\n\n'
        +(f'首选皿底中心{record["receiver_xy_m"]}、z0.74m；瓶口{record["mouth_center_world_m"]}，倾角155°朝+X。right_palm_target_world_matrix及right_arm6_rad见JSON preferred_full_record。\n\n'if record else'此有限族未获得通过端点。\n\n')
        +'实际盖未删除或移动；假设盖已移除且左手安全位。无物理步进/pose驱动写入。原始瞬时持握关系不能证明倾倒时不滑落；更不证明路径、倒12片或真实示范成功。\n')
    print(json.dumps({'clear_candidate_ids':clear,'preferred':preferred,'native_table_shapes':len(shapes)},indent=2))

if __name__=='__main__':main()
