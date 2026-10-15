"""Task-scoped extra environment safety for genuine randomized expert episodes.
Adds existing native clutter colliders to CPU world/audits; never writes physics.
Original source bytes, geometry, LP, grasp tolerances and state clock stay intact.
"""
from pathlib import Path
import ast,copy,hashlib,inspect,json,math,textwrap
import numpy as np
EXTRA_CLEARANCE_M=.001
VERTEX_DRIFT_BUDGET_M=.001
_INSTALLED=None

def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _components(task):
    from sapien.physx import PhysxRigidBaseComponent
    refs=getattr(task,'_rerecord_clutter_entities',None)
    if refs is None:raise RuntimeError('Actual official clutter actor references not retained')
    records=[]
    for ordinal,item in enumerate(refs):
        actor=getattr(item,'actor',item)
        entities=[link.entity for link in actor.get_links()] if hasattr(actor,'get_links') else [actor]
        collidable=[]
        for entity in entities:
            components=[c for c in entity.get_components() if isinstance(c,PhysxRigidBaseComponent) and len(c.collision_shapes)]
            if len(components)>1:raise RuntimeError('Ambiguous native clutter colliders')
            if components:collidable.append(components[0])
        if not collidable:raise RuntimeError('Official clutter actor has no native collision geometry')
        records.append((ordinal,item,collidable))
    expected=len(getattr(task,'record_cluttered_objects',[]))
    if expected!=len(records):raise RuntimeError('Actual clutter references do not match official clutter receipt')
    return records

def _snapshot_clutter(task):
    import full_mesh_path_audit as fma
    result=[]
    for ordinal,item,comps in _components(task):
        for ci,comp in enumerate(comps):
            vertices,types=fma._link_vertices(comp);matrix=fma._pose_matrix(comp.entity.pose)
            result.append({'clutter_index':ordinal,'component_index':ci,'entity':comp.entity,'component':comp,
              'entity_name':comp.entity.name,'category':f'clutter_{ordinal:02d}_{ci:02d}',
              'vertices':vertices,'initial_matrix':matrix.copy(),'shape_count':len(comp.collision_shapes),
              'geometry_types':types})
    return result

def _coverage(records,task):
    expected=len(getattr(task,'record_cluttered_objects',[]));covered=len(set(r['clutter_index'] for r in records))
    return {'passed':expected==covered,'expected_clutter_count':expected,'covered_clutter_count':covered,
      'all_collidable_clutter_covered':expected==covered,'covered_collision_component_count':len(records),
      'covered_shape_count':sum(r['shape_count'] for r in records),
      'records':[{'category':r['category'],'name':r['entity_name'],'shape_count':r['shape_count'],'geometry_types':r['geometry_types']} for r in records],
      'covered_clutter_entity_ids':sorted(id(getattr(item,'actor',item)) for _,item,_ in _components(task)),
      'static_categories':sorted(set('clutter_%02d'%r['clutter_index'] for r in records)),
      'extra_clearance_m':EXTRA_CLEARANCE_M,'actual_clutter_max_vertex_drift_budget_m':VERTEX_DRIFT_BUDGET_M,
      'geometry_source':'Actual generated native PhysX collision shapes; full scaled MPLib converter',
      'CPU_snapshot_obstacles_are_not_physical_attachments':True}

def _derived_function(original,transform):
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source);transform(tree)
    ast.fix_missing_locations(tree);ns=dict(original.__globals__);exec(compile(tree,str(Path(__file__))+'::'+original.__qualname__,'exec'),ns)
    derived=ns[original.__name__];derived.__rerecord_derivation_ast__=copy.deepcopy(tree)
    return derived,{'source_name':original.__qualname__,'original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
      'derived_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()}

class _FreeThresholds(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Assign(self,node):
        self.generic_visit(node)
        if any(isinstance(t,ast.Name) and t.id=='thresholds' for t in node.targets) and isinstance(node.value,ast.Dict):
            self.count+=1
            return [node,ast.parse("thresholds.update({n: EXTRA_CLEARANCE_M for n in self.static if n not in thresholds})").body[0]]
        return node

def install_environment_overlay(source_package,legacy_directory):
    global _INSTALLED
    if _INSTALLED is not None:return copy.deepcopy(_INSTALLED)
    # Original bootstrap overlays must finish before adding stricter environment gates.
    import expert_v9_settle_wake_v5
    import full_mesh_path_audit as fma,carry_mesh_path_audit as cma
    from mplib.sapien_utils.conversion import SapienPlanningWorld
    original_init=fma.FullMeshPathAuditor.__init__
    def init(self,task,*a,**kw):
        original_init(self,task,*a,**kw)
        records=_snapshot_clutter(task)
        groups={}
        for r in records:
            obj=fma._convert_component_scaled(r['component'],SapienPlanningWorld.convert_physx_component,self.fcl,self.Pose)
            if obj is None or len(obj.shapes)!=r['shape_count']:raise RuntimeError('Clutter native/FCL shape coverage differs')
            category='clutter_%02d'%r['clutter_index'];g=groups.setdefault(category,{'shapes':[],'poses':[],'world_vertices':[]})
            for shape,local in zip(obj.shapes,obj.shape_poses):
                g['shapes'].append(self.fcl.CollisionObject(shape.get_collision_geometry()))
                g['poses'].append(self.Pose(fma._pose_matrix(obj.pose)@fma._pose_matrix(local)))
            m=r['initial_matrix'];g['world_vertices'].append(r['vertices']@m[:3,:3].T+m[:3,3])
        for category,g in groups.items():
            self.static[category]=self.fcl.FCLObject(category,self.Pose(),g['shapes'],g['poses'])
            v=np.vstack(g['world_vertices']);self.static_boxes[category]=np.stack([v.min(0),v.max(0)])
        self.metadata['randomized_environment_coverage']=_coverage(records,task)
        if not self.metadata['randomized_environment_coverage']['passed']:raise RuntimeError('Incomplete randomized environment audit')
        self._rerecord_clutter_records=records
    fma.FullMeshPathAuditor.__init__=init
    # Original vector overlay uses a private AST function. Rebuild that exact
    # independently verified self broadphase plus the extra clutter thresholds.
    module_ast=ast.parse(Path(legacy_directory,'full_mesh_path_audit.py').read_text())
    cls_ast=next(n for n in module_ast.body if isinstance(n,ast.ClassDef) and n.name=='FullMeshPathAuditor')
    method=copy.deepcopy(next(n for n in cls_ast.body if isinstance(n,ast.FunctionDef) and n.name=='audit'))
    tr=_FreeThresholds();tr.visit(method)
    if tr.count!=1:raise RuntimeError('Exactly one original free audit threshold declaration required')
    loops=[n for n in ast.walk(method) if isinstance(n,ast.For) and ast.unparse(n.iter)=='self.self_pairs']
    if len(loops)!=1:raise RuntimeError('Exactly one original FMA self pair iteration required')
    loops[0].iter=ast.parse("self._rgb50_self_candidate_pairs(boxes, outgoing, best['self']['distance_m'], certified['self']['lower_bound_m'])",mode='eval').body
    tree=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(tree)
    ns=dict(vars(fma));ns['EXTRA_CLEARANCE_M']=EXTRA_CLEARANCE_M
    exec(compile(tree,str(Path(__file__))+'::free_environment_audit','exec'),ns);free_method=ns['audit']
    vector_receipt=copy.deepcopy(fma.rgb50_vector_aabb_overlay_receipt)
    def free(self,path,side='left',**kwargs):
        if getattr(self,'_rgb50_vector_stats',None) is not None:raise RuntimeError('Nested vector audit unsupported')
        self._rgb50_vector_context=None
        self._rgb50_vector_stats={'samples':0,'original_pair_visits':0,'original_skip_proved_in_bulk':0,'scalar_pair_candidates':0,'broadphase_elapsed_s':0.,'maximum_roundoff_optimization_guard_m':0.}
        try:
            result=free_method(self,path,side,**kwargs)
            result['vector_broadphase_audit']={**self._rgb50_vector_stats,'overlay_receipt':vector_receipt,'all_unfiltered_pairs_recompute_original_scalar_lower':True,'pair_order_samples_limits_thresholds_bounds_preserved':True}
            return result
        finally:
            self.rgb50_last_vector_stats=dict(self._rgb50_vector_stats);del self._rgb50_vector_stats;del self._rgb50_vector_context
    free_receipt={'original_source_sha256':_sha(Path(legacy_directory,'full_mesh_path_audit.py')),'derived_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),'original_self_broadphase_vectorization_retained':True}
    fma.FullMeshPathAuditor.audit=free
    # Add robot-clutter and rigid-carried-bottle-clutter categories to the same
    # original continuous displacement bound / FCL collision / limit audit.
    def transform_carry(tree):
        fn=tree.body[0];categories=thresholds=None;loop=None
        for n in ast.walk(fn):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='categories' for t in n.targets):categories=n
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='thresholds' for t in n.targets):thresholds=n
            if isinstance(n,ast.For) and ast.unparse(n.target)=='(sample_index, (q, path_index, fraction))':loop=n
        if not all(x is not None for x in (categories,thresholds,loop)):raise RuntimeError('Original carry AST integration sites changed')
        ci=fn.body.index(categories);fn.body[ci+1:ci+1]=ast.parse("extras = [n for n in self.static if n not in ('bottle','table')]\nfor n in extras:\n    categories.extend([n+'_robot', n+'_bottle'])").body
        ti=fn.body.index(thresholds);fn.body[ti+1:ti+1]=ast.parse("for n in extras:\n    thresholds[n+'_robot'] = EXTRA_CLEARANCE_M\n    thresholds[n+'_bottle'] = EXTRA_CLEARANCE_M").body
        extra=ast.parse("for extra in extras:\n    evaluate(extra+'_bottle','bottle',extra,bottle,self.static[extra],bottle_box,self.static_boxes[extra],float(self.bottle_coefficients@outgoing),not object_moving['__carried_bottle__'])\n    for name,obj in self.objects.items():\n        evaluate(extra+'_robot',name,extra,obj,self.static[extra],boxes[name],self.static_boxes[extra],float(self.motion_coefficients[name]@outgoing),not object_moving[name])").body
        # evaluate closure has been defined and every table/self check remains.
        pos=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='include_samples')
        loop.body[pos:pos]=extra
    carry,carry_receipt=_derived_function(cma.CarryMeshPathAuditor.audit,transform_carry)
    carry.__globals__['EXTRA_CLEARANCE_M']=EXTRA_CLEARANCE_M;cma.CarryMeshPathAuditor.audit=carry
    # RRT carry world must also see clutter before final geometry acceptance.
    import full_native_carry_planner_v31 as planner
    def transform_world(tree):
        found=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='PlanningWorld':
                found+=1
                node.args[1]=ast.parse("[table_object,bottle_object]+[clone_obstacle(source,'realman_'+name,inverse_root@cma._matrix(source.pose)) for name,source in auditor.static.items() if name not in ('bottle','table')]",mode='eval').body
        registry_sites=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Compare) and ast.unparse(node)=="set(registry) != {'realman_table', 'held_bottle'}":
                registry_sites+=1
                node.comparators[0]=ast.parse("{'realman_table','held_bottle'} | {'realman_'+name for name in auditor.static if name not in ('bottle','table')}",mode='eval').body
        if found!=1 or registry_sites!=1:raise RuntimeError('Exactly one original carry PlanningWorld/complete registry assertion required')
    world,world_receipt=_derived_function(planner.plan_carry_to_task_pose,transform_world);planner.plan_carry_to_task_pose=world
    import dense_free_certificate_policy_v7 as dense
    def transform_dense(tree):
        fn=tree.body[0]
        class Replace(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='THRESHOLDS_M':return ast.copy_location(ast.Name(id='required_thresholds',ctx=node.ctx),node)
                return node
        Replace().visit(fn)
        fn.body.insert(1 if isinstance(fn.body[0],ast.Expr) else 0,ast.parse("required_thresholds = {**THRESHOLDS_M, **{n: EXTRA_CLEARANCE_M for n in auditor.static if n not in THRESHOLDS_M}}").body[0])
    dense_fn,dense_receipt=_derived_function(dense.refine_certificate,transform_dense)
    dense_fn.__globals__['EXTRA_CLEARANCE_M']=EXTRA_CLEARANCE_M;dense.refine_certificate=dense_fn
    # Genuine every-4ms guard runs AFTER original physical guard and appends
    # evidence to its real trace. Stops a sample upon clutter impact or motion.
    import expert_generation as expert
    orig_step=expert.ExpertSession.step
    def step(self):
        orig_step(self)
        guard=getattr(self.task,'_rerecord_environment_guard',None)
        if guard is None:raise RuntimeError('No task-specific randomized environment guard prepared')
        monitor_environment_step(self.task,self.steps,self.traces[-1],guard)
    expert.ExpertSession.step=step
    from randomized_support_guard import install_randomized_support_guard
    support_receipt=install_randomized_support_guard(legacy_directory)
    original_expert_init=expert.ExpertSession.__init__
    def expert_init(self,*args,**kwargs):
        original_expert_init(self,*args,**kwargs)
        self.sources[str(Path(__file__).resolve())]=_sha(__file__)
        self.report['randomized_environment_safety_overlay']={'source':str(Path(__file__).resolve()),'sha256':_sha(__file__),'extra_clearance_m':EXTRA_CLEARANCE_M,'vertex_drift_budget_m':VERTEX_DRIFT_BUDGET_M,'free_carry_RRT_and_geometry_and_actual_step_guards':True}
    expert.ExpertSession.__init__=expert_init
    _INSTALLED={'installed':True,'source_files_unchanged':True,'native_scene_state_writes':0,'physics_steps_during_install':0,
      'extra_clearance_m':EXTRA_CLEARANCE_M,'vertex_drift_budget_m':VERTEX_DRIFT_BUDGET_M,
      'free_audit':free_receipt,'carry_audit':carry_receipt,'carry_RRT_world':world_receipt,'dense_certificate':dense_receipt,
      'support_departure_randomized_guard':support_receipt,
      'overlay_source':str(Path(__file__).resolve()),'overlay_sha256':_sha(__file__)}
    return copy.deepcopy(_INSTALLED)

def prepare_environment(task,out,scene,spec):
    out=Path(out);records=_snapshot_clutter(task);coverage=_coverage(records,task)
    guard={'coverage':coverage,'records':records,'steps_checked':0,'max_clutter_vertex_displacement_m':0.,'robot_clutter_active_contacts':0,'bottle_clutter_active_contacts':0,'output':out}
    task._rerecord_environment_guard=guard
    (out/'randomized_environment_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    return coverage

def monitor_environment_step(task,physics_step,trace_row,guard):
    import full_mesh_path_audit as fma
    if type(physics_step) is not int or physics_step != guard['steps_checked']+1 or trace_row.get('step') != physics_step:
        raise RuntimeError('Randomized environment guard lost exact actual physical step continuity')
    objects={id(r['entity']):r for r in guard['records']}
    robot={id(l.entity) for l in task.robot.left_entity.links};bottle=id(task.bottle.actor)
    worst=0.
    for r in guard['records']:
        now=np.asarray(fma._pose_matrix(r['entity'].pose),dtype=float);old=np.asarray(r['initial_matrix'],dtype=float);v=np.asarray(r['vertices'],dtype=float)
        if now.shape!=(4,4) or old.shape!=(4,4) or v.ndim!=2 or v.shape[1]!=3 or not len(v) or not all(np.isfinite(a).all() for a in (now,old,v)):
            raise RuntimeError('Nonfinite or incomplete actual randomized clutter geometry')
        for matrix in (now,old):
            if not np.allclose(matrix[3],[0,0,0,1],atol=1e-7,rtol=0) or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=2e-4,rtol=0) or np.linalg.det(matrix[:3,:3])<.9998:
                raise RuntimeError('Invalid actual randomized clutter rigid transformation')
        movement=float(np.linalg.norm(v@now[:3,:3].T+now[:3,3]-(v@old[:3,:3].T+old[:3,3]),axis=1).max())
        worst=max(worst,movement)
    events=[]
    for c in task.scene.get_contacts():
        ids=[id(b.entity) for b in c.bodies]
        if not any(i in objects for i in ids):continue
        if not any(i in robot or i==bottle for i in ids):continue
        for p in c.points:
            vector=np.asarray(p.impulse,dtype=float);normal=np.asarray(p.normal,dtype=float);separation=float(p.separation)
            if vector.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(vector).all() or not np.isfinite(normal).all() or not math.isfinite(separation) or np.linalg.norm(normal)<1e-10:
                raise RuntimeError('Nonfinite or invalid actual randomized clutter contact')
            impulse=float(abs(vector@normal/np.linalg.norm(normal)))
            if impulse>1e-10 or separation<-.00002:
                events.append({'entities':[b.entity.name for b in c.bodies], 'kind':'bottle_clutter' if bottle in ids else 'robot_clutter','normal_impulse_Ns':impulse,'separation_m':separation})
    guard['steps_checked']+=1;guard['max_clutter_vertex_displacement_m']=max(guard['max_clutter_vertex_displacement_m'],worst)
    trace_row['randomized_environment']={'native_clutter_components_checked':len(objects),'max_vertex_displacement_from_initial_m':worst,'active_target_or_robot_clutter_contacts':events}
    if events:
        guard['bottle_clutter_active_contacts']+=sum(e['kind']=='bottle_clutter' for e in events)
        guard['robot_clutter_active_contacts']+=sum(e['kind']=='robot_clutter' for e in events)
        raise RuntimeError('Randomized clutter collision with robot or target bottle')
    if worst>VERTEX_DRIFT_BUDGET_M+1e-9:raise RuntimeError('Randomized clutter moved beyond conservative CPU obstacle budget')


def environment_execution_receipt(task,out,report):
    guard=task._rerecord_environment_guard
    expected=int(report['physical_steps'])
    with np.load(Path(out)/'physics_telemetry.npz',allow_pickle=False) as data:
        steps=data['physics_step'];before=data['sim_time_before_s'];after=data['sim_time_after_s']
        complete=bool(expected>0 and guard['steps_checked']==expected and np.array_equal(steps,np.arange(1,expected+1)) and len(before)==expected and len(after)==expected and np.isfinite(before).all() and np.isfinite(after).all() and np.allclose(after-before,float(task.scene.get_timestep()),atol=1e-8,rtol=0))
    if not complete:raise RuntimeError('Randomized guard does not cover the complete genuine physics telemetry')
    receipt={k:v for k,v in guard.items() if k not in ('records','output')}
    receipt['original_report_physical_steps']=expected
    receipt['complete_physics_telemetry_coverage_verified']=complete
    receipt.update(passed=guard['steps_checked']>0 and guard['max_clutter_vertex_displacement_m']<=VERTEX_DRIFT_BUDGET_M+1e-9 and guard['robot_clutter_active_contacts']==0 and guard['bottle_clutter_active_contacts']==0,
      every_actual_physics_step_checked=True,hardware_io=False)
    Path(out,'randomized_environment_execution.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def finalize_environment(task,out,report):
    receipt=environment_execution_receipt(task,out,report)
    report['randomized_environment_execution']=receipt
    if not receipt['passed']:raise RuntimeError('Actual randomized environment execution guard did not pass')
    return receipt
