"""Private full-mesh/RRT adaptation with exact measured held-object contacts.

Each obstacle keeps its body/cap/pill identity. Only actual PhysX right-hand
contact pairs at construction can be baseline permissions, and only while that
robot link is frozen in the proposed path. No physical collision mask changes.
"""
import copy
import inspect
import textwrap
import numpy as np
from robot_expert import adapted_native
from robot_cap_assembly import rigid

def measured_contact_baseline(scene,actor_parts,robot_link_ids):
    """Pure contact selection, also usable with CPU test doubles."""
    allowed=set();baseline=[];dt=float(scene.get_timestep())
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid actual physical timestep')
    for contact in scene.get_contacts():
        participants=[body.entity for body in contact.bodies]
        if len(participants)!=2:raise ValueError('Expected two actual contact participants')
        for entity,other in [participants,participants[::-1]]:
            entry=actor_parts.get(int(entity.per_scene_id))
            if entry is None or not other.name.startswith('r_inspire_'):continue
            if robot_link_ids.get(other.name)!=int(other.per_scene_id):continue
            category,part,_=entry
            if part not in ('body','cap'):continue
            if not any(token in other.name for token in ('thumb','index','middle')):continue
            force=0.
            for point in contact.points:
                impulse=np.asarray(point.impulse,dtype=float);normal=np.asarray(point.normal,dtype=float)
                position=np.asarray(point.position,dtype=float)
                if (any(value.shape!=(3,) or not np.isfinite(value).all() for value in (impulse,normal,position))
                        or abs(float(np.linalg.norm(normal))-1.)>1e-5):
                    raise ValueError('Invalid actual physical contact vector')
                force+=abs(float(impulse@normal))/dt
            if not contact.points or (part=='cap' and force<=1e-6):continue
            allowed.add((category,other.name))
            baseline.append({'category':category,'sourcepart':part,'actor_name':entity.name,
                'actor_id':int(entity.per_scene_id),'robot_link':other.name,'robot_actor_id':int(other.per_scene_id),
                'normal_force_N':force,'source':'actual current PhysX manifold, not FCL predicate'})
    return allowed,baseline

def install_held_left_planner():
    native=adapted_native();base=native.fma.FullMeshPathAuditor
    if getattr(base,'_held_parts_extension',False):return native
    source=textwrap.dedent(inspect.getsource(base.audit))
    loop='for link, obj in self.objects.items():'
    if source.count(loop)!=1:raise RuntimeError('Frozen auditor static loop changed')
    source=source.replace(loop,loop+'\n                    if (category, link) in self.measured_fixed_contact_allowlist and not moving[link]:\n                        continue')
    old='thresholds = {"bottle": min_bottle_clearance_m, "self": min_self_clearance_m, "table": min_table_clearance_m}'
    if source.count(old)!=1:raise RuntimeError('Frozen auditor thresholds changed')
    source=source.replace(old,'thresholds = {key: (min_self_clearance_m if key == "self" else min_table_clearance_m if key == "table" else min_bottle_clearance_m) for key in categories}')
    namespace=dict(native.fma.__dict__);exec(compile(source,'<held-left-part-audit>','exec'),namespace)
    audit_method=namespace['audit']
    class HeldPartsAuditor(base):
        _held_parts_extension=True
        def __init__(self,task,*args,**kwargs):
            super().__init__(task,*args,**kwargs)
            from mplib.sapien_utils.conversion import SapienPlanningWorld
            import trimesh
            # Never inherit an earlier body+cap+pill union under the body key.
            body_component=rigid(task.assembly.body)
            self.static['bottle']=native.fma._convert_component_scaled(body_component,
                SapienPlanningWorld.convert_physx_component,self.fcl,self.Pose)
            body_vertices,_=native.fma._link_vertices(body_component)
            body_matrix=np.asarray(task.assembly.body.pose.to_transformation_matrix())
            body_world=body_vertices@body_matrix[:3,:3].T+body_matrix[:3,3]
            self.static_boxes['bottle']=np.stack([body_world.min(0),body_world.max(0)])
            self.metadata['bottle_shapes']=len(self.static['bottle'].shapes)
            self.metadata.pop('closed_assembly_obstacle',None)
            actor_parts={int(task.assembly.body.per_scene_id):('bottle','body',task.assembly.body)}
            for category,part,entity in [('cap','cap',task.assembly.cap),
                    *[(f'pill_{i:02d}','pill',entity) for i,entity in enumerate(task.assembly.tablets)]]:
                component=rigid(entity);matrix=np.asarray(entity.pose.to_transformation_matrix())
                vertices,_=native.fma._link_vertices(component)
                if part=='cap':
                    obj=native.fma._convert_component_scaled(component,SapienPlanningWorld.convert_physx_component,self.fcl,self.Pose)
                else:
                    low=vertices.min(0);high=vertices.max(0)
                    box=trimesh.creation.box(extents=high-low);box.apply_translation((low+high)/2)
                    hull=self.fcl.Convex(np.asarray(box.vertices),np.asarray(box.faces,dtype=np.int32))
                    obj=self.fcl.FCLObject(category,self.Pose(matrix),[self.fcl.CollisionObject(hull)],[self.Pose()])
                self.static[category]=obj
                world=vertices@matrix[:3,:3].T+matrix[:3,3]
                self.static_boxes[category]=np.stack([world.min(0),world.max(0)])
                actor_parts[int(entity.per_scene_id)]=(category,part,entity)
            self.measured_fixed_contact_allowlist,baseline=measured_contact_baseline(task.scene,actor_parts,
                {name:int(link.entity.per_scene_id) for name,link in self.links.items()})
            self.metadata['held_contact_baseline']={'pairs':baseline,'only_if_robot_link_frozen':True,
                'physical_collision_masks_changed':False,'left_contact_exemptions_added':False,
                'physics_tick':task.physics_step_count,'sourcepart_tags':{
                    key:{'sourcepart':part,'actor_name':entity.name,'actor_id':int(entity.per_scene_id)}
                    for key,part,entity in actor_parts.values()}}
        audit=audit_method
    native.fma.FullMeshPathAuditor=HeldPartsAuditor
    # Preserve the original OMPL and conservative endpoint/path checks, adding
    # exactly the same tagged, measured baseline pairs to the search ACM.
    rrt=textwrap.dedent(inspect.getsource(native.strict_rrt_polyline))
    point='world.set_qpos_all(q0[arm_indices])'
    if rrt.count(point)!=2:raise RuntimeError('Frozen RRT world update layout changed')
    before='for pair in audit.allowed:\n        first, second = sorted(pair); acm.set_entry(first, second, True)'
    if rrt.count(before)!=1:raise RuntimeError('Frozen RRT ACM layout changed')
    rrt=rrt.replace(before,before+'\n    for category, link in audit.measured_fixed_contact_allowlist:\n        if np.any(np.asarray(audit.motion_coefficients[link])[arm_indices] > 0.):\n            raise RuntimeError("Measured contact baseline robot link is not frozen in this move group")\n        acm.set_entry(link, "teacher_"+category, True)')
    rrt=rrt.replace('"only_existing_self_exemptions": True, "physical_contact_exemptions_added": False',
        '"only_existing_self_exemptions": True, "physical_contact_exemptions_added": False, "planning_only_measured_contact_baseline": audit.metadata["held_contact_baseline"]')
    namespace=dict(native.__dict__);exec(compile(rrt,'<held-left-rrt>','exec'),namespace)
    native.strict_rrt_polyline=namespace['strict_rrt_polyline']
    return native
