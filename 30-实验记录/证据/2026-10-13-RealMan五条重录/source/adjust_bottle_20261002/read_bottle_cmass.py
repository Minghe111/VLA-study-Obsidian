from sapien.physx import PhysxRigidDynamicComponent
c=task.bottle.actor.find_component_by_type(PhysxRigidDynamicComponent)
a={'mass':float(c.mass),'inertia':np.asarray(c.inertia).tolist(),'cmass_local_pose':c.cmass_local_pose.to_transformation_matrix().tolist()}
(BASE/'bottle_mass_inertia_cmass_20261003.json').write_text(json.dumps(a,indent=2));print('BOTTLE_CMASS',a,flush=True)
