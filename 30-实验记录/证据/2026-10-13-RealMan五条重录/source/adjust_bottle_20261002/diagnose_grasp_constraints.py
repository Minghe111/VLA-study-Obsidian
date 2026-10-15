ix=np.flatnonzero(ok.ravel());flatq=allq.reshape(-1,6)
state=mg.kinematics.get_state(ta.to_device(flatq[ix]));spheres=state.link_spheres_tensor.detach().cpu().numpy()
world=spheres[...,:3]@rootR.T+origin.p
radius=spheres[...,3]
delta=np.abs(world-np.array([0,0,.715]))-np.array([.6,.35,.025])
signed=np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(np.max(delta,axis=-1),0)
margin=signed-radius
rank=np.argsort(np.min(margin,axis=1))[::-1]
print('SPHERE_CONFIG_KEYS',vars(mg.kinematics.kinematics_config).keys(),flush=True)
print('BEST_TABLE_MARGINS',np.min(margin,axis=1)[rank[:10]],flush=True)
for k in rank[:8]:
 i,j=np.unravel_index(ix[k],ok.shape);entry=metadata[i];worst=int(np.argmin(margin[k]))
 print('DIAG',entry,'q',flatq[ix[k]].tolist(),'worst_sphere',worst,'world',world[k,worst].tolist(),'radius',float(radius[k,worst]),'minz',float(np.min(world[k,:,2]-radius[k])),flush=True)
checker_no_world=IKSolver(IKSolverConfig.load_from_robot_config(cfg,None,tensor_args=ta,num_seeds=1,self_collision_check=True,self_collision_opt=True,use_cuda_graph=False))
c=checker_no_world.check_constraints(JointState.from_position(ta.to_device(flatq[ix])))
print('SELF_FEASIBLE',int(c.feasible.sum()),'of',len(ix),flush=True)