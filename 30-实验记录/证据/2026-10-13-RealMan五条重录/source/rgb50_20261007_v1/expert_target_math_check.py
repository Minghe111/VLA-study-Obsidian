"""Synthetic frame-conversion checks; no simulator, IK or physical-success claim."""
import json
import numpy as np
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import mat2quat, quat2mat
from expert_generation_v5 import rigid_functional_point_target

axes=[np.array([1.,0.,0.]),np.array([-1.,0.,0.]),np.array([0.,0.,1.]),
      np.array([0.,0.,-1.]),np.array([1.,1.,0.])/np.sqrt(2)]
rows=[]
for axis in axes:
    x=np.cross(axis,[1.,0.,0.])
    if np.linalg.norm(x)<.1:x=np.cross(axis,[0.,0.,1.])
    x/=np.linalg.norm(x);z=np.cross(x,axis)
    bottle=np.eye(4);bottle[:3,:3]=np.stack([x,axis,z],axis=1)
    current=np.array([.14,-.105,.85]);target=np.array([.25,-.12,.95])
    rotation=axangle2mat([1.,2.,3.],.37)
    live=[.23,-.21,.86,*mat2quat(rotation)]
    for yaw in [0.,30.,-60.,180.]:
        pose,delta=rigid_functional_point_target(live,bottle,current,target,yaw)
        point_error=np.linalg.norm((np.asarray(pose[:3])-delta@np.asarray(live[:3]))-(target-delta@current))
        rotation_error=np.max(abs(quat2mat(pose[-4:])-delta@rotation))
        up_error=np.linalg.norm(delta@axis-np.array([0.,0.,1.]))
        assert point_error<1e-12 and rotation_error<1e-12 and up_error<1e-12
        rows.append({'axis':axis.tolist(),'yaw_deg':yaw,'point_error_m':point_error,
                     'rotation_matrix_error':rotation_error,'upright_axis_error':up_error})
print(json.dumps({'passed':True,'scope':'synthetic rigid-transform math only; no simulator or IK',
                  'checked_cases':len(rows),'rows':rows}),flush=True)
