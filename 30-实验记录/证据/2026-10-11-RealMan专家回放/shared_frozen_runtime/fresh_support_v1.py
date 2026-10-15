"""Actual native tabletop/bottle support, unchanged depth/footprint budgets."""
from __future__ import annotations
import numpy as np

def matrix(value):
    v=np.asarray(value,dtype=float)
    if v.shape!=(4,4) or not np.isfinite(v).all() or not np.allclose(v[3],[0,0,0,1],atol=1e-9):raise ValueError('Finite homogeneous pose required')
    return v

def native_tabletop(component):
    signs=np.asarray([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    entity=matrix(component.entity.pose.to_transformation_matrix());rows=[];chunks=[]
    for i,s in enumerate(component.collision_shapes):
        if 'Box' not in type(s).__name__ or not hasattr(s,'half_size'):raise ValueError('Native support table must retain Box collisions')
        half=np.asarray(s.half_size,dtype=float)
        if half.shape!=(3,) or not np.isfinite(half).all() or np.any(half<=0):raise ValueError('Invalid native half_size')
        p=entity@matrix(s.local_pose.to_transformation_matrix());w=(signs*half)@p[:3,:3].T+p[:3,3]
        b=np.stack([w.min(0),w.max(0)]);a=np.abs(p[:3,:3])
        aligned=bool(np.allclose(a@a.T,np.eye(3),atol=1e-7) and np.allclose(a,np.round(a),atol=1e-7) and np.allclose(a[2],[0,0,1],atol=1e-7))
        rows.append({'shape':i,'type':type(s).__name__,'half_size_m':half.tolist(),'world_bounds_m':b.tolist(),'aligned':aligned,'top_z_m':float(b[1,2])});chunks.append(w)
    if not chunks:raise ValueError('Missing actual table collision shapes')
    allw=np.concatenate(chunks);whole=np.stack([allw.min(0),allw.max(0)]);top=float(whole[1,2])
    if abs(top-.74)>1e-6:raise ValueError('Fixed official tabletop must remain .74m')
    candidates=[r for r in rows if r['aligned'] and abs(r['top_z_m']-top)<=1e-7]
    if len(candidates)!=1:raise ValueError('Unique horizontal native tabletop required')
    b=np.asarray(candidates[0]['world_bounds_m'])
    if not np.allclose(b[1,:2]-b[0,:2],[1.2,.7],atol=1e-6):raise ValueError('Official tabletop footprint changed')
    if np.any(whole[0,:2]<b[0,:2]-1e-7) or np.any(whole[1,:2]>b[1,:2]+1e-7):raise ValueError('Native legs outside tabletop')
    return {'table_top_z':top,'tabletop_bounds_world_m':b.tolist(),'whole_table_bounds_world_m':whole.tolist(),'native_shapes':rows,'collision_shapes_removed':0}

class ActualSupport:
    def __init__(self,task,vertices):
        from sapien.physx import PhysxRigidBaseComponent
        self.task=task;self.vertices=np.asarray(vertices,dtype=float).copy()
        tables=[a for a in task.scene.get_all_actors() if a.name=='table']
        if len(tables)!=1:raise ValueError('Unique real support table required')
        self.table=native_tabletop(tables[0].find_component_by_type(PhysxRigidBaseComponent));self.top=self.table['table_top_z'];self.bounds=np.asarray(self.table['tabletop_bounds_world_m'])
        w=self.world_vertices();self.baseline=float(w[:,2].min()-self.top)
        if not np.isfinite(w).all() or not -.0005<=self.baseline<.001:raise ValueError('Initial bounded support gap invalid')
        self.floor=self.baseline-.000020;self.released=False;self.release_step=None;self.rows=[]
    def world_vertices(self):
        p=matrix(self.task.bottle.get_pose().to_transformation_matrix())
        return self.vertices@p[:3,:3].T+p[:3,3]
    def check(self,step):
        if step!=len(self.rows)+1:raise ValueError('Every executed physical support sample must be checked')
        w=self.world_vertices();gap=float(w[:,2].min()-self.top);reason=None
        if not np.isfinite(w).all():reason='Nonfinite native bottle vertices'
        elif np.any(w[:,:2]<self.bounds[0,:2]) or np.any(w[:,:2]>self.bounds[1,:2]):reason='Bottle leaves actual tabletop footprint'
        elif gap<(0. if self.released else self.floor):reason='Original actual support signed floor / released clearance violated'
        if gap>=.001 and not self.released:self.released=True;self.release_step=step
        row={'step':step,'signed_gap_m':gap,'baseline_m':self.baseline,'floor_m':self.floor,'released':self.released,'release_step':self.release_step,'violation':reason}
        self.rows.append(row)
        if reason:raise RuntimeError(reason+': '+str(row))
        return row
    def summary(self,last_step):
        return {'passed':bool(len(self.rows)==last_step and not any(r['violation'] for r in self.rows)),'samples':len(self.rows),'actual_step_coverage':len(self.rows)==last_step,'baseline_signed_gap_m':self.baseline,'floor_signed_gap_m':self.floor,'release_step':self.release_step,'native_table_geometry':self.table,'source':'Current actual cooked collision vertices; no predicted attachment or expert support path'}
