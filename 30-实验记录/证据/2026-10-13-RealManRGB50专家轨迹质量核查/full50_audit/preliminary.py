from pathlib import Path
import json
import numpy as np

ROOT=Path('/tmp/realman-data-motion-audit-20261013')

def path_metric(points):
    if len(points)<2:return {'path_m':0.,'straight_m':0.,'path_to_straight':None}
    length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    straight=float(np.linalg.norm(points[-1]-points[0]))
    return {'path_m':length,'straight_m':straight,'path_to_straight':length/straight if straight>1e-6 else None}

def main():
    receipt=json.loads((ROOT/'numeric_receipt.json').read_text())
    joints=[];commands=[];paths=[]
    for row in receipt['episodes']:
        i=row['index'];a=np.load(ROOT/row['numeric_file']);before=a['realman_dexterous_physics/measured_state24_before'];after=a['realman_dexterous_physics/measured_state24_after']
        states=np.vstack((before[:1],after));t=np.r_[a['realman_dexterous_physics/sim_time_before_s'][0],a['realman_dexterous_physics/sim_time_after_s']]
        applied=a['realman_dexterous_physics/applied_command24'];tc=a['realman_dexterous_physics/sim_time_before_s'];tl=a['realman_dexterous/sample_time_s']
        for arm,base in [('left',0),('right',12)]:
            q=states[:,base:base+6];u=np.unwrap(q,axis=0);cmd=applied[:,base:base+6]
            close=np.any(applied[:,base+6:base+11]<950.,axis=1);hits=np.flatnonzero(close);ct=float(tc[hits[0]]) if len(hits) else None
            keep=t<=ct+1e-10 if ct is not None else np.ones(len(t),bool)
            for j in range(6):
                raw=np.ptp(q[:,j])*180/np.pi;uw=np.ptp(u[:,j])*180/np.pi
                pre=np.ptp(u[keep,j])*180/np.pi if np.sum(keep)>1 else 0
                joints.append({'index':i,'arm':arm,'active':arm==row['arm'],'J':j+1,'raw_range_deg':float(raw),'unwrapped_range_deg':float(uw),'net_deg':float((u[-1,j]-u[0,j])*180/np.pi),'cumulative_deg':float(np.abs(np.diff(u[:,j])).sum()*180/np.pi),'preclose_range_deg':float(pre),'closure_command_time_s':ct})
                dc=np.diff(cmd[:,j]); commands.append({'index':i,'arm':arm,'J':j+1,'dense_max_command_jump_deg':float(np.max(np.abs(dc))*180/np.pi),'near_2pi_jumps':int(np.sum((np.abs(np.abs(dc)-2*np.pi)<.1)&(np.abs(dc)>np.pi)))})
            lp=a[f'state/{arm}_ee_poses'][:,:3];lkeep=tl<=ct+1e-10 if ct is not None else np.ones(len(tl),bool)
            p=path_metric(lp[lkeep]);p.update({'index':i,'arm':arm,'active':arm==row['arm'],'closure_command_time_s':ct,'samples':int(np.sum(lkeep))});paths.append(p)
        a.close()
    result={'scope':'Preliminary all50 only; per-joint ranges are not six-joint sums; preclose means SDK command<950, not physical contact',
            'max_actual_joint_ranges':sorted(joints,key=lambda x:x['unwrapped_range_deg'],reverse=True)[:12],
            'max_preclose_actual_joint_ranges':sorted([x for x in joints if x['active']],key=lambda x:x['preclose_range_deg'],reverse=True)[:12],
            'max_dense_command_jumps':sorted(commands,key=lambda x:x['dense_max_command_jump_deg'],reverse=True)[:8],
            'max_active_preclose_endpoint_path':sorted([x for x in paths if x['active']],key=lambda x:x['path_m'],reverse=True)[:8],
            'max_active_preclose_endpoint_detour_ratio':sorted([x for x in paths if x['active'] and x['path_to_straight'] is not None],key=lambda x:x['path_to_straight'],reverse=True)[:8],
            'actual_single_joint_ranges_ge360_count':sum(x['unwrapped_range_deg']>=360 for x in joints),'near2pi_command_jumps_total':sum(x['near_2pi_jumps'] for x in commands)}
    out=ROOT/'full50_audit/preliminary.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
