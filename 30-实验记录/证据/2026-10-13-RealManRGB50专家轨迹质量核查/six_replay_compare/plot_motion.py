"""Static scientific plots from read-only persisted numerical evidence."""
from pathlib import Path
import gzip,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent;ROOT=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据')
EX=ROOT/'2026-10-11-RealMan专家回放';MO=ROOT/'2026-10-12-RealMan80000原场景本机推理';NU=OUT.parent/'numeric'
metrics=json.loads((OUT/'six_replay_motion_comparison.json').read_text())
initial={e['index']:e for e in json.loads((MO/'source_initial_records.json').read_text())['episodes']}
def raw(kind,index):
    folder=EX/f'episode_{index:07d}' if kind=='Expert replay' else MO/'episodes'/f'ep{index:02d}_model80000'
    named=json.loads((folder/'initial_state.json').read_text())['actual_physical_contract']['full_qpos_joint_names']
    q=[initial[index]['source_initial_qpos36']];t=[0.]
    with gzip.open(folder/'actual_physical_trace.jsonl.gz','rt') as f:
        for line in f:r=json.loads(line);q.append(r['actual_qpos36_rad']);t.append(r['sim_time_s'])
    return np.array(t),np.array(q)*180/np.pi,named
def original(index):
    names=json.loads((EX/f'episode_{index:07d}/initial_state.json').read_text())['actual_physical_contract']['full_qpos_joint_names']
    with np.load(NU/f'episode_{index:07d}.npz') as z:
        t=np.r_[0.,z['realman_dexterous_physics/sim_time_after_s']];q=np.vstack([z['realman_dexterous_physics/full36_qpos_before_rad'][0],z['realman_dexterous_physics/full36_qpos_after_rad']])*180/np.pi
    return t,q,names
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10.5,4.8));styles=[('Original expert dataset','#14213d','-'),('Expert replay','#2a9d8f','--'),('Model80000','#e76f51','-')]
for label,col,ls in styles:
    t,q,n=original(17) if label=='Original expert dataset' else raw(label,17)
    ax.plot(t,q[:,n.index('l_joint6')],label=label,color=col,ls=ls,lw=1.7,alpha=.92)
start=initial[17]['source_initial_qpos36'][n.index('l_joint6')]*180/np.pi
ax.axhline(start-360,color='#666666',ls=':',label='One negative full turn from start')
ax.axvline(9.016,color='#2a9d8f',ls=':',alpha=.7);ax.axvline(8.916,color='#e76f51',ls=':',alpha=.7)
ax.text(9.2,-90,'Closing starts about9s\nFull turn already occurred',fontsize=10)
ax.set(xlabel='Simulation time (s)',ylabel='Raw actual left joint6 angle (deg)',title='Episode17: a real full wrist turn before closing',xlim=(0,24));ax.grid(alpha=.18);ax.legend(loc='upper right');fig.tight_layout();fig.savefig(OUT/'episode17_real_wrist_full_turn.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7.5),sharex=True)
for ax,joint in zip(axs.flat,['r_joint1','r_joint3','r_joint4','r_joint6']):
    for label,col,ls in styles:
        t,q,n=original(31) if label=='Original expert dataset' else raw(label,31)
        ax.plot(t,q[:,n.index(joint)],label=label,color=col,ls=ls,lw=1.6,alpha=.9)
    ax.axvline(9.092,color='#2a9d8f',ls=':',alpha=.6);ax.axvline(13.228,color='#e76f51',ls=':',alpha=.6)
    ax.set(title=joint,ylabel='Raw actual angle (deg)',xlabel='Simulation time (s)',xlim=(0,24));ax.grid(alpha=.18)
axs[0,0].legend(fontsize=9);fig.suptitle('Episode31: model elbow/wrist route differs from the original expert',fontsize=14)
fig.text(.5,.015,'Dotted lines: expert closes around9.1s; model around13.2s. Source expert ends at14.76s; model diagnostic runs24s.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(OUT/'episode31_expert_model_arm_routes.png',dpi=180);plt.close(fig)
print('PLOTS_READY',flush=True)
