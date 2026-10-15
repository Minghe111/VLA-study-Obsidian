"""Root-queued only: strict CPU bottle/table certificate for saved WORLD lift."""
import importlib
import json
import hashlib
from pathlib import Path
import numpy as np
import bottle_table_clearance_certificate as btcc


def run_world_table_certificate(task,base,reference):
    importlib.reload(btcc);base=Path(base)
    source=base/'curobo_world_z_lift_probe_20261003.npz'
    data=np.load(source);path=data['position']
    certificate=btcc.certify_bottle_table_path(task,path,'left',palm_to_bottle=reference,clearance_m=0.)
    certificate.update(source_command_npz=str(source),source_command_npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(btcc.__file__).read_bytes()).hexdigest())
    output=base/'world_z_lift_table_certificate_20261003.json'
    with output.open('x',encoding='utf-8') as f:json.dump(certificate,f,indent=2)
    print('WORLD_TABLE_CERTIFICATE',certificate['passed'],'evaluations',certificate['evaluations'],
        'sampled_min',certificate['sampled_minimum_gap'],'continuous',certificate['continuous_lower_bound'],
        'collisions',certificate['collisions'],'unresolved',certificate['unresolved_intervals'],'report',str(output),flush=True)
    return certificate


if 'task' in globals():
    world_z_lift_table_certificate=run_world_table_certificate(task,BASE,carry_reference[0])
elif __name__=='__main__':print(json.dumps(btcc.run_cpu_tests(),indent=2))
