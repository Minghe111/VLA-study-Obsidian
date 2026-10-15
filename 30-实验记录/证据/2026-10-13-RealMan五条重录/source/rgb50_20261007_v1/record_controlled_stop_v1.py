"""Record a stopped, excluded simulation trial without changing its artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
work=Path(__file__).resolve().parent
root=work/'smoke_worker_right_upright_roll_v11'
payload={'accepted':False,'excluded_from_dataset':True,'worker_incomplete':True,
    'reason':'Root latest instruction requested v9 palm-roll single-factor trials; newly started v11 roll queue was stopped before switching version.',
    'controlled_signal':'SIGTERM','pid':1300043,
    'pid_command_was_exactly_checked_before_signal':True,
    'partial_output_preserved':True,'physical_steps_not_inferred_from_partial_report':True,
    'hardware_io':False,'created_utc':datetime.now(timezone.utc).isoformat(),
    'existing_artifacts_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*.json')}}
with (root/'controlled_stop_receipt.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload),flush=True)
