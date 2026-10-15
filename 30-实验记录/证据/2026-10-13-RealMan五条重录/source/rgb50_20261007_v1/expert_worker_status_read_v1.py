"""Read compact status for specified worker roots; never write or admit data."""
from pathlib import Path
import json
import sys

for argument in sys.argv[1:]:
    root = Path(argument)
    status_path = root / 'worker_status.json'
    try:
        status = json.loads(status_path.read_bytes()) if status_path.is_file() else {}
    except (json.JSONDecodeError, OSError):
        print(json.dumps({'root': str(root.resolve()), 'status_write_in_progress': True}), flush=True)
        continue
    receipts = [{key: receipt.get(key) for key in ('name', 'accepted', 'elapsed_s', 'physics_steps', 'error')}
                for receipt in status.get('receipts', [])]
    finished_names = {receipt['name'] for receipt in receipts}
    active = [child.name for child in root.iterdir() if child.is_dir() and child.name not in finished_names
              and (child / 'sample_spec.json').is_file()]
    print(json.dumps({'scope': 'Read-only worker result status; independent formal admission not inferred',
                      'root': str(root.resolve()), 'jobs_total': status.get('jobs_total'),
                      'finished': status.get('finished'), 'worker_ended': status.get('worker_ended', False),
                      'receipts': receipts, 'active_or_not_yet_status_recorded': active}), flush=True)
