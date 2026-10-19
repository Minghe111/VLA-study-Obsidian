"""Run one explicitly supplied inference command and sample whole-card GPU use."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def sample():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=3, check=True,
    )
    return [dict(index=int(parts[0]), name=parts[1].strip(),
                 used_mib=float(parts[2]), total_mib=float(parts[3]))
            for parts in (line.split(',') for line in result.stdout.splitlines())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('An explicit command is required')
    if args.report.exists():
        raise FileExistsError(args.report)
    started = time.monotonic()
    baseline = sample()
    readings = [{'elapsed_s': 0., 'gpus': baseline}]
    errors = []
    process = subprocess.Popen(command)
    while process.poll() is None:
        try:
            readings.append({'elapsed_s': time.monotonic()-started, 'gpus': sample()})
        except Exception as error:
            errors.append(repr(error))
        time.sleep(.5)
    result = dict(
        status='completed' if process.returncode == 0 else 'child_failed',
        returncode=process.returncode, command=command,
        elapsed_s=time.monotonic()-started, baseline=baseline,
        scope='Sampled whole-card use including desktop and simulator; not process-attributed and not exact peak.',
        readings=readings, errors=errors,
        sampled_peak_mib=max(gpu['used_mib'] for row in readings for gpu in row['gpus']),
    )
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
