"""Staged exporter binding for source-pinned streamed full native audit files."""
import argparse,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_frozen_stream_export_v1',HERE/'streaming_export.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
original_read=base.read_json

def read_json(path,limit=base.JSON_LIMIT):
 path=Path(path)
 if path.name=='actual36_geometry_audit.json' and path.stat().st_size>limit:
  summary=original_read(path.with_name(path.stem+'_summary.json'))
  binding=summary.pop('bounded_full_audit_file')
  base.require(binding['path']==str(path.resolve()) and binding['sha256']==base.sha(path) and binding['all_samples_written']==summary['recorded_samples'],'Bounded full geometry summary is not bound to complete original audit')
  report=original_read(path.parent/'expert_report.json')
  base.require(summary==report['actual36_geometry_audit'],'Bounded geometry summary differs from original live report')
  pins=report['source_hashes'];helper=[p for p in pins if Path(p).name=='bounded_trace_overlay.py']
  base.require(len(helper)==1 and base.sha(helper[0])==pins[helper[0]],'Actual bounded-audit producer was not registered')
  return summary
 return original_read(path,limit)
base.read_json=read_json
original_export=base.export_candidate

def export_candidate(candidate,output,**kwargs):
 producer_before={str(Path(__file__).resolve()):base.sha(__file__),str(HERE/'streaming_export.py'):base.sha(HERE/'streaming_export.py')}
 result=original_export(candidate,output,**kwargs)
 base.require(all(base.sha(p)==v for p,v in producer_before.items()),'Staged export producer changed')
 receipt={'schema':'bounded-full-audit-export-binding-v1','export_receipt_sha256':base.sha(Path(output)/'export_receipt.json'),
  'candidate':str(Path(candidate).resolve()),'producer_sha256':producer_before,'accepted':False}
 summary=Path(candidate)/'actual36_geometry_audit_summary.json'
 if summary.exists():receipt['full_audit_summary']={'path':str(summary.resolve()),'sha256':base.sha(summary)}
 base.write_json(Path(output)/'staged_export_binding.json',receipt)
 return result
base.export_candidate=export_candidate
if __name__=='__main__':base.main()
