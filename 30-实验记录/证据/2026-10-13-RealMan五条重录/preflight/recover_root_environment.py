"""Recover only documented file-write bodies from this task's own history.
No historical shell, GPU, SSH, credential, or user-message replay.
"""
from pathlib import Path
import json,re,ast,hashlib
ROOT=Path(__file__).resolve().parent.parent
OLD='/tmp/realman-rerecord5-20261013'
LOGS=[Path('/home/admin123/.codex/sessions/2026/09/18/rollout-2026-09-18T20-35-35-01a0b483-dd68-7d91-a6cd-8f9bb7d31ed6.jsonl'),Path('/home/admin123/.codex/sessions/2026/10/13/rollout-2026-10-13T21-43-04-01a0b483-dd68-7d91-a6cd-8f9bb7d31ed6_01a13580-a216-7370-a16f-be1a8c2866ac.jsonl')]
TARGETS=('planning/randomized_environment_guard.py','preflight/test_environment_guard.py')
receipt=[]
for log in LOGS:
 for ordinal,line in enumerate(log.open()):
  o=json.loads(line);v=o.get('payload',{})
  if o.get('type')!='response_item' or v.get('type')!='custom_tool_call':continue
  js=v.get('input','')
  for m in re.finditer(r'(?:"cmd"|cmd)\s*:\s*',js):
   try:cmd,_=json.JSONDecoder().raw_decode(js[m.end():])
   except (ValueError,TypeError):continue
   if not isinstance(cmd,str) or not any(OLD+'/'+target in cmd for target in TARGETS):continue
   modified=[]
   # Recover exact literal heredoc payloads only.
   for cat in re.finditer(r"cat > ([^\n]+) <<'PY'\n([\s\S]*?)\nPY(?:\n|$)",cmd):
    name=cat.group(1).strip()
    if name not in [OLD+'/'+p for p in TARGETS]:continue
    rel=name[len(OLD)+1:];p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(cat.group(2)+'\n');modified.append(rel)
   # Apply this task's explicit source edits, never unrelated process commands.
   for py in re.finditer(r"python3 - <<'PY'\n([\s\S]*?)\nPY(?:\n|$)",cmd):
    body=py.group(1)
    if not any(OLD+'/'+p in body for p in TARGETS) or '.write_text(' not in body:continue
    # All such known edits are source string operations only. Preserve original
    # quoted path contents inside generated code, remap paths in edit operands.
    tree=ast.parse(body)
    class Paths(ast.NodeTransformer):
     def visit_Constant(self,node):
      if isinstance(node.value,str) and node.value.startswith(OLD+'/') and '\n' not in node.value:
       node.value=str(ROOT)+node.value[len(OLD):]
      return node
    tree=Paths().visit(tree);ast.fix_missing_locations(tree)
    forbidden=(ast.Import,ast.ImportFrom)
    for n in ast.walk(tree):
     if isinstance(n,ast.Import) and any(a.name not in ('ast','json','re','hashlib','shlex') for a in n.names):raise RuntimeError('Unexpected recovery import')
     if isinstance(n,ast.ImportFrom) and n.module!='pathlib':raise RuntimeError('Unexpected recovery import')
     if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('system','run','Popen','connect','unlink','remove','rename'):raise RuntimeError('Unsafe historical operation')
    exec(compile(tree,'recovered-safe-file-edit','exec'),{})
    modified.extend(p for p in TARGETS if OLD+'/'+p in body)
   if modified:receipt.append({'session_file':str(log),'tool_event_ordinal':ordinal,'call_id':v.get('call_id'),'modified_files':modified,'shell_replayed':False})
actual={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in TARGETS}
(ROOT/'preflight/environment_recovery_receipt.json').write_text(json.dumps({'events':receipt,'final_sha256':actual,'source':'Only actual task file-write tool payloads','credential_or_shell_replay':False},indent=2))
print('Recovered environment guard and tests',len(receipt),'actual historical write events')
