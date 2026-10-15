"""Lossless disk-backed JSON arrays and bounded mutable recording tail.

No physics/contacts/rows are changed. Every row persists; only residency changes.
"""
from __future__ import annotations
from collections import deque
import copy,hashlib,json,os
from pathlib import Path

MiB=1024**2

def stream_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(MiB),b''):h.update(b)
 return h.hexdigest()

class LazyJSONList(list):
 def __init__(self,path,*,index_path=None,max_record_bytes=8*MiB,chunk_bytes=MiB,_offsets=None,_selection=None,_overrides=None):
  self.path=Path(path);self.max_record_bytes=max_record_bytes;self.chunk_bytes=chunk_bytes
  self.offsets=_offsets if _offsets is not None else self._index()
  self.selection=range(len(self.offsets)) if _selection is None else _selection
  self.overrides={} if _overrides is None else _overrides
 def _index(self):
  decoder=json.JSONDecoder();offsets=[];buf='';base=0;phase='open';eof=False
  with self.path.open(encoding='utf-8') as f:
   while True:
    if not buf and not eof:
     extra=f.read(self.chunk_bytes);buf+=extra;eof=not extra
    consumed=len(buf)-len(buf.lstrip());base+=len(buf[:consumed].encode('utf-8'));buf=buf[consumed:]
    if phase=='open':
     if not buf:raise ValueError('Truncated JSON array header')
     if buf[0]!='[':raise ValueError('Expected JSON array')
     base+=1;buf=buf[1:];phase='first';continue
    if phase=='end':
     if buf.strip():raise ValueError('Trailing JSON array data')
     if eof:return offsets
     extra=f.read(self.chunk_bytes);buf+=extra;eof=not extra;continue
    if not buf:
     if eof:raise ValueError('Truncated JSON array')
     continue
    if phase=='separator':
     if buf[0]==']':base+=1;buf=buf[1:];phase='end';continue
     if buf[0]!=',':raise ValueError('Missing JSON row separator')
     base+=1;buf=buf[1:];phase='row';continue
    if buf[0]==']':
     if phase!='first':raise ValueError('Trailing JSON comma')
     base+=1;buf=buf[1:];phase='end';continue
    while True:
     try:row,end=decoder.raw_decode(buf);break
     except json.JSONDecodeError as error:
      if eof:raise ValueError('Truncated or invalid JSON row') from error
      if len(buf.encode('utf-8'))>self.max_record_bytes:raise ValueError('JSON row exceeds fixed memory budget')
      extra=f.read(self.chunk_bytes);buf+=extra;eof=not extra
    raw=buf[:end].encode('utf-8')
    if len(raw)>self.max_record_bytes:raise ValueError('JSON row exceeds fixed memory budget')
    if not isinstance(row,dict):raise ValueError('Recorded JSON rows must be dictionaries')
    offsets.append((base,len(raw)));base+=len(raw);buf=buf[end:];phase='separator'
 def __len__(self):return len(self.selection)
 def __bool__(self):return len(self)>0
 def __iter__(self):
  with self.path.open('rb') as f:
   for idx in self.selection:
    if idx in self.overrides:yield self.overrides[idx];continue
    start,size=self.offsets[idx];f.seek(start);payload=f.read(size)
    if len(payload)!=size:raise ValueError('Indexed JSON row changed/truncated')
    yield json.loads(payload)
 def __getitem__(self,index):
  if isinstance(index,slice):return LazyJSONList(self.path,max_record_bytes=self.max_record_bytes,chunk_bytes=self.chunk_bytes,_offsets=self.offsets,_selection=self.selection[index],_overrides=self.overrides)
  real=self.selection[index]
  if real in self.overrides:return self.overrides[real]
  start,size=self.offsets[real]
  with self.path.open('rb') as f:f.seek(start);payload=f.read(size)
  if len(payload)!=size:raise ValueError('Indexed JSON row changed/truncated')
  return json.loads(payload)
 def __setitem__(self,index,value):
  if isinstance(index,slice):raise ValueError('Row range replacement forbidden')
  if not isinstance(value,dict):raise ValueError('Recorded row replacement must remain a dictionary')
  self.overrides[self.selection[index]]=value

def copy_with_overrides(samples,overrides=None):
 if isinstance(samples,LazyJSONList):
  return LazyJSONList(samples.path,max_record_bytes=samples.max_record_bytes,chunk_bytes=samples.chunk_bytes,_offsets=samples.offsets,_selection=samples.selection,_overrides={**copy.deepcopy(samples.overrides),**copy.deepcopy(overrides or {})})
 raise TypeError('Expected bounded immutable row sequence')

class PendingTraceList(list):
 def __init__(self,path,*,tail_rows=50,default=None):
  self.path=Path(path);self.part=self.path.with_name(self.path.name+'.part')
  if self.path.exists() or self.part.exists():raise FileExistsError('Refusing prior trace overwrite')
  self.stream=self.part.open('x',encoding='utf-8');self.stream.write('[')
  self.tail=deque(maxlen=tail_rows);self.count=0;self.written=0;self.pending=None;self.sealed=False;self.default=default
 def _flush_pending(self):
  if self.pending is None:return
  if self.written:self.stream.write(', ')
  self.stream.write(json.dumps(self.pending,default=self.default));self.stream.flush()
  self.written+=1;self.pending=None
 def append(self,row):
  if self.sealed:raise RuntimeError('No steps allowed after trace sealed')
  if not isinstance(row,dict):raise TypeError('Recorded rows must remain dictionaries')
  self._flush_pending();self.pending=row;self.tail.append(row);self.count+=1
 def __len__(self):return self.count
 def __bool__(self):return self.count>0
 def __getitem__(self,index):
  if self.sealed:return LazyJSONList(self.path)[index]
  if isinstance(index,slice):
   start,stop,stride=index.indices(self.count)
   if start<self.count-len(self.tail):raise IndexError('Requested rows outside bounded mutable tail')
   return [self.tail[i-(self.count-len(self.tail))] for i in range(start,stop,stride)]
  index=index if index>=0 else self.count+index
  if index<self.count-len(self.tail) or index>=self.count:raise IndexError('Requested row outside bounded mutable tail')
  return self.tail[index-(self.count-len(self.tail))]
 def flush_json(self,path=None):
  if path is not None and Path(path)!=self.path:raise ValueError('Cannot relabel recorded trace path')
  if not self.sealed:
   self._flush_pending();self.stream.write(']');self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.part.rename(self.path);self.sealed=True
  return LazyJSONList(self.path)
 def __iter__(self):
  if not self.sealed:raise RuntimeError('Full trace iteration requires a completed seal')
  return iter(LazyJSONList(self.path))

class DiskJSONAppender(PendingTraceList):
 def close(self):return self.flush_json()


def write_json_stream(path,data,*,indent=2,default=None,allow_nan=False):
 path=Path(path);part=path.with_name(path.name+'.writing')
 if path.exists() or part.exists():raise FileExistsError('Refusing prior JSON audit overwrite')
 encoder=json.JSONEncoder(indent=indent,default=default,allow_nan=allow_nan)
 try:
  with part.open('x',encoding='utf-8') as f:
   buffer=[];size=0
   for chunk in encoder.iterencode(data):
    # JSONEncoder may emit an entire large string in one chunk. Its default
    # ensure_ascii=True guarantees these pieces are ASCII bytes on disk.
    for start in range(0,len(chunk),16384):
     piece=chunk[start:start+16384]
     if size+len(piece)>65536:f.write(''.join(buffer));buffer=[];size=0
     buffer.append(piece);size+=len(piece)
     if size==65536:f.write(''.join(buffer));buffer=[];size=0
   if buffer:f.write(''.join(buffer))
   f.flush();os.fsync(f.fileno())
  part.rename(path)
 except BaseException:raise
 return path
