"""Load hash-bound selected originals, retaining source identity and whole QA."""
import json,hashlib
from pathlib import Path
from functools import lru_cache
from datasets import Dataset
from .data_manifest import compute_content_hash,compute_normalized_prompt_hash
@lru_cache(maxsize=16)
def _read(path,digest):
 p=Path(path)
 if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Risk5 materialized source hash mismatch')
 rows=[json.loads(l) for l in p.open()]
 for r in rows:
  if not r['instruction'].strip() or not r['response'].strip():raise ValueError('Empty QA')
  if r['content_hash']!=compute_content_hash(r['instruction'],r['response']):raise ValueError('Original QA content mismatch')
  if r['normalized_prompt_hash']!=compute_normalized_prompt_hash(r['instruction']):raise ValueError('Prompt identity mismatch')
  if r['full_tokens']>1024 or not r['token_boundary_valid']:raise ValueError('Whole QA exceeds sequence budget')
 return rows
def materialized_rows(source):
 if hashlib.sha256(Path(source['admission_path']).read_bytes()).hexdigest()!=source['admission_sha256']:raise ValueError('Risk5 admission receipt differs')
 return _read(source['path'],source['sha256'])
def load_source(entry):
 if entry['active'] is not True:raise ValueError('Inactive source')
 rows=materialized_rows(entry['source'])
 return Dataset.from_list([{k:r[k] for k in ['instruction','response','source_id','content_hash','normalized_prompt_hash','full_tokens','token_boundary_valid','abc','group_id','partition','task']} for r in rows])
def rebuild(dataset_name,dataset,is_benign,source,records):
 rows=materialized_rows(source);out=[]
 for record in records:
  rank=record['split_rank'];r=rows[record['raw_row_index']]
  for k in ['source_id','content_hash','normalized_prompt_hash','partition']:
   if r[k]!=record[k]:raise ValueError('Risk5 record binding differs: '+k)
  if (r['abc'] in ['A','B'])!=is_benign:raise ValueError('Risk5 A/B/C role mismatch')
  out.append(dict(instruction=r['instruction'],response=r['response'],source_id=r['source_id'],content_hash=r['content_hash'],normalized_prompt_hash=r['normalized_prompt_hash'],raw_row_index=record['raw_row_index'],sample_id=rank,split_rank=rank,dataset_name=dataset_name))
 return Dataset.from_list(out)
