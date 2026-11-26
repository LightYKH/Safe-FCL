import argparse,collections,itertools
from common import *
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args();s=a.stage;cfg=load(s/'inputs/stage.json');t=cfg['task'];verify(s/'provenance/COLLECTION_LOCK.json')
sys.path.insert(0,cfg['collector_overlay'])
import numpy as np,torch
from lora_classifier.probe_collection_app import validate_probe_collection_run_summary
from lora_classifier.probe_delta_store import load_and_validate_probe_delta_round_shard
from lora_classifier.lora_classifier_common import build_feature_matrix,apply_normalization
summary=load(s/'collection/probe-collection-run.json');validate_probe_collection_run_summary(summary)
assert (s/'collection/COMPLETE').read_text().strip()==summary['run_sha256']
assert summary['collection_r0_tensor_sha256']==cfg['parent_tensor_sha256']
spec=summary['lora_spec'];keys=spec['ordered_keys'];sizes={k:int(np.prod(spec['tensors'][k]['shape'])) for k in keys};assert sum(sizes.values())==40960
features=[];labels=[];ids=[];groups=[];steps=[];text_sources=collections.defaultdict(set);splits=None
lookup={(g['round_index'],g['client_id']):split for split,gs in splits.items() for g in gs} if splits else {}
selected=collections.defaultdict(list)
for r in range(cfg['collection_rounds']):
 folder=s/'collection'/f'round-{r:03d}';assert sha(folder/'index.json')==summary['rounds'][r]['index_sha256']
 raw=load_and_validate_probe_delta_round_shard(folder,expected_round_index=r,expected_manifest_sha256=summary['manifest_sha256'],expected_code_commit=summary['code_commit'],expected_lora_spec=spec)
 for rec in raw['records']:
  m=rec['metadata'];group=(r,m['client_id']);assert m['label']==int(m['client_id']>=5)
  if cfg['method']=='shadow':
   assert m['delta_base_tensor_sha256']==cfg['parent_tensor_sha256'] and m['learning_rate']==5e-5
  features.append(apply_normalization(build_feature_matrix([rec['delta_tensors']],keys,sizes),'l2')[0]);labels.append(m['label']);ids.append(m['record_id']);groups.append(group);steps.append(m['step_index'])
  if splits:
   split=lookup[group];selected[split].append(m['record_id']);text_sources[split].update(x['source_id'] for x in m['step_samples'])
assert len(ids)==len(set(ids))==cfg['collection_rounds']*30
x=torch.from_numpy(np.stack(features));assert x.shape[1]==40960 and torch.isfinite(x).all();assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=1e-5)
meta=dict(ordered_keys=keys,key_sizes=sizes,normalize='l2',filter_policy='first_B',layer_idx=0,threshold=.8,feature_dim=40960)
target=s/'features'/f'task{t}.pt';assert not target.exists()
torch.save(dict(features=x,labels=torch.tensor(labels),record_ids=ids,groups=groups,steps=steps,feature_metadata=meta,source=str(s/'collection')),target)
from sklearn.model_selection import train_test_split
train_ix,val_ix=train_test_split(np.arange(len(ids)),test_size=.1,shuffle=True,random_state=2023,stratify=np.asarray(labels))
selected={'train':[ids[i] for i in train_ix],'validation':[ids[i] for i in val_ix],'test':[]}
assert len(selected['train'])==270 and len(selected['validation'])==30
assert not set(selected['train'])&set(selected['validation'])
audit=dict(task=t,features_sha256=sha(target),collection_run_sha256=summary['run_sha256'],records=len(ids),benign=labels.count(0),malicious=labels.count(1),split_record_ids=dict(selected),source_text_overlap_counts={a+'__'+b:len(text_sources[a]&text_sources[b]) for a,b in itertools.combinations(text_sources,2)},split_recipe='exact linear stratified update-level 90/10 seed2023',scope='Matches linear update-level split; clients/texts/collection trajectory may be shared; no held-out probe test set')

write(s/'features'/f'task{t}.splits.json',audit);write(s/'features/COMPLETE.json',dict(tasks={str(t):audit}))
print('MATERIALIZED',t,audit['records'],flush=True)
