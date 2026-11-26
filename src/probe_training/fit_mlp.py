"""Continue the uncompressed orthogonal MLP on the exact linear 90/10 update split."""
import argparse,copy,time,csv
from common import *
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args();s=a.stage;cfg=load(s/'inputs/stage.json');t=cfg['task'];verify(s/'provenance/COLLECTION_LOCK.json')
sys.path.insert(0,str(ROOT/'mlp_code'))
import experiment as base
from models import make_model
from orthogonal import extend_bases,Projector,projected_adamw,coverage
from probe_artifact import initialize_probe
base.ROOT=s;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
device='cuda' if torch.cuda.is_available() else 'cpu';data={};raw=None
for task in range(1,t+1):
 f=Path(cfg['feature_sources'][str(task)]);audit=load(f/f'task{task}.splits.json');assert sha(f/f'task{task}.pt')==audit['features_sha256']
 raw=torch.load(f/f'task{task}.pt',map_location='cpu',weights_only=False);ids=raw['record_ids'];index={rid:i for i,rid in enumerate(ids)};data[task]={}
 assert len(ids)==300 and set(audit['split_record_ids']['train']).isdisjoint(audit['split_record_ids']['validation'])
 for split,n in [('train',270),('validation',30)]:
  chosen=audit['split_record_ids'][split];assert len(chosen)==n;ix=torch.tensor([index[rid] for rid in chosen]);y=raw['labels'][ix].float().to(device);assert set(y.cpu().tolist())=={0.,1.}
  data[task][split]=dict(x=raw['features'][ix].float().to(device),y=y,ids=chosen)
base.seed_all(2023);model=make_model('mlp1').to(device);artifact=initialize_probe(model,cfg,raw['feature_metadata']);folder=s/'probe_runs'/f'task{t}';folder.mkdir(exist_ok=True);bases={};statistics={}
if cfg['orthogonal'] and t>1:
 if t>=4:
  from basis_memory import build_bases
  reference=copy.deepcopy(model).cpu()
  bases,statistics=build_bases(reference,{str(i):cfg['feature_sources'][str(i)] for i in range(1,t)},rank_caps=(128,32),expected_train_count=270,expected_class_count=None)
  assert statistics['total_train_records']==(t-1)*270 and statistics['history_split']=='train';del reference
 else:
  if t>2:bases=torch.load(Path(cfg['previous_stage'])/'probe_runs'/f'task{t-1}'/'basis.pt',map_location=device,weights_only=False)['bases']
  bases,statistics=extend_bases(model,data[t-1]['train']['x'],bases,t-1)
 torch.save(dict(bases=bases,statistics=statistics,through_task=t-1,reference_checkpoint=cfg['previous_probe'],reference_checkpoint_sha256=cfg['previous_probe_sha256'],newly_used_training_record_ids=data[t-1]['train']['ids']),folder/'basis.pt');write(folder/'basis_statistics.json',statistics)
recipe=cfg['probe_training'];assert recipe['positive_class_weight'] is True and recipe['extra_current_resampling'] is False
base.seed_all(2023+1000*t);generator=torch.Generator().manual_seed(2023+1000*t)
train,val=data[t]['train'],data[t]['validation'];pos_weight=(train['y'].eq(0).sum()/train['y'].eq(1).sum()) if recipe['positive_class_weight'] else None
criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
@torch.no_grad()
def loss(d):
 model.eval();return float(criterion(model(d['x']),d['y']))
def fit():
 optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay'])
 best=float('inf');bad=0;updates=0;selected=None;start=time.monotonic();batch=recipe['batch_size']
 with (folder/'epochs.jsonl').open('w') as log:
  for epoch in range(1,recipe['epochs']+1):
   model.train();running=0.;batches=0;samples_seen=0
   for ix in torch.randperm(len(train['y']),generator=generator).split(batch):
    x,y=train['x'][ix],train['y'][ix];samples_seen+=len(ix)
    optimizer.zero_grad(set_to_none=True);value=criterion(model(x),y);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['grad_clip'],error_if_nonfinite=True);optimizer.step();updates+=1;running+=float(value.detach());batches+=1
   validation=loss(val);train_eval=loss(train)
   if validation<best-recipe['min_delta']:
    best=validation;bad=0;selected=dict(model_state_dict=base.cpu_state(model),optimizer_state=copy.deepcopy(optimizer.state_dict()),rng_state=dict(torch=torch.get_rng_state(),generator=generator.get_state(),cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None),epoch=epoch)
   else:bad+=1
   assert samples_seen==270, 'No extra resampling is allowed'
   row=dict(epoch=epoch,samples_seen=samples_seen,positive_class_weight=float(pos_weight),train_bce=running/batches,train_eval_bce=train_eval,validation_bce=validation,best_epoch=selected['epoch'],best_validation_bce=best,updates=updates);log.write(json.dumps(row)+'\n');log.flush()
   if epoch==1 or epoch%10==0:print('PROBE_EPOCH',t,row,flush=True)
   if bad>=recipe['patience']:break
 torch.save(dict(model_state_dict=base.cpu_state(model),epoch=epoch),folder/'last_epoch.pt');model.load_state_dict(selected['model_state_dict']);assert abs(loss(val)-best)<1e-6
 info=dict(recipe=recipe,positive_class_weight=float(pos_weight),resampling=False,examples_per_epoch=270,epochs_completed=epoch,selected_epoch=selected['epoch'],best_epoch=selected['epoch'],best_validation_bce=best,restored_validation_bce=loss(val),updates=updates,elapsed_seconds=time.monotonic()-start,early_stopping=True,stop_reason='patience' if bad>=recipe['patience'] else 'epoch_cap',selection='current-task validation BCE only',train_samples=270,validation_samples=30,test_samples=0,optimizer_between_tasks='reset',training_record_ids=train['ids'],replay=False)
 return info,selected
if bases:
 with (folder/'projection_updates.jsonl').open('w') as log:
  projector=Projector(model,bases,lambda row:log.write(json.dumps(row)+'\n'))
  with projected_adamw(projector):info,selected=fit()
  info['protection']=projector.summary()
  assert info['protection']['steps']>0, 'Orthogonal final-update projection must actually execute'
  if t>=4:info['protection'].update(method='svd_basis_orthogonal',old_hard_projection=True,cumulative_risk_constraint=False,rank_caps=[128,32],history_split='train',task_class_balance='equal_task_equal_class')
else:info,selected=fit()
cp=folder/'probe.pt';torch.save(dict(model_state_dict=base.cpu_state(model),optimizer_state=selected['optimizer_state'],rng_state=selected['rng_state'],architecture='mlp1',seed=2023,task=t,method='svd_basis_orthogonal' if t>=4 else 'orthogonal',training=info,previous_checkpoint=cfg['previous_probe'],previous_checkpoint_sha256=cfg['previous_probe_sha256'],stage_config_sha256=sha(s/'inputs/stage.json')),cp)
artifact.update(probe_training_recipe=copy.deepcopy(recipe),positive_class_weight=float(pos_weight),extra_current_resampling=False,model_state_dict=base.cpu_state(model),task_id=t,source_checkpoint=str(cp),source_checkpoint_sha256=sha(cp),source_previous_checkpoint_sha256=cfg['previous_probe_sha256'],selected_epoch=info['selected_epoch'],epochs_completed=info['epochs_completed'])
if t>=4:artifact.update(continual_protection='svd_basis_orthogonal',protection_basis_compressed=True,rank_caps=[128,32],cumulative_risk_constraint=False)
torch.save(artifact,s/f'task{t}/probe_model.pt');metrics={};model.eval()
with torch.no_grad():
 for task,sets in data.items():metrics[str(task)]={split:base.metric(model(d['x']).cpu(),d['y'].cpu()) for split,d in sets.items()}
write(folder/'metrics.json',dict(stage=t,training=info,metrics=metrics,parameter_coverage=coverage(model)))
write(s/'state/PROBE_COMPLETE.json',dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),probe_sha256=sha(s/f'task{t}/probe_model.pt'),previous_probe_sha256=cfg['previous_probe_sha256'],training=info))
print('PROBE COMPLETE',t,info['selected_epoch'],info['epochs_completed'],flush=True)
