"""Prepare a fresh, hash-bound stage from locally obtained model/data artifacts.

Input manifests retain their native integrity and provenance checks. This command
does not synthesize missing data, bypass licenses, or reuse old training locks.
"""
import argparse
import shutil
import sys
from pathlib import Path
from materialize_code import materialize, copy_overlay
from release_utils import (
    ROOT, assert_profile, freeze, load, method_config_path, normalize_method, offline, sha, write,
)

def finalize(stage):
    import torch
    stage=Path(stage).resolve()
    cfg=load(stage/'inputs/stage.json')
    args=load(stage/'inputs/train_args.json')
    if cfg['probe_kind']:
        probe=stage/f"task{cfg['task']}/probe_model.pt"
        artifact=torch.load(probe,map_location='cpu',weights_only=True)
        expected_task=min(cfg['task'],4)
        if artifact['task_id']!=expected_task:raise ValueError('Wrong probe stage')
        args.update(prefilter_classifier_path=str(probe),prefilter_classifier_sha256=sha(probe))
        if cfg['probe_kind']=='mlp':
            source=Path(artifact['source_checkpoint'])
            if sha(source)!=artifact['source_checkpoint_sha256']:raise ValueError('Probe training checkpoint identity mismatch')
            write(stage/'provenance/REPAIR_CONTRACT.json',dict(probe_training=cfg['probe_training'],probe_architecture=artifact['architecture'],orthogonal_enabled=cfg['orthogonal'],probe_checkpoints={str(expected_task):dict(sha256=sha(source))}))
    write(stage/'inputs/train_args.json',args)
    freeze(stage)
    print('Stage frozen:',stage)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True)
    p.add_argument('--task',type=int,required=True,choices=range(1,6))
    p.add_argument('--order',choices=['order1','order2'],default='order2')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--probe',type=Path)
    p.add_argument('--previous-stage',type=Path)
    p.add_argument('--data-dir',type=Path,required=True)
    a=p.parse_args()
    method=normalize_method(a.method)
    offline()
    stage=a.run_dir.resolve()
    if stage.exists():raise FileExistsError('Use a fresh run directory: '+str(stage))
    spec=load(method_config_path(method))
    cfg=spec['method'].copy();cfg.update(task=a.task,stage_id=a.task,order=a.order,public_method=method)
    sequence=load(ROOT/'configs/protocol.json')['task_orders'][a.order]
    cfg['original_task_id']=sequence[a.task-1]
    for path in [a.model,a.parent,a.manifest,a.schedule,a.data_dir]:
        if not path.exists():raise FileNotFoundError(path)
    if a.task>1 and cfg['probe_kind'] and a.previous_stage is None:
        raise ValueError('Continual probes require --previous-stage')
    materialize(method,stage/'code',a.task,a.order)
    for sub in ['inputs','manifests','schedules','provenance','scripts','state','training','features','probe_runs','collection','scratch',f'task{a.task}']:
        (stage/sub).mkdir(parents=True,exist_ok=True)
    copy_overlay(ROOT/'src/probe_models',stage/'mlp_code')
    copy_overlay(ROOT/'src/probe_training',stage/'scripts')
    shutil.copy2(ROOT/'scripts/probe_common.py',stage/'scripts/common.py')
    sys.path.insert(0,str(stage/'code'))
    from utils.formal_method_chain import adapter_tensor_sha256
    parent=a.parent.resolve()
    parent_names=['adapter_config.json','adapter_model.safetensors']
    if a.task>1:parent_names+=['ONLINE_CHECKPOINT_PROVENANCE.json']
    if cfg['gate'] and a.task>1:parent_names+=['MODULE_GATE_STATE.json']
    cfg.update(parent_checkpoint=str(parent),parent_files={n:sha(parent/n) for n in parent_names},parent_tensor_sha256=adapter_tensor_sha256(parent/'adapter_model.safetensors'),model_directory=str(a.model.resolve()),local_data_dir=str(a.data_dir.resolve()),previous_stage=str(a.previous_stage.resolve()) if a.previous_stage else None,previous_probe=None,previous_probe_sha256=None,feature_sources={str(a.task):str(stage/'features')})
    if a.previous_stage:
        prev=a.previous_stage.resolve();old=load(prev/'inputs/stage.json')
        if old['task']!=a.task-1 or normalize_method(old['public_method'])!=method or old['order']!=a.order:raise ValueError('Previous stage lineage mismatch')
        probe=prev/f'task{a.task-1}/probe_model.pt'
        cfg.update(previous_probe=str(probe),previous_probe_sha256=sha(probe),feature_sources=dict(old['feature_sources'],**{str(a.task):str(stage/'features')}))
    manifest=stage/'manifests'/f'task{a.task}.data_manifest.json'
    schedule=stage/'schedules'/f'task{a.task}.schedule.json'
    shutil.copy2(a.manifest,manifest);shutil.copy2(a.schedule,schedule)
    for suffix in ['.sha256','.audit.json','.split.json']:
        extra=Path(str(a.manifest)+suffix)
        if extra.exists():shutil.copy2(extra,Path(str(manifest)+suffix))
    args=spec['training'].copy()
    args.update(model_name_or_path=str(a.model.resolve()),local_data_dir=str(a.data_dir.resolve()),formal_task_id=a.task,continual_round_offset=(a.task-1)*100,initial_lora_checkpoint=str(parent),initial_lora_config_sha256=cfg['parent_files']['adapter_config.json'],initial_lora_adapter_sha256=cfg['parent_files']['adapter_model.safetensors'],formal_expected_stage_base_tensor_sha256=cfg['parent_tensor_sha256'],data_manifest_path=str(manifest),data_manifest_sha256=sha(manifest),continual_schedule_path=str(schedule),continual_schedule_sha256=sha(schedule),benign_dataset_names=f"risk5/task{cfg['original_task_id']}_benign",malicious_dataset_names=f"risk5/task{cfg['original_task_id']}_malicious",output_dir=str(stage/'training/online_probe'/f"{cfg['method']}_level"/f'task{a.task}_r{(a.task-1)*100+1:03d}_r{a.task*100:03d}'))
    assert_profile(args)
    write(stage/'inputs/stage.json',cfg)
    write(stage/'inputs/train_args.json',args)
    if cfg['gate']:shutil.copy2(method_config_path(method, 'gates'),stage/'inputs/module_gate_config.json')
    if a.probe:shutil.copy2(a.probe,stage/f'task{a.task}/probe_model.pt')
    if a.task==5 and cfg['probe_kind'] and not a.probe:shutil.copy2(cfg['previous_probe'],stage/'task5/probe_model.pt')
    if cfg['probe_kind'] is None or (stage/f'task{a.task}/probe_model.pt').exists():finalize(stage)
    else:print('Stage prepared. Collect and fit its probe, then run finalize_stage.py:',stage)

if __name__=='__main__':main()
