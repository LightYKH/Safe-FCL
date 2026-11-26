"""Collect auxiliary trajectories from the current stage's own parent adapter."""
import argparse
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path
from materialize_code import materialize
from release_utils import ROOT,load,write,sha,offline

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();s=a.stage.resolve();cfg=load(s/'inputs/stage.json');t=cfg['task']
    if t==5 or cfg['probe_kind'] is None:raise ValueError('This stage does not collect a probe')
    offline()
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip()
    if not branch:raise ValueError('Use a clean local branch when collecting')
    os.environ['SAFE_FCL_PROBE_EXPECTED_BRANCH']=branch
    overlay=s/'collector'
    if not overlay.exists():materialize(cfg['public_method'],overlay,t,cfg['order'],True)
    sys.path.insert(0,str(overlay))
    from lora_classifier import probe_collection_app as app,probe_collection_plan as cp,probe_collection_runtime as runtime
    from utils.manifest_training_data import validate_manifest_training_provenance
    from lora_classifier.probe_collection_runner import clone_lora_state
    from utils.lora_state_guard import canonical_lora_tensor_sha256
    manifest=s/'manifests'/f'task{t}.data_manifest.json'
    planfile=s/'collection/probe-collection-plan.json'
    if not planfile.exists():
        plan=cp.build_probe_collection_plan(manifest,sha(manifest),cfg['model_directory'],commit,s/'collection',repository_root=ROOT,manifest_provenance_validator=lambda m,r:validate_manifest_training_provenance(m,r,allow_experiment_copy=True))
        cp.write_probe_collection_plan(plan,s/'collection')
    plan=load(planfile)
    cfg['collector_overlay']=str(overlay)
    write(s/'inputs/stage.json',cfg)
    sources=[p for folder in ['code','collector','inputs','scripts','manifests','schedules'] for p in (s/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    sources += [Path(cfg['parent_checkpoint'])/n for n in cfg['parent_files']]
    lock=s/'provenance/COLLECTION_LOCK.json'
    if lock.exists():
        for name,digest in load(lock)['files'].items():
            if sha(name)!=digest:raise ValueError('Collection input changed: '+name)
    else:write(lock,dict(files={str(p):sha(p) for p in sources}))
    original=runtime.initialize_probe_lora_model
    def initialize(*args,**kwargs):
        import torch
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict,set_peft_model_state_dict
        result=original(*args,**kwargs)
        state=load_file(str(Path(cfg['parent_checkpoint'])/'adapter_model.safetensors'),device='cpu')
        if set(state)!=set(result['initial_global_state']):raise ValueError('Parent adapter key mismatch')
        set_peft_model_state_dict(result['model'],state)
        actual=clone_lora_state(get_peft_model_state_dict(result['model']))
        if not all(torch.equal(actual[k],state[k]) for k in state):raise ValueError('Parent adapter load is not exact')
        if canonical_lora_tensor_sha256(actual)!=cfg['parent_tensor_sha256']:raise ValueError('Parent tensor mismatch')
        result['initial_global_state']=actual
        return result
    runtime.initialize_probe_lora_model=initialize
    dependencies=app._dependencies(None)
    dependencies['validate_manifest_provenance']=lambda m,r:validate_manifest_training_provenance(m,r,allow_experiment_copy=True)
    result=app.run_probe_collection(mode='preflight-only' if a.preflight else 'full',plan_path=planfile,plan_sha256=plan['plan_sha256'],manifest_path=manifest,manifest_sha256=sha(manifest),model_directory=cfg['model_directory'],expected_code_commit=commit,expected_bitsandbytes_version='0.44.1',output_root=s/'collection',scratch_root=s/'scratch',local_data_dir=cfg['local_data_dir'],repository_root=ROOT,expected_r0_tensor_sha256=cfg['parent_tensor_sha256'],dependency_provider=lambda:dependencies)
    write(s/'state'/('COLLECTION_PREFLIGHT.json' if a.preflight else 'COLLECTION_COMPLETE.json'),result)

if __name__=='__main__':main()
