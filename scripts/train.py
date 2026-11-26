"""Validate or execute a prepared stage using the fixed 4x4/512 profile."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from release_utils import assert_profile, load, offline, verify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--preflight',action='store_true',help='CPU input and lineage validation; no training')
    a=p.parse_args();stage=a.stage.resolve()
    offline();args=load(stage/'inputs/train_args.json');assert_profile(args);verify(stage)
    sys.path.insert(0,str(stage/'code'))
    from config import ScriptArguments, FedArguments
    from dataclasses import fields
    from utils.online_experiment_contract import validate_online_arguments
    from utils.frozen_continual_schedule import load_frozen_schedule
    from utils.manifest_training_data import load_training_manifest,validate_manifest_training_provenance,validate_manifest_training_arguments
    script_fields={f.name for f in fields(ScriptArguments)}
    fed_fields={f.name for f in fields(FedArguments)}
    script=ScriptArguments(**{k:v for k,v in args.items() if k in script_fields})
    fed=FedArguments(**{k:([v] if k in ['benign_num_clients','malicious_num_clients','benign_dataset_names','malicious_dataset_names'] and not isinstance(v,list) else v) for k,v in args.items() if k in fed_fields})
    validate_online_arguments(script,fed)
    load_frozen_schedule(args['continual_schedule_path'],args['continual_schedule_sha256'],task_id=args['formal_task_id'],continual_round_offset=args['continual_round_offset'])
    manifest=load_training_manifest(args['data_manifest_path'],args['data_manifest_sha256'])
    validate_manifest_training_arguments(manifest,script,fed)
    validate_manifest_training_provenance(manifest,stage/'code',allow_experiment_copy=True)
    if script.prefilter_enable:
        from utils.online_probe_guard import OnlineProbeGuard
        OnlineProbeGuard(script.prefilter_classifier_path,script.prefilter_classifier_sha256,threshold=script.prefilter_threshold,task_id=script.formal_task_id)
    print('Preflight passed: batch=4, accumulation=4, seq_length=512; inputs and lineage validated.',flush=True)
    if a.preflight:return
    cli=[]
    for key,value in args.items():
        if value is None:continue
        cli.extend(['--'+key,str(value).lower() if isinstance(value,bool) else str(value)])
    env=os.environ.copy();env['PYTHONPATH']=str(stage/'code')
    result=subprocess.run([sys.executable,str(stage/'code/main_sft.py'),*cli],cwd=stage/'code',env=env)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
