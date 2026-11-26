"""Assemble a method's actual source, without starting training."""
import argparse
import re
import shutil
from pathlib import Path
from release_utils import ROOT, load, method_config_path, normalize_method

def copy_overlay(source, target):
    if source.is_dir():shutil.copytree(source,target,dirs_exist_ok=True)

def materialize(method, destination, task=1, order='order2', collection=False):
    method = normalize_method(method)
    config=method_config_path(method)
    if task not in range(1,6):raise ValueError('task must be 1..5')
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    copy_overlay(ROOT/'src/core',destination)
    copy_overlay(ROOT/'src/variants'/method,destination)
    if collection:
        if task==5:raise ValueError('Task5 freezes Probe4; no collection or Probe5 fitting')
        if load(config)['method']['probe_kind'] is None:raise ValueError('Baseline does not collect a probe')
        copy_overlay(ROOT/'src/collection',destination)
    elif task==5:
        copy_overlay(ROOT/'src/ood',destination)
        guard=destination/'utils/online_probe_guard.py'
        if load(config)['method']['probe_kind'] and '_FrozenTask4Guard' not in guard.read_text():
            with guard.open('a',encoding='utf-8') as f:
                f.write('\n_FrozenTask4Guard=OnlineProbeGuard\nclass OnlineProbeGuard(_FrozenTask4Guard):\n    def __init__(self,path,expected_sha256,*,threshold,task_id):\n        if task_id != 5:raise ValueError("This stage requires Task5")\n        super().__init__(path,expected_sha256,threshold=threshold,task_id=4)\n        self.identity.update(probe_training_task_id=4,training_task_id=5,probe_weights_frozen=True)\n')
        if method=='linear_shadow':
            contract=destination/'utils/online_experiment_contract.py'
            text=contract.read_text(encoding='utf-8').replace("if task not in (1,2,3,4) or method != 'shadow':","if task != 5 or method != 'shadow':")
            contract.write_text(text,encoding='utf-8')
    sequence=load(ROOT/'configs/protocol.json')['task_orders'][order]
    names={1:'数字安全',2:'隐私与信息保护',3:'财产、身份及信息欺骗',4:'身体、公共及物质安全',5:'歧视与人际伤害'}
    task_names={i:names[d] for i,d in enumerate(sequence,1)}
    datasets={i:{'benign':f'risk5/task{d}_benign','malicious':f'risk5/task{d}_malicious'} for i,d in enumerate(sequence,1)}
    for relative,name,value in [('utils/continual_data_manifest.py','TASK_DATASETS',datasets),('utils/frozen_continual_schedule.py','TASK_NAMES',task_names)]:
        p=destination/relative
        text=p.read_text(encoding='utf-8')
        text,count=re.subn(r'^'+name+r' = .*$',name+' = '+repr(value),text,flags=re.M)
        if count!=1:raise ValueError('Expected one mapping in '+relative)
        p.write_text(text,encoding='utf-8')
    p=destination/'main_sft.py'
    text=p.read_text(encoding='utf-8').replace('(3,1,4,2,5)',repr(tuple(sequence)))
    p.write_text(text,encoding='utf-8')
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--task',type=int,default=1)
    p.add_argument('--order',choices=['order1','order2'],default='order2')
    p.add_argument('--collection',action='store_true')
    a=p.parse_args()
    print(materialize(a.method,a.destination,a.task,a.order,a.collection))
