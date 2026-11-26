"""Run the retained probe fitting recipe on a prepared collection's features."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from release_utils import load, offline

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();s=a.stage.resolve();cfg=load(s/'inputs/stage.json');t=cfg['task']
    if t==5:raise ValueError('Task5 uses the frozen Probe4; fitting is forbidden')
    if cfg['probe_kind'] is None:raise ValueError('Baseline has no probe')
    if (s/f'task{t}/probe_model.pt').exists():raise FileExistsError('Probe already exists')
    offline();env=os.environ.copy();env['PYTHONPATH']=str(s/'code')
    if not (s/'features/COMPLETE.json').exists():
        subprocess.run([sys.executable,str(s/'scripts/materialize.py'),'--stage',str(s)],env=env,check=True)
    cmd=[sys.executable,str(s/'scripts/fit_mlp.py'),'--stage',str(s)]
    subprocess.run(cmd,env=env,check=True)

if __name__=='__main__':main()
