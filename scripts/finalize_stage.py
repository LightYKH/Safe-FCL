import argparse
from pathlib import Path
from prepare_stage import finalize

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Bind the newly fitted probe and freeze the stage.')
    p.add_argument('--stage',type=Path,required=True)
    finalize(p.parse_args().stage)
