"""No-sample Shadow/module retention using bounded streaming parameter statistics.

No texts, response losses, labels, or historical per-client update records are
read by this module. It consumes the existing Shadow security factors and the
current-round A/B factors only. Novelty is a risk modifier, not a safety label.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from utils.lora_state_guard import canonical_lora_tensor_sha256

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)

def module_pairs(state):
    pairs={}
    for key in sorted(state):
        if '.lora_A.' not in key and '.lora_B.' not in key:raise ValueError(f'unsupported adapter key {key}')
        prefix,tail=key.split('.lora_',1)
        if not prefix.endswith(('.q_proj','.v_proj')):raise ValueError('only q/v adapters supported')
        pairs.setdefault(prefix,{})[tail[0]]=key
    if any(set(v)!=set('AB') for v in pairs.values()):raise ValueError('unpaired A/B factors')
    return pairs

def effective_norm(a,b,a0,b0,scale):
    """Exact Frobenius norm of scale*(B A - B0 A0), via small Gram matrices."""
    left=torch.cat((b,b0),dim=1).double()
    right=torch.cat((a,-a0),dim=0).double()
    square=((left.T@left)*(right@right.T)).sum()
    value=float(square)
    if not math.isfinite(value):raise FloatingPointError('nonfinite BA norm')
    # Gram cancellation can produce tiny negative roundoff at identical factors.
    if value < -1e-8*max(float((left.T@left).abs().sum()*(right@right.T).abs().sum()),1e-30):
        raise FloatingPointError('invalid negative BA squared norm')
    return math.sqrt(max(0.,value))*abs(scale)

def projection(name,dim,width,device):
    gen=torch.Generator(device='cpu')
    gen.manual_seed(int(hashlib.sha256(('2023:'+name).encode()).hexdigest()[:15],16))
    idx=torch.randint(width,(dim,),generator=gen)
    signs=torch.randint(2,(dim,),generator=gen).float()*2-1
    matrix=torch.zeros(width,dim);matrix[idx,torch.arange(dim)]=signs
    return matrix.to(device)

def deviation(stats,lognorm,unit,cfg):
    if stats['observations']<cfg['warmup_observations']:return 0.,dict(warmup=True,magnitude=0.,direction=0.)
    std=max(math.sqrt(max(stats['log_variance'],0.)),cfg['log_std_floor'])
    magnitude=max(0.,(lognorm-stats['log_mean'])/std-cfg['magnitude_z_allowance'])
    mean=np.asarray(stats['direction_mean'],dtype=np.float64)
    length=float(np.linalg.norm(mean))
    if length<cfg['minimum_direction_resultant'] or np.linalg.norm(unit)<1e-12:
        direction=0.
    else:
        cosine=float(np.clip(np.dot(mean/length,unit),-1.,1.))
        direction=max(0.,1.-cosine-cfg['direction_allowance'])
    value=min(cfg['max_deviation'],magnitude+cfg['direction_weight']*direction)
    return value,dict(warmup=False,magnitude=magnitude,direction=direction)

def update_statistics(stats,items,cfg):
    """One EMA update per round; equal client contribution, no stored trajectories."""
    if not items:return
    normmean=float(np.mean([x[0] for x in items]));direction=np.mean([x[1] for x in items],axis=0)
    if stats['observations']==0:
        stats['log_mean']=normmean
        stats['log_variance']=float(np.mean([(x[0]-normmean)**2 for x in items]))
        stats['direction_mean']=direction.tolist()
    else:
        rho=cfg['ema_rate'];old=stats['log_mean']
        stats['log_mean']=(1-rho)*old+rho*normmean
        stats['log_variance']=(1-rho)*(stats['log_variance']+rho*(normmean-old)**2)
        stats['direction_mean']=((1-rho)*np.asarray(stats['direction_mean'])+rho*direction).tolist()
    stats['observations']+=len(items)
    stats['rounds']+=1

def gate_value(security,dev,strength):
    return math.exp(-strength*(1-float(np.clip(security,0.,1.)))*dev)

class ModuleGate:
    def __init__(self,root,task,parent,state,output):
        self.root=Path(root);self.task=int(task);self.output=Path(output)
        self.cfgpath=self.root/'inputs/module_gate_config.json'
        self.cfg=json.loads(self.cfgpath.read_text());self.config_sha=sha(self.cfgpath)
        self.pairs=module_pairs(state);self.projections={};self.last_round=(self.task-1)*100
        self.parent_stats_sha=None
        if self.task==1:
            self.statistics={m:dict(observations=0,rounds=0,log_mean=0.,log_variance=0.,
                direction_mean=[0.]*(self.cfg['sketch_width']**2)) for m in self.pairs}
        elif self.task in (2,3,4,5):
            path=Path(parent)/'MODULE_GATE_STATE.json';obj=json.loads(path.read_text())
            provenance=json.loads((Path(parent)/'ONLINE_CHECKPOINT_PROVENANCE.json').read_text())
            assert sha(path)==provenance['module_gate']['state_sha256']
            assert obj['config_sha256']==self.config_sha and obj['global_round']==(self.task-1)*100
            assert obj['global_adapter_tensor_sha256']==canonical_lora_tensor_sha256(state)
            self.statistics=obj['statistics'];self.parent_stats_sha=sha(path)
            assert set(self.statistics)==set(self.pairs)
        else:raise ValueError('frozen experiment supports Task1/2/3/4/5 only')
        if len(self.pairs)!=self.cfg['expected_modules']:raise ValueError('unexpected module count')
        for name,s in self.statistics.items():
            assert len(s['direction_mean'])==self.cfg['sketch_width']**2
        atomic(self.output/'module_gate_initial.json',dict(task=self.task,config_sha256=self.config_sha,
            parent_stats_sha256=self.parent_stats_sha,modules=len(self.pairs),
            observations={k:v['observations'] for k,v in self.statistics.items()},
            sample_memory=False,statistics_values_bound=len(self.pairs)*(self.cfg['sketch_width']**2+4)))
        print('MODULE_GATE_INITIALIZED='+json.dumps(dict(task=self.task,modules=len(self.pairs),sample_memory=False)),flush=True)

    def features(self,module,local,old,lr):
        pair=self.pairs[module];a=local[pair['A']].float();b=local[pair['B']].float()
        a0=old[pair['A']].float();b0=old[pair['B']].float()
        scale=self.cfg['lora_alpha']/self.cfg['lora_rank']
        norm=effective_norm(a,b,a0,b0,scale)
        if module not in self.projections:
            self.projections[module]=(projection(module+':out',b.shape[0],self.cfg['sketch_width'],b.device),
                                      projection(module+':in',a.shape[1],self.cfg['sketch_width'],a.device))
        outproj,inproj=self.projections[module]
        sketch=scale*((outproj@b)@(a@inproj.T)-(outproj@b0)@(a0@inproj.T))
        vector=sketch.flatten().double().cpu().numpy();length=float(np.linalg.norm(vector))
        if not np.isfinite(vector).all():raise FloatingPointError('nonfinite BA sketch')
        unit=vector/max(length,1e-30)
        # Normalize known stage LR reset/decay, while recording unscaled BA norm.
        lognorm=math.log(max(norm/lr,self.cfg['norm_floor']))
        return lognorm,unit,norm

    @torch.no_grad()
    def aggregate(self,old,original,local_states,clients,effective,actual,lr,round_number):
        if round_number!=self.last_round+1:raise ValueError('nonsequential module-gate rounds')
        if lr<=0:raise ValueError('positive frozen LR required')
        t0=time.time();total=sum(effective[c] for c in clients)
        if total<=0:raise ValueError('empty aggregation weight')
        weights={c:effective[c]/total for c in clients}
        security={c:float(np.clip(effective[c]/actual[c],0.,1.)) for c in clients}
        stats_updates={m:[] for m in self.pairs};gates={c:{} for c in clients};details={}
        for module in self.pairs:
            rows=[];stats=self.statistics[module]
            for c in clients:
                lognorm,unit,norm=self.features(module,local_states[c],old,lr)
                dev,parts=deviation(stats,lognorm,unit,self.cfg)
                gate=gate_value(security[c],dev,self.cfg['strength'])
                gates[c][module]=gate
                eligible=(security[c]>=self.cfg['trusted_security_min'] and dev<=self.cfg['statistics_max_deviation']
                          and gate>=self.cfg['statistics_min_gate'] and norm>self.cfg['zero_update_floor'])
                if eligible:stats_updates[module].append((lognorm,unit))
                rows.append(dict(client=c,security=security[c],weight=weights[c],gate=gate,deviation=dev,
                                 effective_update_norm=norm,statistics_eligible=eligible,**parts))
            details[module]=rows
        out={}
        for module,pair in self.pairs.items():
            for key in pair.values():
                # Preserve exact baseline arithmetic in neutral/warmup modules.
                if all(gates[c][module]==1. for c in clients):out[key]=original[key]
                else:
                    delta=torch.zeros_like(old[key],dtype=torch.float32)
                    for c in clients:
                        delta.add_(local_states[c][key].float()-old[key].float(),alpha=weights[c]*gates[c][module])
                    out[key]=(old[key].float()+delta).to(old[key].dtype)
        if not all(torch.isfinite(x).all() for x in out.values()):raise FloatingPointError('nonfinite gated adapter')
        # Update reference only AFTER scoring every client against the old state.
        for m,items in stats_updates.items():update_statistics(self.statistics[m],items,self.cfg)
        self.last_round=round_number
        retained={m:sum(weights[c]*gates[c][m] for c in clients) for m in self.pairs}
        record=dict(round=round_number,task=self.task,decision='module_gated' if min(retained.values())<1-1e-8 else 'baseline_equivalent',
            clients=list(clients),learning_rate=lr,modules=details,retained_mass=retained,
            mean_retained_mass=float(np.mean(list(retained.values()))),minimum_retained_mass=min(retained.values()),
            trusted_module_observations=sum(len(v) for v in stats_updates.values()),seconds=time.time()-t0,
            samples_read=0,historical_trajectories_stored=0)
        self.publish(record)
        return out

    def skip(self,round_number):
        if round_number!=self.last_round+1:raise ValueError('nonsequential skipped round')
        self.last_round=round_number
        self.publish(dict(round=round_number,task=self.task,decision='original_shadow_skip',samples_read=0))

    def publish(self,record):
        with (self.output/'module_gate_rounds.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        atomic(self.output/'module_gate_latest.json',record)
        print('MODULE_GATE='+json.dumps({k:v for k,v in record.items() if k not in ('modules','retained_mass')}),flush=True)

    def save_checkpoint(self,directory,state,round_number):
        assert self.last_round==round_number
        path=Path(directory)/'MODULE_GATE_STATE.json'
        if path.exists():raise FileExistsError(path)
        atomic(path,dict(schema='safe-fcl.module-statistics.v1',config_sha256=self.config_sha,
            global_round=round_number,task=self.task,global_adapter_tensor_sha256=canonical_lora_tensor_sha256(state),
            parent_stats_sha256=self.parent_stats_sha,statistics=self.statistics,
            per_task_arrays=False,raw_samples=False,raw_historical_updates=False))
        return dict(state_sha256=sha(path),config_sha256=self.config_sha,modules=len(self.pairs),sample_memory=False)
