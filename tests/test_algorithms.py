"""Small independent numerical oracles; no real model/data or GPU required."""
import copy
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','DATASETS_OFFLINE']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/core'))
import torch

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

class AlgorithmTests(unittest.TestCase):
    def test_lora_effective_norm_matches_dense(self):
        gate=module('gate_test',ROOT/'src/core/federated_learning/module_gate.py')
        gen=torch.Generator().manual_seed(7)
        a,a0=[torch.randn(3,7,generator=gen,dtype=torch.float64) for _ in range(2)]
        b,b0=[torch.randn(5,3,generator=gen,dtype=torch.float64) for _ in range(2)]
        expected=float((2*(b@a-b0@a0)).norm())
        self.assertAlmostEqual(gate.effective_norm(a,b,a0,b0,2),expected,places=10)
        self.assertLess(gate.effective_norm(a,b,a,b,2),1e-6)

    def test_final_adamw_update_projection_includes_bias(self):
        ortho=module('ortho_test',ROOT/'src/probe_models/orthogonal.py')
        torch.manual_seed(3)
        model=torch.nn.Sequential(torch.nn.Linear(3,2)).double()
        u=torch.tensor([[1.],[0.],[0.],[1.]],dtype=torch.float64)/(2**.5)
        initial=ortho.packed(ortho.blocks(model)['0']).clone()
        projector=ortho.Projector(model,{'0':u},None)
        original=torch.optim.AdamW
        with ortho.projected_adamw(projector):
            optimizer=torch.optim.AdamW(model.parameters(),lr=.03,weight_decay=.1)
            for _ in range(3):
                optimizer.zero_grad();model(torch.randn(5,3,dtype=torch.float64)).square().mean().backward();optimizer.step()
        delta=ortho.packed(ortho.blocks(model)['0'])-initial
        self.assertLess(float((delta@u).norm()),1e-10)
        self.assertGreater(float(delta.norm()),1e-5)
        self.assertIs(torch.optim.AdamW,original)

    def test_fedavg_weighted_oracle(self):
        fed=module('fed_test',ROOT/'src/core/federated_learning/fed_global.py')
        out=fed.global_aggregate(SimpleNamespace(fed_alg='fedavg'),{'w':torch.zeros(2)},[{'w':torch.tensor([1.,3.])},{'w':torch.tensor([5.,7.])}],[1,3],[0,1],0)
        if isinstance(out,tuple):out=out[0]
        self.assertTrue(torch.equal(out['w'],torch.tensor([4.,6.])))

    def test_main_and_collection_bind_four_microbatches(self):
        from trl import SFTTrainer
        from federated_learning.formal_sft_trainer import FormalFederatedSFTTrainer, _STEP_AUDIT_KEY
        collection=module('collection_trainer_test',ROOT/'src/collection/lora_classifier/probe_sft_trainer.py')
        ProbeSFTTrainer=collection.ProbeSFTTrainer
        _STEP_SAMPLES_KEY=collection._STEP_SAMPLES_KEY
        original=SFTTrainer.training_step
        def step(self,model,inputs):
            self.assert_size=inputs['x'].shape[0]
            loss=(model(inputs['x'])-inputs['y']).square().mean()/4
            loss.backward();return loss.detach()
        SFTTrainer.training_step=step
        try:
            for kind,cls,key in [('main',FormalFederatedSFTTrainer,_STEP_AUDIT_KEY),('collection',ProbeSFTTrainer,_STEP_SAMPLES_KEY)]:
                class Collector:
                    def __init__(self):self.records=[]
                    def validate_step_samples(self,x):return copy.deepcopy(x)
                    def record_optimizer_step(self,state,samples):self.records.append(copy.deepcopy(samples))
                    def abort(self):pass
                torch.manual_seed(5);model=torch.nn.Linear(3,1,bias=False)
                optimizer=torch.optim.SGD(model.parameters(),lr=.01)
                x=torch.randn(16,3);y=torch.randn(16,1)
                reference=copy.deepcopy(model)
                ropt=torch.optim.SGD(reference.parameters(),lr=.01)
                (reference(x)-y).square().mean().backward();ropt.step()
                trainer=object.__new__(cls);trainer._optimizer_callback_count=0
                trainer.accelerator=SimpleNamespace(optimizer_step_was_skipped=False)
                if kind=='main':trainer._pending_audit=None;trainer._audit_records=[];trainer.delta_tracker=None
                else:
                    trainer.probe_collector=Collector();trainer._pending_step_samples=None;trainer._optimizer_skip_count=0;trainer._probe_training_failed=False
                    trainer._probe_state_extractor=lambda m:{'weight':m.weight.detach().clone()}
                for i in range(4):
                    samples=[dict(sample_id=j) for j in range(4*i,4*i+4)]
                    trainer.training_step(model,dict(x=x[4*i:4*i+4],y=y[4*i:4*i+4],**{key:dict(samples=samples) if kind=='main' else samples}))
                optimizer.step()
                if kind=='main':trainer.record_optimizer_step(model);bound=trainer._audit_records[0]['samples']
                else:trainer._record_after_optimizer(model);bound=trainer.probe_collector.records[0]
                self.assertEqual([r['sample_id'] for r in bound],list(range(16)))
                self.assertTrue(torch.allclose(model.weight,reference.weight,atol=1e-7))
                if kind=='main':trainer._pending_audit=dict(samples=bound[:14])
                else:trainer._pending_step_samples=bound[:14]
                with self.assertRaises(RuntimeError):
                    trainer.record_optimizer_step(model) if kind=='main' else trainer._record_after_optimizer(model)
        finally:SFTTrainer.training_step=original

    def test_fully_masked_microbatch_rejected(self):
        from federated_learning.formal_sft_trainer import FormalFederatedSFTTrainer
        obj=object.__new__(FormalFederatedSFTTrainer)
        with self.assertRaisesRegex(ValueError,'no supervised'):
            obj.training_step(None,{'labels':torch.full((2,512),-100)})


if __name__=='__main__':unittest.main()
