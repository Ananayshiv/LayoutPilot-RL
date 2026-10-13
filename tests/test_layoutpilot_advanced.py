import torch
import numpy as np
from layoutpilot.train import ActorCritic, PosterEnv,ppo_update,rollout
from layoutpilot.evaluate_advanced import rollout_score,compare

def test_ppo_kl_guardrail_and_observability():
    policy=ActorCritic(width=16);env=PosterEnv(seed=3,horizon=3)
    data=rollout(env,policy,'cpu',20,.99,.95)
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    d=ppo_update(policy,optimizer,data,'cpu',epochs=2,batch_size=8,target_kl=.02,return_diagnostics=True)
    assert d['minibatches_updated']>0
    assert np.isfinite(d['approx_kl']) and 0<=d['clip_fraction']<=1
    assert isinstance(d['early_stopped'],bool)

def test_identical_checkpoint_paired_score_zero(tmp_path):
    m=ActorCritic(width=16)
    ckpt=tmp_path/'policy.pt';torch.save({'model':m.state_dict(),'width':16},ckpt)
    result=compare(str(ckpt),str(ckpt),n=3,horizon=3,out=str(tmp_path/'eval'))
    assert abs(result['means']['delta'])<1e-9
    assert abs(result['extension_minus_ppo_95pct_bootstrap_ci'][0])<1e-9
