import torch
from layoutpilot.train import PosterEnv,ActorCritic,score,rollout,ppo_update

def test_environment():
    e=PosterEnv(seed=5,horizon=3);obs=e.reset();assert obs.shape==(24,)
    for i in range(3):
        obs,r,done,info=e.step(24)
    assert done and 'quality' in info and isinstance(score(e.rects),float)

def test_ppo_update():
    torch.manual_seed(1);env=PosterEnv(seed=2)
    model=ActorCritic(width=32);batch=rollout(env,model,torch.device('cpu'),32,.99,.95)
    assert len(batch['advantages'])==32
    optim=torch.optim.Adam(model.parameters(),lr=1e-3)
    before=model.actor.weight.detach().clone()
    loss=ppo_update(model,optim,batch,torch.device('cpu'),epochs=1,batch_size=16)
    assert torch.isfinite(torch.tensor(loss)) and not torch.equal(model.actor.weight,before)


def test_design_briefs_and_svg(tmp_path):
    from layoutpilot.train import TEMPLATES,export_svg,observation,greedy_action
    import xml.etree.ElementTree as ET
    for style in TEMPLATES:
        env=PosterEnv(seed=3,template=style)
        assert env.template_name==style
        assert observation(env.rects,env.target).shape==(24,)
        out=tmp_path/f'{style}.svg'
        export_svg(env.rects,out,style=style)
        root=ET.parse(out).getroot()
        assert root.tag.endswith('svg')
        assert sum(t.tag.endswith('g') for t in root)>=3
        assert 0<=greedy_action(env)<25


def test_optional_expert_warmstart():
    from layoutpilot.train import expert_warmstart
    model=ActorCritic(width=32)
    loss=expert_warmstart(model,PosterEnv(seed=7),torch.device('cpu'),examples=24,epochs=1,batch_size=16)
    assert torch.isfinite(torch.tensor(loss))


def test_hard_artboard_constraints():
    from layoutpilot.train import clamp_to_artboard
    import numpy as np
    boxes=np.array([[-.2,.9,1.2,.7],[.8,.8,.5,.5],[.9,-.5,.8,.3]])
    safe=clamp_to_artboard(boxes)
    assert np.all(safe[:,:2]>=.04-1e-6)
    assert np.all(safe[:,:2]+safe[:,2:]<=.96+1e-6)
