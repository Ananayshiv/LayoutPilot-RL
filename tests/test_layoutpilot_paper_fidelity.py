"""PPO mathematical objectives and GAE terminal masking."""
import numpy as np
import torch
from layoutpilot.train import clipped_policy_loss, compute_gae, PosterEnv, observation


def test_ratio_clipping_positive_and_negative_advantages():
    old=torch.zeros(2)
    new=torch.tensor([np.log(1.5), np.log(.5)],dtype=torch.float32)
    adv=torch.tensor([1.,-1.])
    # min(.5* -1, .8* -1)=-.8; min(1.5,1.2)=1.2 -> loss=-.2.
    expected=-(1.2-.8)/2
    assert torch.allclose(clipped_policy_loss(new,old,adv,clip=.2),torch.tensor(expected,dtype=torch.float32),atol=1e-6)


def test_gae_terminal_masks_future_rewards():
    rewards=[1.,7.,2.];values=[.5,.6,.7];dones=[0.,1.,0.]
    g=compute_gae(rewards,values,dones,last_value=123.,gamma=1.,lam=1.)
    # at terminal index 1: advantage=7-.6=6.4; index 0 includes only
    # 1 + gamma*6.4+(.6-.5) =7.5, not transition after terminal.
    assert abs(float(g[1])-6.4)<1e-5
    assert abs(float(g[0])-7.5)<1e-5
    assert abs(float(g[2])-(2+123-.7))<1e-5


def test_action_noop_and_observation_target_information():
    env=PosterEnv(seed=99,template='editorial',horizon=3)
    before=env.rects.copy()
    x=observation(env.rects,env.target)
    assert not np.isnan(x).any()
    env.step(24)
    assert np.array_equal(before,env.rects)
    assert np.all(env.rects[:,:2]+env.rects[:,2:]<=.96001)


def test_ppo_actor_critic_supports_standard_cartpole_shapes_and_gradients():
    from layoutpilot.train import ActorCritic, ppo_update
    model=ActorCritic(width=16,obs_dim=4,action_dim=2)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    obs=np.random.randn(16,4).astype('float32')
    with torch.no_grad():
        logits,v=model(torch.from_numpy(obs))
        actions=torch.distributions.Categorical(logits=logits).sample()
        lp=torch.distributions.Categorical(logits=logits).log_prob(actions)
    data={'obs':obs,'actions':actions.numpy(),'logp':lp.numpy(),
          'advantages':np.linspace(-1,1,16,dtype=np.float32),
          'returns':v.numpy() + .1}
    loss=ppo_update(model,optimizer,data,torch.device('cpu'),epochs=2,batch_size=8)
    assert np.isfinite(loss)
    assert model(torch.zeros(1,4))[0].shape==(1,2)
