"""Independent PPO-Clip on Gymnasium CartPole-v1 as a canonical RL sanity check.

This verifies generality of the algorithm outside the custom poster environment.
No Stable-Baselines implementation, pretraining, or RL helper library is used.
Optional dependency: pip install gymnasium
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from layoutpilot.train import ActorCritic, compute_gae, ppo_update


def train_cartpole(steps=8192,seed=42,out='cartpole_run',rollout_size=256,epochs=4):
    import gymnasium as gym
    np.random.seed(seed)
    torch.manual_seed(seed)
    env=gym.make('CartPole-v1')
    obs,_=env.reset(seed=seed)
    env.action_space.seed(seed)
    model=ActorCritic(width=64,obs_dim=4,action_dim=2)
    optimizer=torch.optim.Adam(model.parameters(),lr=2.5e-4,eps=1e-5)
    episode_rewards=[]
    this_return=0.
    for start in range(0,steps,rollout_size):
        count=min(steps-start,rollout_size)
        history={k:[] for k in ('obs','actions','logp','values','rewards','dones')}
        for _ in range(count):
            x=torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits,v=model(x);dist=Categorical(logits=logits);a=dist.sample()
            obs_next,reward,terminated,truncated,_=env.step(int(a.item()))
            # A time-limit truncation is a bootstrap boundary, not an MDP terminal.
            # Bootstrap truncated terminal state into reward, then mask the GAE
            # recursion because the environment must reset to a new episode.
            adjusted_reward=float(reward)
            if truncated and not terminated:
                with torch.no_grad():
                    _,vnext=model(torch.tensor(obs_next,dtype=torch.float32).unsqueeze(0))
                adjusted_reward += .99*float(vnext.item())
            done=terminated or truncated
            for k,vv in [('obs',np.asarray(obs,dtype=np.float32).copy()),('actions',int(a.item())),
                ('logp',float(dist.log_prob(a).item())),('values',float(v.item())),
                ('rewards',adjusted_reward),('dones',float(done))]: history[k].append(vv)
            this_return+=float(reward)
            if done:
                episode_rewards.append(this_return)
                this_return=0.
                obs,_=env.reset()
            else:obs=obs_next
        with torch.no_grad():
            _,last_v=model(torch.tensor(obs,dtype=torch.float32).unsqueeze(0))
        v=np.asarray(history['values'],dtype=np.float32)
        advantage=compute_gae(history['rewards'],v,history['dones'],float(last_v.item()),.99,.95)
        history['advantages']=advantage;history['returns']=v+advantage
        ppo_update(model,optimizer,history,torch.device('cpu'),epochs=epochs,batch_size=64,clip=.2,entropy=.01)
    # Deterministic evaluation, independent of training random seeds.
    evals=[]
    for i in range(10):
        obs,_=env.reset(seed=seed+10_000+i)
        tot=0
        for _ in range(500):
            with torch.no_grad():
                logits,_=model(torch.tensor(obs,dtype=torch.float32).unsqueeze(0))
            obs,r,done,trunc,_=env.step(int(logits.argmax(-1)))
            tot+=float(r)
            if done or trunc:break
        evals.append(tot)
    env.close()
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    result={'total_steps':steps,'training_episodes':len(episode_rewards),
        'mean_recent_training_reward':float(np.mean(episode_rewards[-20:])) if episode_rewards else None,
        'mean_greedy_test_reward':float(np.mean(evals)),'test_rewards':evals,
        'note':'CartPole baseline sanity check, not an original PPO paper benchmark replication.'}
    (path/'cartpole_metrics.json').write_text(json.dumps(result,indent=2))
    torch.save({'model':model.state_dict(),'width':64,'obs_dim':4,'action_dim':2},path/'cartpole_policy.pt')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=8192)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='cartpole_run')
    args=p.parse_args()
    train_cartpole(args.steps,args.seed,args.out)
