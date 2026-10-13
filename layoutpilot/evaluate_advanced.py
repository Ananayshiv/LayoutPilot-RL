"""Paired fixed-seed policy comparison for LayoutPilot with bootstrap uncertainty.

Uses the identical starting layout for PPO-only and BC+PPO. Reports a handcrafted
geometric quality score, never calls this a user/aesthetic-preference benchmark.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from layoutpilot.train import ActorCritic,PosterEnv,score,observation,greedy_action,N_ACTS


def bootstrap_ci(values,seed=2026,repeats=2000):
    data=np.asarray(values,dtype=float)
    if not len(data):raise ValueError('No values')
    rng=np.random.default_rng(seed)
    return [float(z) for z in np.quantile(data[rng.integers(0,len(data),(repeats,len(data)))].mean(axis=1),[.025,.975])]


def load_policy(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=ActorCritic(width=int(checkpoint['width'])).eval()
    model.load_state_dict(checkpoint['model'])
    return model


def rollout_score(model,seed,horizon=35,oracle=False,random=False):
    env=PosterEnv(seed=seed,horizon=horizon)
    obs=env.reset()
    initial=score(env.rects,env.target)
    for i in range(horizon):
        if oracle:a=greedy_action(env)
        elif random:a=int(env.rng.integers(N_ACTS))
        else:
            with torch.inference_mode():
                a=int(model(torch.from_numpy(obs).unsqueeze(0))[0].argmax(-1).item())
        obs,_,done,_=env.step(a)
        if done:break
    return initial,score(env.rects,env.target)


def compare(base,extended,n=40,seed_start=20000,horizon=35,out='paired_evaluation'):
    if n<=0:raise ValueError('n must be >0')
    base_model=load_policy(base);ext_model=load_policy(extended)
    rows=[]
    for s in range(seed_start,seed_start+n):
        ini,b=rollout_score(base_model,s,horizon)
        ini2,e=rollout_score(ext_model,s,horizon)
        _,r=rollout_score(None,s,horizon,random=True)
        _,g=rollout_score(None,s,horizon,oracle=True)
        assert abs(ini-ini2)<1e-7
        rows.append(dict(seed=s,initial=ini,random=r,ppo=b,bc_ppo=e,greedy=g,delta=e-b))
    result={'n':n,'test_seeds':[seed_start,seed_start+n-1],
            'means':{k:float(np.mean([row[k] for row in rows])) for k in ['initial','random','ppo','bc_ppo','greedy','delta']},
            'extension_minus_ppo_95pct_bootstrap_ci':bootstrap_ci([row['delta'] for row in rows]),
            'training_note':'PPO-only and BC+PPO checkpoints must use same budget and different output paths',
            'quality_note':'Synthetic handcrafted geometric score; not human preferences or industrial design assessment'}
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'paired_scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (dest/'paired_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ppo',required=True);p.add_argument('--bc-ppo',required=True)
    p.add_argument('--n',type=int,default=40);p.add_argument('--seed-start',type=int,default=20000)
    p.add_argument('--horizon',type=int,default=35);p.add_argument('--out',default='paired_evaluation')
    args=p.parse_args();compare(args.ppo,args.bc_ppo,args.n,args.seed_start,args.horizon,args.out)
