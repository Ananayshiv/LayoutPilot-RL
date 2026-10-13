"""PPO from scratch for layout optimization; no Stable-Baselines or Gym dependency.
python train.py --steps 10000    # CPU-compatible
"""
import argparse,csv,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from PIL import Image,ImageDraw,ImageFont

TARGET=np.array([[.08,.06,.84,.20],[.14,.32,.72,.43],[.16,.82,.68,.12]],dtype=np.float32)
# Three editable poster regions: headline, visual artwork, call to action.
# Their target geometry changes per brief; the RL policy must condition on it.
TEMPLATES={
    'minimal': TARGET,
    'editorial': np.array([[.08,.07,.84,.15],[.07,.30,.86,.49],[.55,.87,.38,.08]],dtype=np.float32),
    'split': np.array([[.08,.09,.39,.25],[.52,.09,.40,.69],[.08,.77,.37,.11]],dtype=np.float32),
}
N_ELEMS=3;N_ACTS=25;OBS_DIM=24

def observation(rects,target=TARGET):
    r=np.asarray(rects,dtype=np.float32).reshape(3,4)
    return np.concatenate([r.reshape(-1),((r-target)*4).reshape(-1)]).astype(np.float32)


def overlap(a,b):
    x0=max(float(a[0]),float(b[0]));y0=max(float(a[1]),float(b[1]))
    x1=min(float(a[0]+a[2]),float(b[0]+b[2]));y1=min(float(a[1]+a[3]),float(b[1]+b[3]))
    inter=max(x1-x0,0)*max(y1-y0,0)
    smaller=float(min(a[2]*a[3],b[2]*b[3]))
    # Occupancy of the smaller object: text legibility is lost even when
    # a small headline/CTA intersects a huge art block.
    return inter/max(smaller,1e-8)


def score(rects,target=TARGET):
    """Design proxy; higher is better, NOT human preference or learned aesthetics."""
    r=np.asarray(rects).reshape(3,4)
    centers=r[:,:2]+r[:,2:]/2; tc=target[:,:2]+target[:,2:]/2
    positioning=np.abs(centers-tc).mean()
    sizing=np.abs(r[:,2:]-target[:,2:]).mean()
    collisions=sum(overlap(r[i],r[j]) for i,j in [(0,1),(0,2),(1,2)])/3
    overflow=(np.maximum(-r[:,:2],0)+np.maximum(r[:,:2]+r[:,2:]-1,0)).mean()
    align=np.abs(centers[:,0]-tc[:,0]).mean()
    return float(1.-1.1*positioning-.7*sizing-.85*collisions-1.5*overflow-.18*align)


def clamp_to_artboard(r):
    """Hard safety constraints: all editable elements stay inside 4% margins."""
    r=np.asarray(r,dtype=np.float32).copy()
    r[:,2:]=np.clip(r[:,2:],.08,.86)
    r[:,0]=np.minimum(np.maximum(r[:,0],.04),.96-r[:,2])
    r[:,1]=np.minimum(np.maximum(r[:,1],.04),.96-r[:,3])
    return r


class PosterEnv:
    def __init__(self,seed=42,horizon=35,template='random'):
        if template!='random' and template not in TEMPLATES:raise ValueError('Unknown design template')
        self.rng=np.random.default_rng(seed);self.horizon=horizon;self.template=template;self.reset()
    def reset(self):
        self.template_name=(str(self.rng.choice(list(TEMPLATES))) if self.template=='random' else self.template)
        self.target=TEMPLATES[self.template_name]
        r=self.target+self.rng.normal(0,[.17,.17,.16,.10],(3,4)).astype(np.float32)
        r[:,:2]=np.clip(r[:,:2],-.05,.9);r[:,2:]=np.clip(r[:,2:],.10,.92)
        self.rects=clamp_to_artboard(r);self.t=0;return observation(self.rects,self.target)
    def step(self,action):
        old=score(self.rects,self.target)
        if int(action)!=24:
            item=int(action)//8;delta=int(action)%8
            axis=delta//2;sign=1 if delta%2==0 else -1
            val=.045 if axis<2 else .035
            self.rects[item,axis]+=sign*val
            if axis<2:self.rects[item,axis]=np.clip(self.rects[item,axis],-.1,.95)
            else:self.rects[item,axis]=np.clip(self.rects[item,axis],.08,.95)
        self.rects=clamp_to_artboard(self.rects)
        self.t+=1
        new=score(self.rects,self.target);reward=(new-old)*12-.003
        terminated=self.t>=self.horizon
        return observation(self.rects,self.target),float(reward),bool(terminated),{'quality':new}


class ActorCritic(nn.Module):
    def __init__(self,width=128,obs_dim=OBS_DIM,action_dim=N_ACTS):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(obs_dim,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.actor=nn.Linear(width,action_dim);self.critic=nn.Linear(width,1)
        nn.init.orthogonal_(self.actor.weight,gain=.01)
        nn.init.orthogonal_(self.critic.weight,gain=1.)
    def forward(self,x):
        z=self.encoder(x);return self.actor(z),self.critic(z).squeeze(-1)


def rollout(env,model,device,count,gamma,lam):
    obs=observation(env.rects,env.target);arrays={k:[] for k in ('obs','actions','logp','values','rewards','dones')}
    for _ in range(count):
        x=torch.tensor(obs,device=device,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits,value=model(x);dist=Categorical(logits=logits);a=dist.sample()
        nxt,r,done,_=env.step(int(a.item()))
        for k,v in [('obs',obs.copy()),('actions',int(a.item())),('logp',float(dist.log_prob(a).item())),('values',float(value.item())),('rewards',r),('dones',float(done))]: arrays[k].append(v)
        obs=env.reset() if done else nxt
    with torch.no_grad():
        _,bootstrap=model(torch.tensor(obs,device=device,dtype=torch.float32).unsqueeze(0))
    values=np.asarray(arrays['values'],np.float32)
    adv=compute_gae(arrays['rewards'],values,arrays['dones'],float(bootstrap.item()),gamma,lam)
    arrays['returns']=adv+values;arrays['advantages']=adv
    return arrays


def clipped_policy_loss(new_log_prob,old_log_prob,advantages,clip=.2):
    """PPO-Clip: -E[min(r*A, clip(r,1-e,1+e)*A)] (Schulman et al.)."""
    ratio=(new_log_prob-old_log_prob).exp()
    return -torch.minimum(ratio*advantages,
        ratio.clamp(1-clip,1+clip)*advantages).mean()


def compute_gae(rewards,values,dones,last_value,gamma=.99,lam=.95):
    """GAE(lambda): terminal transitions MUST mask bootstrap and recursion."""
    n=len(rewards)
    advantages=np.zeros(n,dtype=np.float32)
    gae=0.
    for i in range(n-1,-1,-1):
        nonterminal=1.-float(dones[i])
        next_value=values[i+1] if i<n-1 else last_value
        delta=rewards[i]+gamma*next_value*nonterminal-values[i]
        gae=delta+gamma*lam*nonterminal*gae
        advantages[i]=gae
    return advantages


def ppo_update(model,opt,data,device,epochs=4,batch_size=128,clip=.2,entropy=.01,target_kl=None,return_diagnostics=False):
    obs=torch.tensor(np.asarray(data['obs']),dtype=torch.float32,device=device)
    actions=torch.tensor(data['actions'],dtype=torch.int64,device=device)
    oldlog=torch.tensor(data['logp'],dtype=torch.float32,device=device)
    returns=torch.tensor(data['returns'],dtype=torch.float32,device=device)
    advantages=torch.tensor(data['advantages'],dtype=torch.float32,device=device)
    advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    losses=[];kls=[];clip_fracs=[];stopped=False
    for epoch in range(epochs):
        for idx in torch.randperm(len(obs),device=device).split(batch_size):
            logits,v=model(obs[idx]);dist=Categorical(logits=logits)
            logratio=dist.log_prob(actions[idx])-oldlog[idx]
            ratio=logratio.exp()
            with torch.no_grad():
                approx_kl=((ratio-1)-logratio).mean().item()
                clip_fraction=((ratio-1).abs()>clip).float().mean().item()
            if target_kl is not None and approx_kl>1.5*target_kl:
                stopped=True;break
            policy_loss=clipped_policy_loss(dist.log_prob(actions[idx]),oldlog[idx],advantages[idx],clip)
            value_loss=(returns[idx]-v).square().mean()
            loss=policy_loss+.5*value_loss-entropy*dist.entropy().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
            losses.append(float(loss.detach()));kls.append(float(approx_kl));clip_fracs.append(float(clip_fraction))
        if stopped:break
    stats={'loss':float(np.mean(losses)) if losses else 0.0,
           'approx_kl':float(np.mean(kls)) if kls else 0.0,
           'clip_fraction':float(np.mean(clip_fracs)) if clip_fracs else 0.0,
           'early_stopped':stopped,'minibatches_updated':len(losses)}
    return stats if return_diagnostics else stats['loss']

def greedy_action(env):
    """Strong design-score oracle baseline; it directly inspects the reward."""
    prior=env.rects.copy();clock=env.t;best=24;best_score=score(prior,env.target)
    for a in range(N_ACTS):
        env.rects=prior.copy();env.t=clock
        _,_,_,info=env.step(a)
        if info['quality']>best_score+1e-8:best_score=info['quality'];best=a
    env.rects=prior;env.t=clock
    return best


def expert_warmstart(model,env,device,examples=0,epochs=6,batch_size=128,lr=1e-3):
    """Optional imitation from transparent reward-greedy demonstrations.

    This is BEHAVIORAL CLONING + PPO when enabled (not pure PPO). It can
    substantially accelerate learning in a tight Colab budget.
    """
    if examples<=0:return float('nan')
    xs=[];ys=[]
    # Sample fresh perturbed layouts; avoid a trivial dataset dominated by no-ops.
    attempts=0
    while len(xs)<examples and attempts<examples*20:
        attempts+=1
        env.reset()
        a=greedy_action(env)
        if a==24:continue
        xs.append(observation(env.rects,env.target).copy());ys.append(a)
    if len(xs)<examples:raise RuntimeError('Unable to collect diverse expert examples')
    features=torch.as_tensor(np.asarray(xs),dtype=torch.float32,device=device)
    actions=torch.as_tensor(ys,dtype=torch.long,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    losses=[]
    for _ in range(epochs):
        for ids in torch.randperm(examples,device=device).split(batch_size):
            logits,_=model(features[ids]);loss=nn.functional.cross_entropy(logits,actions[ids])
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses[-max(1,examples//batch_size):]))


def evaluate(model,seeds,device,random_policy=False,horizon=35,greedy_policy=False):
    qualities=[];initial=[]
    for seed in seeds:
        env=PosterEnv(seed=seed,horizon=horizon);obs=env.reset()
        # save initial quality on exactly same independent seeds
        initial.append(score(env.rects,env.target))
        for j in range(env.horizon):
            if greedy_policy:a=greedy_action(env)
            elif random_policy: a=int(env.rng.integers(N_ACTS))
            else:
                with torch.no_grad():a=int(model(torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0))[0].argmax(-1).item())
            obs,_,done,info=env.step(a)
            if done:break
        qualities.append(score(env.rects,env.target))
    return float(np.mean(initial)),float(np.mean(qualities))


def _font(sz,bold=False):
    # System fonts on Colab + Linux; independent of bundled proprietary fonts.
    paths=(['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if bold else
           ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])
    for path in paths:
        try:return ImageFont.truetype(path,max(8,int(sz)))
        except OSError:pass
    return ImageFont.load_default()


def render(rects,path,width=600,height=800,style='minimal'):
    """Poster PNG preview. Export separately to genuinely editable SVG."""
    palettes={
        'minimal':('#F4F1EB',['#14213D','#F1B85B','#187C7D'],'#182438'),
        'editorial':('#F5F0EA',['#D34D3F','#18324C','#E4AF58'],'#1F2D37'),
        'split':('#141720',['#F9C846','#4262D9','#D96350'],'#F6F2E8'),
    }
    bg,colors,ink=palettes.get(style,palettes['minimal'])
    im=Image.new('RGB',(width,height),bg);d=ImageDraw.Draw(im)
    d.text((42,9),'STUDIO  /  26     —     DESIGN EXPLORATION',fill=ink,font=_font(13,bold=True))
    # Draw artwork first; drawing text panels on top improves legibility under partial overlap.
    for i in [1,0,2]:
        x,y,w,h=rects[i]
        box=[int(x*width),int(y*height),int((x+w)*width),int((y+h)*height)]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        d.rounded_rectangle(box,radius=16,fill=colors[i])
        bw=box[2]-box[0];bh=box[3]-box[1]
        if i==1:
            cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            rad=max(5,min(bw,bh)*.24)
            d.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),fill='#F6EFE4')
            d.arc((cx-rad*.7,cy-rad*.7,cx+rad*.7,cy+rad*.7),45,290,fill='#D65350',width=6)
        else:
            label='CREATE / WITHOUT LIMITS' if i==0 else 'EXPLORE MORE  >'
            textcolor=('#FFFFFF' if i==0 else '#132438')
            availw=max(bw-24,10);availh=max(bh-14,10)
            size=min(34 if i==0 else 19, max(8,int(availh*.35)))
            while size>8 and d.textbbox((0,0),label,font=_font(size,bold=True))[2]>availw:
                size-=1
            d.text((box[0]+12,box[1]+max(5,(bh-size)//2-3)),label,
                   fill=textcolor,font=_font(size,bold=True))
    d.text((42,height-18),'CREATIVE LAB      /      REINFORCEMENT LEARNING',fill=ink,font=_font(11))
    im.save(path)


def export_svg(rects,path,style='minimal',width=600,height=800):
    """Native SVG primitives/text, not an embedded PNG; editable in Illustrator."""
    from xml.sax.saxutils import escape
    colors={
        'minimal':('#F4F1EB',['#14213D','#FCBD65','#187C7D']),
        'editorial':('#F5F0EA',['#D34D3F','#18324C','#E4AF58']),
        'split':('#141720',['#F9C846','#4262D9','#D96350']),
    }
    bg,palette=colors.get(style,colors['minimal'])
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<title>Editable {escape(style)} AI-designed poster</title>',
           f'<rect id="background" width="{width}" height="{height}" fill="{bg}"/>']
    labels=['CREATE WITHOUT LIMITS','VISUAL CANVAS','EXPLORE NOW']
    for i,(x,y,w,h) in enumerate(rects):
        x,y,w,h=(round(float(x*width),2),round(float(y*height),2),round(float(w*width),2),round(float(h*height),2))
        lines += [f'<g id="editable-layer-{i+1}" data-role="{["headline","artwork","cta"][i]}">',
                  f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{palette[i]}"/>']
        if i==1:
            cx=x+w/2;cy=y+h/2;r=min(w,h)*.24
            lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="#F6EFE4"/>')
        else:
            textcolor='#132438' if i==2 else '#FFFFFF'
            lines.append(f'<text x="{x+12:.2f}" y="{y+28:.2f}" font-family="sans-serif" font-size="18" font-weight="bold" fill="{textcolor}">{escape(labels[i])}</text>')
        lines.append('</g>')
    lines.append('</svg>')
    Path(path).write_text('\n'.join(lines),encoding='utf-8')


def run(args):
    np.random.seed(args.seed);random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cpu') # discrete lightweight simulator performs better on CPU
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    env=PosterEnv(seed=args.seed,horizon=args.horizon)
    model=ActorCritic(width=args.width).to(device)
    bc_loss=expert_warmstart(model,env,device,examples=args.warmstart_examples,epochs=args.warmstart_epochs)
    if args.warmstart_examples:
        print(f'Imitation warm-start used {args.warmstart_examples} reward-greedy examples, last BC loss {bc_loss:.4f}. Now PPO finetuning.')
        env.reset()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-5)
    hist=[];nupdates=max(1,(args.steps+args.rollout-1)//args.rollout)
    best_val=-float('inf');best_weights=None;best_update=0
    before_env=PosterEnv(seed=10001,horizon=args.horizon);initial_rects=before_env.rects.copy()
    for update in range(1,nupdates+1):
        batch=rollout(env,model,device,args.rollout,args.gamma,args.gae_lambda)
        diagnostics=ppo_update(model,opt,batch,device,epochs=args.epochs,batch_size=args.minibatch,
                               target_kl=args.target_kl if args.target_kl>0 else None,return_diagnostics=True)
        loss=diagnostics['loss']
        if update==1 or update==nupdates or update%max(1,nupdates//10)==0:
            q0,qa=evaluate(model,range(1000,1020),device,horizon=args.horizon)
            hist.append([update,update*args.rollout,loss,q0,qa,diagnostics['approx_kl'],diagnostics['clip_fraction'],int(diagnostics['early_stopped'])]);print(f"update {update}/{nupdates}: start={q0:.3f} PPO={qa:.3f} loss={loss:.3f} KL={diagnostics['approx_kl']:.4f}")
            # Select policy using validation seeds 1000..1019, never the final test seeds.
            if qa>best_val:
                best_val=qa;best_update=update
                best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    seeds=list(range(20000,20000+args.eval_episodes))
    qstart,qppo=evaluate(model,seeds,device,horizon=args.horizon)
    _,qrandom=evaluate(model,seeds,device,random_policy=True,horizon=args.horizon)
    _,qgreedy=evaluate(model,seeds,device,greedy_policy=True,horizon=args.horizon)
    demo=PosterEnv(seed=10001,horizon=args.horizon)
    obs=observation(demo.rects,demo.target)
    for _ in range(args.horizon):
        with torch.no_grad():action=int(model(torch.tensor(obs,dtype=torch.float32).unsqueeze(0))[0].argmax(-1).item())
        obs,*_=demo.step(action)
    render(initial_rects,out/'before.png',style=before_env.template_name)
    render(demo.rects,out/'after.png',style=demo.template_name)
    export_svg(demo.rects,out/'editable_poster.svg',style=demo.template_name)
    # Explicit geometric greedy oracle reference, NOT PPO output.
    oracle=PosterEnv(seed=10001,horizon=args.horizon)
    for _ in range(oracle.horizon):
        act=greedy_action(oracle);oracle.step(act)
    render(oracle.rects,out/'greedy_oracle.png',style=oracle.template_name)
    export_svg(oracle.rects,out/'greedy_oracle.svg',style=oracle.template_name)
    torch.save({'model':model.state_dict(),'width':args.width,'steps':args.steps,'selected_update':best_update},out/'policy.pt')
    with (out/'training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['update','steps','ppo_loss','mean_initial_quality','mean_ppo_quality','approx_kl','clip_fraction','kl_early_stop']);w.writerows(hist)
    import json
    metrics={'n_eval_episodes':args.eval_episodes,'eval_seed_start':20000,'validation_seed_start':1000,'target_kl':args.target_kl,'initial_score':qstart,'random_policy_score':qrandom,'greedy_oracle_score':qgreedy,'ppo_policy_score':qppo,'improvement_over_initial':qppo-qstart,'improvement_over_random':qppo-qrandom,'templates':list(TEMPLATES),'selected_validation_update':best_update,'best_validation_score':best_val,'expert_warmstart_examples':args.warmstart_examples,'bc_train_loss':bc_loss if args.warmstart_examples else None,'disclaimer':'Geometric proxy reward, NOT a validated human aesthetic measure.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Saved policy.pt, before.png, after.png, editable_poster.svg, greedy_oracle.png/svg, metrics.json, training.csv to',out)
    print(metrics)


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=16000)
    p.add_argument('--rollout',type=int,default=512);p.add_argument('--horizon',type=int,default=35)
    p.add_argument('--width',type=int,default=128);p.add_argument('--minibatch',type=int,default=128)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--target-kl',type=float,default=.02,help='PPO stability guardrail; 0 disables KL early stopping')
    p.add_argument('--gae-lambda',type=float,default=.95);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--eval-episodes',type=int,default=60);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='outputs')
    p.add_argument('--warmstart-examples',type=int,default=0,help='Optional greedy-expert BC examples; setting >0 makes this BC+PPO, not pure PPO')
    p.add_argument('--warmstart-epochs',type=int,default=6)
    return p.parse_args(argv)

if __name__=='__main__':run(parse_args())
