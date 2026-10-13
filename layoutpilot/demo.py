"""Interactive poster layout optimizer. Upload trained policy.pt and run `python demo.py`.
Self-contained demonstration; no external design software or proprietary SDK required.
"""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from layoutpilot.train import PosterEnv,ActorCritic,observation,render,export_svg,TEMPLATES


def generate(policy,template='minimal',seed=42,horizon=35):
    env=PosterEnv(seed=int(seed),template=template,horizon=int(horizon))
    before=env.rects.copy();obs=observation(env.rects,env.target)
    for _ in range(env.horizon):
        with torch.inference_mode():
            logits,_=policy(torch.from_numpy(obs).unsqueeze(0))
            action=int(logits.argmax(-1).item())
        obs,_,done,_=env.step(action)
        if done:break
    out=Path(tempfile.mkdtemp(prefix='poster_'))
    render(before,out/'before.png',style=template)
    render(env.rects,out/'after.png',style=template)
    export_svg(env.rects,out/'editable.svg',style=template)
    return Image.open(out/'before.png'),Image.open(out/'after.png'),str(out/'editable.svg')


def create_app(checkpoint='outputs/policy.pt'):
    import gradio as gr
    state=torch.load(checkpoint,weights_only=True,map_location='cpu')
    model=ActorCritic(width=state.get('width',128))
    model.load_state_dict(state['model']);model.eval()
    return gr.Interface(lambda kind,seed:generate(model,kind,int(seed)),
        inputs=[gr.Dropdown(list(TEMPLATES),label='Design brief',value='editorial'),
                gr.Slider(0,10000,value=42,step=1,label='Layout seed')],
        outputs=[gr.Image(label='Before'),gr.Image(label='After'),gr.File(label='Editable layered SVG')],
        title='LayoutPilot · Poster Design Agent',
        description='PPO-trained geometric design optimizer. Templates/quality function are hand-defined, not human preference models. SVG output remains editable.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='outputs/policy.pt')
    create_app(p.parse_args().checkpoint).launch(share=False)
