# LayoutPilot — PPO Layout Optimization and Learned-Reward Diagnostics

PyTorch implementation of discrete-action actor–critic **PPO-Clip**, GAE, KL monitoring, and optional behavior-cloning (BC) initialization from a greedy layout solver. Includes an independently defined geometric gold metric and diagnostic tools for learned-reward optimization, extreme tails, ensemble uncertainty and proposal shift.

## Measured results — 60 held-out procedural layouts
| Method | Mean geometric score ↑ |
|---|---:|
| Initial layouts | 0.6888 |
| Random policy | 0.6338 |
| PPO-only (8,192 environment steps) | 0.7213 |
| BC + PPO (8,192 environment steps) | **0.7347** |
| Reward-greedy solver | 0.9479 |

BC+PPO minus PPO-only = **+0.0134** (reported paired bootstrap 95% CI **[−0.0128, +0.0429]**). This interval includes zero. These values measure a handcrafted geometry function, **not** human-rated poster quality. Example SVG outputs still require aesthetic review.

## Research methodology
The PPO reward proxy can disagree with independent occlusion/whitespace metrics. Learned-regressor Best-of-N experiments maintain disjoint training/evaluation templates; advanced protocols probe high-score tails, uncertainty-aware acquisition, and shifted proposal distributions. Preliminary smoke trials **do not establish reward hacking or mitigation**. Do not interpret Goodhart's law as a guaranteed empirical downturn.

The finite-horizon observation currently omits remaining time. With `gamma < 1`, undiscounted score differences need not optimize terminal score exclusively. These limitations are documented rather than hidden.

## Reproduce
```bash
pip install -r requirements.txt
python -m layoutpilot.train --steps 8192 --warmstart-examples 0 --seed 42 --out runs/ppo
python -m layoutpilot.train --steps 8192 --warmstart-examples 700 --seed 42 --out runs/bc_ppo
python -m layoutpilot.evaluate_tail_uncertainty --smoke
python -m pytest -q
```

Model checkpoints and corresponding images were saved in Colab/Google Drive; they are not included in Git history.

**References:** Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347); Gao et al., [Scaling Laws for Reward Model Overoptimization](https://proceedings.mlr.press/v202/gao23h.html) (methodological inspiration, not an LLM-result reproduction).
