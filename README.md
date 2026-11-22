# LayoutPilot — PPO Layout Optimization and Learned-Reward Diagnostics

PyTorch implementation of discrete-action actor–critic **PPO-Clip**, GAE, KL monitoring, and optional behavior-cloning (BC) initialization from a greedy layout solver. Includes an independently defined geometric gold metric and diagnostic tools for learned-reward optimization, extreme tails, ensemble uncertainty and proposal shift.

**Headline (v2, 5 seeds, 200 held-out layouts):** with 524K environment steps, PPO reaches **0.921** and closes **~90% of the gap** to a greedy oracle that queries the true reward 25 times per step, while making **zero** reward queries itself. See the [budget sweep](#budget-sweep-is-ppo-under-trained-v2).

## v1 results — single run, 60 held-out procedural layouts (superseded by the sweep below)
| Method | Mean geometric score ↑ |
|---|---:|
| Initial layouts | 0.6888 |
| Random policy | 0.6338 |
| PPO-only (8,192 environment steps) | 0.7213 |
| BC + PPO (8,192 environment steps) | **0.7347** |
| Reward-greedy solver | 0.9479 |

BC+PPO minus PPO-only = **+0.0134** (reported paired bootstrap 95% CI **[−0.0128, +0.0429]**). This interval includes zero. These values measure a handcrafted geometry function, **not** human-rated poster quality. Example SVG outputs still require aesthetic review.

## Budget sweep: is PPO under-trained? (v2)

The v1 numbers above come from **8,192 environment steps**, which is only 16 PPO updates of 512 steps. The obvious question is whether PPO is fundamentally weak here or just under-trained. `layoutpilot.sweep` answers it with one variable at a time:

- **Budgets:** 8,192 / 131,072 / 524,288 environment steps (same rollout size, learning rate, and network as v1)
- **Seeds:** 5 per cell, for both PPO and BC+PPO (700 greedy-expert examples, as in v1)
- **Evaluation:** **200** held-out layouts (seeds 20000+, up from 60), with per-layout scores saved so every comparison is **paired**
- **Metrics:** mean ± std over seeds, the **fraction of the greedy gap closed** = (policy − initial) / (greedy − initial), and paired-bootstrap 95% CIs for BC+PPO − PPO and policy − greedy at each budget

```bash
python -m layoutpilot.sweep --budgets 8192 131072 524288 --seeds 0 1 2 3 4   # CPU, resumable
python -m layoutpilot.sweep --aggregate-only                                    # writes runs/sweep/RESULTS.md
```

Measured on GitHub Actions (CPU), 200 held-out layouts, 5 seeds per cell. Initial layouts 0.6781, random policy 0.6407, **greedy oracle 0.9495**. Raw results: [`runs/sweep/RESULTS.md`](runs/sweep/RESULTS.md).

| Method | Env steps | Score (mean ± std, 5 seeds) | Greedy gap closed |
|---|---:|---:|---:|
| PPO | 8,192 | 0.7138 ± 0.0152 | 13.1% ± 5.6% |
| BC + PPO | 8,192 | 0.7310 ± 0.0215 | 19.5% ± 7.9% |
| PPO | 131,072 | 0.7938 ± 0.0076 | 42.6% ± 2.8% |
| BC + PPO | 131,072 | 0.7950 ± 0.0212 | 43.1% ± 7.8% |
| **PPO** | **524,288** | **0.9214 ± 0.0166** | **89.7% ± 6.1%** |
| BC + PPO | 524,288 | 0.9061 ± 0.0211 | 84.0% ± 7.8% |

Findings (paired bootstrap 95% CIs over the 200 layouts):
1. **v1 was under-trained, not a weak method.** With 64× more experience, PPO goes from closing 13% to **~90% of the gap to the greedy oracle** (0.921 vs 0.950; still significantly below it, Δ −0.028 [−0.034, −0.022]), while making zero reward queries at inference.
2. **Behaviour cloning helps early and hurts late.** BC+PPO beats PPO at 8K steps (+0.017 [+0.013, +0.022]), ties at 131K (+0.001 [−0.005, +0.007]), and is significantly worse at 524K (−0.015 [−0.020, −0.011]). This is consistent with the greedy-expert initialization biasing the policy toward myopic one-step moves that PPO must later unlearn; that mechanism is a hypothesis, not yet tested.
3. The v1 single-run numbers (PPO 0.7213, BC+PPO 0.7347 at 8K) fall inside the v2 seed spread.

**Why greedy is not the bar RL must clear:** the greedy solver is an **oracle with simulator access**. At every step it tries all 25 actions against the *true* score, which is 875 reward evaluations per episode. The PPO policy makes **zero** reward queries at inference: it amortizes the search into one forward pass per step. That matters whenever the reward is expensive or unavailable at deployment time, for example a human rating or a learned reward model, which is exactly the setting the reward-overoptimization experiments below study. The meaningful question is therefore how much of the oracle's improvement a query-free policy recovers, and at what training budget.

## Research methodology
The PPO reward proxy can disagree with independent occlusion/whitespace metrics. Learned-regressor Best-of-N experiments maintain disjoint training/evaluation templates; advanced protocols probe high-score tails, uncertainty-aware acquisition, and shifted proposal distributions. Preliminary smoke trials **do not establish reward hacking or mitigation**. Do not interpret Goodhart's law as a guaranteed empirical downturn.

The finite-horizon observation currently omits remaining time. With `gamma < 1`, undiscounted score differences need not optimize terminal score exclusively. These limitations are documented rather than hidden.

## Reproduce
```bash
python -m pytest -q
python -m layoutpilot.sweep --budgets 8192 131072 524288 --seeds 0 1 2 3 4
pip install -r requirements.txt
python -m layoutpilot.train --steps 8192 --warmstart-examples 0 --seed 42 --out runs/ppo
python -m layoutpilot.train --steps 8192 --warmstart-examples 700 --seed 42 --out runs/bc_ppo
python -m layoutpilot.evaluate_tail_uncertainty --smoke
python -m pytest -q
```

Model checkpoints and corresponding images were saved in Colab/Google Drive; they are not included in Git history.

**References:** Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347); Gao et al., [Scaling Laws for Reward Model Overoptimization](https://proceedings.mlr.press/v202/gao23h.html) (methodological inspiration, not an LLM-result reproduction).
