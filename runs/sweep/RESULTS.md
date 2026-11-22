# LayoutPilot sweep results

Held-out layouts: 200 (seeds 20000+). Initial 0.6781 · random policy 0.6407 · greedy oracle 0.9495 (25 reward queries per step).

| Method | Env steps | Seeds | Score (mean ± std) | Greedy gap closed |
|---|---:|---:|---:|---:|
| bc_ppo | 8,192 | 5 | 0.7310 ± 0.0215 | 19.5% ± 7.9% |
| ppo | 8,192 | 5 | 0.7138 ± 0.0152 | 13.1% ± 5.6% |
| bc_ppo | 131,072 | 5 | 0.7950 ± 0.0212 | 43.1% ± 7.8% |
| ppo | 131,072 | 5 | 0.7938 ± 0.0076 | 42.6% ± 2.8% |
| bc_ppo | 524,288 | 5 | 0.9061 ± 0.0211 | 84.0% ± 7.8% |
| ppo | 524,288 | 5 | 0.9214 ± 0.0166 | 89.7% ± 6.1% |

| Steps | Comparison (B − A) | Δ score | 95% CI | Significant |
|---:|---|---:|---:|:---:|
| 8,192 | bc_ppo − ppo | +0.0172 | [+0.0129, +0.0216] | yes |
| 8,192 | bc_ppo − greedy_oracle | -0.2184 | [-0.2318, -0.2049] | yes |
| 8,192 | ppo − greedy_oracle | -0.2357 | [-0.2506, -0.2206] | yes |
| 131,072 | bc_ppo − ppo | +0.0012 | [-0.0051, +0.0074] | no |
| 131,072 | bc_ppo − greedy_oracle | -0.1545 | [-0.1668, -0.1424] | yes |
| 131,072 | ppo − greedy_oracle | -0.1557 | [-0.1682, -0.1434] | yes |
| 524,288 | bc_ppo − ppo | -0.0153 | [-0.0200, -0.0109] | yes |
| 524,288 | bc_ppo − greedy_oracle | -0.0433 | [-0.0505, -0.0369] | yes |
| 524,288 | ppo − greedy_oracle | -0.0280 | [-0.0344, -0.0220] | yes |

Gap closed = (policy − initial) / (greedy − initial), per seed. CIs: paired bootstrap over held-out layouts using seed-averaged per-layout scores. The policy needs 0 reward queries at inference; the greedy oracle needs 25 per step.
