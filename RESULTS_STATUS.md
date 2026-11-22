# Reported Colab run

- PPO-only and BC+PPO checkpoints trained at an 8,192-environment-step configuration.
- Evaluation: 60 synthetic starting layouts; handcrafted geometric score.
- Initial **0.6888**, random **0.6338**, PPO-only **0.7213**, BC+PPO **0.7347**, greedy oracle **0.9479**.
- BC+PPO minus PPO-only: **+0.0134**, paired 95% bootstrap interval **[-0.0128, 0.0429]**.

The interval includes zero; do not claim robust BC superiority. The learned-proxy and extreme-tail experiments are exploratory and not validated as Goodhart-law findings. These figures are user-provided Colab outputs, not rerun on GitHub.

# v2 sweep (pending)

`layoutpilot/sweep.py` reruns PPO and BC+PPO at 8,192 / 131,072 / 524,288 steps × 5 seeds on 200 held-out layouts with per-layout scores, paired bootstrap CIs, and the fraction of the greedy-oracle gap closed. No v2 numbers exist until `runs/sweep/RESULTS.md` is produced; the v1 figures above remain the only measured results.
