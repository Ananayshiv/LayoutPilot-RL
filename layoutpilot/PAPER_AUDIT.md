# Paper fidelity audit — PPO-Clip (Schulman et al., 2017)

**Publication:** https://arxiv.org/abs/1707.06347

## Reproduced PPO-Clip algorithm

- A learnable actor and critic with categorically sampled actions.
- On-policy rollouts storing `log pi_old(a|s)`, values, rewards, terminal boundaries.
- GAE(lambda) using `delta = reward + gamma*next_value*(1-done) - value`, correctly masking terminal transitions.
- Clipped surrogate: `-mean(min(r*A, clamp(r,1-e,1+e)*A))`, `r=exp(logp_new-logp_old)`.
- Advantages normalized before updating; value-function MSE, entropy bonus, gradient clipping.
- Multiple minibatch epochs over fixed rollout observations; separate training and evaluation seeds.

## What is NOT claimed

- Original 2017 MuJoCo/Atari performance benchmarks have **not** been reproduced.
- The `PosterEnv` is entirely original and has **handcrafted geometry scoring**, not an 
- `cartpole_sanity.py` is optional verification of the same PPO optimizer on the standard CartPole task. This is not part of the original paper experiments.

## Extension (explicitly non-paper)

- Three conditional creative design briefs with move/resize actions, reusable rendering templates, and Illustrator-editable SVG groups.
- Optional `--warmstart-examples > 0` trains from greedy-expert demonstrations before PPO; label as **BC+PPO**, not PPO alone. Paper baseline is **`--warmstart-examples 0`**.
- Held-out test layout scores compare random, initialized, PPO/BC+PPO and reward-greedy oracle, selected only on separate validation seeds.

## Audit status

Algebra unit tests, terminal masking test, finite PPO update and SVG output are available. Prior hybrid local test is on an **earlier V3 implementation**, not a V4 result. V4 needs matched-budget PPO-only vs BC+PPO runs before any improvement claim.

## V6 stability and experiment checks

PPO minibatches now monitor the approximate reverse KL `(ratio-1)-logratio`, the fraction with `|ratio-1|>clip`, and optionally stop further update minibatches if KL exceeds `1.5 * --target-kl`. This practical engineering guardrail is not presented as a separate theorem from the original paper. The independent `evaluate_advanced.py` creates paired starting layouts, deterministic policies and a bootstrap confidence interval for **BC+PPO minus pure PPO** (one checkpoint per approach). It does not quantify variation across training seeds.
