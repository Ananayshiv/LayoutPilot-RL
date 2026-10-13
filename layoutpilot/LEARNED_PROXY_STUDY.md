# Learned proxy reward over-optimization: experimental protocol

This repository contains a **small, controlled study design** inspired by Gao, Schulman and Hilton, *Scaling Laws for Reward Model Overoptimization* (arXiv 2210.10760; ICML 2023). It is not a replication of their reward-model or human-preference experiments.

## Hypothesis

A proxy fitted on scarce gold labels may be exploited by stronger selection: as the Best-of-N budget increases, the **predicted proxy score** increases while the **fixed held-out gold score** eventually plateaus or decreases. This is a falsifiable hypothesis, **not a guaranteed result**. Differences in proxy accuracy and extrapolation may vary substantially across training seeds.

## Data, labels, and holdouts

- `layoutpilot.learned_proxy.ProxyMLP` is a two-hidden-layer MLP over **12 raw rectangle coordinates** (3 editable elements × `x,y,w,h`), with GELU and a sigmoid output in `[0,1]`. Target geometry and the hand-engineered gold components are not input features.
- The **gold label** reuses the pre-existing target-blind `gold_metrics`: approximate hard text occlusion, reserved-region overlap, whitespace imbalance, whitespace shortfall and text capacity. It is an *engineered geometric constraint*, **not human preference or text-rendering truth**.
- A procedural **template** is a continuous geometry target obtained by perturbing one of three design families. Train IDs `10000..`, validation IDs `200000..` and final test IDs `400000..` are disjoint. Thus 64 distinct test targets are withheld, not merely 64 RNG seeds for the same three shapes.
- Each training seed generates a single pool of 10,000 (or the largest requested budget) gold-labeled proposals; the **100, 1,000 and 10,000** subsets are nested. The separate validation set selects the best checkpoint. Test gold labels are never used to train or select the proxy.
- Candidate proposals are i.i.d. from a **fixed template-conditioned mixture** of moderate/large geometric perturbations and broad random rectangles. The proposal does not depend on gold or on the learned proxy parameters.

## Optimization and auditability

For a fixed template and random generator, sample a candidate bank of size `N_max`. Rank **only by predicted `f_theta`**. At each nested budget `N`, select `argmax_{i<=N} f_theta(x_i)`. Only **after** the selected indices have been fixed may the evaluator query the gold oracle. Paired N=1 references use the exact first candidate of each bank; untouched starting-layout gold is logged separately.

`best_of_n` measures the **number of candidates inspected**, not PPO gradient steps. The pipeline is **Best-of-N rather than PPO**; no RL policy is updated.

## Exact KL in both directions

Let `q0(x | template)` denote the candidate proposal and `qN(x | template)` the distribution of the Best-of-N winner for a *fixed trained proxy*. If candidates are i.i.d. and the proxy score is continuous (no score ties), write `F` for its CDF under `q0`:

\[
\frac{dq_N(x)}{dq_0(x)} = N F(f_\theta(x))^{N-1}.
\]

The probability-integral transform gives two analytic quantities in **nats**:

\[
D_{KL}(q_N\Vert q_0) = \log N - \frac{N-1}{N},
\qquad
D_{KL}(q_0\Vert q_N) = N-1-\log N.
\]

The file logs both as `kl_selection_to_proposal` and `kl_proposal_to_selection`, respectively. These are conditional **selection-distribution KLs**, not categorical PPO-policy KLs, and do not imply the learned proxy got better. They require the specified i.i.d. sampling / tie-free assumptions; if finite-precision ties become material, the formulas need a tie-aware correction. The plot uses the first (forward) KL, consistent with the primary BoN convention in Gao et al.

## Evaluation and uncertainty

Default: 64 distinct held-out templates, two random candidate banks per template, three proxy-training seeds, nine nested search budgets, and training-label budgets of 100 / 1,000 / 10,000. We report raw per-template scores, `gold_change_vs_n1`, `proxy_change_vs_n1`, changes from untouched geometry, and gold submetrics. Training diagnostics include validation MSE, MAE and Pearson correlation. Validation-selected checkpoints are stored under the **gitignored** experiment output directory.

The `summary_ci.csv` uses a **hierarchical bootstrap** over proxy-training seeds and test templates, clustering candidate-bank repetitions within a template. With only three seeds, uncertainty over training initialization remains poorly estimated; these intervals are screening estimates, not publication-ready significance tests. Use more seeds, larger test families, additional independent proposal distributions and predeclared contrasts before making research claims.

A convincing proxy overoptimization finding requires increasing predicted score with N, followed by a reproducible degradation in *paired held-out gold* scores at higher N. An increase in both is *not* reward hacking. A proxy/gold correlation of 0.81 reported elsewhere is not evidence of an overoptimization law by itself.

## Commands

From the repository root:

```bash
python -m pytest -q tests/test_layoutpilot_learned_proxy.py
python -m layoutpilot.evaluate_learned_proxy_hack --smoke \
  --test-templates 64 --out experiment_outputs/learned_proxy_smoke

# Three seeds, full budget regimes, 64 truly held-out target layouts:
python -m layoutpilot.evaluate_learned_proxy_hack \
  --seeds 0 1 2 --train-budgets 100 1000 10000 \
  --best-of-n 1 2 4 8 16 32 64 128 256 \
  --test-templates 64 --repeats 2 --updates 300 \
  --out experiment_outputs/learned_proxy_study
```

Outputs: `per_template.csv`, `summary_ci.csv`, `proxy_training.csv`, `proxy_gold_vs_kl.png`, `experiment.json` and the selected proxy checkpoints. Outputs are ignored by Git. The smoke flag always labels its results `PIPELINE_SMOKE_TEST`; non-smoke runs are `UNVALIDATED_RESEARCH_RUN` until independently evaluated.

## Limitations

1. Gold is a hand-designed geometry function, not human/aesthetic preference. Proxy training and evaluation share geometric ingredients; no claim of human-alignment measurement is appropriate.
2. Held-out targets are procedurally generated from three base design families, not externally collected design briefs; a second independent test generator would strengthen out-of-distribution evaluation.
3. Analytic KL depends on the search algorithm and N, and is independent of the measured proxy/gold scores. Both KL directions are logged to prevent confusion with PPO drift.
4. Proxy dataset scarcity, model initialization, proposal tails and clipping can all influence measured divergence. Their effects require factorial ablations.

Primary reference: Gao et al., [ICML 2023](https://proceedings.mlr.press/v202/gao23h.html).
