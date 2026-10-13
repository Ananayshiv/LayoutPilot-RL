# Learned-reward overoptimization: reproducible Best-of-N

This is an **engineered-gold proxy study** inspired by Gao, Schulman & Hilton,
*Scaling Laws for Reward Model Overoptimization* (OpenAI preprint 2022;
ICML 2023). It is **not** a reproduction of preference-model RLHF, an experiment
with human ratings, nor an automatic demonstration of Goodhart's law.

## Explicit research design

The fixed gold function is `evaluate_reward_hacking.gold_metrics(rects)['gold_score']`
(occlusion, reserved-region infringement, whitespace distribution and minimum
text capacity). Training labels come exclusively from this gold function.
Candidates are chosen *only* by a separately fitted 12-feature MLP reward proxy;
gold scores are consulted only for training and post-selection diagnostics and
oracle references. Unit tests verify the vectorized gold implementation matches
the unchanged scalar implementation.

For **each budget B**, `floor(0.8 B)` examples from disjoint training templates
fit the regressor; the remaining `B-floor(0.8B)` examples from separate validation
templates select an early-stopped checkpoint. **Total supervised label budget is B**,
not B plus validation labels. Validation labels remain clean; optional Gaussian
label noise of standard deviation `--label-noise` is added **only to training
labels**, with clamping to [0,1]. Nested train/validation prefixes are used across
label budgets. Test-set Pearson r and ordinary R² are measured on another
disjoint, labeled *diagnostic* template family, **before any Best-of-N search**.
Those diagnostic labels are extra evaluation expenditure, not training labels.
R² may legitimately be negative.

Each proxy/search seed pair is independent. For a fixed search seed and held-out
template, all regimes reuse the same nested i.i.d. candidate pool, allowing
paired comparisons. Search ranks only learned proxy scores. A true-gold oracle
selects the maximal gold score in the same candidate pool **as a post-hoc
reference**. A random baseline is its first candidate, corresponding to N=1.
The two evaluation template sets are independent of training, validation and
diagnostic templates and of one another:

- **Discovery** set: selects the Best-of-N budget with the highest *mean gold*
  and separately the budget with the highest *mean proxy*.
- **Confirmation** set: evaluates the difference in gold at those two
  preselected budgets. No hyperparameter or search-budget selection uses its gold.

The **predeclared primary statistic** is

\[
\Delta_{\mathrm{confirm}}=
G_{\mathrm{confirm}}(N^*_{G,\mathrm{discovery}})-
G_{\mathrm{confirm}}(N^*_{P,\mathrm{discovery}}).
\]

This operationalizes "**Max Gold minus Peak Gold**": maximum-gold search budget
as selected on discovery minus gold at the peak-proxy search budget, both
scored on independent confirmation layouts. A positive gap is consistent with
proxy overoptimization; the interval must exclude zero for stronger evidence.
Selecting an empirical maximum on the same confirmation data would cause
optimistic bias; this protocol avoids that. The statistic may be negative.

### KL conventions

For an i.i.d. continuous proxy-score proposal (and a selection distribution on
augmented samples if randomized tie breaking is included),

\[
D_{KL}(q_N\Vert q_0)=\ln N-\frac{N-1}{N},\qquad
D_{KL}(q_0\Vert q_N)=N-1-\ln N.
\]

The plot uses `sqrt(KL(q_N || q_0))` as in the BoN scaling analysis. Both KL
directions are in `per_template.csv` and `curves.csv`. **Caveat:** actual geometry
candidates are finite and can have duplicate or quantized network scores; these
are idealized continuous-rank KL values, not empirically measured policy KL.
They are not PPO policy KL, and the reverse KL can be extremely large.

95% confidence intervals use hierarchical paired bootstraps across independently
seeded proxy/search pairs and held-out templates. These are exploratory intervals
with only 10 seed pairs; multiple-comparison adjustments are not implied.
The gold oracle is a diagnostic upper bound within each sampled candidate bank,
not a deployable proxy-only optimizer.

## Full execution command

Run from the project root. For free Colab, select T4 if available and use
`--device cuda`; replace with `cpu` if a GPU is unavailable. A full run trains
60 small MLPs and evaluates two 64-template test splits; it is substantially
more expensive than the smoke check and may exceed a single free session.

```bash
python -m layoutpilot.evaluate_reward_hacking_rigorous \
  --label-budgets 256 512 1024 2048 4096 8192 \
  --best-of-n 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
  --proxy-seeds 0 1 2 3 4 5 6 7 8 9 \
  --search-seeds 100 101 102 103 104 105 106 107 108 109 \
  --train-templates 32 --val-templates 16 \
  --diagnostic-templates 16 --diagnostic-labels 512 \
  --discovery-templates 64 --confirmation-templates 64 \
  --label-noise 0.10 --width 32 --epochs 120 --patience 12 \
  --batch-size 128 --learning-rate 0.001 \
  --proposal-scale 1.0 --broad-prob 0.20 \
  --inference-batch-size 8192 \
  --bootstrap-draws 1000 --bootstrap-seed 2026 \
  --device cuda --out experiment_outputs/rigorous_reward_hacking
```

A fast correctness check (not scientific evidence):

```bash
python -m layoutpilot.evaluate_reward_hacking_rigorous --smoke \
  --out experiment_outputs/rigorous_reward_hacking_smoke
python -m pytest -q tests/test_layoutpilot_reward_hacking_rigorous.py
```

Regenerate the plot from CSV without retraining:

```bash
python -m layoutpilot.plot_reward_hacking_rigorous \
  --run-dir experiment_outputs/rigorous_reward_hacking
```

## Outputs and interpretation

| File | Purpose |
|---|---|
| `proxy_quality.csv` | R², Pearson r, 80/20 counts, clean-val loss, selected epoch per proxy seed/budget |
| `per_template.csv` | Proxy-selected layouts, gold, oracle, random, both KLs, seed and template IDs |
| `curves.csv` | Paired hierarchical confidence intervals for score and reference curves |
| `primary_gap.csv` | Confirmed max-minus-peak gold and confidence bounds |
| `summary.json` | All hyperparameters, template splits, primary metric and provenance |
| `summary_table.md` | Human-readable R² vs. search outcome |
| `proxy_gold_vs_sqrt_kl.png` | Gold/proxy panels with random and gold-oracle controls |

Reporting a positive confirmed gap is meaningful only alongside its interval,
seed-wise variability, baseline performance and clean held-out proxy fit.
Data scarcity, label noise or reduced network width **can** induce reward-model
errors but **do not guarantee** a Gold-score turnover. Report negative or null
results unchanged.

**Primary source:** Gao, Schulman, Hilton (2023),
https://proceedings.mlr.press/v202/gao23h.html
