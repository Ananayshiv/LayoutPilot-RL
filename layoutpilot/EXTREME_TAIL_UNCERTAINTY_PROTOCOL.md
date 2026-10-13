# Extreme-Tail and Uncertainty-Aware Proxy Optimization

**Status:** implemented research protocol, **not a demonstrated Goodhart effect**. The fixed "gold" is an engineered score for geometric text occlusion, reserved regions and whitespace, **not human preferences**. The learned proxy is trained with gold labels, but all search candidate selection uses learned predictions only. The gold oracle is a post-selection reference, never a selector used by the treatments.

## Research questions and controls

1. **Extreme-tail failure.** As Best-of-N increases, does conditional prediction error among the proxy's highest-scored candidates grow, or do rank correlations (Spearman and tied Kendall tau-b) deteriorate? Compute top-K (by predicted proxy, *not gold*) ranking at the largest N and signed tail bias, MAE and nominal-90% interval coverage in the top 5% of candidates. Keep both top-K and tail fraction fixed across label-budget regimes.
2. **Mitigation.** Compare standard selection `argmax mean` with uncertainty-penalized selection `argmax (mean - lambda * calibrated_uncertainty)` on the exact same nested candidate banks. `lambda` must be fixed before evaluating confirmation data. The primary intervention comparison is paired `gold_conservative - gold_standard` at **predeclared maximum N**, with hierarchical seed/template 95% confidence intervals. Even positive differences do **not** establish that reward hacking existed unless a separately confirmed downturn is also observed.
3. **Generalization.** Repeat the protocol under `base` and `wide_proposals` (different scale and broad-mixture probability). The optional `spatial_jitter` perturbation changes candidate positions and dimensions, then projects them onto the artboard. Identical gold metric; disjoint template families for training, early stopping, calibration, diagnostic, discovery and confirmation.
4. **Null results.** Report every regime, negative R², miscalibrated intervals, absent turnover and failed mitigation. No experiment or plotting function inserts synthetic downward trends.

## Proxy ensemble and label budgeting

Each label budget B includes exactly floor(0.8B) gold-labeled training samples and B-floor(0.8B) validation samples. Half the 20% validation split performs early stopping. The remaining half calibrates standardized absolute ensemble errors to a nominal 90% interval radius using an order statistic. Each ensemble member independently bootstraps the **same** training-label pool (no extra queried labels); different parameter seeds and stochastic batch orders yield multiple predictors. An explicit variance floor protects against zero ensemble disagreement. The raw ensemble standard deviation is a proxy for epistemic uncertainty, **not** the full predictive distribution. The calibration interval is an empirical marginal construction under exchangeability, **not** a per-candidate coverage guarantee, particularly after top-N selection or distribution shift.

Let `mu(x)` be the ensemble mean, `s(x)=max(std_members(x),sigma_floor)`, and `c90` the calibration-set 90th percentile of `|gold-mu|/s` using the finite-sample order statistic. Define `sigma_hat(x)=c90*s(x)` and the conservative acquisition function

\[
 S_\lambda(x)=\mu(x)-\lambda\sigma_{hat}(x).
\]

`lambda=0` reproduces standard Best-of-N. This is a **penalty**, not a guaranteed lower confidence bound. Calibration is also checked independently on a diagnostic split before search, and on selected high-scoring candidates afterward. The latter is an important selection-induced shift test.

## Selection, oracle and KL conventions

Each candidate pool is generated i.i.d. conditional on the target template. Prefixes of one pool implement all search budgets, so N=1 is the random baseline and each treatment sees the **same** candidates. Freeze all treatment choices before computing candidate-bank gold values. The gold oracle chooses the maximum gold among the same N candidates only for regret measurement. Regret is oracle gold minus selected gold (nonnegative up to rounding).

For continuous, non-tied acquisition ranks, the idealized **selection-distribution** KL is

\[
 D_{KL}(q_N\Vert q_0)=\log N-\frac{N-1}{N},\quad
 D_{KL}(q_0\Vert q_N)=N-1-\log N.
\]

These are **not** PPO policy KLs or the actual KL of clamped, quantized geometric outputs. The same idealized formula applies to deterministic rankings by standard and conservative continuous acquisitions. On the plot the x-axis is `sqrt(KL(q_N||q_0))`.

## Turnover without cherry-picking

The exploratory gold peak is measured on **discovery** templates only. Choose a budget maximizing discovery gold, freeze that budget, and compare its gold against the predeclared maximum search budget on **confirmation** templates. Report signed difference and hierarchical paired bootstrap interval. Positive interval evidence is only considered if there are at least ten independent training/search seeds and 50 confirmation templates. The `--smoke` mode can never produce a confirmed finding.

## Full experiment (one canonical command)

From repository root, with dependencies from `requirements.txt` installed:

```bash
python -m layoutpilot.evaluate_tail_uncertainty \
  --label-budgets 256 1024 4096 \
  --best-of-n 1 4 16 64 256 1024 8192 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --environments base wide_proposals \
  --train-templates 32 --val-templates 16 \
  --diagnostic-templates 16 --diagnostic-labels 512 \
  --discovery-templates 50 --confirmation-templates 50 \
  --ensemble-members 5 --width 32 --epochs 80 --patience 10 \
  --batch-size 128 --learning-rate 0.001 --label-noise 0.10 \
  --sigma-floor 0.01 --nominal-coverage 0.90 \
  --conservative-lambda 1.0 --top-k 32 --tail-fraction 0.05 \
  --turnover-epsilon 0.01 --proposal-scale 1.0 --broad-prob 0.2 \
  --bootstrap-draws 1000 --device cpu \
  --out experiment_outputs/tail_uncertainty
```

For the pipeline test (not valid for making scientific claims):

```bash
python -m layoutpilot.evaluate_tail_uncertainty \
  --smoke --out experiment_outputs/tail_uncertainty_smoke
```

## Outputs

- `proxy_quality.csv`: independent diagnostic R²/r, interval coverage, split allocation and calibration radius.
- `per_template.csv`: both selectors, post-selection gold, regret, oracle, random baseline, interval coverage by N, budget, environment, template and independent seed.
- `tail_per_template.csv` and `tail_summary.csv`: top-K rank correlation, top-tail signed bias, MAE, calibrated coverage and seed/template intervals where defined.
- `curves.csv`: paired hierarchical 95% intervals for gold/proxy/regret versus idealized sqrt-KL.
- `mitigation_turnover.csv`: primary conservative-minus-standard gold and separately confirmed turnover; null statuses preserved.
- `gold_proxy_regret_vs_kl.png`: gold, proxy and oracle regret against search strength for each environment.
- `tail_reliability_by_budget.png`: top-tail MAE, Spearman ranking and interval coverage against labeled-data budget.
- `summary.json`: exact arguments, template IDs, experiment status, and evidence caveats.

## Reporting limitations

- This is a study of **synthetic geometric reward optimization**. It does not measure aesthetic quality or general alignment.
- The shifted protocol changes candidate distributions, not the learned gold function. It is an OOD proposal test, **not a new independent task or human-rated second environment**.
- Deep ensembles can be confidently wrong in shared blind spots; disagreement is not guaranteed to increase with epistemic error. The mitigation may fail.
- Bootstrap assumes independent seed pairs and exchangeable held-out templates within each environment. It does not substitute for diverse external dataset validation.
- Do not choose `lambda`, tail-K, shift severity or label budgets based on confirmation scores. If tuning is needed, introduce a separate development split and rerun confirmation on untouched data.

**Foundational context:** Gao, Schulman & Hilton (ICML 2023), *Scaling Laws for Reward Model Overoptimization*; Lakshminarayanan, Pritzel & Blundell (NeurIPS 2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. This is an independent layout study, not a reproduction of those papers' empirical findings.
