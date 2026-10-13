# Preliminary empirical note — learned proxy Best-of-N

This is **not** a publication claim. The experiment was run on 64 procedurally distinct held-out target templates (2 candidate-bank replicates each), **3 independently trained proxy seeds**, 100/1,000/10,000 nested gold-label budgets, and BoN up to 256 candidates. MLP optimization used **120 gradient updates per trained model**, below the default 300. Confidence intervals are 95% hierarchical screening intervals over three proxy seeds and test templates; with just three seeds they should not be treated as definitive inferential evidence.

| Proxy training labels | N=1 predicted / gold | N=256 predicted / gold | Paired gold change N=256 − N=1, 95% interval |
|---|---|---|---|
| 100 | 0.713 / 0.723 | 0.769 / 0.725 | +0.002 [−0.061, +0.047] |
| 1,000 | 0.713 / 0.723 | 0.765 / 0.744 | +0.021 [−0.030, +0.071] |
| 10,000 | 0.707 / 0.723 | 0.755 / 0.742 | +0.020 [−0.030, +0.070] |

**Interpretation:** Proxy optimization is measurable and increases average predicted score under all three data regimes, but **there is no robust multi-seed evidence of gold-score degradation** at these settings. A single 100-label seed exhibited deterioration, but that behavior was not consistent across seeds. We therefore cannot claim a reward-overoptimization finding or an empirical scaling law.

Configuration is recorded in `experiment.json` in the separate results archive. To reproduce the run from the source checkout:

```bash
python -m layoutpilot.evaluate_learned_proxy_hack \
  --seeds 0 1 2 --train-budgets 100 1000 10000 \
  --best-of-n 1 2 4 8 16 32 64 128 256 \
  --test-templates 64 --updates 120 --val-labels 256 \
  --repeats 2 --bootstrap-draws 300 \
  --out experiment_outputs/learned_proxy_final_three_seed
```

A publishable study needs substantially more independent proxy-training seeds, multiple proposal distributions, larger selection budgets, predeclared comparisons and an external or perceptually grounded gold evaluator.
