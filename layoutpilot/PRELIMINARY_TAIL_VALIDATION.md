# End-to-End Validation — Not a Converged Research Result

The separately archived validation run used three training/search seeds, 50 discovery + 50 confirmation templates per environment, two data budgets (100, 256 labels), a two-model ensemble, six training epochs and Best-of-N values 1, 8, 32, 128. This was intended to test end-to-end reproducibility and interpretation, not demonstrate prevention or a robust turnover.

| Environment | Labels | Conservative minus standard gold at N=128 | Hierarchical 95% CI |
|---|---:|---:|---|
| Base | 100 | +0.0274 | [−0.0038,+0.0626] |
| Base | 256 | −0.0062 | [−0.0441,+0.0303] |
| Wide proposals | 100 | +0.0302 | [−0.0092,+0.0680] |
| Wide proposals | 256 | +0.0242 | [−0.0519,+0.1017] |

All differences have CIs spanning zero. Held-out proxy R² was negative in all six trained models, so these results do **not** evaluate calibrated uncertainty mitigation with a well-fitted proxy. No intervention or turnover claim is supported. A full study requires convergent training, 10+ independent seeds and untampered confirmation templates.
