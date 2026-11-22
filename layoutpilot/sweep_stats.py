"""Torch-free aggregation for multi-seed, multi-budget LayoutPilot sweeps.

Each run directory contains the metrics.json written by layoutpilot.train with a
`per_episode` block. Every run is evaluated on the SAME held-out layout seeds
(20000..), so per-episode scores are paired across methods and seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, stdev

import numpy as np


def load_runs(root: str | Path) -> list[dict]:
    """Collect <root>/<method>/steps_<N>/seed_<S>/metrics.json."""
    runs = []
    for path in sorted(Path(root).glob("*/steps_*/seed_*/metrics.json")):
        m = json.loads(path.read_text())
        if "per_episode" not in m:
            raise ValueError(f"{path} predates per-episode logging; rerun it")
        runs.append({"method": path.parents[2].name, "steps": int(path.parents[1].name.split("_", 1)[1]),
                     "seed": int(path.parent.name.split("_", 1)[1]), "metrics": m})
    if not runs:
        raise FileNotFoundError(f"No runs under {root}/<method>/steps_<N>/seed_<S>/")
    return runs


def gap_closed(initial: float, policy: float, oracle: float) -> float:
    """Fraction of the initial->greedy-oracle improvement the policy achieves."""
    denom = oracle - initial
    if abs(denom) < 1e-12:
        raise ValueError("Oracle does not improve on initial layouts; gap undefined")
    return (policy - initial) / denom


def episode_matrix(runs: list[dict], method: str, steps: int, key: str = "policy") -> np.ndarray:
    """[n_seeds, n_episodes] per-episode scores for one cell of the sweep."""
    rows = [r["metrics"]["per_episode"][key] for r in runs if r["method"] == method and r["steps"] == steps]
    if not rows:
        raise ValueError(f"No runs for {method} @ {steps}")
    lengths = {len(x) for x in rows}
    if len(lengths) != 1:
        raise ValueError("Runs were evaluated on different numbers of episodes")
    return np.asarray(rows, dtype=np.float64)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iterations: int = 5000, seed: int = 0) -> dict:
    """CI for mean(b - a) over paired episodes (inputs: per-episode, seed-averaged)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Paired 1-D episode vectors of equal length required")
    d = b - a
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (iterations, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def summarize(runs: list[dict], iterations: int = 5000) -> dict:
    cells = sorted({(r["method"], r["steps"]) for r in runs}, key=lambda x: (x[1], x[0]))
    table = []
    for method, steps in cells:
        group = [r for r in runs if r["method"] == method and r["steps"] == steps]
        scores = [r["metrics"]["ppo_policy_score"] for r in group]
        gaps = [gap_closed(r["metrics"]["initial_score"], r["metrics"]["ppo_policy_score"],
                           r["metrics"]["greedy_oracle_score"]) for r in group]
        table.append({"method": method, "steps": steps, "n_seeds": len(group),
                      "score_mean": fmean(scores), "score_std": stdev(scores) if len(scores) > 1 else 0.0,
                      "gap_closed_mean": fmean(gaps), "gap_closed_std": stdev(gaps) if len(gaps) > 1 else 0.0})
    ref = runs[0]["metrics"]
    baselines = {k: fmean(ref["per_episode"][k]) for k in ("initial", "random", "greedy_oracle")}
    for r in runs:  # every run must share the same held-out layouts
        for k in ("initial", "greedy_oracle"):
            if not np.allclose(r["metrics"]["per_episode"][k], ref["per_episode"][k]):
                raise ValueError("Runs disagree on held-out layouts; evaluation seeds differ")
    comparisons = []
    methods = sorted({m for m, _ in cells})
    for steps in sorted({s for _, s in cells}):
        present = [m for m in methods if (m, steps) in cells]
        if "ppo" in present and "bc_ppo" in present:
            a = episode_matrix(runs, "ppo", steps).mean(axis=0)
            b = episode_matrix(runs, "bc_ppo", steps).mean(axis=0)
            comparisons.append({"steps": steps, "b": "bc_ppo", "a": "ppo", **paired_bootstrap(a, b, iterations)})
        oracle = np.asarray(ref["per_episode"]["greedy_oracle"])
        for m in present:
            pol = episode_matrix(runs, m, steps).mean(axis=0)
            comparisons.append({"steps": steps, "b": m, "a": "greedy_oracle",
                                **paired_bootstrap(oracle, pol, iterations)})
    return {"n_eval_episodes": len(ref["per_episode"]["initial"]), "baselines": baselines,
            "table": table, "comparisons": comparisons}


def to_markdown(summary: dict) -> str:
    b = summary["baselines"]
    lines = [f"Held-out layouts: {summary['n_eval_episodes']} (seeds 20000+). "
             f"Initial {b['initial']:.4f} · random policy {b['random']:.4f} · "
             f"greedy oracle {b['greedy_oracle']:.4f} (25 reward queries per step).", "",
             "| Method | Env steps | Seeds | Score (mean ± std) | Greedy gap closed |",
             "|---|---:|---:|---:|---:|"]
    for row in summary["table"]:
        lines.append(f"| {row['method']} | {row['steps']:,} | {row['n_seeds']} | "
                     f"{row['score_mean']:.4f} ± {row['score_std']:.4f} | "
                     f"{100 * row['gap_closed_mean']:.1f}% ± {100 * row['gap_closed_std']:.1f}% |")
    lines += ["", "| Steps | Comparison (B − A) | Δ score | 95% CI | Significant |", "|---:|---|---:|---:|:---:|"]
    for c in summary["comparisons"]:
        lines.append(f"| {c['steps']:,} | {c['b']} − {c['a']} | {c['delta']:+.4f} | "
                     f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {'yes' if c['significant'] else 'no'} |")
    lines += ["", "Gap closed = (policy − initial) / (greedy − initial), per seed. CIs: paired bootstrap over "
              "held-out layouts using seed-averaged per-layout scores. The policy needs 0 reward queries at "
              "inference; the greedy oracle needs 25 per step."]
    return "\n".join(lines)
