"""Multi-seed, multi-budget PPO vs BC+PPO sweep with paired statistics.

CPU is enough (small MLP, NumPy simulator). Resumable: completed runs are skipped.

    python -m layoutpilot.sweep --budgets 8192 131072 524288 --seeds 0 1 2 3 4
    python -m layoutpilot.sweep --aggregate-only

Writes runs/sweep/<method>/steps_<N>/seed_<S>/metrics.json and runs/sweep/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from layoutpilot.sweep_stats import load_runs, summarize, to_markdown

METHODS = {"ppo": 0, "bc_ppo": 700}  # value = greedy-expert BC examples (v1 used 700)


def train_cell(root: Path, method: str, steps: int, seed: int, eval_episodes: int, extra: list[str]):
    from layoutpilot.train import parse_args, run  # requires torch
    out = root / method / f"steps_{steps}" / f"seed_{seed}"
    if (out / "metrics.json").exists():
        print(f"[skip] {out}")
        return
    argv = ["--steps", str(steps), "--seed", str(seed), "--out", str(out),
            "--eval-episodes", str(eval_episodes), "--warmstart-examples", str(METHODS[method]), *extra]
    print(f"[run] {method} steps={steps} seed={seed}")
    run(parse_args(argv))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="runs/sweep")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8192, 131072, 524288])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--rollout", type=int, default=512,
                    help="env steps per PPO update; kept at the v1 value so only the budget varies")
    ap.add_argument("--bootstrap-iterations", type=int, default=5000)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if not args.aggregate_only:
        for steps in args.budgets:
            rollout = str(min(args.rollout, steps))
            for method in args.methods:
                for seed in args.seeds:
                    train_cell(root, method, steps, seed, args.eval_episodes, ["--rollout", rollout])
    summary = summarize(load_runs(root), iterations=args.bootstrap_iterations)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary)
    (root / "RESULTS.md").write_text("# LayoutPilot sweep results\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
