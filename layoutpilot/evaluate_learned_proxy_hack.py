"""Learned reward proxy vs fixed gold: reproducible Best-of-N research probe.

Pipeline: train a proxy on gold-labeled layouts from TRAIN templates; checkpoint
on separate VALIDATION templates; sample IID proposals on 64+ HELD-OUT templates;
select using proxy ONLY; then evaluate gold on selected layouts. This does not
use PPO. The conditional proposal q0 and the induced BoN selection policy qN
have exactly known KLs when proxy scores have a continuous distribution:
  KL(qN || q0) = log(N) - (N-1)/N
  KL(q0 || qN) = (N-1) - log(N)
These measure KL between SELECTION DISTRIBUTIONS, not a PPO policy's KL.
The KL formula assumes IID candidates and no proxy-score ties (see README).

Run: python -m layoutpilot.evaluate_learned_proxy_hack --smoke
Full: python -m layoutpilot.evaluate_learned_proxy_hack --train-budgets 100 1000 10000
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from layoutpilot.learned_proxy import (
    make_templates, proposal_samples, geometry_features,
    labeled_samples, fit_proxy, predict_proxy)
from layoutpilot.evaluate_reward_hacking import gold_metrics


def bon_kl(n: int) -> tuple[float, float]:
    """Exact forward/reverse conditional BoN-vs-proposal KL, in nats.

    For continuous proxy score F, dq_N/dq_0 = N*F(f(x))^(N-1).
    Therefore E_qN(log ratio)=log N-(N-1)/N,
    and E_q0(-log ratio)=(N-1)-log N.
    """
    if n < 1:
        raise ValueError('BoN candidate count must be positive')
    return math.log(n) - (n - 1) / n, (n - 1) - math.log(n)


def choose_proxy_only(scores: np.ndarray, budgets: list[int]) -> dict[int, int]:
    """Return candidate indices; cannot observe the true gold score by design."""
    a = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError('proxy scores must be finite')
    if not budgets or min(budgets) < 1 or max(budgets) > len(a):
        raise ValueError('budgets must lie within candidate bank')
    # Strict > tie handling picks the earliest best. Actual ties should be
    # checked: the analytic KL requires a continuous score distribution.
    indices = np.zeros(len(a), dtype=np.int64)
    current = 0
    for j in range(1, len(a)):
        if a[j] > a[current]:
            current = j
        indices[j] = current
    return {n: int(indices[n-1]) for n in budgets}


def evaluate_candidate_bank(model, templates, *, budgets, repeats, base_seed, device='cpu'):
    """A proxy-only search; gold is computed AFTER the choice has been frozen."""
    if repeats < 1 or len(templates) < 1:
        raise ValueError('nonempty eval templates/repeats required')
    rows = []
    biggest = max(budgets)
    for template in templates:
        for rep in range(repeats):
            rng = np.random.default_rng(base_seed + template.uid * 10_007 + rep * 1009)
            bank = proposal_samples(template.target, biggest, rng)
            proxy = predict_proxy(model, geometry_features(bank), device=device)
            chosen = choose_proxy_only(proxy, budgets)
            # All oracle calls occur AFTER proxy-only candidate selection.
            initial = gold_metrics(template.target)['gold_score']
            # Reference outcome is the first iid draw in the *same* proposal bank.
            # Compute it only after all candidate indices have been frozen.
            reference_gold = gold_metrics(bank[chosen[1]])['gold_score']
            reference_proxy = float(proxy[chosen[1]])
            for n in budgets:
                idx = chosen[n]
                metrics = gold_metrics(bank[idx])   # never used in choice
                kl_fwd, kl_rev = bon_kl(n)
                rows.append({
                    'template_uid': template.uid, 'template_family': template.family,
                    'repeat': rep, 'best_of_n': n,
                    'proxy_score': float(proxy[idx]),
                    'gold_score': metrics['gold_score'],
                    'gold_delta_from_start': metrics['gold_score'] - initial,
                    'gold_change_vs_n1': metrics['gold_score'] - reference_gold,
                    'proxy_change_vs_n1': float(proxy[idx]) - reference_proxy,
                    'starting_gold_score': initial,
                    'hard_text_occlusion': metrics['hard_text_occlusion'],
                    'whitespace_imbalance': metrics['whitespace_imbalance'],
                    'reserved_text_overlap': metrics['reserved_text_overlap'],
                    'text_capacity_penalty': metrics['text_capacity_penalty'],
                    'kl_selection_to_proposal': kl_fwd,
                    'kl_proposal_to_selection': kl_rev,
                    'selected_index': idx,
                })
    return rows


def hierarchical_bootstrap(rows, *, metrics, draws=500, seed=921):
    """95% paired hierarchical intervals, resampling TRAIN seeds & TEST templates.

    First aggregate repeats within a (proxy_seed, template_uid) pair; avoid
    treating repeated candidates from one template as independent test items.
    """
    if draws < 10:
        raise ValueError('bootstrap draws must be >= 10')
    rng = np.random.default_rng(seed)
    combos = sorted({(r['data_budget'], r['best_of_n']) for r in rows})
    output = []
    for budget, n in combos:
        selected = [r for r in rows if r['data_budget'] == budget and r['best_of_n'] == n]
        seeds = sorted({r['proxy_seed'] for r in selected})
        templates = sorted({r['template_uid'] for r in selected})
        index = {}
        for s in seeds:
            for t in templates:
                group = [r for r in selected if r['proxy_seed'] == s and r['template_uid'] == t]
                if not group:
                    raise ValueError('missing seed-template evaluation block')
                index[s, t] = {m: np.mean([r[m] for r in group]) for m in metrics}
        for metric in metrics:
            matrix = np.array([[index[s,t][metric] for t in templates] for s in seeds])
            samples = np.empty(draws)
            for i in range(draws):
                si = rng.integers(0, len(seeds), size=len(seeds))
                ti = rng.integers(0, len(templates), size=len(templates))
                samples[i] = float(matrix[np.ix_(si, ti)].mean())
            low, high = np.quantile(samples, [.025, .975])
            fwd, rev = bon_kl(n)
            output.append({'data_budget': budget, 'best_of_n': n,
                           'metric': metric, 'mean': float(matrix.mean()),
                           'ci95_low': float(low), 'ci95_high': float(high),
                           'n_proxy_seeds': len(seeds), 'n_test_templates': len(templates),
                           'kl_selection_to_proposal': fwd,
                           'kl_proposal_to_selection': rev})
    return output


def _csv(path, rows):
    if not rows:
        raise ValueError('empty CSV')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot(summary, out, *, status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    metrics = [('proxy_score', 'Optimized learned proxy'),
               ('gold_score', 'Held-out fixed gold')]
    for ax, (metric, label) in zip(axes, metrics):
        for size in sorted({r['data_budget'] for r in summary}):
            points = sorted((r for r in summary if r['data_budget'] == size
                             and r['metric'] == metric), key=lambda r: r['best_of_n'])
            x = np.array([r['kl_selection_to_proposal'] for r in points])
            y = np.array([r['mean'] for r in points])
            lo = np.array([r['ci95_low'] for r in points])
            hi = np.array([r['ci95_high'] for r in points])
            ax.plot(x, y, marker='.', label=f'{size:,} gold labels')
            ax.fill_between(x, lo, hi, alpha=.12)
        ax.set(xlabel=r'$D_{KL}(q_N\|q_0)$ (nats; induced Best-of-N selection)',
               ylabel=label)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(f'{status} | Proxy-only Best-of-N, gold evaluated after selection',fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(args):
    budgets = sorted(set(args.train_budgets))
    bon = sorted(set(args.best_of_n))
    if min(budgets) < 2 or min(bon) < 1 or 1 not in bon:
        raise ValueError('positive training budgets and Best-of-N including N=1 required')
    if args.test_templates < 50:
        raise ValueError('at least 50 held-out templates required')
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError('proxy training seeds must be unique and nonempty')
    torch.set_num_threads(min(2, torch.get_num_threads()))
    train = make_templates(args.train_templates, 10_000)
    val = make_templates(args.val_templates, 200_000)
    test = make_templates(args.test_templates, 400_000)
    uid_sets = [set(x.uid for x in seq) for seq in [train, val, test]]
    assert not (uid_sets[0] & uid_sets[1] or uid_sets[0] & uid_sets[2] or uid_sets[1] & uid_sets[2])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = 'PIPELINE_SMOKE_TEST' if args.smoke else 'UNVALIDATED_RESEARCH_RUN'
    rows, fit_info = [], []
    # A common candidate bank is used for each (template, repeat) across
    # training seeds and data budgets, reducing evaluation variance.
    for seed in args.seeds:
        vx, vy = labeled_samples(val, args.val_labels, seed=seed + 700_000)
        # Nested data regimes ensure that N=100 is a subset of N=1,000,
        # which in turn is a subset of N=10,000 for each training seed.
        all_tx, all_ty = labeled_samples(train, max(budgets), seed=seed + 60_000)
        for size in budgets:
            tx, ty = all_tx[:size], all_ty[:size]
            model, diagnostics = fit_proxy(tx, ty, vx, vy, seed=seed + 1_000,
                                           width=args.width, updates=args.updates,
                                           batch_size=args.batch_size, lr=args.lr,
                                           device=args.device)
            checkpoint_name = f'proxy_seed{seed}_labels{size}.pt'
            torch.save({'state_dict': model.state_dict(), 'width': args.width,
                        'seed': seed, 'n_train_labels': size,
                        'best_validation_step': diagnostics['selected_step']},
                       out / checkpoint_name)
            validation_predictions = predict_proxy(model, vx, device=args.device)
            corr = float(np.corrcoef(validation_predictions, vy)[0, 1]) if (
                np.std(validation_predictions) > 1e-12 and np.std(vy) > 1e-12) else float('nan')
            predicted = evaluate_candidate_bank(
                model, test, budgets=bon, repeats=args.repeats,
                base_seed=987_654, device=args.device)
            for r in predicted:
                r.update({'proxy_seed': seed, 'data_budget': size})
            rows.extend(predicted)
            fit_info.append({'proxy_seed': seed, 'data_budget': size,
                             'n_train_examples': len(tx),
                             'validation_mse': diagnostics['val_mse'],
                             'validation_mae': float(np.abs(validation_predictions - vy).mean()),
                             'validation_pearson_r': corr,
                             'best_training_step': diagnostics['selected_step'],
                             'checkpoint': checkpoint_name})
            print(f'proxy seed={seed} labels={size} '
                  f'val MSE={diagnostics["val_mse"]:.5f} '
                  f'BoN gold N={bon[-1]}: '
                  f'{np.mean([r["gold_score"] for r in predicted if r["best_of_n"]==bon[-1]]):.4f}',
                  flush=True)
    metrics = ['proxy_score', 'gold_score', 'gold_delta_from_start',
               'gold_change_vs_n1', 'proxy_change_vs_n1',
               'hard_text_occlusion', 'whitespace_imbalance']
    stats = hierarchical_bootstrap(rows, metrics=metrics, draws=args.bootstrap_draws)
    _csv(out / 'per_template.csv', rows)
    _csv(out / 'summary_ci.csv', stats)
    _csv(out / 'proxy_training.csv', fit_info)
    _plot(stats, out / 'proxy_gold_vs_kl.png', status=status)
    meta = {'experiment_status': status,
            'method': 'learned-proxy best-of-n selection; NOT PPO',
            'gold_is_human_preference': False,
            'gold_definition': 'target-blind engineered occlusion, whitespace and capacity',
            'KL_convention': 'qN=induced Best-of-N distribution, q0=proposal; both directions analytic in nats under IID continuous proxy scores',
            'CI_method': '95% hierarchical bootstrap over proxy seeds and held-out templates; repeats clustered within template',
            'train_template_uids': [x.uid for x in train],
            'validation_template_uids': [x.uid for x in val],
            'test_template_uids': [x.uid for x in test],
            'evaluation_gold_used_in_selection': False,
            'training_data_budget_design': 'nested labeled datasets within each proxy seed',
            'config': {k: v for k, v in vars(args).items() if k != 'out'},
            'caveat': 'A proxy/gold divergence is not guaranteed or established by a smoke test.'}
    (out / 'experiment.json').write_text(json.dumps(meta, indent=2) + '\n')
    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true', help='tiny training, still >=50 held-out templates')
    p.add_argument('--train-budgets', type=int, nargs='+', default=[100, 1000, 10000])
    p.add_argument('--best-of-n', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--train-templates', type=int, default=32)
    p.add_argument('--val-templates', type=int, default=12)
    p.add_argument('--test-templates', type=int, default=64)
    p.add_argument('--val-labels', type=int, default=512)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--updates', type=int, default=300)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--bootstrap-draws', type=int, default=500)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--out', default='experiment_outputs/learned_proxy_study')
    args = p.parse_args(argv)
    if args.smoke:
        args.train_budgets = [100, 1000]
        args.best_of_n = [1, 2, 4, 8, 16]
        args.seeds = [0]
        args.updates = 20
        args.val_labels = 96
        args.repeats = 1
        args.bootstrap_draws = 100
    return args


if __name__ == '__main__':
    run(parse_args())
