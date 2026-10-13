"""Reproducible, small-scale reward-model overoptimization experiment.

Run from repository root:
    python -m layoutpilot.evaluate_reward_hacking_rigorous --smoke
    python -m layoutpilot.evaluate_reward_hacking_rigorous --help

The gold is a FIXED ENGINEERED LAYOUT METRIC, not human preference.
This study is inspired by Gao, Schulman & Hilton (ICML 2023), NOT a
replication of their language-model experiments. Best-of-N alone is optimized;
no PPO occurs in this script. Larger search budgets need not cause a downturn.

Key controls:
- Exactly B gold labels per proxy regime: floor(.8B) train + remainder val.
- Validation selects checkpoints; an independent diagnostic split reports r/R2.
- Noise, if requested, affects training labels only (not validation or gold).
- Nested training/validation datasets across B, shared proposal pools across B.
- Candidate ranking uses ONLY proxy outputs. Gold evaluation comes afterward.
- Gold oracle is a REFERENCE ONLY; it never selects proxy candidates.
- Gold-max budget chosen on discovery templates, tested on confirmation templates.
- Independent replicates are paired (proxy_seed[i], search_seed[i]); both sets
  are independently seeded. Bootstrap resamples replicate pairs, not candidate
  layouts or selection budgets as though independent.
- KL(q_N || q_0) is analytic for i.i.d. continuous proxy ranks:
  log(N)-(N-1)/N.  KL(q_0 || q_N)=(N-1)-log(N).
  Ties and finite/quantized candidate support make these IDEALIZED values;
  they are *not* measured PPO policy KLs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from layoutpilot.learned_proxy import (
    make_templates, labeled_samples, geometry_features, ProxyMLP, predict_proxy,
)
from layoutpilot.evaluate_reward_hacking import gold_metrics


DEFAULT_LABEL_BUDGETS = (256, 512, 1024, 2048, 4096, 8192)
DEFAULT_SEARCH_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def score_kl(n: int) -> tuple[float, float]:
    """(D_KL(qN||q0), D_KL(q0||qN)), nats, ideal continuous rank model."""
    if n < 1:
        raise ValueError('n must be >= 1')
    return math.log(n) - (n - 1) / n, (n - 1) - math.log(n)


def pearson_and_r2(y: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y, np.float64).ravel()
    p = np.asarray(predictions, np.float64).ravel()
    if y.size < 2 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Need >=2 finite, paired regression observations')
    mse = float(np.mean((p - y) ** 2))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((p - y) ** 2) / denom) if denom > 1e-15 else None
    r = (float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 1e-12 and np.std(p) > 1e-12
         else None)
    return {'pearson_r': r, 'r2': r2, 'mse': mse, 'mae': float(np.abs(p - y).mean())}


def _batch_gold(rects: np.ndarray) -> np.ndarray:
    """Vectorized, equation-equivalent gold_score of gold_metrics(), Bx3x4.

    Strict regression tests compare this with the unmodified scalar gold metric.
    It is called only after learned proxy selections have been frozen.
    """
    r = np.asarray(rects, np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 4) or not np.isfinite(r).all():
        raise ValueError('expected finite (batch, 3, 4) rectangles')
    if len(r) == 0:
        return np.empty(0, np.float64)
    h, art, c = r[:, 0], r[:, 1], r[:, 2]

    def inset(z):
        return np.stack([z[:, 0] + .06*z[:, 2], z[:, 1]+.18*z[:, 3],
                         np.maximum(1e-6, .88*z[:, 2]),
                         np.maximum(1e-6, .64*z[:, 3])], axis=1)

    def overlap_fraction(a, b):
        x0 = np.maximum(a[:, 0], b[:, 0]); y0 = np.maximum(a[:, 1], b[:, 1])
        x1 = np.minimum(a[:, 0]+a[:, 2], b[:, 0]+b[:, 2]);
        y1 = np.minimum(a[:, 1]+a[:, 3], b[:, 1]+b[:, 3])
        return np.maximum(0., x1-x0)*np.maximum(0., y1-y0)/np.maximum(1e-8, a[:, 2]*a[:, 3])

    ih, ic = inset(h), inset(c)
    hard = (.5*np.minimum(1., overlap_fraction(ih, art)+overlap_fraction(ih, c))
            +.5*np.minimum(1., overlap_fraction(ic, art)+overlap_fraction(ic, h)))
    mast = np.broadcast_to([0., 0., 1., .075], (len(r), 4))
    foot = np.broadcast_to([0., .925, 1., .075], (len(r), 4))
    reserved = np.mean([np.maximum(overlap_fraction(r[:, i], mast),
                                   overlap_fraction(r[:, i], foot)) for i in range(3)], axis=0)

    coords = np.linspace(.04 + .92/32, .96 - .92/32, 16)
    yy, xx = np.meshgrid(coords, coords, indexing='ij')
    occ = np.zeros((len(r), 16, 16), dtype=bool)
    for i in range(3):
        x, y, w, ht = (r[:, i, j, None, None] for j in range(4))
        occ |= (xx[None] >= x) & (xx[None] < x + w) & (yy[None] >= y) & (yy[None] < y + ht)
    white = ~occ
    quadrants = np.stack([white[:, :8, :8].mean((1, 2)), white[:, :8, 8:].mean((1, 2)),
                          white[:, 8:, :8].mean((1, 2)), white[:, 8:, 8:].mean((1, 2))], axis=1)
    imbalance = np.clip(quadrants.std(axis=1)/.35, 0., 1.)
    shortfall = np.clip((.20-white.mean(axis=(1, 2)))/.20, 0., 1.)
    capacity = np.clip((np.maximum(0, .42-h[:, 2])/.42 + np.maximum(0, .12-h[:, 3])/.12
                       + np.maximum(0, .26-c[:, 2])/.26 + np.maximum(0, .07-c[:, 3])/.07)/4,
                       0., 1.)
    return np.clip(1 - .40*hard - .20*reserved - .20*imbalance -.10*shortfall -.10*capacity, 0., 1.)


def candidate_bank(target: np.ndarray, n: int, rng: np.random.Generator,
                   *, proposal_scale: float = 1., broad_prob: float = .20) -> np.ndarray:
    """Vectorized version of existing mixed Gaussian/broad proposal.

    Identical proposal family, no gold/proxy involvement. Vectorization makes
    N=8192 practical without per-layout Python loops.
    """
    if n < 1 or proposal_scale <= 0 or not (0 <= broad_prob <= 1):
        raise ValueError('invalid proposal settings')
    target = np.asarray(target, np.float32).reshape(3, 4)
    noise = rng.standard_normal((n, 3, 4)).astype(np.float32)
    scales = rng.choice(np.array([.35, 1., 2.1], np.float32), n, p=[.3, .4, .3])
    out = target[None] + noise*scales[:, None, None]*proposal_scale*\
          np.array([.18, .18, .21, .12], np.float32)[None, None]
    broad = rng.random(n) < broad_prob
    count = int(broad.sum())
    if count:
        out[broad, :, :2] = rng.uniform(.02, .86, (count, 3, 2))
        out[broad, :, 2:] = rng.uniform(.09, .70, (count, 3, 2))
    out[:, :, 2:] = np.clip(out[:, :, 2:], .08, .86)
    out[:, :, 0] = np.minimum(np.maximum(out[:, :, 0], .04), .96-out[:, :, 2])
    out[:, :, 1] = np.minimum(np.maximum(out[:, :, 1], .04), .96-out[:, :, 3])
    return out.astype(np.float32)


def select_proxy_only(proxy_scores: np.ndarray, budgets: tuple[int, ...]) -> dict[int, int]:
    scores = np.asarray(proxy_scores, np.float64)
    if len(scores) < max(budgets) or not np.isfinite(scores).all():
        raise ValueError('insufficient or invalid proxy predictions')
    # A single nested candidate bank yields paired evaluations across N.
    best, selected = 0, {}
    targets = set(budgets)
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
        if i+1 in targets:
            selected[i+1] = best
    return selected


@dataclass
class TrainConfig:
    width: int
    epochs: int
    batch_size: int
    patience: int
    learning_rate: float
    label_noise: float
    device: str


def fit_budget_proxy(train_pool, val_pool, budget: int, *, seed: int,
                     config: TrainConfig) -> tuple[ProxyMLP, dict[str, Any]]:
    """Exactly B labels including train+val; early stop on CLEAN validation.

    The only noisy labels are train labels; fresh validation targets are
    noiseless, and all held-out gold scores stay untouched.
    """
    n_train = int(.8*budget)
    n_val = budget - n_train
    if n_train < 2 or n_val < 2:
        raise ValueError('budget too small for 80/20 train/val')
    tr_x, tr_y = train_pool[0][:n_train], train_pool[1][:n_train]
    va_x, va_y = val_pool[0][:n_val], val_pool[1][:n_val]
    rng = np.random.default_rng(seed + 1951)
    train_y = np.clip(tr_y + rng.normal(0, config.label_noise, size=len(tr_y)), 0, 1).astype(np.float32)
    torch.manual_seed(seed)
    model = ProxyMLP(config.width).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    tx = torch.tensor(tr_x, dtype=torch.float32, device=config.device)
    ty = torch.tensor(train_y, dtype=torch.float32, device=config.device)
    vx = torch.tensor(va_x, dtype=torch.float32, device=config.device)
    vy = torch.tensor(va_y, dtype=torch.float32, device=config.device)
    gen = torch.Generator().manual_seed(seed + 41)
    best_state, best_epoch, best_loss, stale = None, 0, float('inf'), 0
    history = []
    for epoch in range(1, config.epochs+1):
        model.train()
        order = torch.randperm(len(tx), generator=gen).tolist()
        for start in range(0, len(order), config.batch_size):
            idx = order[start:start+config.batch_size]
            loss = (model(tx[idx]) - ty[idx]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            vloss = float((model(vx) - vy).square().mean().item())
        history.append(vloss)
        if vloss < best_loss - 1e-7:
            best_state = copy.deepcopy(model.state_dict()); best_loss = vloss
            best_epoch = epoch; stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError('early stopping failed to select a checkpoint')
    model.load_state_dict(best_state)
    model.eval()
    return model, {'budget': budget, 'train_labels': n_train, 'val_labels': n_val,
                   'label_noise_sigma': config.label_noise, 'best_epoch': best_epoch,
                   'epochs_run': len(history), 'early_stopped': len(history) < config.epochs,
                   'validation_mse': best_loss, 'final_epoch_val_mse': history[-1]}


def _write_csv(path: Path, data: list[dict]) -> None:
    if not data:
        raise ValueError('cannot write an empty CSV')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def _no_nan(obj):
    """Standards-compliant JSON: don't emit NaN/Infinity tokens."""
    if isinstance(obj, dict):
        return {k: _no_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_no_nan(x) for x in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _mean_bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int):
    """Compute mean/CI from paired, independent replicate units."""
    arr = np.asarray(values, float).reshape(-1)
    if len(arr) == 0:
        raise ValueError('empty statistic')
    idx = rng.integers(0, len(arr), size=(draws, len(arr)))
    boot = arr[idx].mean(axis=1)
    return float(arr.mean()), *map(float, np.quantile(boot, [.025, .975]))


def curve_summary(rows: list[dict], draws: int, seed: int) -> list[dict]:
    """Hierarchical bootstrap across paired seeds and held-out templates.

    All configurations use the same layout template IDs. Keep template draws
    shared across seeds within a bootstrap replicate and re-use candidates
    across BoN search budgets, so inference is paired.
    """
    rng = np.random.default_rng(seed)
    result = []
    combos = sorted({(r['budget'], r['subset'], r['best_of_n']) for r in rows})
    for budget, subset, n in combos:
        grp = [r for r in rows if (r['budget'], r['subset'], r['best_of_n']) == (budget, subset, n)]
        pairs = sorted({r['pair_id'] for r in grp})
        templates = sorted({r['template_uid'] for r in grp})
        if len(grp) != len(pairs)*len(templates):
            raise ValueError('missing/duplicated paired seed-template evaluation')
        index = {(r['pair_id'], r['template_uid']): r for r in grp}
        item = {'budget': budget, 'subset': subset, 'best_of_n': n,
                'sqrt_kl_qn_q0': math.sqrt(score_kl(n)[0]),
                'kl_qn_q0': score_kl(n)[0], 'kl_q0_qn': score_kl(n)[1],
                'seed_pairs': len(pairs), 'templates': len(templates)}
        # Resample both axes and retain paired observations within each cell.
        si = rng.integers(0, len(pairs), (draws, len(pairs)))
        ti = rng.integers(0, len(templates), (draws, len(templates)))
        for key in ('proxy_score', 'gold_score', 'oracle_gold', 'random_gold',
                    'random_proxy', 'gold_delta_from_random'):
            matrix = np.asarray([[index[p, t][key] for t in templates] for p in pairs], float)
            samples = np.array([matrix[np.ix_(si[j], ti[j])].mean() for j in range(draws)])
            item[key] = float(matrix.mean())
            item[key+'_ci_low'], item[key+'_ci_high'] = map(float, np.quantile(samples, [.025,.975]))
        result.append(item)
    return result


def primary_gap(rows: list[dict], budgets: tuple[int,...],
                draws: int, seed: int) -> list[dict]:
    """Predeclared metric: confirmed gold at DEV-best N minus at proxy-peak N.

    Select gold-best and proxy-peak search budgets using discovery templates.
    Never choose them from confirmation data. A negative observed gap remains
    negative; no synthetic 'Goodhart' claim is forced.
    """
    rng = np.random.default_rng(seed)
    output = []
    for size in sorted({r['budget'] for r in rows}):
        pairs = sorted({r['pair_id'] for r in rows if r['budget'] == size})
        selected = []
        for pid in pairs:
            dev = [r for r in rows if r['budget'] == size and r['pair_id'] == pid and r['subset']=='discovery']
            confirm = [r for r in rows if r['budget'] == size and r['pair_id'] == pid and r['subset']=='confirmation']
            dev_gold = {n: np.mean([r['gold_score'] for r in dev if r['best_of_n']==n]) for n in budgets}
            dev_proxy = {n: np.mean([r['proxy_score'] for r in dev if r['best_of_n']==n]) for n in budgets}
            gold_best = max(budgets, key=lambda n: (dev_gold[n], -n))
            proxy_peak = max(budgets, key=lambda n: (dev_proxy[n], n))
            gold_by_template = {n: {r['template_uid']: r['gold_score'] for r in confirm
                                     if r['best_of_n']==n} for n in (gold_best, proxy_peak)}
            tids = sorted(gold_by_template[gold_best])
            if len(tids) < 1 or set(tids) != set(gold_by_template[proxy_peak]):
                raise ValueError('missing confirmation data')
            diff = np.array([gold_by_template[gold_best][t] - gold_by_template[proxy_peak][t]
                             for t in tids], dtype=float)
            selected.append({'pair_id': pid, 'gold_best_n_discovery': gold_best,
                             'proxy_peak_n_discovery': proxy_peak,
                             'gold_gap_discovery': float(dev_gold[gold_best]-dev_gold[proxy_peak]),
                             'confirm_gap': float(diff.mean()), 'template_diffs': diff.tolist()})
        # Two-stage bootstrap: independently sampled replicate pairs and
        # within-pair confirmation templates. Chosen budgets are FROZEN.
        values = np.array([s['template_diffs'] for s in selected])
        si = rng.integers(0, len(selected), (draws, len(selected)))
        boot = []
        for idx in si:
            boot.append(np.mean([values[i, rng.integers(0, values.shape[1], values.shape[1])].mean()
                                 for i in idx]))
        lo, hi = map(float, np.quantile(boot, [.025, .975]))
        output.append({'budget': size,
                       'metric': 'max_gold_minus_peak_proxy_gold_confirmed',
                       'definition': 'gold on independent confirmation templates at discovery-best-gold N, minus gold at discovery-peak-proxy N',
                       'estimate': float(values.mean()), 'ci95_low': lo, 'ci95_high': hi,
                       'positive_gap_supported_by_ci': bool(lo > 0),
                       'replicates': [{k:v for k,v in x.items() if k != 'template_diffs'}
                                      for x in selected]})
    return output


def plot_curves(stats: list[dict], out: Path, *, status: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    discovery = [r for r in stats if r['subset'] == 'confirmation']
    sizes = sorted({r['budget'] for r in discovery})
    for size in sizes:
        points = sorted((r for r in discovery if r['budget'] == size),
                        key=lambda r: r['best_of_n'])
        xs = np.array([r['sqrt_kl_qn_q0'] for r in points])
        for axis, key in ((axes[0], 'gold_score'), (axes[1], 'proxy_score')):
            mean = np.array([r[key] for r in points])
            lower = np.array([r[key+'_ci_low'] for r in points])
            upper = np.array([r[key+'_ci_high'] for r in points])
            line, = axis.plot(xs, mean, marker='.', markersize=4,
                              label=f'{size:,} labels')
            axis.fill_between(xs, lower, upper, alpha=.12, color=line.get_color())
    # Gold oracle is max gold in the SAME nested candidate pools.
    reference = sorted((r for r in discovery if r['budget'] == sizes[-1]),
                       key=lambda r: r['best_of_n'])
    x = [r['sqrt_kl_qn_q0'] for r in reference]
    axes[0].plot(x, [r['oracle_gold'] for r in reference], 'k--', lw=1.8,
                 label='True-gold oracle (reference only)')
    axes[0].axhline(float(np.mean([r['random_gold'] for r in reference])),
                    color='gray', linestyle=':', label='Random single proposal')
    axes[1].axhline(float(np.mean([r['random_proxy'] for r in reference])),
                    color='gray', linestyle=':', label='Random single proposal')
    axes[0].set_ylabel('Independent engineered gold score')
    axes[1].set_ylabel('Learned proxy score')
    for ax in axes:
        ax.set_xlabel(r'$\sqrt{D_{KL}(q_N\Vert q_0)}$ (idealized BoN rank model)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(f'{status}: learned-proxy Best-of-N | independent confirmation layouts', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _validate(args) -> tuple[tuple[int, ...], tuple[int, ...]]:
    budgets = tuple(sorted(set(args.label_budgets)))
    search = tuple(sorted(set(args.best_of_n)))
    if not budgets or min(budgets) < 10 or not search or min(search) != 1:
        raise ValueError('budgets must be >=10 and search budgets include N=1')
    if len(args.proxy_seeds) == 0 or len(args.proxy_seeds) != len(args.search_seeds):
        raise ValueError('provide equally many independently seeded proxy/search pairs')
    if len(set(args.proxy_seeds)) != len(args.proxy_seeds) or len(set(args.search_seeds)) != len(args.search_seeds):
        raise ValueError('all proxy and all search seeds must be unique')
    if not args.smoke and len(args.proxy_seeds) < 10:
        raise ValueError('full experiment requires >=10 independent proxy/search seed pairs')
    if not args.smoke and (args.discovery_templates < 50 or args.confirmation_templates < 50):
        raise ValueError('full experiment requires >=50 discovery and >=50 confirmation templates')
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.width < 1:
        raise ValueError('invalid training hyperparameters')
    if args.label_noise < 0 or args.proposal_scale <= 0 or not (0 <= args.broad_prob <= 1):
        raise ValueError('invalid noise / proposal parameters')
    if args.bootstrap_draws < 50:
        raise ValueError('bootstrap draws must be >=50')
    return budgets, search


def run(args) -> dict:
    label_budgets, search_budgets = _validate(args)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(width=args.width, epochs=args.epochs,
                      batch_size=args.batch_size, patience=args.patience,
                      learning_rate=args.learning_rate,
                      label_noise=args.label_noise, device=args.device)
    train_templates = make_templates(args.train_templates, 10_000)
    val_templates = make_templates(args.val_templates, 110_000)
    diagnostic_templates = make_templates(args.diagnostic_templates, 210_000)
    dev_templates = make_templates(args.discovery_templates, 310_000)
    confirm_templates = make_templates(args.confirmation_templates, 410_000)
    all_uids = [x.uid for group in (train_templates, val_templates, diagnostic_templates,
                                     dev_templates, confirm_templates) for x in group]
    if len(set(all_uids)) != len(all_uids):
        raise ValueError('template sets overlap')

    fit_rows = []
    trained: dict[tuple[int,int], ProxyMLP] = {}
    # PHASE 1: train *all* models and report independent diagnostic R2
    # BEFORE ANY BEST-OF-N CANDIDATE SEARCH.
    for pair, proxy_seed in enumerate(args.proxy_seeds):
        n_max = max(label_budgets)
        train_max = int(.8*n_max); val_max = n_max-train_max
        train_pool = labeled_samples(train_templates, train_max, seed=proxy_seed+5031)
        val_pool = labeled_samples(val_templates, val_max, seed=proxy_seed+6031)
        # Diagnostic labels do not count toward training N; they are collected
        # from a disjoint template family and NEVER used for early stopping.
        diag_x, diag_y = labeled_samples(diagnostic_templates, args.diagnostic_labels,
                                         seed=proxy_seed+7031)
        for budget in label_budgets:
            model, diag = fit_budget_proxy(train_pool, val_pool, budget,
                                           seed=proxy_seed+budget*17, config=cfg)
            predicted = predict_proxy(model, diag_x, device=args.device)
            quality = pearson_and_r2(diag_y, predicted)
            fit_rows.append({'pair_id': pair, 'proxy_seed': proxy_seed, **diag, **quality})
            trained[pair, budget] = model
            print(f'PRE-SEARCH proxy_seed={proxy_seed} labels={budget} '
                  f'train={diag["train_labels"]} val={diag["val_labels"]} '
                  f'heldout_r={quality["pearson_r"]!s} '
                  f'heldout_R2={quality["r2"]!s}', flush=True)
    _write_csv(root/'proxy_quality.csv', fit_rows)

    # PHASE 2: proxy-only selection; do not query TEST gold until ALL learned
    # proxy rankings for this candidate pool have been irreversibly frozen.
    rows = []
    for pair, search_seed in enumerate(args.search_seeds):
        for subset, templates in [('discovery', dev_templates),
                                  ('confirmation', confirm_templates)]:
            for template in templates:
                rng = np.random.default_rng(search_seed*1_000_003 + template.uid)
                bank = candidate_bank(template.target, max(search_budgets), rng,
                                      proposal_scale=args.proposal_scale,
                                      broad_prob=args.broad_prob)
                features = geometry_features(bank)
                selections: dict[int, dict[int,int]] = {}
                predictions: dict[int, np.ndarray] = {}
                for size in label_budgets:
                    proxy = predict_proxy(trained[pair,size], features, batch_size=args.inference_batch_size,
                                          device=args.device)
                    # NO gold labels or gold metrics occur in this loop.
                    selections[size] = select_proxy_only(proxy, search_budgets)
                    predictions[size] = proxy
                # TRUE GOLD is evaluated only after proxy choices are frozen.
                gold = _batch_gold(bank)
                oracle = np.maximum.accumulate(gold)
                for size in label_budgets:
                    for n in search_budgets:
                        index = selections[size][n]
                        kl_fwd, kl_rev = score_kl(n)
                        rows.append({'pair_id': pair, 'proxy_seed': args.proxy_seeds[pair],
                                     'search_seed': search_seed,
                                     'budget': size, 'subset': subset,
                                     'template_uid': template.uid, 'best_of_n': n,
                                     'selected_index': index,
                                     'proxy_score': float(predictions[size][index]),
                                     'gold_score': float(gold[index]),
                                     'oracle_gold': float(oracle[n-1]),
                                     'random_gold': float(gold[0]),
                                     'random_proxy': float(predictions[size][0]),
                                     'gold_delta_from_random': float(gold[index]-gold[0]),
                                     'kl_qn_q0': kl_fwd, 'kl_q0_qn': kl_rev,
                                     'sqrt_kl_qn_q0': math.sqrt(kl_fwd)})
        print(f'Finished paired replicate {pair+1}/{len(args.proxy_seeds)}', flush=True)
    _write_csv(root/'per_template.csv', rows)
    curves = curve_summary(rows, args.bootstrap_draws, args.bootstrap_seed)
    primary = primary_gap(rows, search_budgets, args.bootstrap_draws, args.bootstrap_seed+100)
    _write_csv(root/'curves.csv', curves)
    flat_primary = [{k:v for k,v in p.items() if k not in ('replicates', 'definition')}
                    for p in primary]
    _write_csv(root/'primary_gap.csv', flat_primary)

    status = ('PIPELINE_SMOKE_TEST' if args.smoke else 'EMPIRICAL_EXPERIMENT_UNREPLICATED')
    plot_curves(curves, root/'proxy_gold_vs_sqrt_kl.png', status=status)
    table = ['| Labels N (train/val) | Held-out r | Held-out R² | Gold@BoN=1 | Gold@max BoN | Confirmed max−peak gold (95% CI) |',
             '|---:|---:|---:|---:|---:|---:|']
    for size in label_budgets:
        quality = [r for r in fit_rows if r['budget'] == size]
        rr = [r['pearson_r'] for r in quality if r['pearson_r'] is not None]
        r2 = [r['r2'] for r in quality if r['r2'] is not None]
        get = lambda n: next(x for x in curves if x['budget']==size and x['subset']=='confirmation' and x['best_of_n']==n)
        p = next(x for x in primary if x['budget']==size)
        first, last = get(search_budgets[0]), get(search_budgets[-1])
        tr_n, va_n = int(.8*size), size-int(.8*size)
        table.append(f'| {size:,} ({tr_n:,}/{va_n:,}) | {np.mean(rr):.3f} | {np.mean(r2):.3f} | '
                     f'{first["gold_score"]:.3f} | {last["gold_score"]:.3f} | '
                     f'{p["estimate"]:+.3f} [{p["ci95_low"]:+.3f},{p["ci95_high"]:+.3f}] |')
    text = '\n'.join(table) + '\n'
    print('\n'+text, flush=True)
    (root/'summary_table.md').write_text(text, encoding='utf-8')
    summary = {'status': status,
               'method': 'learned proxy, proxy-only Best-of-N; not PPO',
               'citation': 'Gao, Schulman, Hilton, ICML 2023 (inspiration, not reproduction)',
               'gold_definition': 'fixed engineered layout-occlusion/whitespace score, not human preferences',
               'primary_metric': 'max_gold_minus_peak_proxy_gold_confirmed',
               'selection': 'discovery chooses best-gold/peak-proxy budgets; independent confirmation evaluates difference',
               'label_budget': '80% training and 20% clean validation labels counted in N; held-out diagnostic labels separate',
               'experiment_status_warning': 'Label noise and restricted capacity do not guarantee a downturn.',
               'kl_assumption': 'Ideal analytic BoN rank KL for IID continuous proxy scores; not measured KL of discrete geometry marginals',
               'ci_method': '95% paired hierarchical bootstrap over independent proxy/search replicate pairs and templates',
               'hyperparameters': {k:v for k,v in vars(args).items() if k!='out'},
               'train_templates': [t.uid for t in train_templates],
               'validation_templates': [t.uid for t in val_templates],
               'diagnostic_templates': [t.uid for t in diagnostic_templates],
               'discovery_templates': [t.uid for t in dev_templates],
               'confirmation_templates': [t.uid for t in confirm_templates],
               'proxy_quality': fit_rows, 'primary_metric_results': primary,
               'outputs': ['proxy_quality.csv','per_template.csv','curves.csv','primary_gap.csv',
                           'proxy_gold_vs_sqrt_kl.png','summary_table.md','summary.json']}
    (root/'summary.json').write_text(json.dumps(_no_nan(summary), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--label-budgets', type=int, nargs='+', default=list(DEFAULT_LABEL_BUDGETS))
    parser.add_argument('--best-of-n', type=int, nargs='+', default=list(DEFAULT_SEARCH_BUDGETS))
    parser.add_argument('--proxy-seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--search-seeds', type=int, nargs='+', default=list(range(100, 110)))
    parser.add_argument('--train-templates', type=int, default=32)
    parser.add_argument('--val-templates', type=int, default=16)
    parser.add_argument('--diagnostic-templates', type=int, default=16)
    parser.add_argument('--diagnostic-labels', type=int, default=512)
    parser.add_argument('--discovery-templates', type=int, default=64)
    parser.add_argument('--confirmation-templates', type=int, default=64)
    parser.add_argument('--label-noise', type=float, default=.10)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--proposal-scale', type=float, default=1.)
    parser.add_argument('--broad-prob', type=float, default=.20)
    parser.add_argument('--inference-batch-size', type=int, default=8192)
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=2026)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--out', default='experiment_outputs/rigorous_reward_hacking')
    parser.add_argument('--smoke', action='store_true', help='Run a small pipeline check; never interpret as finding')
    args = parser.parse_args(argv)
    if args.smoke:
        args.label_budgets = [256, 512]
        args.best_of_n = [1, 2, 4, 8, 16, 32]
        args.proxy_seeds = [0]
        args.search_seeds = [100]
        args.epochs = 8
        args.patience = 3
        args.diagnostic_labels = 64
        args.discovery_templates = 8
        args.confirmation_templates = 8
        args.bootstrap_draws = 100
    return args


if __name__ == '__main__':
    run(parse_args())
