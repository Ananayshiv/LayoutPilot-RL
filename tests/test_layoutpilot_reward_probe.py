"""Tests that discriminate target-blind gold constraints from target proxy Q."""
import numpy as np
import pytest
import torch
from layoutpilot.evaluate_reward_hacking import (
    _intersection_fraction,
    evaluate_baselines,
    evaluate_geometry,
    evaluate_policy,
    gold_metrics,
    initial_policy_kl,
    reference_observations,
)
from layoutpilot.train import ActorCritic


def test_gold_ignores_proxy_target():
    """Same geometry, different targets: proxy changes, independent gold cannot.

    Also construct a proxy-optimal but text-obscured layout: its Q improves
    when scored against its own bad target, but its gold score worsens.
    This would fail if gold accidentally read the target or mirrored Q.
    """
    clean = np.array([[.15, .14, .46, .16],
                      [.15, .40, .64, .30],
                      [.15, .83, .42, .09]], dtype=np.float32)
    occluded = clean.copy()
    occluded[1] = [.15, .15, .64, .55]  # artwork covers headline ink

    clean_target = clean.copy()
    bad_target = occluded.copy()
    a = evaluate_geometry(clean, clean_target)
    b = evaluate_geometry(clean, bad_target)
    assert a['proxy_score'] > b['proxy_score']  # target sensitivity
    assert a['gold_score'] == pytest.approx(b['gold_score'], abs=1e-12)
    assert a['hard_text_occlusion'] == b['hard_text_occlusion']

    # A safe but badly positioned/sized artwork block can score worse on Q
    # than a target-matching overlapping block, even though gold prefers it.
    safe_off_target = clean.copy()
    safe_off_target[1] = [.05, .75, .18, .15]
    proxy_good = evaluate_geometry(occluded, bad_target)
    proxy_bad = evaluate_geometry(safe_off_target, bad_target)
    assert proxy_good['proxy_score'] > proxy_bad['proxy_score']
    assert proxy_good['hard_text_occlusion'] > proxy_bad['hard_text_occlusion']
    assert proxy_good['gold_score'] < proxy_bad['gold_score']


def test_gold_penalizes_hard_occlusion():
    clean = np.array([[.15, .15, .45, .14], [.12, .36, .60, .32],
                      [.15, .82, .40, .10]])
    bad = clean.copy()
    bad[1] = [.16, .16, .50, .55]
    assert gold_metrics(bad)['hard_text_occlusion'] > gold_metrics(clean)['hard_text_occlusion']
    assert gold_metrics(bad)['gold_score'] < gold_metrics(clean)['gold_score']


def test_baselines_share_starts_and_noop_is_zero_delta():
    baselines = evaluate_baselines([10000, 10001], 4)
    assert len(baselines) == 6
    for seed in (10000, 10001):
        rows = [r for r in baselines if r['eval_seed'] == seed]
        assert len({r['initial_proxy'] for r in rows}) == 1
        assert len({r['initial_gold'] for r in rows}) == 1
        noop = next(r for r in rows if r['baseline'] == 'noop')
        assert noop['proxy_delta'] == pytest.approx(0., abs=1e-12)
        assert noop['gold_delta'] == pytest.approx(0., abs=1e-12)
    assert evaluate_baselines([10000, 10001], 4) == baselines


def test_probe_uses_fixed_eval_seeds_and_exposes_per_episode_data():
    model = ActorCritic(width=16)
    a = evaluate_policy(model, [10000, 10001], 3)
    b = evaluate_policy(model, [10000, 10001], 3)
    assert a == b and len(a) == 2
    assert all(0 <= e['gold_score'] <= 1 for e in a)
    assert all(e['proxy_delta'] == pytest.approx(e['proxy_score']-e['initial_proxy']) for e in a)


def test_kl_from_initial_zero_and_positive_after_policy_change():
    model = ActorCritic(width=16)
    states = reference_observations([10000, 10001], 5)
    assert states.shape[0] > 2
    with torch.no_grad():
        ref = model(states)[0].clone()
    assert initial_policy_kl(model, states, ref) == pytest.approx(0., abs=1e-6)
    with torch.no_grad():
        model.actor.bias[0].add_(2.5)
    assert initial_policy_kl(model, states, ref) > 0.01
    assert torch.equal(reference_observations([10000, 10001], 5), states)


def test_area_fraction():
    a = np.array([0, 0, 1, 1]); b = np.array([0, 0, .5, 1])
    assert abs(_intersection_fraction(a, b)-.5) < 1e-12


def test_probe_exports_labeled_baselines_kl_and_scores(tmp_path):
    """Exercise full CSV/PNG/metadata contract with two tiny PPO updates."""
    import csv
    import json
    from layoutpilot.evaluate_reward_hacking import parse_args, run
    args = parse_args(['--seeds', '0', '1', '--steps', '32', '--rollout', '16',
                       '--epochs', '1', '--width', '16', '--horizon', '3',
                       '--eval-episodes', '2', '--out', str(tmp_path)])
    rows = run(args)
    assert len(rows) == 6  # 2 training seeds * (step 0, 16, 32)
    assert all(r['kl_from_initial'] < 1e-6 for r in rows if r['ppo_steps'] == 0)
    with (tmp_path / 'reference_baselines.csv').open() as handle:
        baselines = list(csv.DictReader(handle))
    assert {r['baseline'] for r in baselines} == {'noop', 'random', 'greedy'}
    assert all(float(r['proxy_delta']) == pytest.approx(0., abs=1e-12)
               for r in baselines if r['baseline'] == 'noop')
    assert (tmp_path / 'reward_proxy_vs_gold.png').stat().st_size > 1000
    assert json.loads((tmp_path / 'summary.json').read_text())['experiment_status'] == 'PIPELINE_SMOKE_TEST'
    assert json.loads((tmp_path / 'config.json').read_text())['gold_is_human_preference'] is False
