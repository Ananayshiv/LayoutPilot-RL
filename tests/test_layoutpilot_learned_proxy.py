"""Failure-sensitive tests for the learned reward proxy study."""
import csv
import json
import math
import numpy as np
import pytest
from layoutpilot.learned_proxy import (
    make_templates, proposal_samples, geometry_features, labeled_samples,
    ProxyMLP, fit_proxy, predict_proxy)
from layoutpilot.evaluate_learned_proxy_hack import (
    bon_kl, choose_proxy_only, hierarchical_bootstrap, parse_args, run)


def test_exact_kl_conventions():
    assert bon_kl(1) == (0., 0.)
    for n in (2, 10, 100):
        f, r = bon_kl(n)
        assert f == pytest.approx(math.log(n) - (n-1)/n)
        assert r == pytest.approx(n-1-math.log(n))
        assert f > 0 and r > f
    with pytest.raises(ValueError):
        bon_kl(0)


def test_template_splits_procedural_and_disjoint():
    a = make_templates(12, 10_000)
    b = make_templates(12, 200_000)
    c = make_templates(64, 400_000)
    assert len({r.uid for r in a+b+c}) == 88
    assert len({tuple(r.target.ravel().round(5)) for r in c}) == 64


def test_proposal_and_features():
    t = make_templates(1, 10000)[0]
    x = proposal_samples(t.target, 20, np.random.default_rng(0))
    assert x.shape == (20, 3, 4)
    assert geometry_features(x).shape == (20, 12)
    assert np.isfinite(x).all()
    assert np.all(x[:,:,:2] >= .04-1e-5)
    with pytest.raises(ValueError):
        geometry_features(np.zeros((2,3)))


def test_proxy_selection_does_not_query_gold(monkeypatch):
    import layoutpilot.evaluate_learned_proxy_hack as exp
    def never_gold(*args, **kwargs):
        raise AssertionError('Selection must never call gold oracle')
    monkeypatch.setattr(exp, 'gold_metrics', never_gold)
    assert choose_proxy_only(np.array([.1,.8,.3,.7]), [1,2,3,4]) == {1:0, 2:1, 3:1, 4:1}
    assert choose_proxy_only(np.array([0.1,0.9,0.7]), [1,2,3])[3] == 1
    with pytest.raises(ValueError):
        choose_proxy_only(np.array([0.1,0.2]),[3])


def test_proxy_training_learns_and_validates():
    train = make_templates(3, 10_000)
    val = make_templates(3, 200_000)
    tx, ty = labeled_samples(train, 60, seed=7)
    vx, vy = labeled_samples(val, 24, seed=11)
    model, info = fit_proxy(tx, ty, vx, vy, seed=0, width=16, updates=12)
    ypred = predict_proxy(model, geometry_features(proposal_samples(
        train[0].target, 3, np.random.default_rng(2))))
    assert ypred.shape == (3,)
    assert np.all((ypred>=0)&(ypred<=1))
    assert math.isfinite(info['val_mse']) and info['selected_step'] >= 1


def test_hierarchical_ci_no_fake_independent_repeats():
    rows=[]
    for seed in (0,1):
        for template in (100,101,102):
            for repeat in (0,1):
                rows.append({'proxy_seed':seed,'template_uid':template,'data_budget':100,
                             'best_of_n':2,'repeat':repeat,'gold_score':.5+.1*seed})
    out = hierarchical_bootstrap(rows,metrics=['gold_score'], draws=50)
    assert len(out)==1 and out[0]['n_test_templates']==3
    assert out[0]['mean']==pytest.approx(.55)
    assert out[0]['ci95_low']<=out[0]['ci95_high']


def test_standalone_smoke_produces_heldout_and_two_kl_directions(tmp_path):
    args = parse_args(['--smoke', '--test-templates','50', '--out',str(tmp_path)])
    out = run(args)
    assert len(out)>0
    meta = json.loads((tmp_path/'experiment.json').read_text())
    assert meta['experiment_status']=='PIPELINE_SMOKE_TEST'
    assert len(meta['test_template_uids'])==50
    assert not set(meta['test_template_uids']) & set(meta['train_template_uids'])
    assert (tmp_path/'proxy_gold_vs_kl.png').stat().st_size > 1000
    assert (tmp_path/'proxy_seed0_labels100.pt').exists()
    with (tmp_path/'proxy_training.csv').open() as handle:
        train_rows = list(csv.DictReader(handle))
    assert {'validation_mae', 'validation_pearson_r', 'checkpoint'} <= set(train_rows[0])
    with (tmp_path/'per_template.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==50*5*2
    assert all(float(r['kl_selection_to_proposal'])==pytest.approx(0.) for r in rows if r['best_of_n']=='1')
    assert all(float(r['kl_proposal_to_selection'])>0 for r in rows if int(r['best_of_n'])>1)


def test_selection_freezes_before_any_test_gold_query(monkeypatch):
    """Catch an implementation that quietly ranks candidates using the gold oracle."""
    import layoutpilot.evaluate_learned_proxy_hack as exp
    import torch
    from layoutpilot.evaluate_reward_hacking import gold_metrics as real_gold
    chosen = {'finished': False}
    real_choose = exp.choose_proxy_only

    def audited_select(scores, budgets):
        assert not chosen['finished']
        result = real_choose(scores, budgets)
        chosen['finished'] = True
        return result

    def gold_after_selection_only(rects):
        assert chosen['finished'], 'test gold queried before proxy selection finished'
        return real_gold(rects)

    monkeypatch.setattr(exp, 'choose_proxy_only', audited_select)
    monkeypatch.setattr(exp, 'gold_metrics', gold_after_selection_only)
    model = ProxyMLP(8)
    rows = exp.evaluate_candidate_bank(model, make_templates(1, 400000),
                                       budgets=[1, 2], repeats=1, base_seed=10)
    assert len(rows) == 2
    assert rows[0]['gold_change_vs_n1'] == pytest.approx(0.)
    assert rows[0]['proxy_change_vs_n1'] == pytest.approx(0.)
