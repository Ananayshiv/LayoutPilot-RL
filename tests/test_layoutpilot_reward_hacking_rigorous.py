"""Mathematical, data-integrity and end-to-end tests for rigorous BoN study."""
import csv
import json
import math
import numpy as np
import pytest

from layoutpilot.evaluate_reward_hacking import gold_metrics
from layoutpilot.learned_proxy import make_templates, labeled_samples
from layoutpilot.evaluate_reward_hacking_rigorous import (
    TrainConfig, fit_budget_proxy, _batch_gold, candidate_bank, score_kl,
    pearson_and_r2, select_proxy_only, primary_gap, parse_args, run,
)


def test_batch_gold_is_numerically_identical_to_fixed_scalar_oracle():
    t = make_templates(3, 40_000)
    banks = [candidate_bank(x.target, 37, np.random.default_rng(i)) for i,x in enumerate(t)]
    banks.append(np.array([[[.04,.04,.08,.08],[.04,.04,.86,.86],[.04,.04,.08,.08]]], np.float32))
    for bank in banks:
        expected = [gold_metrics(x)['gold_score'] for x in bank]
        assert _batch_gold(bank) == pytest.approx(expected, abs=1e-12)
    assert _batch_gold(np.zeros((0,3,4))).shape == (0,)


def test_analytic_kl_and_sqrt_x_axis():
    assert score_kl(1) == (0., 0.)
    for n in [2, 8, 8192]:
        f, rev = score_kl(n)
        assert f == pytest.approx(math.log(n)-(n-1)/n)
        assert rev == pytest.approx(n-1-math.log(n))
        assert rev >= f > 0
    with pytest.raises(ValueError):
        score_kl(0)


def test_proxy_training_spends_exact_label_budget():
    train = make_templates(4, 10_000)
    val = make_templates(3, 110_000)
    tx = labeled_samples(train, 60, 12)
    vx = labeled_samples(val, 20, 19)
    model, info = fit_budget_proxy(tx, vx, 70, seed=5,
        config=TrainConfig(width=8, epochs=4, batch_size=16, patience=2,
                           learning_rate=.001, label_noise=.05, device='cpu'))
    assert info['train_labels'] == 56
    assert info['val_labels'] == 14
    assert info['train_labels']+info['val_labels']==70
    assert 1 <= info['best_epoch'] <= info['epochs_run'] <= 4
    assert np.isfinite(info['validation_mse'])


def test_heldout_r2_can_be_negative_and_reports_nonconstant_r():
    a = pearson_and_r2(np.array([0., 1., 2.]), np.array([0., 2., 4.]))
    assert a['r2'] < 0 and a['pearson_r'] == pytest.approx(1.)
    assert pearson_and_r2(np.array([1., 1., 1.]), np.array([1., 1., 1.]))['r2'] is None


def test_proxy_ranking_never_needs_gold(monkeypatch):
    import layoutpilot.evaluate_reward_hacking_rigorous as m
    monkeypatch.setattr(m, '_batch_gold', lambda _: (_ for _ in ()).throw(AssertionError('gold leaked')))
    indices = select_proxy_only(np.array([.1,.5,.4,.3,.8]), (1,2,4,5))
    assert indices == {1:0,2:1,4:1,5:4}


def test_primary_metric_uses_discovery_to_choose_n_and_confirmation_to_measure():
    rows=[]
    for pid in (0,1):
        for subset in ('discovery','confirmation'):
            for tid in (10,11):
                g={1:.5,2:.9 if subset=='discovery' else .7,4:.6 if subset=='discovery' else .4}
                p={1:.1,2:.5,4:.95}
                for n in (1,2,4):
                    rows.append({'budget':256,'pair_id':pid,'subset':subset,
                                 'template_uid':tid,'best_of_n':n,
                                 'gold_score':g[n],'proxy_score':p[n]})
    result = primary_gap(rows,(1,2,4),draws=100,seed=0)[0]
    assert result['estimate'] == pytest.approx(.3)
    assert result['ci95_low'] == pytest.approx(.3)
    assert {r['gold_best_n_discovery'] for r in result['replicates']}=={2}
    assert {r['proxy_peak_n_discovery'] for r in result['replicates']}=={4}


def test_end_to_end_smoke_has_both_references_and_clean_provenance(tmp_path):
    cfg=parse_args(['--smoke','--out',str(tmp_path)])
    result=run(cfg)
    assert result['status']=='PIPELINE_SMOKE_TEST'
    assert len(result['proxy_quality'])==2
    assert len(result['primary_metric_results'])==2
    for q in result['proxy_quality']:
        assert q['train_labels']+q['val_labels']==q['budget']
        assert 'r2' in q and 'pearson_r' in q
    groups=[set(result[k]) for k in ('train_templates','validation_templates',
         'diagnostic_templates','discovery_templates','confirmation_templates')]
    assert all(not groups[i]&groups[j] for i in range(len(groups)) for j in range(i+1,len(groups)))
    assert (tmp_path/'proxy_gold_vs_sqrt_kl.png').stat().st_size>1000
    for name in ('proxy_quality.csv','curves.csv','per_template.csv','primary_gap.csv','summary_table.md','summary.json'):
        assert (tmp_path/name).exists()
    with (tmp_path/'per_template.csv').open() as f:
        data=list(csv.DictReader(f))
    assert len(data)==2*2*8*6  # budgets * subsets * templates * BoN
    for row in data:
        assert float(row['gold_score']) <= float(row['oracle_gold']) + 1e-12
        if int(row['best_of_n']) == 1:
            assert float(row['gold_score']) == pytest.approx(float(row['random_gold']))
            assert float(row['sqrt_kl_qn_q0']) == 0.
    json.loads((tmp_path/'summary.json').read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def test_full_protocol_requires_ten_paired_seeds_and_50plus_templates():
    cfg=parse_args(['--proxy-seeds','0','1','--search-seeds','10','11'])
    from layoutpilot.evaluate_reward_hacking_rigorous import _validate
    with pytest.raises(ValueError,match='>=10'):
        _validate(cfg)
