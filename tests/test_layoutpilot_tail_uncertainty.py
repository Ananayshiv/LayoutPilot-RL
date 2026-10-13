"""Invariants: selection isolation, budget accounting, shift, tie-safe tail, null findings."""
import json
import numpy as np
import pytest
from layoutpilot.tail_diagnostics import rank_correlations,top_tail_metrics,turnover_diagnostic
from layoutpilot.shift_protocol import proposal_environment
from layoutpilot.uncertainty_proxy import EnsembleConfig, DeepEnsemble, train_ensemble
from layoutpilot.learned_proxy import make_templates,labeled_samples, ProxyMLP
from layoutpilot.evaluate_tail_uncertainty import (
    select_acquisition_only,hierarchical_ci,paired_mitigation,parse_args,run,
)


def test_tied_rank_correlations_and_undefined():
    p=np.array([1.,2.,3.,4.]);g=np.array([4.,3.,2.,1.])
    c=rank_correlations(p,g)
    assert c['spearman']==pytest.approx(-1)
    assert c['kendall_tau_b']==pytest.approx(-1)
    d=rank_correlations([1,1,1],[1,2,3])
    assert d['spearman'] is None and d['kendall_tau_b'] is None
    t=rank_correlations([1,1,2],[0,0,5])
    assert t['spearman']==pytest.approx(1)
    assert t['kendall_tau_b']==pytest.approx(1)


def test_high_proxy_tail_can_have_high_error_and_inverted_ranking():
    p=np.array([.1,.2,.6,.7,.8,.9])
    gold=np.array([.1,.2,.9,.7,.4,.1])
    sd=np.full(6,.05)
    t=top_tail_metrics(p,gold,sd,top_k=3,tail_fraction=1/3,interval_radius=1.)
    assert t['top_k_effective']==3
    assert t['tail_count']==2
    assert t['tail_spearman']==pytest.approx(-1)
    assert t['tail_kendall_tau_b']==pytest.approx(-1)
    assert t['tail_bias_proxy_minus_gold'] >.5
    assert t['tail_interval_coverage']==0
    assert t['overall_mae'] < t['tail_mae']


def test_exploratory_turnover_does_not_force_positive_result():
    no=turnover_diagnostic([1,2,4,8],[.2,.3,.4,.5],epsilon=.01)
    assert no['observed_turnover'] is False
    yes=turnover_diagnostic([1,2,4,8],[.2,.7,.6,.4],epsilon=.01)
    assert yes['peak_n']==2 and yes['observed_turnover'] is True


def test_environment_shift_is_deterministic_and_constrained():
    template=make_templates(1,1234)[0]
    a=proposal_environment(template.target,20,np.random.default_rng(42),'base')
    b=proposal_environment(template.target,20,np.random.default_rng(42),'spatial_jitter')
    assert not np.array_equal(a,b)
    assert np.all(b[:,:,0]>=.04-1e-6) and np.all(b[:,:,1]>=.04-1e-6)
    assert np.all(b[:,:,0]+b[:,:,2]<=.960001)
    assert np.all(b[:,:,1]+b[:,:,3]<=.960001)
    assert np.array_equal(b,proposal_environment(template.target,20,np.random.default_rng(42),'spatial_jitter'))
    with pytest.raises(ValueError):
        proposal_environment(template.target,4,np.random.default_rng(1),'invalid')


def test_ensemble_train_spends_only_budget_with_independent_calibration():
    tr=labeled_samples(make_templates(3,10000),80,seed=10)
    va=labeled_samples(make_templates(3,110000),20,seed=20)
    config=EnsembleConfig(members=2,width=8,epochs=3,patience=2,batch_size=32,
                          lr=.001,sigma_floor=.01,nominal_coverage=.90,
                          label_noise=0.,device='cpu')
    model,log=train_ensemble(tr,va,100,config,seed=2)
    assert log['train_labels']==80
    assert log['val_labels']==20
    assert log['early_stop_labels']==10
    assert log['calibration_labels']==10
    assert log['train_labels']+log['early_stop_labels']+log['calibration_labels']==100
    pred,sd=model.predict(tr[0][:4])
    assert pred.shape==sd.shape==(4,) and (sd>=.01).all()
    assert np.isfinite(model.radius90) and model.radius90>=0
    score,mean,unc=model.acquisition(tr[0][:4],lam=1.)
    assert np.all(score<=mean)
    assert np.allclose(unc,sd)
    with pytest.raises(ValueError):
        model.acquisition(tr[0][:4],lam=-1.)


def test_proxy_selection_requires_no_gold_argument(monkeypatch):
    import layoutpilot.evaluate_tail_uncertainty as study
    monkeypatch.setattr(study,'_batch_gold',lambda bank:(_ for _ in ()).throw(AssertionError('gold leaked')))
    ids=study.select_acquisition_only(np.array([.1,.5,.4,.6]),[1,2,4])
    assert ids=={1:0,2:1,4:3}


def test_hierarchical_bootstrap_and_ineligible_smoke_claims():
    rng=np.random.default_rng(7)
    assert hierarchical_ci(np.full((2,4),.3),rng=rng,draws=100)==pytest.approx((.3,.3,.3))
    rows=[]
    for subset in ('discovery','confirmation'):
        for tid in range(4):
            for n in (1,2,4):
                for sel in ('standard','conservative'):
                    gold=(.3 if n==1 else .7 if n==2 else .5)+(0.02 if sel=='conservative' else 0)
                    rows.append({'subset':subset,'pair_id':0,'environment':'base',
                        'budget':100,'template_uid':tid,'selector':sel,'best_of_n':n,'gold_score':gold})
    result=paired_mitigation(rows,budgets=[100],search_budgets=[1,2,4],
            draws=100,seed=5,epsilon=.01)[0]
    assert result['conservative_minus_standard_gold_at_max_n']==pytest.approx(.02)
    assert result['confirmed_turnover_drop_standard']==pytest.approx(.2)
    assert result['mitigation_supported'] is False
    assert result['turnover_supported'] is False


def test_end_to_end_smoke_files_and_no_gold_selection_leak(tmp_path):
    out=tmp_path/'probe'
    report=run(parse_args(['--smoke','--out',str(out)]))
    assert report['status']=='PIPELINE_SMOKE_TEST'
    assert len(report['proxy_quality'])==1
    assert report['proxy_quality'][0]['train_labels']==80
    assert report['proxy_quality'][0]['early_stop_labels']+report['proxy_quality'][0]['calibration_labels']==20
    for k in ('proxy_quality.csv','per_template.csv','tail_per_template.csv','curves.csv',
              'tail_summary.csv','mitigation_turnover.csv','gold_proxy_regret_vs_kl.png',
              'tail_reliability_by_budget.png','summary.json'):
        assert (out/k).stat().st_size>30
    assert len(report['mitigation_and_turnover'])==2
    assert all(not m['mitigation_supported'] for m in report['mitigation_and_turnover'])
    x=json.loads((out/'summary.json').read_text())
    groups=[set(v) for v in x['split_template_uids'].values()]
    assert all(not groups[i]&groups[j] for i in range(len(groups)) for j in range(i+1,len(groups)))
