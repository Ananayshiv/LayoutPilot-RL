"""Extremal Goodhart diagnostics and calibrated-uncertainty mitigation study.

Run from repository root: python -m layoutpilot.evaluate_tail_uncertainty --smoke
The independently engineered 'gold' is not human ratings. Experiments may
show monotonic gold improvement, zero mitigation or failed calibration.

No gold scores are accessed until *all* candidate selection indices for a
proposal bank have been finalized using learned proxy predictions alone.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from layoutpilot.learned_proxy import make_templates, labeled_samples, geometry_features
from layoutpilot.evaluate_reward_hacking_rigorous import (
    _batch_gold, score_kl, pearson_and_r2, _no_nan, select_proxy_only,
)
from layoutpilot.uncertainty_proxy import EnsembleConfig, train_ensemble
from layoutpilot.tail_diagnostics import top_tail_metrics, turnover_diagnostic
from layoutpilot.shift_protocol import ENVIRONMENTS, proposal_environment


def write_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError('no rows')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=records[0].keys(),extrasaction='ignore')
        writer.writeheader();writer.writerows(records)


def select_acquisition_only(acquisition, search_budgets):
    """Pure proxy/uncertainty selection; intentionally no gold parameter."""
    return select_proxy_only(np.asarray(acquisition),tuple(search_budgets))


def hierarchical_ci(data: np.ndarray, *, rng, draws: int):
    """Hierarchical paired bootstrap of (independent replicate, template)."""
    a=np.asarray(data,dtype=float)
    if a.ndim!=2 or a.size==0 or np.any(~np.isfinite(a)):
        raise ValueError('bootstrap needs finite [replicates, templates] matrix')
    ns,nt=a.shape
    si=rng.integers(0,ns,size=(draws,ns))
    ti=rng.integers(0,nt,size=(draws,nt))
    means=a[si[:,:,None],ti[:,None,:]].mean(axis=(1,2))
    return float(a.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def aggregate_curves(rows, draws, seed):
    rng=np.random.default_rng(seed)
    groups=defaultdict(list)
    for row in rows:
        groups[row['environment'],row['budget'],row['selector'],row['best_of_n']].append(row)
    output=[]
    for (env,b,sel,n),group in sorted(groups.items()):
        pids=sorted({r['pair_id'] for r in group})
        tids=sorted({r['template_uid'] for r in group})
        if len(group)!=len(pids)*len(tids):
            raise ValueError('missing or duplicate held-out template rows')
        mapping={(r['pair_id'],r['template_uid']):r for r in group}
        row={'environment':env,'budget':b,'selector':sel,'best_of_n':n,
             'sqrt_kl':math.sqrt(score_kl(n)[0]),
             'kl_selected_to_proposal':score_kl(n)[0],
             'kl_proposal_to_selected':score_kl(n)[1],
             'replicates':len(pids),'templates':len(tids)}
        for key in ('proxy_score','gold_score','selection_regret','oracle_gold','random_gold',
                    'predicted_uncertainty','selected_interval_coverage'):
            arr=np.asarray([[mapping[i,t][key] for t in tids] for i in pids],dtype=float)
            est,lo,hi=hierarchical_ci(arr,rng=rng,draws=draws)
            row[key]=est;row[key+'_ci_low']=lo;row[key+'_ci_high']=hi
        output.append(row)
    return output


def aggregate_tail(rows, draws, seed):
    rng=np.random.default_rng(seed)
    output=[]; groups=defaultdict(list)
    for row in rows:
        groups[row['environment'],row['budget']].append(row)
    for (env,b),group in sorted(groups.items()):
        pids=sorted({r['pair_id'] for r in group}); tids=sorted({r['template_uid'] for r in group})
        mapping={(r['pair_id'],r['template_uid']):r for r in group}
        if len(mapping)!=len(group) or len(mapping)!=len(pids)*len(tids):
            raise ValueError('missing/duplicate tail observations')
        rec={'environment':env,'budget':b,'replicates':len(pids),'templates':len(tids)}
        for key in ('tail_spearman','tail_kendall_tau_b','tail_bias_proxy_minus_gold',
                    'tail_mae','overall_mae','tail_interval_coverage','overall_interval_coverage',
                    'tail_mean_sigma'):
            a=np.asarray([[mapping[i,t][key] if mapping[i,t][key] is not None else np.nan
                           for t in tids] for i in pids],float)
            valid=a[np.isfinite(a)]
            rec[key+'_valid_units']=int(valid.size)
            if valid.size==0:
                rec[key]=rec[key+'_ci_low']=rec[key+'_ci_high']=None
            elif np.isfinite(a).all():
                rec[key],rec[key+'_ci_low'],rec[key+'_ci_high']=hierarchical_ci(a,rng=rng,draws=draws)
            else:
                # Some rank correlations are undefined on tied/constant tails;
                # do not silently assign 0. Report descriptive valid-unit mean.
                rec[key]=float(valid.mean());rec[key+'_ci_low']=rec[key+'_ci_high']=None
        output.append(rec)
    return output


def paired_mitigation(rows, *, budgets, search_budgets, draws, seed, epsilon):
    """Predeclared largest-N treatment effect and frozen discovery turnover.

    The treatment compares standard vs conservative on IDENTICAL candidates.
    Returns effect CI and turnover separately, without asserting success.
    """
    rng=np.random.default_rng(seed)
    output=[]
    for env in sorted({r['environment'] for r in rows}):
        for budget in budgets:
            for pid in sorted({r['pair_id'] for r in rows}):
                dev=[r for r in rows if r['environment']==env and r['budget']==budget and
                     r['pair_id']==pid and r['subset']=='discovery']
                conf=[r for r in rows if r['environment']==env and r['budget']==budget and
                      r['pair_id']==pid and r['subset']=='confirmation']
                if not conf:
                    raise ValueError('missing independent confirmation split')
                # Predeclare Nmax and compare paired final layouts for treatment.
                max_n=max(search_budgets)
                result={}
                for sel in ('standard','conservative'):
                    d={n:float(np.mean([r['gold_score'] for r in dev if
                          r['selector']==sel and r['best_of_n']==n])) for n in search_budgets}
                    peak=turnover_diagnostic(np.array(search_budgets),
                                             np.array([d[n] for n in search_budgets]),epsilon=epsilon)
                    result[sel]=(d,peak)
                if pid==min({r['pair_id'] for r in rows}):
                    pass
                tids=sorted({r['template_uid'] for r in conf})
                def values(sel,n,key='gold_score'):
                    mapping={r['template_uid']:r[key] for r in conf if
                             r['selector']==sel and r['best_of_n']==n}
                    if set(mapping)!=set(tids):
                        raise ValueError('missing confirmation paired observation')
                    return np.array([mapping[t] for t in tids],float)
                gain=values('conservative',max_n)-values('standard',max_n)
                # Turnover peak selected ONLY on discovery, evaluated on conf.
                peak_n=result['standard'][1]['peak_n']
                confirmed_drop=values('standard',peak_n)-values('standard',max_n)
                base={'environment':env,'budget':budget,'pair_id':pid,
                      'peak_n_on_discovery':peak_n,'max_n':max_n,
                      'turnover_observed_on_discovery':result['standard'][1]['observed_turnover'],
                      'discovery_drop':result['standard'][1]['max_gold_minus_final_gold'],
                      'mitigation_by_template':gain,'drop_by_template':confirmed_drop}
                output.append(base)
    summary=[]
    for env in sorted({x['environment'] for x in output}):
        for budget in budgets:
            group=sorted((x for x in output if x['environment']==env and x['budget']==budget),
                         key=lambda x:x['pair_id'])
            effects=np.stack([x['mitigation_by_template'] for x in group])
            drops=np.stack([x['drop_by_template'] for x in group])
            effect=hierarchical_ci(effects,rng=rng,draws=draws)
            drop=hierarchical_ci(drops,rng=rng,draws=draws)
            summary.append({'environment':env,'budget':budget,'max_n':max(search_budgets),
                'conservative_minus_standard_gold_at_max_n':effect[0],
                'effect_ci_low':effect[1],'effect_ci_high':effect[2],
                'mitigation_supported':bool(effect[1]>0 and len(group)>=10 and effects.shape[1]>=50),
                'confirmed_turnover_drop_standard':drop[0],
                'turnover_ci_low':drop[1],'turnover_ci_high':drop[2],
                'turnover_supported':bool(drop[1]>epsilon and len(group)>=10 and effects.shape[1]>=50),
                'discovery_turnover_seeds':int(sum(x['turnover_observed_on_discovery'] for x in group)),
                'replicates':len(group),'confirmation_templates':effects.shape[1]})
    return summary


def plot_study(curves, tails, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    envs=sorted({r['environment'] for r in curves})
    budgets=sorted({r['budget'] for r in curves})
    fig,ax=plt.subplots(len(envs),3,figsize=(14,4*len(envs)),squeeze=False)
    for ei,env in enumerate(envs):
        for b in (budgets[0],budgets[-1]):
            for sel,ls in [('standard','-'),('conservative','--')]:
                subset=sorted((r for r in curves if r['environment']==env and r['budget']==b and
                                r['selector']==sel),key=lambda r:r['best_of_n'])
                x=[r['sqrt_kl'] for r in subset]
                name=f'{b} labels / {sel}'
                for axis,key in [(ax[ei,0],'gold_score'),(ax[ei,1],'proxy_score'),
                                 (ax[ei,2],'selection_regret')]:
                    line,=axis.plot(x,[r[key] for r in subset],linestyle=ls,marker='.',label=name)
                    axis.fill_between(x,[r[key+'_ci_low'] for r in subset],
                                      [r[key+'_ci_high'] for r in subset],alpha=.10,color=line.get_color())
        baseline=sorted((r for r in curves if r['environment']==env and
                         r['budget']==budgets[-1] and r['selector']=='standard'),
                         key=lambda r:r['best_of_n'])
        ax[ei,0].plot([r['sqrt_kl'] for r in baseline],
                      [r['oracle_gold'] for r in baseline],color='black',ls=':',label='Gold oracle')
        ax[ei,0].axhline(baseline[0]['random_gold'],color='gray',ls='-.',label='Random N=1')
        for j,title in enumerate(('True gold', 'Predicted proxy mean', 'Gold-oracle regret')):
            ax[ei,j].set_title(f'{env}: {title}')
            ax[ei,j].set_xlabel(r'$\sqrt{D_{KL}(q_N\Vert q_0)}$ (idealized)')
            ax[ei,j].grid(alpha=.22)
            ax[ei,j].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)


def plot_tail_reliability(tail_summary, out):
    """Visualize extreme-tail error and ranking quality over label budgets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    envs=sorted({r['environment'] for r in tail_summary})
    fig,axes=plt.subplots(len(envs),3,figsize=(12,3.4*len(envs)),squeeze=False)
    specs=(('tail_mae','Top-proxy-tail MAE'),
           ('tail_spearman','Top-K Spearman rho'),
           ('tail_interval_coverage','Selected-tail interval coverage'))
    for ei,env in enumerate(envs):
        subset=sorted((r for r in tail_summary if r['environment']==env),
                      key=lambda r:r['budget'])
        for j,(key,title) in enumerate(specs):
            x=np.asarray([r['budget'] for r in subset],dtype=int)
            vals=np.asarray([r[key] if r[key] is not None else np.nan for r in subset],float)
            ax=axes[ei,j]
            ax.plot(x,vals,marker='o')
            lows=np.asarray([r[key+'_ci_low'] if r[key+'_ci_low'] is not None else np.nan for r in subset],float)
            highs=np.asarray([r[key+'_ci_high'] if r[key+'_ci_high'] is not None else np.nan for r in subset],float)
            ax.fill_between(x,lows,highs,alpha=.17)
            ax.set_xscale('log')
            ax.set_title(f'{env}: {title}')
            ax.set_xlabel('Gold-label budget (train + validation)')
            ax.grid(alpha=.2)
            if key=='tail_interval_coverage':
                ax.axhline(.90,color='black',ls=':',lw=1,label='Nominal 90%')
                ax.legend(fontsize=7)
            if key=='tail_spearman':
                ax.set_ylim(-1.05,1.05)
    fig.tight_layout()
    fig.savefig(out,dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--label-budgets',nargs='+',type=int,default=[256,1024,4096])
    p.add_argument('--best-of-n',nargs='+',type=int,default=[1,4,16,64,256,1024,8192])
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--environments',nargs='+',choices=ENVIRONMENTS,default=['base','wide_proposals'])
    p.add_argument('--train-templates',type=int,default=32)
    p.add_argument('--val-templates',type=int,default=16)
    p.add_argument('--diagnostic-templates',type=int,default=16)
    p.add_argument('--diagnostic-labels',type=int,default=512)
    p.add_argument('--discovery-templates',type=int,default=50)
    p.add_argument('--confirmation-templates',type=int,default=50)
    p.add_argument('--ensemble-members',type=int,default=5)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--label-noise',type=float,default=.10)
    p.add_argument('--sigma-floor',type=float,default=.01)
    p.add_argument('--nominal-coverage',type=float,default=.90)
    p.add_argument('--conservative-lambda',type=float,default=1.)
    p.add_argument('--top-k',type=int,default=32)
    p.add_argument('--tail-fraction',type=float,default=.05)
    p.add_argument('--turnover-epsilon',type=float,default=.01)
    p.add_argument('--proposal-scale',type=float,default=1.)
    p.add_argument('--broad-prob',type=float,default=.20)
    p.add_argument('--jitter',type=float,default=.045)
    p.add_argument('--bootstrap-draws',type=int,default=1000)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--out',type=str,default='experiment_outputs/tail_uncertainty')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    if args.smoke:
        args.label_budgets=[100];args.best_of_n=[1,4,16,32];args.seeds=[0]
        args.discovery_templates=4;args.confirmation_templates=4
        args.diagnostic_labels=32;args.ensemble_members=2
        args.epochs=5;args.patience=2;args.bootstrap_draws=100
    return args


def run(args):
    sizes=sorted(set(args.label_budgets)); search=sorted(set(args.best_of_n))
    if not sizes or min(sizes)<30 or not search or search[0]!=1 or args.top_k<2:
        raise ValueError('invalid budgets')
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds:
        raise ValueError('independent seeds must be unique')
    if not args.smoke and (len(args.seeds)<3 or min(args.discovery_templates,args.confirmation_templates)<50):
        raise ValueError('study requires at least 3 independent seeds and 50 templates per evaluation split')
    if args.conservative_lambda<0 or args.bootstrap_draws<50 or not (0<args.tail_fraction<=1):
        raise ValueError('invalid statistical/selection hyperparameters')
    if args.ensemble_members < 2 or args.label_noise < 0 or args.sigma_floor <= 0:
        raise ValueError('invalid ensemble settings')
    if not args.smoke and args.bootstrap_draws < 500:
        raise ValueError('full experiment requires at least 500 bootstrap draws')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    torch.set_num_threads(min(torch.get_num_threads(),2))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    all_groups={
        'train':make_templates(args.train_templates,10_000),
        'validation':make_templates(args.val_templates,110_000),
        'diagnostic':make_templates(args.diagnostic_templates,210_000),
        'discovery':make_templates(args.discovery_templates,310_000),
        'confirmation':make_templates(args.confirmation_templates,410_000),
    }
    uids=[t.uid for group in all_groups.values() for t in group]
    if len(uids)!=len(set(uids)):
        raise ValueError('template IDs overlap across splits')
    config=EnsembleConfig(args.ensemble_members,args.width,args.epochs,args.patience,
                          args.batch_size,args.learning_rate,args.sigma_floor,
                          args.nominal_coverage,args.label_noise,args.device)
    trained={};fit_records=[]
    # Phase 1: all training, in-budget early stopping/calibration, and held-out
    # diagnostics BEFORE search. This diagnostic split never selects hyperparams.
    for pair,seed in enumerate(args.seeds):
        tr_max=int(.8*max(sizes)); val_max=max(sizes)-tr_max
        train_pool=labeled_samples(all_groups['train'],tr_max,seed=seed*91+112)
        val_pool=labeled_samples(all_groups['validation'],val_max,seed=seed*91+113)
        diag_x,diag_gold=labeled_samples(all_groups['diagnostic'],args.diagnostic_labels,
                                          seed=seed*91+114)
        for b in sizes:
            ensemble,detail=train_ensemble(train_pool,val_pool,b,config,seed=seed*100_003+b)
            mu,sd=ensemble.predict(diag_x)
            q=pearson_and_r2(diag_gold,mu)
            coverage=float(np.mean(np.abs(mu-diag_gold)<=ensemble.radius90*sd))
            fit_records.append({'pair_id':pair,'seed':seed,'budget':b,
                'r2_diagnostic':q['r2'],'pearson_diagnostic':q['pearson_r'],
                'mae_diagnostic':q['mae'],
                'calibration_coverage_on_diagnostic':coverage,
                'calibration_coverage_target':args.nominal_coverage,
                'calibration_radius':ensemble.radius90,
                'train_labels':detail['train_labels'],
                'early_stop_labels':detail['early_stop_labels'],
                'calibration_labels':detail['calibration_labels'],
                'validation_labels':detail['val_labels'],
                'member_count':args.ensemble_members})
            trained[pair,b]=ensemble
            print(f'PRE-SEARCH pair={pair} B={b} diagnostic_R2={q["r2"]} '
                  f'coverage={coverage:.3f}',flush=True)
    write_csv(out/'proxy_quality.csv',fit_records)

    # Phase 2: candidate selection using proxy statistics ONLY, for each
    # independent environment. Gold is called AFTER every selector is frozen.
    per_template=[];tails=[]
    for pair,seed in enumerate(args.seeds):
        for subset in ('discovery','confirmation'):
            for template in all_groups[subset]:
                for env in args.environments:
                    env_idx=ENVIRONMENTS.index(env)
                    # Distinct environment-specific RNG streams and same bank
                    # for all label-budget and conservative-standard comparisons.
                    rng=np.random.default_rng(seed*1_000_003+template.uid*37+env_idx*713)
                    bank=proposal_environment(template.target,max(search),rng,env,
                               scale=args.proposal_scale,broad_prob=args.broad_prob,
                               jitter=args.jitter)
                    features=geometry_features(bank)
                    frozen={}
                    for b in sizes:
                        ensemble=trained[pair,b]
                        mu,sigma=ensemble.predict(features)
                        selections={}
                        selections['standard']=select_acquisition_only(mu,search)
                        acquisitions=mu-args.conservative_lambda*ensemble.radius90*sigma
                        selections['conservative']=select_acquisition_only(acquisitions,search)
                        frozen[b]=(mu,sigma,ensemble.radius90,selections)
                    # GOLD BARRIER: every selector for every budget and N has
                    # already been frozen above, before accessing gold.
                    gold=_batch_gold(bank)
                    oracle=np.maximum.accumulate(gold)
                    for b in sizes:
                        mu,sigma,radius,selections=frozen[b]
                        tail=top_tail_metrics(mu,gold,sigma,top_k=args.top_k,
                                    tail_fraction=args.tail_fraction,interval_radius=radius)
                        tails.append({'pair_id':pair,'seed':seed,'budget':b,
                             'subset':subset,'environment':env,'template_uid':template.uid,**tail})
                        for selector,indices in selections.items():
                            for n in search:
                                idx=indices[n]
                                per_template.append({'pair_id':pair,'seed':seed,
                                   'subset':subset,'environment':env,'budget':b,
                                   'template_uid':template.uid,'selector':selector,
                                   'best_of_n':n,'selected_index':idx,
                                   'proxy_score':float(mu[idx]),'gold_score':float(gold[idx]),
                                   'predicted_uncertainty':float(radius*sigma[idx]),
                                   'selected_interval_coverage':float(abs(mu[idx]-gold[idx])<=radius*sigma[idx]),
                                   'oracle_gold':float(oracle[n-1]),
                                   'selection_regret':float(oracle[n-1]-gold[idx]),
                                   'random_gold':float(gold[0])})
        print(f'Completed search replicate {pair+1}/{len(args.seeds)}',flush=True)
    write_csv(out/'per_template.csv',per_template)
    write_csv(out/'tail_per_template.csv',tails)
    confirm=[r for r in per_template if r['subset']=='confirmation']
    confirm_tail=[r for r in tails if r['subset']=='confirmation']
    curves=aggregate_curves(confirm,args.bootstrap_draws,104729)
    tail_summary=aggregate_tail(confirm_tail,args.bootstrap_draws,104759)
    mitigation=paired_mitigation(per_template,budgets=sizes,search_budgets=search,
              draws=args.bootstrap_draws,seed=104761,epsilon=args.turnover_epsilon)
    write_csv(out/'curves.csv',curves)
    write_csv(out/'tail_summary.csv',tail_summary)
    write_csv(out/'mitigation_turnover.csv',mitigation)
    plot_study(curves,tail_summary,out/'gold_proxy_regret_vs_kl.png')
    plot_tail_reliability(tail_summary,out/'tail_reliability_by_budget.png')
    status='PIPELINE_SMOKE_TEST' if args.smoke else 'EXPLORATORY_EMPIRICAL_STUDY_NOT_PREREGISTERED'
    proxy_fit_health={b:{'mean_diagnostic_r2':(float(np.mean([x['r2_diagnostic'] for x in fit_records if x['budget']==b and x['r2_diagnostic'] is not None])) if any(x['budget']==b and x['r2_diagnostic'] is not None for x in fit_records) else None), 'nonnegative_r2_seeds':sum(x['budget']==b and x['r2_diagnostic'] is not None and x['r2_diagnostic']>=0 for x in fit_records)} for b in sizes}
    summary={'status':status, 'proxy_fit_health':proxy_fit_health,
       'confirmation_evidence_eligibility':'requires >=10 independent seed pairs, >=50 confirmation templates and independently verified proxy convergence; smoke or exploratory runs never qualify as established findings',
       'gold':'fixed, hand-engineered layout geometric metric; not human feedback',
       'primary_intervention':'gold(conservative)-gold(standard) on identical candidate bank at predeclared maximum N',
       'conservative_acquisition':'ensemble mean - lambda * split-calibrated 90% interval radius',
       'uncertainty_limitation':'ensemble disagreement is not a guaranteed uncertainty bound; shift may break calibration',
       'tail_measurement':'top K by predicted mean from entire candidate bank; selection gold queried only post-selection',
       'turnover':'discovery peak budget frozen before independent confirmation measurement',
       'kl':'analytic ideal i.i.d continuous proxy-rank selection KL; conservative rank distribution has same ideal KL only under continuous scores',
       'ci':'paired hierarchical resampling over independent training/search seeds and templates',
       'findings_warning':'non-turnover and unsuccessful mitigation are valid outcomes',
       'hyperparameters':vars(args),
       'split_template_uids':{k:[t.uid for t in v] for k,v in all_groups.items()},
       'proxy_quality':fit_records,'mitigation_and_turnover':mitigation,
       'outputs':['proxy_quality.csv','per_template.csv','tail_per_template.csv','curves.csv',
                  'tail_summary.csv','mitigation_turnover.csv','gold_proxy_regret_vs_kl.png',
                  'tail_reliability_by_budget.png','summary.json']}
    (out/'summary.json').write_text(json.dumps(_no_nan(summary),indent=2,allow_nan=False)+'\n')
    print('\n| Environment | Labels | ΔGold conservative-standard @maxN | 95% CI | Supported? | Confirmed turnover? |')
    print('|---|---:|---:|---|---|---|')
    for x in mitigation:
        print(f'| {x["environment"]} | {x["budget"]} | '
              f'{x["conservative_minus_standard_gold_at_max_n"]:+.4f} | '
              f'[{x["effect_ci_low"]:+.4f},{x["effect_ci_high"]:+.4f}] | '
              f'{x["mitigation_supported"]} | {x["turnover_supported"]} |')
    return summary

if __name__=='__main__':
    run(parse_args())
