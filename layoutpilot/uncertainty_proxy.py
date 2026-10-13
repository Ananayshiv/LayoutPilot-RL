"""Bootstrap deep ensemble with budget-preserving 80/20 labels.

80% train; within the 20%, half is for checkpoint early stopping and half is
for calibration. No extra labels are purchased. Conformal standardized-error
radius is estimated ONLY on calibration records, never on test candidates.
Ensemble spread is epistemic *heuristic*, not a posterior variance guarantee.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from layoutpilot.learned_proxy import ProxyMLP, predict_proxy


@dataclass(frozen=True)
class EnsembleConfig:
    members: int = 5
    width: int = 32
    epochs: int = 80
    patience: int = 10
    batch_size: int = 128
    lr: float = .001
    sigma_floor: float = .01
    nominal_coverage: float = .90
    label_noise: float = .10
    device: str = 'cpu'


def _quantile_higher(x: np.ndarray, p: float):
    # Split-conformal finite-sample order statistic (clip if target > sample).
    a = np.sort(np.asarray(x, float).reshape(-1))
    if len(a)==0:
        raise ValueError('no calibration samples')
    index = min(len(a)-1, max(0, math.ceil((len(a)+1)*p)-1))
    return float(a[index])


class DeepEnsemble:
    def __init__(self, models: list[ProxyMLP], *, floor: float, radius90: float, device: str):
        if len(models)<2 or floor<=0 or radius90<0:
            raise ValueError('need >=2 models, positive sigma floor and valid radius')
        self.models=models
        self.floor=float(floor)
        self.radius90=float(radius90)
        self.device=device

    def predict(self, features: np.ndarray, *, batch_size=4096):
        member_predictions = np.stack([predict_proxy(m, features,batch_size=batch_size,
                                                      device=self.device)
                                      for m in self.models], axis=0)
        mean = member_predictions.mean(0).astype(np.float64)
        # ddof=1 is sample SD over ensemble means, not predictive aleatoric noise.
        sd = member_predictions.std(0,ddof=1).astype(np.float64)
        sigma = np.maximum(sd,self.floor)
        return mean, sigma

    def acquisition(self, features: np.ndarray, *, lam: float, batch_size=4096):
        if lam < 0:
            raise ValueError('lambda must be >=0')
        mu,sigma = self.predict(features,batch_size=batch_size)
        # Calibrated to a held-out 90%-coverage *radius*; is not a Gaussian SD.
        return mu - lam*self.radius90*sigma,mu,sigma


def train_ensemble(train_pool, val_pool, budget: int, config: EnsembleConfig,
                   *, seed: int) -> tuple[DeepEnsemble, dict[str, Any]]:
    if budget<30 or config.members<2 or config.epochs<1 or config.patience<1 or config.batch_size<1:
        raise ValueError('invalid ensemble hyperparameters')
    if config.lr<=0 or config.sigma_floor<=0 or not (.5<config.nominal_coverage<1):
        raise ValueError('invalid calibration settings')
    n_train=int(.8*budget); n_val=budget-n_train
    n_early=(n_val+1)//2; n_cal=n_val-n_early
    if n_cal<3 or len(train_pool[0])<n_train or len(val_pool[0])<n_val:
        raise ValueError('not enough labeled samples')
    tx = np.asarray(train_pool[0][:n_train],np.float32)
    ty = np.asarray(train_pool[1][:n_train],np.float32)
    vx = np.asarray(val_pool[0][:n_early],np.float32)
    vy = np.asarray(val_pool[1][:n_early],np.float32)
    cx = np.asarray(val_pool[0][n_early:n_val],np.float32)
    cy = np.asarray(val_pool[1][n_early:n_val],np.float32)
    members=[]; diagnostics=[]
    for member in range(config.members):
        rng = np.random.default_rng(seed*9973+member*617+43)
        bootstrap_idx = rng.integers(0,n_train,n_train)
        xb=torch.tensor(tx[bootstrap_idx],device=config.device)
        noisy=np.clip(ty[bootstrap_idx]+rng.normal(0,config.label_noise,n_train),0,1)
        yb=torch.tensor(noisy,dtype=torch.float32,device=config.device)
        xv=torch.tensor(vx,device=config.device)
        yv=torch.tensor(vy,device=config.device)
        torch.manual_seed(seed*1009+member*43)
        model=ProxyMLP(config.width).to(config.device)
        opt=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=1e-4)
        gen=torch.Generator().manual_seed(seed*371+member+17)
        best=float('inf'); best_state=None; stale=0; selected_epoch=0
        for epoch in range(1,config.epochs+1):
            model.train()
            order=torch.randperm(n_train,generator=gen).tolist()
            for start in range(0,n_train,config.batch_size):
                idx=order[start:start+config.batch_size]
                loss=(model(xb[idx])-yb[idx]).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step()
            model.eval()
            with torch.inference_mode():
                val_loss=float((model(xv)-yv).square().mean())
            if val_loss < best-1e-7:
                best=val_loss;best_state=copy.deepcopy(model.state_dict());selected_epoch=epoch;stale=0
            else:
                stale+=1
            if stale>=config.patience:
                break
        if best_state is None:
            raise RuntimeError('no model checkpoint selected')
        model.load_state_dict(best_state);model.eval()
        members.append(model)
        diagnostics.append({'member':member,'best_epoch':selected_epoch,'early_stop_mse':best})
    ensemble=DeepEnsemble(members,floor=config.sigma_floor,radius90=1,device=config.device)
    mu,sigma=ensemble.predict(cx)
    residual = np.abs(mu-cy)/sigma
    radius=_quantile_higher(residual,config.nominal_coverage)
    ensemble.radius90=radius
    quality={'label_budget':budget,'train_labels':n_train,'val_labels':n_val,
             'early_stop_labels':n_early,'calibration_labels':n_cal,
             'member_diagnostics':diagnostics,'calibration_radius':radius,
             'calibration_coverage':float(np.mean(residual<=radius)),
             'calibration_mae':float(np.mean(np.abs(mu-cy))),
             'uncertainty_definition':'max(ensemble member sample SD, sigma_floor); confidence radius from split calibration',
             'nominal_coverage':config.nominal_coverage}
    return ensemble,quality
