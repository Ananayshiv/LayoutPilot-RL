"""Data, model and training primitives for learned-proxy reward experiments.

Gold labels are *only* used for supervised proxy training/validation, and
post-selection evaluation. Search is implemented in a separate module.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch import nn
from layoutpilot.train import TEMPLATES, clamp_to_artboard
from layoutpilot.evaluate_reward_hacking import gold_metrics


@dataclass(frozen=True)
class Template:
    uid: int
    target: np.ndarray
    family: str


def make_templates(count: int, seed_start: int) -> list[Template]:
    """Procedurally distinct target geometries; UID ranges must not overlap."""
    if count < 1:
        raise ValueError('count must be positive')
    bases = tuple(TEMPLATES.items())
    items = []
    for uid in range(seed_start, seed_start + count):
        rng = np.random.default_rng(uid)
        family, base = bases[int(rng.integers(len(bases)))]
        target = np.asarray(base, dtype=np.float32).copy()
        target += rng.normal(0, [.12, .12, .11, .07], (3, 4)).astype(np.float32)
        target = clamp_to_artboard(target)
        items.append(Template(uid, target, family))
    return items


def proposal_samples(target: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample i.i.d. candidate layouts from a template-conditional proposal q0.

    Candidate rank is NOT influenced by gold scores. Sampling mixture includes
    out-of-distribution geometry to expose possible proxy extrapolation errors.
    """
    if n < 1:
        raise ValueError('n must be positive')
    target = np.asarray(target, dtype=np.float32).reshape(3, 4)
    noise = rng.normal(0, 1, (n, 3, 4)).astype(np.float32)
    scales = rng.choice(np.array([.35, 1.0, 2.1], dtype=np.float32), size=n,
                        p=[.3, .4, .3])
    sample = target[None, :, :] + noise * scales[:, None, None] * np.array(
        [.18, .18, .21, .12], dtype=np.float32)[None, None, :]
    # Twenty percent of the proposal is a broad independent mixture component.
    broad = rng.random(n) < .2
    if broad.any():
        k = int(broad.sum())
        sample[broad, :, :2] = rng.uniform(.02, .86, (k, 3, 2))
        sample[broad, :, 2:] = rng.uniform(.09, .7, (k, 3, 2))
    # clamp_to_artboard processes one layout, not a batch.
    return np.stack([clamp_to_artboard(r) for r in sample]).astype(np.float32)


def geometry_features(rects: np.ndarray) -> np.ndarray:
    """Target-blind 12-dimensional geometry features (no gold leakage)."""
    r = np.asarray(rects, dtype=np.float32)
    if r.shape[-2:] != (3, 4):
        raise ValueError('rects must end in (3,4)')
    if not np.all(np.isfinite(r)):
        raise ValueError('non-finite rectangle coordinates')
    return r.reshape(*r.shape[:-2], 12)


def labeled_samples(templates: list[Template], n: int, seed: int):
    """Gold evaluated for training LABELS, never during proxy optimization."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, len(templates), size=n)
    x = np.empty((n, 12), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    for i, j in enumerate(group):
        rect = proposal_samples(templates[int(j)].target, 1, rng)[0]
        x[i] = geometry_features(rect)
        y[i] = gold_metrics(rect)['gold_score']
    return x, y


class ProxyMLP(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def fit_proxy(train_x, train_y, val_x, val_y, *, seed: int,
              width: int = 64, updates: int = 250, batch_size: int = 128,
              lr: float = 0.001, device: str = 'cpu'):
    """Deterministic supervised training, validation-based checkpoint selection.

    Validation labels are used ONLY for selecting the checkpoint, not for search.
    """
    if updates < 1 or batch_size < 1 or width < 1:
        raise ValueError('updates, batch_size and width must be positive')
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    model = ProxyMLP(width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tx = torch.from_numpy(np.asarray(train_x, dtype=np.float32)).to(dev)
    ty = torch.from_numpy(np.asarray(train_y, dtype=np.float32)).to(dev)
    vx = torch.from_numpy(np.asarray(val_x, dtype=np.float32)).to(dev)
    vy = torch.from_numpy(np.asarray(val_y, dtype=np.float32)).to(dev)
    gen = torch.Generator(device='cpu').manual_seed(seed + 821)
    best = float('inf')
    state = None
    best_step = 0
    log = []
    for step in range(1, updates + 1):
        idx = torch.randint(len(tx), (min(batch_size, len(tx)),), generator=gen).to(dev)
        model.train()
        loss = (model(tx[idx]) - ty[idx]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 10 == 0 or step == updates:
            model.eval()
            with torch.no_grad():
                val_mse = float((model(vx) - vy).square().mean().item())
            log.append({'step': step, 'minibatch_mse': float(loss.item()), 'val_mse': val_mse})
            if val_mse < best:
                best = val_mse
                best_step = step
                state = copy.deepcopy(model.state_dict())
    assert state is not None
    model.load_state_dict(state)
    model.eval()
    return model, {'val_mse': best, 'selected_step': best_step, 'training_log': log}


def predict_proxy(model: ProxyMLP, features: np.ndarray, *, batch_size: int = 4096,
                  device: str = 'cpu') -> np.ndarray:
    """Batched forward pass: no calls to gold_metrics."""
    arr = np.asarray(features, dtype=np.float32).reshape(-1, 12)
    outputs = []
    dev = torch.device(device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arr), batch_size):
            x = torch.from_numpy(arr[start:start+batch_size]).to(dev)
            outputs.append(model(x).detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0,), dtype=np.float32)
