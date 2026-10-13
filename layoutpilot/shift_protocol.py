"""Explicit test-only proposal shifts; train/validation gold stays unchanged."""
from __future__ import annotations
import numpy as np
from layoutpilot.evaluate_reward_hacking_rigorous import candidate_bank

ENVIRONMENTS = ('base', 'wide_proposals', 'spatial_jitter')


def proposal_environment(target, n, rng: np.random.Generator, env: str,
                         *, scale=1., broad_prob=.2, jitter=.045):
    if env not in ENVIRONMENTS:
        raise ValueError(f'unknown environment: {env}')
    if jitter < 0:
        raise ValueError('jitter must be nonnegative')
    if env == 'wide_proposals':
        return candidate_bank(target,n,rng,proposal_scale=scale*1.55,
                              broad_prob=max(broad_prob, .65))
    bank = candidate_bank(target,n,rng,proposal_scale=scale,broad_prob=broad_prob)
    if env == 'base' or jitter == 0:
        return bank
    # Shift *candidate* geometry, independently of network/reward, and enforce
    # the same artboard constraints as the production base environment.
    offsets = rng.normal(0,jitter,size=bank.shape).astype(np.float32)
    offsets[...,2:] *= .6
    r = bank + offsets
    r[...,2:] = np.clip(r[...,2:],.08,.86)
    r[...,0] = np.minimum(np.maximum(r[...,0],.04), .96-r[...,2])
    r[...,1] = np.minimum(np.maximum(r[...,1],.04), .96-r[...,3])
    return r.astype(np.float32)
