import json

import numpy as np
import pytest

from layoutpilot.sweep_stats import (episode_matrix, gap_closed, load_runs, paired_bootstrap, summarize,
                                     to_markdown)


def _write(root, method, steps, seed, policy, initial, oracle, random):
    d = root / method / f"steps_{steps}" / f"seed_{seed}"
    d.mkdir(parents=True)
    m = {"initial_score": float(np.mean(initial)), "ppo_policy_score": float(np.mean(policy)),
         "greedy_oracle_score": float(np.mean(oracle)), "random_policy_score": float(np.mean(random)),
         "per_episode": {"initial": list(initial), "policy": list(policy), "greedy_oracle": list(oracle),
                         "random": list(random)}}
    (d / "metrics.json").write_text(json.dumps(m))


def _fake_sweep(root):
    rng = np.random.default_rng(0)
    initial = rng.uniform(.6, .75, 50)
    oracle = initial + rng.uniform(.2, .3, 50)
    random = initial - .05
    for seed in range(3):
        for steps, frac in ((8192, .2), (131072, .8)):
            noise = rng.normal(0, .01, 50)
            _write(root, "ppo", steps, seed, initial + frac * (oracle - initial) + noise, initial, oracle, random)
            _write(root, "bc_ppo", steps, seed, initial + (frac + .05) * (oracle - initial) + noise, initial, oracle, random)
    return initial, oracle


def test_gap_closed():
    assert gap_closed(.6, .8, 1.0) == pytest.approx(.5)
    assert gap_closed(.6, 1.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        gap_closed(.7, .8, .7)


def test_paired_bootstrap():
    a = np.linspace(0, 1, 100)
    res = paired_bootstrap(a, a + .1, iterations=500)
    assert res["delta"] == pytest.approx(.1) and res["significant"]
    res = paired_bootstrap(a, a, iterations=200)
    assert res["delta"] == 0 and not res["significant"]
    with pytest.raises(ValueError):
        paired_bootstrap(a, a[:10])


def test_summarize_sweep(tmp_path):
    _fake_sweep(tmp_path)
    runs = load_runs(tmp_path)
    assert len(runs) == 12
    assert episode_matrix(runs, "ppo", 8192).shape == (3, 50)
    s = summarize(runs, iterations=300)
    rows = {(r["method"], r["steps"]): r for r in s["table"]}
    assert rows[("ppo", 131072)]["gap_closed_mean"] == pytest.approx(.8, abs=.05)
    assert rows[("bc_ppo", 8192)]["n_seeds"] == 3
    bc_vs_ppo = [c for c in s["comparisons"] if c["a"] == "ppo" and c["steps"] == 131072][0]
    assert bc_vs_ppo["delta"] > 0 and bc_vs_ppo["significant"]
    vs_oracle = [c for c in s["comparisons"] if c["a"] == "greedy_oracle" and c["b"] == "ppo" and c["steps"] == 8192][0]
    assert vs_oracle["delta"] < 0 and vs_oracle["significant"]
    md = to_markdown(s)
    assert "Greedy gap closed" in md and "bc_ppo − ppo" in md and "131,072" in md


def test_rejects_inconsistent_runs(tmp_path):
    _fake_sweep(tmp_path)
    bad = tmp_path / "ppo" / "steps_8192" / "seed_0" / "metrics.json"
    m = json.loads(bad.read_text())
    m["per_episode"]["initial"][0] += 1
    bad.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        summarize(load_runs(tmp_path), iterations=10)
    legacy = tmp_path / "ppo" / "steps_8192" / "seed_9"
    legacy.mkdir(parents=True)
    (legacy / "metrics.json").write_text(json.dumps({"ppo_policy_score": .7}))
    with pytest.raises(ValueError):
        load_runs(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path / "nothing")
