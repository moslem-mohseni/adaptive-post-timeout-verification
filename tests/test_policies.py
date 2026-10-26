import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from policies import Adaptive, ContractGuard, Ep, FixedWait, Oracle, make_ctx
from world import sample_service, true_episode


def test_probe_never_false_positive():
    rng = np.random.default_rng(0)
    for _ in range(300):
        svc = sample_service(rng)
        tr = true_episode(rng, svc)
        ep = Ep(svc, tr)
        if tr["lost"]:
            assert not ep.probe() and not ep.evidence()


def test_oracle_has_no_duplicates_or_false_success():
    rng = np.random.default_rng(1)
    for _ in range(300):
        svc = sample_service(rng)
        tr = true_episode(rng, svc)
        r = Ep(svc, tr).finish(Oracle().run(Ep(svc, tr), {}))
        assert r["dup"] == 0 and r["false_success"] == 0


def test_episodes_conditioned_on_timeout():
    rng = np.random.default_rng(2)
    for _ in range(100):
        svc = sample_service(rng)
        tr = true_episode(rng, svc)
        assert tr["lost"] or tr["c"] > 0  # sanity


def test_adaptive_runs_and_returns_valid_decision():
    rng = np.random.default_rng(3)
    for i in range(20):
        svc = sample_service(rng)
        tr = true_episode(rng, svc)
        ctx = make_ctx(np.random.default_rng(i), svc)
        d = Adaptive().run(Ep(svc, tr), ctx)
        assert d in ("success", "retry", "escalate")
