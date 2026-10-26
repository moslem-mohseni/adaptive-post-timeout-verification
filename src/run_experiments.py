"""Paper 3 experiments: policies x services, sensitivity, ablations. Writes CSVs to ../results."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from policies import (COSTS, Adaptive, AdaptiveCfg, AlwaysProbe, ContractGuard, Ep, FixedWait,
                      ImmediateRetry, Oracle, make_ctx)
from world import PATTERNS, sample_service, true_episode

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def policies(best_wait):
    return [
        ImmediateRetry(), AlwaysProbe(), FixedWait(best_wait), FixedWait(2.0 if best_wait != 2.0 else 4.0), ContractGuard(),
        Adaptive(AdaptiveCfg(name="adaptive")),
        Adaptive(AdaptiveCfg(use_staleness=False, name="adaptive_no_staleness")),
        Adaptive(AdaptiveCfg(use_evidence=False, name="adaptive_no_evidence")),
        Adaptive(AdaptiveCfg(use_contract=False, name="adaptive_no_contract")),
        Oracle(),
    ]


def run_block(n, seed0, prior_mode, scale, n_hist, pols, costs=COSTS, pattern=None,retry_loss=0.0,sigma_factor=1.0):
    rows = []
    for i in range(n):
        seed = seed0 + i
        rng = np.random.default_rng(seed)
        svc = sample_service(rng, pattern)
        truth = true_episode(rng, svc,sigma_factor=sigma_factor)
        # Independent counterfactual retry outcome, shared by every policy;
        # a separate stream preserves the original world when loss is added.
        truth['retry_lost'] = bool(np.random.default_rng(seed*13+9).random()<retry_loss)
        for pol in pols:
            ctx_rng = np.random.default_rng(seed * 7 + 1)
            ctx = make_ctx(ctx_rng, svc, prior_mode, scale, n_hist)
            ep = Ep(svc, truth, costs,retry_loss=retry_loss)
            dec = pol.run(ep, ctx)
            r = ep.finish(dec)
            r.update(policy=pol.name, seed=seed, pattern=svc.pattern, lag_mean=svc.lag_mean,
                     idem_doc=int(svc.idem_doc), idem_true=int(svc.idem_true),
                     bound=svc.bound_doc if svc.bound_doc else 0.0, bound_true=int(svc.bound_true),
                     prior=prior_mode, scale=scale, n_hist=n_hist,sigma_factor=sigma_factor)
            rows.append(r)
    return rows


def tune_wait(n=600):
    """Pick the best fixed wait on a development stream (seeds disjoint from test)."""
    best, best_c = None, 1e9
    for w in (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0):
        df = pd.DataFrame(run_block(n, 900000, "oracle_prior", 1.0, 60, [FixedWait(w)]))
        c = df.cost.mean()
        if c < best_c:
            best, best_c = w, c
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    bw = tune_wait()
    print("best fixed wait", bw,flush=True)
    open(os.path.join(RES, "best_wait.txt"), "w").write(str(bw))
    pols = policies(bw)
    # E1: main (known pattern parameters)
    pd.DataFrame(run_block(a.n, 0, "oracle_prior", 1.0, 60, pols)).to_csv(os.path.join(RES, "e1_main.csv"), index=False)
    print('E1 complete',flush=True)
    # E2: learned prior from n_hist logged writes
    rows = []
    for nh in (20, 60, 300):
        rows += run_block(a.n // 2, 0, "learned", 1.0, nh, [pols[5], pols[2], pols[4]])
    pd.DataFrame(rows).to_csv(os.path.join(RES, "e2_learned.csv"), index=False)
    print('E2 complete',flush=True)
    # E3: misspecified latency prior
    rows = []
    for sc in (0.25, 0.5, 1.0, 2.0, 4.0):
        rows += run_block(a.n // 2, 0, "misspecified", sc, 60, [pols[5], pols[2], pols[4]])
    pd.DataFrame(rows).to_csv(os.path.join(RES, "e3_misspec.csv"), index=False)
    print('E3 complete',flush=True)
    # E4: cost-model sensitivity (duplicate penalty)
    rows = []
    for dup in (10.0, 30.0, 100.0, 300.0, 1000.0):
        c = dict(COSTS, dup=dup, false_success=dup)
        for r in run_block(a.n // 2, 0, "oracle_prior", 1.0, 60, [pols[5], pols[2], pols[4], pols[1], pols[0]], costs=c):
            r["dup_cost"] = dup
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "e4_costs.csv"), index=False)
    print('E4 complete',flush=True)
    # Revision sensitivities fixed before observing these fresh seed streams.
    rows=[]
    for rho in (0.0,0.05,0.15,0.30):
        rows+=run_block(a.n//2,50000,'oracle_prior',1.0,60,[pols[0],pols[2],pols[4],pols[5],pols[-1]],retry_loss=rho)
    pd.DataFrame(rows).to_csv(os.path.join(RES,'e5_retry_loss.csv'),index=False)
    print('E5 complete',flush=True)
    rows=[]
    for factor in (1.0,1.5,2.0):
        rows+=run_block(a.n//2,60000,'oracle_prior',1.0,60,[pols[2],pols[4],pols[5]],sigma_factor=factor)
    pd.DataFrame(rows).to_csv(os.path.join(RES,'e6_heavy_tail.csv'),index=False)
    print('E6 complete',flush=True)
    print("done")


if __name__ == "__main__":
    main()
