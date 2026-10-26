"""Tables and figures for Paper 3."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stats_utils import holm, mean_ci, paired_diff_ci, sign_flip_p

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main():
    e1 = pd.read_csv(os.path.join(RES, "e1_main.csv"))
    bw = float(open(os.path.join(RES, "best_wait.txt")).read())
    fixed_best = f"fixed_wait_{bw:g}s"
    short = f"fixed_wait_{2.0 if bw != 2.0 else 4.0:g}s"
    order = ["immediate_retry", "probe_then_retry", short, fixed_best, "contract_guard",
             "adaptive_no_staleness", "adaptive_no_evidence", "adaptive_no_contract", "adaptive", "oracle"]
    order = [o for o in order if o in set(e1.policy)]
    rows = []
    for p in order:
        g = e1[e1.policy == p]
        m, lo, hi = mean_ci(g.cost.values)
        rows.append(dict(policy=p, cost=m, cost_lo=lo, cost_hi=hi, dup=g.dup.mean(), false_success=g.false_success.mean(),
                         escalated=g.escalated.mean(), time=g.time.mean(), probes=g.probes.mean(), evidence=g.evidence.mean()))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RES, "table_main.csv"), index=False)
    print(tab.round(3).to_string())

    # paired comparisons of adaptive vs each other policy
    ad = e1[e1.policy == "adaptive"].set_index("seed").cost
    pr, ps = [], []
    for p in order:
        if p in ("adaptive", "oracle"):
            continue
        o = e1[e1.policy == p].set_index("seed").cost.loc[ad.index]
        m, lo, hi, dz = paired_diff_ci(ad.values, o.values)
        pv = sign_flip_p(ad.values, o.values, n_perm=2000)
        pr.append(dict(vs=p, mean_diff=m, lo=lo, hi=hi, dz=dz, p=pv))
    pdf = pd.DataFrame(pr)
    pdf["p_holm"] = holm(pdf.p.values)
    pdf.to_csv(os.path.join(RES, "paired_vs_adaptive.csv"), index=False)
    print(pdf.round(4).to_string())

    # by pattern and by status-lag
    bp = e1[e1.policy.isin(["immediate_retry", fixed_best, "contract_guard", "adaptive", "oracle"])].groupby(["pattern", "policy"]).agg(
        cost=("cost", "mean"), dup=("dup", "mean"), false_success=("false_success", "mean"), time=("time", "mean")).reset_index()
    bp.to_csv(os.path.join(RES, "by_pattern.csv"), index=False)
    bl = e1[e1.policy.isin([fixed_best, "contract_guard", "adaptive"])].groupby(["lag_mean", "policy"]).agg(
        cost=("cost", "mean"), dup=("dup", "mean"), false_success=("false_success", "mean")).reset_index()
    bl.to_csv(os.path.join(RES, "by_lag.csv"), index=False)
    # documentation errors: where the contract is wrong
    wrong = e1[e1.policy.isin(["contract_guard", "adaptive", "adaptive_no_contract"])].copy()
    wrong["doc_wrong"] = ((wrong.idem_doc == 1) & (wrong.idem_true == 0)).astype(int)
    bd = wrong[wrong.idem_doc == 1].groupby(["doc_wrong", "policy"]).agg(cost=("cost", "mean"), dup=("dup", "mean"), n=("cost", "size")).reset_index()
    bd.to_csv(os.path.join(RES, "by_doc_error.csv"), index=False)
    print(bd.round(3).to_string())
    print(bp.round(2).to_string())

    # E2 learned prior
    e2 = pd.read_csv(os.path.join(RES, "e2_learned.csv"))
    t2 = e2.groupby(["n_hist", "policy"]).cost.mean().unstack().round(3)
    t2.to_csv(os.path.join(RES, "table_learned_prior.csv"))
    print(t2)
    # E3 misspecification
    e3 = pd.read_csv(os.path.join(RES, "e3_misspec.csv"))
    t3 = e3.groupby(["scale", "policy"]).agg(cost=("cost", "mean"), dup=("dup", "mean"), fs=("false_success", "mean")).unstack()
    t3.to_csv(os.path.join(RES, "table_misspec.csv"))
    print(t3.round(3))
    # E4 cost sensitivity
    e4 = pd.read_csv(os.path.join(RES, "e4_costs.csv"))
    t4 = e4.groupby(["dup_cost", "policy"]).cost.mean().unstack().round(2)
    t4.to_csv(os.path.join(RES, "table_cost_sensitivity.csv"))
    print(t4)

    # Figure 1: cost bars with CI, and error components
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=300)
    sel = [o for o in order if o not in ("adaptive_no_staleness", "adaptive_no_evidence", "adaptive_no_contract")]
    t = tab.set_index("policy").loc[sel]
    y = np.arange(len(sel))
    ax[0].barh(y, t.cost, xerr=[t.cost - t.cost_lo, t.cost_hi - t.cost], color=["#9aa5b1"] * (len(sel) - 2) + ["#1f77b4", "#2ca02c"], height=0.6)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([s.replace("_", " ") for s in sel], fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Mean recovery cost (units)")
    comp = t[["dup", "false_success", "escalated"]]
    comp.plot(kind="barh", stacked=True, ax=ax[1], color=["#d62728", "#ff7f0e", "#7f7f7f"], width=0.6, legend=True)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].set_ylabel("")
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Episode rate")
    ax[1].legend(["duplicate", "false success", "escalated"], frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_main.png"))
    plt.close(fig)

    # Figure 2: misspecification + learned prior
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=300)
    ms = e3.groupby(["scale", "policy"]).cost.mean().unstack()
    for p, c in (("adaptive", "#1f77b4"), (fixed_best, "#7f7f7f"), ("contract_guard", "#ff7f0e")):
        ax[0].plot(ms.index, ms[p], "-o", ms=3, color=c, label=p.replace("_", " "))
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("Prior latency scale error (x true median)")
    ax[0].set_ylabel("Mean recovery cost")
    ax[0].legend(frameon=False, fontsize=7)
    t2p = e2.groupby(["n_hist", "policy"]).cost.mean().unstack()
    for p, c in (("adaptive", "#1f77b4"), (fixed_best, "#7f7f7f"), ("contract_guard", "#ff7f0e")):
        ax[1].plot(t2p.index, t2p[p], "-o", ms=3, color=c, label=p.replace("_", " "))
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Logged historical writes used to fit the prior")
    ax[1].set_ylabel("Mean recovery cost")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_robustness.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
