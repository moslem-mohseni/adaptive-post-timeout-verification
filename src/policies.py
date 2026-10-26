"""Recovery policies after a write timeout, and the episode environment (Paper 3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world import PATTERNS, T_TIMEOUT, Service, draw_latents, fit_prior, true_episode

# cost model (abstract units; sensitivity analysed in the paper)
COSTS = dict(dup=100.0, false_success=100.0, escalate=25.0, probe=2.0, evidence=6.0,
             per_second=1.0, retry=1.0)
PROBE_DELAY, EVIDENCE_DELAY = 0.2, 1.0


class Ep:
    """One timed-out write. Policies interact through wait/probe/evidence/finish."""

    def __init__(self, svc: Service, truth: dict, costs=COSTS, retry_loss=0.0):
        self.svc, self.truth, self.costs = svc, truth, costs
        self.t = T_TIMEOUT
        self.n_probe = self.n_evid = 0
        self.info_cost = 0.0
        self.retry_loss = retry_loss

    def wait(self, dt):
        self.t += dt

    def probe(self):
        """Status endpoint: never a false positive, but lags behind commits."""
        self.n_probe += 1
        self.info_cost += self.costs["probe"]
        self.t += PROBE_DELAY
        tr = self.truth
        return (not tr["lost"]) and tr["c"] + tr["lag"] <= self.t

    def evidence(self):
        """Independent authoritative evidence (e.g. downstream ledger): sees every commit so far."""
        self.n_evid += 1
        self.info_cost += self.costs["evidence"]
        self.t += EVIDENCE_DELAY
        tr = self.truth
        return (not tr["lost"]) and tr["c"] <= self.t

    def finish(self, decision: str):
        tr, c = self.truth, self.costs
        dup = false_succ = esc = retry_missing = premature = 0
        loss = 0.0
        if decision == "success":
            if tr["lost"]:
                false_succ, loss = 1, c["false_success"]
            premature = int(not tr['lost'] and tr['c'] > self.t)
        elif decision == "retry":
            loss += c["retry"]
            retry_lost = tr.get('retry_lost', False)
            if (not tr["lost"]) and not self.svc.idem_true and not retry_lost:
                dup, loss = 1, loss + c["dup"]
            if tr['lost'] and retry_lost:
                retry_missing, loss = 1, loss + c['false_success']
        else:
            esc, loss = 1, c["escalate"]
        elapsed = self.t - T_TIMEOUT
        total = loss + self.info_cost + c["per_second"] * elapsed
        effects=int(not tr['lost'])
        if decision=='retry' and not tr.get('retry_lost',False):
            effects=1 if self.svc.idem_true else effects+1
        return dict(decision=decision, dup=dup, false_success=false_succ, escalated=esc,
                    cost=total, time=elapsed, probes=self.n_probe, evidence=self.n_evid,
                    lost=int(tr["lost"]),retry_missing=retry_missing,premature_success=premature,
                    retry_loss=self.retry_loss,eventual_effects=effects)


# ---------------------------------------------------------------- baselines

class FixedWait:
    """Wait w seconds, check status once; present -> success, absent -> retry."""
    def __init__(self, w):
        self.w, self.name = w, f"fixed_wait_{w:g}s"

    def run(self, ep, ctx):
        ep.wait(self.w)
        return "success" if ep.probe() else "retry"


class ImmediateRetry:
    name = "immediate_retry"

    def run(self, ep, ctx):
        return "retry"


class AlwaysProbe:
    name = "probe_then_retry"

    def run(self, ep, ctx):
        return "success" if ep.probe() else "retry"


class ContractGuard:
    """Fixed rule from documented guarantees: idempotency key -> retry; documented in-flight bound
    -> wait it out, read independent evidence; otherwise wait w and probe."""
    name = "contract_guard"

    def __init__(self, w=2.0):
        self.w = w

    def run(self, ep, ctx):
        svc = ep.svc
        if svc.idem_doc:
            return "retry"
        if svc.bound_doc is not None:
            ep.wait(max(0.0, svc.bound_doc - (ep.t - 0.0)))
            return "success" if ep.evidence() else "retry"
        ep.wait(self.w)
        return "success" if ep.probe() else "retry"


class Oracle:
    """Lower bound: knows the latent truth (not implementable)."""
    name = "oracle"

    def run(self, ep, ctx):
        tr = ep.truth
        if tr["lost"]:
            if tr.get('retry_lost', False):
                return 'escalate'
            return "retry"
        return "success"


# ---------------------------------------------------------------- adaptive controller

@dataclass
class AdaptiveCfg:
    use_staleness: bool = True     # model status-endpoint lag
    use_evidence: bool = True      # allow independent-evidence reads
    use_contract: bool = True      # use documented idempotency / bound
    waits: tuple = (0.0, 1.0, 3.0, 8.0)
    n_particles: int = 1500
    max_steps: int = 8
    name: str = "adaptive"


class Adaptive:
    """Particle belief over (lost, commit time, lag); one-step value-of-information lookahead."""

    def __init__(self, cfg: AdaptiveCfg | None = None):
        self.cfg = cfg or AdaptiveCfg()
        self.name = self.cfg.name

    def _prior(self, svc: Service, rng, params):
        cfg = self.cfg
        p = params
        lag_mean = svc.lag_mean if cfg.use_staleness else 0.0
        bound = svc.bound_doc if cfg.use_contract else None
        d = draw_latents(rng, p["rho"], p["alpha"], p["mu"], p["sigma"], lag_mean,
                         bound=bound, n=cfg.n_particles)
        return d["lost"], d["c"], d["lag"]

    def _terminal(self, lost, costs, p_idem, retry_loss=0.0):
        n = max(len(lost), 1)
        p_lost = lost.sum() / n if len(lost) else 0.5
        l_succ = costs["false_success"] * p_lost
        l_retry = costs["retry"] + (1-retry_loss) * (1 - p_idem) * costs["dup"] * (1 - p_lost) + retry_loss*p_lost*costs['false_success']
        l_esc = costs["escalate"]
        return min((l_succ, "success"), (l_retry, "retry"), (l_esc, "escalate"))

    def run(self, ep, ctx):
        cfg, costs = self.cfg, ep.costs
        rng = ctx["rng"]
        lost, c, lag = self._prior(ep.svc, rng, ctx["params"])
        p_idem = 0.9 if (cfg.use_contract and ep.svc.idem_doc) else 0.0
        for _ in range(cfg.max_steps):
            if len(lost) < 10:
                break
            cur_loss, _ = self._terminal(lost, costs, p_idem, ep.retry_loss)
            best_net, best = 0.0, None
            kinds = ["probe"] + (["evidence"] if cfg.use_evidence else [])
            for kind in kinds:
                delay = PROBE_DELAY if kind == "probe" else EVIDENCE_DELAY
                ucost = costs[kind]
                for w in cfg.waits:
                    tt = ep.t + w + delay
                    vis = (~lost) & ((c + lag <= tt) if kind == "probe" else (c <= tt))
                    pp = vis.mean()
                    after = 0.0
                    for mask, prob in ((vis, pp), (~vis, 1 - pp)):
                        if prob > 0:
                            after += prob * self._terminal_loss_only(lost[mask], costs, p_idem, kind_present=(mask is vis),retry_loss=ep.retry_loss)
                    net = cur_loss - after - ucost - costs["per_second"] * (w + delay)
                    if net > best_net:
                        best_net, best = net, (kind, w, tt)
            if best is None:
                break
            kind, w, tt = best
            ep.wait(w)
            seen = ep.probe() if kind == "probe" else ep.evidence()
            vis = (~lost) & ((c + lag <= ep.t) if kind == "probe" else (c <= ep.t))
            keep = vis if seen else ~vis
            if seen:
                return "success"
            lost, c, lag = lost[keep], c[keep], lag[keep]
        if len(lost) < 10:
            return "escalate"
        return self._terminal(lost, costs, p_idem, ep.retry_loss)[1]

    def _terminal_loss_only(self, lost_sub, costs, p_idem, kind_present, retry_loss=0.0):
        if kind_present:
            return 0.0  # seeing a commit means success with certainty (no false positives)
        return self._terminal(lost_sub, costs, p_idem,retry_loss)[0]


def make_ctx(rng, svc, mode="oracle_prior", scale=1.0, n_hist=60):
    """Controller's belief parameters: true pattern parameters, perturbed, or learned from history."""
    base = PATTERNS[svc.pattern]
    if mode == "learned":
        params = fit_prior(rng, svc.pattern, n_hist)
    elif mode == "misspecified":
        params = dict(base)
        params["mu"] = base["mu"] + np.log(scale)
    else:
        params = dict(base)
    return dict(rng=rng, params=params)
