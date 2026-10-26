"""Simulated services and the latent world behind a timed-out write (Paper 3).

A write request is sent at t=0 with client timeout T. Latent truth:
  lost      : the write never commits
  c         : commit time (if not lost)
  ack_lost  : the acknowledgement is lost (write may have committed)
  lag       : status-endpoint visibility lag after commit (0 for strongly consistent services)
Episodes are conditioned on observing a timeout: lost, or ack lost, or commit+ack_delay > T.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

T_TIMEOUT = 2.0
PATTERNS = {
    # rho = P(write lost), alpha = P(ack lost), (mu, sigma) = lognormal commit latency
    "ack_loss":     dict(rho=0.03, alpha=0.15, mu=math.log(0.30), sigma=0.5),
    "slow_backend": dict(rho=0.03, alpha=0.02, mu=math.log(1.50), sigma=1.0),
    "lossy":        dict(rho=0.25, alpha=0.05, mu=math.log(0.50), sigma=0.6),
}


@dataclass
class Service:
    pattern: str
    lag_mean: float        # 0 => strong status endpoint
    idem_doc: bool         # documentation claims idempotency keys are honoured
    idem_true: bool        # they really are
    bound_doc: float | None  # documented max in-flight time (None if absent)
    bound_true: bool       # documented bound really holds


def sample_service(rng: np.random.Generator, pattern: str | None = None) -> Service:
    pattern = pattern or rng.choice(list(PATTERNS))
    lag = float(rng.choice([0.0, 1.0, 3.0], p=[0.4, 0.3, 0.3]))
    idem_doc = bool(rng.random() < 0.5)
    idem_true = idem_doc and rng.random() < 0.9      # 10% of documented keys are not honoured
    bound = float(rng.choice([6.0, 12.0])) if rng.random() < 0.3 else None
    bound_true = bound is not None and rng.random() < 0.9
    return Service(pattern, lag, idem_doc, idem_true, bound, bound_true)


def draw_latents(rng, rho, alpha, mu, sigma, lag_mean, bound=None, n=1, ack_delay_sigma=0.1,
                 condition_on_timeout=True, enforce_bound=True):
    """Vectorised draw of latent worlds conditioned on a timeout. Returns dict of arrays (length n)."""
    out = {k: [] for k in ("lost", "c", "lag", "ack_lost", "ack_delay")}
    have = 0
    while have < n:
        m = max(4 * (n - have), 64)
        lost = rng.random(m) < rho
        c = rng.lognormal(mu, sigma, m)
        if bound is not None and enforce_bound:
            c = np.minimum(c, bound)  # service guarantees commit within `bound`
        ack_lost = rng.random(m) < alpha
        ack_delay = rng.lognormal(math.log(0.05), ack_delay_sigma, m)
        timeout = lost | ack_lost | (c + ack_delay > T_TIMEOUT)
        lag = rng.exponential(lag_mean, m) if lag_mean > 0 else np.zeros(m)
        keep = timeout if condition_on_timeout else np.ones(m, bool)
        for k, v in (("lost", lost), ("c", c), ("lag", lag), ("ack_lost", ack_lost), ("ack_delay", ack_delay)):
            out[k].append(v[keep])
        have += int(keep.sum())
    return {k: np.concatenate(v)[:n] for k, v in out.items()}


def true_episode(rng, svc: Service, sigma_factor=1.0):
    p = dict(PATTERNS[svc.pattern])
    p['sigma'] *= sigma_factor
    d = draw_latents(rng, p["rho"], p["alpha"], p["mu"], p["sigma"], svc.lag_mean,
                     bound=svc.bound_doc, n=1, enforce_bound=svc.bound_true)
    return dict(lost=bool(d["lost"][0]), c=float(d["c"][0]), lag=float(d["lag"][0]),
                ack_lost=bool(d['ack_lost'][0]),ack_delay=float(d['ack_delay'][0]))


def fit_prior(rng, pattern: str, n_hist: int):
    """Estimate pattern parameters from n_hist unconditioned historical writes (monitoring logs)."""
    p = PATTERNS[pattern]
    lost = rng.random(n_hist) < p["rho"]
    c = rng.lognormal(p["mu"], p["sigma"], n_hist)
    ack_lost = rng.random(n_hist) < p["alpha"]
    obs_c = np.log(c[~lost])
    mu_hat = float(obs_c.mean())
    sg_hat = float(max(obs_c.std(ddof=1), 0.2))
    rho_hat = float(np.clip(lost.mean(), 0.005, 0.6))
    alpha_hat = float(np.clip(ack_lost[~lost].mean(), 0.005, 0.6))
    return dict(rho=rho_hat, alpha=alpha_hat, mu=mu_hat, sigma=sg_hat)
