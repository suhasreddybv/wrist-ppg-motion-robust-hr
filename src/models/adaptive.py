"""Stage 4b - adaptive cancellation of motion from the PPG, in the time domain.

NLMS with the three wrist-ACC axes as reference inputs. Each 8 s window is filtered
independently, starting from zero weights, so no state crosses a window boundary and
nothing leaks between subjects.

The sample loop runs over the 512 samples of a window, vectorised across all windows
of a subject at once: every window carries its own weight vector and they are updated
in parallel. A per-window Python loop would be ~33 million iterations over the cohort;
this is 512 numpy operations per subject.

A batch least-squares variant is also provided. It is the solution NLMS converges to
within a window, so it stands in for RLS: same reference model, no step size, no
convergence transient. Its advantage is that it cannot be blamed on a badly chosen mu;
its disadvantage is that it cannot track within a window.

Filter order and step size are hyperparameters, chosen in `src/eval/stage4.py` inside
each LOSO fold on training subjects only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveConfig:
    order: int = 8           # FIR taps per reference channel
    mu: float = 0.1          # NLMS step size
    eps: float = 1e-6        # regularisation in the NLMS normalisation
    method: str = "nlms"     # "nlms" or "ls" (batch least squares, converged-RLS stand-in)
    ridge: float = 1e-3      # ridge term for the least-squares variant

    def label(self) -> str:
        return (f"{self.method},order={self.order},mu={self.mu}" if self.method == "nlms"
                else f"{self.method},order={self.order},ridge={self.ridge}")


def _design(acc: np.ndarray, order: int) -> np.ndarray:
    """(n_windows, samples, 3) -> (n_windows, samples, 3*order) of lagged references."""
    n, t, c = acc.shape
    acc = acc - acc.mean(axis=1, keepdims=True)
    out = np.zeros((n, t, c * order), dtype=float)
    for lag in range(order):
        out[:, lag:, lag * c:(lag + 1) * c] = acc[:, :t - lag, :]
    return out


def nlms_cancel(bvp: np.ndarray, acc: np.ndarray, cfg: AdaptiveConfig) -> np.ndarray:
    """Remove the ACC-predictable part of each BVP window; returns the residual.

    bvp: (n_windows, samples) band-passed and z-scored. acc: (n_windows, samples, 3) in g.
    """
    x = _design(acc, cfg.order)                      # (n, t, p)
    n, t, p = x.shape
    w = np.zeros((n, p))
    residual = np.empty_like(bvp)
    for k in range(t):
        xk = x[:, k, :]                              # (n, p)
        y = np.einsum("np,np->n", w, xk)             # filter output
        e = bvp[:, k] - y                            # residual = what ACC cannot explain
        norm = np.einsum("np,np->n", xk, xk) + cfg.eps
        w += (cfg.mu * e / norm)[:, None] * xk
        residual[:, k] = e
    return residual


def ls_cancel(bvp: np.ndarray, acc: np.ndarray, cfg: AdaptiveConfig) -> np.ndarray:
    """Batch least-squares cancellation: the solution NLMS converges to within a window."""
    x = _design(acc, cfg.order)
    xtx = np.einsum("ntp,ntq->npq", x, x)
    xty = np.einsum("ntp,nt->np", x, bvp)
    p = xtx.shape[-1]
    scale = np.trace(xtx, axis1=1, axis2=2) / p
    xtx = xtx + (cfg.ridge * np.maximum(scale, 1e-12))[:, None, None] * np.eye(p)
    w = np.linalg.solve(xtx, xty[..., None])[..., 0]
    return bvp - np.einsum("ntp,np->nt", x, w)


def cancel(bvp: np.ndarray, acc: np.ndarray, cfg: AdaptiveConfig) -> np.ndarray:
    if cfg.method == "nlms":
        return nlms_cancel(bvp, acc, cfg)
    if cfg.method == "ls":
        return ls_cancel(bvp, acc, cfg)
    raise ValueError(f"unknown method {cfg.method!r}")
