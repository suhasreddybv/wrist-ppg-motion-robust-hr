"""Global sequence decoding over candidate spectral peaks.

Diagnostic A (D-032, D-047) says the correct peak is present in about two thirds
of failed windows and greedy selection takes a different one. Greedy tracking has
to commit per window and can only undo a commitment by resetting; decoding the
whole sequence chooses the path that is best overall, so an early wrong choice is
revised when later windows disagree with it.

Model. Each window's states are the top-K peaks of its (masked) spectrum.
  emission(t, j)   = log of the peak's height normalised by the window maximum,
                     so the strongest peak scores 0 and weaker ones score below it.
  transition(i, j) = -(f_j - f_i)^2 / (2 sigma^2), a Gaussian penalty on
                     heart-rate change between consecutive 2 s windows.
A window with no admissible peak falls back to a coarse uniform grid with equal
emission everywhere, so the decoder can pass through it without being forced onto
a spurious line.

Two decoders:
  decode_full      - Viterbi over the whole session. NON-CAUSAL: it uses future
                     windows, so it is not comparable with SpaMa, SpaMaPlus or the
                     greedy tracker. Reported only as an upper bound on what
                     selection alone can achieve.
  decode_fixed_lag - the deployable form. The estimate for window t is emitted
                     after window t+L has been seen, so it is causal with a known
                     delay of L windows (2 s each). This is the version placed
                     beside the published numbers.

K, sigma and L are hyperparameters, chosen inside each LOSO fold.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

NEG = -1e18
FALLBACK_GRID_BPM = np.arange(40.0, 200.0, 10.0)


def top_k_peaks(spectra: np.ndarray, freqs_bpm: np.ndarray, k: int,
                prominence: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Per window, the k most prominent peaks: (freqs, log-normalised heights).

    Windows with no peak get the fallback grid with flat emission, so the decoder
    passes through on the transition term alone.
    """
    n = len(spectra)
    f = np.full((n, k), np.nan)
    e = np.full((n, k), NEG)
    for i, spec in enumerate(spectra):
        top = spec.max()
        idx, props = find_peaks(spec, prominence=prominence * top)
        if len(idx) == 0:
            m = min(k, len(FALLBACK_GRID_BPM))
            f[i, :m] = FALLBACK_GRID_BPM[:m]
            e[i, :m] = 0.0                      # flat: no evidence either way
            continue
        order = np.argsort(spec[idx])[::-1][:k]
        sel = idx[order]
        f[i, :len(sel)] = freqs_bpm[sel]
        e[i, :len(sel)] = np.log(np.maximum(spec[sel] / top, 1e-12))
    return f, e


def _step(prev_score: np.ndarray, prev_f: np.ndarray, f: np.ndarray, e: np.ndarray,
          sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """One Viterbi step: best predecessor for each state of the current window."""
    d = f[None, :] - prev_f[:, None]                 # (prev, cur)
    trans = -(d ** 2) / (2.0 * sigma ** 2)
    total = prev_score[:, None] + trans
    total = np.where(np.isnan(total), NEG, total)
    back = np.argmax(total, axis=0)
    return total[back, np.arange(len(f))] + e, back


def _forward(freqs: np.ndarray, emis: np.ndarray, sigma: float):
    n, k = freqs.shape
    scores = np.full((n, k), NEG)
    backs = np.zeros((n, k), dtype=int)
    scores[0] = np.where(np.isnan(freqs[0]), NEG, emis[0])
    for t in range(1, n):
        scores[t], backs[t] = _step(scores[t - 1], freqs[t - 1], freqs[t], emis[t], sigma)
        scores[t] = np.where(np.isnan(freqs[t]), NEG, scores[t])
    return scores, backs


def decode_full(freqs: np.ndarray, emis: np.ndarray, sigma: float) -> np.ndarray:
    """Non-causal: the whole session at once. An upper bound, not a deployable method."""
    n, _ = freqs.shape
    scores, backs = _forward(freqs, emis, sigma)
    out = np.empty(n)
    j = int(np.argmax(scores[-1]))
    for t in range(n - 1, -1, -1):
        out[t] = freqs[t, j]
        j = int(backs[t, j])
    return out


def decode_fixed_lag(freqs: np.ndarray, emis: np.ndarray, sigma: float, lag: int) -> np.ndarray:
    """Causal with a delay of `lag` windows: window t is emitted once t+lag is seen."""
    n, _ = freqs.shape
    scores, backs = _forward(freqs, emis, sigma)
    out = np.empty(n)
    for t in range(n):
        end = min(t + lag, n - 1)
        j = int(np.argmax(scores[end]))
        for u in range(end, t, -1):               # walk back to the window being emitted
            j = int(backs[u, j])
        out[t] = freqs[t, j]
    return out
