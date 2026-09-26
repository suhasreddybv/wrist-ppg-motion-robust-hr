"""Stage 4 - peak tracking with an estimate-based reset.

Heart rate does not jump 40 bpm in two seconds, so the search is constrained to a
window around a prediction formed from recent estimates. The reset triggers on the
estimates themselves - when the constrained choice keeps landing far from the
prediction, the track is stale and is re-seeded - so the tracker needs no quality
measure at all. That matters here: the SQI is unusable under motion (D-016), and a
gate built on it would fail exactly where tracking is needed.

This mirrors the published SpaMaPlus tracker (Reiss et al. 2019): mean filter over
recent estimates, pick the spectral peak nearest the prediction, reset after
repeated large jumps.

It runs on any in-band spectrum - plain b2-zp, masked, or cancelled - and is
reported on b2-zp alone as its own ablation row.

`history`, `half_width_bpm`, `jump_bpm` and `reset_after` are hyperparameters,
chosen inside each LOSO fold on training subjects only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class TrackerConfig:
    half_width_bpm: float = 12.0   # "window" mode: hard search window around the prediction
    history: int = 6               # estimates in the mean filter that forms the prediction
    jump_bpm: float = 12.0         # a choice this far from the prediction counts as a jump
    reset_after: int = 3           # consecutive jumps before the track is re-seeded
    mode: str = "nearest_peak"     # "nearest_peak" (SpaMaPlus-like) or "window"
    prominence: float = 0.05       # peak prominence for the candidate list, relative to max

    def label(self) -> str:
        span = f"+-{self.half_width_bpm}" if self.mode == "window" else f"prom={self.prominence}"
        return f"{self.mode},{span},hist={self.history},jump={self.jump_bpm},reset={self.reset_after}"


def _peaks(spectrum: np.ndarray, freqs_bpm: np.ndarray, prominence: float) -> np.ndarray:
    """Candidate frequencies: prominent local maxima, falling back to the argmax."""
    idx, _ = find_peaks(spectrum, prominence=prominence * spectrum.max())
    if not len(idx):
        return np.array([freqs_bpm[int(spectrum.argmax())]])
    return freqs_bpm[idx]


def peak_candidates(spectra: np.ndarray, freqs_bpm: np.ndarray, prominence: float) -> list[np.ndarray]:
    """Candidate peaks per window, computed once and reused across tracker configurations."""
    return [_peaks(spec, freqs_bpm, prominence) for spec in spectra]


def track(spectra: np.ndarray, freqs_bpm: np.ndarray, cfg: TrackerConfig,
          candidates: list[np.ndarray] | None = None) -> np.ndarray:
    """Constrained peak selection over a sequence of in-band spectra (one row per window).

    spectra: (n_windows, n_bins) power, already masked or cancelled as required.
    freqs_bpm: (n_bins,) bin centres in bpm. Returns (n_windows,) estimates in bpm.
    """
    n = len(spectra)
    out = np.empty(n)
    recent: deque[float] = deque(maxlen=max(1, cfg.history))
    jumps = 0

    for i in range(n):
        spec = spectra[i]
        unconstrained = float(freqs_bpm[int(spec.argmax())])
        if not recent:
            out[i] = unconstrained
            recent.append(unconstrained)
            continue

        prediction = float(np.mean(recent))
        if cfg.mode == "nearest_peak":
            # Choose the prominent peak closest to the prediction, wherever it is. There is
            # no hard search window: a hard window forces the jump test onto the global
            # argmax, which under motion disagrees with the prediction almost every window
            # and resets the track continuously - leaving the tracker a near no-op.
            cand = candidates[i] if candidates is not None else _peaks(spec, freqs_bpm, cfg.prominence)
            choice = float(cand[int(np.argmin(np.abs(cand - prediction)))])
        else:
            near = np.abs(freqs_bpm - prediction) <= cfg.half_width_bpm
            if near.any():
                idx = np.flatnonzero(near)
                choice = float(freqs_bpm[idx[int(spec[idx].argmax())]])
            else:
                choice = unconstrained

        # The jump test looks at the chosen peak against the prediction. In "window" mode the
        # choice is capped by the window, so the unconstrained argmax is used instead; that is
        # what makes "window" mode reset so readily.
        probe = choice if cfg.mode == "nearest_peak" else unconstrained
        jumps = jumps + 1 if abs(probe - prediction) > cfg.jump_bpm else 0

        if jumps >= cfg.reset_after:
            choice = unconstrained
            recent.clear()
            jumps = 0

        out[i] = choice
        recent.append(choice)
    return out
