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
    # Sustained-wrongness reset (D-037). The jump test cannot see a smoothly tracked wrong
    # estimate; these look at whether the tracked peak is still a credible peak at all.
    sustained_rule: str = "none"   # "none", "rank" or "ratio"
    rank_max: int = 2              # "rank": tracked peak must be within the top rank_max peaks
    ratio_min: float = 0.3         # "ratio": tracked height / spectrum max must stay above this
    sustained_after: int = 3       # consecutive failing windows before re-seeding

    def label(self) -> str:
        span = f"+-{self.half_width_bpm}" if self.mode == "window" else f"prom={self.prominence}"
        base = f"{self.mode},{span},hist={self.history},jump={self.jump_bpm},reset={self.reset_after}"
        if self.sustained_rule == "none":
            return base
        arg = self.rank_max if self.sustained_rule == "rank" else self.ratio_min
        return f"{base},{self.sustained_rule}={arg},after={self.sustained_after}"


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
          candidates: list[np.ndarray] | None = None, return_resets: bool = False,
          bound_bpm: float | None = None):
    """Constrained peak selection over a sequence of in-band spectra (one row per window).

    spectra: (n_windows, n_bins) power, already masked or cancelled as required.
    freqs_bpm: (n_bins,) bin centres in bpm. Returns (n_windows,) estimates in bpm.
    """
    if bound_bpm is not None:
        keep = freqs_bpm >= bound_bpm
        if keep.any():
            spectra = spectra[:, keep]
            freqs_bpm = freqs_bpm[keep]
            if candidates is not None:
                candidates = [c[c >= bound_bpm] for c in candidates]

    n = len(spectra)
    out = np.empty(n)
    resets = np.zeros(n, dtype=bool)
    recent: deque[float] = deque(maxlen=max(1, cfg.history))
    jumps = sustained = 0

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
            # A bound can remove every candidate from a window; fall back to the (bounded) argmax.
            choice = (float(cand[int(np.argmin(np.abs(cand - prediction)))]) if len(cand)
                      else unconstrained)
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

        # Sustained wrongness: the tracked peak stops being a credible peak, without ever
        # jumping. Rank counts how many peaks stand above it; ratio compares its height with
        # the spectral maximum.
        if cfg.sustained_rule != "none":
            j = int(np.argmin(np.abs(freqs_bpm - choice)))
            height = float(spec[j])
            if cfg.sustained_rule == "rank":
                cand = candidates[i] if candidates is not None else _peaks(spec, freqs_bpm, cfg.prominence)
                if len(cand):
                    heights = spec[np.searchsorted(freqs_bpm, cand).clip(0, len(spec) - 1)]
                    rank = 1 + int((heights > height + 1e-12).sum())
                else:
                    rank = 1
                failing = rank > cfg.rank_max
            else:
                top = float(spec.max())
                failing = top > 0 and (height / top) < cfg.ratio_min
            sustained = sustained + 1 if failing else 0
        else:
            sustained = 0

        if jumps >= cfg.reset_after or sustained >= cfg.sustained_after:
            choice = unconstrained
            recent.clear()
            jumps = sustained = 0
            resets[i] = True

        out[i] = choice
        recent.append(choice)
    return (out, resets) if return_resets else out
