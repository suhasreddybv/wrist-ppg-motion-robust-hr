"""Stage 3 - the three baselines, reported before any motion handling.

b0  global mean HR of the training subjects. The floor: anything worse is broken.
b1  previous window's ground-truth HR. Uses labels, so it is an ORACLE reference
    and not a deployable method. It measures how much of the task is pure
    temporal smoothness.
b2  naive spectral peak: FFT of the band-passed window, argmax in 0.4-4 Hz,
    converted to bpm. No motion handling of any kind.
b2-zp  the same estimator zero-padded to nfft=4096, so the peak is located on a
    ~0.94 bpm grid instead of a 7.5 bpm one. Zero-padding interpolates the
    spectrum; it adds no true resolution and cannot undo a motion-locked peak.
    Stage 4 gains are reported against b2-zp, so that finer spectral spacing is
    never mistaken for motion handling.

b2 resolution: an 8 s window at 64 Hz gives 1/8 Hz bins, i.e. 7.5 bpm. The
estimate is therefore quantised, which puts a floor of roughly 1.9 bpm MAE on it
even with a perfect peak. `nfft` can zero-pad for a finer grid, but the default
is the plain FFT the spec calls for.
"""
from __future__ import annotations

import numpy as np

from src.features.preprocess import CARDIAC_BAND_HZ

ZERO_PAD_NFFT = 4096   # b2-zp: ~0.94 bpm spectral spacing, vs 7.5 bpm for the plain FFT

ORACLE_METHODS = frozenset({"b1"})


def global_mean_hr(train_hr: np.ndarray) -> float:
    """b0: one constant, computed on training subjects only."""
    return float(np.mean(train_hr))


def previous_window_hr(hr: np.ndarray, fallback: float) -> np.ndarray:
    """b1 (oracle): predict each window with the previous window's ground truth.

    The first window has no predecessor and uses `fallback`, which must come from
    training subjects; using its own label would leak the answer.
    """
    out = np.empty_like(np.asarray(hr, dtype=float))
    out[0] = fallback
    out[1:] = hr[:-1]
    return out


def spectral_peak_hr(windows: np.ndarray, fs: int = 64,
                     band: tuple[float, float] = CARDIAC_BAND_HZ,
                     nfft: int | None = None) -> np.ndarray:
    """b2: bpm of the largest in-band FFT bin, per window. No motion handling."""
    w = np.asarray(windows, dtype=float)
    n = nfft or w.shape[1]
    freqs = np.fft.rfftfreq(n, 1 / fs)
    power = np.abs(np.fft.rfft(w, n=n, axis=1)) ** 2
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    peak_idx = power[:, in_band].argmax(axis=1)
    return freqs[in_band][peak_idx] * 60.0


def bin_width_bpm(fs: int = 64, window_s: int = 8, nfft: int | None = None) -> float:
    """Spacing of the b2 frequency grid, in bpm."""
    n = nfft or fs * window_s
    return fs / n * 60.0
