"""Stage 4a - accelerometer-informed spectral masking.

The taxonomy (D-025) rules out a single-line approach: 58% of confirmed locks sit
on the 0.5x stride subharmonic rather than the step fundamental, so masking only
the dominant ACC peak would leave most of the damage in place.

Motion spectrum: the power spectra of the three wrist-ACC axes (resampled to
64 Hz) are summed, and the power spectrum of the magnitude signal |acc| is added
to that sum. Per-axis spectra carry direction-specific cadence lines whose
strength depends on how the watch happens to sit on the wrist; the magnitude is
orientation-invariant and reinforces lines present in the movement as a whole but
suppresses pure rotations. Summing power keeps both, is linear, and avoids
choosing an orientation.

Masking: the top-K prominent peaks of that motion spectrum, each with its 0.5x
and 2x harmonics, are attenuated in the PPG power spectrum with a multiplicative
Gaussian notch. The notch width is tied to the measured width of each ACC peak
rather than a fixed constant, so a sharp treadmill-like cadence is notched
narrowly and a smeared one broadly. Bins are down-weighted, never zeroed: a heart
rate that coincides with a cadence line must remain findable.

K, the prominence threshold, the width factor and the notch depth are
hyperparameters. They are never chosen here - `src/eval/stage4.py` selects them
inside each LOSO fold on training subjects only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from src.features.preprocess import CARDIAC_BAND_HZ
from src.models.baselines import ZERO_PAD_NFFT

ACC_NFFT = 2048
HARMONICS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class MaskConfig:
    """Hyperparameters for spectral masking. Chosen on training subjects only."""

    n_peaks: int = 3            # K
    prominence: float = 0.05    # peak prominence, fraction of the spectrum maximum
    width_factor: float = 2.0   # notch sigma = width_factor x measured half-width of the ACC peak
    depth: float = 0.1          # residual gain at the notch centre (0 = zeroing, 1 = no masking)
    min_sigma_hz: float = 0.02  # floor on notch sigma, so a razor-thin peak still removes something
    motion_gate_g: float = 0.02  # below this ACC magnitude SD the window is treated as still

    def label(self) -> str:
        return (f"K={self.n_peaks},prom={self.prominence},width={self.width_factor},"
                f"depth={self.depth},gate={self.motion_gate_g}")


@dataclass(frozen=True)
class MaskResult:
    hr_bpm: np.ndarray          # (n_windows,) estimate from the masked spectrum
    masked_energy: np.ndarray   # (n_windows,) share of in-band PPG power removed
    n_lines: np.ndarray         # (n_windows,) number of notch centres applied


def motion_spectrum(acc_win: np.ndarray, fs: int, nfft: int = ACC_NFFT
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Combined ACC power spectrum per window: 3 axes summed, plus the magnitude signal.

    acc_win: (n_windows, samples, 3) in g. Returns (freqs_hz, power) with power
    normalised so each window's maximum is 1.
    """
    acc = acc_win - acc_win.mean(axis=1, keepdims=True)
    mag = np.linalg.norm(acc_win, axis=2)
    mag = mag - mag.mean(axis=1, keepdims=True)
    taper = np.hanning(acc.shape[1])[None, :, None]

    p_axes = (np.abs(np.fft.rfft(acc * taper, n=nfft, axis=1)) ** 2).sum(axis=2)
    p_mag = np.abs(np.fft.rfft(mag * taper[:, :, 0], n=nfft, axis=1)) ** 2
    power = p_axes + p_mag
    peak = power.max(axis=1, keepdims=True)
    return np.fft.rfftfreq(nfft, 1 / fs), power / np.where(peak > 0, peak, 1.0)


def _lines_for_window(freqs: np.ndarray, power: np.ndarray, cfg: MaskConfig
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Notch centres (Hz) and sigmas (Hz) for one window's motion spectrum."""
    band = (freqs >= CARDIAC_BAND_HZ[0] * 0.5) & (freqs <= CARDIAC_BAND_HZ[1] * 2)
    idx = np.flatnonzero(band)
    peaks, props = signal.find_peaks(power[idx], prominence=cfg.prominence)
    if not len(peaks):
        return np.empty(0), np.empty(0)

    order = np.argsort(props["prominences"])[::-1][:cfg.n_peaks]
    peaks = peaks[order]
    widths = signal.peak_widths(power[idx], peaks, rel_height=0.5)[0]
    df = freqs[1] - freqs[0]
    half_width_hz = np.maximum(widths * df / 2.0, cfg.min_sigma_hz)

    centres, sigmas = [], []
    for p, hw in zip(peaks, half_width_hz):
        f0 = freqs[idx][p]
        for h in HARMONICS:
            centres.append(f0 * h)
            sigmas.append(cfg.width_factor * hw * h)   # harmonics scale in width too
    return np.asarray(centres), np.asarray(sigmas)


def mask_gains(acc_windows: np.ndarray, fs: int, f_band_hz: np.ndarray, cfg: MaskConfig
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per-window multiplicative gain over the in-band frequency grid, plus the line count.

    Gains depend only on the accelerometer and the config, so they are computed once and
    reused for the plain spectrum and for an adaptive-cancellation residual.
    """
    acc_freqs, acc_power = motion_spectrum(acc_windows, fs)
    # A still wrist has no motion lines to find: the normalised ACC spectrum is noise and
    # peak-picking would invent notches. Windows below the gate are left unmasked.
    motion = np.linalg.norm(acc_windows, axis=2).std(axis=1)
    still = motion < cfg.motion_gate_g

    gains = np.ones((len(acc_windows), len(f_band_hz)))
    n_lines = np.zeros(len(acc_windows), dtype=int)
    for i in range(len(acc_windows)):
        if still[i]:
            continue
        centres, sigmas = _lines_for_window(acc_freqs, acc_power[i], cfg)
        if not len(centres):
            continue
        g = np.ones_like(f_band_hz)
        for c, sg in zip(centres, sigmas):
            g *= 1.0 - (1.0 - cfg.depth) * np.exp(-0.5 * ((f_band_hz - c) / max(sg, 1e-6)) ** 2)
        gains[i] = g
        n_lines[i] = len(centres)
    return gains, n_lines


def band_spectra(bvp_windows: np.ndarray, fs: int, band: tuple[float, float] = CARDIAC_BAND_HZ,
                 nfft: int = ZERO_PAD_NFFT) -> tuple[np.ndarray, np.ndarray]:
    """In-band power spectra of each window, on the b2-zp grid. No taper - see apply_masking."""
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    power = np.abs(np.fft.rfft(bvp_windows, n=nfft, axis=1)) ** 2
    return freqs[in_band], power[:, in_band]


def apply_masking(bvp_windows: np.ndarray, acc_windows: np.ndarray, fs: int,
                  cfg: MaskConfig, band: tuple[float, float] = CARDIAC_BAND_HZ,
                  nfft: int = ZERO_PAD_NFFT) -> MaskResult:
    """Mask motion lines out of each PPG spectrum, then take the in-band argmax.

    No taper on the PPG side: b2-zp takes a plain FFT, and the ablation must isolate
    masking. Tapering would reduce leakage and quietly contribute its own gain; if it is
    ever wanted it becomes its own ablation row. The ACC side is tapered because it only
    has to locate motion lines, not to stay comparable with a baseline.
    """
    f_band, spec = band_spectra(bvp_windows, fs, band, nfft)
    gains, n_lines = mask_gains(acc_windows, fs, f_band, cfg)
    masked = spec * gains
    total = spec.sum(axis=1)
    removed = np.where(total > 0, 1.0 - masked.sum(axis=1) / np.where(total > 0, total, 1.0), 0.0)
    hr = f_band[masked.argmax(axis=1)] * 60.0
    return MaskResult(hr, removed, n_lines)
