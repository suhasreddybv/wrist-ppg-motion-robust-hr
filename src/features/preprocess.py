"""Stage 2 - band-pass, alignment, per-window z-scoring and a signal-quality index.

Band: 0.4-4 Hz (24-240 bpm), 4th-order Butterworth, zero-phase via filtfilt.
The band is deliberately wide. A narrower 0.8-2.5 Hz band appears to solve motion
artifact by construction, because it excludes the frequencies where walking and
cycling cadence collide with heart rate, and it makes the estimator useless above
150 bpm - exactly the stairs and cycling regime. `bandpass` takes an explicit band
so a narrow-band comparison can be run, and results from it must be labelled.

Two implementation choices, both visible in the output:
- The continuous record is filtered before windowing. filtfilt on an isolated 8 s
  window leaves edge transients at a 0.4 Hz cutoff; filtering the record once and
  then cutting windows does not. Windowing is unchanged (Stage 1 grid).
- Wrist ACC is resampled 32 -> 64 Hz (polyphase) so it shares a time base with the
  BVP, which time-domain adaptive cancellation in Stage 4 needs. The 32 Hz windows
  stay available on the WindowedSubject.

Perfusion index is not computed: the E4 BVP is zero-centred with the DC component
removed (verified; see data/README.md), so it cannot be derived.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from src.data.loader import SubjectRecord
from src.data.windows import WindowedSubject, window_array, window_subject

CARDIAC_BAND_HZ = (0.4, 4.0)
FILTER_ORDER = 4
PEAK_BAND_HZ = 0.25      # half-width around the spectral peak for concentration
MIN_HR_BPM, MAX_HR_BPM = 24.0, 240.0
BEAT_HALF_S = 0.25       # half-width of a beat template


@dataclass(frozen=True)
class QualityIndex:
    """Per-window signal quality. Each array is (n_windows,)."""

    template_corr: np.ndarray        # mean correlation of each beat with the window's template
    spectral_concentration: np.ndarray  # in-band power near the dominant peak / total in-band power
    out_of_band_ratio: np.ndarray    # power outside 0.4-4 Hz / total power, from the unfiltered window
    combined: np.ndarray             # spectral_concentration; see note on the template term below


@dataclass(frozen=True)
class PreprocessedSubject:
    """Stage 1 windows plus their filtered, z-scored form."""

    windows: WindowedSubject
    bvp_filtered: np.ndarray   # (n_windows, 512) band-passed, then z-scored per window
    acc_64: np.ndarray         # (n_windows, 512, 3) wrist ACC resampled to 64 Hz, in g
    sqi: QualityIndex
    band_hz: tuple[float, float]
    fs: int = 64

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def subject_id(self) -> str:
        return self.windows.subject_id


def bandpass(x: np.ndarray, fs: int, band: tuple[float, float] = CARDIAC_BAND_HZ,
             order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0."""
    lo, hi = band
    if not 0 < lo < hi < fs / 2:
        raise ValueError(f"band {band} invalid for fs={fs}")
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def zscore_windows(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-score each window independently. Flat windows stay at zero rather than blowing up."""
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True)
    return (w - mu) / np.maximum(sd, eps)


def resample_acc(acc: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resample of (samples, 3) accelerometer data."""
    if fs_out % fs_in:
        raise ValueError(f"{fs_in} -> {fs_out} Hz is not an integer upsample")
    return signal.resample_poly(acc, fs_out // fs_in, 1, axis=0)


def _spectral_concentration(w: np.ndarray, fs: int, band=CARDIAC_BAND_HZ,
                            half_width=PEAK_BAND_HZ) -> np.ndarray:
    freqs = np.fft.rfftfreq(w.shape[1], 1 / fs)
    power = np.abs(np.fft.rfft(w, axis=1)) ** 2
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    p_band = power[:, in_band]
    f_band = freqs[in_band]
    peak_f = f_band[p_band.argmax(axis=1)]
    near = np.abs(f_band[None, :] - peak_f[:, None]) <= half_width
    total = p_band.sum(axis=1)
    return np.where(total > 0, (p_band * near).sum(axis=1) / np.maximum(total, 1e-20), 0.0)


def _out_of_band_ratio(raw_w: np.ndarray, fs: int, band=CARDIAC_BAND_HZ) -> np.ndarray:
    """Fraction of power outside the cardiac band, measured on the unfiltered window."""
    w = raw_w - raw_w.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(w.shape[1], 1 / fs)
    power = np.abs(np.fft.rfft(w, axis=1)) ** 2
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    total = power.sum(axis=1)
    return np.where(total > 0, power[:, ~in_band].sum(axis=1) / np.maximum(total, 1e-20), 1.0)


def _template_correlation(w: np.ndarray, fs: int) -> np.ndarray:
    """Mean correlation between each detected beat and the window's average beat.

    Peaks are found on the filtered, z-scored window with a refractory distance set by
    MAX_HR_BPM. Windows with fewer than three beats score 0: not enough to form a template.
    """
    half = int(BEAT_HALF_S * fs)
    min_dist = int(fs * 60 / MAX_HR_BPM)
    out = np.zeros(len(w))
    for i, x in enumerate(w):
        peaks, _ = signal.find_peaks(x, distance=min_dist, prominence=0.3)
        peaks = peaks[(peaks >= half) & (peaks < len(x) - half)]
        if len(peaks) < 3:
            continue
        beats = np.stack([x[p - half:p + half] for p in peaks])
        beats = beats - beats.mean(axis=1, keepdims=True)
        template = beats.mean(axis=0)
        t_norm = np.linalg.norm(template)
        b_norm = np.linalg.norm(beats, axis=1)
        good = (b_norm > 0) & (t_norm > 0)
        if not good.any():
            continue
        out[i] = float(np.mean((beats[good] @ template) / (b_norm[good] * t_norm)))
    return out


def composite_with_template(template_corr: np.ndarray, concentration: np.ndarray) -> np.ndarray:
    """The pre-decision composite, kept so `validate_sqi` can reproduce the comparison."""
    return np.clip(template_corr, 0, 1) * concentration


def quality_index(bvp_filtered_z: np.ndarray, bvp_raw_w: np.ndarray, fs: int,
                  band=CARDIAC_BAND_HZ) -> QualityIndex:
    """Per-window quality. The composite is spectral concentration alone.

    The beat-template term was dropped on 22 Sep 2026 after validation against error
    (results/sqi_validation.csv): pooled over non-transient windows, the composite
    without it scores AUROC 0.722 for detecting |error| > 10 bpm, inside the 95% CI
    of the full composite ([0.69, 0.76] around 0.729). The simpler score is
    statistically indistinguishable, so the term goes. It is still computed and
    reported as a component.
    """
    tc = _template_correlation(bvp_filtered_z, fs)
    sc = _spectral_concentration(bvp_filtered_z, fs, band)
    ob = _out_of_band_ratio(bvp_raw_w, fs, band)
    return QualityIndex(tc, sc, ob, sc.copy())


def preprocess_subject(rec: SubjectRecord, band: tuple[float, float] = CARDIAC_BAND_HZ,
                       acc_fs_out: int = 64) -> PreprocessedSubject:
    """Window one subject, band-pass and z-score its BVP, and score window quality."""
    ws = window_subject(rec)
    n = len(ws)
    bvp, acc = rec["wrist_bvp"], rec["wrist_acc"]

    filtered = bandpass(bvp.data, bvp.fs, band)
    bvp_f = zscore_windows(window_array(filtered, bvp.fs, n))

    acc_up = resample_acc(acc.data, acc.fs, acc_fs_out)
    acc_64 = window_array(acc_up, acc_fs_out, n)

    sqi = quality_index(bvp_f, ws.bvp, bvp.fs, band)
    return PreprocessedSubject(windows=ws, bvp_filtered=bvp_f, acc_64=acc_64, sqi=sqi,
                               band_hz=band, fs=bvp.fs)
