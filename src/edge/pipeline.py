"""The frozen mask+tracker(hist=15) pipeline, re-run under the arithmetic and filtering a wrist
device would use. One function per stage, parameterised by a `Variant`; the tracker and the
per-fold configurations are the frozen ones (results/edge_fold_configs.csv).

Variants (docs/edge-budget.md):
  reference    float64, zero-phase band-pass over the whole record - the published pipeline (14.70 bpm)
  causal_f64   float64, causal band-pass (a device cannot filter backwards in time)
  causal_f32   float32 throughout the signal path: native on a Cortex-M4F
  q15          16-bit data: BVP from a 16-bit converter at an in-fold full scale, the band-pass as
               32-bit-state biquads, windows and spectra in 16-bit block floating point, Q15 FFTs,
               accelerometer at 16 bits over +-4 g, notch gains in Q15
  q15_state16  as q15 but the band-pass keeps only 16-bit state between sections (the naive form)
  q8           as q15 but windows, spectra and gains stored in 8 bits, accelerometer at its native
               1/64 g (8 bits over +-2 g)
The tracker itself runs in float64 in every variant: it handles one number per window and is not
where a device spends its arithmetic.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace

import numpy as np
import scipy.fft
from scipy import signal

from src.data.loader import REPO_ROOT, SHIFT_S, WINDOW_S, load_subject
from src.data.windows import window_array, window_subject
from src.edge.fixed_point import block_quantize, fft_fixed, quantize, sosfilt_fixed
from src.eval.stage4 import MASK_GRID, TRACK_GRID
from src.features.preprocess import CARDIAC_BAND_HZ, FILTER_ORDER, bandpass, resample_acc, zscore_windows
from src.models.baselines import ZERO_PAD_NFFT
from src.models.masking import ACC_NFFT, _lines_for_window
from src.models.tracker import peak_candidates, track

FS = 64
CHUNK = 256


@dataclass(frozen=True)
class Variant:
    name: str
    causal: bool
    arith: str                  # "f64", "f32" or "fixed"
    state_bits: int = 32        # fixed: band-pass state between sections
    store_bits: int = 16        # fixed: windows, spectra, gains
    acc_native: bool = False    # fixed: keep the accelerometer at 1/64 g (8 bits over +-2 g)


VARIANTS = (
    Variant("reference", causal=False, arith="f64"),
    Variant("causal_f64", causal=True, arith="f64"),
    Variant("causal_f32", causal=True, arith="f32"),
    Variant("q15", causal=True, arith="fixed"),
    Variant("q15_state16", causal=True, arith="fixed", state_bits=16),
    Variant("q8", causal=True, arith="fixed", store_bits=8, acc_native=True),
)


def band_sos() -> np.ndarray:
    return signal.butter(FILTER_ORDER, CARDIAC_BAND_HZ, btype="bandpass", fs=FS, output="sos")


def fold_configs() -> dict[str, tuple]:
    """Subject -> (MaskConfig, TrackerConfig at hist=15), as each LOSO fold selected them."""
    out = {}
    for r in csv.DictReader(open(REPO_ROOT / "results" / "edge_fold_configs.csv")):
        m = next(c for c in MASK_GRID if c.label() == r["mask_config"])
        t = replace(next(c for c in TRACK_GRID if c.label() == r["tracker_config"]), history=int(r["history"]))
        out[r["subject"]] = (m, t)
    return out


# ------------------------------------------------------------------ the band-pass

def filter_bvp(bvp: np.ndarray, v: Variant, full_scale: float, zi_init: bool = False) -> np.ndarray:
    """Band-pass the raw BVP record (or a burst of it) under the variant's arithmetic.
    `full_scale` is the converter range for the fixed variants (chosen in-fold). `zi_init`
    starts a causal float filter in steady state at the first sample (a burst after waking);
    otherwise a causal filter starts from rest."""
    sos = band_sos()
    if not v.causal:
        return bandpass(bvp, FS)
    if v.arith in ("f64", "f32"):
        dt = np.float64 if v.arith == "f64" else np.float32
        s, x = sos.astype(dt), np.asarray(bvp, dt)
        if zi_init:
            return signal.sosfilt(s, x, zi=(signal.sosfilt_zi(s) * x[0]).astype(dt))[0]
        return signal.sosfilt(s, x)
    codes = quantize(bvp, 16, full_scale) / full_scale * 2 ** 15          # 16-bit converter codes
    if v.state_bits == 32:
        y = sosfilt_fixed(sos, codes * 2 ** 14, data_bits=32) / 2 ** 14   # Q15 codes carried in Q29, 2 bits headroom
    else:
        y = sosfilt_fixed(sos, np.round(codes / 4), data_bits=16) * 4     # Q13: 2 bits headroom in 16 bits
    return np.clip(np.round(y), -2 ** 15, 2 ** 15 - 1)                    # back to 16-bit samples


# ------------------------------------------------------------------ spectra

def _fft_power(x: np.ndarray, nfft: int, v: Variant) -> np.ndarray:
    """|FFT|^2 of each row (real input) under the variant; scale-free for the fixed path."""
    if v.arith == "f64":
        return np.abs(np.fft.rfft(x, n=nfft, axis=1)) ** 2
    if v.arith == "f32":
        return (np.abs(scipy.fft.rfft(np.asarray(x, np.float32), n=nfft, axis=1)) ** 2).astype(np.float32)
    peak = np.max(np.abs(x), axis=1, keepdims=True)
    xs = x / np.where(peak > 0, peak, 1.0) * (1 - 2.0 ** -15)            # block exponent: into [-1, 1)
    out = np.empty((len(x), nfft // 2 + 1))
    for i in range(0, len(x), CHUNK):
        out[i:i + CHUNK] = np.abs(fft_fixed(xs[i:i + CHUNK], nfft, 15)) ** 2
    return out


def bvp_spectra(filtered: np.ndarray, n_windows: int, v: Variant) -> tuple[np.ndarray, np.ndarray]:
    """In-band power spectra of the z-scored windows, on the b2-zp grid (no taper, as frozen)."""
    w = window_array(filtered, FS, n_windows)
    w = zscore_windows(np.asarray(w, np.float32 if v.arith == "f32" else float))
    if v.arith == "fixed":
        w = block_quantize(w, v.store_bits)
    freqs = np.fft.rfftfreq(ZERO_PAD_NFFT, 1 / FS)
    band = (freqs >= CARDIAC_BAND_HZ[0]) & (freqs <= CARDIAC_BAND_HZ[1])
    p = _fft_power(w, ZERO_PAD_NFFT, v)[:, band]
    if v.arith == "fixed":
        p = block_quantize(p, v.store_bits)
    return freqs[band], p


def acc_windows(acc_g: np.ndarray, acc_fs: int, n_windows: int, v: Variant) -> np.ndarray:
    """Wrist ACC (g, 32 Hz) resampled to 64 Hz and windowed, at the variant's resolution."""
    a = acc_g
    if v.arith == "fixed":
        a = quantize(a, 8, 2.0)                                            # the E4's own 1/64 g, +-2 g
    up = resample_acc(a, acc_fs, FS)
    if v.arith == "fixed":
        up = quantize(up, 8, 2.0) if v.acc_native else quantize(up, 16, 4.0)
    w = window_array(up, FS, n_windows)
    return np.asarray(w, np.float32) if v.arith == "f32" else w


def motion_power(acc_win: np.ndarray, v: Variant) -> tuple[np.ndarray, np.ndarray]:
    """The masking module's motion spectrum (3 axes + magnitude, Hann taper, 2048 points, each
    window normalised to its maximum), under the variant. One block exponent per window across
    the four signals, so their relative power is preserved."""
    acc = acc_win - acc_win.mean(axis=1, keepdims=True)
    mag = np.linalg.norm(acc_win, axis=2)
    mag = mag - mag.mean(axis=1, keepdims=True)
    taper = np.hanning(acc.shape[1])
    if v.arith == "fixed":
        taper = quantize(taper, 16, 1.0)
    sig = np.concatenate([acc * taper[None, :, None], (mag * taper[None, :])[:, :, None]], axis=2)   # (W, S, 4)
    if v.arith == "fixed":
        peak = np.max(np.abs(sig), axis=(1, 2), keepdims=True)
        sig = sig / np.where(peak > 0, peak, 1.0) * (1 - 2.0 ** -15)
    W = len(sig)
    flat = np.moveaxis(sig, 2, 1).reshape(W * 4, -1)
    p = _fft_power(flat, ACC_NFFT, v).reshape(W, 4, -1).sum(axis=1)
    peak = p.max(axis=1, keepdims=True)
    p = p / np.where(peak > 0, peak, 1.0)
    if v.arith == "fixed":
        p = block_quantize(p, v.store_bits)
    return np.fft.rfftfreq(ACC_NFFT, 1 / FS), p


def gains_for(acc_win: np.ndarray, f_band_hz: np.ndarray, cfg, v: Variant) -> np.ndarray:
    """`masking.mask_gains` with the motion spectrum computed under the variant; identical logic."""
    acc_freqs, acc_power = motion_power(acc_win, v)
    motion = np.linalg.norm(acc_win, axis=2).std(axis=1)
    gains = np.ones((len(acc_win), len(f_band_hz)))
    for i in range(len(acc_win)):
        if motion[i] < cfg.motion_gate_g:
            continue
        centres, sigmas = _lines_for_window(acc_freqs, acc_power[i], cfg)
        if not len(centres):
            continue
        g = np.ones_like(f_band_hz)
        for c, sg in zip(centres, sigmas):
            g *= 1.0 - (1.0 - cfg.depth) * np.exp(-0.5 * ((f_band_hz - c) / max(sg, 1e-6)) ** 2)
        gains[i] = g
    if v.arith == "fixed":
        gains = quantize(gains, v.store_bits, 1.0)
    return gains


def estimate(spec: np.ndarray, gains: np.ndarray, f_band_hz: np.ndarray, tcfg, v: Variant) -> np.ndarray:
    masked = spec * gains
    if v.arith == "fixed":
        masked = block_quantize(masked, v.store_bits)
    f_bpm = f_band_hz * 60.0
    masked = np.asarray(masked, float)
    return track(masked, f_bpm, tcfg, peak_candidates(masked, f_bpm, tcfg.prominence))


def run_subject(sid: str, v: Variant, configs: dict, full_scale: float) -> dict:
    rec = load_subject(sid)
    ws = window_subject(rec)
    n = len(ws)
    mcfg, tcfg = configs[sid]
    filtered = filter_bvp(rec["wrist_bvp"].data, v, full_scale)
    f_band, spec = bvp_spectra(filtered, n, v)
    accw = acc_windows(rec["wrist_acc"].data, rec["wrist_acc"].fs, n, v)
    gains = gains_for(accw, f_band, mcfg, v)
    return dict(sid=sid, hr=ws.hr, activity=ws.activity, pred=estimate(spec, gains, f_band, tcfg, v))


def bvp_full_scale(train_sids: list[str], p999: dict[str, float]) -> float:
    """The converter range for a fold: the largest 99.9th percentile of |BVP| among the training
    subjects. Larger excursions saturate, as they would on a converter sized from training data."""
    return max(p999[s] for s in train_sids)
