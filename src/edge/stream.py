"""The device path as a device runs it: one 2-second step at a time, float32, with the band-pass
state carried between steps, a 512-sample ring buffer, and an incremental tracker. It must give
the batch causal_f32 pipeline's estimates exactly (tests/test_edge_stream.py); `src.edge.latency`
times it.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import scipy.fft
from scipy import signal

from src.edge.pipeline import FS, VARIANTS, band_sos, gains_for
from src.features.preprocess import CARDIAC_BAND_HZ
from src.models.baselines import ZERO_PAD_NFFT
from src.models.tracker import TrackerConfig, _peaks

DEVICE = next(v for v in VARIANTS if v.name == "causal_f32")
WINDOW, STEP = 8 * FS, 2 * FS


class TrackerState:
    """One step of `src.models.tracker.track`, carrying its state between calls."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.recent: deque[float] = deque(maxlen=max(1, cfg.history))
        self.jumps = self.sustained = 0

    def step(self, spec: np.ndarray, freqs_bpm: np.ndarray, cand: np.ndarray) -> float:
        cfg = self.cfg
        unconstrained = float(freqs_bpm[int(spec.argmax())])
        if not self.recent:
            self.recent.append(unconstrained)
            return unconstrained
        prediction = float(np.median(self.recent) if cfg.pred_stat == "median" else np.mean(self.recent))
        if cfg.mode == "nearest_peak":
            choice = float(cand[int(np.argmin(np.abs(cand - prediction)))]) if len(cand) else unconstrained
        else:
            near = np.abs(freqs_bpm - prediction) <= cfg.half_width_bpm
            if near.any():
                idx = np.flatnonzero(near)
                choice = float(freqs_bpm[idx[int(spec[idx].argmax())]])
            else:
                choice = unconstrained
        probe = choice if cfg.mode == "nearest_peak" else unconstrained
        self.jumps = self.jumps + 1 if abs(probe - prediction) > cfg.jump_bpm else 0
        if cfg.sustained_rule != "none":
            j = int(np.argmin(np.abs(freqs_bpm - choice)))
            height = float(spec[j])
            if cfg.sustained_rule == "rank":
                if len(cand):
                    heights = spec[np.searchsorted(freqs_bpm, cand).clip(0, len(spec) - 1)]
                    rank = 1 + int((heights > height + 1e-12).sum())
                else:
                    rank = 1
                failing = rank > cfg.rank_max
            else:
                top = float(spec.max())
                failing = top > 0 and (height / top) < cfg.ratio_min
            self.sustained = self.sustained + 1 if failing else 0
        else:
            self.sustained = 0
        if self.jumps >= cfg.reset_after or self.sustained >= cfg.sustained_after:
            choice = unconstrained
            self.recent.clear()
            self.jumps = self.sustained = 0
        self.recent.append(choice)
        return choice


class DeviceStream:
    """Feed raw BVP 128 samples at a time (2 s at 64 Hz) with the window's 64-Hz ACC; get one
    estimate per step once 8 s are buffered."""

    def __init__(self, mask_cfg, tracker_cfg):
        self.sos = band_sos().astype(np.float32)
        self.zi = np.zeros((self.sos.shape[0], 2), np.float32)
        self.buf = np.zeros(0, np.float32)
        freqs = np.fft.rfftfreq(ZERO_PAD_NFFT, 1 / FS)
        self.band = (freqs >= CARDIAC_BAND_HZ[0]) & (freqs <= CARDIAC_BAND_HZ[1])
        self.f_band = freqs[self.band]
        self.f_bpm = self.f_band * 60.0
        self.mask_cfg = mask_cfg
        self.tracker = TrackerState(tracker_cfg)
        self.timings: dict[str, float] = {}

    def push(self, bvp_new: np.ndarray, acc_window_64: np.ndarray | None = None, clock=None) -> float | None:
        t = clock or (lambda: 0.0)
        t0 = t()
        y, self.zi = signal.sosfilt(self.sos, np.asarray(bvp_new, np.float32), zi=self.zi)
        self.buf = np.concatenate([self.buf, y.astype(np.float32)])[-WINDOW:]
        t1 = t()
        if len(self.buf) < WINDOW or acc_window_64 is None:
            return None
        w = self.buf
        w = (w - w.mean()) / max(float(w.std()), 1e-12)
        t2 = t()
        spec = (np.abs(scipy.fft.rfft(w[None, :], n=ZERO_PAD_NFFT, axis=1)) ** 2)[0, self.band].astype(np.float32)
        t3 = t()
        g = gains_for(np.asarray(acc_window_64, np.float32)[None], self.f_band, self.mask_cfg, DEVICE)[0]
        t4 = t()
        masked = np.asarray(spec * g, float)
        cand = _peaks(masked, self.f_bpm, self.tracker.cfg.prominence)
        est = self.tracker.step(masked, self.f_bpm, cand)
        t5 = t()
        self.timings = dict(filter=t1 - t0, zscore=t2 - t1, ppg_fft=t3 - t2, acc_spectra_and_gains=t4 - t3, peaks_and_tracker=t5 - t4)
        return est
