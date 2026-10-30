"""Duty-cycled operation: the device sleeps between updates and, on waking, records a burst of B
windows, filters it from steady state at its first sample, and runs the frozen tracker from an
empty history; the estimate at the burst's last window is the update. Accuracy is measured
against burst length, on the device path (causal float32), LOSO with each fold's frozen
configuration.

Update points are every 15th window (one per 30 s) from window 29 on, the same points for every
B, so the comparison across burst lengths and with the continuous pipeline is paired. Each point
is an independent cold start - what a device waking every five minutes faces; the 5-minute
interval itself only enters the energy budget. The LED is on for 8 + 2 (B - 1) seconds per update.

Writes results/edge_burst.csv and results/edge_burst_per_activity.csv.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import scipy.fft

from src.data.loader import REPO_ROOT, SHIFT_S, SUBJECT_IDS, WINDOW_S, load_subject
from src.data.windows import window_subject
from src.edge.evaluate import N_BOOT, SEED, paired
from src.edge.pipeline import FS, VARIANTS, acc_windows, bvp_spectra, filter_bvp, fold_configs, gains_for
from src.eval.loso import ACTIVITY_NAMES
from src.features.preprocess import CARDIAC_BAND_HZ
from src.models.baselines import ZERO_PAD_NFFT
from src.models.tracker import peak_candidates, track

BURSTS = (1, 2, 4, 8, 15, 30)
STRIDE = 15
DEVICE = next(v for v in VARIANTS if v.name == "causal_f32")


def led_on_s(b: int) -> int:
    return WINDOW_S + SHIFT_S * (b - 1)


def burst_estimates(bvp: np.ndarray, gains: np.ndarray, f_band: np.ndarray, tcfg, points: np.ndarray, b: int,
                    zi_init: bool = True) -> np.ndarray:
    """One estimate per update point from a cold-start burst of b windows ending there. `zi_init=False`
    starts the filter from rest, which makes a burst from window 0 identical to the continuous run (tested)."""
    per, step = WINDOW_S * FS, SHIFT_S * FS
    freqs = np.fft.rfftfreq(ZERO_PAD_NFFT, 1 / FS)
    band = (freqs >= CARDIAC_BAND_HZ[0]) & (freqs <= CARDIAC_BAND_HZ[1])
    f_bpm = f_band * 60.0
    out = np.empty(len(points))
    for n, k in enumerate(points):
        first = k - b + 1
        seg = filter_bvp(bvp[first * step: k * step + per], DEVICE, 1.0, zi_init=zi_init)
        w = np.stack([seg[i * step: i * step + per] for i in range(b)]).astype(np.float32)
        w = (w - w.mean(axis=1, keepdims=True)) / np.maximum(w.std(axis=1, keepdims=True), 1e-12)
        p = (np.abs(scipy.fft.rfft(w, n=ZERO_PAD_NFFT, axis=1)) ** 2)[:, band].astype(float)
        masked = p * gains[first:k + 1]
        out[n] = track(masked, f_bpm, tcfg, peak_candidates(masked, f_bpm, tcfg.prominence))[-1]
    return out


def main() -> None:
    configs = fold_configs()
    rows, act_rows, per_subject = [], [], {b: {} for b in BURSTS}
    cont = {}
    for sid in SUBJECT_IDS:
        rec = load_subject(sid)
        ws = window_subject(rec)
        n = len(ws)
        mcfg, tcfg = configs[sid]
        bvp = rec["wrist_bvp"].data
        f_band, spec = bvp_spectra(filter_bvp(bvp, DEVICE, 1.0), n, DEVICE)
        gains = gains_for(acc_windows(rec["wrist_acc"].data, rec["wrist_acc"].fs, n, DEVICE), f_band, mcfg, DEVICE)
        continuous = track(np.asarray(spec * gains, float), f_band * 60.0, tcfg, peak_candidates(np.asarray(spec * gains, float), f_band * 60.0, tcfg.prominence))
        points = np.arange(max(BURSTS) - 1, n, STRIDE)
        hr, act = ws.hr[points], ws.activity[points]
        cont[sid] = (np.abs(continuous[points] - hr), act)
        for b in BURSTS:
            est = burst_estimates(bvp, gains, f_band, tcfg, points, b)
            per_subject[b][sid] = (np.abs(est - hr), act)
        print(f"  {sid}: {len(points)} update points", flush=True)
    rng = np.random.default_rng(SEED)
    c_mae = np.array([cont[s][0].mean() for s in SUBJECT_IDS])
    rows.append(dict(burst_windows="continuous", led_on_s="", update_points=sum(len(cont[s][0]) for s in SUBJECT_IDS),
                     pooled_mae=round(float(c_mae.mean()), 3), minus_continuous="", ci_lo="", ci_hi="", subjects_worse=""))
    for b in BURSTS:
        m = np.array([per_subject[b][s][0].mean() for s in SUBJECT_IDS])
        mean, lo, hi = paired(m - c_mae, rng)
        rows.append(dict(burst_windows=b, led_on_s=led_on_s(b), update_points=sum(len(per_subject[b][s][0]) for s in SUBJECT_IDS),
                         pooled_mae=round(float(m.mean()), 3), minus_continuous=round(mean, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                         subjects_worse=int((m - c_mae > 1e-9).sum())))
        print(f"  B={b:>2} ({led_on_s(b):>2} s of LED): MAE {m.mean():.2f}, minus continuous {mean:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
    for act, name in ACTIVITY_NAMES.items():
        row = dict(activity=name)
        vals = [cont[s][0][cont[s][1] == act].mean() for s in SUBJECT_IDS if (cont[s][1] == act).any()]
        row["continuous"] = round(float(np.mean(vals)), 3); row["folds"] = len(vals)
        for b in BURSTS:
            vals = [per_subject[b][s][0][per_subject[b][s][1] == act].mean() for s in SUBJECT_IDS if (per_subject[b][s][1] == act).any()]
            row[f"burst_{b}"] = round(float(np.mean(vals)), 3)
        act_rows.append(row)
    for data, fn in ((rows, "edge_burst.csv"), (act_rows, "edge_burst_per_activity.csv")):
        with open(REPO_ROOT / "results" / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        print(f"wrote results/{fn}")


if __name__ == "__main__":
    sys.exit(main())
