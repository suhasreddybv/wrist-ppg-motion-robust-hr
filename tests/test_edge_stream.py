"""The streaming device path equals the batch pipeline, and the incremental tracker equals
`track` step for step, across every tracker configuration in the grids."""
from __future__ import annotations

import numpy as np
import pytest

from src.edge.pipeline import VARIANTS, acc_windows, bvp_spectra, filter_bvp, gains_for
from src.edge.stream import STEP, WINDOW, DeviceStream, TrackerState
from src.eval.stage4 import MASK_GRID, TRACK_GRID
from src.eval.week3 import RESET_GRID
from src.models.tracker import _peaks, peak_candidates, track

DEVICE = next(v for v in VARIANTS if v.name == "causal_f32")


@pytest.mark.parametrize("cfg", TRACK_GRID + RESET_GRID, ids=lambda c: c.label())
def test_incremental_tracker_equals_track(cfg):
    rng = np.random.default_rng(0)
    f_bpm = np.linspace(24, 240, 231)
    spectra = rng.gamma(1.0, 1.0, size=(200, 231))
    spectra[:, 60] += np.linspace(5, 0, 200)          # a drifting true peak among noise
    cand = peak_candidates(spectra, f_bpm, cfg.prominence)
    ref = track(spectra, f_bpm, cfg, cand)
    st = TrackerState(cfg)
    got = np.array([st.step(spectra[i], f_bpm, cand[i]) for i in range(len(spectra))])
    np.testing.assert_array_equal(got, ref)


def test_stream_equals_batch_on_a_synthetic_record():
    rng = np.random.default_rng(1)
    fs, seconds = 64, 120
    t = np.arange(fs * seconds) / fs
    bvp = 50 * np.sin(2 * np.pi * 1.3 * t) + 10 * rng.normal(size=t.size)
    acc32 = np.stack([np.sin(2 * np.pi * 1.8 * np.arange(32 * seconds) / 32) * 0.3, 0.1 * rng.normal(size=32 * seconds),
                      1 + 0.05 * rng.normal(size=32 * seconds)], axis=1)
    n = (seconds - 8) // 2 + 1
    mcfg, tcfg = MASK_GRID[0], TRACK_GRID[-1]
    f_band, spec = bvp_spectra(filter_bvp(bvp, DEVICE, 1.0), n, DEVICE)
    accw = acc_windows(acc32, 32, n, DEVICE)
    masked = np.asarray(spec * gains_for(accw, f_band, mcfg, DEVICE), float)
    batch = track(masked, f_band * 60, tcfg, peak_candidates(masked, f_band * 60, tcfg.prominence))
    dev = DeviceStream(mcfg, tcfg)
    dev.push(bvp[: WINDOW - STEP])
    got = []
    for k in range(n):
        lo = WINDOW - STEP + k * STEP
        got.append(dev.push(bvp[lo: lo + STEP], accw[k]))
    np.testing.assert_allclose(got, batch, atol=1e-9)


def test_a_burst_from_the_first_window_reproduces_the_continuous_run():
    """Guards the duty-cycle result: the cold-start penalty is the cold start, not the burst code."""
    from src.edge.burst import burst_estimates
    rng = np.random.default_rng(2)
    fs, seconds = 64, 100
    t = np.arange(fs * seconds) / fs
    bvp = 50 * np.sin(2 * np.pi * (1.2 + 0.2 * np.sin(2 * np.pi * t / 60)) * t) + 15 * rng.normal(size=t.size)
    acc32 = np.stack([0.3 * np.sin(2 * np.pi * 1.7 * np.arange(32 * seconds) / 32), 0.1 * rng.normal(size=32 * seconds),
                      1 + 0.05 * rng.normal(size=32 * seconds)], axis=1)
    n = (seconds - 8) // 2 + 1
    mcfg, tcfg = MASK_GRID[0], TRACK_GRID[-1]
    f_band, spec = bvp_spectra(filter_bvp(bvp, DEVICE, 1.0), n, DEVICE)
    gains = gains_for(acc_windows(acc32, 32, n, DEVICE), f_band, mcfg, DEVICE)
    masked = np.asarray(spec * gains, float)
    cont = track(masked, f_band * 60, tcfg, peak_candidates(masked, f_band * 60, tcfg.prominence))
    for k in (0, 5, n - 1):
        got = burst_estimates(bvp, gains, f_band, tcfg, np.array([k]), k + 1, zi_init=False)
        assert got[0] == pytest.approx(cont[k], abs=1e-9)
