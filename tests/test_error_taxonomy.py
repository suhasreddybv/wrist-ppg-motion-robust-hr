import numpy as np
import pytest

from src.eval.error_taxonomy import classify, lowest_in_band_bpm

FLOOR = lowest_in_band_bpm()


def _one(est, true_hr, acc):
    lab, harm = classify(np.array([est]), np.array([true_hr]), np.array([acc]), FLOOR)
    return lab[0], harm[0]


def test_lowest_in_band_bin():
    assert FLOOR == pytest.approx(24.375, abs=1e-3)      # 0.40625 Hz on the 4096-point grid
    assert lowest_in_band_bpm(nfft=512) == pytest.approx(30.0)   # the coarse b2 grid


def test_floor_wins_over_acc_lock():
    """Precedence is a decision: a floor-pinned window that also sits on an ACC harmonic
    is counted as floor, because the fix is a search bound rather than motion handling."""
    label, harmonic = _one(est=FLOOR, true_hr=150.0, acc=2 * FLOOR)   # also 0.5x of ACC
    assert label == "floor"
    assert harmonic == ""


def test_acc_lock_records_which_harmonic():
    assert _one(est=100.0, true_hr=150.0, acc=100.0) == ("acc_locked", "1x")
    assert _one(est=100.0, true_hr=150.0, acc=50.0) == ("acc_locked", "2x")
    assert _one(est=50.0, true_hr=150.0, acc=100.0) == ("acc_locked", "0.5x")


def test_acc_lock_wins_over_true_harmonic():
    # estimate is both 0.5x the truth and on the ACC fundamental: ACC takes precedence
    label, harmonic = _one(est=75.0, true_hr=150.0, acc=75.0)
    assert label == "acc_locked" and harmonic == "1x"


def test_harmonic_of_true_hr():
    assert _one(est=75.0, true_hr=150.0, acc=40.0)[0] == "harmonic"
    assert _one(est=160.0, true_hr=80.0, acc=40.0)[0] == "harmonic"


def test_other_when_nothing_matches():
    assert _one(est=95.0, true_hr=150.0, acc=42.0)[0] == "other"


def test_tolerances_are_edges_not_ranges():
    assert _one(est=FLOOR + 2.0, true_hr=150.0, acc=42.0)[0] == "floor"
    assert _one(est=FLOOR + 2.1, true_hr=150.0, acc=42.0)[0] == "other"
    assert _one(est=103.0, true_hr=150.0, acc=100.0)[0] == "acc_locked"
    assert _one(est=103.1, true_hr=150.0, acc=100.0)[0] == "other"


def test_every_window_gets_exactly_one_label():
    rng = np.random.default_rng(0)
    n = 500
    est = rng.uniform(24, 240, n)
    hr = rng.uniform(50, 180, n)
    acc = rng.uniform(24, 240, n)
    lab, _ = classify(est, hr, acc, FLOOR)
    assert len(lab) == n
    assert set(np.unique(lab)) <= {"floor", "acc_locked", "harmonic", "other"}
