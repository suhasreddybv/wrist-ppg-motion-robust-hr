import numpy as np
import pytest

from src.eval import loso
from src.eval.loso import (
    SubjectArrays,
    loso_predictions,
    per_activity_rows,
    per_subject_rows,
)
from src.eval.metrics import bland_altman, mae, pearson_r, rmse
from src.models.baselines import (
    bin_width_bpm,
    global_mean_hr,
    previous_window_hr,
    spectral_peak_hr,
)
from tests.test_loader import needs_data

FS = 64


def _window(hr_bpm, n=512, fs=FS):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * (hr_bpm / 60) * t)


# --- baselines ---------------------------------------------------------------

def test_bin_width_is_7_5_bpm():
    assert bin_width_bpm() == pytest.approx(7.5)
    assert bin_width_bpm(nfft=4096) == pytest.approx(0.9375)


@pytest.mark.parametrize("hr", [60.0, 75.0, 90.0, 120.0, 180.0])
def test_spectral_peak_recovers_a_clean_sine_within_one_bin(hr):
    est = spectral_peak_hr(_window(hr)[None, :], fs=FS)[0]
    assert abs(est - hr) <= bin_width_bpm()


def test_spectral_peak_zero_padding_sharpens_the_estimate():
    hr = 71.0   # deliberately off-grid
    coarse = spectral_peak_hr(_window(hr)[None, :], fs=FS)[0]
    fine = spectral_peak_hr(_window(hr)[None, :], fs=FS, nfft=4096)[0]
    assert abs(fine - hr) < abs(coarse - hr)


def test_spectral_peak_ignores_out_of_band_energy():
    w = _window(75) * 0.2 + 5 * np.sin(2 * np.pi * 12 * np.arange(512) / FS)
    assert abs(spectral_peak_hr(w[None, :], fs=FS)[0] - 75) <= bin_width_bpm()


def test_spectral_peak_locks_onto_a_stronger_motion_component():
    """The failure mode b2 exists to expose: a large cadence peak wins the argmax."""
    hr, cadence = 150.0, 110.0
    w = _window(hr) + 4 * _window(cadence)
    assert abs(spectral_peak_hr(w[None, :], fs=FS)[0] - cadence) <= bin_width_bpm()


def test_global_mean_hr():
    assert global_mean_hr(np.array([60.0, 80.0, 100.0])) == pytest.approx(80.0)


def test_previous_window_hr_shifts_and_uses_fallback():
    hr = np.array([70.0, 72.0, 75.0, 71.0])
    out = previous_window_hr(hr, fallback=65.0)
    np.testing.assert_allclose(out, [65.0, 70.0, 72.0, 75.0])


# --- metrics -----------------------------------------------------------------

def test_metrics_known_values():
    y = np.array([60.0, 70.0, 80.0])
    p = np.array([62.0, 68.0, 85.0])
    assert mae(y, p) == pytest.approx(3.0)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 25) / 3))
    assert pearson_r(y, p) == pytest.approx(0.9639, abs=1e-4)


def test_bland_altman_bias_and_limits():
    y = np.zeros(100)
    p = np.concatenate([np.full(50, 4.0), np.full(50, 6.0)])
    bias, lo, hi = bland_altman(y, p)
    assert bias == pytest.approx(5.0)
    assert lo < bias < hi


# --- LOSO mechanics ----------------------------------------------------------

def _fake_subjects():
    """Three subjects; the third has no activity 8, mimicking S6's missing activities."""
    return [
        SubjectArrays("A", np.full(10, 60.0), np.array([0] * 2 + [1] * 4 + [8] * 4),
                      np.zeros(10, bool), np.full(10, 61.0)),
        SubjectArrays("B", np.full(10, 80.0), np.array([0] * 2 + [1] * 4 + [8] * 4),
                      np.array([False] * 8 + [True] * 2), np.full(10, 95.0)),
        SubjectArrays("C", np.full(10, 100.0), np.array([0] * 2 + [1] * 8),
                      np.zeros(10, bool), np.full(10, 101.0)),
    ]


def test_b0_uses_training_subjects_only():
    subs = _fake_subjects()
    preds = loso_predictions(subs)
    assert preds["A"]["b0"][0] == pytest.approx(90.0)    # mean of B and C, not A
    assert preds["C"]["b0"][0] == pytest.approx(70.0)    # mean of A and B


def test_b1_first_window_does_not_use_its_own_label():
    subs = _fake_subjects()
    hr = np.array([70.0, 90.0] + [80.0] * 8)
    subs[0] = SubjectArrays("A", hr, subs[0].activity, subs[0].clipped, subs[0].b2)
    preds = loso_predictions(subs)
    assert preds["A"]["b1"][0] == preds["A"]["b0"][0]    # fallback, not 70.0
    assert preds["A"]["b1"][1] == 70.0


def test_per_activity_fold_counts_reflect_missing_activities():
    rows = per_activity_rows(_fake_subjects(), loso_predictions(_fake_subjects()))
    by = {(r["method"], r["activity"]): r for r in rows}
    assert by[("b2", "sitting")]["n_folds"] == 3
    assert by[("b2", "working")]["n_folds"] == 2          # subject C has no activity 8
    assert by[("b2", "POOLED (incl. transient)")]["n_folds"] == 3
    assert ("b2", "cycling") not in by      # no subject performed it: no row, no crash


def test_pooled_includes_transients_and_activity_rows_exclude_them():
    subs = _fake_subjects()
    rows = per_activity_rows(subs, loso_predictions(subs))
    activities = {r["activity"] for r in rows}
    assert "transient" not in activities
    pooled = next(r for r in rows if r["method"] == "b2" and r["activity"].startswith("POOLED"))
    assert pooled["n_windows"] == 30                       # all windows of all three subjects


def test_b2_noclip_drops_flagged_windows_only():
    subs = _fake_subjects()
    preds = loso_predictions(subs)
    rows = {(r["method"], r["subject"]): r for r in per_subject_rows(subs, preds)}
    assert rows[("b2", "B")]["n_windows"] == 10
    assert rows[("b2_noclip", "B")]["n_windows"] == 8
    assert rows[("b2", "A")]["n_windows"] == rows[("b2_noclip", "A")]["n_windows"] == 10


def test_oracle_flag_is_carried_into_outputs():
    subs = _fake_subjects()
    rows = per_subject_rows(subs, loso_predictions(subs))
    assert all(r["oracle"] for r in rows if r["method"] == "b1")
    assert not any(r["oracle"] for r in rows if r["method"] != "b1")
    assert "ORACLE" in loso.METHOD_LABELS["b1"]


# --- real data ---------------------------------------------------------------

@needs_data
def test_b2_is_accurate_at_rest_and_degrades_with_motion():
    """Guards the stop conditions: b2 must work when still, and must not beat the oracle."""
    subs = build = loso.build_subject_arrays(["S1", "S2", "S3"])
    preds = loso_predictions(subs)
    by = {(r["method"], r["activity"]): r for r in per_activity_rows(subs, preds)}
    assert by[("b2", "sitting")]["mae"] < 8.0
    assert by[("b2", "stairs")]["mae"] > by[("b2", "sitting")]["mae"]
    for activity in ("sitting", "stairs", "cycling", "walking"):
        assert by[("b1", activity)]["mae"] < by[("b2", activity)]["mae"], activity
        assert by[("b2", activity)]["mae"] >= 3.0, activity
    assert len(build) == 3
