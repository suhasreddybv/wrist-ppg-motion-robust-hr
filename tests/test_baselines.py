import numpy as np
import pytest

from src.eval import loso
from src.eval.loso import (
    POOLED_ALL,
    POOLED_NO_TRANSIENT,
    SubjectArrays,
    clip_fraction_rows,
    loso_predictions,
    per_activity_rows,
    per_subject_rows,
)
from src.eval.metrics import bland_altman, mae, mape, pearson_r, rmse
from src.models.baselines import (
    ZERO_PAD_NFFT,
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


def test_zero_padding_gives_a_finer_grid_but_no_new_resolution():
    """b2-zp must sharpen a clean peak and leave a motion-locked one where it was."""
    assert bin_width_bpm(nfft=ZERO_PAD_NFFT) < 1.0
    clean = _window(71.0)[None, :]
    coarse = spectral_peak_hr(clean, fs=FS)[0]
    fine = spectral_peak_hr(clean, fs=FS, nfft=ZERO_PAD_NFFT)[0]
    assert abs(fine - 71.0) < abs(coarse - 71.0)

    locked = (_window(150.0) + 4 * _window(110.0))[None, :]
    assert abs(spectral_peak_hr(locked, fs=FS, nfft=ZERO_PAD_NFFT)[0] - 110.0) < 2.0


def test_global_mean_hr():
    assert global_mean_hr(np.array([60.0, 80.0, 100.0])) == pytest.approx(80.0)


def test_previous_window_hr_shifts_and_uses_fallback():
    hr = np.array([70.0, 72.0, 75.0, 71.0])
    out = previous_window_hr(hr, fallback=65.0)
    np.testing.assert_allclose(out, [65.0, 70.0, 72.0, 75.0])


# --- metrics -----------------------------------------------------------------

def test_mape_is_relative():
    """The same 10 bpm error is 16.7% at 60 bpm and 6.2% at 160."""
    assert mape(np.array([60.0]), np.array([70.0])) == pytest.approx(100 / 6, abs=1e-6)
    assert mape(np.array([160.0]), np.array([170.0])) == pytest.approx(6.25, abs=1e-6)


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


def test_pooled_rows_reported_with_and_without_transients():
    subs = _fake_subjects()
    rows = {(r["method"], r["activity"]): r for r in per_activity_rows(subs, loso_predictions(subs))}
    assert rows[("b2", POOLED_ALL)]["n_windows"] == 30
    assert rows[("b2", POOLED_NO_TRANSIENT)]["n_windows"] == 24   # 2 transient windows each
    assert rows[("b2", POOLED_ALL)]["n_folds"] == 3


def test_both_aggregations_are_reported_and_differ_when_folds_are_unequal():
    subs = _fake_subjects()
    row = next(r for r in per_activity_rows(subs, loso_predictions(subs))
               if r["method"] == "b2" and r["activity"] == POOLED_ALL)
    assert {"mae_mean_of_folds", "mae_pooled", "mape_mean_of_folds", "mape_pooled",
            "mape_sd_across_folds", "rmse_pooled"} <= set(row)
    # A: |61-60|=1, B: |95-80|=15, C: |101-100|=1 -> fold mean 5.667, pooled the same here
    assert row["mae_mean_of_folds"] == pytest.approx(17 / 3, abs=1e-3)


def test_per_subject_rows_cover_both_scopes():
    subs = _fake_subjects()
    rows = {(r["method"], r["subject"], r["scope"]): r
            for r in per_subject_rows(subs, loso_predictions(subs))}
    assert rows[("b2", "A", "all")]["n_windows"] == 10
    assert rows[("b2", "A", "no_transient")]["n_windows"] == 8
    assert rows[("b2", "A", "all")]["mape"] > 0


def test_clip_fraction_rows_report_distribution():
    subs = _fake_subjects()
    subs[1] = SubjectArrays("B", subs[1].hr, subs[1].activity, subs[1].clipped, subs[1].b2,
                            subs[1].b2_zp, clip_fraction=np.r_[np.zeros(8), np.full(2, 0.25)])
    rows = {r["activity"]: r for r in clip_fraction_rows(subs)}
    assert "working" in rows
    assert rows["working"]["windows_flagged_pct"] > 0
    assert rows["working"]["max_clip_fraction_pct"] == pytest.approx(25.0)


def test_b2_noclip_drops_flagged_windows_only():
    subs = _fake_subjects()
    preds = loso_predictions(subs)
    rows = {(r["method"], r["subject"]): r for r in per_subject_rows(subs, preds) if r["scope"] == "all"}
    assert rows[("b2", "B")]["n_windows"] == 10
    assert rows[("b2_noclip", "B")]["n_windows"] == 8
    assert rows[("b2", "A")]["n_windows"] == rows[("b2_noclip", "A")]["n_windows"] == 10


def test_oracle_flag_is_carried_into_outputs():
    subs = _fake_subjects()
    rows = per_subject_rows(subs, loso_predictions(subs))
    assert all(r["oracle"] for r in rows if r["method"] == "b1")
    assert "b2_zp" in loso.METHOD_LABELS and loso.BASELINE_FOR_STAGE4 == "b2_zp"
    assert not any(r["oracle"] for r in rows if r["method"] != "b1")
    assert "ORACLE" in loso.METHOD_LABELS["b1"]


# --- real data ---------------------------------------------------------------

@needs_data
def test_b2_is_accurate_at_rest_and_degrades_with_motion():
    """Guards the stop conditions: b2 must work when still, and must not beat the oracle."""
    subs = build = loso.build_subject_arrays(["S1", "S2", "S3"])
    preds = loso_predictions(subs)
    by = {(r["method"], r["activity"]): r for r in per_activity_rows(subs, preds)}
    assert by[("b2", "sitting")]["mae_mean_of_folds"] < 8.0
    assert by[("b2", "stairs")]["mae_mean_of_folds"] > by[("b2", "sitting")]["mae_mean_of_folds"]
    for activity in ("sitting", "stairs", "cycling", "walking"):
        assert by[("b1", activity)]["mae_mean_of_folds"] < by[("b2", activity)]["mae_mean_of_folds"], activity
        assert by[("b2", activity)]["mae_mean_of_folds"] >= 3.0, activity
    # zero-padding must not "fix" motion lock (stop condition from the 21 Sep follow-ups)
    for activity in ("stairs", "table soccer", "walking"):
        gain = by[("b2", activity)]["mae_mean_of_folds"] - by[("b2_zp", activity)]["mae_mean_of_folds"]
        assert gain <= 3.0, (activity, gain)
    assert len(build) == 3


# --- AUROC used by the SQI validation ---------------------------------------

def test_auroc_perfect_and_chance():
    from src.eval.validate_sqi import auroc
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(score, np.array([False, False, True, True])) == pytest.approx(1.0)
    assert auroc(score, np.array([True, True, False, False])) == pytest.approx(0.0)
    assert auroc(np.ones(4), np.array([True, False, True, False])) == pytest.approx(0.5)  # all ties


def test_auroc_matches_rank_definition_with_ties():
    from src.eval.validate_sqi import auroc
    score = np.array([1.0, 2.0, 2.0, 3.0])
    pos = np.array([False, True, False, True])
    # positives hold ranks 2.5 and 4 of 4; U = (2.5 + 4) - 3 = 3.5 over 2*2 pairs
    assert auroc(score, pos) == pytest.approx(3.5 / 4)
