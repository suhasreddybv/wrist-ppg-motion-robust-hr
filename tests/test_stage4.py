import numpy as np
import pytest

from src.models.adaptive import AdaptiveConfig, cancel, ls_cancel, nlms_cancel
from src.models.masking import (
    MaskConfig,
    apply_masking,
    band_spectra,
    mask_gains,
    motion_spectrum,
)
from src.models.tracker import TrackerConfig, track

FS = 64
N = 512


def _sine(f_bpm, n=N, fs=FS, phase=0.0, amp=1.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * (f_bpm / 60) * t + phase)


def _acc(f_bpm, n=N, amp=0.5):
    """Three-axis accelerometer with a cadence line on each axis."""
    a = np.zeros((n, 3))
    for k in range(3):
        a[:, k] = amp * _sine(f_bpm, n, phase=0.4 * k)
    a[:, 2] += 1.0          # gravity
    return a


# --- motion spectrum ---------------------------------------------------------

def test_motion_spectrum_finds_the_cadence_line():
    acc = _acc(110.0)[None, :, :]
    freqs, power = motion_spectrum(acc, FS)
    peak_bpm = freqs[power[0].argmax()] * 60
    assert abs(peak_bpm - 110.0) < 3.0
    assert power.max() == pytest.approx(1.0)      # normalised per window


def test_motion_spectrum_uses_all_axes_not_just_one():
    """A line present on a single axis must still appear in the combined spectrum."""
    acc = np.zeros((1, N, 3))
    acc[0, :, 0] = _sine(140.0)
    acc[0, :, 2] += 1.0
    freqs, power = motion_spectrum(acc, FS)
    assert abs(freqs[power[0].argmax()] * 60 - 140.0) < 3.0


# --- masking -----------------------------------------------------------------

def test_masking_recovers_hr_when_motion_dominates():
    """The failure b2 exhibits: a cadence peak larger than the cardiac peak."""
    hr, cadence = 150.0, 110.0
    bvp = (_sine(hr) + 4 * _sine(cadence))[None, :]
    acc = _acc(cadence)[None, :, :]
    unmasked = apply_masking(bvp, acc, FS, MaskConfig(depth=1.0))   # depth 1 = no attenuation
    # depth is a POWER gain: a 4x-amplitude cadence carries 16x the cardiac power, so a
    # depth of 0.1 leaves it on top (16 x 0.1 = 1.6). The notch has to be deeper than the
    # motion-to-cardiac power ratio, which is why the ablation grid reaches 0.02.
    masked = apply_masking(bvp, acc, FS, MaskConfig(n_peaks=1, depth=0.02, width_factor=2.0))
    assert abs(unmasked.hr_bpm[0] - cadence) < 3.0
    assert abs(masked.hr_bpm[0] - hr) < 4.0


def test_masking_is_skipped_when_the_wrist_is_still():
    """No motion means no lines to mask; the estimate must be untouched."""
    bvp = _sine(72.0)[None, :]
    acc = np.zeros((1, N, 3))
    acc[0, :, 2] = 1.0 + 0.0005 * np.random.default_rng(0).normal(size=N)
    r = apply_masking(bvp, acc, FS, MaskConfig())
    assert r.n_lines[0] == 0
    assert r.masked_energy[0] == 0.0
    assert abs(r.hr_bpm[0] - 72.0) < 1.5


def test_masking_down_weights_rather_than_zeroes():
    """A heart rate that coincides with the cadence must remain findable."""
    hr = 110.0
    bvp = _sine(hr)[None, :]
    acc = _acc(hr)[None, :, :]
    r = apply_masking(bvp, acc, FS, MaskConfig(n_peaks=1, depth=0.1))
    assert abs(r.hr_bpm[0] - hr) < 4.0      # still the largest bin after attenuation
    assert 0.0 < r.masked_energy[0] < 1.0


def test_mask_gains_are_bounded_and_reusable():
    f_hz, _ = band_spectra(_sine(80.0)[None, :], FS)
    gains, n_lines = mask_gains(_acc(110.0)[None, :, :], FS, f_hz, MaskConfig())
    assert gains.shape == (1, len(f_hz))
    assert np.all(gains > 0) and np.all(gains <= 1.0)
    assert n_lines[0] > 0


def test_deeper_and_wider_masking_removes_more_energy():
    bvp = (_sine(150.0) + 4 * _sine(110.0))[None, :]
    acc = _acc(110.0)[None, :, :]
    gentle = apply_masking(bvp, acc, FS, MaskConfig(n_peaks=1, depth=0.5, width_factor=0.5))
    heavy = apply_masking(bvp, acc, FS, MaskConfig(n_peaks=3, depth=0.05, width_factor=2.0))
    assert heavy.masked_energy[0] > gentle.masked_energy[0]


# --- adaptive cancellation ---------------------------------------------------

def test_nlms_removes_an_acc_correlated_component():
    rng = np.random.default_rng(0)
    hr = _sine(75.0)
    motion = rng.normal(size=N)
    acc = np.zeros((1, N, 3))
    acc[0, :, 0] = motion
    acc[0, :, 2] = 1.0
    bvp = (hr + 3 * motion)[None, :]
    res = nlms_cancel(bvp, acc, AdaptiveConfig(order=4, mu=0.5))
    # the residual should look more like the cardiac component than the input did
    assert np.corrcoef(res[0, N // 2:], hr[N // 2:])[0, 1] > np.corrcoef(bvp[0], hr)[0, 1]


def test_ls_cancel_removes_a_linear_reference_exactly():
    rng = np.random.default_rng(1)
    hr = _sine(75.0)
    motion = rng.normal(size=N)
    acc = np.zeros((1, N, 3))
    acc[0, :, 0] = motion
    bvp = (hr + 2.5 * motion)[None, :]
    res = ls_cancel(bvp, acc, AdaptiveConfig(order=1, ridge=1e-9))
    assert np.corrcoef(res[0], hr)[0, 1] > 0.99


def test_cancel_dispatches_and_rejects_unknown_methods():
    bvp = _sine(75.0)[None, :]
    acc = _acc(100.0)[None, :, :]
    assert cancel(bvp, acc, AdaptiveConfig(method="ls")).shape == bvp.shape
    with pytest.raises(ValueError, match="unknown method"):
        cancel(bvp, acc, AdaptiveConfig(method="kalman"))


def test_cancellation_does_not_leak_across_windows():
    """Each window is filtered from zero weights: window order must not matter."""
    rng = np.random.default_rng(2)
    bvp = rng.normal(size=(4, N))
    acc = rng.normal(size=(4, N, 3))
    cfg = AdaptiveConfig(order=4, mu=0.2)
    forward = nlms_cancel(bvp, acc, cfg)
    reverse = nlms_cancel(bvp[::-1], acc[::-1], cfg)[::-1]
    np.testing.assert_allclose(forward, reverse, atol=1e-12)


# --- tracker -----------------------------------------------------------------

def _spectra_from(peaks_bpm, f_bpm, width=2.0):
    return np.stack([np.exp(-0.5 * ((f_bpm - p) / width) ** 2) for p in peaks_bpm])


def test_tracker_rejects_a_single_window_jump():
    f = np.arange(24, 241, 1.0)
    truth = [80, 81, 82, 83, 84]
    spec = _spectra_from(truth, f)
    spec[3] += 3 * np.exp(-0.5 * ((f - 150.0) / 2.0) ** 2)   # one bad window
    out = track(spec, f, TrackerConfig(half_width_bpm=12, jump_bpm=12, reset_after=3))
    assert abs(out[3] - 83) <= 12          # constrained, did not follow the spike
    assert out[3] != 150.0


def test_tracker_resets_after_persistent_disagreement():
    """A genuine rate change must be followed once it persists."""
    f = np.arange(24, 241, 1.0)
    seq = [80, 80, 80, 160, 160, 160, 160, 160]
    out = track(_spectra_from(seq, f), f, TrackerConfig(half_width_bpm=12, jump_bpm=12, reset_after=2))
    assert abs(out[-1] - 160) <= 2         # re-seeded onto the new rate
    assert abs(out[0] - 80) <= 2


def test_tracker_is_a_no_op_on_a_stable_track():
    f = np.arange(24, 241, 1.0)
    seq = [75, 76, 77, 76, 75]
    out = track(_spectra_from(seq, f), f, TrackerConfig())
    np.testing.assert_allclose(out, seq, atol=1.0)


def test_tracker_first_window_is_unconstrained():
    f = np.arange(24, 241, 1.0)
    out = track(_spectra_from([180, 180], f), f, TrackerConfig())
    assert abs(out[0] - 180) <= 1.0


# --- Week 3: search bound and sustained-wrongness reset ----------------------

def test_bound_removes_the_low_region_from_selection():
    f = np.arange(24, 241, 1.0)
    spec = _spectra_from([30.0], f)                  # the only peak is below the bound
    assert track(spec, f, TrackerConfig()) == pytest.approx(30.0, abs=1.0)
    bounded = track(spec, f, TrackerConfig(), bound_bpm=42.0)
    assert bounded[0] >= 42.0


def test_bound_with_no_surviving_candidates_still_returns_an_estimate():
    f = np.arange(24, 241, 1.0)
    spec = _spectra_from([30.0, 31.0, 30.5], f)
    out = track(spec, f, TrackerConfig(mode="nearest_peak"), bound_bpm=42.0)
    assert np.all(out >= 42.0) and np.all(np.isfinite(out))


def test_rank_reset_fires_on_a_smoothly_tracked_wrong_estimate():
    """The failure the jump test cannot see: the track stays put while the spectrum moves."""
    f = np.arange(24, 241, 1.0)
    # window 0 seeds at 80; afterwards the dominant peak is 150 and 80 is a minor peak
    spec = np.stack([_spectra_from([80.0], f)[0]]
                    + [_spectra_from([150.0], f)[0] + 0.25 * _spectra_from([80.0], f)[0]
                       for _ in range(6)])
    smooth = track(spec, f, TrackerConfig(jump_bpm=100, reset_after=99))
    with_rank = track(spec, f, TrackerConfig(jump_bpm=100, reset_after=99,
                                             sustained_rule="rank", rank_max=1, sustained_after=2))
    assert abs(smooth[-1] - 80.0) <= 2.0          # jump test alone never lets go
    assert abs(with_rank[-1] - 150.0) <= 2.0      # rank test re-seeds onto the real peak


def test_ratio_reset_fires_on_a_collapsing_peak():
    f = np.arange(24, 241, 1.0)
    spec = np.stack([_spectra_from([80.0], f)[0]]
                    + [_spectra_from([150.0], f)[0] + 0.1 * _spectra_from([80.0], f)[0]
                       for _ in range(6)])
    out = track(spec, f, TrackerConfig(jump_bpm=100, reset_after=99,
                                       sustained_rule="ratio", ratio_min=0.5, sustained_after=2))
    assert abs(out[-1] - 150.0) <= 2.0


def test_sustained_reset_does_not_fire_on_a_healthy_track():
    f = np.arange(24, 241, 1.0)
    spec = _spectra_from([75, 76, 77, 76, 75], f)
    out, resets = track(spec, f, TrackerConfig(sustained_rule="rank", rank_max=2, sustained_after=2),
                        return_resets=True)
    assert not resets.any()
    np.testing.assert_allclose(out, [75, 76, 77, 76, 75], atol=1.0)
