import numpy as np
import pytest

from src.data import loader
from src.features.preprocess import (
    CARDIAC_BAND_HZ,
    bandpass,
    composite_with_template,
    preprocess_subject,
    quality_index,
    resample_acc,
    zscore_windows,
)
from tests.test_loader import needs_data
from tests.test_windows import _fake_record

FS = 64


def _sine(f, n=512, fs=FS, phase=0.0):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f * t + phase)


# --- band-pass ---------------------------------------------------------------

@pytest.mark.parametrize("f, lo, hi", [
    (0.05, 0.0, 0.05),   # far below the band: rejected
    (0.2, 0.0, 0.5),     # below 0.4 Hz
    (1.2, 0.95, 1.05),   # mid-band (72 bpm): passed
    (3.0, 0.9, 1.05),    # 180 bpm: passed, still in band
    (10.0, 0.0, 0.05),   # above the band: rejected
])
def test_bandpass_gain(f, lo, hi):
    x = _sine(f, n=64 * 60)
    y = bandpass(x, FS)
    gain = y[FS * 5:-FS * 5].std() / x.std()
    assert lo <= gain <= hi, f"{f} Hz gain {gain:.3f}"


def test_bandpass_is_zero_phase():
    x = _sine(1.2, n=64 * 30)
    y = bandpass(x, FS)
    core = slice(FS * 5, -FS * 5)
    lags = np.arange(-FS, FS + 1)
    xc = [np.corrcoef(x[core], np.roll(y, l)[core])[0, 1] for l in lags]
    assert lags[int(np.argmax(xc))] == 0   # no group delay


@pytest.mark.parametrize("f, bpm, wide_min, narrow_max", [
    (2.5, 150, 0.98, 0.55),   # the narrow band's own corner: already down to 0.50
    (3.0, 180, 0.95, 0.10),
    (3.5, 210, 0.75, 0.02),
])
def test_wide_band_keeps_high_heart_rates_that_a_narrow_band_discards(f, bpm, wide_min, narrow_max):
    """0.8-2.5 Hz would discard the stairs/cycling regime; the wide band keeps it.

    Gain tapers towards the 4 Hz corner because filtfilt applies the response twice:
    1.00 to 150 bpm, 0.96 at 180, 0.80 at 210, 0.50 at 240. Only 0.06% of PPG-DaLiA
    labels exceed 180 bpm (max 187), so the taper costs nothing here.
    """
    x = _sine(f, n=64 * 60)
    core = slice(FS * 5, -FS * 5)
    wide = bandpass(x, FS, CARDIAC_BAND_HZ)[core].std() / x.std()
    narrow = bandpass(x, FS, (0.8, 2.5))[core].std() / x.std()
    assert wide >= wide_min
    assert narrow <= narrow_max


def test_bandpass_rejects_impossible_band():
    with pytest.raises(ValueError):
        bandpass(_sine(1.0), FS, (0.4, 40.0))   # above Nyquist


# --- z-scoring and resampling ------------------------------------------------

def test_zscore_per_window():
    w = np.array([_sine(1.2) * 3 + 10, _sine(2.0) * 0.1 - 5])
    z = zscore_windows(w)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-12)


def test_zscore_flat_window_does_not_explode():
    z = zscore_windows(np.zeros((1, 512)))
    assert np.all(z == 0)


def test_resample_acc_doubles_rate_and_preserves_gravity():
    rng = np.random.default_rng(0)
    acc = rng.normal(0, 0.05, (320, 3))
    acc[:, 2] += 1.0
    up = resample_acc(acc, 32, 64)
    assert up.shape == (640, 3)
    assert abs(np.median(np.linalg.norm(up, axis=1)) - 1.0) < 0.02


def test_resample_acc_rejects_non_integer_ratio():
    with pytest.raises(ValueError):
        resample_acc(np.zeros((320, 3)), 32, 100)


# --- signal quality ----------------------------------------------------------

def _pulse_train(hr_bpm=72, n=512, fs=FS, noise=0.0, seed=0):
    """A crude but periodic pulse-like waveform at a given heart rate."""
    t = np.arange(n) / fs
    f = hr_bpm / 60
    x = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, n)
    return x


def test_sqi_high_for_clean_pulse_low_for_noise():
    clean = _pulse_train()[None, :]
    noise = np.random.default_rng(1).normal(0, 1, (1, 512))
    q_clean = quality_index(clean, clean, FS)
    q_noise = quality_index(noise, noise, FS)
    assert q_clean.template_corr[0] > 0.9
    assert q_clean.spectral_concentration[0] > 0.5
    assert q_clean.combined[0] > q_noise.combined[0]
    # The composite is spectral concentration alone since the template term was dropped,
    # so noise no longer gets multiplied down: it scores ~0.39 rather than ~0.1. What the
    # SQI must preserve is the ordering and a clear margin, not a fixed absolute value.
    assert q_noise.combined[0] < 0.5
    assert q_clean.combined[0] - q_noise.combined[0] > 0.3


def test_out_of_band_ratio_flags_high_frequency_content():
    in_band = _sine(1.2)[None, :]
    out_band = _sine(20.0)[None, :]
    q_in = quality_index(in_band, in_band, FS)
    q_out = quality_index(out_band, out_band, FS)
    assert q_in.out_of_band_ratio[0] < 0.05
    assert q_out.out_of_band_ratio[0] > 0.95


def test_sqi_components_are_bounded():
    rng = np.random.default_rng(2)
    w = np.stack([_pulse_train(hr_bpm=h, noise=0.3, seed=h) for h in (50, 80, 120, 180)])
    q = quality_index(w, w + rng.normal(0, 0.1, w.shape), FS)
    for arr in (q.template_corr, q.spectral_concentration, q.out_of_band_ratio, q.combined):
        assert np.all(arr >= -1) and np.all(arr <= 1)


# --- end to end --------------------------------------------------------------

def test_preprocess_synthetic_subject():
    pre = preprocess_subject(_fake_record())
    assert len(pre) == 7
    assert pre.bvp_filtered.shape == (7, 512)
    assert pre.acc_64.shape == (7, 512, 3)
    np.testing.assert_allclose(pre.bvp_filtered.mean(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(pre.bvp_filtered.std(axis=1), 1, atol=1e-10)
    assert pre.band_hz == CARDIAC_BAND_HZ


@needs_data
def test_preprocess_s1_shapes_and_alignment():
    rec = loader.load_subject("S1")
    pre = preprocess_subject(rec)
    assert len(pre) == len(rec.label) == 4603
    assert pre.bvp_filtered.shape == (4603, 512)
    assert pre.acc_64.shape == (4603, 512, 3)
    np.testing.assert_array_equal(pre.windows.hr, rec.label)
    # ACC resampled to 64 Hz still reads ~1 g
    assert abs(np.median(np.linalg.norm(pre.acc_64.reshape(-1, 3), axis=1)) - 1.0) < 0.05


@needs_data
def test_sqi_is_higher_at_rest_than_in_motion():
    """Sitting should score better than walking and cycling. If not, the SQI is not measuring quality."""
    pre = preprocess_subject(loader.load_subject("S1"))
    act = pre.windows.activity
    sitting = pre.sqi.combined[act == 1].mean()
    walking = pre.sqi.combined[act == 7].mean()
    cycling = pre.sqi.combined[act == 4].mean()
    assert sitting > walking
    assert sitting > cycling


# --- SQI composite after the template term was dropped -----------------------

def test_composite_is_spectral_concentration_only():
    """The template term was dropped on 22 Sep 2026; see results/sqi_validation.csv."""
    w = np.stack([_pulse_train(hr_bpm=h, noise=0.2, seed=h) for h in (60, 90, 140)])
    q = quality_index(w, w, FS)
    np.testing.assert_allclose(q.combined, q.spectral_concentration)
    assert np.any(q.template_corr > 0)          # still computed and reported


def test_composite_with_template_is_kept_for_reproducing_the_decision():
    w = _pulse_train()[None, :]
    q = quality_index(w, w, FS)
    old = composite_with_template(q.template_corr, q.spectral_concentration)
    assert old[0] <= q.spectral_concentration[0] + 1e-12
