"""The fixed-point simulators against exact references, with hand-checkable cases."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from src.edge.fixed_point import block_quantize, fft_fixed, quantize, sosfilt_fixed


def test_quantize_grid_rounding_and_saturation():
    q = quantize(np.array([0.0, 0.2, -1.0, 0.999, 1.5, -1.5]), bits=3, full_scale=1.0)   # grid step 0.25
    assert q.tolist() == [0.0, 0.25, -1.0, 0.75, 0.75, -1.0]


def test_block_quantize_is_per_row_and_scale_free():
    x = np.array([[1.0, -0.5, 0.25], [100.0, -50.0, 25.0]])
    q = block_quantize(x, bits=8)
    np.testing.assert_allclose(q[1], 100 * q[0], rtol=1e-12)          # one exponent per row
    assert np.max(np.abs(q - x) / np.max(np.abs(x), axis=1, keepdims=True)) < 2 ** -7


def test_fft_fixed_without_rounding_is_the_exact_fft_over_n():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(3, 100))
    got = fft_fixed(x, 256, frac_bits=None)
    np.testing.assert_allclose(got, np.fft.rfft(x, n=256, axis=1) / 256, atol=1e-12)


def test_fft_fixed_q15_error_is_at_the_q15_noise_floor():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(4, 512))
    exact = np.fft.rfft(x, n=4096, axis=1) / 4096
    err = np.abs(fft_fixed(x, 4096, 15) - exact)
    # each of the 12 stages adds up to ~1 LSB per component before the next halving shrinks it, so the
    # worst case approaches 2 LSB per component, 2*sqrt(2) in magnitude (first estimate of "< 2" was wrong)
    assert err.max() < 2 * np.sqrt(2) * 2 ** -15
    # and coarser data really is coarser
    err8 = np.abs(fft_fixed(x, 4096, 7) - exact)
    assert err8.max() > 50 * err.max()


def _snr_db(ref, got):
    return 20 * np.log10(np.std(ref) / np.std(got - ref))


def test_sosfilt_fixed_with_32_bit_state_matches_the_float_filter():
    """The CMSIS 32x64 biquad: 32-bit coefficients and state, 64-bit accumulator."""
    sos = signal.butter(4, [0.4, 4.0], btype="bandpass", fs=64, output="sos")
    t = np.arange(64 * 60) / 64
    x = 0.4 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 0.05 * t)
    xi = np.round(x * 2 ** 29)
    ref = signal.sosfilt(sos, xi)
    got = sosfilt_fixed(sos, xi, data_bits=32)
    assert _snr_db(ref[640:], got[640:]) > 100          # measured 127 dB


def test_sosfilt_fixed_with_16_bit_state_loses_most_of_its_resolution():
    """Rounding every section's output to 16 bits at a 0.4 Hz cutoff, 64 Hz sampling: poles near the
    unit circle amplify the rounding. This is why the 32-bit-state form is the one the edge path uses."""
    sos = signal.butter(4, [0.4, 4.0], btype="bandpass", fs=64, output="sos")
    t = np.arange(64 * 60) / 64
    x = 0.4 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 0.05 * t)
    xi = np.round(x * 2 ** 13)
    ref = signal.sosfilt(sos, xi)
    got = sosfilt_fixed(sos, xi, data_bits=16)
    assert 20 < _snr_db(ref[640:], got[640:]) < 40      # measured 31 dB


def test_sosfilt_fixed_saturates_rather_than_wrapping():
    sos = np.array([[4.0, 0.0, 0.0, 1.0, 0.0, 0.0]]) * [1, 1, 1, 1, 1, 1]       # a gain of 3.99..: Q2.29 limit
    sos[0, 0] = 3.9
    got = sosfilt_fixed(sos, np.array([20000, -20000, 1000]))
    assert got.tolist() == [32767.0, -32768.0, 3900.0]
