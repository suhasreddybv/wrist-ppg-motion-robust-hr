import numpy as np
import pytest

from src.models.viterbi import decode_fixed_lag, decode_full, top_k_peaks

F = np.arange(24, 241, 1.0)


def _spec(peaks, heights, width=2.0):
    s = np.zeros_like(F)
    for p, h in zip(peaks, heights):
        s = s + h * np.exp(-0.5 * ((F - p) / width) ** 2)
    return s


def _session(specs, k=4):
    S = np.stack(specs)
    return (*top_k_peaks(S, F, k), F[S.argmax(axis=1)])


def test_top_k_peaks_ranks_by_height_and_normalises():
    f, e = top_k_peaks(np.stack([_spec([80, 140], [1.0, 0.5])]), F, 3)
    assert abs(f[0, 0] - 80) <= 1 and abs(f[0, 1] - 140) <= 1
    assert e[0, 0] == pytest.approx(0.0, abs=1e-6)          # tallest peak is the reference
    assert e[0, 1] == pytest.approx(np.log(0.5), abs=0.02)


def test_window_with_no_peak_gets_a_flat_fallback():
    flat = np.ones_like(F)
    f, e = top_k_peaks(np.stack([flat]), F, 4)
    assert np.isfinite(f[0]).all()
    assert np.allclose(e[0], 0.0)      # no evidence: the transition term decides


def test_decoding_overrules_a_brief_louder_motion_line():
    """The failure Diagnostic A describes: the true peak survives but is not the tallest."""
    specs = ([_spec([80, 140], [1.0, 0.3])]
             + [_spec([80, 140], [0.6, 1.0])] * 3        # motion louder, cardiac still present
             + [_spec([80, 140], [1.0, 0.3])])
    f, e, greedy = _session(specs)
    assert set(np.round(greedy)) == {80.0, 140.0}          # greedy follows the motion line
    assert np.allclose(np.round(decode_full(f, e, sigma=6.0)), 80.0)


def test_decoding_still_follows_a_genuine_sustained_change():
    specs = [_spec([80], [1.0])] * 4 + [_spec([120], [1.0])] * 8
    f, e, _ = _session(specs)
    out = decode_full(f, e, sigma=12.0)
    assert abs(out[0] - 80) <= 2 and abs(out[-1] - 120) <= 2


def test_fixed_lag_is_causal_and_approaches_full_decoding_as_lag_grows():
    specs = ([_spec([80, 140], [1.0, 0.3])]
             + [_spec([80, 140], [0.6, 1.0])] * 4
             + [_spec([80, 140], [1.0, 0.3])] * 2)
    f, e, _ = _session(specs)
    full = decode_full(f, e, sigma=6.0)
    near = decode_fixed_lag(f, e, sigma=6.0, lag=6)
    short = decode_fixed_lag(f, e, sigma=6.0, lag=0)
    assert np.allclose(near, full)                                   # enough lag -> same path
    assert np.mean(np.abs(short - full)) >= np.mean(np.abs(near - full))


def test_fixed_lag_zero_cannot_see_the_future():
    """With no lag the decoder commits on evidence so far, so a later window cannot change it."""
    specs = [_spec([80, 140], [0.6, 1.0])] * 3 + [_spec([80, 140], [1.0, 0.2])] * 3
    f, e, _ = _session(specs)
    out0 = decode_fixed_lag(f, e, sigma=6.0, lag=0)
    out5 = decode_fixed_lag(f, e, sigma=6.0, lag=5)
    assert not np.allclose(out0, out5)


def test_tighter_sigma_resists_jumps_more():
    # the spike window must still contain the cardiac peak, or there is no choice to make
    specs = ([_spec([80, 150], [1.0, 0.2])] * 3 + [_spec([80, 150], [0.35, 1.0])]
             + [_spec([80, 150], [1.0, 0.2])] * 3)
    f, e, _ = _session(specs)
    # a 70 bpm excursion and back costs 2 x (70^2)/(2 sigma^2); it is only worth paying
    # when sigma is large enough for that to fall under the emission gap of ~1.05 nats
    loose = decode_full(f, e, sigma=200.0)
    tight = decode_full(f, e, sigma=3.0)
    assert abs(loose[3] - 150) <= 2          # effectively unconstrained: follows the spike
    assert abs(tight[3] - 80) <= 2           # tight transition rejects it
