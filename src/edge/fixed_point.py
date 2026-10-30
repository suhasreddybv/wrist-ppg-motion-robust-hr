"""Fixed-point arithmetic, simulated in numpy, for the frozen pipeline's signal path.

What is modelled, and what is not (docs/edge-budget.md says the same):

- `quantize`: round to nearest on a signed grid of `bits` bits over [-full_scale, full_scale),
  saturating outside. The simulation keeps the dequantised value, so every downstream step sees
  exactly the values a fixed-point implementation would hold.
- `block_quantize`: block floating point - one shared exponent per window (scale = the window's
  maximum magnitude), then `bits`-bit mantissas. This is how a fixed-point spectrum or window is
  stored without wasting range; every later step of the pipeline is invariant to a per-window
  scale (argmax, relative prominence, multiplicative masks), so only the rounding matters.
- `fft_fixed`: an iterative radix-2 decimation-in-time FFT that halves after every butterfly stage
  and rounds the real and imaginary parts to `frac_bits` fractional bits after every stage and
  every twiddle multiply, with twiddles rounded the same way. That is the scaling scheme of
  CMSIS-DSP's Q15 transforms (output = FFT / N). It is a model of that arithmetic, not a
  bit-exact copy of `arm_rfft_q15`, which computes a real FFT through an N/2 complex transform
  and a split step. With `frac_bits=None` it returns exactly `np.fft.fft(x, n) / n`.
- `sosfilt_fixed`: a cascade of direct-form-I biquads with coefficients rounded to Q2.29 (the
  CMSIS-DSP recommendation for low-cutoff filters is 32-bit coefficients with a 64-bit
  accumulator, `arm_biquad_cas_df1_32x64_q31`), integer state and a 16-bit saturating output per
  section. Python integers make the accumulator exact.
"""
from __future__ import annotations

import numpy as np


def quantize(x: np.ndarray, bits: int, full_scale: float) -> np.ndarray:
    """Signed `bits`-bit grid over [-full_scale, full_scale), round to nearest, saturate."""
    levels = 2 ** (bits - 1)
    q = np.clip(np.round(np.asarray(x, float) / full_scale * levels), -levels, levels - 1)
    return q * (full_scale / levels)


def block_quantize(x: np.ndarray, bits: int, axis: int = -1) -> np.ndarray:
    """One exponent per row (scale = max |x| along `axis`), `bits`-bit mantissas."""
    x = np.asarray(x, float)
    peak = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(peak > 0, peak, 1.0) * (1 + 2.0 ** -(bits - 1))   # keep the peak inside the grid
    return quantize(x / scale, bits, 1.0) * scale


def _round(z: np.ndarray, frac_bits: int | None) -> np.ndarray:
    if frac_bits is None:
        return z
    s = 2.0 ** frac_bits
    return (np.round(z.real * s) + 1j * np.round(z.imag * s)) / s


def _bitrev(n: int) -> np.ndarray:
    bits = int(np.log2(n))
    idx = np.arange(n)
    rev = np.zeros(n, dtype=int)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def fft_fixed(x: np.ndarray, nfft: int, frac_bits: int | None = 15) -> np.ndarray:
    """FFT / nfft of each row of real `x` (values in [-1, 1)), zero-padded to `nfft`, with
    per-stage halving and rounding. Returns the first nfft // 2 + 1 bins (the real-input half)."""
    x = np.atleast_2d(np.asarray(x, float))
    if nfft & (nfft - 1):
        raise ValueError("nfft must be a power of two")
    a = np.zeros((x.shape[0], nfft), dtype=complex)
    a[:, :x.shape[1]] = x
    a = _round(a, frac_bits)[:, _bitrev(nfft)]
    m = 1
    while m < nfft:
        half, m = m, 2 * m
        tw = _round(np.exp(-2j * np.pi * np.arange(half) / m), frac_bits)
        a = a.reshape(x.shape[0], nfft // m, m)
        top, bot = a[:, :, :half], _round(a[:, :, half:] * tw, frac_bits)
        a = np.concatenate([_round((top + bot) / 2, frac_bits), _round((top - bot) / 2, frac_bits)], axis=2)
    return a.reshape(x.shape[0], nfft)[:, : nfft // 2 + 1]


COEF_FRAC_BITS = 29          # Q2.29: 32-bit coefficients covering |c| < 4
DATA_BITS = 16


def sos_to_fixed(sos: np.ndarray, frac_bits: int = COEF_FRAC_BITS) -> list[tuple[int, ...]]:
    """Integer coefficients (b0, b1, b2, a1, a2) per section, a0 normalised to 1."""
    out = []
    for b0, b1, b2, a0, a1, a2 in np.asarray(sos, float):
        if abs(a0 - 1) > 1e-12:
            raise ValueError("a0 must be 1")
        out.append(tuple(int(round(c * 2 ** frac_bits)) for c in (b0, b1, b2, a1, a2)))
    return out


def sosfilt_fixed(sos: np.ndarray, x_int: np.ndarray, frac_bits: int = COEF_FRAC_BITS,
                  data_bits: int = DATA_BITS) -> np.ndarray:
    """Causal biquad cascade on integer samples; each section's output rounded to an integer and
    saturated to `data_bits`. Starts from rest (zero state)."""
    lo, hi = -(2 ** (data_bits - 1)), 2 ** (data_bits - 1) - 1
    rnd = 1 << (frac_bits - 1)
    y = [int(v) for v in np.asarray(x_int)]
    for b0, b1, b2, a1, a2 in sos_to_fixed(sos, frac_bits):
        x1 = x2 = y1 = y2 = 0
        out = [0] * len(y)
        for i, xn in enumerate(y):
            acc = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            yn = (acc + rnd) >> frac_bits
            yn = hi if yn > hi else lo if yn < lo else yn
            out[i] = yn
            x2, x1, y2, y1 = x1, xn, y1, yn
        y = out
    return np.asarray(y, dtype=float)
