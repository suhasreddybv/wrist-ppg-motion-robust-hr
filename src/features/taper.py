"""One switch for the analysis window, so a taper cannot be half-applied.

A Hann taper reduces spectral leakage, which is the mechanism D-029 named as the
cost of taking plain FFTs: leakage from a strong motion line spreads past any
notch a mask can place. Tapering is therefore a candidate improvement - but it
changes b2-zp itself, so it invalidates every baseline and ablation number in the
repository at once.

It is controlled by one environment variable, `PPG_TAPER=1`, and every FFT of an
8 s window in this package goes through `analysis_window()`. A taper applied to
some paths and not others would be worse than none: the baselines and the methods
would no longer be measured on the same spectrum.

Accelerometer spectra are tapered unconditionally. They only locate motion lines
and are never compared against a baseline, so leakage there is a pure cost.
"""
from __future__ import annotations

import functools
import os

import numpy as np


def taper_enabled() -> bool:
    return os.environ.get("PPG_TAPER", "0") == "1"


@functools.lru_cache(maxsize=8)
def analysis_window(n: int) -> np.ndarray:
    """Hann when the taper is on, a rectangular window (all ones) when it is off."""
    return np.hanning(n) if taper_enabled() else np.ones(n)


def label() -> str:
    return "hann" if taper_enabled() else "rectangular"
