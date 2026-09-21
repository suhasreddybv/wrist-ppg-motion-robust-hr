"""The motion-collision figure: why wrist PPG needs motion handling.

Shows the 0.4-4 Hz BVP spectrogram over the estimator's own windows, with the ECG
ground-truth HR, the b2-zp estimate and the dominant wrist-ACC frequency drawn on
top. During stairs and walking the estimate follows the accelerometer line rather
than the heart-rate line.

Subject selection rule (not by eye): the subject whose stairs b2-zp MAE is closest
to the cohort median stairs b2-zp MAE. Recorded in the caption and printed here.

Deviation from the 21 Sep scope, which asked for the ACC fundamental plus its 2x
harmonic: the 2x line is replaced by the 0.5x subharmonic. Measured on S4, 0.0% of
walking and stairs windows lock to 2x, while 35% and 47% lock to 0.5x - the stride
rate, one per two steps. Drawing 2x would show a line nothing follows.

Writes figures/motion_collision.png.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, load_subject  # noqa: E402
from src.eval.loso import build_subject_arrays  # noqa: E402
from src.eval.metrics import mae  # noqa: E402
from src.features.preprocess import CARDIAC_BAND_HZ, preprocess_subject  # noqa: E402
from src.models.baselines import ZERO_PAD_NFFT, spectral_peak_hr  # noqa: E402

PANELS = [(7, "walking"), (2, "stairs"), (1, "sitting")]
MAX_WINDOWS = 120            # 4 minutes of 2 s-shifted windows, for legibility
SPEC_NFFT = 2048
HARMONICS = (0.5, 1.0)       # stride and step; 2x is drawn by no window in this data
BPM_LIM = (24, 240)
NEAR_BPM = 10.0              # "tracks" tolerance for the stop-condition check

# Okabe-Ito, colour-blind safe; line styles differ so the figure survives greyscale.
C_TRUE, C_ACC, C_EST = "#0072B2", "#E69F00", "#D55E00"


def choose_subject(subjects) -> tuple[str, float, float]:
    """The subject whose stairs b2-zp MAE is closest to the cohort median."""
    stairs = {}
    for s in subjects:
        m = s.activity == 2
        if m.any():
            stairs[s.subject_id] = mae(s.hr[m], s.b2_zp[m])
    median = float(np.median(list(stairs.values())))
    sid = min(stairs, key=lambda k: abs(stairs[k] - median))
    return sid, stairs[sid], median


def acc_dominant_bpm(acc_win: np.ndarray, fs: int) -> np.ndarray:
    """Peak frequency of the ACC magnitude spectrum per window, in bpm, in band."""
    mag = np.linalg.norm(acc_win, axis=2)
    mag = mag - mag.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(SPEC_NFFT, 1 / fs)
    taper = np.hanning(mag.shape[1])
    power = np.abs(np.fft.rfft(mag * taper, n=SPEC_NFFT, axis=1)) ** 2
    band = (freqs >= CARDIAC_BAND_HZ[0]) & (freqs <= CARDIAC_BAND_HZ[1])
    return freqs[band][power[:, band].argmax(axis=1)] * 60.0


def _block(pre, activity: int) -> slice:
    """Longest contiguous run of this activity, truncated to MAX_WINDOWS."""
    m = pre.windows.activity == activity
    idx = np.flatnonzero(m)
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(splits, key=len)
    return slice(int(run[0]), int(run[0]) + min(len(run), MAX_WINDOWS))


def main() -> None:
    subjects = build_subject_arrays()
    sid, sid_mae, median_mae = choose_subject(subjects)
    print(f"Subject rule: stairs b2-zp MAE closest to cohort median "
          f"({median_mae:.2f} bpm) -> {sid} at {sid_mae:.2f} bpm")

    pre = preprocess_subject(load_subject(sid))
    freqs = np.fft.rfftfreq(SPEC_NFFT, 1 / pre.fs) * 60.0
    band = (freqs >= BPM_LIM[0]) & (freqs <= BPM_LIM[1])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    tracking = {}
    for ax, (activity, name) in zip(axes, PANELS):
        sl = _block(pre, activity)
        w = pre.bvp_filtered[sl]
        # Hann taper before the FFT: without it, leakage from an 8 s rectangular window
        # buries the cardiac ridge in sidelobes and the spectrogram reads as noise.
        power = np.abs(np.fft.rfft(w * np.hanning(w.shape[1]), n=SPEC_NFFT, axis=1)) ** 2
        power = power[:, band] / power[:, band].max(axis=1, keepdims=True)

        t = pre.windows.start_s[sl]
        true_hr = pre.windows.hr[sl]
        est = spectral_peak_hr(w, fs=pre.fs, band=pre.band_hz, nfft=ZERO_PAD_NFFT)
        acc_hz = acc_dominant_bpm(pre.acc_64[sl], pre.fs)

        ax.pcolormesh(t, freqs[band], 10 * np.log10(power.T + 1e-12),
                      cmap="Greys", shading="nearest", vmin=-18, vmax=0)
        # When the wrist is still there is no motion peak to find, so the ACC lines are noise.
        # They are muted in that panel rather than hidden, to keep the contrast honest.
        acc_alpha = 0.3 if name == "sitting" else 1.0
        ax.plot(t, true_hr, color=C_TRUE, lw=2.2, label="ECG ground-truth HR")
        ax.plot(t, acc_hz, color=C_ACC, lw=1.8, ls="--", alpha=acc_alpha,
                label="dominant wrist-ACC frequency (step rate)")
        ax.plot(t, 0.5 * acc_hz, color=C_ACC, lw=1.3, ls=":", alpha=acc_alpha,
                label="ACC 0.5x (stride rate)")
        ax.plot(t, est, color=C_EST, ls="none", marker="x", ms=4, mew=1.2, label="b2-zp estimate")
        ax.set_ylim(*BPM_LIM)
        ax.set_ylabel("bpm")
        ax.set_title(f"{sid} — {name}, t = {t[0]:.0f}–{t[-1] + 8:.0f} s", fontsize=10)

        lines = np.column_stack([h * acc_hz for h in HARMONICS])
        near_acc = float(np.mean(np.min(np.abs(est[:, None] - lines), axis=1) <= NEAR_BPM))
        near_hr = float(np.mean(np.abs(est - true_hr) <= NEAR_BPM))
        tracking[name] = (near_acc, near_hr)
        ax.text(0.995, 0.04, f"estimate within {NEAR_BPM:.0f} bpm of an ACC line: {near_acc:.0%} | "
                             f"of true HR: {near_hr:.0%}",
                transform=ax.transAxes, ha="right", fontsize=8,
                bbox=dict(fc="white", ec="0.7", alpha=0.85))

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Wrist PPG spectrum under motion: the estimate follows the accelerometer, not the heart",
                 fontsize=12)
    fig.tight_layout()
    out = REPO_ROOT / "figures" / "motion_collision.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(REPO_ROOT)}")

    print("\nTracking check (fraction of windows within 10 bpm):")
    for name, (a, h) in tracking.items():
        print(f"  {name:<9} ACC line {a:6.1%}   true HR {h:6.1%}")
    a, h = tracking["stairs"]
    if a <= h:
        print("\nSTOP: on stairs the b2-zp estimate does not track the ACC line more often than "
              "the true HR. This contradicts the structured-error finding; do not publish the figure.")
    else:
        print("\nStop condition: not triggered (stairs estimate tracks the ACC line).")


if __name__ == "__main__":
    main()
