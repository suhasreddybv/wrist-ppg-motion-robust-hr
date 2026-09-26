"""Is 'accelerometer lock' a real attribution, or arithmetic?

An estimate lands within +-T bpm of one of three ACC lines (0.5x, 1x, 2x the
dominant motion frequency) by chance at a rate that grows with T. The null here
pairs each window's PPG estimate with the ACC spectrum of a *different* window of
the same activity, which destroys the pairing while keeping both marginal
distributions, and repeats it over many shuffles.

Windows are shuffled within an activity across all subjects, so the null also
absorbs between-subject differences in heart rate and cadence.

Writes results/acc_lock_permutation.csv.
"""
from __future__ import annotations

import csv

import numpy as np

from src.data.loader import REPO_ROOT, SUBJECT_IDS, load_subject
from src.eval.error_taxonomy import ERROR_BPM
from src.eval.loso import ACTIVITY_NAMES
from src.eval.plot_motion_collision import acc_dominant_bpm
from src.features.preprocess import preprocess_subject
from src.models.baselines import ZERO_PAD_NFFT, spectral_peak_hr

TOLERANCES = (3.0, 8.0, 20.0)
N_SHUFFLES = 20
HARMONICS = (0.5, 1.0, 2.0)
SEED = 42


def lock_share(est: np.ndarray, acc: np.ndarray, tol: float) -> float:
    lines = np.column_stack([h * acc for h in HARMONICS])
    return float(np.mean(np.min(np.abs(est[:, None] - lines), axis=1) <= tol))


def main() -> None:
    rng = np.random.default_rng(SEED)
    per_activity: dict[int, tuple[list, list]] = {a: ([], []) for a in ACTIVITY_NAMES}
    for sid in SUBJECT_IDS:
        pre = preprocess_subject(load_subject(sid))
        est = spectral_peak_hr(pre.bvp_filtered, fs=pre.fs, band=pre.band_hz, nfft=ZERO_PAD_NFFT)
        acc = acc_dominant_bpm(pre.acc_64, pre.fs)
        big = np.abs(est - pre.windows.hr) > ERROR_BPM
        for a in ACTIVITY_NAMES:
            m = (pre.windows.activity == a) & big
            if m.any():
                per_activity[a][0].append(est[m])
                per_activity[a][1].append(acc[m])

    rows = []
    print(f"Observed vs permuted 'accelerometer lock' share, error windows only "
          f"({N_SHUFFLES} shuffles)\n")
    print(f"{'activity':<15}{'n':>7}" + "".join(f"{'+-' + str(int(t)) + ' bpm':>26}" for t in TOLERANCES))
    all_est, all_acc = [], []
    for a, name in ACTIVITY_NAMES.items():
        est = np.concatenate(per_activity[a][0])
        acc = np.concatenate(per_activity[a][1])
        all_est.append(est)
        all_acc.append(acc)
        line = f"{name:<15}{len(est):>7}"
        for tol in TOLERANCES:
            obs = lock_share(est, acc, tol)
            null = np.array([lock_share(est, rng.permutation(acc), tol) for _ in range(N_SHUFFLES)])
            excess = obs - null.mean()
            line += f"{obs:>10.1%}{null.mean():>8.1%}{excess:>+8.1%}"
            rows.append(dict(activity=name, tolerance_bpm=tol, n_error_windows=len(est),
                             observed=round(obs, 4), permuted_mean=round(float(null.mean()), 4),
                             permuted_sd=round(float(null.std()), 4),
                             excess=round(float(excess), 4),
                             exceeds_null=bool(obs > null.mean() + 2 * null.std())))
        print(line)

    est, acc = np.concatenate(all_est), np.concatenate(all_acc)
    line = f"{'ALL':<15}{len(est):>7}"
    for tol in TOLERANCES:
        obs = lock_share(est, acc, tol)
        null = np.array([lock_share(est, rng.permutation(acc), tol) for _ in range(N_SHUFFLES)])
        line += f"{obs:>10.1%}{null.mean():>8.1%}{obs - null.mean():>+8.1%}"
        rows.append(dict(activity="ALL (excl. transient)", tolerance_bpm=tol, n_error_windows=len(est),
                         observed=round(obs, 4), permuted_mean=round(float(null.mean()), 4),
                         permuted_sd=round(float(null.std()), 4),
                         excess=round(float(obs - null.mean()), 4),
                         exceeds_null=bool(obs > null.mean() + 2 * null.std())))
    print(line)
    print("\ncolumns per tolerance: observed | permuted | excess")

    with open(REPO_ROOT / "results" / "acc_lock_permutation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote results/acc_lock_permutation.csv")

    for tol in TOLERANCES:
        r = next(x for x in rows if x["activity"] == "ALL (excl. transient)" and x["tolerance_bpm"] == tol)
        verdict = "survives the null" if r["exceeds_null"] else "DOES NOT EXCEED the null - do not use"
        print(f"  +-{int(tol):>2} bpm: observed {r['observed']:.1%} vs permuted {r['permuted_mean']:.1%} "
              f"(sd {r['permuted_sd']:.1%}) -> {verdict}")


if __name__ == "__main__":
    main()
