"""Two diagnostics that decide what the Stage 4 result means.

A. Does the cardiac peak survive masking? For every masked window still in error,
   does the masked spectrum contain any peak within 3 bpm of the true heart rate?
   A high share means masking preserves the information and the residual problem is
   selection - which is what the superadditive combined gain implies. A low share
   means motion destroyed the cardiac component, and no single-window spectral
   method can recover it: a limitation of the signal, not of the method.
   The share is compared against a chance rate built the same way as the
   accelerometer-lock null (D-026): pair each window with another window's true HR
   from the same activity.

B. Error persistence. Mean error says nothing about how long an error lasts. A
   tracker that reduces MAE while lengthening individual errors is worse for a
   wearable, not better. Run lengths are measured on consecutive windows in time,
   within a contiguous block of one activity, and the tracker's resets are counted
   per activity.

Configurations are the ones each fold actually selected, read back from
results/stage4_per_subject.csv.

Writes results/surviving_peak.csv, results/error_persistence.csv and
figures/error_persistence.png.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, SUBJECT_IDS, load_subject  # noqa: E402
from src.eval.loso import ACTIVITY_NAMES  # noqa: E402
from src.eval.stage4 import config_from_label  # noqa: E402
from src.features.preprocess import preprocess_subject  # noqa: E402
from src.models.masking import band_spectra, mask_gains  # noqa: E402
from src.models.tracker import peak_candidates, track  # noqa: E402

ERROR_BPM = 10.0
PEAK_TOL_BPM = 3.0
N_SHUFFLES = 20
LONG_RUN_WINDOWS = 15          # 30 s at a 2 s shift
SEED = 42
PROMINENCE = 0.05


def _selected() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    with open(REPO_ROOT / "results" / "stage4_per_subject.csv") as f:
        for r in csv.DictReader(f):
            out[r["method"]][r["subject"]] = r["config"]
    return out


def _runs(flags: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs."""
    if not flags.any():
        return []
    d = np.diff(np.r_[0, flags.view(np.int8), 0])
    return list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    sel = _selected()
    subs = []
    for sid in SUBJECT_IDS:
        pre = preprocess_subject(load_subject(sid))
        f_hz, spec = band_spectra(pre.bvp_filtered, pre.fs)
        f_bpm = f_hz * 60.0
        hr, act = pre.windows.hr, pre.windows.activity

        mcfg = config_from_label(sel["+mask"][sid])
        gains, _ = mask_gains(pre.acc_64, pre.fs, f_hz, mcfg)
        masked = spec * gains
        mask_est = f_bpm[masked.argmax(axis=1)]

        mt_mask_label, mt_track_label = sel["+mask+tracker"][sid].split("|")
        g2, _ = mask_gains(pre.acc_64, pre.fs, f_hz, config_from_label(mt_mask_label))
        masked2 = spec * g2
        cand2 = peak_candidates(masked2, f_bpm, PROMINENCE)
        mt_est, resets = track(masked2, f_bpm, config_from_label(mt_track_label), cand2, return_resets=True)

        subs.append(dict(sid=sid, hr=hr, activity=act, f_bpm=f_bpm, masked=masked,
                         base=f_bpm[spec.argmax(axis=1)], mask_est=mask_est,
                         mt_est=mt_est, resets=resets,
                         cand=peak_candidates(masked, f_bpm, PROMINENCE),
                         heights=[masked[i][np.searchsorted(f_bpm, c)] if len(c) else np.empty(0)
                                  for i, c in enumerate(peak_candidates(masked, f_bpm, PROMINENCE))]))
        print(f"  {sid} done")

    # ---------- A: does a cardiac peak survive? ----------
    rows_a = []
    print(f"\nA. masked windows still in error: is there a peak within {PEAK_TOL_BPM:.0f} bpm of the true HR?")
    print(f"{'activity':<15}{'err windows':>12}{'survives':>10}{'chance':>9}{'excess':>9}"
          f"{'rank 1':>9}{'rank 2':>8}{'rank 3':>8}{'rank >3':>9}")
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        surv, ranks, chance = [], [], []
        for s in subs:
            m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
            m = m & (np.abs(s["mask_est"] - s["hr"]) > ERROR_BPM)
            idx = np.flatnonzero(m)
            for i in idx:
                cand, h = s["cand"][i], s["heights"][i]
                if not len(cand):
                    surv.append(False)
                    continue
                hit = np.abs(cand - s["hr"][i]) <= PEAK_TOL_BPM
                surv.append(bool(hit.any()))
                if hit.any():
                    order = np.argsort(h)[::-1]
                    ranks.append(int(np.flatnonzero(np.isin(order, np.flatnonzero(hit)))[0]) + 1)
            if len(idx):
                pool = s["hr"][m]
                for _ in range(N_SHUFFLES):
                    shuffled = rng.permutation(pool)
                    chance.append(np.mean([
                        bool(len(s["cand"][i]) and (np.abs(s["cand"][i] - shuffled[k]) <= PEAK_TOL_BPM).any())
                        for k, i in enumerate(idx)]))
        if not surv:
            continue
        surv = np.array(surv)
        ranks = np.array(ranks) if ranks else np.array([0])
        share, ch = float(surv.mean()), float(np.mean(chance))
        r1, r2, r3 = [float(np.mean(ranks == k)) for k in (1, 2, 3)]
        print(f"{name:<15}{len(surv):>12}{share:>10.1%}{ch:>9.1%}{share - ch:>+9.1%}"
              f"{r1:>9.1%}{r2:>8.1%}{r3:>8.1%}{1 - r1 - r2 - r3:>9.1%}")
        rows_a.append(dict(activity=name, n_error_windows=len(surv),
                           survives=round(share, 4), chance=round(ch, 4),
                           excess=round(share - ch, 4), rank1=round(r1, 4), rank2=round(r2, 4),
                           rank3=round(r3, 4), rank_gt3=round(1 - r1 - r2 - r3, 4),
                           median_rank=int(np.median(ranks))))

    # ---------- B: error persistence ----------
    rows_b, dists = [], {}
    print(f"\nB. runs of consecutive windows with error > {ERROR_BPM:.0f} bpm")
    print(f"{'activity':<15}{'method':<14}{'runs':>7}{'median':>8}{'p90':>7}{'longest':>9}"
          f"{'>30s share':>12}{'resets/1k win':>15}")
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        for method, key in (("b2-zp", "base"), ("mask+tracker", "mt_est")):
            runs, n_reset, n_win = [], 0, 0
            for s in subs:
                m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
                err = np.abs(s[key] - s["hr"]) > ERROR_BPM
                # runs are contiguous in time within a block of this activity
                blocks = np.split(np.flatnonzero(m), np.flatnonzero(np.diff(np.flatnonzero(m)) > 1) + 1)
                for b in blocks:
                    if len(b):
                        runs += _runs(err[b])
                n_reset += int(s["resets"][m].sum())
                n_win += int(m.sum())
            if not runs:
                continue
            runs = np.array(runs)
            long_share = float(runs[runs > LONG_RUN_WINDOWS].sum() / runs.sum())
            dists[(name, method)] = runs
            print(f"{name:<15}{method:<14}{len(runs):>7}{np.median(runs):>8.0f}"
                  f"{np.percentile(runs, 90):>7.0f}{runs.max():>9}{long_share:>12.1%}"
                  f"{(n_reset / n_win * 1000 if method != 'b2-zp' else float('nan')):>15.1f}")
            rows_b.append(dict(activity=name, method=method, n_runs=len(runs),
                               n_error_windows=int(runs.sum()),
                               median_run=float(np.median(runs)),
                               p90_run=float(np.percentile(runs, 90)), longest_run=int(runs.max()),
                               share_in_runs_over_30s=round(long_share, 4),
                               resets_per_1000_windows=None if method == "b2-zp"
                               else round(n_reset / n_win * 1000, 2)))

    for name, rows, fn in [("surviving_peak", rows_a, "surviving_peak.csv"),
                           ("error_persistence", rows_b, "error_persistence.csv")]:
        with open(REPO_ROOT / "results" / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote results/{fn}")

    acts = [a for a in ACTIVITY_NAMES.values()]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    bins = np.arange(1, 32, 2)
    for ax, a in zip(axes.ravel(), acts):
        for method, colour in (("b2-zp", "#0072B2"), ("mask+tracker", "#D55E00")):
            r = dists.get((a, method))
            if r is not None:
                ax.hist(np.clip(r, 1, 30), bins=bins, alpha=0.55, label=method, color=colour,
                        weights=np.ones(len(r)) / len(r))
        ax.axvline(LONG_RUN_WINDOWS, color="black", ls="--", lw=1)
        ax.set_title(a, fontsize=10)
        ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("run length (windows of 2 s; 15 = 30 s, dashed)")
    fig.supylabel("share of runs")
    fig.suptitle("How long does an error last? Consecutive windows with error > 10 bpm")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "error_persistence.png", dpi=130)
    print("wrote figures/error_persistence.png")


if __name__ == "__main__":
    main()
