"""Stage 5 - the evaluation items the pipeline spec asks for beyond MAE.

- Fitzpatrick stratification, reported per subject rather than as a stratum mean:
  the strata are n=1, n=11 and n=3, which settles nothing statistically. It is
  reported because almost no published work on this dataset reports it at all.
- Pearson r and Bland-Altman bias with 95% limits of agreement, pooled and per
  activity, for b2-zp and mask+tracker.
- Three figures: per-activity MAE against the b0 constant, Bland-Altman, and the
  best and worst windows of the full method chosen by rule.

Writes results/skin_type.csv, results/agreement.csv, figures/per_activity_mae.png,
figures/bland_altman.png and figures/best_worst_windows.png.
"""
from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, SUBJECT_IDS, load_subject  # noqa: E402
from src.eval.diagnostics import PROMINENCE, _selected  # noqa: E402
from src.eval.loso import ACTIVITY_NAMES  # noqa: E402
from src.eval.metrics import bland_altman, mae, mape, pearson_r, rmse  # noqa: E402
from src.eval.stage4 import config_from_label  # noqa: E402
from src.features.preprocess import CARDIAC_BAND_HZ, preprocess_subject  # noqa: E402
from src.models.masking import band_spectra, mask_gains  # noqa: E402
from src.models.tracker import peak_candidates, track  # noqa: E402

C_BASE, C_FULL, C_B0 = "#0072B2", "#D55E00", "#666666"


def build():
    sel = _selected()
    subs = []
    for sid in SUBJECT_IDS:
        rec = load_subject(sid)
        pre = preprocess_subject(rec)
        f_hz, spec = band_spectra(pre.bvp_filtered, pre.fs)
        f_bpm = f_hz * 60.0
        mlabel, tlabel = sel["+mask+tracker"][sid].split("|")
        gains, _ = mask_gains(pre.acc_64, pre.fs, f_hz, config_from_label(mlabel))
        masked = spec * gains
        cand = peak_candidates(masked, f_bpm, PROMINENCE)
        full = track(masked, f_bpm, config_from_label(tlabel), cand)
        subs.append(dict(sid=sid, skin=rec.skin_type, hr=pre.windows.hr,
                         activity=pre.windows.activity, start_s=pre.windows.start_s,
                         base=f_bpm[spec.argmax(axis=1)], full=full, f_bpm=f_bpm,
                         masked=masked, bvp=pre.bvp_filtered, acc=pre.acc_64, fs=pre.fs))
        print(f"  {sid} done")
    return subs


def main() -> None:
    subs = build()

    # ---------- Fitzpatrick ----------
    print("\nFitzpatrick skin type - per subject, because the strata are n=1, n=11 and n=3")
    print(f"{'type':<6}{'subject':<9}{'b2-zp':>9}{'mask+tracker':>15}")
    skin_rows = []
    for t in (2, 3, 4):
        members = [s for s in subs if s["skin"] == t]
        for s in members:
            b, f = mae(s["hr"], s["base"]), mae(s["hr"], s["full"])
            print(f"{t:<6}{s['sid']:<9}{b:>9.2f}{f:>15.2f}")
            skin_rows.append(dict(skin_type=t, subject=s["sid"], n_in_stratum=len(members),
                                  mae_b2zp=round(b, 3), mae_mask_tracker=round(f, 3),
                                  mape_b2zp=round(mape(s["hr"], s["base"]), 3),
                                  mape_mask_tracker=round(mape(s["hr"], s["full"]), 3)))
        m_b = np.mean([mae(s["hr"], s["base"]) for s in members])
        m_f = np.mean([mae(s["hr"], s["full"]) for s in members])
        print(f"{'':<6}{'(mean, n=' + str(len(members)) + ')':<9}{m_b:>9.2f}{m_f:>15.2f}")

    # ---------- agreement ----------
    agr_rows = []
    print(f"\nAgreement with the ECG reference\n{'activity':<15}{'method':<14}{'r':>7}"
          f"{'bias':>8}{'LoA low':>10}{'LoA high':>10}{'RMSE':>8}")
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "POOLED (excl. transient)")]:
        for method, key in (("b2-zp", "base"), ("mask+tracker", "full")):
            t, p = [], []
            for s in subs:
                m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
                t.append(s["hr"][m])
                p.append(s[key][m])
            t, p = np.concatenate(t), np.concatenate(p)
            r = pearson_r(t, p)
            bias, lo, hi = bland_altman(t, p)
            print(f"{name:<15}{method:<14}{r:>7.3f}{bias:>8.2f}{lo:>10.2f}{hi:>10.2f}{rmse(t, p):>8.2f}")
            agr_rows.append(dict(activity=name, method=method, n_windows=len(t),
                                 pearson_r=round(r, 4), bias_bpm=round(bias, 3),
                                 loa_low=round(lo, 3), loa_high=round(hi, 3),
                                 rmse=round(rmse(t, p), 3), mae=round(mae(t, p), 3)))

    for rows, fn in ((skin_rows, "skin_type.csv"), (agr_rows, "agreement.csv")):
        with open(REPO_ROOT / "results" / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote results/{fn}")

    # ---------- figure: per-activity MAE with the b0 constant ----------
    ab = {(r["method"], r["activity"]): float(r["mae"])
          for r in csv.DictReader(open(REPO_ROOT / "results" / "stage4_ablation.csv"))}
    b0 = {r["activity"]: float(r["mae_mean_of_folds"])
          for r in csv.DictReader(open(REPO_ROOT / "results" / "baselines_per_activity.csv"))
          if r["method"] == "b0"}
    acts = list(ACTIVITY_NAMES.values())
    x = np.arange(len(acts))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, [ab[("b2-zp", a)] for a in acts], 0.38, label="b2-zp", color=C_BASE)
    ax.bar(x + 0.2, [ab[("+mask+tracker", a)] for a in acts], 0.38, label="mask+tracker", color=C_FULL)
    for i, a in enumerate(acts):
        ax.plot([i - 0.42, i + 0.42], [b0[a]] * 2, color=C_B0, lw=2,
                label="b0 constant (predict the mean)" if i == 0 else None)
    ax.set_xticks(x, acts, rotation=30, ha="right")
    ax.set_ylabel("MAE (bpm), mean of per-fold MAEs")
    ax.set_title("Per-activity error. Bars above the grey line are worse than predicting a constant.")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "per_activity_mae.png", dpi=130)
    print("wrote figures/per_activity_mae.png")

    # ---------- figure: Bland-Altman ----------
    t = np.concatenate([s["hr"][s["activity"] != 0] for s in subs])
    p = np.concatenate([s["full"][s["activity"] != 0] for s in subs])
    bias, lo, hi = bland_altman(t, p)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hexbin((t + p) / 2, p - t, gridsize=60, cmap="Greys", bins="log", mincnt=1)
    for y, lab, ls in ((bias, f"bias {bias:+.2f}", "-"), (lo, f"LoA {lo:+.1f}", "--"),
                       (hi, f"LoA {hi:+.1f}", "--")):
        ax.axhline(y, color=C_FULL, ls=ls, lw=1.5, label=lab)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("mean of estimate and ECG reference (bpm)")
    ax.set_ylabel("estimate − reference (bpm)")
    ax.set_title("Bland–Altman, mask+tracker vs ECG (non-transient windows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "bland_altman.png", dpi=130)
    print("wrote figures/bland_altman.png")

    # ---------- figure: best and worst windows, chosen by rule ----------
    # Worst: the largest absolute error in the cohort. Best: among windows the method got
    # to within 1 bpm, the one with the most wrist motion - the hardest window it got right.
    worst = max(((s, int(np.argmax(np.abs(s["full"] - s["hr"])))) for s in subs),
                key=lambda sw: abs(sw[0]["full"][sw[1]] - sw[0]["hr"][sw[1]]))
    best = None
    for s in subs:
        ok = np.flatnonzero((np.abs(s["full"] - s["hr"]) <= 1.0) & (s["activity"] != 0))
        if len(ok):
            motion = np.linalg.norm(s["acc"][ok], axis=2).std(axis=1)
            k = ok[int(np.argmax(motion))]
            if best is None or motion.max() > best[2]:
                best = (s, int(k), float(motion.max()))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for row, (s, i, title) in enumerate([(best[0], best[1], "best: hardest window estimated within 1 bpm"),
                                         (worst[0], worst[1], "worst: largest error in the cohort")]):
        t_axis = np.arange(s["bvp"].shape[1]) / s["fs"]
        axes[row, 0].plot(t_axis, s["bvp"][i], lw=0.8, color="black")
        axes[row, 0].set_ylabel(f"{s['sid']} w{i}\n{ACTIVITY_NAMES.get(int(s['activity'][i]), 'transient')}")
        axes[row, 0].set_title(f"{title} — band-passed BVP", fontsize=10)
        spec = s["masked"][i] / s["masked"][i].max()
        axes[row, 1].plot(s["f_bpm"], spec, color=C_BASE, lw=1.2)
        axes[row, 1].axvline(s["hr"][i], color="#009E73", ls="--", lw=1.6, label=f"true {s['hr'][i]:.0f}")
        axes[row, 1].axvline(s["full"][i], color=C_FULL, ls=":", lw=1.6, label=f"estimate {s['full'][i]:.0f}")
        axes[row, 1].set_title("masked spectrum", fontsize=10)
        axes[row, 1].legend(fontsize=8)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 1].set_xlabel("bpm")
    fig.suptitle("mask+tracker: best and worst windows, selected by rule")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "best_worst_windows.png", dpi=125)
    print("wrote figures/best_worst_windows.png")


if __name__ == "__main__":
    main()
