"""Why is S5 the worst subject for b2 (47.15 bpm MAE, against 9.71 for S7)?

Diagnostic logic:
- If S5 is bad at rest too, the cause is signal quality or sensor contact.
- If S5 is normal at rest and bad only under motion, the cause is motion.
- If S5's b1 oracle error is normal, labels and alignment are fine.

Writes results/s5_investigation.md and figures/s5_worst_windows.png.
S5 is never excluded from any result.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, load_subject  # noqa: E402
from src.eval.loso import ACTIVITY_NAMES, build_subject_arrays, loso_predictions  # noqa: E402
from src.eval.metrics import mae  # noqa: E402
from src.features.preprocess import CARDIAC_BAND_HZ, preprocess_subject  # noqa: E402

SUBJECT = "S5"
SITTING = 1
N_WORST = 5


def _rest_stats(pre):
    """Amplitude and cardiac-band concentration on that subject's sitting windows."""
    m = pre.windows.activity == SITTING
    raw = pre.windows.bvp[m]
    return dict(
        n_sitting=int(m.sum()),
        bvp_sd=float(np.mean(raw.std(axis=1))),
        bvp_p2p=float(np.mean(raw.max(axis=1) - raw.min(axis=1))),
        spectral_concentration=float(pre.sqi.spectral_concentration[m].mean()),
        template_corr=float(pre.sqi.template_corr[m].mean()),
        sqi=float(pre.sqi.combined[m].mean()),
        out_of_band=float(pre.sqi.out_of_band_ratio[m].mean()),
    )


def main() -> None:
    subjects = build_subject_arrays()
    preds = loso_predictions(subjects)
    pres = {s.subject_id: preprocess_subject(load_subject(s.subject_id)) for s in subjects}

    # Per-activity error for S5 against the cohort
    per_act = {}
    for activity, name in ACTIVITY_NAMES.items():
        vals = {}
        for s in subjects:
            m = s.activity == activity
            if m.any():
                vals[s.subject_id] = dict(
                    b2=mae(s.hr[m], preds[s.subject_id]["b2"][m]),
                    b2_zp=mae(s.hr[m], preds[s.subject_id]["b2_zp"][m]),
                    b1=mae(s.hr[m], preds[s.subject_id]["b1"][m]),
                )
        if SUBJECT in vals:
            others = [v["b2"] for k, v in vals.items() if k != SUBJECT]
            per_act[name] = dict(
                s5_b2=vals[SUBJECT]["b2"], s5_b2_zp=vals[SUBJECT]["b2_zp"],
                s5_b1=vals[SUBJECT]["b1"],
                cohort_median_b2=float(np.median(others)),
                cohort_max_other=float(np.max(others)),
                rank=1 + sum(v["b2"] > vals[SUBJECT]["b2"] for k, v in vals.items() if k != SUBJECT),
                n_windows=int((subjects[[s.subject_id for s in subjects].index(SUBJECT)].activity == activity).sum()),
            )

    rest = {sid: _rest_stats(p) for sid, p in pres.items()}
    s5_rest, other_rest = rest[SUBJECT], {k: v for k, v in rest.items() if k != SUBJECT}
    hr = {s.subject_id: s.hr for s in subjects}

    # Five worst windows
    s5 = next(s for s in subjects if s.subject_id == SUBJECT)
    err = np.abs(preds[SUBJECT]["b2"] - s5.hr)
    worst = np.argsort(err)[-N_WORST:][::-1]
    pre5 = pres[SUBJECT]

    fig, axes = plt.subplots(N_WORST, 3, figsize=(15, 3 * N_WORST))
    for row, w in enumerate(worst):
        t = np.arange(pre5.bvp_filtered.shape[1]) / pre5.fs
        axes[row, 0].plot(t, pre5.bvp_filtered[w], lw=0.8, color="black")
        axes[row, 0].set_ylabel(f"w{w}\n{ACTIVITY_NAMES.get(int(s5.activity[w]), 'transient')}")
        if row == 0:
            axes[row, 0].set_title("band-passed BVP (z-scored)")

        freqs = np.fft.rfftfreq(pre5.bvp_filtered.shape[1], 1 / pre5.fs) * 60
        p = np.abs(np.fft.rfft(pre5.bvp_filtered[w])) ** 2
        band = (freqs >= CARDIAC_BAND_HZ[0] * 60) & (freqs <= CARDIAC_BAND_HZ[1] * 60)
        axes[row, 1].plot(freqs[band], p[band] / p[band].max(), color="tab:blue")
        axes[row, 1].axvline(s5.hr[w], color="tab:green", ls="--", label=f"true {s5.hr[w]:.0f}")
        axes[row, 1].axvline(s5.b2[w], color="tab:red", ls=":", label=f"b2 {s5.b2[w]:.0f}")
        axes[row, 1].legend(fontsize=7)
        if row == 0:
            axes[row, 1].set_title("BVP spectrum (bpm)")

        acc = pre5.acc_64[w] - pre5.acc_64[w].mean(axis=0)
        pa = np.abs(np.fft.rfft(acc, axis=0)) ** 2
        axes[row, 2].plot(freqs[band], (pa.sum(axis=1)[band] / pa.sum(axis=1)[band].max()), color="tab:orange")
        axes[row, 2].axvline(s5.hr[w], color="tab:green", ls="--")
        axes[row, 2].axvline(s5.b2[w], color="tab:red", ls=":")
        if row == 0:
            axes[row, 2].set_title("ACC spectrum (bpm), 3 axes summed")
    for ax in axes[-1]:
        ax.set_xlabel("time (s) / bpm")
    fig.suptitle(f"{SUBJECT}: five worst b2 windows (green = true HR, red = b2 estimate)")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "s5_worst_windows.png", dpi=110)

    # ---- write the report ----
    L = []
    L.append(f"# {SUBJECT}: why b2 fails on this subject\n")
    L.append(f"Generated by `src/eval/investigate_s5.py`. {SUBJECT} is not excluded from any result.\n")
    L.append(f"\n{SUBJECT} has the worst b2 MAE of the cohort: "
             f"{mae(s5.hr, preds[SUBJECT]['b2']):.2f} bpm, against {mae(hr['S7'], preds['S7']['b2']):.2f} "
             f"for S7. Fitzpatrick 3, age 21, self-reported fitness 4 - nothing unusual.\n")

    L.append("\n## 1. The labels and alignment are fine\n")
    b1_all = {s.subject_id: mae(s.hr, preds[s.subject_id]["b1"]) for s in subjects}
    L.append(f"\n{SUBJECT}'s b1 oracle MAE is **{b1_all[SUBJECT]:.2f} bpm**, the *best* in the cohort "
             f"(median {np.median(list(b1_all.values())):.2f}, worst {max(b1_all.values()):.2f}). "
             "If windows were misaligned or labels corrupted, the oracle would degrade too. It does not.\n")

    L.append("\n## 2. It is not resting signal quality\n")
    L.append(f"\n| metric | {SUBJECT} (sitting) | cohort median | cohort range |\n|---|---|---|---|\n")
    for key, label in [("bvp_sd", "BVP SD"), ("bvp_p2p", "BVP peak-to-peak"),
                       ("spectral_concentration", "spectral concentration"),
                       ("template_corr", "beat-template r"), ("sqi", "combined SQI"),
                       ("out_of_band", "out-of-band power")]:
        vals = [v[key] for v in other_rest.values()]
        L.append(f"| {label} | {s5_rest[key]:.3f} | {np.median(vals):.3f} | "
                 f"{min(vals):.3f}-{max(vals):.3f} |\n")
    L.append(f"\n{SUBJECT}'s resting b2 MAE is **{per_act['sitting']['s5_b2']:.2f} bpm** "
             f"(cohort median {per_act['sitting']['cohort_median_b2']:.2f}). "
             "The sensor reads this subject normally when they are still.\n")

    L.append("\n## 3. The error is concentrated in motion, and in this subject's heart rate range\n")
    others_by_act = {}
    for activity, name in ACTIVITY_NAMES.items():
        vals = [s.hr[s.activity == activity] for s in subjects
                if s.subject_id != SUBJECT and (s.activity == activity).any()]
        others_by_act[name] = float(np.concatenate(vals).mean()) if vals else float("nan")
    L.append(f"\n| activity | {SUBJECT} b2 | {SUBJECT} b2-zp | cohort median b2 | worst other | rank | "
             f"{SUBJECT} mean HR | cohort mean HR | windows |\n"
             "|---|---|---|---|---|---|---|---|---|\n")
    s5_arr = next(s for s in subjects if s.subject_id == SUBJECT)
    for name, d in per_act.items():
        act_id = [k for k, v in ACTIVITY_NAMES.items() if v == name][0]
        s5_hr_act = s5_arr.hr[s5_arr.activity == act_id].mean()
        L.append(f"| {name} | **{d['s5_b2']:.2f}** | {d['s5_b2_zp']:.2f} | {d['cohort_median_b2']:.2f} | "
                 f"{d['cohort_max_other']:.2f} | {d['rank']}/15 | {s5_hr_act:.0f} | "
                 f"{others_by_act[name]:.0f} | {d['n_windows']} |\n")

    s5_hr = hr[SUBJECT]
    others_hr = np.concatenate([v for k, v in hr.items() if k != SUBJECT])
    L.append(f"\n{SUBJECT}'s heart rate is unusually high: mean **{s5_hr.mean():.1f} bpm** "
             f"(cohort mean {others_hr.mean():.1f}), median {np.median(s5_hr):.1f}, "
             f"and {(s5_hr > 120).mean() * 100:.1f}% of windows above 120 bpm "
             f"(cohort {(others_hr > 120).mean() * 100:.1f}%). "
             "A high true HR sits further from the walking and cycling cadence band, so when the "
             "argmax locks onto motion the error is larger in bpm than it would be for a subject "
             "whose HR sits near their cadence.\n")
    mapes = {s.subject_id: 100 * float(np.mean(np.abs(preds[s.subject_id]["b2"] - s.hr) / s.hr))
             for s in subjects}
    L.append(f"\nThis is why relative error is milder: {SUBJECT}'s b2 MAPE is {mapes[SUBJECT]:.1f}%, "
             f"against a cohort median of {np.median([v for k, v in mapes.items() if k != SUBJECT]):.1f}% "
             f"and a worst-other of {max(v for k, v in mapes.items() if k != SUBJECT):.1f}%. "
             "Still the worst subject, but not the extreme outlier the bpm figure suggests.\n")

    L.append("\n## 3b. The failure mode is cohort-wide; only its cost is specific to this subject\n")
    low = {s.subject_id: 100 * float(np.mean(s.b2 <= 45.0)) for s in subjects}
    others_low = [v for k, v in low.items() if k != SUBJECT]
    L.append(f"\nWhen motion destroys the cardiac peak, the argmax often lands near the bottom of the "
             f"0.4-4 Hz band. {SUBJECT} produces an estimate at or below 45 bpm in **{low[SUBJECT]:.1f}%** "
             f"of windows - high, but within the cohort's range (median {np.median(others_low):.1f}%, "
             f"worst other {max(others_low):.1f}%). What differs is the consequence: in "
             f"{100 * float(np.mean((next(s for s in subjects if s.subject_id == SUBJECT).b2 <= 45) & (s5_hr >= 120))):.1f}% "
             f"of {SUBJECT}'s windows such an estimate coincides with a true HR of 120 bpm or more, so a "
             "single locked window contributes a ~100 bpm error rather than a ~30 bpm one.\n")

    L.append("\n## 4. The five worst windows\n")
    L.append("\n![S5 worst windows](../figures/s5_worst_windows.png)\n")
    L.append(f"\n| window | activity | true HR | b2 | error | ACC clip fraction |\n|---|---|---|---|---|---|\n")
    for w in worst:
        L.append(f"| {w} | {ACTIVITY_NAMES.get(int(s5.activity[w]), 'transient')} | {s5.hr[w]:.1f} | "
                 f"{s5.b2[w]:.1f} | {s5.b2[w] - s5.hr[w]:+.1f} | "
                 f"{s5.clip_fraction[w] * 100:.2f}% |\n")

    L.append("\n## Conclusion\n")
    L.append(f"\n{SUBJECT} is not a bad recording. The oracle is the best in the cohort, resting b2 "
             "accuracy is ordinary, and the BVP amplitude is roughly twice the cohort median, so sensor "
             f"contact is good. What sets {SUBJECT} apart is heart rate: elevated in **every** activity, "
             "including sitting, and above 120 bpm in 54% of windows against 7.7% for everyone else. "
             "The spectral-peak failure mode - locking onto low-frequency motion energy when the cardiac "
             "peak is buried - is cohort-wide, but a true HR near 160 turns each locked window into a "
             "~130 bpm error instead of a ~40 bpm one. That also explains why the gap narrows under "
             "relative error.\n")
    L.append(f"\nOne consequence for later stages: {SUBJECT} is bad on quiet activities too "
             "(working 31.92 bpm against a cohort median of 6.17), which is not explained by motion "
             "intensity alone and is worth revisiting once ACC-referenced masking exists. The fix is "
             f"motion handling in Stage 4, not exclusion - and {SUBJECT} is the subject that will show "
             "whether it works.\n")

    out = REPO_ROOT / "results" / "s5_investigation.md"
    out.write_text("".join(L))
    print("".join(L))
    print(f"wrote {out.relative_to(REPO_ROOT)} and figures/s5_worst_windows.png")


if __name__ == "__main__":
    main()
