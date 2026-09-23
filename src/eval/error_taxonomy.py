"""What kind of error is it? A taxonomy of b2-zp failures, before any motion handling.

Every window with |b2-zp error| > 10 bpm gets exactly one label. The order below is
itself a decision: a window can satisfy several rules at once - a stride-locked
estimate near the band floor, say - and the first match wins.

    1. floor        estimate within 2 bpm of the lowest in-band frequency (24.375 bpm)
    2. acc_locked   estimate within 3 bpm of the dominant wrist-ACC frequency or its
                    2x or 0.5x harmonic; which one is recorded in `acc_harmonic`
    3. harmonic     estimate within 3 bpm of 2x or 0.5x the TRUE HR
    4. other        everything else

`floor` precedes `acc_locked` because the two have different fixes: a search-bound
change versus motion handling. Counting a floor-pinned window as accelerometer-locked
would credit motion handling with a gain that a lower bound could have had.

Writes results/error_taxonomy.csv and figures/error_taxonomy.png.
"""
from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, SUBJECT_IDS, load_subject  # noqa: E402
from src.eval.loso import ACTIVITY_NAMES  # noqa: E402
from src.eval.plot_motion_collision import acc_dominant_bpm  # noqa: E402
from src.features.preprocess import CARDIAC_BAND_HZ, preprocess_subject  # noqa: E402
from src.models.baselines import ZERO_PAD_NFFT, spectral_peak_hr  # noqa: E402

ERROR_BPM = 10.0
FLOOR_TOL_BPM = 2.0
LOCK_TOL_BPM = 3.0
BOTTOM_OF_BAND_BPM = 5.0      # "bottom 5 bpm of the band", for the floor question
LABELS = ("floor", "acc_locked", "harmonic", "other")
OTHER_SHARE_WARN = 0.40


def lowest_in_band_bpm(nfft: int = ZERO_PAD_NFFT, fs: int = 64) -> float:
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    return float(freqs[freqs >= CARDIAC_BAND_HZ[0]][0] * 60.0)


def classify(est: np.ndarray, true_hr: np.ndarray, acc_bpm: np.ndarray,
             floor_bpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Label each window; returns (label, acc_harmonic) as object arrays."""
    label = np.full(len(est), "other", dtype=object)
    harmonic = np.full(len(est), "", dtype=object)

    is_floor = np.abs(est - floor_bpm) <= FLOOR_TOL_BPM

    d1 = np.abs(est - acc_bpm)
    d2 = np.abs(est - 2 * acc_bpm)
    dh = np.abs(est - 0.5 * acc_bpm)
    stack = np.vstack([d1, d2, dh])
    which = stack.argmin(axis=0)
    is_acc = stack.min(axis=0) <= LOCK_TOL_BPM

    is_harm = (np.abs(est - 2 * true_hr) <= LOCK_TOL_BPM) | (np.abs(est - 0.5 * true_hr) <= LOCK_TOL_BPM)

    label[is_harm] = "harmonic"                       # applied in reverse precedence order,
    label[is_acc & ~is_floor] = "acc_locked"          # so earlier rules overwrite later ones
    harmonic[is_acc & ~is_floor] = np.array(["1x", "2x", "0.5x"], dtype=object)[which[is_acc & ~is_floor]]
    label[is_floor] = "floor"
    harmonic[is_floor] = ""
    return label, harmonic


def main() -> None:
    floor_bpm = lowest_in_band_bpm()
    print(f"lowest in-band frequency on the b2-zp grid: {floor_bpm:.3f} bpm "
          f"(floor tolerance +-{FLOOR_TOL_BPM}, lock tolerance +-{LOCK_TOL_BPM})\n")

    subs = []
    for sid in SUBJECT_IDS:
        pre = preprocess_subject(load_subject(sid))
        est = spectral_peak_hr(pre.bvp_filtered, fs=pre.fs, band=pre.band_hz, nfft=ZERO_PAD_NFFT)
        acc = acc_dominant_bpm(pre.acc_64, pre.fs)
        hr = pre.windows.hr
        lab, harm = classify(est, hr, acc, floor_bpm)
        subs.append(dict(sid=sid, est=est, hr=hr, acc=acc, err=est - hr, label=lab, harmonic=harm,
                         activity=pre.windows.activity, sqi=pre.sqi.combined))

    big = {s["sid"]: np.abs(s["err"]) > ERROR_BPM for s in subs}

    rows = []
    # --- per activity, cohort-wide -------------------------------------------
    print(f"{'activity':<15}{'err windows':>12}" + "".join(f"{l:>12}" for l in LABELS))
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        lab, sqi, err = [], [], []
        for s in subs:
            m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
            m = m & big[s["sid"]]
            lab.append(s["label"][m]); sqi.append(s["sqi"][m]); err.append(s["err"][m])
        lab, sqi, err = np.concatenate(lab), np.concatenate(sqi), np.concatenate(err)
        n = len(lab)
        if not n:
            continue
        line = f"{name:<15}{n:>12}"
        for lname in LABELS:
            k = lab == lname
            line += f"{k.sum() / n:>11.1%}"
            rows.append(dict(scope="activity", name=name, label=lname, n_error_windows=n,
                             count=int(k.sum()), share=round(float(k.sum() / n), 4),
                             median_signed_error=round(float(np.median(err[k])), 2) if k.any() else None,
                             median_sqi=round(float(np.median(sqi[k])), 4) if k.any() else None))
        print(line)

    # --- per class, cohort-wide ----------------------------------------------
    print(f"\n{'class':<12}{'share':>8}{'median signed err':>20}{'median SQI':>13}{'ACC harmonic mix':>26}")
    lab = np.concatenate([s["label"][big[s["sid"]] & (s["activity"] != 0)] for s in subs])
    sqi = np.concatenate([s["sqi"][big[s["sid"]] & (s["activity"] != 0)] for s in subs])
    err = np.concatenate([s["err"][big[s["sid"]] & (s["activity"] != 0)] for s in subs])
    harm = np.concatenate([s["harmonic"][big[s["sid"]] & (s["activity"] != 0)] for s in subs])
    ok_sqi = np.median(sqi)
    for lname in LABELS:
        k = lab == lname
        mix = ""
        if lname == "acc_locked" and k.any():
            vals, counts = np.unique(harm[k], return_counts=True)
            mix = " ".join(f"{v}:{c / k.sum():.0%}" for v, c in zip(vals, counts))
        print(f"{lname:<12}{k.sum() / len(lab):>8.1%}{np.median(err[k]) if k.any() else float('nan'):>20.1f}"
              f"{np.median(sqi[k]) if k.any() else float('nan'):>13.3f}{mix:>26}")
        rows.append(dict(scope="cohort", name="ALL (excl. transient)", label=lname,
                         n_error_windows=len(lab), count=int(k.sum()),
                         share=round(float(k.sum() / len(lab)), 4),
                         median_signed_error=round(float(np.median(err[k])), 2) if k.any() else None,
                         median_sqi=round(float(np.median(sqi[k])), 4) if k.any() else None))
    print(f"(median SQI over all error windows: {ok_sqi:.3f}; a class above this is one the "
          f"current quality measure cannot see)")

    # --- S5 and the median subject -------------------------------------------
    pooled_mae = {s["sid"]: float(np.mean(np.abs(s["err"]))) for s in subs}
    median_sid = min(pooled_mae, key=lambda k: abs(pooled_mae[k] - np.median(list(pooled_mae.values()))))
    print(f"\nmedian subject by pooled |b2-zp error|: {median_sid} "
          f"({pooled_mae[median_sid]:.2f} bpm; cohort median {np.median(list(pooled_mae.values())):.2f})")
    print(f"{'subject':<10}{'err windows':>12}" + "".join(f"{l:>12}" for l in LABELS))
    for sid in ("S5", median_sid):
        s = next(x for x in subs if x["sid"] == sid)
        m = big[sid] & (s["activity"] != 0)
        line = f"{sid:<10}{int(m.sum()):>12}"
        for lname in LABELS:
            k = s["label"][m] == lname
            line += f"{k.sum() / m.sum():>11.1%}"
            rows.append(dict(scope="subject", name=sid, label=lname, n_error_windows=int(m.sum()),
                             count=int(k.sum()), share=round(float(k.sum() / m.sum()), 4),
                             median_signed_error=round(float(np.median(s["err"][m][k])), 2) if k.any() else None,
                             median_sqi=round(float(np.median(s["sqi"][m][k])), 4) if k.any() else None))
        print(line)

    # --- the floor question ---------------------------------------------------
    lo_edge = CARDIAC_BAND_HZ[0] * 60
    print(f"\nEstimates in the bottom {BOTTOM_OF_BAND_BPM:.0f} bpm of the band "
          f"(<= {lo_edge + BOTTOM_OF_BAND_BPM:.0f} bpm), as a share of ALL windows:")
    print(f"{'activity':<15}{'all windows':>12}{'bottom-of-band':>16}{'true HR min':>13}"
          f"{'p1':>8}{'p5':>8}")
    floor_rows = []
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        est, hr = [], []
        for s in subs:
            m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
            est.append(s["est"][m]); hr.append(s["hr"][m])
        est, hr = np.concatenate(est), np.concatenate(hr)
        share = float(np.mean(est <= lo_edge + BOTTOM_OF_BAND_BPM))
        print(f"{name:<15}{len(est):>12}{share:>15.1%}{hr.min():>13.1f}"
              f"{np.percentile(hr, 1):>8.1f}{np.percentile(hr, 5):>8.1f}")
        floor_rows.append(dict(scope="floor", name=name, label="bottom_of_band",
                               n_error_windows=len(est), count=int((est <= lo_edge + BOTTOM_OF_BAND_BPM).sum()),
                               share=round(share, 4),
                               median_signed_error=None,
                               median_sqi=None,
                               true_hr_min=round(float(hr.min()), 1),
                               true_hr_p1=round(float(np.percentile(hr, 1)), 1),
                               true_hr_p5=round(float(np.percentile(hr, 5)), 1)))
    rows = [dict(r, true_hr_min=None, true_hr_p1=None, true_hr_p5=None) for r in rows] + floor_rows

    with open(REPO_ROOT / "results" / "error_taxonomy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- figure ---------------------------------------------------------------
    acts = [n for n in ACTIVITY_NAMES.values()]
    shares = {l: [next((r["share"] for r in rows if r["scope"] == "activity" and r["name"] == a
                        and r["label"] == l), 0.0) for a in acts] for l in LABELS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bottom = np.zeros(len(acts))
    colours = {"floor": "#0072B2", "acc_locked": "#E69F00", "harmonic": "#009E73", "other": "#999999"}
    for l in LABELS:
        axes[0].bar(acts, shares[l], bottom=bottom, label=l, color=colours[l])
        bottom += np.array(shares[l])
    axes[0].set_ylabel(f"share of windows with |error| > {ERROR_BPM:.0f} bpm")
    axes[0].set_title("What kind of error, by activity")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)

    all_est = np.concatenate([s["est"][s["activity"] != 0] for s in subs])
    all_hr = np.concatenate([s["hr"][s["activity"] != 0] for s in subs])
    axes[1].hist(all_est, bins=np.arange(24, 241, 2.5), alpha=0.75, label="b2-zp estimate", color="#D55E00")
    axes[1].hist(all_hr, bins=np.arange(24, 241, 2.5), alpha=0.6, label="true HR", color="#0072B2")
    axes[1].axvline(lo_edge + BOTTOM_OF_BAND_BPM, color="black", ls="--", lw=1,
                    label=f"bottom {BOTTOM_OF_BAND_BPM:.0f} bpm of band")
    axes[1].axvline(float(all_hr.min()), color="#0072B2", ls=":", lw=1.4,
                    label=f"lowest true HR ({all_hr.min():.1f} bpm)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("bpm")
    axes[1].set_title("Estimates extend far below the true-HR range (no label under 41.7 bpm)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "error_taxonomy.png", dpi=130)

    other_share = float(np.mean(lab == "other"))
    floor_share = float(np.mean(lab == "floor"))
    print(f"\nwrote results/error_taxonomy.csv and figures/error_taxonomy.png")
    if other_share > OTHER_SHARE_WARN:
        print(f"NOTE: 'other' is {other_share:.1%} of error windows, above the {OTHER_SHARE_WARN:.0%} "
              "threshold. Reported as-is; tolerances were not widened.")
    if floor_share < 0.10:
        print(f"STOP CONDITION: 'floor' is {floor_share:.1%} of error windows, below 10%. "
              "Recommend dropping the floor idea - it was inferred from five S5 windows.")
    else:
        print(f"'floor' is {floor_share:.1%} of error windows (>= 10%): the floor question stands.")


if __name__ == "__main__":
    main()
