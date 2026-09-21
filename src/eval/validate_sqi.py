"""Does the SQI predict error, or does it only track activity?

Ranking activities is weak evidence: motion lowers SQI and raises error, so the
ranking follows without the SQI saying anything about an individual window. The
test that matters is within an activity.

3a. Median and mean |b2-zp error| by SQI decile, deciles computed within activity.
3b. AUROC for detecting |error| > 10 bpm, per activity and pooled, with a bootstrap
    95% CI that resamples SUBJECTS, not windows: windows within a subject are
    correlated, so resampling windows would understate the interval. Every score is
    oriented so that HIGHER MEANS WORSE (the quality terms are negated, out-of-band
    power is not), so 0.5 is chance and above 0.5 means the term finds bad windows.
3c. The same for each component alone, and for the composite without the template
    term, to decide whether that term stays.

No thresholds are tuned here. If Stage 4 needs one it is chosen inside each LOSO
fold on training subjects only.

Writes results/sqi_validation.csv and figures/sqi_vs_error.png.
"""
from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, load_subject  # noqa: E402
from src.eval.loso import ACTIVITY_NAMES  # noqa: E402
from src.features.preprocess import composite_with_template, preprocess_subject  # noqa: E402
from src.models.baselines import ZERO_PAD_NFFT, spectral_peak_hr  # noqa: E402

ERROR_BPM = 10.0
N_BOOT = 2000
N_DECILES = 10
SEED = 42


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC; `positive` marks the windows we want a high score to find."""
    if positive.all() or not positive.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos = int(positive.sum())
    n_neg = len(score) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def boot_ci(per_subject: list[tuple[np.ndarray, np.ndarray]], rng) -> tuple[float, float, float]:
    """AUROC with a 95% CI from resampling subjects with replacement."""
    scores = np.concatenate([s for s, _ in per_subject])
    pos = np.concatenate([p for _, p in per_subject])
    point = auroc(scores, pos)
    n = len(per_subject)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        s = np.concatenate([per_subject[i][0] for i in pick])
        p = np.concatenate([per_subject[i][1] for i in pick])
        a = auroc(s, p)
        if not np.isnan(a):
            boots.append(a)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    rng = np.random.default_rng(SEED)
    subs = []
    for sid in [f"S{i}" for i in range(1, 16)]:
        pre = preprocess_subject(load_subject(sid))
        est = spectral_peak_hr(pre.bvp_filtered, fs=pre.fs, band=pre.band_hz, nfft=ZERO_PAD_NFFT)
        q = pre.sqi
        subs.append(dict(
            sid=sid, activity=pre.windows.activity, err=np.abs(est - pre.windows.hr),
            composite=q.combined, template=q.template_corr,
            concentration=q.spectral_concentration, out_of_band=q.out_of_band_ratio,
            # computed explicitly, so this comparison still reproduces after the
            # template term was dropped from the shipped composite
            with_template=composite_with_template(q.template_corr, q.spectral_concentration),
        ))

    # 3a: error by SQI decile within activity
    print("3a. |b2-zp error| by within-activity SQI decile (median bpm)")
    decile_curves = {}
    for act, name in ACTIVITY_NAMES.items():
        sc = np.concatenate([s["composite"][s["activity"] == act] for s in subs])
        er = np.concatenate([s["err"][s["activity"] == act] for s in subs])
        if len(sc) < N_DECILES:
            continue
        edges = np.quantile(sc, np.linspace(0, 1, N_DECILES + 1))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(sc, edges[1:-1]), 0, N_DECILES - 1)
        med = np.array([np.median(er[idx == d]) if (idx == d).any() else np.nan
                        for d in range(N_DECILES)])
        decile_curves[name] = med
        print(f"  {name:<14} D1 {med[0]:6.1f} -> D10 {med[-1]:6.1f}  (drop {med[0] - med[-1]:+.1f})")

    # 3b/3c: AUROC per activity and pooled, for the composite and each component
    # (label, key, sign): sign +1 where a high value already means a bad window.
    variants = {"composite (shipped)": ("composite", -1),
                "composite with template (pre-decision)": ("with_template", -1),
                "template only": ("template", -1),
                "spectral concentration only": ("concentration", -1),
                "out-of-band power": ("out_of_band", +1)}
    rows = []
    print(f"\n3b/3c. AUROC for detecting |error| > {ERROR_BPM:.0f} bpm (95% CI, subject bootstrap)")
    header = f"{'activity':<14}" + "".join(f"{v:>30}" for v in variants)
    print(header)
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "POOLED (excl. transient)")]:
        line = f"{name:<14}"
        for label, (key, sign) in variants.items():
            per_subject = []
            for s in subs:
                m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
                if m.sum() < 20:
                    continue
                score = sign * s[key][m]   # oriented so higher = worse window
                per_subject.append((score, s["err"][m] > ERROR_BPM))
            if len(per_subject) < 3:
                continue
            point, lo, hi = boot_ci(per_subject, rng)
            rows.append(dict(activity=name, variant=label, auroc=round(point, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             n_subjects=len(per_subject),
                             n_windows=int(sum(len(p) for p, _ in per_subject)),
                             includes_chance=bool(lo <= 0.5 <= hi)))
            line += f"{point:>16.3f} [{lo:.2f},{hi:.2f}]"
        print(line)

    # decision rule
    by = {(r["activity"], r["variant"]): r for r in rows}
    pooled_full = by[("POOLED (excl. transient)", "composite with template (pre-decision)")]
    pooled_not = by[("POOLED (excl. transient)", "spectral concentration only")]
    drop_template = pooled_full["ci_lo"] <= pooled_not["auroc"] <= pooled_full["ci_hi"]
    print(f"\nDecision: composite {pooled_full['auroc']:.3f} "
          f"[{pooled_full['ci_lo']:.2f},{pooled_full['ci_hi']:.2f}] vs without template "
          f"{pooled_not['auroc']:.3f} -> "
          + ("DROP the template term (within CI of the full composite)."
             if drop_template else "KEEP the template term (outside the CI)."))
    shipped = [r for r in rows if r["variant"] == "composite (shipped)"]
    weak = [r["activity"] for r in shipped if r["includes_chance"]]
    inverted = [r["activity"] for r in shipped if r["ci_hi"] < 0.5]
    print("Activities where the composite CI includes 0.5 (no discrimination): "
          + (", ".join(weak) if weak else "none"))
    print("Activities where the composite is ANTI-predictive (CI entirely below 0.5): "
          + (", ".join(inverted) if inverted else "none")
          + ". Stage 4 must not use SQI resets in any activity named on either line.")

    with open(REPO_ROOT / "results" / "sqi_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, med in decile_curves.items():
        axes[0].plot(range(1, N_DECILES + 1), med, marker="o", ms=3.5, label=name)
    axes[0].set_xlabel("within-activity SQI decile (1 = worst)")
    axes[0].set_ylabel("median |b2-zp error| (bpm)")
    axes[0].set_title("Does error fall as SQI rises, inside an activity?")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    comp = [r for r in rows if r["variant"] == "composite (shipped)"]
    y = np.arange(len(comp))
    axes[1].errorbar([r["auroc"] for r in comp], y,
                     xerr=[[r["auroc"] - r["ci_lo"] for r in comp], [r["ci_hi"] - r["auroc"] for r in comp]],
                     fmt="o", color="#0072B2", capsize=3)
    axes[1].axvline(0.5, color="#D55E00", ls="--", lw=1, label="chance")
    axes[1].set_yticks(y, [r["activity"] for r in comp], fontsize=8)
    axes[1].set_xlabel(f"AUROC for |error| > {ERROR_BPM:.0f} bpm (95% CI, subject bootstrap)")
    axes[1].set_title("Does the SQI find bad windows?")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "sqi_vs_error.png", dpi=130)
    print("\nwrote results/sqi_validation.csv and figures/sqi_vs_error.png")


if __name__ == "__main__":
    main()
