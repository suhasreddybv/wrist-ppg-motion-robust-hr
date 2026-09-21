"""Run the Stage 3 baselines LOSO and write the result tables.

Writes results/baselines_per_activity.csv, results/baselines_per_subject.csv and
results/clip_fraction_by_activity.csv, then checks the stop conditions:
  - b2 must not beat the b1 oracle on any activity (would suggest leakage);
  - b2 MAE under 3 bpm on stairs or cycling is implausible;
  - b2-zp must not improve stairs, table soccer or walking by more than 3 bpm
    over b2 - zero-padding interpolates the spectrum and cannot undo motion lock.
"""
from __future__ import annotations

import csv

from src.data.loader import REPO_ROOT
from src.eval.loso import (
    METHOD_LABELS,
    POOLED_ALL,
    build_subject_arrays,
    clip_fraction_rows,
    loso_predictions,
    per_activity_rows,
    per_subject_rows,
)
from src.models.baselines import ZERO_PAD_NFFT, bin_width_bpm

RESULTS = REPO_ROOT / "results"
ZP_STOP_BPM = 3.0
ZP_STOP_ACTIVITIES = ("stairs", "table soccer", "walking")


def _write(rows: list[dict], name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/{name}")


def check_stop_conditions(act_rows: list[dict]) -> list[str]:
    warnings = []
    by = {(r["method"], r["activity"]): r for r in act_rows}
    for (method, activity), row in by.items():
        if method != "b2":
            continue
        b1 = by.get(("b1", activity))
        if b1 and row["mae_mean_of_folds"] < b1["mae_mean_of_folds"]:
            warnings.append(
                f"STOP: b2 ({row['mae_mean_of_folds']:.2f}) beats the b1 oracle "
                f"({b1['mae_mean_of_folds']:.2f}) on {activity}. Check alignment and band-pass.")
        if activity in {"stairs", "cycling"} and row["mae_mean_of_folds"] < 3.0:
            warnings.append(
                f"STOP: b2 MAE on {activity} is {row['mae_mean_of_folds']:.2f} bpm, implausibly low.")
        if activity in ZP_STOP_ACTIVITIES:
            zp = by.get(("b2_zp", activity))
            gain = row["mae_mean_of_folds"] - zp["mae_mean_of_folds"] if zp else 0.0
            if gain > ZP_STOP_BPM:
                warnings.append(
                    f"STOP: b2-zp improves {activity} by {gain:.2f} bpm over b2 (> {ZP_STOP_BPM}). "
                    "Zero-padding should not fix motion lock; something else changed.")
    return warnings


def main() -> None:
    print(f"b2 grid: {bin_width_bpm():.2f} bpm per bin | "
          f"b2-zp grid (nfft={ZERO_PAD_NFFT}): {bin_width_bpm(nfft=ZERO_PAD_NFFT):.2f} bpm\n")
    subjects = build_subject_arrays()
    preds = loso_predictions(subjects)

    act_rows = per_activity_rows(subjects, preds)
    subj_rows = per_subject_rows(subjects, preds)
    clip_rows = clip_fraction_rows(subjects)

    print(f"{'method':<11}{'activity':<28}{'folds':>6}{'windows':>9}"
          f"{'MAE(fold)':>11}{'MAE(pool)':>11}{'SD':>7}{'worst':>8}{'MAPE(fold)':>12}{'MAPE(pool)':>12}")
    for r in act_rows:
        print(f"{r['method']:<11}{r['activity']:<28}{r['n_folds']:>6}{r['n_windows']:>9}"
              f"{r['mae_mean_of_folds']:>11.2f}{r['mae_pooled']:>11.2f}{r['mae_sd_across_folds']:>7.2f}"
              f"{r['mae_worst_fold']:>8.2f}{r['mape_mean_of_folds']:>12.2f}{r['mape_pooled']:>12.2f}")

    print(f"\nPer-subject MAE / MAPE, all windows:\n{'subject':<9}"
          + "".join(f"{m:>18}" for m in METHOD_LABELS))
    for subj in subjects:
        cells = {r["method"]: r for r in subj_rows
                 if r["subject"] == subj.subject_id and r["scope"] == "all"}
        print(f"{subj.subject_id:<9}" + "".join(
            f"{cells[m]['mae']:>11.2f}/{cells[m]['mape']:>5.1f}" for m in METHOD_LABELS))

    print(f"\nACC clipping by activity:\n{'activity':<15}{'windows':>9}{'flagged%':>10}"
          f"{'mean frac%':>12}{'p95 frac%':>11}{'max frac%':>11}")
    for r in clip_rows:
        print(f"{r['activity']:<15}{r['n_windows']:>9}{r['windows_flagged_pct']:>10.1f}"
              f"{r['mean_clip_fraction_pct']:>12.3f}{r['p95_clip_fraction_pct']:>11.3f}"
              f"{r['max_clip_fraction_pct']:>11.2f}")

    _write(act_rows, "baselines_per_activity.csv")
    _write(subj_rows, "baselines_per_subject.csv")
    _write(clip_rows, "clip_fraction_by_activity.csv")

    by = {(r["method"], r["activity"]): r for r in act_rows}
    b2, zp = by[("b2", POOLED_ALL)], by[("b2_zp", POOLED_ALL)]
    print(f"\nb2-zp vs b2 pooled: {b2['mae_mean_of_folds']:.2f} -> {zp['mae_mean_of_folds']:.2f} bpm")

    warnings = check_stop_conditions(act_rows)
    print("\n" + ("\n".join(warnings) if warnings else "Stop conditions: none triggered."))


if __name__ == "__main__":
    main()
