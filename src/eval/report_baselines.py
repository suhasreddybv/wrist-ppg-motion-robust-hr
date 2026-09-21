"""Run the Stage 3 baselines LOSO and write the result tables.

Writes results/baselines_per_activity.csv and results/baselines_per_subject.csv,
then checks the stop conditions from the 21 Sep scope:
  - b2 must not beat the b1 oracle on any activity (would suggest leakage);
  - b2 MAE under 3 bpm on stairs or cycling is implausible and must be checked.
"""
from __future__ import annotations

import csv

from src.data.loader import REPO_ROOT
from src.eval.loso import (
    METHOD_LABELS,
    build_subject_arrays,
    loso_predictions,
    per_activity_rows,
    per_subject_rows,
)
from src.models.baselines import bin_width_bpm

RESULTS = REPO_ROOT / "results"


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
        if method == "b2":
            b1 = by.get(("b1", activity))
            if b1 and row["mae"] < b1["mae"]:
                warnings.append(
                    f"STOP: b2 ({row['mae']:.2f}) beats the b1 oracle ({b1['mae']:.2f}) on {activity}. "
                    "Check window/label alignment and the band-pass before trusting this.")
            if activity in {"stairs", "cycling"} and row["mae"] < 3.0:
                warnings.append(
                    f"STOP: b2 MAE on {activity} is {row['mae']:.2f} bpm, implausibly low. "
                    "Check alignment and filtering.")
    return warnings


def main() -> None:
    print(f"b2 frequency grid: {bin_width_bpm():.2f} bpm per bin\n")
    subjects = build_subject_arrays()
    preds = loso_predictions(subjects)

    act_rows = per_activity_rows(subjects, preds)
    subj_rows = per_subject_rows(subjects, preds)

    print(f"{'method':<11}{'activity':<28}{'folds':>6}{'windows':>9}{'MAE':>8}{'SD':>7}{'worst':>8}{'RMSE':>8}")
    for r in act_rows:
        print(f"{r['method']:<11}{r['activity']:<28}{r['n_folds']:>6}{r['n_windows']:>9}"
              f"{r['mae']:>8.2f}{r['mae_sd_across_folds']:>7.2f}{r['mae_worst_fold']:>8.2f}{r['rmse']:>8.2f}")

    print(f"\n{'subject':<9}" + "".join(f"{m:>12}" for m in METHOD_LABELS))
    for subj in subjects:
        cells = {r["method"]: r["mae"] for r in subj_rows if r["subject"] == subj.subject_id}
        print(f"{subj.subject_id:<9}" + "".join(f"{cells[m]:>12.2f}" for m in METHOD_LABELS))

    _write(act_rows, "baselines_per_activity.csv")
    _write(subj_rows, "baselines_per_subject.csv")

    warnings = check_stop_conditions(act_rows)
    print("\n" + ("\n".join(warnings) if warnings else "Stop conditions: none triggered."))


if __name__ == "__main__":
    main()
