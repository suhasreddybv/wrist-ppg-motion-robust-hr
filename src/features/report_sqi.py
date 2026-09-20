"""Signal quality by activity, across all subjects.

Sanity check for Stage 2: if the SQI does not fall during walking and cycling
relative to sitting, it is not measuring what it claims to.

Writes results/02_sqi_by_activity.csv.
"""
import csv

import numpy as np

from src.data.loader import REPO_ROOT, load_all
from src.features.preprocess import preprocess_subject

ACTIVITY_NAMES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling",
    5: "driving", 6: "lunch", 7: "walking", 8: "working",
}


def main() -> None:
    rows = []
    per_activity = {a: [] for a in ACTIVITY_NAMES}
    for pre in (preprocess_subject(rec) for rec in load_all()):
        act = pre.windows.activity
        for a in ACTIVITY_NAMES:
            m = act == a
            if m.any():
                per_activity[a].append((pre.subject_id, m.sum(),
                                        pre.sqi.combined[m].mean(),
                                        pre.sqi.template_corr[m].mean(),
                                        pre.sqi.spectral_concentration[m].mean(),
                                        pre.sqi.out_of_band_ratio[m].mean()))
        print(f"  {pre.subject_id}: {len(pre)} windows, mean SQI {pre.sqi.combined.mean():.3f}")

    print(f"\n{'activity':<14}{'windows':>9}{'subjects':>10}{'SQI':>8}{'templ r':>9}"
          f"{'spec conc':>11}{'out-of-band':>13}")
    for a, name in ACTIVITY_NAMES.items():
        entries = per_activity[a]
        if not entries:
            continue
        n = sum(e[1] for e in entries)
        w = np.array([e[1] for e in entries], dtype=float)
        mean = lambda i: float(np.average([e[i] for e in entries], weights=w))  # noqa: E731
        row = dict(activity=name, windows=n, subjects=len(entries),
                   sqi=round(mean(2), 4), template_corr=round(mean(3), 4),
                   spectral_concentration=round(mean(4), 4),
                   out_of_band_ratio=round(mean(5), 4))
        rows.append(row)
        print(f"{name:<14}{n:>9}{len(entries):>10}{row['sqi']:>8.3f}{row['template_corr']:>9.3f}"
              f"{row['spectral_concentration']:>11.3f}{row['out_of_band_ratio']:>13.3f}")

    out = REPO_ROOT / "results" / "02_sqi_by_activity.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
