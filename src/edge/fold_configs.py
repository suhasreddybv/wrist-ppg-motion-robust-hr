"""Record the configuration each LOSO fold selected for the +mask+tracker row, so the edge
work runs on a committed artifact rather than re-running the 80-configuration grid.

Reconciliation: with those configurations and the prediction rule switched to hist=15 (D-046),
the pooled MAE must equal results/prediction_rule.csv's "mean,hist=15" (14.704) to the third
decimal, or the script stops. Writes results/edge_fold_configs.csv.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import replace

import numpy as np

from src.data.loader import REPO_ROOT
from src.eval.metrics import mae
from src.eval.stage4 import TRACK_GRID, fold_select
from src.eval.week3 import build
from src.models.tracker import peak_candidates, track

HISTORY = 15


def main() -> int:
    subs = build()
    _, chosen, _ = fold_select(subs, "mask_track")
    rows, per = [], []
    for s in subs:
        ml, tl = chosen[s["sid"]].split("|")
        tcfg = replace(next(c for c in TRACK_GRID if c.label() == tl), history=HISTORY)
        masked = s["spec"] * s["gains"][ml]
        pred = track(masked, s["f_bpm"], tcfg, peak_candidates(masked, s["f_bpm"], tcfg.prominence))
        per.append(mae(s["hr"], pred))
        rows.append(dict(subject=s["sid"], mask_config=ml, tracker_config=tl, history=HISTORY, mae_hist15=round(per[-1], 4)))
    pooled = float(np.mean(per))
    ref = next(float(r["pooled_mae"]) for r in csv.DictReader(open(REPO_ROOT / "results" / "prediction_rule.csv")) if r["rule"] == "mean,hist=15")
    print(f"pooled MAE with the fold configurations at hist=15: {pooled:.3f} (prediction_rule.csv: {ref:.3f})")
    if abs(pooled - ref) > 5e-4:
        print("STOP: does not reconcile with the committed headline")
        return 2
    with open(REPO_ROOT / "results" / "edge_fold_configs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote results/edge_fold_configs.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
