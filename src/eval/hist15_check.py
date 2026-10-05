"""Diagnostic B and the b0 table for the 15-window prediction headline (D-046).

A 15-window mean is 30 s of memory. Longer memory should slow the response to a
genuine rate change and lengthen lock-ins, so the 1.10 bpm gain may have been
bought with persistence. Diagnostic B was never rerun for it; this does that, and
the per-activity table against the b0 constant, for hist=15 against hist=6 and
b2-zp. No bound, so the numbers sit beside the published figures.

Writes results/hist15_persistence.csv and results/hist15_vs_b0.csv.
"""
from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np

from src.data.loader import REPO_ROOT
from src.eval.diagnostics import ERROR_BPM, LONG_RUN_WINDOWS, _runs
from src.eval.loso import ACTIVITY_NAMES
from src.eval.metrics import mae
from src.eval.stage4 import TRACK_GRID, fold_select
from src.eval.week3 import build
from src.models.tracker import peak_candidates, track

HISTORIES = (6, 15)


def main() -> None:
    subs = build()
    _, mt_chosen, _ = fold_select(subs, "mask_track")

    preds = {"b2-zp": {s["sid"]: s["base"] for s in subs}}
    for h in HISTORIES:
        preds[f"hist={h}"] = {}
        for s in subs:
            ml, tl = mt_chosen[s["sid"]].split("|")
            tcfg = next(c for c in TRACK_GRID if c.label() == tl)
            masked = s["spec"] * s["gains"][ml]
            cand = peak_candidates(masked, s["f_bpm"], tcfg.prominence)
            preds[f"hist={h}"][s["sid"]] = track(masked, s["f_bpm"], replace(tcfg, history=h), cand)

    methods = ["b2-zp", "hist=6", "hist=15"]
    print(f"\nDiagnostic B for the 15-window prediction\n{'activity':<15}{'method':<10}{'runs':>7}"
          f"{'median':>8}{'p90':>8}{'longest':>9}{'>30s share':>12}")
    rows = []
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        for m in methods:
            runs = []
            for s in subs:
                sel = (s["activity"] == act) if act is not None else (s["activity"] != 0)
                err = np.abs(preds[m][s["sid"]] - s["hr"]) > ERROR_BPM
                blocks = np.split(np.flatnonzero(sel), np.flatnonzero(np.diff(np.flatnonzero(sel)) > 1) + 1)
                for b in blocks:
                    if len(b):
                        runs += _runs(err[b])
            if not runs:
                continue
            runs = np.array(runs)
            share = float(runs[runs > LONG_RUN_WINDOWS].sum() / runs.sum())
            print(f"{name:<15}{m:<10}{len(runs):>7}{np.median(runs):>8.0f}"
                  f"{np.percentile(runs, 90):>8.0f}{runs.max():>9}{share:>12.1%}")
            rows.append(dict(activity=name, method=m, n_runs=len(runs),
                             median_run=float(np.median(runs)),
                             p90_run=float(np.percentile(runs, 90)), longest_run=int(runs.max()),
                             share_in_runs_over_30s=round(share, 4)))
        print()

    b0 = {r["activity"]: float(r["mae_mean_of_folds"])
          for r in csv.DictReader(open(REPO_ROOT / "results" / "baselines_per_activity.csv"))
          if r["method"] == "b0"}
    print(f"{'activity':<15}{'b0':>8}{'b2-zp':>9}{'hist=6':>9}{'hist=15':>9}{'verdict (hist=15)':>22}")
    b0_rows = []
    for act, name in ACTIVITY_NAMES.items():
        vals = {}
        for m in methods:
            per = [mae(s["hr"][s["activity"] == act], preds[m][s["sid"]][s["activity"] == act])
                   for s in subs if (s["activity"] == act).any()]
            vals[m] = float(np.mean(per))
        verdict = "beats b0" if vals["hist=15"] < b0[name] else "WORSE THAN b0"
        print(f"{name:<15}{b0[name]:>8.2f}{vals['b2-zp']:>9.2f}{vals['hist=6']:>9.2f}"
              f"{vals['hist=15']:>9.2f}{verdict:>22}")
        b0_rows.append(dict(activity=name, b0=round(b0[name], 3),
                            b2_zp=round(vals["b2-zp"], 3), hist6=round(vals["hist=6"], 3),
                            hist15=round(vals["hist=15"], 3),
                            hist15_beats_b0=bool(vals["hist=15"] < b0[name]),
                            hist15_vs_hist6=round(vals["hist=15"] - vals["hist=6"], 3)))

    for data, fn in ((rows, "hist15_persistence.csv"), (b0_rows, "hist15_vs_b0.csv")):
        with open(REPO_ROOT / "results" / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote results/{fn}")


if __name__ == "__main__":
    main()
