"""D-034: does the prediction rule explain the gap to SpaMaPlus?

D-034 named one difference from SpaMaPlus: prediction from a mean over the last
six estimates rather than the last one alone. That entry was written without
checking the code - the tracker has used a six-window mean since it was built
(`deque(maxlen=history)` with `history=6` in every grid configuration).

So the question is not whether to add the mean filter but whether the prediction
rule matters at all. This sweeps the window length from 1 (the last estimate
alone) to 15, and tries a median as a robust alternative, with everything else
held at each fold's own chosen configuration. Reported WITHOUT the bound, for
comparability with the published numbers.

Writes results/prediction_rule.csv.
"""
from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np

from src.data.loader import REPO_ROOT
from src.eval.metrics import mae, mape
from src.eval.stage4 import TRACK_GRID, fold_select
from src.eval.week3 import build
from src.models.tracker import peak_candidates, track

HISTORIES = (1, 2, 3, 6, 10, 15, 20, 30, 45, 60)
STATS = ("mean", "median")


def main() -> None:
    subs = build()
    _, mt_chosen, _ = fold_select(subs, "mask_track")
    pooled = lambda p: float(np.mean([mae(s["hr"], p[s["sid"]]) for s in subs]))  # noqa: E731

    # Reconciliation: reproduce the ablation's +mask+tracker exactly, to be sure the sweep's
    # hist=6 cell is the same computation and not a near-miss.
    exact = {}
    for s_ in subs:
        exact[s_["sid"]] = s_["mask_track"][mt_chosen[s_["sid"]]]
    recon = pooled(exact)
    print(f"reconciliation: ablation +mask+tracker = {recon:.3f} bpm")
    assert abs(recon - 15.799) < 0.01, (
        f"the sweep no longer reproduces the ablation's +mask+tracker ({recon:.3f} vs 15.799); "
        "the pipelines have diverged and the sweep cannot be compared to the ablation")

    variants: dict[str, dict[str, np.ndarray]] = {}
    for h in HISTORIES:
        for stat in STATS:
            key = f"{stat},hist={h}"
            variants[key] = {}
            for s in subs:
                ml, tl = mt_chosen[s["sid"]].split("|")
                tcfg = next(c for c in TRACK_GRID if c.label() == tl)
                masked = s["spec"] * s["gains"][ml]
                cand = peak_candidates(masked, s["f_bpm"], tcfg.prominence)
                variants[key][s["sid"]] = track(
                    masked, s["f_bpm"], replace(tcfg, history=h, pred_stat=stat), cand)

    # in-fold selection across the prediction rules
    chosen = {}
    for held in subs:
        scores = {k: np.mean([mae(s["hr"], variants[k][s["sid"]])
                              for s in subs if s["sid"] != held["sid"]]) for k in variants}
        chosen[held["sid"]] = min(scores, key=scores.get)
    selected = {s["sid"]: variants[chosen[s["sid"]]][s["sid"]] for s in subs}

    rows = []
    print(f"\nprediction rule sweep, no bound (each fold's own mask and tracker otherwise)")
    print(f"{'rule':<20}{'pooled MAE':>12}{'MAPE':>8}{'vs hist=6 mean':>17}")
    ref = pooled(variants["mean,hist=6"])
    for k, v in sorted(variants.items(), key=lambda kv: pooled(kv[1])):
        m = pooled(v)
        print(f"{k:<20}{m:>12.2f}{np.mean([mape(s['hr'], v[s['sid']]) for s in subs]):>8.2f}"
              f"{m - ref:>+17.2f}")
        rows.append(dict(rule=k, pooled_mae=round(m, 3),
                         pooled_mape=round(float(np.mean([mape(s["hr"], v[s["sid"]]) for s in subs])), 3),
                         delta_vs_mean6=round(m - ref, 3)))
    print(f"\nin-fold selected across rules: {pooled(selected):.2f} bpm")
    for k in sorted(set(chosen.values())):
        print(f"  {sum(x == k for x in chosen.values())}/15 folds chose {k}")
    rows.append(dict(rule="IN-FOLD SELECTED", pooled_mae=round(pooled(selected), 3),
                     pooled_mape=round(float(np.mean([mape(s["hr"], selected[s["sid"]]) for s in subs])), 3),
                     delta_vs_mean6=round(pooled(selected) - ref, 3)))

    # Diagnostic A on the residual errors of the selected rule
    from src.eval.diagnostics import ERROR_BPM, PEAK_TOL_BPM
    surv = []
    for s in subs:
        m = (s["activity"] != 0) & (np.abs(selected[s["sid"]] - s["hr"]) > ERROR_BPM)
        if not m.any():
            continue
        masked = s["spec"] * s["gains"][mt_chosen[s["sid"]].split("|")[0]]
        cand = peak_candidates(masked, s["f_bpm"], 0.05)
        for i in np.flatnonzero(m):
            c = cand[i]
            surv.append(bool(len(c) and (np.abs(c - s["hr"][i]) <= PEAK_TOL_BPM).any()))
    print(f"\nDiagnostic A on the selected rule: a peak within {PEAK_TOL_BPM:.0f} bpm of the truth "
          f"survives in {np.mean(surv):.1%} of {len(surv)} remaining error windows "
          f"(masked baseline 58.8%, Week 3 best 60.4%)")
    rows.append(dict(rule="diagnostic A survives", pooled_mae=round(float(np.mean(surv)), 4),
                     pooled_mape=len(surv), delta_vs_mean6=None))

    # the selected rule with the fold-derived bound, for the repository's second headline
    from src.eval.week3 import MARGIN_GRID, fold_bound  # noqa: E402
    bounded = {}
    for s_ in subs:
        train = [x["hr"] for x in subs if x["sid"] != s_["sid"]]
        b = fold_bound(train, MARGIN_GRID[0])
        ml, tl = mt_chosen[s_["sid"]].split("|")
        tcfg = next(c for c in TRACK_GRID if c.label() == tl)
        masked = s_["spec"] * s_["gains"][ml]
        cand = peak_candidates(masked, s_["f_bpm"], tcfg.prominence)
        best_rule = max(set(chosen.values()), key=list(chosen.values()).count)
        stat, hist = best_rule.split(",")[0], int(best_rule.split("hist=")[1])
        bounded[s_["sid"]] = track(masked, s_["f_bpm"],
                                   replace(tcfg, history=hist, pred_stat=stat,
                                           sustained_rule="ratio", ratio_min=0.3, sustained_after=5),
                                   cand, bound_bpm=b)
    print(f"selected rule + bound + ratio reset: {pooled(bounded):.2f} bpm "
          f"(Week 3 best with hist=6 was 12.50)")
    rows.append(dict(rule="selected + bound + reset", pooled_mae=round(pooled(bounded), 3),
                     pooled_mape=round(float(np.mean([mape(s_["hr"], bounded[s_["sid"]]) for s_ in subs])), 3),
                     delta_vs_mean6=None))

    with open(REPO_ROOT / "results" / "prediction_rule.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/prediction_rule.csv")


if __name__ == "__main__":
    main()
