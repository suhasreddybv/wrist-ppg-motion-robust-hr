"""C1 and C2: is the bound robust, and does the reset do anything without it?

C1. Every fold chose a zero margin, which sits at the tightest edge of the grid -
    exactly where in-sample gain is maximised. The bound is then the training
    sample's minimum, a biased estimate of the population minimum, and it will be
    too high for a bradycardic, athletic or beta-blocked subject. This traces
    pooled MAE against margin, from 0 down to 30 bpm below the training minimum.
    If most of the gain survives a 30 bpm margin the bound is robust; if it
    collapses, it is a cohort-specific ceiling rather than a component.

C2. "Without bound: 15.80" was the mask+tracker figure, which predates the reset.
    The honest without-bound number is the best configuration without a bound -
    mask, tracker and reset together. If it is still 15.80, the reset does nothing
    without the bound, and that is a finding.

Writes results/bound_sensitivity.csv and results/without_bound_best.csv.
"""
from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np

from src.data.loader import REPO_ROOT
from src.eval.metrics import mae, mape
from src.eval.stage4 import MASK_GRID, TRACK_GRID, fold_select
from src.eval.week3 import MARGIN_GRID, RESET_GRID, build, fold_bound
from src.models.masking import mask_gains  # noqa: F401  (imported for symmetry with week3)
from src.models.tracker import peak_candidates, track

MARGINS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)


def main() -> None:
    subs = build()
    _, mt_chosen, _ = fold_select(subs, "mask_track")
    base_mae = {s["sid"]: mae(s["hr"], s["base"]) for s in subs}
    pooled = lambda p: float(np.mean([mae(s["hr"], p[s["sid"]]) for s in subs]))  # noqa: E731

    # per-fold bound at each margin
    bounds = {m: {} for m in MARGINS}
    for held in subs:
        train = [s["hr"] for s in subs if s["sid"] != held["sid"]]
        for m in MARGINS:
            bounds[m][held["sid"]] = fold_bound(train, m)

    print(f"\nC1. bound sensitivity\n{'margin':>7}{'bound (bpm)':>13}{'+bound alone':>14}"
          f"{'mask+track+bound':>19}{'windows changed':>17}{'gain kept':>11}")
    rows, full_at_margin = [], {}
    for m in MARGINS:
        # +bound alone
        b_alone, changed = {}, 0
        for s in subs:
            keep = s["f_bpm"] >= bounds[m][s["sid"]]
            b_alone[s["sid"]] = s["f_bpm"][keep][s["spec"][:, keep].argmax(axis=1)]
            changed += int((b_alone[s["sid"]] != s["base"]).sum())
        # mask+tracker with the fold's own configuration, bounded at this margin
        mt = {}
        for s in subs:
            ml, tl = mt_chosen[s["sid"]].split("|")
            tcfg = next(c for c in TRACK_GRID if c.label() == tl)
            masked = s["spec"] * s["gains"][ml]
            cand = peak_candidates(masked, s["f_bpm"], tcfg.prominence)
            mt[s["sid"]] = track(masked, s["f_bpm"], tcfg, cand, bound_bpm=bounds[m][s["sid"]])
        full_at_margin[m] = mt
        gain_alone = float(np.mean([base_mae[s["sid"]] - mae(s["hr"], b_alone[s["sid"]]) for s in subs]))
        rows.append(dict(margin_bpm=m, mean_bound_bpm=round(float(np.mean(list(bounds[m].values()))), 2),
                         mae_bound_alone=round(pooled(b_alone), 3),
                         gain_bound_alone=round(gain_alone, 3),
                         mae_mask_tracker_bound=round(pooled(mt), 3),
                         windows_changed=changed,
                         share_changed=round(changed / sum(len(s["hr"]) for s in subs), 4)))
        print(f"{m:>7.0f}{np.mean(list(bounds[m].values())):>13.1f}{pooled(b_alone):>14.2f}"
              f"{pooled(mt):>19.2f}{changed:>17}{'':>11}")
    g0 = rows[0]["gain_bound_alone"]
    for r in rows:
        r["share_of_margin0_gain"] = round(r["gain_bound_alone"] / g0, 3)
    print(f"\n  gain of the bound alone, as a share of the margin-0 gain:")
    for r in rows:
        print(f"    margin {r['margin_bpm']:>4.0f} -> bound {r['mean_bound_bpm']:>5.1f} bpm: "
              f"{r['gain_bound_alone']:>+5.2f} bpm ({r['share_of_margin0_gain']:.0%} of the margin-0 gain)")

    # ---------- C2: the best configuration WITHOUT a bound ----------
    print("\nC2. best configuration without a bound")
    reset_preds = {}
    for rcfg in RESET_GRID:
        key = (f"{rcfg.sustained_rule}={rcfg.rank_max if rcfg.sustained_rule == 'rank' else rcfg.ratio_min},"
               f"after={rcfg.sustained_after}")
        reset_preds[key] = {}
        for s in subs:
            ml, tl = mt_chosen[s["sid"]].split("|")
            tcfg = next(c for c in TRACK_GRID if c.label() == tl)
            merged = replace(tcfg, sustained_rule=rcfg.sustained_rule, rank_max=rcfg.rank_max,
                             ratio_min=rcfg.ratio_min, sustained_after=rcfg.sustained_after)
            masked = s["spec"] * s["gains"][ml]
            cand = peak_candidates(masked, s["f_bpm"], tcfg.prominence)
            reset_preds[key][s["sid"]] = track(masked, s["f_bpm"], merged, cand)   # no bound
    chosen = {}
    for held in subs:
        scores = {k: np.mean([mae(s["hr"], reset_preds[k][s["sid"]])
                              for s in subs if s["sid"] != held["sid"]]) for k in reset_preds}
        chosen[held["sid"]] = min(scores, key=scores.get)
    best = {s["sid"]: reset_preds[chosen[s["sid"]]][s["sid"]] for s in subs}

    mt_nobound, _, _ = fold_select(subs, "mask_track")
    rows2 = [dict(configuration="mask+tracker (Week 2, no reset, no bound)",
                  mae=round(pooled(mt_nobound), 3), mape=round(float(np.mean(
                      [mape(s["hr"], mt_nobound[s["sid"]]) for s in subs])), 3)),
             dict(configuration="mask+tracker+reset, no bound",
                  mae=round(pooled(best), 3), mape=round(float(np.mean(
                      [mape(s["hr"], best[s["sid"]]) for s in subs])), 3))]
    for r in rows2:
        print(f"  {r['configuration']:<44}{r['mae']:>8.2f}")
    delta = rows2[0]["mae"] - rows2[1]["mae"]
    print(f"  the reset is worth {delta:+.2f} bpm without a bound "
          f"(it is worth {15.80 - 12.50:+.2f} with one, from 12.97 to 12.50: {12.97 - 12.50:+.2f})")
    for k in sorted(set(chosen.values())):
        print(f"  selected by {sum(v == k for v in chosen.values())}/15 folds: {k}")

    res = REPO_ROOT / "results"
    for data, fn in ((rows, "bound_sensitivity.csv"), (rows2, "without_bound_best.csv")):
        with open(res / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote results/{fn}")


if __name__ == "__main__":
    main()
