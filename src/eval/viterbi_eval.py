"""Viterbi decoding against the greedy tracker, LOSO.

Rows: b2-zp, the current headline (mask + greedy tracker, 15-window prediction),
fixed-lag Viterbi (causal, the comparable one) and full-session Viterbi
(non-causal, an upper bound on what selection alone can achieve).

K, sigma and the lag are chosen inside each fold on training subjects. No bound,
so every figure sits beside the published numbers.

Writes results/viterbi_ablation.csv, results/viterbi_per_subject.csv,
results/viterbi_diagnostics.csv.
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import replace

import numpy as np

from src.data.loader import REPO_ROOT
from src.eval.diagnostics import ERROR_BPM, LONG_RUN_WINDOWS, PEAK_TOL_BPM, _runs
from src.eval.loso import ACTIVITY_NAMES, POOLED_ALL, POOLED_NO_TRANSIENT
from src.eval.metrics import mae, mape
from src.eval.stage4 import TRACK_GRID, fold_select, rows_for
from src.eval.week3 import build
from src.models.tracker import peak_candidates, track
from src.models.viterbi import decode_fixed_lag, decode_full, top_k_peaks

K_GRID = (3, 4, 6)
SIGMA_GRID = (3.0, 6.0, 10.0, 15.0)
LAG_GRID = (1, 2, 4, 8)          # windows; 2 s each
HEADLINE_HISTORY = 15
SEED = 42


def main() -> None:
    subs = build()
    _, mt_chosen, _ = fold_select(subs, "mask_track")
    pooled = lambda p: float(np.mean([mae(s["hr"], p[s["sid"]]) for s in subs]))  # noqa: E731

    preds = {"b2-zp": {s["sid"]: s["base"] for s in subs}}

    # the current headline: mask + greedy tracker with the 15-window prediction
    headline = {}
    masked_by_sid, cand_by_sid = {}, {}
    for s in subs:
        ml, tl = mt_chosen[s["sid"]].split("|")
        tcfg = next(c for c in TRACK_GRID if c.label() == tl)
        masked = s["spec"] * s["gains"][ml]
        masked_by_sid[s["sid"]] = masked
        cand_by_sid[s["sid"]] = peak_candidates(masked, s["f_bpm"], tcfg.prominence)
        headline[s["sid"]] = track(masked, s["f_bpm"],
                                   replace(tcfg, history=HEADLINE_HISTORY), cand_by_sid[s["sid"]])
    preds["+mask+tracker(hist=15)"] = headline

    # decoder states: top-K peaks of the same masked spectra
    print("\nextracting candidate peaks")
    states = {}
    for k in K_GRID:
        states[k] = {s["sid"]: top_k_peaks(masked_by_sid[s["sid"]], s["f_bpm"], k) for s in subs}
        print(f"  K={k} done")

    full_variants, lag_variants = {}, {}
    for k, sg in itertools.product(K_GRID, SIGMA_GRID):
        full_variants[f"K={k},sigma={sg}"] = {
            s["sid"]: decode_full(*states[k][s["sid"]], sigma=sg) for s in subs}
        for L in LAG_GRID:
            lag_variants[f"K={k},sigma={sg},lag={L}"] = {
                s["sid"]: decode_fixed_lag(*states[k][s["sid"]], sigma=sg, lag=L) for s in subs}
        print(f"  decoded K={k}, sigma={sg}")

    def select(variants):
        chosen = {}
        for held in subs:
            scores = {key: np.mean([mae(s["hr"], v[s["sid"]]) for s in subs if s["sid"] != held["sid"]])
                      for key, v in variants.items()}
            chosen[held["sid"]] = min(scores, key=scores.get)
        return {s["sid"]: variants[chosen[s["sid"]]][s["sid"]] for s in subs}, chosen

    preds["+mask+viterbi-fixedlag"], lag_chosen = select(lag_variants)
    preds["+mask+viterbi-full"], full_chosen = select(full_variants)

    order = ["b2-zp", "+mask+tracker(hist=15)", "+mask+viterbi-fixedlag", "+mask+viterbi-full"]
    rng = np.random.default_rng(SEED)
    base = {s["sid"]: mae(s["hr"], s["base"]) for s in subs}
    rows, per_subject = [], []
    print(f"\n{'method':<28}{'pooled':>9}{'MAPE':>8}{'no-trans':>10}{'gain':>8}{'95% CI':>18}{'improved':>10}")
    for label in order:
        rows += rows_for(subs, preds[label], label)
        for s in subs:
            per_subject.append(dict(method=label, subject=s["sid"],
                                    mae=round(mae(s["hr"], preds[label][s["sid"]]), 3),
                                    mape=round(mape(s["hr"], preds[label][s["sid"]]), 3)))
        d = np.array([base[s["sid"]] - mae(s["hr"], preds[label][s["sid"]]) for s in subs])
        bs = [d[rng.integers(0, 15, 15)].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = next(r for r in rows if r["method"] == label and r["activity"] == POOLED_ALL)
        pn = next(r for r in rows if r["method"] == label and r["activity"] == POOLED_NO_TRANSIENT)
        print(f"{label:<28}{p['mae']:>9.2f}{p['mape']:>8.2f}{pn['mae']:>10.2f}{d.mean():>+8.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}{f'{sum(d > 0)}/15':>10}")

    # paired comparison against the headline, not just against b2-zp
    head = {s["sid"]: mae(s["hr"], headline[s["sid"]]) for s in subs}
    for label in ("+mask+viterbi-fixedlag", "+mask+viterbi-full"):
        d = np.array([head[s["sid"]] - mae(s["hr"], preds[label][s["sid"]]) for s in subs])
        bs = [d[rng.integers(0, 15, 15)].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {label} vs the greedy headline: {d.mean():+.2f} bpm "
              f"[{lo:+.2f}, {hi:+.2f}], {sum(d > 0)}/15 improved")

    print(f"\nchosen configurations")
    for name, chosen in (("fixed-lag", lag_chosen), ("full", full_chosen)):
        for key in sorted(set(chosen.values())):
            n = sum(v == key for v in chosen.values())
            lag_s = f" ({int(key.split('lag=')[1]) * 2} s delay)" if "lag=" in key else ""
            print(f"  {name:<10} {n:>2}/15 folds: {key}{lag_s}")

    print(f"\nper-activity MAE\n{'activity':<15}" + "".join(f"{m[:22]:>24}" for m in order))
    b0 = {r["activity"]: float(r["mae_mean_of_folds"])
          for r in csv.DictReader(open(REPO_ROOT / "results" / "baselines_per_activity.csv"))
          if r["method"] == "b0"}
    for act in ACTIVITY_NAMES.values():
        line = f"{act:<15}"
        for m in order:
            r = next((x for x in rows if x["method"] == m and x["activity"] == act), None)
            line += f"{r['mae']:>24.2f}" if r else f"{'-':>24}"
        fl = next(x for x in rows if x["method"] == "+mask+viterbi-fixedlag" and x["activity"] == act)
        line += "   beats b0" if fl["mae"] < b0[act] else "   WORSE than b0"
        print(line)

    # Diagnostics A and B on the fixed-lag result
    diag = []
    fl = preds["+mask+viterbi-fixedlag"]
    surv = []
    for s in subs:
        m = (s["activity"] != 0) & (np.abs(fl[s["sid"]] - s["hr"]) > ERROR_BPM)
        for i in np.flatnonzero(m):
            c = cand_by_sid[s["sid"]][i]
            surv.append(bool(len(c) and (np.abs(c - s["hr"][i]) <= PEAK_TOL_BPM).any()))
    print(f"\nDiagnostic A on fixed-lag Viterbi: {np.mean(surv):.1%} of {len(surv)} error windows "
          f"still contain the true peak (greedy headline 65.7%)")
    diag.append(dict(metric="diagnostic_A_survives", activity="ALL (excl. transient)",
                     value=round(float(np.mean(surv)), 4), n=len(surv)))

    print(f"\nDiagnostic B\n{'activity':<15}{'method':<22}{'runs':>7}{'median':>8}{'p90':>8}"
          f"{'longest':>9}{'>30s':>9}")
    for act, name in [(2, "stairs"), (4, "cycling"), (7, "walking"), (None, "ALL (excl. transient)")]:
        for label in ("+mask+tracker(hist=15)", "+mask+viterbi-fixedlag"):
            runs = []
            for s in subs:
                sel = (s["activity"] == act) if act is not None else (s["activity"] != 0)
                err = np.abs(preds[label][s["sid"]] - s["hr"]) > ERROR_BPM
                blocks = np.split(np.flatnonzero(sel), np.flatnonzero(np.diff(np.flatnonzero(sel)) > 1) + 1)
                for b in blocks:
                    if len(b):
                        runs += _runs(err[b])
            runs = np.array(runs)
            share = float(runs[runs > LONG_RUN_WINDOWS].sum() / runs.sum())
            print(f"{name:<15}{label:<22}{len(runs):>7}{np.median(runs):>8.0f}"
                  f"{np.percentile(runs, 90):>8.0f}{runs.max():>9}{share:>8.1%}")
            diag.append(dict(metric="persistence", activity=name, value=round(share, 4),
                             n=len(runs), method=label, median_run=float(np.median(runs)),
                             p90_run=float(np.percentile(runs, 90)), longest_run=int(runs.max())))

    res = REPO_ROOT / "results"
    for data, fn in ((rows, "viterbi_ablation.csv"), (per_subject, "viterbi_per_subject.csv"),
                     (diag, "viterbi_diagnostics.csv")):
        keys = sorted({k for d in data for k in d})
        with open(res / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        print(f"wrote results/{fn}")


if __name__ == "__main__":
    main()
