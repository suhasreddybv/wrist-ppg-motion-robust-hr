"""Week 3, block 1: break the lock-in.

Two components that address the same failure from opposite sides.

1. A physiological lower bound on the search space. D-024 measured that estimates
   below 40 bpm are always wrong in this cohort while the lowest ECG label is
   41.7 bpm. The bound is derived INSIDE each fold as (minimum training label -
   margin), never from the global minimum and never hand-picked, and the margin is
   selected on training subjects. b2-zp is left untouched as the reference.

2. A sustained-wrongness reset. The jump reset cannot see a smoothly tracked wrong
   estimate, which is why error episodes reach 13 minutes (D-033). Two formulations
   are tried: a hard rank test (the tracked peak must stay within the top-N peaks)
   and a continuous height-ratio test.

Selection is staged, not joint: mask and tracker are chosen first on the
+mask+tracker row, then the bound margin with those fixed, then the reset rule.
A joint grid would be ~2,000 configurations per subject. The staging is greedy and
is reported as such.

Disclosure: SpaMa and SpaMaPlus search the full band, so the bound is a departure
from the published comparison. Both figures are reported and the without-bound one
is used whenever comparing to published numbers.

Writes results/week3_ablation.csv, results/week3_per_subject.csv,
results/week3_bound_effect.csv, results/week3_reset_rates.csv.
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import replace

import numpy as np

from src.data.loader import REPO_ROOT, SUBJECT_IDS, load_subject
from src.eval.loso import ACTIVITY_NAMES, POOLED_ALL, POOLED_NO_TRANSIENT
from src.eval.metrics import mae, mape, rmse
from src.eval.stage4 import MASK_GRID, TRACK_GRID, fold_select, rows_for
from src.features.preprocess import preprocess_subject
from src.models.masking import band_spectra, mask_gains
from src.models.tracker import peak_candidates, track

MARGIN_GRID = (0.0, 5.0, 10.0)      # bound = min(training labels) - margin
RESET_GRID = ([replace(TRACK_GRID[0], sustained_rule="rank", rank_max=r, sustained_after=a)
               for r, a in itertools.product((2, 3), (2, 3, 5))]
              + [replace(TRACK_GRID[0], sustained_rule="ratio", ratio_min=q, sustained_after=a)
                 for q, a in itertools.product((0.3, 0.5), (2, 3, 5))])
SEED = 42
ERROR_BPM = 10.0


def fold_bound(train_hr: list[np.ndarray], margin: float) -> float:
    """The rule, stated once: the lowest heart rate seen in training, less a margin."""
    return float(min(h.min() for h in train_hr) - margin)


def build():
    subs = []
    for sid in SUBJECT_IDS:
        pre = preprocess_subject(load_subject(sid))
        f_hz, spec = band_spectra(pre.bvp_filtered, pre.fs)
        f_bpm = f_hz * 60.0
        rec = dict(sid=sid, hr=pre.windows.hr, activity=pre.windows.activity, f_bpm=f_bpm,
                   spec=spec, acc=pre.acc_64, fs=pre.fs, f_hz=f_hz,
                   base=f_bpm[spec.argmax(axis=1)], mask={}, mask_track={}, gains={})
        for cfg in MASK_GRID:
            g, _ = mask_gains(pre.acc_64, pre.fs, f_hz, cfg)
            rec["gains"][cfg.label()] = g
            rec["mask"][cfg.label()] = f_bpm[(spec * g).argmax(axis=1)]
        cand = peak_candidates(spec, f_bpm, TRACK_GRID[-1].prominence)
        rec["track"] = {c.label(): track(spec, f_bpm, c, cand) for c in TRACK_GRID}
        for mcfg in MASK_GRID:
            masked = spec * rec["gains"][mcfg.label()]
            mcand = peak_candidates(masked, f_bpm, TRACK_GRID[-1].prominence)
            for tcfg in TRACK_GRID:
                rec["mask_track"][f"{mcfg.label()}|{tcfg.label()}"] = track(masked, f_bpm, tcfg, mcand)
        subs.append(rec)
        print(f"  {sid} built")
    return subs


def main() -> None:
    subs = build()
    by_sid = {s["sid"]: s for s in subs}
    preds = {"b2-zp": {s["sid"]: s["base"] for s in subs}}

    # --- rows that do not involve the bound -------------------------------------
    for label, variant in (("+tracker", "track"), ("+mask", "mask"), ("+mask+tracker", "mask_track")):
        p, chosen, _ = fold_select(subs, variant)
        preds[label] = p
        if label == "+mask+tracker":
            mt_chosen = chosen

    # --- +bound alone: restrict the plain spectrum --------------------------------
    print("\nderiving the bound inside each fold")
    bound_rows, bounds = [], {}
    margin_scores = {m: {} for m in MARGIN_GRID}
    for held in subs:
        train = [s["hr"] for s in subs if s["sid"] != held["sid"]]
        for m in MARGIN_GRID:
            margin_scores[m][held["sid"]] = fold_bound(train, m)
    # choose the margin on training subjects: score each margin by training MAE of +bound
    def bound_pred(s, b):
        keep = s["f_bpm"] >= b
        return s["f_bpm"][keep][s["spec"][:, keep].argmax(axis=1)]

    per_margin = {m: {s["sid"]: bound_pred(s, margin_scores[m][s["sid"]]) for s in subs}
                  for m in MARGIN_GRID}
    chosen_margin = {}
    for held in subs:
        scores = {m: np.mean([mae(s["hr"], per_margin[m][s["sid"]])
                              for s in subs if s["sid"] != held["sid"]]) for m in MARGIN_GRID}
        chosen_margin[held["sid"]] = min(scores, key=scores.get)
    preds["+bound"] = {s["sid"]: per_margin[chosen_margin[s["sid"]]][s["sid"]] for s in subs}
    for s in subs:
        b = margin_scores[chosen_margin[s["sid"]]][s["sid"]]
        bounds[s["sid"]] = b
        changed = preds["+bound"][s["sid"]] != s["base"]
        bound_rows.append(dict(subject=s["sid"], margin=chosen_margin[s["sid"]], bound_bpm=round(b, 2),
                               windows=len(s["hr"]), windows_changed=int(changed.sum()),
                               share_changed=round(float(changed.mean()), 4),
                               mae_before=round(mae(s["hr"], s["base"]), 3),
                               mae_after=round(mae(s["hr"], preds["+bound"][s["sid"]]), 3)))
        print(f"  {s['sid']}: margin {chosen_margin[s['sid']]:.0f} -> bound {b:.1f} bpm, "
              f"{changed.sum()} of {len(changed)} windows changed")

    # --- +mask+tracker+bound: fixed mask/tracker per fold, margin chosen in-fold ---
    distinct = sorted(set(mt_chosen.values()))
    mtb = {cfg: {m: {} for m in MARGIN_GRID} for cfg in distinct}
    for cfg in distinct:
        ml, tl = cfg.split("|")
        mcfg = next(c for c in MASK_GRID if c.label() == ml)
        tcfg = next(c for c in TRACK_GRID if c.label() == tl)
        for s in subs:
            masked = s["spec"] * s["gains"][mcfg.label()]
            mcand = peak_candidates(masked, s["f_bpm"], tcfg.prominence)
            for m in MARGIN_GRID:
                b = margin_scores[m][s["sid"]]
                mtb[cfg][m][s["sid"]] = track(masked, s["f_bpm"], tcfg, mcand, bound_bpm=b)
    chosen_margin_mt = {}
    for held in subs:
        cfg = mt_chosen[held["sid"]]
        scores = {m: np.mean([mae(s["hr"], mtb[cfg][m][s["sid"]])
                              for s in subs if s["sid"] != held["sid"]]) for m in MARGIN_GRID}
        chosen_margin_mt[held["sid"]] = min(scores, key=scores.get)
    preds["+mask+tracker+bound"] = {s["sid"]: mtb[mt_chosen[s["sid"]]][chosen_margin_mt[s["sid"]]][s["sid"]]
                                    for s in subs}

    # --- + rank/ratio reset on top -------------------------------------------------
    print("\nsustained-wrongness reset: two formulations")

    def reset_label(c):
        arg = c.rank_max if c.sustained_rule == "rank" else c.ratio_min
        return f"{c.sustained_rule}={arg},after={c.sustained_after}"

    reset_preds, reset_flags = {}, {}
    for rcfg in RESET_GRID:
        reset_preds[reset_label(rcfg)], reset_flags[reset_label(rcfg)] = {}, {}
        for s in subs:
            cfg = mt_chosen[s["sid"]]
            ml, tl = cfg.split("|")
            tcfg = next(c for c in TRACK_GRID if c.label() == tl)
            merged = replace(tcfg, sustained_rule=rcfg.sustained_rule, rank_max=rcfg.rank_max,
                             ratio_min=rcfg.ratio_min, sustained_after=rcfg.sustained_after)
            masked = s["spec"] * s["gains"][ml]
            mcand = peak_candidates(masked, s["f_bpm"], tcfg.prominence)
            b = margin_scores[chosen_margin_mt[s["sid"]]][s["sid"]]
            est, res = track(masked, s["f_bpm"], merged, mcand, return_resets=True, bound_bpm=b)
            reset_preds[reset_label(rcfg)][s["sid"]] = est
            reset_flags[reset_label(rcfg)][s["sid"]] = res
    chosen_reset = {}
    for held in subs:
        scores = {k: np.mean([mae(s["hr"], reset_preds[k][s["sid"]])
                              for s in subs if s["sid"] != held["sid"]]) for k in reset_preds}
        chosen_reset[held["sid"]] = min(scores, key=scores.get)
    preds["+mask+tracker+bound+rankreset"] = {s["sid"]: reset_preds[chosen_reset[s["sid"]]][s["sid"]]
                                              for s in subs}
    for k in sorted(set(chosen_reset.values())):
        print(f"  selected by {sum(v == k for v in chosen_reset.values())}/15 folds: {k} "
              f"(on top of each fold's own tracker configuration)")
    # every reset rule scored, so the two formulations can be compared
    print(f"  {'rule':<22}{'pooled MAE':>12}")
    for k in sorted(reset_preds, key=lambda k: np.mean([mae(s['hr'], reset_preds[k][s['sid']]) for s in subs])):
        print(f"  {k:<22}{np.mean([mae(s['hr'], reset_preds[k][s['sid']]) for s in subs]):>12.2f}")

    # --- reset rates and degeneracy check -----------------------------------------
    rate_rows = []
    print(f"\nreset rate per activity (windows per 1,000) for the selected reset rule")
    print(f"{'activity':<15}{'rate':>8}{'flag':>8}")
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        fired = win = 0
        for s in subs:
            m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
            fired += int(reset_flags[chosen_reset[s["sid"]]][s["sid"]][m].sum())
            win += int(m.sum())
        rate = fired / win
        flag = "DEGENERATE" if rate > 0.2 else ""
        print(f"{name:<15}{rate * 1000:>8.1f}{flag:>12}")
        rate_rows.append(dict(activity=name, resets_per_1000_windows=round(rate * 1000, 2),
                              n_windows=win, degenerate=bool(rate > 0.2)))

    # --- tables --------------------------------------------------------------------
    order = ["b2-zp", "+bound", "+tracker", "+mask", "+mask+tracker",
             "+mask+tracker+bound", "+mask+tracker+bound+rankreset"]
    rows, per_subject = [], []
    rng = np.random.default_rng(SEED)
    base = {s["sid"]: mae(s["hr"], s["base"]) for s in subs}
    print(f"\n{'method':<32}{'pooled':>9}{'MAPE':>8}{'no-trans':>10}{'gain':>8}{'95% CI':>18}{'improved':>10}")
    for label in order:
        rows += rows_for(subs, preds[label], label)
        for s in subs:
            per_subject.append(dict(method=label, subject=s["sid"],
                                    mae=round(mae(s["hr"], preds[label][s["sid"]]), 3),
                                    mape=round(mape(s["hr"], preds[label][s["sid"]]), 3),
                                    rmse=round(rmse(s["hr"], preds[label][s["sid"]]), 3)))
        d = np.array([base[s["sid"]] - mae(s["hr"], preds[label][s["sid"]]) for s in subs])
        bs = [d[rng.integers(0, 15, 15)].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = next(r for r in rows if r["method"] == label and r["activity"] == POOLED_ALL)
        pn = next(r for r in rows if r["method"] == label and r["activity"] == POOLED_NO_TRANSIENT)
        print(f"{label:<32}{p['mae']:>9.2f}{p['mape']:>8.2f}{pn['mae']:>10.2f}{d.mean():>+8.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}{f'{sum(d > 0)}/15':>10}")

    print(f"\nper-activity MAE\n{'activity':<15}" + "".join(f"{m[:14]:>16}" for m in order))
    for act in ACTIVITY_NAMES.values():
        line = f"{act:<15}"
        for m in order:
            r = next((x for x in rows if x["method"] == m and x["activity"] == act), None)
            line += f"{r['mae']:>16.2f}" if r else f"{'-':>16}"
        print(line)

    res = REPO_ROOT / "results"
    for data, fn in ((rows, "week3_ablation.csv"), (per_subject, "week3_per_subject.csv"),
                     (bound_rows, "week3_bound_effect.csv"), (rate_rows, "week3_reset_rates.csv")):
        with open(res / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote results/{fn}")

    # ---- rerun the diagnostics on the best configuration ----
    best = preds["+mask+tracker+bound+rankreset"]
    from src.eval.diagnostics import ERROR_BPM as DIAG_ERR, LONG_RUN_WINDOWS, _runs

    print(f"\nDiagnostic B, rerun: runs of consecutive windows with error > {DIAG_ERR:.0f} bpm")
    print(f"{'activity':<15}{'method':<16}{'runs':>7}{'median':>8}{'p90':>7}{'longest':>9}{'>30s share':>12}")
    persist_rows = []
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        for method, p in (("b2-zp", preds["b2-zp"]), ("mask+tracker", preds["+mask+tracker"]),
                          ("week3 best", best)):
            runs = []
            for s in subs:
                m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
                err = np.abs(p[s["sid"]] - s["hr"]) > DIAG_ERR
                blocks = np.split(np.flatnonzero(m), np.flatnonzero(np.diff(np.flatnonzero(m)) > 1) + 1)
                for b in blocks:
                    if len(b):
                        runs += _runs(err[b])
            if not runs:
                continue
            runs = np.array(runs)
            share = float(runs[runs > LONG_RUN_WINDOWS].sum() / runs.sum())
            print(f"{name:<15}{method:<16}{len(runs):>7}{np.median(runs):>8.0f}"
                  f"{np.percentile(runs, 90):>7.0f}{runs.max():>9}{share:>12.1%}")
            persist_rows.append(dict(activity=name, method=method, n_runs=len(runs),
                                     n_error_windows=int(runs.sum()),
                                     median_run=float(np.median(runs)),
                                     p90_run=float(np.percentile(runs, 90)),
                                     longest_run=int(runs.max()),
                                     share_in_runs_over_30s=round(share, 4)))

    print("\nagainst the b0 constant (mean of per-fold MAE)")
    b0 = {r["activity"]: float(r["mae_mean_of_folds"])
          for r in csv.DictReader(open(res / "baselines_per_activity.csv")) if r["method"] == "b0"}
    print(f"{'activity':<15}{'b0':>8}{'b2-zp':>9}{'mask+trk':>10}{'week3':>9}{'verdict':>22}")
    b0_rows = []
    for act in ACTIVITY_NAMES.values():
        g = lambda m: next(r["mae"] for r in rows if r["method"] == m and r["activity"] == act)  # noqa: E731
        w3 = g("+mask+tracker+bound+rankreset")
        verdict = "beats b0" if w3 < b0[act] else "STILL WORSE THAN b0"
        print(f"{act:<15}{b0[act]:>8.2f}{g('b2-zp'):>9.2f}{g('+mask+tracker'):>10.2f}{w3:>9.2f}{verdict:>22}")
        b0_rows.append(dict(activity=act, b0=b0[act], b2_zp=g("b2-zp"),
                            mask_tracker=g("+mask+tracker"), week3_best=w3,
                            beats_b0=bool(w3 < b0[act]),
                            mask_tracker_beat_b0=bool(g("+mask+tracker") < b0[act])))

    print(f"\nDiagnostic A, rerun on the new residual errors")
    from src.eval.diagnostics import PEAK_TOL_BPM
    from src.models.tracker import peak_candidates as _pc
    surv_rows = []
    print(f"{'activity':<15}{'err windows':>13}{'survives':>10}")
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        surv = []
        for s in subs:
            m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
            m = m & (np.abs(best[s["sid"]] - s["hr"]) > DIAG_ERR)
            if not m.any():
                continue
            masked = s["spec"] * s["gains"][mt_chosen[s["sid"]].split("|")[0]]
            cand = _pc(masked, s["f_bpm"], 0.05)
            for i in np.flatnonzero(m):
                c = cand[i]
                surv.append(bool(len(c) and (np.abs(c - s["hr"][i]) <= PEAK_TOL_BPM).any()))
        if not surv:
            continue
        print(f"{name:<15}{len(surv):>13}{np.mean(surv):>10.1%}")
        surv_rows.append(dict(activity=name, n_error_windows=len(surv),
                              survives=round(float(np.mean(surv)), 4)))

    for data, fn in ((persist_rows, "week3_persistence.csv"), (b0_rows, "week3_vs_b0.csv"),
                     (surv_rows, "week3_surviving_peak.csv")):
        with open(res / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote results/{fn}")

    np.save(res / "week3_best_predictions.npy",
            np.array([preds["+mask+tracker+bound+rankreset"][s["sid"]] for s in subs], dtype=object),
            allow_pickle=True)


if __name__ == "__main__":
    main()
