"""Stage 4 ablation: masking, tracking, and adaptive cancellation, LOSO.

Every gain is measured against b2-zp. Hyperparameters are chosen inside each fold:
for held-out subject i, the configuration minimising the mean per-subject MAE over
the other 14 is selected and applied to subject i. The full grid is also scored
globally, so a gain that only exists when the configuration is chosen on the test
data shows up as the difference between the two.

Variants share one spectral grid, so the only difference between rows is the
component under test.

Writes results/stage4_ablation.csv, results/stage4_per_subject.csv,
results/stage4_selected_configs.csv and figures/stage4_ablation.png.
"""
from __future__ import annotations

import csv
import itertools
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, SUBJECT_IDS, load_subject  # noqa: E402
from src.eval.loso import ACTIVITY_NAMES, POOLED_ALL, POOLED_NO_TRANSIENT  # noqa: E402
from src.eval.metrics import mae, mape, rmse  # noqa: E402
from src.features.preprocess import preprocess_subject  # noqa: E402
from src.models.adaptive import AdaptiveConfig, cancel  # noqa: E402
from src.models.masking import MaskConfig, band_spectra, mask_gains  # noqa: E402
from src.models.tracker import TrackerConfig, peak_candidates, track  # noqa: E402

# Notch depth is a POWER gain, and a cadence line can carry 16x the power of the cardiac
# peak, so a depth of 0.1 still leaves motion dominant. The grid therefore spans deep
# (0.02) to gentle (0.2) notches and lets the fold choose.
MASK_GRID = [MaskConfig(n_peaks=k, prominence=p, width_factor=w, depth=d)
             for k, p, w, d in itertools.product((1, 3), (0.1,), (0.5, 2.0), (0.02, 0.2))]
# Two tracker modes: a hard search window, and SpaMaPlus-like nearest-peak selection.
TRACK_GRID = ([TrackerConfig(mode="window", half_width_bpm=h, history=6, jump_bpm=j, reset_after=r)
               for h, j, r in itertools.product((12.0, 20.0), (15.0,), (2, 3))]
              + [TrackerConfig(mode="nearest_peak", history=6, jump_bpm=j, reset_after=r)
                 for j, r in itertools.product((10.0, 15.0, 25.0), (2, 3))])
ADAPT_GRID = [AdaptiveConfig(method=m, order=o, mu=mu)
              for m, o, mu in [("nlms", 8, 0.1), ("nlms", 8, 0.05), ("nlms", 16, 0.1), ("ls", 8, 0.0)]]


def config_from_label(label: str):
    """Recover the MaskConfig / TrackerConfig a fold selected, from its recorded label."""
    for cfg in MASK_GRID + TRACK_GRID:
        if cfg.label() == label:
            return cfg
    raise KeyError(f"no configuration in the grids matches {label!r}")


def _estimates(spec: np.ndarray, f_bpm: np.ndarray) -> np.ndarray:
    return f_bpm[spec.argmax(axis=1)]


def build(run_adaptive: bool) -> list[dict]:
    """Per-subject predictions for every configuration of every variant."""
    subs = []
    for sid in SUBJECT_IDS:
        t0 = time.time()
        pre = preprocess_subject(load_subject(sid))
        f_hz, spec = band_spectra(pre.bvp_filtered, pre.fs)
        f_bpm = f_hz * 60.0
        rec = dict(sid=sid, hr=pre.windows.hr, activity=pre.windows.activity,
                   clip_fraction=pre.windows.clip_fraction, f_bpm=f_bpm,
                   base=_estimates(spec, f_bpm))

        # masking: gains depend only on the accelerometer, so they are reused below
        rec["mask"], rec["mask_energy"], gains = {}, {}, {}
        for cfg in MASK_GRID:
            g, _ = mask_gains(pre.acc_64, pre.fs, f_hz, cfg)
            masked = spec * g
            gains[cfg.label()] = g
            rec["mask"][cfg.label()] = _estimates(masked, f_bpm)
            total = spec.sum(axis=1)
            rec["mask_energy"][cfg.label()] = float(np.mean(1.0 - masked.sum(axis=1) / np.where(total > 0, total, 1)))

        # tracking on the plain spectrum, and on each masked spectrum
        cand = peak_candidates(spec, f_bpm, TRACK_GRID[-1].prominence)
        rec["track"] = {c.label(): track(spec, f_bpm, c, cand) for c in TRACK_GRID}
        rec["mask_track"] = {}
        for mcfg in MASK_GRID:
            masked = spec * gains[mcfg.label()]
            mcand = peak_candidates(masked, f_bpm, TRACK_GRID[-1].prominence)
            for tcfg in TRACK_GRID:
                rec["mask_track"][f"{mcfg.label()}|{tcfg.label()}"] = track(masked, f_bpm, tcfg, mcand)

        if run_adaptive:
            rec["adapt"], rec["adapt_track"] = {}, {}
            for acfg in ADAPT_GRID:
                res = cancel(pre.bvp_filtered, pre.acc_64, acfg)
                _, aspec = band_spectra(res, pre.fs)
                rec["adapt"][acfg.label()] = _estimates(aspec, f_bpm)
                acand = peak_candidates(aspec, f_bpm, TRACK_GRID[-1].prominence)
                for tcfg in TRACK_GRID:
                    rec["adapt_track"][f"{acfg.label()}|{tcfg.label()}"] = track(aspec, f_bpm, tcfg, acand)
        subs.append(rec)
        print(f"  {sid}: {len(rec['hr'])} windows, {time.time() - t0:.1f}s")
    return subs


def fold_select(subs: list[dict], variant: str) -> tuple[np.ndarray, dict, dict]:
    """LOSO predictions with the configuration chosen on training subjects only.

    Returns (per-subject predictions, chosen config per fold, globally best config).
    """
    keys = list(subs[0][variant])
    train_mae = {k: {s["sid"]: mae(s["hr"], s[variant][k]) for s in subs} for k in keys}
    chosen, preds = {}, {}
    for held in subs:
        scores = {k: np.mean([v for sid, v in train_mae[k].items() if sid != held["sid"]]) for k in keys}
        best = min(scores, key=scores.get)
        chosen[held["sid"]] = best
        preds[held["sid"]] = held[variant][best]
    global_best = min(keys, key=lambda k: np.mean(list(train_mae[k].values())))
    return preds, chosen, global_best


def rows_for(subs, preds_by_sid, method: str) -> list[dict]:
    out = []
    scopes = list(ACTIVITY_NAMES.items()) + [(None, POOLED_ALL), ("no_transient", POOLED_NO_TRANSIENT)]
    for key, name in scopes:
        maes, mapes, rmses = [], [], []
        n_win = 0
        for s in subs:
            if key is None:
                m = np.ones(len(s["hr"]), bool)
            elif key == "no_transient":
                m = s["activity"] != 0
            else:
                m = s["activity"] == key
            if not m.any():
                continue
            p = preds_by_sid[s["sid"]]
            maes.append(mae(s["hr"][m], p[m]))
            mapes.append(mape(s["hr"][m], p[m]))
            rmses.append(rmse(s["hr"][m], p[m]))
            n_win += int(m.sum())
        if not maes:
            continue
        out.append(dict(method=method, activity=name, n_folds=len(maes), n_windows=n_win,
                        mae=round(float(np.mean(maes)), 3),
                        mae_sd_across_folds=round(float(np.std(maes, ddof=1)), 3),
                        mae_worst_fold=round(float(np.max(maes)), 3),
                        mape=round(float(np.mean(mapes)), 3),
                        rmse=round(float(np.mean(rmses)), 3)))
    return out


def main(run_adaptive: bool = True) -> None:
    print("building per-subject predictions for every configuration...")
    subs = build(run_adaptive)

    variants = [("+tracker", "track"), ("+mask", "mask"), ("+mask+tracker", "mask_track")]
    if run_adaptive:
        variants += [("+adaptive", "adapt"), ("+adaptive+tracker", "adapt_track")]

    rows = rows_for(subs, {s["sid"]: s["base"] for s in subs}, "b2-zp")
    per_subject = [dict(method="b2-zp", subject=s["sid"], config="-",
                        mae=round(mae(s["hr"], s["base"]), 3),
                        mape=round(mape(s["hr"], s["base"]), 3)) for s in subs]
    selected, leak_rows = [], []

    for label, variant in variants:
        preds, chosen, global_best = fold_select(subs, variant)
        rows += rows_for(subs, preds, label)
        for s in subs:
            per_subject.append(dict(method=label, subject=s["sid"], config=chosen[s["sid"]],
                                    mae=round(mae(s["hr"], preds[s["sid"]]), 3),
                                    mape=round(mape(s["hr"], preds[s["sid"]]), 3)))
        n_distinct = len(set(chosen.values()))
        selected.append(dict(method=label, n_distinct_configs=n_distinct,
                             global_best=global_best,
                             folds_agreeing_with_global=sum(c == global_best for c in chosen.values())))
        # the same variant scored with the globally best configuration, i.e. chosen on all 15
        global_preds = {s["sid"]: s[variant][global_best] for s in subs}
        in_fold = np.mean([mae(s["hr"], preds[s["sid"]]) for s in subs])
        global_fit = np.mean([mae(s["hr"], global_preds[s["sid"]]) for s in subs])
        leak_rows.append(dict(method=label, mae_config_chosen_in_fold=round(float(in_fold), 3),
                              mae_config_chosen_globally=round(float(global_fit), 3),
                              optimism=round(float(in_fold - global_fit), 3)))

    # masked energy of the selected mask configurations
    mask_preds, mask_chosen, _ = fold_select(subs, "mask")
    energy = float(np.mean([s["mask_energy"][mask_chosen[s["sid"]]] for s in subs]))

    print(f"\n{'method':<20}{'folds':>6}{'windows':>9}{'MAE':>8}{'SD':>7}{'worst':>8}{'MAPE':>8}")
    for r in rows:
        if r["activity"] in (POOLED_ALL, POOLED_NO_TRANSIENT):
            print(f"{r['method'] + ' ' + ('[all]' if r['activity'] == POOLED_ALL else '[no trans]'):<20}"
                  f"{r['n_folds']:>6}{r['n_windows']:>9}{r['mae']:>8.2f}{r['mae_sd_across_folds']:>7.2f}"
                  f"{r['mae_worst_fold']:>8.2f}{r['mape']:>8.2f}")

    print(f"\nper-activity MAE\n{'activity':<15}" + "".join(f"{m:>16}" for m, _ in [("b2-zp", 0)] + variants))
    for act in ACTIVITY_NAMES.values():
        line = f"{act:<15}"
        for m, _ in [("b2-zp", 0)] + variants:
            r = next((x for x in rows if x["method"] == m and x["activity"] == act), None)
            line += f"{r['mae']:>16.2f}" if r else f"{'-':>16}"
        print(line)

    print(f"\nconfiguration selection (in-fold vs global):")
    for lr, se in zip(leak_rows, selected):
        print(f"  {lr['method']:<20} in-fold {lr['mae_config_chosen_in_fold']:6.2f} | "
              f"global {lr['mae_config_chosen_globally']:6.2f} | optimism {lr['optimism']:+5.2f} | "
              f"{se['n_distinct_configs']} distinct configs, {se['folds_agreeing_with_global']}/15 match global")
    print(f"\nmean masked energy of selected mask configs: {energy:.1%}"
          + ("   FLAG: above 30%" if energy > 0.30 else ""))

    res = REPO_ROOT / "results"
    for name, data in [("stage4_ablation.csv", rows), ("stage4_per_subject.csv", per_subject),
                       ("stage4_selected_configs.csv", [dict(s, **l) for s, l in zip(selected, leak_rows)])]:
        with open(res / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote results/{name}")

    # figure: per-activity MAE by method
    acts = list(ACTIVITY_NAMES.values())
    methods = ["b2-zp"] + [m for m, _ in variants]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(methods):
        vals = [next((x["mae"] for x in rows if x["method"] == m and x["activity"] == a), np.nan) for a in acts]
        ax.bar(np.arange(len(acts)) + i * width, vals, width, label=m)
    ax.set_xticks(np.arange(len(acts)) + 0.4 - width / 2, acts, rotation=45, ha="right")
    ax.set_ylabel("MAE (bpm), mean of per-fold MAEs")
    ax.set_title("Stage 4 ablation, leave-one-subject-out")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "stage4_ablation.png", dpi=130)
    print("wrote figures/stage4_ablation.png")


if __name__ == "__main__":
    import sys
    main(run_adaptive="--no-adaptive" not in sys.argv)
