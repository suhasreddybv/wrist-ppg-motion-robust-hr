"""Mechanistic check for Stage 4a: does masking remove the failure it targets?

An MAE change is not evidence that masking worked. If masking does what it claims,
the `acc_locked` share of error windows must fall - most visibly in walking, where
it is 55.1% before masking. If MAE improves while `acc_locked` does not fall, the
gain came from somewhere else and is reported as such.

Mask configurations are selected the same way as in the ablation: inside each LOSO
fold, on training subjects only.

Writes results/masked_taxonomy.csv.
"""
from __future__ import annotations

import csv

import numpy as np

from src.data.loader import REPO_ROOT, SUBJECT_IDS, load_subject
from src.eval.error_taxonomy import ERROR_BPM, LABELS, classify, lowest_in_band_bpm
from src.eval.loso import ACTIVITY_NAMES
from src.eval.plot_motion_collision import acc_dominant_bpm
from src.eval.stage4 import MASK_GRID, fold_select
from src.features.preprocess import preprocess_subject
from src.models.masking import band_spectra, mask_gains


def main() -> None:
    floor = lowest_in_band_bpm()
    subs = []
    for sid in SUBJECT_IDS:
        pre = preprocess_subject(load_subject(sid))
        f_hz, spec = band_spectra(pre.bvp_filtered, pre.fs)
        f_bpm = f_hz * 60.0
        rec = dict(sid=sid, hr=pre.windows.hr, activity=pre.windows.activity,
                   acc=acc_dominant_bpm(pre.acc_64, pre.fs),
                   base=f_bpm[spec.argmax(axis=1)], mask={})
        for cfg in MASK_GRID:
            g, _ = mask_gains(pre.acc_64, pre.fs, f_hz, cfg)
            rec["mask"][cfg.label()] = f_bpm[(spec * g).argmax(axis=1)]
        subs.append(rec)
        print(f"  {sid} done")

    preds, chosen, _ = fold_select(subs, "mask")

    rows = []
    print(f"\n{'activity':<15}{'base err win':>13}{'masked err win':>16}"
          + "".join(f"{l + ' before/after':>26}" for l in LABELS))
    for act, name in list(ACTIVITY_NAMES.items()) + [(None, "ALL (excl. transient)")]:
        b_lab, m_lab = [], []
        for s in subs:
            m = (s["activity"] == act) if act is not None else (s["activity"] != 0)
            be = np.abs(s["base"] - s["hr"]) > ERROR_BPM
            me = np.abs(preds[s["sid"]] - s["hr"]) > ERROR_BPM
            lb, _ = classify(s["base"], s["hr"], s["acc"], floor)
            lm, _ = classify(preds[s["sid"]], s["hr"], s["acc"], floor)
            b_lab.append(lb[m & be])
            m_lab.append(lm[m & me])
        b_lab, m_lab = np.concatenate(b_lab), np.concatenate(m_lab)
        line = f"{name:<15}{len(b_lab):>13}{len(m_lab):>16}"
        for l in LABELS:
            before = float(np.mean(b_lab == l)) if len(b_lab) else float("nan")
            after = float(np.mean(m_lab == l)) if len(m_lab) else float("nan")
            line += f"{before:>12.1%}{after:>13.1%}"
            rows.append(dict(activity=name, label=l, n_error_windows_before=len(b_lab),
                             n_error_windows_after=len(m_lab),
                             share_before=round(before, 4), share_after=round(after, 4),
                             share_change=round(after - before, 4),
                             count_before=int(np.sum(b_lab == l)), count_after=int(np.sum(m_lab == l))))
        print(line)

    with open(REPO_ROOT / "results" / "masked_taxonomy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote results/masked_taxonomy.csv")

    walk = next(r for r in rows if r["activity"] == "walking" and r["label"] == "acc_locked")
    allr = next(r for r in rows if r["activity"].startswith("ALL") and r["label"] == "acc_locked")
    print(f"\nacc_locked share, walking: {walk['share_before']:.1%} -> {walk['share_after']:.1%} "
          f"({walk['count_before']} -> {walk['count_after']} windows)")
    print(f"acc_locked share, all:     {allr['share_before']:.1%} -> {allr['share_after']:.1%}")
    if allr["share_change"] >= 0:
        print("The targeted failure did NOT fall. Any MAE change came from somewhere else.")


if __name__ == "__main__":
    main()
