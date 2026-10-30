"""Accuracy of the frozen pipeline under each edge variant, LOSO, per subject and per activity,
with paired differences against the reference and against causal float64 (bootstrap over the
15 subjects, 2,000 resamples, seed 42 - the repository's convention).

Writes results/edge_variants_per_subject.csv, edge_variants_per_activity.csv, edge_variants_summary.csv.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from src.data.loader import REPO_ROOT, SUBJECT_IDS, load_subject
from src.edge.pipeline import VARIANTS, bvp_full_scale, fold_configs, run_subject
from src.eval.loso import ACTIVITY_NAMES
from src.eval.metrics import mae, mape

SEED, N_BOOT = 42, 2000


def paired(d: np.ndarray, rng) -> tuple[float, float, float]:
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def write(rows, name):
    with open(REPO_ROOT / "results" / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote results/{name}")


def main() -> None:
    configs = fold_configs()
    p999 = {s: float(np.percentile(np.abs(load_subject(s)["wrist_bvp"].data), 99.9)) for s in SUBJECT_IDS}
    per_subject, per_act, preds = [], [], {}
    for v in VARIANTS:
        preds[v.name] = {}
        for sid in SUBJECT_IDS:
            fs = bvp_full_scale([s for s in SUBJECT_IDS if s != sid], p999)
            r = run_subject(sid, v, configs, fs)
            preds[v.name][sid] = r
            per_subject.append(dict(variant=v.name, subject=sid, windows=len(r["hr"]), full_scale=round(fs, 1),
                                    mae=round(mae(r["hr"], r["pred"]), 4), mape=round(mape(r["hr"], r["pred"]), 4)))
        pooled = np.mean([x["mae"] for x in per_subject if x["variant"] == v.name])
        print(f"  {v.name:<12} pooled MAE {pooled:.3f}", flush=True)
        for act, name in ACTIVITY_NAMES.items():
            vals = [mae(r["hr"][r["activity"] == act], r["pred"][r["activity"] == act]) for r in preds[v.name].values() if (r["activity"] == act).any()]
            per_act.append(dict(variant=v.name, activity=name, folds=len(vals), mae_mean_of_folds=round(float(np.mean(vals)), 3)))
    rng = np.random.default_rng(SEED)
    m = {(x["variant"], x["subject"]): x["mae"] for x in per_subject}
    summary = []
    for v in VARIANTS:
        own = np.array([m[(v.name, s)] for s in SUBJECT_IDS])
        row = dict(variant=v.name, pooled_mae=round(float(own.mean()), 3),
                   pooled_mape=round(float(np.mean([x["mape"] for x in per_subject if x["variant"] == v.name])), 3))
        for base in ("reference", "causal_f64"):
            d = own - np.array([m[(base, s)] for s in SUBJECT_IDS])          # positive: the variant is worse
            mean, lo, hi = paired(d, rng)
            row.update({f"minus_{base}": round(mean, 3), f"minus_{base}_ci_lo": round(lo, 3), f"minus_{base}_ci_hi": round(hi, 3),
                        f"subjects_worse_than_{base}": int((d > 1e-9).sum())})
        summary.append(row)
        print(f"  {v.name:<12} {row['pooled_mae']:.3f}  vs reference {row['minus_reference']:+.3f} [{row['minus_reference_ci_lo']:+.3f}, {row['minus_reference_ci_hi']:+.3f}]"
              f"  vs causal_f64 {row['minus_causal_f64']:+.3f} [{row['minus_causal_f64_ci_lo']:+.3f}, {row['minus_causal_f64_ci_hi']:+.3f}]")
    write(per_subject, "edge_variants_per_subject.csv"); write(per_act, "edge_variants_per_activity.csv"); write(summary, "edge_variants_summary.csv")


if __name__ == "__main__":
    sys.exit(main())
