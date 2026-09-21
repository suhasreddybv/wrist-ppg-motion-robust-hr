"""Leave-one-subject-out evaluation of the Stage 3 baselines.

15 folds, one per subject. Anything a method needs beyond the test subject's own
signal (b0's constant, b1's first-window fallback) is computed on the training
subjects of that fold only.

Reporting rules:
- Two aggregations are reported side by side and never mixed:
  `mae_mean_of_folds` averages per-subject MAE across folds (each subject counts
  once), `mae_pooled` pools every window (long recordings count more). The same
  applies to MAPE and RMSE.
- Every cell carries its fold count: S6 has no activities 6-8, so those rows have
  14 folds, not 15.
- Activity 0 (transient) is excluded from per-activity rows. Pooled is reported
  both including and excluding transients, which are 27% of all windows.
- Windows are flagged when wrist ACC reaches +-2 g, with the continuous fraction
  kept alongside. `b2_noclip` scores b2 on unflagged windows only; note that b2
  does not use the accelerometer, so this comparison cannot attribute error to
  clipping. It becomes meaningful for the ACC-referenced methods in Stage 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.loader import SUBJECT_IDS, load_subject
from src.eval.metrics import mae, mape, rmse
from src.features.preprocess import preprocess_subject
from src.models.baselines import ZERO_PAD_NFFT, global_mean_hr, previous_window_hr, spectral_peak_hr

ACTIVITY_NAMES = {
    1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling",
    5: "driving", 6: "lunch", 7: "walking", 8: "working",
}
TRANSIENT = 0
POOLED_ALL = "POOLED (incl. transient)"
POOLED_NO_TRANSIENT = "POOLED (excl. transient)"
METHOD_LABELS = {
    "b0": "global mean HR (training subjects)",
    "b1": "previous window ground truth (ORACLE, not deployable)",
    "b2": "naive spectral peak, no motion handling",
    "b2_zp": f"naive spectral peak, zero-padded to nfft={ZERO_PAD_NFFT}",
    "b2_noclip": "naive spectral peak, ACC-clipped windows excluded",
}
BASELINE_FOR_STAGE4 = "b2_zp"   # Stage 4 gains are reported against this, not b2


@dataclass(frozen=True)
class SubjectArrays:
    subject_id: str
    hr: np.ndarray
    activity: np.ndarray
    clipped: np.ndarray
    b2: np.ndarray
    b2_zp: np.ndarray | None = None
    clip_fraction: np.ndarray | None = None


def build_subject_arrays(subject_ids=SUBJECT_IDS) -> list[SubjectArrays]:
    """Per-subject ground truth and the fold-independent spectral-peak predictions."""
    out = []
    for sid in subject_ids:
        pre = preprocess_subject(load_subject(sid))
        out.append(SubjectArrays(
            subject_id=sid,
            hr=pre.windows.hr,
            activity=pre.windows.activity,
            clipped=pre.windows.acc_clipped,
            b2=spectral_peak_hr(pre.bvp_filtered, fs=pre.fs, band=pre.band_hz),
            b2_zp=spectral_peak_hr(pre.bvp_filtered, fs=pre.fs, band=pre.band_hz, nfft=ZERO_PAD_NFFT),
            clip_fraction=pre.windows.clip_fraction,
        ))
    return out


def loso_predictions(subjects: list[SubjectArrays]) -> dict[str, dict[str, np.ndarray]]:
    """predictions[subject_id][method] for every window of every fold."""
    preds: dict[str, dict[str, np.ndarray]] = {}
    for held_out in subjects:
        train_hr = np.concatenate([s.hr for s in subjects if s.subject_id != held_out.subject_id])
        b0 = global_mean_hr(train_hr)
        zp = held_out.b2_zp if held_out.b2_zp is not None else held_out.b2
        preds[held_out.subject_id] = {
            "b0": np.full_like(held_out.hr, b0),
            "b1": previous_window_hr(held_out.hr, fallback=b0),
            "b2": held_out.b2,
            "b2_zp": zp,
            "b2_noclip": held_out.b2,   # same estimator; clipped windows dropped when scoring
        }
    return preds


def _mask(subj: SubjectArrays, method: str, activity: int | None,
          drop_transient: bool = False) -> np.ndarray:
    m = np.ones(len(subj.hr), dtype=bool) if activity is None else subj.activity == activity
    if drop_transient:
        m = m & (subj.activity != TRANSIENT)
    if method == "b2_noclip":
        m = m & ~subj.clipped
    return m


def per_subject_rows(subjects, preds) -> list[dict]:
    """One row per method, subject and scope ('all' includes transients)."""
    rows = []
    for subj in subjects:
        for method, pred in preds[subj.subject_id].items():
            for scope, drop in (("all", False), ("no_transient", True)):
                m = _mask(subj, method, None, drop_transient=drop)
                rows.append(dict(
                    method=method, subject=subj.subject_id, scope=scope,
                    n_windows=int(m.sum()),
                    mae=round(mae(subj.hr[m], pred[m]), 3),
                    mape=round(mape(subj.hr[m], pred[m]), 3),
                    rmse=round(rmse(subj.hr[m], pred[m]), 3),
                    clipped_windows=int(subj.clipped.sum()),
                    mean_clip_fraction=round(float(np.mean(subj.clip_fraction)), 5)
                    if subj.clip_fraction is not None else None,
                    oracle=method == "b1",
                ))
    return rows


def per_activity_rows(subjects, preds) -> list[dict]:
    """Per activity and pooled, with fold-mean and window-pooled aggregations side by side."""
    scopes = list(ACTIVITY_NAMES.items()) + [(None, POOLED_ALL), ("no_transient", POOLED_NO_TRANSIENT)]
    rows = []
    for method in METHOD_LABELS:
        for key, name in scopes:
            activity = None if key in (None, "no_transient") else key
            drop = key == "no_transient"
            maes, mapes, rmses, truth, pred_all = [], [], [], [], []
            for subj in subjects:
                m = _mask(subj, method, activity, drop_transient=drop)
                if not m.any():
                    continue          # S6 has no activities 6-8
                pred = preds[subj.subject_id][method]
                maes.append(mae(subj.hr[m], pred[m]))
                mapes.append(mape(subj.hr[m], pred[m]))
                rmses.append(rmse(subj.hr[m], pred[m]))
                truth.append(subj.hr[m])
                pred_all.append(pred[m])
            if not maes:
                continue   # no subject in this set performed the activity; emit no row
            t, p = np.concatenate(truth), np.concatenate(pred_all)
            rows.append(dict(
                method=method, activity=name, n_folds=len(maes), n_windows=len(t),
                mae_mean_of_folds=round(float(np.mean(maes)), 3),
                mae_pooled=round(mae(t, p), 3),
                mae_sd_across_folds=round(float(np.std(maes, ddof=1)), 3),
                mae_worst_fold=round(float(np.max(maes)), 3),
                mape_mean_of_folds=round(float(np.mean(mapes)), 3),
                mape_pooled=round(mape(t, p), 3),
                mape_sd_across_folds=round(float(np.std(mapes, ddof=1)), 3),
                rmse_mean_of_folds=round(float(np.mean(rmses)), 3),
                rmse_pooled=round(rmse(t, p), 3),
                oracle=method == "b1",
            ))
    return rows


def clip_fraction_rows(subjects) -> list[dict]:
    """Distribution of ACC clipping by activity: this will distort ACC-referenced methods."""
    rows = []
    for activity, name in ACTIVITY_NAMES.items():
        fracs, flagged, n = [], 0, 0
        for subj in subjects:
            if subj.clip_fraction is None:
                continue
            m = subj.activity == activity
            if not m.any():
                continue
            fracs.append(subj.clip_fraction[m])
            flagged += int(subj.clipped[m].sum())
            n += int(m.sum())
        if not fracs:
            continue
        f = np.concatenate(fracs)
        rows.append(dict(
            activity=name, n_windows=n,
            windows_flagged_pct=round(flagged / n * 100, 2),
            mean_clip_fraction_pct=round(float(f.mean()) * 100, 3),
            p95_clip_fraction_pct=round(float(np.percentile(f, 95)) * 100, 3),
            max_clip_fraction_pct=round(float(f.max()) * 100, 3),
        ))
    return rows
