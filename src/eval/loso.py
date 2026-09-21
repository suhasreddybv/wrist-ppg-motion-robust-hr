"""Leave-one-subject-out evaluation of the Stage 3 baselines.

15 folds, one per subject. Anything a method needs beyond the test subject's own
signal (b0's constant, b1's first-window fallback) is computed on the training
subjects of that fold only.

Reporting rules:
- Per-activity results average per-subject MAE across folds, so a long recording
  does not outweigh a short one. Every cell carries its fold count: S6 has no
  activities 6-8, so those rows have 14 folds, not 15.
- Activity 0 (transient) is excluded from per-activity rows and included in pooled.
- Windows where wrist ACC reaches +-2 g are flagged, and b2 is additionally
  reported with them excluded (`b2_noclip`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.loader import SUBJECT_IDS, load_subject
from src.eval.metrics import mae, rmse
from src.features.preprocess import preprocess_subject
from src.models.baselines import global_mean_hr, previous_window_hr, spectral_peak_hr

ACTIVITY_NAMES = {
    1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling",
    5: "driving", 6: "lunch", 7: "walking", 8: "working",
}
TRANSIENT = 0
METHOD_LABELS = {
    "b0": "global mean HR (training subjects)",
    "b1": "previous window ground truth (ORACLE, not deployable)",
    "b2": "naive spectral peak, no motion handling",
    "b2_noclip": "naive spectral peak, ACC-clipped windows excluded",
}


@dataclass(frozen=True)
class SubjectArrays:
    subject_id: str
    hr: np.ndarray
    activity: np.ndarray
    clipped: np.ndarray
    b2: np.ndarray


def build_subject_arrays(subject_ids=SUBJECT_IDS) -> list[SubjectArrays]:
    """Per-subject ground truth and the fold-independent b2 prediction."""
    out = []
    for sid in subject_ids:
        pre = preprocess_subject(load_subject(sid))
        out.append(SubjectArrays(
            subject_id=sid,
            hr=pre.windows.hr,
            activity=pre.windows.activity,
            clipped=pre.windows.acc_clipped,
            b2=spectral_peak_hr(pre.bvp_filtered, fs=pre.fs, band=pre.band_hz),
        ))
    return out


def loso_predictions(subjects: list[SubjectArrays]) -> dict[str, dict[str, np.ndarray]]:
    """predictions[subject_id][method] for every window of every fold."""
    preds: dict[str, dict[str, np.ndarray]] = {}
    for held_out in subjects:
        train_hr = np.concatenate([s.hr for s in subjects if s.subject_id != held_out.subject_id])
        b0 = global_mean_hr(train_hr)
        preds[held_out.subject_id] = {
            "b0": np.full_like(held_out.hr, b0),
            "b1": previous_window_hr(held_out.hr, fallback=b0),
            "b2": held_out.b2,
            "b2_noclip": held_out.b2,   # same estimator; clipped windows dropped when scoring
        }
    return preds


def _mask(subj: SubjectArrays, method: str, activity: int | None) -> np.ndarray:
    m = np.ones(len(subj.hr), dtype=bool) if activity is None else subj.activity == activity
    if method == "b2_noclip":
        m = m & ~subj.clipped
    return m


def per_subject_rows(subjects, preds) -> list[dict]:
    """Pooled over all windows of a subject, transients included."""
    rows = []
    for subj in subjects:
        for method, pred in preds[subj.subject_id].items():
            m = _mask(subj, method, None)
            rows.append(dict(
                method=method, subject=subj.subject_id, n_windows=int(m.sum()),
                mae=round(mae(subj.hr[m], pred[m]), 3),
                rmse=round(rmse(subj.hr[m], pred[m]), 3),
                clipped_windows=int(subj.clipped.sum()),
                oracle=method == "b1",
            ))
    return rows


def per_activity_rows(subjects, preds) -> list[dict]:
    """Mean of per-subject MAE across folds, per activity, plus a pooled row."""
    rows = []
    for method in METHOD_LABELS:
        for activity, name in list(ACTIVITY_NAMES.items()) + [(None, "POOLED (incl. transient)")]:
            maes, rmses, n_win = [], [], 0
            for subj in subjects:
                m = _mask(subj, method, activity)
                if not m.any():
                    continue          # S6 has no activities 6-8
                pred = preds[subj.subject_id][method]
                maes.append(mae(subj.hr[m], pred[m]))
                rmses.append(rmse(subj.hr[m], pred[m]))
                n_win += int(m.sum())
            if not maes:
                continue   # no subject in this set performed the activity; emit no row
            rows.append(dict(
                method=method, activity=name, n_folds=len(maes), n_windows=n_win,
                mae=round(float(np.mean(maes)), 3),
                mae_sd_across_folds=round(float(np.std(maes, ddof=1)), 3),
                mae_worst_fold=round(float(np.max(maes)), 3),
                rmse=round(float(np.mean(rmses)), 3),
                oracle=method == "b1",
            ))
    return rows
