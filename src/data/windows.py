"""Stage 1 - 8 s windows at a 2 s shift, aligned to the provided HR labels.

Window i covers [2i, 2i + 8) s, so at 64 Hz the BVP slice is 512 samples and the
32 Hz wrist ACC slice is 256.

The alignment assertion matters more than anything else here: the number of
windows must equal len(label) exactly, for every subject. An off-by-one shifts
every result downstream and is invisible in the metrics, so it fails loudly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.loader import SHIFT_S, WINDOW_S, SubjectRecord

ACTIVITY_RATE = 4
TRANSIENT = 0


class WindowAlignmentError(AssertionError):
    """Raised when windowing does not line up with the label array."""


@dataclass(frozen=True)
class WindowedSubject:
    """All windows of one subject, stacked. Row i of every array is window i."""

    subject_id: str
    bvp: np.ndarray          # (n_windows, WINDOW_S * 64)
    acc: np.ndarray          # (n_windows, WINDOW_S * 32, 3), g
    hr: np.ndarray           # (n_windows,) ground-truth bpm
    activity: np.ndarray     # (n_windows,) modal activity id over the window
    start_s: np.ndarray      # (n_windows,) window start time in seconds
    skin_type: int
    bvp_fs: int = 64
    acc_fs: int = 32

    def __len__(self) -> int:
        return len(self.hr)

    @property
    def is_transient(self) -> np.ndarray:
        return self.activity == TRANSIENT


@dataclass(frozen=True)
class Window:
    subject_id: str
    index: int
    start_s: float
    bvp: np.ndarray
    acc: np.ndarray
    hr: float
    activity: int
    skin_type: int


def n_windows_for(duration_s: float) -> int:
    return int((duration_s - WINDOW_S) // SHIFT_S) + 1


def _strided(x: np.ndarray, per_window: int, step: int, n: int) -> np.ndarray:
    """n non-copying rows of length per_window, starting every step samples."""
    if x.ndim == 1:
        shape, strides = (n, per_window), (step * x.strides[0], x.strides[0])
    else:  # (samples, channels)
        shape = (n, per_window, x.shape[1])
        strides = (step * x.strides[0], x.strides[0], x.strides[1])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)


def window_array(x: np.ndarray, fs: int, n: int, copy: bool = True) -> np.ndarray:
    """Window an arbitrary signal on the same grid: row i covers [2i, 2i + 8) s."""
    w = _strided(x, WINDOW_S * fs, SHIFT_S * fs, n)
    return np.ascontiguousarray(w) if copy else w


def window_subject(rec: SubjectRecord, copy: bool = True) -> WindowedSubject:
    """Cut one subject into label-aligned windows."""
    bvp, acc = rec["wrist_bvp"], rec["wrist_acc"]
    n = n_windows_for(rec.duration_s)
    if n != len(rec.label):
        raise WindowAlignmentError(
            f"{rec.subject_id}: {n} windows from {rec.duration_s} s but {len(rec.label)} labels"
        )

    per_bvp, step_bvp = WINDOW_S * bvp.fs, SHIFT_S * bvp.fs
    per_acc, step_acc = WINDOW_S * acc.fs, SHIFT_S * acc.fs
    per_act, step_act = WINDOW_S * ACTIVITY_RATE, SHIFT_S * ACTIVITY_RATE

    need_bvp = (n - 1) * step_bvp + per_bvp
    need_acc = (n - 1) * step_acc + per_acc
    need_act = (n - 1) * step_act + per_act
    if len(bvp.data) < need_bvp or len(acc.data) < need_acc or len(rec.activity) < need_act:
        raise WindowAlignmentError(
            f"{rec.subject_id}: signals too short for {n} windows "
            f"(BVP {len(bvp.data)}/{need_bvp}, ACC {len(acc.data)}/{need_acc}, "
            f"activity {len(rec.activity)}/{need_act})"
        )

    bvp_w = _strided(bvp.data, per_bvp, step_bvp, n)
    acc_w = _strided(acc.data, per_acc, step_acc, n)
    act_w = _strided(rec.activity, per_act, step_act, n)

    # Modal activity id over the window; ties resolve to the smaller id.
    counts = np.zeros((n, 9), dtype=np.int32)
    for cls in range(9):
        counts[:, cls] = (act_w == cls).sum(axis=1)
    activity = counts.argmax(axis=1).astype(np.int8)

    return WindowedSubject(
        subject_id=rec.subject_id,
        bvp=np.ascontiguousarray(bvp_w) if copy else bvp_w,
        acc=np.ascontiguousarray(acc_w) if copy else acc_w,
        hr=np.asarray(rec.label, dtype=float),
        activity=activity,
        start_s=np.arange(n, dtype=float) * SHIFT_S,
        skin_type=rec.skin_type,
        bvp_fs=bvp.fs,
        acc_fs=acc.fs,
    )


def iter_windows(rec: SubjectRecord):
    """Yield individual Window objects for one subject."""
    ws = window_subject(rec, copy=False)
    for i in range(len(ws)):
        yield Window(
            subject_id=ws.subject_id,
            index=i,
            start_s=float(ws.start_s[i]),
            bvp=ws.bvp[i],
            acc=ws.acc[i],
            hr=float(ws.hr[i]),
            activity=int(ws.activity[i]),
            skin_type=ws.skin_type,
        )
