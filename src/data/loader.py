"""PPG-DaLiA subject loader with validation and an .npz cache.

Every property the rest of the pipeline relies on is checked here and raises
`DataValidationError` on failure. Nothing is silently coerced.

Verified against the data on 16 Sep 2026 (see data/README.md):
- `SX.pkl` top-level keys are activity, label, questionnaire, rpeaks, signal, subject.
- Wrist ACC in the pickle is already in g (median |ACC| ~1.01 at S1). The 1/64 g
  units in the dataset readme apply to the raw E4 CSVs, not to this file.
- Chest EMG, EDA and Temp are constant placeholders and are dropped.
- S6 has 3 and S14 has 1 exact duplicate R-peak indices. They are removed and
  counted; the HR labels around them are smooth, so the ground truth did not use them.
- Every channel implies the same duration exactly, and
  len(label) == (duration - 8) / 2 + 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PPG_DALIA_ROOT", REPO_ROOT / "data" / "PPG_FieldStudy"))
CACHE_DIR = Path(os.environ.get("PPG_DALIA_CACHE", REPO_ROOT / "data" / "cache"))

SUBJECT_IDS = tuple(f"S{i}" for i in range(1, 16))

WINDOW_S = 8
SHIFT_S = 2

TOP_LEVEL_KEYS = {"activity", "label", "questionnaire", "rpeaks", "signal", "subject"}

# (location, key in pickle) -> (record name, sampling rate in Hz, units, expected columns)
# Units: wrist ACC verified as g on load; wrist EDA/TEMP as stated in the dataset readme;
# the readme gives no units for the RespiBAN chest channels.
CHANNELS = {
    ("wrist", "BVP"): ("wrist_bvp", 64, "E4 BVP, arbitrary units (DC removed, verified)", 1),
    ("wrist", "ACC"): ("wrist_acc", 32, "g", 3),
    ("wrist", "EDA"): ("wrist_eda", 4, "uS (per readme)", 1),
    ("wrist", "TEMP"): ("wrist_temp", 4, "degC (per readme)", 1),
    ("chest", "ECG"): ("chest_ecg", 700, "unspecified", 1),
    ("chest", "ACC"): ("chest_acc", 700, "unspecified", 3),
    ("chest", "Resp"): ("chest_resp", 700, "unspecified", 1),
}
CHEST_DUMMY = ("EMG", "EDA", "Temp")
ACTIVITY_RATE = 4
ACTIVITY_IDS = set(range(9))  # 0 transient, 1-8 protocol

ACC_G_RANGE = (0.8, 1.2)   # median |wrist ACC| must fall here if units are g
WRIST_ACC_LIMIT_G = 2.0    # Empatica E4 accelerometer full scale
HR_RANGE_BPM = (30.0, 230.0)

CACHE_VERSION = 2


class DataValidationError(ValueError):
    """Raised when a subject file does not match what the pipeline assumes."""


@dataclass(frozen=True)
class Signal:
    name: str
    data: np.ndarray
    fs: int
    units: str

    @property
    def duration_s(self) -> float:
        return len(self.data) / self.fs


@dataclass(frozen=True)
class Provenance:
    source_path: str
    source_size: int
    source_mtime: float
    sha256: str
    loaded_from: str  # "pickle" or "cache"


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    signals: dict[str, Signal]
    label: np.ndarray          # mean instantaneous HR (bpm), one per 8 s window, 2 s shift
    activity: np.ndarray       # int activity id at 4 Hz
    rpeaks: np.ndarray         # chest ECG sample indices
    skin_type: int             # Fitzpatrick
    duration_s: float
    wrist_acc_clipped_fraction: float
    rpeak_duplicates_removed: int  # exact duplicate indices in the source (S6: 3, S14: 1; labels unaffected)
    provenance: Provenance
    window_s: int = WINDOW_S
    shift_s: int = SHIFT_S
    questionnaire: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> Signal:
        return self.signals[name]


def _fail(subject_id: str, msg: str) -> None:
    raise DataValidationError(f"{subject_id}: {msg}")


def expected_label_count(duration_s: float) -> int:
    return int((duration_s - WINDOW_S) // SHIFT_S) + 1


def validate_raw(raw: dict, subject_id: str) -> None:
    """Check the unpickled dict against every assumption before anything is extracted."""
    missing = TOP_LEVEL_KEYS - set(raw)
    if missing:
        _fail(subject_id, f"missing top-level keys {sorted(missing)}; found {sorted(raw)}")
    if raw["subject"] != subject_id:
        _fail(subject_id, f"file declares subject {raw['subject']!r}")

    sig = raw["signal"]
    for (loc, key), (_, fs, _, ncols) in CHANNELS.items():
        if loc not in sig or key not in sig[loc]:
            _fail(subject_id, f"missing channel signal.{loc}.{key}")
        arr = np.asarray(sig[loc][key])
        if arr.ndim != 2 or arr.shape[1] != ncols:
            _fail(subject_id, f"signal.{loc}.{key} has shape {arr.shape}, expected (N, {ncols})")
        if not np.all(np.isfinite(arr)):
            _fail(subject_id, f"signal.{loc}.{key} contains non-finite values")

    for key in CHEST_DUMMY:
        if key in sig["chest"] and np.ptp(np.asarray(sig["chest"][key])) > 1e-3:
            _fail(subject_id, f"chest.{key} is documented as dummy data but is not constant")

    # One duration, implied exactly by every channel.
    durations = {f"{loc}.{key}": len(sig[loc][key]) / fs for (loc, key), (_, fs, _, _) in CHANNELS.items()}
    durations["activity"] = len(raw["activity"]) / ACTIVITY_RATE
    if len(set(durations.values())) != 1:
        _fail(subject_id, f"channels imply different durations: {durations}")
    duration = next(iter(durations.values()))

    label = np.asarray(raw["label"])
    if label.ndim != 1 or len(label) != expected_label_count(duration):
        _fail(subject_id, f"{len(label)} labels for {duration} s; expected {expected_label_count(duration)} "
                          f"({WINDOW_S} s windows, {SHIFT_S} s shift)")
    if not np.all(np.isfinite(label)) or label.min() < HR_RANGE_BPM[0] or label.max() > HR_RANGE_BPM[1]:
        _fail(subject_id, f"label HR outside {HR_RANGE_BPM} bpm: {label.min():.1f}-{label.max():.1f}")

    acc_mag = np.median(np.linalg.norm(np.asarray(sig["wrist"]["ACC"]), axis=1))
    if not ACC_G_RANGE[0] <= acc_mag <= ACC_G_RANGE[1]:
        _fail(subject_id, f"median |wrist ACC| = {acc_mag:.3f}, expected ~1 g. "
                          "If ~64, the data is in raw 1/64 g units; if ~0.016 it was scaled twice.")

    activity = np.asarray(raw["activity"]).ravel()
    if not np.all(activity == np.round(activity)) or not set(np.unique(activity).astype(int)) <= ACTIVITY_IDS:
        _fail(subject_id, f"activity ids outside 0-8: {np.unique(activity)}")

    rpeaks = np.asarray(raw["rpeaks"])
    ecg_len = len(sig["chest"]["ECG"])
    if len(rpeaks) and (rpeaks.min() < 0 or rpeaks.max() >= ecg_len):
        _fail(subject_id, f"rpeaks outside the ECG length ({ecg_len} samples)")
    if np.any(np.diff(rpeaks) < 0):
        _fail(subject_id, "rpeaks decrease; only exact duplicates are tolerated")

    skin = raw["questionnaire"].get("SKIN")
    if skin not in {1, 2, 3, 4, 5, 6}:
        _fail(subject_id, f"Fitzpatrick skin type {skin!r} not in 1-6")


def _sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _record_from_raw(raw: dict, subject_id: str, provenance: Provenance) -> SubjectRecord:
    sig = raw["signal"]
    signals = {}
    for (loc, key), (name, fs, units, ncols) in CHANNELS.items():
        arr = np.asarray(sig[loc][key])
        signals[name] = Signal(name, arr[:, 0] if ncols == 1 else arr, fs, units)
    acc = signals["wrist_acc"].data
    rpeaks = np.asarray(raw["rpeaks"])
    return SubjectRecord(
        subject_id=subject_id,
        signals=signals,
        label=np.asarray(raw["label"], dtype=float),
        activity=np.asarray(raw["activity"]).ravel().astype(np.int8),
        rpeaks=np.unique(rpeaks),
        skin_type=int(raw["questionnaire"]["SKIN"]),
        duration_s=signals["wrist_bvp"].duration_s,
        wrist_acc_clipped_fraction=float(np.mean(np.any(np.abs(acc) >= WRIST_ACC_LIMIT_G, axis=1))),
        rpeak_duplicates_removed=int(len(rpeaks) - len(np.unique(rpeaks))),
        provenance=provenance,
        questionnaire={k: (v.strip() if isinstance(v, str) else v) for k, v in raw["questionnaire"].items()},
    )


def _cache_path(subject_id: str) -> Path:
    return CACHE_DIR / f"{subject_id}.v{CACHE_VERSION}.npz"


def _write_cache(rec: SubjectRecord) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {f"sig__{n}": s.data for n, s in rec.signals.items()}
    meta = {
        "subject_id": rec.subject_id, "skin_type": rec.skin_type, "duration_s": rec.duration_s,
        "wrist_acc_clipped_fraction": rec.wrist_acc_clipped_fraction,
        "rpeak_duplicates_removed": rec.rpeak_duplicates_removed,
        "questionnaire": rec.questionnaire,
        "source_path": rec.provenance.source_path, "source_size": rec.provenance.source_size,
        "source_mtime": rec.provenance.source_mtime, "sha256": rec.provenance.sha256,
    }
    tmp = _cache_path(rec.subject_id).with_suffix(".tmp.npz")
    np.savez(tmp, label=rec.label, activity=rec.activity, rpeaks=rec.rpeaks,
             meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(_cache_path(rec.subject_id))


def _read_cache(subject_id: str, source: Path) -> SubjectRecord | None:
    path = _cache_path(subject_id)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        st = source.stat()
        if meta["source_size"] != st.st_size or meta["source_mtime"] != st.st_mtime:
            return None  # source changed: rebuild
        signals = {}
        for (_, _), (name, fs, units, _) in CHANNELS.items():
            signals[name] = Signal(name, z[f"sig__{name}"], fs, units)
        return SubjectRecord(
            subject_id=subject_id, signals=signals, label=z["label"], activity=z["activity"],
            rpeaks=z["rpeaks"], skin_type=meta["skin_type"], duration_s=meta["duration_s"],
            wrist_acc_clipped_fraction=meta["wrist_acc_clipped_fraction"],
            rpeak_duplicates_removed=meta["rpeak_duplicates_removed"],
            provenance=Provenance(meta["source_path"], meta["source_size"], meta["source_mtime"],
                                  meta["sha256"], "cache"),
            questionnaire=meta["questionnaire"],
        )


def load_subject(subject_id: str, use_cache: bool = True) -> SubjectRecord:
    """Load, validate and return one subject. First load unpickles and writes the cache."""
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"unknown subject {subject_id!r}; expected one of {SUBJECT_IDS}")
    source = DATA_ROOT / subject_id / f"{subject_id}.pkl"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Set PPG_DALIA_ROOT or see data/README.md.")

    if use_cache and (rec := _read_cache(subject_id, source)) is not None:
        return rec

    with open(source, "rb") as f:
        raw = pickle.load(f, encoding="latin1")  # Python 2 pickle
    validate_raw(raw, subject_id)
    st = source.stat()
    prov = Provenance(str(source), st.st_size, st.st_mtime, _sha256(source), "pickle")
    rec = _record_from_raw(raw, subject_id, prov)
    if use_cache:
        _write_cache(rec)
    return rec


def load_all(subject_ids=SUBJECT_IDS, use_cache: bool = True) -> Iterator[SubjectRecord]:
    """Yield subjects one at a time so only one is in memory."""
    for sid in subject_ids:
        yield load_subject(sid, use_cache=use_cache)
