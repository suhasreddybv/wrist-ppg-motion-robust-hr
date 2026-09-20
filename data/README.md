# Data: PPG-DaLiA

Wrist PPG (Empatica E4) and chest ECG (RespiBAN) from 15 subjects during a ~2.5 h daily-life protocol.

Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). PPG-DaLiA [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53890. Licence: CC BY 4.0.

## Obtaining it

Download from the UCI page above and extract, so that each subject folder sits at:

```
data/PPG_FieldStudy/S1/S1.pkl  ...  data/PPG_FieldStudy/S15/S15.pkl
```

A symlink works, or set `PPG_DALIA_ROOT` to wherever `PPG_FieldStudy/` lives. Only `SX.pkl` is used. `data/` is gitignored except this file, and the loader writes a per-subject `.npz` cache to `data/cache/` (about 3.5 GB for all 15 subjects; override with `PPG_DALIA_CACHE`).

## Verified facts

Checked on the downloaded files on 16 Sep 2026, and enforced by `src/data/loader.py` and `tests/test_loader.py`:

- **Top-level keys** of `SX.pkl` are `activity, label, questionnaire, rpeaks, signal, subject`. The dataset readme's quick-start mentions a nested `'data'` key; it does not exist.
- **Wrist ACC is already in g** in the pickle (S1 median |ACC| = 1.01, range ±2.0). The readme's "1/64 g" applies to `ACC.csv` inside `SX_E4.zip`. Scaling the pickle by 1/64 would be wrong, and the loader rejects either mistake.
- **The accelerometer clips** at the E4's ±2 g full scale: 0.04–0.18% of samples per subject, rising to about 0.9% for S1 during cycling.
- **Durations are exact and consistent.** Every channel implies the same duration to the sample, and `len(label) == (duration − 8) / 2 + 1` for every subject. S1 is 9,212 s (589,568 BVP samples, 4,603 labels).
- **Per-subject durations** range from 7,914 s (S12) to 10,648 s (S10), with **S6 truncated at 5,250 s** and covering activities 1–5 only (no lunch, walking or working).
- **Chest EMG, EDA and Temp are constant placeholders** (−1.5, 0 and −273.15) and are dropped.
- **Duplicate R-peak indices:** S6 has 3, S14 has 1, everyone else 0. HR labels are smooth through all of them, so they were not used to compute ground truth. The loader removes and counts them.
- **BVP has no DC component.** A 60 s seated-rest segment of S1 has mean 0.084 against SD 41.95; the whole record has mean −0.002. Perfusion index cannot be computed.
- **Five subjects have odd-length records** (S2, S3, S5, S7, S15), leaving a 1 s tail that cannot form an 8 s window. Label counts still match the window count exactly.
- **Fitzpatrick skin types** are 2 (S15), 4 (S4, S9, S10) and 3 (everyone else). No type I, V or VI.
- Units for the RespiBAN chest ECG, ACC and Resp are not stated in the dataset readme and are recorded as unspecified.
