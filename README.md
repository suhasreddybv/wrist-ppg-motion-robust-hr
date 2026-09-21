# wrist-ppg-motion-robust-hr

Heart-rate estimation from wrist PPG during real-world movement on PPG-DaLiA, with accelerometer-informed motion compensation. It is evaluated leave-one-subject-out, per activity, with baselines reported first.

**Result.** _Pending. The per-activity MAE table against a naive spectral-peak baseline and Reiss et al. (2019) goes here once evaluation lands (by 4 Oct 2026)._

**Status:** Stages 0–3 are done — validated data layer, label-aligned windowing, band-pass with signal quality, and the four LOSO baselines below. Motion compensation (spectral masking, adaptive cancellation, peak tracking) follows.

## Data layer (Stage 0)

`src/data/loader.py` loads each subject into a validated `SubjectRecord` and caches it. It raises on anything the pipeline assumes and does not hold: missing keys, channels whose durations disagree by even one sample, a label count that doesn't match 8 s windows at a 2 s shift, accelerometer data not in g, non-constant "dummy" channels, or out-of-order R-peaks. Nothing is silently coerced; the single tolerated irregularity (exact duplicate R-peaks in S6 and S14) is counted on the record.

Several facts checked against the files differ from the dataset's own documentation. The most consequential is that **wrist ACC in the pickle is already in g**; applying the readme's 1/64 g scaling would silently break motion compensation. The full list is in [data/README.md](data/README.md).

![S1, 60 s of seated-rest BVP: zero-centred with no DC component](figures/s1_rest_bvp.png)

**The E4 BVP has no DC component.** A minute of seated rest from S1 has mean 0.084 against SD 41.95 (whole record: mean −0.002), so the signal is manufacturer-processed and zero-centred and **perfusion index is not available** as a quality measure. The segment also shows a motion burst around 38–47 s during nominal rest.

## Windowing (Stage 1)

`src/data/windows.py` cuts each subject into 8 s windows at a 2 s shift: 512 BVP samples and 256 wrist-ACC samples per window, with the ground-truth HR, the modal activity id and the skin type attached. Windows are produced by striding rather than copying, so a full subject costs no extra memory until materialised.

The assertion that matters is that **the window count equals `len(label)` exactly, for all 15 subjects**, and it is tested across the whole dataset. An off-by-one here would shift every downstream result invisibly. Five subjects (S2, S3, S5, S7, S15) have odd-length records, so a tail shorter than one 2 s shift is left unused; the test allows that and nothing else.

## Preprocessing (Stage 2)

`src/features/preprocess.py`: 0.4–4 Hz (24–240 bpm) 4th-order Butterworth, zero-phase via `sosfiltfilt`, then per-window z-scoring, plus a per-window signal-quality index.

Two implementation choices, both deliberate:
- **The continuous record is filtered before windowing.** `filtfilt` on an isolated 8 s window leaves edge transients at a 0.4 Hz cutoff. The windowing grid is unchanged.
- **Wrist ACC is resampled 32 → 64 Hz** (polyphase) so it shares a time base with the BVP, which time-domain adaptive cancellation needs in Stage 4. The original 32 Hz windows remain available.

The wide band is kept deliberately. Measured gain is 1.00 up to 150 bpm, 0.96 at 180, 0.80 at 210 and 0.50 at the 4 Hz corner, because zero-phase filtering applies the response twice. A 0.8–2.5 Hz band would pass only 0.07 at 180 bpm, discarding the stairs and cycling regime. In this dataset only 39 of 64,697 labels (0.06%) exceed 180 bpm, maximum 187, so the roll-off costs nothing here.

**Signal quality index** per window, from three components: mean correlation of each beat with the window's own beat template, spectral concentration around the dominant in-band peak, and the fraction of power outside the cardiac band (measured on the unfiltered window). Perfusion index is unavailable, as the BVP has no DC component.

Mean SQI by activity, all 15 subjects, window-weighted (`results/02_sqi_by_activity.csv`):

| Activity | Windows | SQI | Template r | Spectral concentration |
|---|---|---|---|---|
| sitting | 4,569 | **0.636** | 0.855 | 0.739 |
| working | 8,497 | 0.544 | 0.857 | 0.625 |
| driving | 6,843 | 0.491 | 0.848 | 0.573 |
| lunch | 13,554 | 0.480 | 0.832 | 0.572 |
| cycling | 3,473 | 0.468 | 0.851 | 0.542 |
| table soccer | 2,310 | 0.397 | 0.810 | 0.488 |
| stairs | 3,239 | 0.390 | 0.819 | 0.474 |
| walking | 4,697 | **0.389** | 0.801 | 0.483 |

The ordering is what it should be: sitting best, walking and stairs worst. Note that **template correlation barely discriminates** (0.80–0.86 across every activity) — beat detection still finds self-similar peaks in motion-corrupted windows, so essentially all the separation comes from spectral concentration. Treat the template term as weak evidence.

## Baselines (Stage 3)

Four baselines, leave-one-subject-out over 15 folds, before any motion handling. Anything a method needs beyond the test subject's own signal — b0's constant, b1's first-window fallback — comes from that fold's training subjects only.

**Aggregation.** The MAE column below is the **mean of per-fold MAEs**: each subject contributes once, so a long recording does not outweigh a short one. The window-pooled figure is reported beside it in the CSV as `mae_pooled` (and `mape_pooled`); the two agree closely (they differ by at most 2.4 bpm, on the smallest cell — `b2_noclip` table soccer — and by under 1.1 bpm on every other row). Every cell carries its fold count — S6 has no lunch, walking or working, so those rows have 14 folds.

**Relative error.** MAPE is reported alongside MAE, because a 5 bpm error means something different at 60 and at 160 bpm. It is the same framing as MARD for glucose sensors, and it matters here: the subject with the worst MAE is far less extreme in MAPE.

| Activity | Folds | Windows | b0 mean HR | b1 *(oracle)* | b2 peak | **b2-zp** | b2-zp MAPE |
|---|---|---|---|---|---|---|---|
| sitting | 15 | 4,569 | 29.28 | 1.09 | 3.90 | **2.82** | 4.5% |
| stairs | 15 | 3,239 | 30.95 | 1.12 | 39.32 | **38.17** | 31.6% |
| table soccer | 15 | 2,310 | 14.02 | 1.85 | 33.54 | **33.27** | 34.9% |
| cycling | 15 | 3,473 | 33.74 | 0.96 | 30.96 | **29.75** | 23.4% |
| driving | 15 | 6,843 | 14.11 | 1.71 | 14.96 | **13.42** | 15.3% |
| lunch | 14 | 13,554 | 13.70 | 1.71 | 15.50 | **14.26** | 16.4% |
| walking | 14 | 4,697 | 15.46 | 1.25 | 29.66 | **29.61** | 28.6% |
| working | 14 | 8,497 | 17.02 | 1.40 | 9.89 | **8.62** | 10.3% |
| **Pooled, transients included** | 15 | 64,697 | 18.57 | 1.49 | 20.06 | **18.98** | 19.2% |
| **Pooled, transients excluded** | 15 | 47,182 | 19.12 | 1.46 | 18.49 | **17.35** | 17.6% |

MAE in bpm. Transient windows are 17,515 of 64,697 (27%), so both pooled figures are given. b1 uses ground truth and is an oracle reference, not a deployable method; pooled at 1.49 bpm, it shows how much of this task is pure temporal smoothness. Full tables — pooled and fold-mean MAE, MAPE, RMSE, per-fold spread, worst fold — are in `results/baselines_per_activity.csv` and `results/baselines_per_subject.csv`.

**b2-zp is the reference for Stage 4.** It is b2 with the FFT zero-padded to 4,096 points, so the peak is located on a 0.94 bpm grid instead of a 7.5 bpm one. Zero-padding interpolates the spectrum and adds no true resolution, and the results show exactly that: it helps where a clean peak exists (sitting 3.90 → 2.82, working 9.89 → 8.62) and does essentially nothing where the peak is motion-locked (walking 29.66 → 29.61, table soccer −0.27, stairs −1.15). Reporting Stage 4 gains against b2-zp keeps a resolution gain from ever being credited to motion handling.

**The naive spectral peak loses to a constant on five of eight activities** — stairs, table soccer, driving, lunch and walking. Pooled, b2-zp is 18.98 bpm against b0's 18.57. Working is the one genuinely quiet high-duration activity, where b2-zp (8.62) clearly beats b0 (17.02); lunch and driving are *not* quiet in this sense, despite their low motion. Three checks confirm b2 is implemented correctly rather than broken:

- **At rest it is nearly exact.** 89.5% of sitting windows land within one FFT bin, median error −0.1 bpm.
- **The errors are structured.** Median error on stairs is −32.8 bpm: the estimate sits *below* the truth, locked onto lower-frequency motion energy. 13–16% of stairs and walking windows land on a 2× or ½× harmonic.
- **It never beats the oracle** on any activity.

### Accelerometer clipping

The E4 accelerometer saturates at ±2 g. Each window carries both a boolean flag and a continuous `clip_fraction`, and the distribution is extremely uneven (`results/clip_fraction_by_activity.csv`):

| Activity | Windows flagged | Mean clip fraction | p95 | Max |
|---|---|---|---|---|
| table soccer | **47.5%** | 0.52% | 1.95% | 5.5% |
| cycling | **39.0%** | 0.64% | 2.73% | 23.8% |
| walking | 10.9% | 0.09% | 0.78% | 3.9% |
| stairs | 10.7% | 0.11% | 0.78% | 9.0% |
| driving | 10.0% | 0.08% | 0.39% | 5.1% |
| lunch | 1.6% | 0.01% | 0.00% | 2.7% |
| working | 0.7% | 0.00% | 0.00% | 3.1% |
| sitting | 0.1% | 0.00% | 0.00% | 0.4% |

Nearly half of table-soccer windows and two-fifths of cycling windows contain saturated accelerometer samples, against almost none at rest. **This matters for Stage 4, not here:** b2 never reads the accelerometer, so scoring it with and without flagged windows cannot say anything about what clipping causes. The `b2_noclip` rows exist as a descriptive column only; the comparison will be repeated once the ACC-referenced methods exist, where it will mean something.

### Per-subject spread

b2-zp MAE ranges from 8.75 (S7) to 45.87 (S5). **S5 is investigated in full in [results/s5_investigation.md](results/s5_investigation.md)** and is excluded from nothing. In short: its oracle error is the *best* in the cohort and its resting b2 MAE is ordinary (3.95 vs 3.23 median), so alignment and sensor contact are fine. What differs is heart rate — elevated in every activity, mean 125.8 bpm against a cohort mean of 86.6, and above 120 bpm in 54% of windows against 7.7% elsewhere. The low-frequency lock that causes the error is cohort-wide, but a true HR near 160 turns each locked window into a ~130 bpm error instead of a ~40 bpm one. Under MAPE, S5 is 35.8% against a cohort median of 19.1% — still worst, far less extreme.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# obtain PPG-DaLiA first: see data/README.md
pytest                              # 74 tests; 13 need the dataset and skip without it
python -m src.data.plot_rest_bvp    # figures/s1_rest_bvp.png and the DC numbers above
python -m src.features.report_sqi   # results/02_sqi_by_activity.csv
python -m src.eval.report_baselines # results/baselines_*.csv, clipping table, stop-condition checks
python -m src.eval.investigate_s5   # results/s5_investigation.md + figures/s5_worst_windows.png
```

The first real-data run unpickles all 15 subjects (~23 GB) and builds a 3.5 GB cache in about 40 s; later runs take about 3 s.

## Limitations and failure cases

_Per-activity failures and the gap to published benchmarks are written once evaluation exists._ Known from the data layer:

- **Skin types 2–4 only.** There are no type V or VI subjects, so nothing here speaks to the melanin confound in optical HR.
- **BVP is manufacturer-processed**, not raw photodiode output, and perfusion index is unavailable.
- **S6 is truncated** (5,250 s, activities 1–5 only), so folds are not equivalent.
- **The accelerometer clips at ±2 g**, most during cycling and table soccer. Motion references are least reliable exactly where they are most needed.
- **Band-edge roll-off.** Zero-phase filtering halves the amplitude at 240 bpm. It is immaterial on PPG-DaLiA (0.06% of labels above 180 bpm) but would matter on a higher-intensity cohort.
- **Absolute error flatters low-heart-rate subjects.** MAE in bpm is reported with MAPE beside it throughout, because the same locked window costs ~40 bpm for a resting subject and ~130 for S5.
- **The SQI's template term is weak**, varying only 0.80-0.86 between resting and walking; spectral concentration carries the signal.
- Activity durations are imbalanced, and this is a single dataset from a single device.

## References

- Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: Large-scale heart rate estimation with convolutional neural networks. *Sensors*, 19(14), 3079.
- Reiss, A., et al. (2019). PPG-DaLiA [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53890
- Zhang, Z., Pi, Z., & Liu, B. (2015). TROIKA: A general framework for heart rate monitoring using wrist-type photoplethysmographic signals during intensive physical exercise. *IEEE Transactions on Biomedical Engineering*, 62(2), 522–531.

## Licence

MIT (code). Data: CC BY 4.0, see [data/README.md](data/README.md).
