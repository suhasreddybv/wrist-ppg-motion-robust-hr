# wrist-ppg-motion-robust-hr

Heart-rate estimation from wrist PPG during real-world movement on PPG-DaLiA, with accelerometer-informed motion compensation. It is evaluated leave-one-subject-out, per activity, with baselines reported first.

**Result.** _Pending. The per-activity MAE table against a naive spectral-peak baseline and Reiss et al. (2019) goes here once evaluation lands (by 4 Oct 2026)._

**Status:** Stages 0–2 are done — validated data layer, label-aligned windowing, band-pass and signal quality. Baselines, motion compensation and evaluation follow.

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

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# obtain PPG-DaLiA first: see data/README.md
pytest                              # 48 tests; 12 need the dataset and skip without it
python -m src.data.plot_rest_bvp    # figures/s1_rest_bvp.png and the DC numbers above
python -m src.features.report_sqi   # results/02_sqi_by_activity.csv
```

The first real-data run unpickles all 15 subjects (~23 GB) and builds a 3.5 GB cache in about 40 s; later runs take about 3 s.

## Limitations and failure cases

_Per-activity failures and the gap to published benchmarks are written once evaluation exists._ Known from the data layer:

- **Skin types 2–4 only.** There are no type V or VI subjects, so nothing here speaks to the melanin confound in optical HR.
- **BVP is manufacturer-processed**, not raw photodiode output, and perfusion index is unavailable.
- **S6 is truncated** (5,250 s, activities 1–5 only), so folds are not equivalent.
- **The accelerometer clips at ±2 g**, most during cycling and table soccer. Motion references are least reliable exactly where they are most needed.
- **Band-edge roll-off.** Zero-phase filtering halves the amplitude at 240 bpm. It is immaterial on PPG-DaLiA (0.06% of labels above 180 bpm) but would matter on a higher-intensity cohort.
- **The SQI's template term is weak**, varying only 0.80-0.86 between resting and walking; spectral concentration carries the signal.
- Activity durations are imbalanced, and this is a single dataset from a single device.

## References

- Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: Large-scale heart rate estimation with convolutional neural networks. *Sensors*, 19(14), 3079.
- Reiss, A., et al. (2019). PPG-DaLiA [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53890
- Zhang, Z., Pi, Z., & Liu, B. (2015). TROIKA: A general framework for heart rate monitoring using wrist-type photoplethysmographic signals during intensive physical exercise. *IEEE Transactions on Biomedical Engineering*, 62(2), 522–531.

## Licence

MIT (code). Data: CC BY 4.0, see [data/README.md](data/README.md).
