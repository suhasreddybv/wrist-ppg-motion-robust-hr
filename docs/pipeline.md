# The pipeline, stage by stage

How the data layer, windowing, preprocessing, baselines and motion compensation were built and what each stage measured. The front page carries the result; this carries the construction.

*Part of [wrist-ppg-motion-robust-hr](../README.md). Numbers here come from the CSVs in `results/`.*

---

## Data layer (Stage 0)

`src/data/loader.py` loads each subject into a validated `SubjectRecord` and caches it. It raises on anything the pipeline assumes and does not hold: missing keys, channels whose durations disagree by even one sample, a label count that doesn't match 8 s windows at a 2 s shift, accelerometer data not in g, non-constant "dummy" channels, or out-of-order R-peaks. Nothing is silently coerced; the single tolerated irregularity (exact duplicate R-peaks in S6 and S14) is counted on the record.

Several facts checked against the files differ from the dataset's own documentation. The most consequential is that **wrist ACC in the pickle is already in g**; applying the readme's 1/64 g scaling would silently break motion compensation. The full list is in [data/README.md](../data/README.md).

![S1, 60 s of seated-rest BVP: zero-centred with no DC component](../figures/s1_rest_bvp.png)

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

**The composite is spectral concentration alone.** The beat-template term was dropped after validating it against error (below): the composite without it scores AUROC 0.722 for finding bad windows, inside the full composite's 95% CI of [0.69, 0.76]. A term that adds nothing measurable does not stay. It is still computed and reported as a component.

Mean SQI by activity, all 15 subjects, window-weighted — **descriptive only**, since motion lowers SQI and raises error, so this ordering would appear whether or not the SQI said anything about an individual window (`results/02_sqi_by_activity.csv`):

| Activity | Windows | SQI | Template r | Out-of-band |
|---|---|---|---|---|
| sitting | 4,569 | **0.739** | 0.855 | 0.055 |
| working | 8,497 | 0.625 | 0.857 | 0.084 |
| driving | 6,843 | 0.573 | 0.848 | 0.110 |
| lunch | 13,554 | 0.572 | 0.832 | 0.114 |
| cycling | 3,473 | 0.542 | 0.851 | 0.086 |
| table soccer | 2,310 | 0.488 | 0.810 | 0.135 |
| walking | 4,697 | **0.483** | 0.801 | 0.087 |
| stairs | 3,239 | **0.474** | 0.819 | 0.088 |

### Does the SQI predict error?

The real test is *within* an activity. Below: AUROC for detecting a b2-zp error above 10 bpm, oriented so higher means worse, with 95% CIs from bootstrapping **subjects** rather than windows — windows within a subject are correlated, and resampling them would understate the interval (`results/sqi_validation.csv`).

![Error by within-activity SQI decile, and AUROC per activity with confidence intervals](../figures/sqi_vs_error.png)

| Activity | AUROC [95% CI] | Median error, worst vs best SQI decile | Verdict |
|---|---|---|---|
| sitting | 0.895 [0.87, 0.92] | 5.0 → 0.5 | predictive |
| working | 0.797 [0.73, 0.84] | 9.9 → 0.7 | predictive |
| driving | 0.703 [0.65, 0.75] | 13.7 → 1.4 | predictive |
| lunch | 0.685 [0.62, 0.74] | 12.1 → 1.1 | predictive |
| cycling | 0.600 [0.48, 0.70] | 27.5 → 1.1 | **CI includes chance** |
| walking | 0.537 [0.43, 0.63] | 26.0 → 14.1 | **CI includes chance** |
| stairs | 0.429 [0.33, 0.52] | 25.5 → 49.9 | **CI includes chance** |
| table soccer | 0.448 [0.41, 0.48] | 32.0 → 32.5 | **anti-predictive** |
| **Pooled** (excl. transient) | 0.722 [0.69, 0.75] | — | predictive |

**The SQI works where there is little motion and fails where there is a lot** — which is the opposite of where a tracker would want it. In stairs, cycling and walking the CI includes 0.5, so it does not discriminate. In table soccer it is worse than useless: AUROC 0.448 with a CI entirely below 0.5, meaning higher SQI goes with *larger* error, and on stairs the worst-to-best decile trend runs the wrong way (25.5 → 49.9 bpm). The reason is that a motion-locked window can have a very sharp spectral peak — at the stride frequency. Concentration measures peak sharpness, not whether the peak is the heart.

**Consequence for Stage 4:** the peak tracker must not use SQI-triggered resets in stairs, table soccer, cycling or walking. Since activity labels will not exist at inference time, the reset rule needs a motion-aware quality term (e.g. agreement between the PPG peak and the accelerometer spectrum), not this SQI. No threshold has been tuned here; if one is needed it is chosen inside each LOSO fold on training subjects only.

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

b2-zp MAE ranges from 8.75 (S7) to 45.87 (S5). **S5 is investigated in full in [results/s5_investigation.md](s5-investigation.md)** and is excluded from nothing. In short: its oracle error is the *best* in the cohort and its resting b2 MAE is ordinary (3.95 vs 3.23 median), so alignment and sensor contact are fine. What differs is heart rate: S5's labels are higher in every activity, mean 125.8 bpm against a cohort mean of 86.6, and above 120 bpm in 54% of windows against 7.7% elsewhere. That was verified rather than assumed — ECG and PPG agree independently at rest (median 92.7 vs 91.9 bpm over 300 sitting windows), and inspection of chest-ECG strips in high-rate, low-motion, unclipped windows shows the stored R-peaks sitting on QRS complexes with T waves unmarked, RR intervals matching the labels to 0.1 bpm, and no short/long alternation that would indicate T-wave oversensing (0.00% of S5's high-rate windows, against 0.00% for S7 and 0.36% for S10). The low-frequency lock that causes the error is cohort-wide, but a true HR near 160 turns each locked window into a ~130 bpm error instead of a ~40 bpm one. Under MAPE, S5 is 35.8% against a cohort median of 19.1% — still worst, far less extreme.

## Motion compensation (Stage 4)

Two components, ablated separately, LOSO, every gain measured against b2-zp. Hyperparameters are chosen **inside each fold on training subjects only** — a configuration picked on all 15 subjects instead would make the tracker look 2.5 bpm better than it is (`results/stage4_selected_configs.csv`).

- **Masking (4a)** builds a motion spectrum from the three wrist-ACC axes plus the magnitude signal, then notches the top-K prominent peaks together with their ½× and 2× harmonics, with notch width set by the measured width of each ACC peak. Windows where the wrist is still are left alone. Bins are attenuated, never zeroed, so a heart rate that coincides with a cadence line stays findable.
- **Tracking** picks the prominent spectral peak nearest a mean-filtered prediction of recent estimates, and resets when the chosen peak keeps disagreeing with the prediction. **The reset reads only the estimates**, so it needs no quality measure — which matters because the SQI does not work under motion (D-016).

| Method | Pooled MAE | MAPE | Fold SD | Worst fold | Paired gain vs b2-zp [95% CI] |
|---|---|---|---|---|---|
| b2-zp | 18.98 | 19.2% | 9.40 | 45.87 | — |
| +tracker | 17.53 | 17.5% | 6.83 | 34.12 | +1.45 [−1.45, +4.65] |
| +mask | 18.61 | 18.9% | 9.23 | 45.33 | +0.36 [+0.04, +0.72] |
| **+mask+tracker** | **15.80** | **15.6%** | 6.04 | 28.74 | **+3.18 [+1.00, +5.83]** |
| *SpaMa* (Reiss et al. 2019) | *15.56* | — | *7.5* | — | published |
| *SpaMaPlus* (Reiss et al. 2019) | *11.06* | — | *4.8* | — | published |

Pooled over all 64,697 windows **including transients**, which is what the published evaluation covers. CIs are from bootstrapping subjects; gains are paired per subject.

**The tracker alone is not claimed.** Its mean gain is 1.45 bpm, but the CI spans zero and only 9 of 15 subjects improve, with S13 losing 10.5 bpm. It is reported as within per-subject variability, not as a result.

**Against the published baselines.** SpaMa already contains a tracking step, so **mask+tracker against SpaMa is the like-for-like pairing, and 15.80 against 15.56 is a replication rather than a shortfall**. The remaining distance to SpaMaPlus (11.06) comes down to one named difference: SpaMaPlus predicts from a mean over the last six estimates and tracks against that, where this tracker's prediction is dominated by the most recent estimate. That is a specific, testable change, logged as the first Week 3 item, not a mystery.

**The spread across subjects is tighter at matched mean error.** Fold SD falls 9.40 → 6.04 against SpaMa's published 7.5, and the worst fold falls 45.87 → 28.74. The method helps the worst subjects most — S5 gains 17.1 bpm and S6 8.2 — which is the property that matters for a wearable, where a device that is usually good and occasionally catastrophic is worse than one that is uniformly adequate. Per subject we beat SpaMa on 5 of 15 and SpaMaPlus on 0 of 15 (`results/stage4_per_subject.csv`; published values from their Table 10).

**The components are superadditive.** Alone they are worth 1.45 and 0.36 bpm; together 3.18. Masking clears the competing motion line, tracking selects the survivor, and neither does much on its own — masking leaves the wrong peak still selectable, and tracking without masking follows a motion line as happily as a cardiac one.

**Hyperparameter selection is in-fold and costs real accuracy.** Choosing the configuration on training subjects rather than all 15 costs 2.51 bpm for the tracker and 1.40 for the combination, and only 7/15 and 9/15 folds pick the globally best setting (`results/stage4_selected_configs.csv`). The globally chosen figures are never reported. The published comparison selects its parameters the same way, per held-out session, so the comparison stays like-for-like.

### Where it gets worse

| Activity | b2-zp | +tracker | +mask | +mask+tracker |
|---|---|---|---|---|
| sitting | 2.82 | 2.65 | 2.81 | **2.63** |
| working | 8.62 | 5.62 | 8.73 | **4.92** |
| driving | 13.42 | 7.73 | 13.56 | **7.50** |
| lunch | 14.26 | 9.21 | 14.25 | **6.64** |
| table soccer | 33.27 | 29.08 | 33.26 | **26.73** |
| walking | 29.61 | 31.44 | 28.98 | **25.18** |
| cycling | **29.75** | 42.91 | 27.38 | 36.96 |
| stairs | **38.17** | 52.55 | 37.37 | 56.01 |

![Per-activity MAE for b2-zp and mask+tracker with the b0 constant baseline drawn across each activity](../figures/per_activity_mae.png)

**On four activities this method is worse than predicting a constant.** Not merely worse than b2-zp: stairs **56.01 against the b0 constant's 30.95**, table soccer 26.73 against 14.02, walking 25.18 against 15.46, cycling 36.96 against 33.74. A method that loses to "always guess the mean heart rate" on half the protocol has not solved those activities. Tracking assumes the previous estimate is informative; during sustained vigorous motion the spectrum offers a stable *wrong* peak and the tracker holds it. On the quiet activities the same mechanism is transformative — lunch 14.26 → 6.64, working 8.62 → 4.92. Week 3 addresses this directly; see below.

### Does masking remove what it targets?

Yes, and the check is mechanistic rather than an MAE reading (`results/masked_taxonomy.csv`). The `acc_locked` share of error windows falls from 23.3% to 13.1% cohort-wide, and in walking from **55.1% to 26.1%** (1,626 → 694 windows). The residual 13.1% is at the **13.6% chance rate** measured by the permutation null, so after masking almost no *real* accelerometer lock remains.

But total error windows fall only 2.4% (17,485 → 17,074): the errors are reclassified into `other`, not removed. That is the difference between masking's 0.36 bpm and the 3.18 bpm the combination achieves, and it says the remaining failure is not accelerometer lock.

Mean masked energy for the selected configurations is **7.7%** of in-band power, well under the 30% flag.

### Does the cardiac peak survive masking?

Yes — the information is there and the problem is selection. For masked windows still in error, the masked spectrum contains a peak within 3 bpm of the true heart rate **58.8% of the time, against a 36.3% chance rate** built the same way as the accelerometer-lock null: +22.5 points of real excess (`results/surviving_peak.csv`). The excess is largest exactly where the method fails — **cycling 78.1% against 34.1% chance, stairs 65.6% against 36.8%**.

That peak is essentially never the largest one (rank 1: 0.0%, by construction — if it were, the window would not be in error), but it is **rank 2 in 34.9% of cases and rank 3 in 24.2%**. Roughly three in five surviving peaks are in the top three. So a better selection rule has somewhere to go, and this is a selection problem rather than a destroyed-signal problem. The worst window in the cohort makes it concrete:

![Best and worst windows of the full method: BVP trace and masked spectrum](../figures/best_worst_windows.png)

The bottom row is S5 cycling at a true 173 bpm. The cardiac peak is plainly present in the masked spectrum, and the tracker returned 30 bpm.

### How long does an error last?

Mean error is not the whole story for a wearable. Measuring runs of consecutive windows with error over 10 bpm (`results/error_persistence.csv`):

| | Runs | Median | p90 | Longest | Share of error windows in runs > 30 s |
|---|---|---|---|---|---|
| b2-zp | 3,574 | 3 | 10 | 99 | 27.6% |
| mask+tracker | 2,876 | 1 | 6 | **409** | **61.8%** |

![Run-length distribution of error episodes by activity, b2-zp vs mask+tracker](../figures/error_persistence.png)

**The method reduces mean error while making individual errors last far longer, and for a wearable that is the more dangerous failure.** It removes many short errors — the median run drops from 3 windows to 1 — but almost two thirds of remaining error time now sits in episodes longer than 30 seconds, against a quarter before. On stairs the p90 run length goes from 20 windows to **210** (7 minutes), and the longest single error episode in the cohort runs to 409 windows, about 13 minutes.

The cause is visible in the reset counts: the reset fires **0.3 times per 1,000 windows on stairs and 1.4 on cycling**, and never at all on most activities. It triggers on jumps, and a smoothly tracked wrong estimate never jumps. A reset that detects sustained wrongness rather than sudden movement is the obvious next step, and is logged as a Week 3 item.

## Breaking the lock-in (Week 3)

Two components, both aimed at the lock-in rather than at pooled MAE.

**A fold-derived lower bound.** Estimates below ~40 bpm are always wrong in this cohort while the lowest ECG label is 41.7 bpm, so the search space is restricted. The bound is **derived inside each fold** as the minimum training label less a margin chosen on training subjects — never the global minimum, never a hand-picked constant. Every fold selected a zero margin, giving 41.7 bpm for fourteen folds and 41.9 for the fold that holds out the subject carrying the cohort minimum. It changes **5,555 of 64,697 windows (8.6%)**, from 136 windows for S7 to 779 for S5 (`results/week3_bound_effect.csv`).

**But the bound does not generalise, and the sensitivity curve says so** (`results/bound_sensitivity.csv`). Every fold picking a zero margin means the grid saturated at its tightest edge — the setting that maximises in-sample gain — and the bound then sits at the *training sample's* minimum, a biased estimate of any population minimum:

| Margin below the training minimum | Bound | Gain of the bound alone | Share of the margin-0 gain |
|---|---|---|---|
| 0 bpm | 41.7 | +2.04 | 100% |
| 5 bpm | 36.7 | +0.99 | 49% |
| 10 bpm | 31.7 | +0.22 | 11% |
| 15 bpm | 26.7 | +0.01 | 1% |
| 20–30 bpm | ≤21.7 | +0.00 | 0% |

**Half the gain is gone 5 bpm down, and all of it 15 bpm down.** A bound loose enough to be safe for a bradycardic, athletic or beta-blocked subject — where resting rates of 35–40 bpm are ordinary — is worth nothing. This is a property of this cohort's heart-rate floor, not a signal-processing component, and it is the reason the headline figure excludes it.

### The prediction window is worth more than masking

SpaMaPlus predicts from a mean over recent estimates, and so does this tracker — it always has. The open question was not whether to add the mean filter but how long it should be, and that had never been swept (`results/prediction_rule.csv`, everything else held at each fold's own configuration, no bound):

| Prediction rule | Pooled MAE |
|---|---|
| mean over 1 window (the last estimate alone) | 18.41 |
| mean over 6 windows (the setting used all through Week 2) | 15.80 |
| **mean over 15 windows** | **14.70** |
| mean over 20 / 30 / 45 / 60 | 15.18 / 15.66 / 15.79 / 15.92 |
| median, any length | worse at every length |

All 15 folds selected the 15-window mean, and the optimum is interior — 10 and 20 windows are both worse — so this is a real optimum rather than a grid edge. **Lengthening the prediction window is worth +1.10 bpm, three times what masking is worth (+0.36)**, and it is one line of configuration. The median is worse than the mean everywhere, which suggests the prediction benefits from being dragged by outliers rather than protected from them: a genuine rate change shows up in the mean before it can win a majority.

**It interacts with the bound.** The 15-window mean is worth +1.10 bpm without the bound but *costs* 0.56 with it (13.06 against 12.50 for the 6-window mean). Once the bound has removed the low-frequency traps, a long memory is mostly a brake on following real heart-rate changes. The two settings have to be chosen together, not stacked.

**A sustained-wrongness reset.** The jump reset cannot see a smoothly tracked wrong estimate. Two formulations were tried: a hard rank test (the tracked peak must stay within the top-N peaks) and a continuous height-ratio test. **The ratio test won on every fold** — the tracked peak must keep at least 30% of the spectral maximum's height for five consecutive windows — and it beat the best rank formulation by 1.3 bpm pooled (12.50 against 13.82). Both are in `results/week3_ablation.csv`.

**The reset does nothing without the bound.** With the bound it is worth +0.47 bpm (12.97 → 12.50); without one, the best reset configuration scores **15.86 against 15.80 — very slightly worse** (`results/without_bound_best.csv`). Re-seeding only helps when the unconstrained argmax it re-seeds onto is itself constrained to a sane region; without a bound the reset frequently lands back on the low-frequency lock it just escaped. The two components are not independent, and the 15.80 figure is therefore the best available without a bound.

| Method | Pooled MAE | MAPE | Paired gain [95% CI] | Improved |
|---|---|---|---|---|
| b2-zp | 18.98 | 19.2% | — | — |
| +bound | 16.94 | 17.0% | +2.04 [+1.57, +2.63] | **15/15** |
| +tracker | 17.53 | 17.5% | +1.45 [−1.42, +4.53] | 9/15 |
| +mask | 18.61 | 18.9% | +0.36 [+0.05, +0.73] | 10/15 |
| +mask+tracker | 15.80 | 15.6% | +3.18 [+1.00, +5.92] | 11/15 |
| +mask+tracker+bound | 12.97 | 12.8% | +6.00 [+3.38, +9.15] | 13/15 |
| **+mask+tracker+bound+reset** | **12.50** | **12.7%** | **+6.48 [+3.96, +9.47]** | **14/15** |

The bound alone improves **every subject** — the only component in this repository that does.

