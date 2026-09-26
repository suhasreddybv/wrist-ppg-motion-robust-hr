# Design decisions

An append-only record of the design choices in this repository, the measured evidence behind each, and what was rejected. Entries are dated by when the decision landed in git, with the commit cited.

Every number here is copied from a committed results file, test or script output, and the source is named. Nothing is quoted from memory. Entries are never rewritten: a changed decision gets a new entry, and the old entry's **Status** line alone is updated to point at it. Corrections stay visible beside what they correct.

---

## Evaluation design

### D-001 · Leave-one-subject-out, per-activity headline
- **Date / commit:** 2026-09-21 · `2124718`
- **Status:** adopted
- **Decision:** 15 LOSO folds, one per subject. Per-activity results are the headline; pooled is secondary. Anything a method needs beyond the test subject's own signal is computed on that fold's training subjects only.
- **Evidence:** pooled figures are dominated by the long near-stationary blocks — lunch is 13,554 windows and working 8,497, against 2,310 for table soccer (`results/baselines_per_activity.csv`). A pooled number flatters any method.
- **Rejected:** pooled random window splits. Windows overlap by 6 of 8 seconds at a 2 s shift, so neighbouring windows share most of their samples and a random split leaks almost the whole test set into training.
- **Depends on this:** every number in the README, and the fold-count column in both baseline tables.

### D-002 · Baselines before any method; b2-zp is the Stage 4 reference
- **Date / commit:** 2026-09-21 · `a02b426`, `5cd3360`
- **Status:** adopted
- **Decision:** three baselines are reported before any motion handling — b0 global mean, b1 previous-window ground truth (oracle), b2 naive spectral peak — plus b2-zp, the same estimator zero-padded to 4,096 points. Stage 4 gains are measured against **b2-zp**, not b2.
- **Evidence:** zero-padding helps only where a clean peak exists and leaves motion-locked windows alone: sitting 3.90 → 2.82, working 9.89 → 8.62, against stairs 39.32 → 38.17, table soccer −0.27, walking −0.06 bpm (`results/baselines_per_activity.csv`, `mae_mean_of_folds`). The scope's stop condition — no more than ~3 bpm gain on stairs, table soccer or walking — passed, and is re-checked by `src/eval/report_baselines.py`.
- **Rejected:** reporting Stage 4 against b2. A finer spectral grid is worth about 1.1 bpm pooled and would have been credited to motion handling.
- **Depends on this:** every Stage 4 gain claim.

### D-003 · MAE is the mean of per-fold MAEs, with pooled reported beside it
- **Date / commit:** 2026-09-21 · `f5002ed`
- **Status:** adopted
- **Decision:** `mae_mean_of_folds` and `mae_pooled` are separate columns; the README quotes the fold mean. Every cell carries its fold count.
- **Evidence:** recordings differ by 35% in length (7,914 s to 10,648 s, `data/README.md`), so pooling weights subjects unequally. S6 has no lunch, walking or working, giving those rows 14 folds rather than 15 (`results/baselines_per_activity.csv`).
- **Rejected:** a single column named `mae`. It was ambiguous, and the two disagree by up to 2.44 bpm on the smallest cell.
- **Depends on this:** the comparability of every table in the README.

### D-004 · MAPE reported beside MAE throughout
- **Date / commit:** 2026-09-21 · `f93d381`
- **Status:** adopted
- **Decision:** mean absolute percentage error accompanies MAE for every method, activity, subject and pooled row.
- **Evidence:** S5 is the worst subject at 45.87 bpm MAE but 35.8% MAPE against a cohort median of 19.1% — an outlier in bpm, merely worst in relative terms (`results/baselines_per_subject.csv`, `results/s5_investigation.md`).
- **Rejected:** bpm only. The same motion-locked window costs ~40 bpm at a resting rate and ~130 bpm for S5, which makes absolute error partly a measure of who the subject is.
- **Depends on this:** the S5 interpretation in D-020.

### D-005 · Pooled figures given with and without transients
- **Date / commit:** 2026-09-21 · `f5002ed`
- **Status:** adopted
- **Decision:** transient windows (activity 0) are excluded from per-activity rows and reported both ways when pooled.
- **Evidence:** transients are 17,515 of 64,697 windows, 27.1% (`results/baselines_per_activity.csv`, pooled rows). b2-zp is 18.98 bpm including them and 17.35 excluding.
- **Rejected:** silently including them, which inflates pooled error by about 1.6 bpm with no statement of why.

---

## Data layer

### D-006 · Wrist ACC is not rescaled
- **Date / commit:** 2026-09-16 · `0181e04`
- **Status:** adopted
- **Decision:** the accelerometer in `SX.pkl` is already in g and is used as-is. The loader rejects data whose median magnitude is outside 0.8–1.2 g.
- **Evidence:** S1's median |ACC| is 1.01 with a range of ±2.0 (`data/README.md`; asserted in `tests/test_loader.py::test_s1_acc_is_in_g_and_not_rescaled`). The dataset readme's "1/64 g" is stated only for `ACC.csv` inside `SX_E4.zip`.
- **Rejected:** following the pipeline spec and the dataset readme, which said to scale by 1/64. That would have made every motion reference 64× too small, with no error raised and no visible symptom until Stage 4 produced nothing.
- **Depends on this:** all ACC-referenced work — the hero figure, and Stage 4 masking and cancellation.

### D-007 · Exact length relation asserted, not approximate counts
- **Date / commit:** 2026-09-16 · `0181e04`
- **Status:** adopted
- **Decision:** the loader asserts that every channel implies the same duration to the sample, and that `len(label) == (duration − 8) / 2 + 1` for every subject.
- **Evidence:** S1 is 9,212 s with 589,568 BVP samples and 4,603 labels, not the spec's approximate 576,000 and 4,500 (`data/README.md`; pinned in `tests/test_loader.py::test_s1_pinned_shapes`). The relation holds for all 15 subjects including the truncated S6.
- **Rejected:** a ±5% tolerance around the spec's figures. It would have passed while hiding that the spec's numbers were derived from a rounded duration.

### D-008 · Duplicate R-peaks removed and counted
- **Date / commit:** 2026-09-16 · `0181e04`
- **Status:** adopted
- **Decision:** exact duplicate R-peak indices are removed, the count is stored on the record, and out-of-range or decreasing peaks still raise.
- **Evidence:** S6 has 3 duplicates and S14 has 1; all other subjects have none (`data/README.md`; pinned in `tests/test_loader.py`). HR labels are smooth through every duplicate, so the published ground truth did not use them.
- **Rejected:** dropping them silently (hides a defect in a published dataset) and rejecting the subjects (would discard S6's 10,436 and S14's 13,274 good peaks over four bad ones).

### D-009 · Chest channel units recorded as unspecified
- **Date / commit:** 2026-09-16 · `0181e04`
- **Status:** adopted
- **Decision:** the RespiBAN ECG, ACC and Resp channels carry `units="unspecified"`.
- **Evidence:** the dataset readme states units only for the E4 CSVs — μS for EDA, °C for TEMP, 1/64 g for ACC — and says nothing about the chest channels (`PPG_FieldStudy_readme.pdf`, quoted in `data/README.md`).
- **Rejected:** labelling them mV, g and % by convention. Nothing in this pipeline needs chest units, so a plausible guess would have been an unverified claim carried in code.

### D-010 · Perfusion index excluded
- **Date / commit:** 2026-09-16 · `06a4aec`, `e2ea341`
- **Status:** adopted
- **Decision:** the SQI does not use perfusion index.
- **Evidence:** the E4 BVP is zero-centred with the DC component removed — a 60 s seated-rest segment of S1 has mean 0.084 against SD 41.95, and the whole record has mean −0.002 (`data/README.md`, figure from `src/data/plot_rest_bvp.py`).
- **Rejected:** asserting the absence from the device documentation. It was measured instead, and the figure is in the README.

---

## Signal processing

### D-011 · Band-pass 0.4–4 Hz, zero-phase, applied before windowing
- **Date / commit:** 2026-09-20 · `41feb4e`
- **Status:** adopted
- **Decision:** 4th-order Butterworth, 0.4–4 Hz, `sosfiltfilt`, applied to the continuous record; windows are cut afterwards.
- **Evidence:** measured gain is 1.00 to 150 bpm, 0.96 at 180 and 0.50 at the 4 Hz corner, while a 0.8–2.5 Hz band passes ≤0.10 at 180 bpm (`tests/test_preprocess.py::test_wide_band_keeps_high_heart_rates_that_a_narrow_band_discards`). Only 39 of 64,697 labels exceed 180 bpm, maximum 187 (README, Stage 2).
- **Rejected:** a 0.8–2.5 Hz band, which appears to solve motion artifact by construction because it excludes the cadence-collision region, and makes the estimator useless over 150 bpm — the stairs and cycling regime. Also rejected: filtering each 8 s window separately, which leaves `filtfilt` edge transients at a 0.4 Hz cutoff.
- **Depends on this:** the band-edge roll-off limitation in the README; any higher-intensity cohort would need this revisited.

### D-012 · Wrist ACC resampled 32 → 64 Hz
- **Date / commit:** 2026-09-20 · `41feb4e`
- **Status:** adopted
- **Decision:** polyphase resample of the accelerometer to the BVP rate; the native 32 Hz windows stay available on the record.
- **Evidence:** resampling preserves gravity to within 0.02 g (`tests/test_preprocess.py::test_resample_acc_doubles_rate_and_preserves_gravity`); the 64 Hz windows read 1.0 g median on S1 (`tests/test_preprocess.py::test_preprocess_s1_shapes_and_alignment`).
- **Rejected:** keeping the two rates separate and comparing only in the frequency domain. Stage 4b's time-domain adaptive cancellation needs a shared sample clock.

### D-013 · Window count must equal label count
- **Date / commit:** 2026-09-20 · `4ace258`
- **Status:** adopted
- **Decision:** windowing raises `WindowAlignmentError` unless the number of windows equals `len(label)` exactly, for every subject.
- **Evidence:** asserted across all 15 subjects in `tests/test_windows.py::test_window_count_equals_label_count_every_subject`. Five subjects (S2, S3, S5, S7, S15) have odd-length records, leaving a tail shorter than one 2 s shift; that is the only tolerated slack.
- **Rejected:** trusting the arithmetic. An off-by-one would shift every result by 2 s and stay invisible in the metrics.

---

## Signal quality

### D-014 · SQI validated within activity, against error
- **Date / commit:** 2026-09-21 · `97674e7`
- **Status:** adopted
- **Decision:** the SQI is judged by AUROC for detecting a b2-zp error above 10 bpm *within* each activity, with 95% CIs from bootstrapping subjects.
- **Evidence:** `results/sqi_validation.csv`, produced by `src/eval/validate_sqi.py` (2,000 resamples, seed 42).
- **Rejected:** judging it by how it ranks activities. Motion lowers SQI and raises error, so that ranking appears whether or not the score says anything about an individual window. Also rejected: bootstrapping windows, which are correlated within a subject and would understate the interval.
- **Depends on this:** D-015 and D-016.

### D-015 · Beat-template term dropped from the composite
- **Date / commit:** 2026-09-21 · `a0ca800`
- **Status:** adopted
- **Decision:** the composite SQI is spectral concentration alone. The template correlation is still computed and reported as a component.
- **Evidence:** pooled over non-transient windows, the composite without the template term scores AUROC 0.722, inside the full composite's 95% CI of [0.691, 0.762] around 0.729 (`results/sqi_validation.csv`). Template correlation alone is 0.660 [0.60, 0.71].
- **Rejected:** keeping a term that adds nothing measurable. It also barely varies — 0.80 to 0.86 from sitting to walking (`results/02_sqi_by_activity.csv`) — because beat detection still finds self-similar peaks in motion-corrupted windows.
- **Note on out-of-band power:** out-of-band power was **never** part of the composite, so nothing was dropped beyond what the decision rule authorised. At introduction (`41feb4e`) the composite was `clip(template_corr, 0, 1) * spectral_concentration`; `a0ca800` changed only that expression. Out-of-band has always been a reported component. See D-023 for the open question of whether it should be added.

### D-016 · SQI must not trigger tracker resets under motion
- **Date / commit:** 2026-09-21 · `97674e7`, `32ebfaf`
- **Status:** adopted (constraint); implementation pending
- **Decision:** Stage 4's peak tracker may not use SQI-triggered resets during stairs, table soccer, cycling or walking. A motion-aware confidence term is required instead.
- **Evidence:** AUROC of the shipped composite (`results/sqi_validation.csv`): stairs 0.429 [0.334, 0.521], cycling 0.600 [0.482, 0.704], walking 0.537 [0.428, 0.631] — all CIs include chance; table soccer 0.448 [0.406, 0.484], a CI entirely **below** 0.5, meaning higher SQI predicts larger error. Against sitting 0.895 [0.865, 0.921] and working 0.797.
- **Mechanism:** spectral concentration measures how sharp the dominant peak is, not whether it is the heart. A motion-locked window can have a very sharp peak at the stride frequency.
- **Depends on this:** the Stage 4 tracker design. Activity labels do not exist at inference time, so the replacement must be derived from the signals — for example agreement between the PPG peak and the accelerometer spectrum.

---

## Accelerometer clipping

### D-017 · Clipping carried as both a flag and a continuous fraction
- **Date / commit:** 2026-09-21 · `e5ab84b` (flag), `f93d381` (fraction)
- **Status:** adopted
- **Decision:** every window carries `acc_clipped` and `clip_fraction`, set in Stage 1 so all later stages share one definition.
- **Evidence:** flagged-window rates are extremely uneven — table soccer 47.5%, cycling 39.0%, walking 10.9%, stairs 10.7%, driving 10.0%, lunch 1.6%, working 0.7%, sitting 0.1% (`results/clip_fraction_by_activity.csv`). The E4 saturates at ±2 g.
- **Rejected:** a boolean alone. It cannot distinguish one saturated sample from a window that is 24% saturated, and ACC-referenced methods will need the degree.

### D-018 · Correction: the clipping causation claim was retracted
- **Date / commit:** 2026-09-21 · `f5002ed`
- **Status:** adopted (supersedes the claim made in `91deb5e`)
- **Decision:** the earlier README sentence that saturation "marks vigorous motion rather than causing error" is removed. `b2_noclip` is a descriptive column only.
- **Evidence:** b2 computes its estimate from the band-passed BVP and never reads the accelerometer (`src/models/baselines.py`), so scoring it with and without clipped windows cannot test what clipping causes. The observation that prompted the claim — pooled MAE 20.06 → 18.79 when flagged windows are dropped — remains in `results/baselines_per_activity.csv` and means only that flagged windows are harder on average.
- **Depends on this:** the comparison is repeated at Stage 4, where the estimator does read the accelerometer and the question becomes answerable.

---

## Findings handled as decisions

### D-019 · Correction: lunch and driving are not near-stationary
- **Date / commit:** 2026-09-21 · `709481f`
- **Status:** adopted (supersedes the framing in the pipeline spec)
- **Decision:** the README no longer describes lunch as near-stationary. b2 loses to the b0 constant on **five of eight** activities: stairs, table soccer, driving, lunch and walking.
- **Evidence:** lunch b2 15.50 vs b0 13.70; driving 14.96 vs 14.11 (`results/baselines_per_activity.csv`, `mae_mean_of_folds`). Working is the one genuinely quiet high-duration activity, where b2-zp is 8.62 against b0's 17.02.
- **Rejected:** the spec's assumption that pooled error is flattered mainly by lunch and working being stationary. Half of it is, and the other half is wrong.

### D-020 · S5 retained in every result, with its labels verified
- **Date / commit:** 2026-09-21 · `270dc70`, `22dd67c`
- **Status:** adopted
- **Decision:** S5 — the worst subject at 45.87 bpm b2-zp MAE — is excluded from nothing. Its heart-rate labels were verified rather than assumed.
- **Evidence:** resting cross-sensor agreement, ECG median 92.7 bpm against PPG 91.9 over 300 sitting windows; three ECG strips from high-rate, low-motion, unclipped windows show R-peaks on QRS complexes with T waves unmarked, RR-implied rates matching labels to 0.1 bpm, RR CV ≈ 0.01, and 0.00% short/long alternation across all 2,512 high-rate windows against 0.36% for S10 (`results/s5_investigation.md`, `figures/s5_ecg_check.png`).
- **Rejected:** treating b1's low oracle error as evidence the labels are correct — it shows only that they are smooth. Also rejected: excluding S5, and asserting a physiological cause for the elevated rate.
- **Depends on this:** S5 is the subject that will demonstrate whether Stage 4 motion handling works.

### D-021 · Hero figure subject chosen by rule
- **Date / commit:** 2026-09-21 · `40f5115`
- **Status:** adopted
- **Decision:** the motion-collision figure uses the subject whose stairs b2-zp MAE is closest to the cohort median — S4 at 36.24 bpm against a median of 36.24 — and the rule is stated in the script and the caption.
- **Evidence:** `src/eval/plot_motion_collision.py::choose_subject`. On stairs, S4's estimate lands within 10 bpm of an accelerometer line in 70.8% of windows and of the true HR in 10.8%; sitting inverts to 31.7% and 88.3%.
- **Rejected:** picking the most dramatic subject. Also rejected: drawing the ACC 2× harmonic as the spec asked — 0% of S4's stairs and walking windows lock to 2×, while 47% and 35% lock to the 0.5× stride subharmonic, so the figure draws 0.5× instead.

---

## Open

### D-022 · Floor-pinning is a distinct failure mode
- **Date / commit:** — · raised 2026-09-21 in `270dc70`
- **Status:** superseded by D-024
- **Decision:** none yet. Some estimates are pinned at the bottom of the search band rather than locked to a motion frequency, and the two need separating before either is fixed.
- **Evidence:** S5's five worst windows all read exactly 30.0 bpm, the lowest in-band FFT bin, against a true HR near 178 — not a stride frequency. S5 produces an estimate ≤45 bpm in 27.8% of windows, within a cohort range whose median is 16.1% (`results/s5_investigation.md`).
- **Next step:** an error taxonomy — floor-pinned, accelerometer-locked including harmonics, other — before Stage 4a. Any floor imposed must be justified physiologically or chosen inside each fold on training subjects only, and reported as its own ablation step rather than folded into a motion-handling gain.

### D-023 · Whether out-of-band power should enter the composite
- **Date / commit:** — · raised 2026-09-21 in `97674e7`; resolved 2026-09-23 · `c855492`
- **Status:** adopted — kept as a candidate input to the Stage 4 confidence term, on marginal evidence
- **Decision:** none yet. Out-of-band power is reported as a component and is not part of the composite (see D-015).
- **Evidence:** alone it scores AUROC 0.652 [0.623, 0.680] pooled over non-transient windows, below the composite's 0.722 [0.686, 0.750], but it is measured on the unfiltered window and so carries information the in-band concentration cannot (`results/sqi_validation.csv`). Per activity it beats the composite on table soccer (0.553 vs 0.448) and stairs (0.439 vs 0.429) — the two activities where the composite fails.
- **Outcome (2026-09-23):** the two candidate activities were re-tested with the `validate_sqi` bootstrap across five seeds. **Table soccer:** out-of-band 0.553, CI lower bound 0.500–0.503 depending on seed — it excludes chance in all five, but only just. **Stairs:** out-of-band 0.439 with an upper bound of 0.497–0.501, straddling chance, against the composite's 0.429 — no real difference, as expected, and both sit below chance.
- **Decision:** out-of-band power is kept as a *candidate input* to the Stage 4 confidence term, not as evidence that it works. One activity with an effect of 0.553 whose interval touches 0.500 is weak support; it must not be relied on alone, and PPG–ACC spectral agreement remains the primary route.
- **Next step:** test any combination the same way — within activity, subject-bootstrapped, weights chosen inside each fold rather than on the full cohort.

### D-024 · The floor hypothesis is rejected; a lower search bound is proposed on different evidence
- **Date / commit:** 2026-09-23 · `c855492` (supersedes D-022)
- **Status:** implemented 2026-09-26 as D-039; the proposal below stands as written
- **Decision:** floor-pinning as described in D-022 does not exist at cohort scale and that line of work is dropped. Separately, a lower search bound of **40 bpm** is *proposed* on different evidence, to be ablated on its own rather than folded into any motion-handling gain.
- **Evidence against D-022:** windows whose estimate sits within 2 bpm of the lowest in-band frequency (24.375 bpm on the b2-zp grid) are **0.2%** of the 17,485 non-transient error windows — below the 10% stop threshold (`results/error_taxonomy.csv`). D-022 generalised from five S5 windows that all read exactly 30.0 bpm, which is the lowest bin of b2's **coarse 512-point grid**; on b2-zp's 4,096-point grid the pile-up at the exact bin does not survive. The failure those windows showed is real but belongs to `acc_locked` or `other`, not to a distinct floor class.
- **Evidence for a 40 bpm bound:** 7.03% of all non-transient windows produce an estimate in [24, 40) bpm, and **100% of estimates below 40 bpm are errors over 10 bpm**, while the lowest ECG label anywhere in the cohort is 41.7 bpm and only 0.53% of labels fall below 45 (`results/error_taxonomy.csv`, true-HR percentiles per activity).
- **Argument, in one sentence:** the 24–40 bpm region of the search band contains no recoverable heart rate for this cohort and only spurious peaks, so excluding it can only remove errors — but the bound is a property of *this* population, not of physiology, and would be wrong for a bradycardic, athletic or beta-blocked cohort.
- **Rejected:** applying the bound now. It would improve the numbers before the motion work it is meant to be measured against, and a cohort-derived threshold chosen on all 15 subjects is exactly the leak the evaluation design forbids.
- **If implemented:** as a distinct component with its own ablation row, with any data-derived value computed inside each LOSO fold on training subjects only, and with the population caveat stated in the README.
- **Depends on this:** nothing yet.

### D-025 · The dominant failure is accelerometer lock, and two thirds of errors are unexplained
- **Date / commit:** 2026-09-23 · `c855492`
- **Status:** adopted (finding; sets Stage 4 priorities)
- **Decision:** Stage 4 targets accelerometer lock first, and the unexplained majority is tracked as an open measurement rather than assumed to be noise.
- **Evidence:** of 17,485 non-transient windows with |error| > 10 bpm — `acc_locked` 23.3%, `harmonic` 10.2%, `floor` 0.2%, `other` **66.2%** (`results/error_taxonomy.csv`). Accelerometer lock concentrates where expected: 55.1% of walking errors and 38.9% of stairs errors, against 10–16% elsewhere. Within it, **58% sit on the 0.5× stride subharmonic**, 39% on the step fundamental and 3% on 2× — the same asymmetry the hero figure shows (D-021).
- **On `other`:** it exceeds the 40% threshold the scope set, and is reported without widening the tolerances. It is not random: 91.9% of `other` estimates fall *below* the true HR, 64.8% are under 60 bpm, and 31.9% lie within 10 bpm of half the true rate — just outside the ±3 bpm harmonic rule. Loosening the accelerometer tolerance to ±8 bpm would absorb 22.4% of it and to ±20 bpm, 60.6%, which suggests much of `other` is motion lock that the single dominant accelerometer peak does not capture, rather than a separate mechanism.
- **Also:** `acc_locked` (0.485), `harmonic` (0.495) and `other` (0.491) all have a median SQI within 0.01 of the 0.490 median across all error windows, so the current quality measure separates none of the three classes that matter — consistent with D-016. Only `floor` sits lower, at 0.440, and that class is 35 windows.
- **Correction (2026-09-25, `2a40cd2`):** the widened-tolerance sentence above is withdrawn in part. A permutation null — each window's estimate paired with another window's ACC spectrum, same activity, 20 shuffles — gives a chance lock rate of 13.6% at ±3 bpm, 31.7% at ±8 and **67.3% at ±20** (`results/acc_lock_permutation.csv`). So the ±3 headline survives but the true excess is ~10pp, not 23.3%; ±8 carries information only for stairs (+24.4pp) and walking (+6.0pp); and **±20 carries none at all** (69.1% observed against a 67.3% null). The "±20 bpm would absorb 60.6% of `other`" figure is not used anywhere.

### D-026 · Accelerometer-lock attribution is tested against a permutation null
- **Date / commit:** 2026-09-25 · `2a40cd2`
- **Status:** adopted
- **Decision:** any "the estimate sits on a motion line" claim is reported against the rate the same rule produces on mismatched pairs, and tolerances whose observed rate does not clear that null are not used.
- **Evidence:** `results/acc_lock_permutation.csv`. Chance rates are 13.6% (±3 bpm), 31.7% (±8) and 67.3% (±20) against observed 23.3%, 40.4% and 69.1%.
- **Rejected:** quoting raw within-tolerance shares. Three harmonic lines and a ±20 bpm window cover most of the searchable band, so a large number there means nothing.
- **Depends on this:** D-025's corrected wording, and the masking claim in D-030 — post-masking lock of 13.1% is meaningful only because the chance rate is known to be 13.6%.

### D-027 · Motion spectrum from four channels; notch depth is a power gain
- **Date / commit:** 2026-09-25 · `049e134`
- **Status:** adopted
- **Decision:** the motion spectrum sums the power spectra of the three wrist-ACC axes and of the magnitude signal. Masking notches the top-K prominent peaks and each peak's 0.5× and 2× harmonics, with notch σ proportional to the measured width of that ACC peak and a multiplicative Gaussian attenuation rather than zeroing. K, prominence, width factor, depth and a motion gate are chosen in-fold.
- **Evidence:** 58% of confirmed locks sit on the 0.5× stride subharmonic and only 3% on 2× (D-025), so a single-line mask would miss most of the damage. Per-axis spectra depend on how the watch sits on the wrist; the magnitude is orientation-invariant. Selected configurations remove 7.7% of in-band power on average, well under the 30% flag (`results/stage4_selected_configs.csv`).
- **Rejected:** masking the argmax alone; hard-zeroing bins, which destroys a heart rate that coincides with a cadence line; and a fixed notch width. Also rejected: a depth grid bottoming out at 0.1 — **depth is a power gain**, and a cadence line carrying 16× the cardiac power survives it, so the grid reaches 0.02.
- **Also:** windows whose ACC magnitude SD is below a gate are left unmasked. Without it the normalised spectrum of a still wrist is noise and peak-picking invents notches; an early run masked 47% of the spectrum during sitting and made resting error worse.

### D-028 · Tracking selects the nearest prominent peak, and resets on the estimates alone
- **Date / commit:** 2026-09-25 · `049e134`
- **Status:** adopted
- **Decision:** the tracker predicts from a mean filter over recent estimates, selects the prominent spectral peak nearest that prediction with no hard search window, and re-seeds from the unconstrained argmax after `reset_after` consecutive windows whose chosen peak is more than `jump_bpm` from the prediction. No quality measure is involved.
- **Evidence:** D-016 forbids SQI-gated resets under motion, and this rule needs none. The design matters: a hard ±12 bpm window with the jump test on the global argmax scored 17.53 → the same grid with nearest-peak selection reached the numbers in D-030, because under motion the argmax disagrees with the prediction almost every window, so the hard-window variant reset continuously and behaved like no tracker at all.
- **Rejected:** testing the jump against the *constrained* choice. A unit test caught it: the track then creeps by up to `jump_bpm` per window without ever registering a jump, following a genuine 80 → 160 bpm change only as far as 102 (`tests/test_stage4.py::test_tracker_resets_after_persistent_disagreement`).
- **Depends on this:** the whole Stage 4 result. Tracking contributes most of the combined gain.

### D-029 · No taper on the PPG spectrum, so the ablation isolates masking
- **Date / commit:** 2026-09-25 · `049e134`
- **Status:** adopted, with a known cost
- **Decision:** the masked and tracked variants take the same plain FFT as b2-zp. A window function is not applied.
- **Evidence:** tapering reduces spectral leakage and would improve the estimate on its own, so folding it into "masking" would credit motion handling with a gain that a window function delivered.
- **Cost, stated:** leakage from a strong motion line spreads well beyond its own bin, so a narrow notch cannot remove it — visible in `tests/test_stage4.py::test_masking_recovers_hr_when_motion_dominates`, where a 4× cadence needs a notch both deep and wide before the cardiac peak wins. A Hann taper is a candidate improvement and, if adopted, becomes its own ablation row rather than being absorbed into 4a.

### D-030 · Stage 4 result: tracking carries the gain, masking makes it possible, stairs and cycling get worse
- **Date / commit:** 2026-09-25 · `78608f6`
- **Status:** adopted
- **Decision:** the reported method is masking plus tracking. The tracker alone is **not** reported as a result.
- **Evidence** (`results/stage4_ablation.csv`, `results/stage4_per_subject.csv`, paired subject bootstrap): pooled MAE including transients b2-zp 18.98 → +tracker 17.53 → +mask 18.61 → +mask+tracker **15.80**. Paired gains: tracker alone +1.45 bpm [−1.45, +4.65], 9/15 subjects improved — **CI spans zero, within per-subject variability**; masking alone +0.36 [+0.04, +0.72]; together +3.18 [+1.00, +5.83], 11/15 improved.
- **Against published work:** SpaMa 15.56 and SpaMaPlus 11.06 on the same dataset and protocol (Reiss et al. 2019, Tables 4 and 10). Our combination matches SpaMa, which 4a reimplements, and is 4.7 bpm short of SpaMaPlus; per subject we beat SpaMa on 5/15 and SpaMaPlus on 0/15.
- **Where it fails:** stairs 38.17 → 56.01 and cycling 29.75 → 36.96, both far worse than the baseline. Tracking assumes the previous estimate is informative; under sustained vigorous motion the spectrum offers a stable wrong peak and the tracker holds it. The same mechanism halves error on lunch (14.26 → 6.64) and working (8.62 → 4.92).
- **Mechanistic confirmation:** `acc_locked` falls from 23.3% of error windows to 13.1% cohort-wide and 55.1% → 26.1% in walking, landing at the 13.6% chance rate of D-026 — masking removes essentially all real lock. Total error windows fall only 2.4%, so the residual failure is not accelerometer lock (`results/masked_taxonomy.csv`).
- **On hyperparameter choice:** selecting in-fold rather than globally costs 2.51 bpm for the tracker and 1.40 for the combination, and only 7/15 and 9/15 folds pick the global best (`results/stage4_selected_configs.csv`). The globally chosen figures are not reported as results.

### D-031 · Adaptive cancellation deferred; confidence term not attempted
- **Date / commit:** 2026-09-25 · `049e134`
- **Status:** open — deferred to Week 3 hardening
- **Decision:** NLMS and batch least-squares cancellation are implemented and unit-tested but not evaluated in the ablation, and the motion-aware confidence term is not attempted.
- **Reason:** the session's priority order put masking and tracking first, with 4b named as the cut. The published precedent reaches 11.06 with masking and tracking alone, so the ablation's story stands without it. The confidence term was optional and is unnecessary for the reset rule adopted in D-028, which reads only the estimates.
- **What exists:** `src/models/adaptive.py`, vectorised across windows so a subject's 4,600 windows filter in about a second, with tests covering reference removal and the absence of state leaking across windows. `src/eval/stage4.py` already wires `+adaptive` and `+adaptive+tracker` rows; they are produced by running it without `--no-adaptive`.
- **When it runs:** the clipping comparison retracted in D-018 becomes meaningful there — table soccer (47.5% of windows flagged) and cycling (39.0%) are the test cases for a distorted reference degrading cancellation.

### D-032 · The cardiac peak survives masking: the residual failure is selection
- **Date / commit:** 2026-09-26 · `03236e8`
- **Status:** adopted (finding)
- **Decision:** the residual Stage 4 error is treated as a selection problem, not as destroyed signal, and effort goes to the selection rule rather than to recovering the waveform.
- **Evidence:** for masked windows still in error, a peak within 3 bpm of the true HR exists in 58.8% of them against a 36.3% chance rate from the D-026 null — +22.5 points (`results/surviving_peak.csv`). The excess is largest where the method fails worst: cycling 78.1% vs 34.1% chance, stairs 65.6% vs 36.8%. Among surviving peaks, 34.9% are rank 2 and 24.2% rank 3, so three in five are in the top three candidates.
- **Rejected:** the alternative reading, that motion destroys the cardiac component and no single-window spectral method could recover it. If that were true the share would sit at chance; it does not.
- **Depends on this:** D-033 and the Week 3 tracker work. It also means a better selection rule has measurable headroom rather than speculative headroom.

### D-033 · The method trades many short errors for a few very long ones
- **Date / commit:** 2026-09-26 · `03236e8`
- **Status:** adopted (finding); mitigation open, see D-037
- **Decision:** error persistence is reported beside MAE, and the trade is stated as a cost rather than folded into the headline.
- **Evidence** (`results/error_persistence.csv`): b2-zp produces 3,574 error runs, median 3 windows, p90 10, longest 99, with 27.6% of error windows in runs over 30 s. mask+tracker produces 2,876 runs, median 1, p90 6, longest **409** (about 13 minutes), with **61.8%** of error time in runs over 30 s. On stairs p90 goes 20 → 210 windows.
- **Mechanism:** the reset triggers on jumps, and a smoothly tracked wrong estimate never jumps. Resets fire 0.3 times per 1,000 windows on stairs and 1.4 on cycling, and not at all on most activities.
- **Consequence:** for continuous monitoring a shorter mean error with longer episodes may be the worse product. The README states this rather than reporting the MAE gain alone.

### D-034 · Week 3: six-window mean prediction for the tracker
- **Status:** open — expected effect: closes part of the 4.7 bpm gap to SpaMaPlus
- **Decision:** none yet. SpaMaPlus predicts from a mean over the last six estimates; this tracker's mean filter is dominated by the most recent estimate in practice, and that is the one named difference between the two.
- **Target:** SpaMaPlus 11.06 bpm pooled (Reiss et al. 2019, Table 4) against our 15.80.
- **Note:** a change to the prediction rule does not affect b2-zp, so it can be evaluated without invalidating the baselines.

### D-035 · Week 3: Hann taper on the PPG spectrum
- **Status:** open — expected effect: better masking, at the cost of every baseline number
- **Decision:** none yet. See D-029: leakage from strong motion lines is wider than a notch can follow, and a taper would narrow it.
- **Constraint:** tapering changes b2-zp itself, so it invalidates every baseline and ablation figure in the repository. It must not be touched before the ship, and when attempted it requires regenerating the whole results tree in one commit.

### D-036 · Week 3: adaptive cancellation (4b) remains deferred
- **Status:** open — carried over from D-031
- **Note:** implemented and unit-tested in `src/models/adaptive.py`, wired into the ablation, never evaluated. The clipping comparison retracted in D-018 becomes meaningful when it runs.

### D-037 · Week 3: a reset that detects sustained wrongness rather than jumps
- **Status:** implemented 2026-09-26 as D-040
- **Decision:** none yet. This falls directly out of the persistence result: the current reset asks "did the estimate move?", when the failure is an estimate that does not move and is wrong.
- **Candidate signals:** agreement between the tracked peak and the accelerometer harmonic family, the rank of the tracked peak in the current spectrum (D-032 shows the true peak is usually rank 2 or 3), or time since the last reset.
- **Constraint:** any such signal faces the D-014 bar — within-activity AUROC against error, subject-bootstrapped CIs — before it gates anything.

### D-038 · Week 3: split the README, keep the front page to result, figure, table, limitations
- **Status:** open
- **Decision:** none yet. Stage-by-stage detail moves to `docs/`, the README keeps the headline result, the motion-collision figure, the per-activity table and the limitations.
- **Reason:** the README now carries the whole pipeline narrative and is long past the eight-minute reading path the project is optimised for.

### D-039 · The search bound, derived inside each fold
- **Date / commit:** 2026-09-26 · `52e65b1` (implements D-024)
- **Status:** adopted, as a separately ablated component and with a disclosure
- **Decision:** selection is restricted to frequencies at or above a bound computed **inside each fold** as `min(training-subject labels) − margin`, with the margin chosen on training subjects from {0, 5, 10} bpm. The rule lives in `fold_bound()`. b2-zp is left untouched as the reference.
- **Measured:** every fold chose a zero margin, giving 41.7 bpm for fourteen folds and 41.9 for the fold holding out the subject who carries the cohort minimum — so the held-out subject genuinely never contributes to its own bound. It alters **5,555 of 64,697 windows (8.6%)**, between 136 (S7) and 779 (S5) per subject (`results/week3_bound_effect.csv`). Alone it is worth +2.04 bpm [+1.57, +2.63] and **improves all 15 subjects**, the only component here that does.
- **Disclosure:** SpaMa and SpaMaPlus search the full band, so this is a departure from the published protocol. The repository reports **15.80 bpm without the bound** whenever comparing to published numbers and 12.50 with it.
- **Caveat, carried unchanged from D-024:** this is a property of *this* cohort, not of physiology. It would be wrong for a bradycardic, athletic or beta-blocked population, and any deployment would need its own bound or none.

### D-040 · The sustained-wrongness reset is a height-ratio test, not a rank test
- **Date / commit:** 2026-09-26 · `52e65b1` (implements D-037)
- **Status:** adopted
- **Decision:** alongside the jump reset, the track is re-seeded when the tracked peak holds less than 30% of the spectral maximum's height for five consecutive windows. The hard rank test is implemented and reported but not used.
- **Evidence:** the ratio formulation was selected by **15/15 folds** and beats the best rank variant by 1.3 bpm pooled (12.50 against 13.82); all twelve variants are scored in the run output. Rank is the coarser signal — a peak can fall to rank 3 while still being nearly as tall as the maximum, and can be rank 1 in a spectrum with no real structure.
- **Degeneracy check:** resets fire **4.7 times per 1,000 windows** cohort-wide, peaking at 9.6 on stairs — about one window in 104, far below the one-in-five threshold that would indicate the tracker had collapsed into a no-op (`results/week3_reset_rates.csv`). That was the failure that cost 2.7 bpm in D-028, and it is now checked explicitly.
- **Effect on persistence (D-033's numbers):** cohort-wide, error time in runs over 30 s falls from 61.8% to **54.1%**, the longest episode from 409 windows to 323, and on stairs the p90 run from 210 windows to **79**. Still roughly double the baseline's 27.6%: **reduced, not eliminated**.

### D-041 · Both Week 3 components were designed after seeing test-set results
- **Date / commit:** 2026-09-26 · `52e65b1`
- **Status:** adopted (disclosure)
- **Decision:** recorded explicitly, as was done for the tracker revision in D-028.
- **What happened:** the bound was motivated by the cohort's single worst window (S5 cycling, true 173 bpm, estimate 30) and by D-024's finding that estimates under 40 bpm are always wrong — both observed on the full dataset. The reset was motivated by the persistence diagnostic, also computed on the full dataset. **Every hyperparameter is still chosen inside the fold**, so the fitted quantities are clean, but the *design choice* of what to build was informed by the test set.
- **What this means for the numbers:** the in-fold figures are not optimistically biased in the usual sense, but neither are they a blind evaluation of a pre-registered method. A genuinely held-out cohort is the only way to settle that, and this dataset cannot provide one.
- **Rejected:** presenting the Week 3 gain as if the components had been specified in advance.

### D-042 · The per-activity regression was understated in the shipped README
- **Date / commit:** 2026-09-26 · `52e65b1`
- **Status:** adopted (correction)
- **Decision:** the claim that mask+tracker was "worse than predicting a constant on stairs and cycling" is corrected: it was worse than the b0 constant on **four** activities — stairs (56.01 vs 30.95), table soccer (26.73 vs 14.02), walking (25.18 vs 15.46) and cycling (36.96 vs 33.74).
- **How it happened:** the Week 2 brief named stairs and cycling, and that pairing was carried into the README without checking the remaining activities against `results/baselines_per_activity.csv`. Stairs and cycling were the activities worse than *b2-zp*; four were worse than *b0*.
- **After Week 3:** cycling now beats b0 (21.63 vs 33.74). Stairs, table soccer and walking still do not (`results/week3_vs_b0.csv`).

### D-043 · Diagnostic A is unchanged by the Week 3 components
- **Date / commit:** 2026-09-26 · `52e65b1`
- **Status:** adopted (finding)
- **Decision:** the selection headroom identified in D-032 is recorded as still open after Week 3.
- **Evidence:** among the errors the Week 3 method leaves behind, a peak within 3 bpm of the true HR survives in **60.4%** of windows, against 58.8% for the masked baseline (`results/week3_surviving_peak.csv`). MAE improved by 6.48 bpm, but what remains is the same failure in the same proportion.
- **Reading:** the bound and the reset removed error *episodes* without changing the character of the residual error. A better selection rule — not a better filter — is still the open direction.
