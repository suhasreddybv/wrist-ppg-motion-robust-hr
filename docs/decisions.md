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
- **Status:** proposed — not applied; to be implemented, if at all, as its own Stage 4 component
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
- **Also:** every class has a median SQI within 0.01 of the 0.490 median across all error windows, so the current quality measure separates none of them — consistent with D-016.
