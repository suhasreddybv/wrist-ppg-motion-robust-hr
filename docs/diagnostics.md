# Diagnostics

Two questions that decide what the result means: does masking preserve the cardiac information, and how long does an individual error last? Plus what Week 3 and Viterbi did to both.

*Part of [wrist-ppg-motion-robust-hr](../README.md). Numbers here come from the CSVs in `results/`.*

---


## Does the cardiac peak survive masking?

Yes — the information is there and the problem is selection. For masked windows still in error, the masked spectrum contains a peak within 3 bpm of the true heart rate **58.8% of the time, against a 36.3% chance rate** built the same way as the accelerometer-lock null: +22.5 points of real excess (`results/surviving_peak.csv`). The excess is largest exactly where the method fails — **cycling 78.1% against 34.1% chance, stairs 65.6% against 36.8%**.

That peak is essentially never the largest one (rank 1: 0.0%, by construction — if it were, the window would not be in error), but it is **rank 2 in 34.9% of cases and rank 3 in 24.2%**. Roughly three in five surviving peaks are in the top three. So a better selection rule has somewhere to go, and this is a selection problem rather than a destroyed-signal problem. The worst window in the cohort makes it concrete:

![Best and worst windows of the full method: BVP trace and masked spectrum](../figures/best_worst_windows.png)

The bottom row is S5 cycling at a true 173 bpm. The cardiac peak is plainly present in the masked spectrum, and the tracker returned 30 bpm.

## How long does an error last?

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

### Did it fix what it targeted?

**Per activity against the b0 constant** (`results/week3_vs_b0.csv`):

| Activity | b0 | mask+tracker | Week 3 | |
|---|---|---|---|---|
| sitting | 29.28 | 2.63 | **2.44** | beats b0 |
| working | 17.02 | 4.92 | **4.67** | beats b0 |
| driving | 14.11 | 7.50 | **7.61** | beats b0 |
| lunch | 13.70 | 6.64 | **7.41** | beats b0 |
| cycling | 33.74 | 36.96 ✗ | **21.63** | **now beats b0** |
| stairs | 30.95 | 56.01 ✗ | 35.84 | still worse |
| table soccer | 14.02 | 26.73 ✗ | 21.69 | still worse |
| walking | 15.46 | 25.18 ✗ | 22.05 | still worse |

**Cycling is fixed — 36.96 → 21.63, from worse than a constant to comfortably better. Stairs, table soccer and walking are not**, though stairs improves by 20 bpm. Three of eight activities still lose to guessing the mean.

**Error persistence** (`results/week3_persistence.csv`), the other target:

| | Runs | Median | p90 | Longest | Error time in runs > 30 s |
|---|---|---|---|---|---|
| b2-zp | 3,574 | 3 | 10 | 99 | 27.6% |
| mask+tracker | 2,876 | 1 | 6 | 409 | 61.8% |
| **Week 3** | 2,965 | 2 | 7 | **323** | **54.1%** |

Improved but not solved. On stairs the p90 run falls from **210 windows to 79** and on cycling from 48 to 12; cohort-wide, long-run error time falls from 61.8% to 54.1%. It remains double the baseline's 27.6%, so **the persistence trade is reduced, not eliminated**.

**Reset rate** (`results/week3_reset_rates.csv`): 4.7 per 1,000 windows cohort-wide, peaking at 9.6 on stairs — roughly one window in 104, far below the one-in-five degeneracy threshold. The tracker has not collapsed into a no-op.

**Diagnostic A on the new residual errors: still unconsumed, and rising.** A peak within 3 bpm of the true HR survives in **60.4%** of the errors the bound and reset leave behind, against 58.8% before — and **65.7%** of those the 15-window prediction leaves behind. Every component so far has reduced error magnitude while leaving the selection failure intact, and the share of residual errors that contain the right answer keeps going up. Greedy per-window selection cannot use that information; a method that decodes the whole sequence at once can, which is what comes next.

