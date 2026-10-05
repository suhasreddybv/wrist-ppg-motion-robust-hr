# Agreement, strata and error structure

Bland-Altman and Pearson r per activity, Fitzpatrick stratification, and the published-benchmark comparison in full.

*Part of [wrist-ppg-motion-robust-hr](../README.md). Numbers here come from the CSVs in `results/`.*

---

## Agreement, strata and error structure (Stage 5)

**Agreement with the ECG reference** (`results/agreement.csv`), non-transient windows: pooled Pearson r is 0.399 for b2-zp and 0.402 for mask+tracker, with Bland–Altman bias improving from −14.74 to −11.66 bpm and limits of agreement from [−66.3, +36.9] to [−58.8, +35.4]. The pooled correlation barely moves because between-activity spread dominates it; per activity the change is large — driving r 0.368 → 0.699, lunch 0.283 → 0.779, working 0.336 → 0.725, while stairs falls 0.196 → 0.130. The bias is negative everywhere: this estimator systematically *under*-reads, which follows from motion lines sitting below the cardiac rate.

![Bland–Altman plot of mask+tracker against the ECG reference](figures/bland_altman.png)

**Fitzpatrick skin type** (`results/skin_type.csv`). The cohort has one type-2 subject, eleven type-3 and three type-4, **so this establishes almost nothing statistically** and is reported per subject rather than as a stratum mean that would imply precision it does not have. It is here because almost no published work on this dataset reports it at all.

| Type | Subjects | b2-zp | mask+tracker |
|---|---|---|---|
| 2 | S15 | 14.38 | 13.86 |
| 3 | S1, S2, S3, S5, S6, S7, S8, S11, S12, S13, S14 (n=11) | 8.75–45.87 (mean 18.94) | 10.40–28.74 (mean 15.98) |
| 4 | S4, S9, S10 (n=3) | 14.51–26.05 (mean 20.64) | 7.50–24.25 (mean 15.77) |

Within-stratum spread dwarfs any difference between strata: the type-3 subjects alone span 8.75 to 45.87 bpm. There are **no type V or VI subjects at all**, so nothing here speaks to the melanin confound that matters most in optical heart rate.

