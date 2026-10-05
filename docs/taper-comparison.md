# Hann taper: measured, not adopted

Dated snapshot, 5 October 2026. The repository's canonical numbers use a **rectangular**
analysis window (a plain FFT). This records what a Hann taper does, and why it was not made
canonical.

Switch: `PPG_TAPER=1` routes every 8 s FFT in the package through `analysis_window()` — the
PPG spectra for b2 and b2-zp, the masked spectra every Stage 4 method uses, and both SQI
spectra. Accelerometer spectra are tapered either way; they only locate motion lines and are
never compared against a baseline.

## The delta table

| Quantity | Rectangular | Hann | Δ |
|---|---|---|---|
| baseline b0 | 18.57 | 18.57 | 0.00 |
| baseline b1 (oracle) | 1.49 | 1.49 | 0.00 |
| baseline b2 | 20.06 | 19.67 | −0.39 |
| baseline b2-zp | 18.98 | 19.14 | +0.16 |
| +tracker | 17.53 | 15.19 | **−2.34** |
| +mask | 18.61 | 19.13 | **+0.52** |
| +mask+tracker (6-window prediction) | 15.80 | 15.25 | −0.55 |
| +mask+tracker (15-window prediction) | 14.70 | 15.25 | +0.55 |
| prediction rule selected in-fold | **14.70** | **14.81** | +0.11 |
| Diagnostic A, surviving-peak share | 66% | 40% | −26 pp |

## What it shows

**The headline does not move.** With the prediction rule selected in-fold the two agree to
0.11 bpm, and no baseline moves by more than 0.39 bpm — well inside the 1 bpm threshold that
would have forced a decision either way.

**But the attribution changes a lot.** The tracker is worth 2.34 bpm more under a taper, and
masking is worth 0.52 bpm *less*. That is the mechanism D-029 predicted: leakage from a strong
motion line spreads past any notch a mask can place, so part of what masking appeared to
contribute was leakage mitigation that a window function does more cheaply. Under a taper the
cadence line is narrow enough that tracking alone handles most of it.

**Diagnostic A falls from 66% to 40%**, its largest movement from any single change. With less
leakage, fewer spurious peaks survive near the true rate, so the residual errors contain the
right answer less often. The selection headroom is partly a leakage artefact.

## Why it is not canonical

The repository closes today. Making the taper canonical means regenerating every CSV and
figure and re-verifying every README number end to end; a partially migrated results tree —
some numbers tapered, some not — is worse than either state, and that is the realistic outcome
of starting the migration in the last hours of the block. Since the headline is unchanged, the
migration would buy consistency with a standard practice, not accuracy.

It is therefore reported as a comparison, with the pre-taper numbers canonical and the reason
stated. The switch is committed and defaulted off, so adopting it later is a one-line change
plus a full regeneration.

**What would change this:** any future work that touches the spectra at all should turn the
taper on first and regenerate everything in one commit, because the attribution results above
— not the headline — are what a reader of the ablation would be misled by.
