# On a wrist: the inference budget

Week 7, on the frozen mask+tracker(hist=15) pipeline (D-046), LOSO, every fold at the configuration it selected on its training subjects (`results/edge_fold_configs.csv`, reconciled to the published 14.704 bpm before anything else ran). Decision-log entries D-057 to D-059. Every number here is generated from `results/edge_*.csv` by `src/edge/write_docs.py`.

## What was asked, and why there is no ONNX model

The plan asked for an ONNX export, int8 quantisation, latency, model size and accuracy lost per activity, and a one-page budget for a five-minute update interval on a two-year battery. This pipeline has no learned weights: a 4th-order Butterworth band-pass, a 4,096-point spectrum of each 8-s PPG window, four 2,048-point accelerometer spectra, Gaussian notches at the motion lines and a peak tracker, with a handful of hyperparameters chosen in each fold. ONNX export and post-training int8 quantisation are tools for networks. The equivalent questions for signal processing were answered instead: what fixed-point arithmetic costs, how long a window takes, how much memory it needs, and where the energy goes.

## 1. Arithmetic: what a device would change, and what it costs

Each variant is the frozen pipeline with one class of change (`src/edge/pipeline.py`; the fixed-point simulators in `src/edge/fixed_point.py` are tested against exact references). Paired differences are per subject against causal float64, the first variant a device could actually run; bootstrap over the 15 subjects, 2,000 resamples, seed 42.

| Variant | Pooled MAE, bpm | Minus causal float64, paired [95 % CI] | Subjects worse than causal float64 |
|---|---|---|---|
| Reference: float64, zero-phase band-pass over the record (the published 14.70) | **14.70** | +0.44 [-0.31, +1.20] | 9 of 15 |
| Causal band-pass, float64 | **14.27** | — | — |
| Causal, float32 (native on a Cortex-M4F) | **14.27** | +0.00 [+0.00, +0.00] | 0 of 15 |
| Q15: 16-bit data, 32-bit-state band-pass, Q15 FFTs | **15.08** | +0.81 [-0.75, +2.51] | 10 of 15 |
| Q15 with a 16-bit-state band-pass (the naive form) | **17.12** | +2.85 [+0.99, +5.13] | 12 of 15 |
| 8-bit storage of windows, spectra and gains; ACC at its native 1/64 g | **14.51** | +0.24 [-0.87, +1.49] | 9 of 15 |

**Read it this way.** The published pipeline filters the whole record forwards and backwards (zero-phase), which needs the future; a device filters causally, and that is 14.27 bpm against 14.70 — not worse, and the interval includes zero. Float32, the native arithmetic of a Cortex-M4F, changes nothing to two decimals. Fixed point with 16-bit data and 8-bit storage moves the pooled error by less than a bpm in either direction with intervals spanning zero: the tracker is path-dependent, so any small perturbation flips some lock-in episodes one way and some the other, and the 8-bit variant landing below Q15 is that noise, not an ordering. The one arithmetic choice that is clearly worse is a band-pass that rounds each biquad's output to 16 bits: at a 0.4 Hz cutoff and 64 Hz sampling the poles sit close to the unit circle and amplify the rounding (31 dB signal-to-error in `tests/test_edge_fixed_point.py`, against 127 dB with 32-bit state). The CMSIS-DSP library provides the 32-bit-state form for exactly this case.

Per activity, mean of per-subject MAE (bpm):

| Activity | reference | causal f64 | causal f32 | Q15 | Q15, 16-bit state | 8-bit |
|---|---|---|---|---|---|---|
| sitting | 2.59 | 2.61 | 2.61 | 2.58 | 3.29 | 2.61 |
| stairs | 54.92 | 56.84 | 56.84 | 56.62 | 57.87 | 55.55 |
| table soccer | 27.42 | 24.08 | 24.08 | 27.18 | 33.27 | 28.44 |
| cycling | 36.57 | 34.34 | 34.34 | 35.84 | 40.05 | 36.07 |
| driving | 5.57 | 6.11 | 6.11 | 8.19 | 8.36 | 6.98 |
| lunch | 5.24 | 5.53 | 5.53 | 5.69 | 6.61 | 5.58 |
| walking | 24.75 | 20.99 | 20.99 | 22.75 | 21.87 | 21.91 |
| working | 4.66 | 4.75 | 4.75 | 4.92 | 5.80 | 4.84 |

## 2. Latency and compute

The device path was written as a device runs it (`src/edge/stream.py`): one 2-s step at a time, the band-pass state carried, a 512-sample ring buffer, an incremental tracker. A test asserts it gives the batch pipeline's estimates exactly, and that its tracker equals the original step for step across every configuration in the grids. Timed on one thread on this machine over 1,490 windows of S1 (macOS cannot pin a process to a core; one thread is the stand-in):

| Stage of one streaming step | Median | 95th percentile |
|---|---|---|
| total per window | 163.0 µs | 168.8 µs |
| filter | 16.3 µs | 17.3 µs |
| zscore | 9.9 µs | 10.4 µs |
| ppg_fft | 18.5 µs | 19.3 µs |
| acc_spectra_and_gains | 100.5 µs | 104.3 µs |
| peaks_and_tracker | 16.7 µs | 17.7 µs |

What a Cortex-M4F would execute per window, from a published FFT benchmark and loop sizes read off the code; the cycle costs assumed for the non-FFT arithmetic are stated in `src/edge/latency.py`:

| Work per 8-s window | Cortex-M4F cycles | Basis |
|---|---|---|
| PPG real FFT, 4096 points (CMSIS-DSP 5, Cortex-M4F, published) | 331,857 | FFT4CM4F benchmark |
| ACC real FFTs, 4 x 2048 points (published) | 606,904 | FFT4CM4F benchmark |
| band-pass, 128 new samples x 4 biquads x 5 MAC | 7,680 | loop size from the code x assumed cycles per op |
| z-score, 512 samples (mean, variance, scale) | 4,608 | loop size from the code x assumed cycles per op |
| PPG power spectrum, in-band bins | 2,079 | loop size from the code x assumed cycles per op |
| ACC detrend, magnitude, taper: 512 x 4 signals | 24,576 | loop size from the code x assumed cycles per op |
| ACC power, 4 x 1025 bins, sum and normalise | 36,900 | loop size from the code x assumed cycles per op |
| ACC peak search over the motion band | 5,000 | loop size from the code x assumed cycles per op |
| notch gains: lines x in-band bins x (exp + 4 MAC) | 77,616 | loop size from the code x assumed cycles per op |
| mask multiply and PPG peak search | 5,313 | loop size from the code x assumed cycles per op |
| tracker step | 200 | loop size from the code x assumed cycles per op |
| total per window | 1,102,733 | 17.23 ms at 64 MHz |

That is 17.2 ms per window at 64 MHz, 0.9 % of the 2-s step.

## 3. Memory

RAM the device path holds per window, and the flash for CMSIS-DSP 5's FFT tables (sizes from `arm_common_tables.h`):

| Buffer | float32 | Q15 | 8-bit storage |
|---|---|---|---|
| PPG window ring buffer, 512 samples | 2,048 B | 1,024 B | 512 B |
| real-FFT input and output buffers, 4096 points (reused for the ACC FFTs) | 32,768 B | 24,576 B | 24,576 B |
| ACC window, 3 axes (64 Hz resampled; q8 keeps the native 32 Hz, 1/64 g) | 6,144 B | 3,072 B | 768 B |
| motion power spectrum, 1025 bins | 4,100 B | 2,050 B | 1,025 B |
| in-band PPG spectrum and notch gains, 2 x 231 bins | 1,848 B | 924 B | 462 B |
| band-pass state (4 biquads) and tracker history (15) | 92 B | 124 B | 124 B |
| RAM total | 47,000 B | 31,770 B | 27,467 B |
| FFT tables in flash (CMSIS-DSP arm_common_tables.h sizes) | 60,368 B | 47,936 B | 47,936 B |

All three fit an nRF52840 (256 KB RAM, 1 MB flash) with room to spare.

## 4. Five-minute updates: what a cold start costs

A device that updates every five minutes sleeps between updates; on waking it records a burst, filters it from steady state at its first sample, and runs the tracker from an empty history (`src/edge/burst.py`). Accuracy at every 15th window (4,291 update points, the same points for every burst length), paired against the continuous pipeline at those points:

| Burst per update | LED on per update | MAE, bpm | Minus continuous, paired [95 % CI] | Average current, low-power LED | CR2032 life | Average current, datasheet LED | CR2032 life |
|---|---|---|---|---|---|---|---|
| continuous (2-s steps) | always | **14.33** | — | 284.6 µA | 1.1 months | 976.1 µA | 0.3 months |
| 1 window | 8 s | **19.19** | +4.86 [+2.93, +7.17] | 11.5 µA | 27.5 months | 29.9 µA | 10.5 months |
| 2 windows | 10 s | **19.50** | +5.16 [+3.19, +7.14] | 13.3 µA | 23.6 months | 36.4 µA | 8.7 months |
| 4 windows | 14 s | **20.37** | +6.04 [+4.26, +7.84] | 17.1 µA | 18.4 months | 49.3 µA | 6.4 months |
| 8 windows | 22 s | **19.34** | +5.00 [+3.22, +7.17] | 24.5 µA | 12.8 months | 75.2 µA | 4.2 months |
| 15 windows | 36 s | **19.45** | +5.12 [+3.53, +6.83] | 37.6 µA | 8.4 months | 120.6 µA | 2.6 months |
| 30 windows | 66 s | **18.81** | +4.48 [+3.02, +6.13] | 65.6 µA | 4.8 months | 217.8 µA | 1.4 months |

Per activity (bpm):

| Activity | Continuous | 1 window | 4 windows | 15 windows | 30 windows |
|---|---|---|---|---|---|
| sitting | 2.23 | 3.54 | 3.13 | 3.49 | 2.84 |
| stairs | 56.80 | 34.18 | 37.52 | 38.73 | 41.98 |
| table soccer | 23.41 | 33.89 | 32.16 | 33.84 | 31.28 |
| cycling | 35.17 | 26.39 | 28.92 | 28.80 | 30.02 |
| driving | 6.19 | 14.46 | 15.21 | 14.59 | 12.09 |
| lunch | 5.65 | 14.32 | 15.85 | 14.08 | 12.72 |
| walking | 20.76 | 35.12 | 30.78 | 31.26 | 31.59 |
| working | 4.47 | 8.85 | 10.36 | 8.88 | 8.64 |

**Read it this way.** Every cold-start burst lands near 19 bpm — the published no-tracker baseline is 18.98 — whatever its length. The tracker's value is continuity: in quiet activities a continuously running tracker stays on the heart rate (lunch 5.6 bpm continuous), and a cold start picks whichever peak is largest and tracks from there. On stairs the effect reverses, because the continuous tracker had been locked onto a motion line for minutes and a cold start escapes it. A test asserts that a burst started from the record's first window reproduces the continuous run exactly, so the penalty is the cold start, not the burst code. Carrying the tracker's last estimate across sleeps would be the obvious remedy; it is a new component and was not built here.

## 5. The budget, on one page

Average battery current = sleep + accelerometer + (PPG front end + LED while sampling + compute) / 300 s, from the figures below; two years is 17,532 h; a CR2032 is 230 mAh. The LED is reported at the datasheet's own operating point (62 mA, 117.3 µs pulses, scaled to 64 samples/s) and at an assumed lower point (31 mA, 58.7 µs) — the LED current a wrist needs depends on skin and fit, and the datasheet says its LED current is trimmed by up to ±50 %.

- **Continuous tracking** (14.33 bpm): 285–976 µA; two years needs 4,989–17,114 mAh; a CR2032 lasts 1.1 months at best.
- **One window every five minutes** (19.19 bpm): 11.5 µA at the lower LED point; two years needs 201 mAh, so a CR2032 lasts 27.5 months — the only configuration that reaches two years on a coin cell. At the datasheet LED point it lasts 10.5 months.
- **Compute** is 0.8–10 % of the energy in every configuration. Faster or quantised arithmetic cannot move the battery life; LED on-time and the accuracy cost of turning it off are the whole trade.

Sources for the hardware figures:

- **sleep:** nRF52840 Product Brief v2.3: 3.16 uA, System ON, full RAM retention, RTC, DC/DC at 3 V.
- **cpu:** nRF52840 Product Specification, I_CPU0: 3.3 mA, CoreMark at 64 MHz from flash, HFXO, DC/DC.
- **afe_vdd:** MAX86140/MAX86141 datasheet: MAX86141 VDD supply current 23 uA at 50 sps, 46 uA at 100 sps (LP_BOOST = 1, LED off), 1.8 V.
- **afe_vled:** MAX86140/MAX86141 datasheet: VLED supply current 370 uA at 50 sps, 735 uA at 100 sps (62 mA, 117.3 us, single pulse), VLED = 5.0 V.
- **acc:** ADXL362 datasheet: 1.8 uA at 100 Hz output data rate, 2.0 V.
- **battery:** VARTA CR 2032 data sheet: 230 mAh typical, 5.6 kOhm load to 2.0 V, 20 C.
- **ASSUMPTION:** converter efficiency 85% for the 1.8 V front-end and 5 V LED rails; the reduced LED point; coin-cell capacity at the datasheet's light load (LED pulses would need a reservoir capacitor, and a CR2032's capacity falls at high pulse currents).
- **FFT cycles:** FFT4CM4F benchmark README (github.com/Stenzel/FFT4CM4F), CMSIS-DSP 5 on an STM32F446 (Cortex-M4F): 331,857 cycles for a 4,096-point and 151,726 for a 2,048-point real forward transform.

## Limitations

- **Simulated, not measured on silicon.** Latency was measured on a Mac; the microcontroller figures are a published benchmark plus counted operations; energy is datasheet arithmetic. No board was run.
- **The fixed-point FFT is a model** of CMSIS-DSP's Q15 scaling (halve and round every stage), not a bit-exact copy of `arm_rfft_q15`.
- **The tracker stays in float64** in every variant; it handles one number per window.
- **The duty-cycle result is for this tracker.** A tracker that carries state across sleeps, or a front end that samples sparsely but continuously, would change it; neither was built.
- **E4 data.** BVP and accelerometer are the Empatica E4's own outputs; a different front end would have different noise.
