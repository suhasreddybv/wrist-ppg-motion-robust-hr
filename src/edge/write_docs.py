"""docs/edge-budget.md and the README's Week 7 section, generated from results/edge_*.csv so every
number is the committed one; tests/test_edge_docs.py asserts both match what this writes."""
from __future__ import annotations

import csv
from pathlib import Path

from src.edge.budget import CAPACITY_MAH, ETA, SOURCES

ROOT = Path(__file__).resolve().parents[2]
START, END = "<!-- edge-budget:start -->", "<!-- edge-budget:end -->"
LABEL = {"reference": "Reference: float64, zero-phase band-pass over the record (the published 14.70)",
         "causal_f64": "Causal band-pass, float64", "causal_f32": "Causal, float32 (native on a Cortex-M4F)",
         "q15": "Q15: 16-bit data, 32-bit-state band-pass, Q15 FFTs", "q15_state16": "Q15 with a 16-bit-state band-pass (the naive form)",
         "q8": "8-bit storage of windows, spectra and gains; ACC at its native 1/64 g"}


def _rows(name):
    with open(ROOT / "results" / name) as f:
        return list(csv.DictReader(f))


def _n() -> dict:
    s = {r["variant"]: r for r in _rows("edge_variants_summary.csv")}
    b = {r["burst_windows"]: r for r in _rows("edge_burst.csv")}
    bud = _rows("edge_budget.csv")
    low = {r["burst_windows"]: r for r in bud if r["led"].startswith("low")}
    hi = {r["burst_windows"]: r for r in bud if r["led"].startswith("datasheet")}
    lat = {r["stage"]: r for r in _rows("edge_latency.csv")}
    comp = {r["item"]: r for r in _rows("edge_compute.csv")}
    mem = {r["item"]: r for r in _rows("edge_memory.csv")}
    return dict(s=s, b=b, low=low, hi=hi, lat=lat, comp=comp, mem=mem)


def variant_table(n) -> str:
    out = ["| Variant | Pooled MAE, bpm | Minus causal float64, paired [95 % CI] | Subjects worse than causal float64 |", "|---|---|---|---|"]
    for k, lab in LABEL.items():
        r = n["s"][k]
        d = "—" if k == "causal_f64" else f"{float(r['minus_causal_f64']):+.2f} [{float(r['minus_causal_f64_ci_lo']):+.2f}, {float(r['minus_causal_f64_ci_hi']):+.2f}]"
        out.append(f"| {lab} | **{float(r['pooled_mae']):.2f}** | {d} | {'—' if k == 'causal_f64' else r['subjects_worse_than_causal_f64'] + ' of 15'} |")
    return "\n".join(out)


def activity_table() -> str:
    rows = _rows("edge_variants_per_activity.csv")
    acts = list(dict.fromkeys(r["activity"] for r in rows))
    v = {(r["variant"], r["activity"]): float(r["mae_mean_of_folds"]) for r in rows}
    out = ["| Activity | " + " | ".join(["reference", "causal f64", "causal f32", "Q15", "Q15, 16-bit state", "8-bit"]) + " |", "|---" * 7 + "|"]
    for a in acts:
        out.append(f"| {a} | " + " | ".join(f"{v[(k, a)]:.2f}" for k in LABEL) + " |")
    return "\n".join(out)


def burst_table(n) -> str:
    out = ["| Burst per update | LED on per update | MAE, bpm | Minus continuous, paired [95 % CI] | Average current, low-power LED | CR2032 life | Average current, datasheet LED | CR2032 life |",
           "|---|---|---|---|---|---|---|---|"]
    for k in ["continuous", "1", "2", "4", "8", "15", "30"]:
        r = n["b"][k]
        lab = "continuous (2-s steps)" if k == "continuous" else f"{k} window{'s' if k != '1' else ''}"
        on = "always" if k == "continuous" else f"{r['led_on_s']} s"
        d = "—" if k == "continuous" else f"{float(r['minus_continuous']):+.2f} [{float(r['ci_lo']):+.2f}, {float(r['ci_hi']):+.2f}]"
        lo, hi = n["low"][k], n["hi"][k]
        out.append(f"| {lab} | {on} | **{float(r['pooled_mae']):.2f}** | {d} | {float(lo['avg_current_uA']):.1f} µA | {float(lo['cr2032_months']):.1f} months | "
                   f"{float(hi['avg_current_uA']):.1f} µA | {float(hi['cr2032_months']):.1f} months |")
    return "\n".join(out)


def burst_activity_table() -> str:
    rows = _rows("edge_burst_per_activity.csv")
    cols = ["continuous", "burst_1", "burst_4", "burst_15", "burst_30"]
    out = ["| Activity | Continuous | 1 window | 4 windows | 15 windows | 30 windows |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['activity']} | " + " | ".join(f"{float(r[c]):.2f}" for c in cols) + " |")
    return "\n".join(out)


def compute_table(n) -> str:
    out = ["| Work per 8-s window | Cortex-M4F cycles | Basis |", "|---|---|---|"]
    for r in _rows("edge_compute.csv"):
        out.append(f"| {r['item']} | {int(r['cycles']):,} | {r['source']} |")
    return "\n".join(out)


def memory_table() -> str:
    out = ["| Buffer | float32 | Q15 | 8-bit storage |", "|---|---|---|---|"]
    for r in _rows("edge_memory.csv"):
        out.append(f"| {r['item']} | {int(r['float32']):,} B | {int(r['q15']):,} B | {int(r['q8']):,} B |")
    return "\n".join(out)


def latency_table(n) -> str:
    out = ["| Stage of one streaming step | Median | 95th percentile |", "|---|---|---|"]
    for r in _rows("edge_latency.csv"):
        out.append(f"| {r['stage']} | {float(r['median_us']):.1f} µs | {float(r['p95_us']):.1f} µs |")
    return "\n".join(out)


def headline(n) -> dict:
    s, b, low, hi, lat, comp = n["s"], n["b"], n["low"], n["hi"], n["lat"], n["comp"]
    cyc = int(comp["total per window"]["cycles"])
    return dict(
        ref=f"{float(s['reference']['pooled_mae']):.2f}", c64=f"{float(s['causal_f64']['pooled_mae']):.2f}", c32=f"{float(s['causal_f32']['pooled_mae']):.2f}",
        q15=f"{float(s['q15']['pooled_mae']):.2f}", q15d=f"{float(s['q15']['minus_causal_f64']):+.2f}", q15lo=f"{float(s['q15']['minus_causal_f64_ci_lo']):+.2f}", q15hi=f"{float(s['q15']['minus_causal_f64_ci_hi']):+.2f}",
        q8=f"{float(s['q8']['pooled_mae']):.2f}", q8d=f"{float(s['q8']['minus_causal_f64']):+.2f}", q8lo=f"{float(s['q8']['minus_causal_f64_ci_lo']):+.2f}", q8hi=f"{float(s['q8']['minus_causal_f64_ci_hi']):+.2f}",
        s16=f"{float(s['q15_state16']['pooled_mae']):.2f}", s16d=f"{float(s['q15_state16']['minus_causal_f64']):+.2f}", s16lo=f"{float(s['q15_state16']['minus_causal_f64_ci_lo']):+.2f}", s16hi=f"{float(s['q15_state16']['minus_causal_f64_ci_hi']):+.2f}",
        cont=f"{float(b['continuous']['pooled_mae']):.2f}", b1=f"{float(b['1']['pooled_mae']):.2f}", b30=f"{float(b['30']['pooled_mae']):.2f}",
        b1d=f"{float(b['1']['minus_continuous']):+.2f}", b1lo=f"{float(b['1']['ci_lo']):+.2f}", b1hi=f"{float(b['1']['ci_hi']):+.2f}",
        b30d=f"{float(b['30']['minus_continuous']):+.2f}", b30lo=f"{float(b['30']['ci_lo']):+.2f}", b30hi=f"{float(b['30']['ci_hi']):+.2f}",
        low1=f"{float(low['1']['avg_current_uA']):.1f}", low1m=f"{float(low['1']['cr2032_months']):.1f}", low1cap=f"{float(low['1']['capacity_for_2y_mAh']):.0f}",
        lowc=f"{float(low['continuous']['avg_current_uA']):.0f}", lowcm=f"{float(low['continuous']['cr2032_months']):.1f}", lowccap=f"{float(low['continuous']['capacity_for_2y_mAh']):,.0f}",
        hic=f"{float(hi['continuous']['avg_current_uA']):.0f}", hicm=f"{float(hi['continuous']['cr2032_months']):.1f}", hiccap=f"{float(hi['continuous']['capacity_for_2y_mAh']):,.0f}",
        hi1m=f"{float(hi['1']['cr2032_months']):.1f}", share_lo=f"{float(hi['1']['compute_share_pct']):.1f}", share_hi=f"{float(low['continuous']['compute_share_pct']):.0f}",
        fft_share=f"{100 * sum(int(r['cycles']) for k, r in comp.items() if 'FFT' in k) / cyc:.0f}",
        points=f"{int(b['1']['update_points']):,}", lat_n=f"{int(lat['total per window']['windows']):,}", step_pct=f"{cyc / 64e6 / 2.0 * 100:.1f}",
        lunch=f"{float(next(r for r in _rows('edge_burst_per_activity.csv') if r['activity'] == 'lunch')['continuous']):.1f}",
        cyc=f"{cyc / 1e6:.2f}", ms=f"{cyc / 64e6 * 1e3:.1f}", mac=f"{float(lat['total per window']['median_us']):.0f}",
        ram32=f"{int(n['mem']['RAM total']['float32']) / 1024:.0f}", ram15=f"{int(n['mem']['RAM total']['q15']) / 1024:.0f}",
        fl32=f"{int(n['mem']['FFT tables in flash (CMSIS-DSP arm_common_tables.h sizes)']['float32']) / 1024:.0f}")


def readme_section() -> str:
    n = _n(); h = headline(n)
    return f"""{START}
## On a wrist: the inference budget (Week 7)

The frozen pipeline was run the way a watch would run it ([docs/edge-budget.md](docs/edge-budget.md)). It has no learned weights — a band-pass, spectra, accelerometer notches and a tracker — so there is nothing for ONNX export or int8 quantisation of a network to act on; the edge question is fixed-point arithmetic, latency, memory and energy, and that is what was measured. Nothing was refitted: every variant uses the configuration each LOSO fold selected.

- **Arithmetic is not the constraint.** A causal band-pass (a device cannot filter backwards) gives {h['c64']} bpm against the published zero-phase {h['ref']}; float32 is identical ({h['c32']}); 16-bit fixed point gives {h['q15']} ({h['q15d']} [{h['q15lo']}, {h['q15hi']}] against causal float64) and 8-bit storage {h['q8']} ({h['q8d']} [{h['q8lo']}, {h['q8hi']}]) — neither distinguishable from float. The one arithmetic choice that costs accuracy is a band-pass that keeps only 16-bit state between sections: {h['s16']} bpm ({h['s16d']} [{h['s16lo']}, {h['s16hi']}]), because a 0.4 Hz cutoff at 64 Hz puts the poles next to the unit circle.
- **Compute is not the constraint either.** About {h['cyc']} million Cortex-M4F cycles per 8-s window — {h['ms']} ms at 64 MHz, {h['fft_share']} % of it the five FFTs — and {h['ram32']} KB of RAM in float32 ({h['ram15']} KB in Q15). On this Mac the streaming path takes {h['mac']} µs per window on one thread.
- **The LED is the constraint, and duty cycling costs the tracker.** Tracking continuously gives {h['cont']} bpm but draws {h['lowc']}–{h['hic']} µA, {h['lowcm']} months or less on a CR2032. Waking every five minutes for one window draws {h['low1']} µA at a reduced LED setting — {h['low1m']} months on a CR2032 — but scores {h['b1']} bpm ({h['b1d']} [{h['b1lo']}, {h['b1hi']}]): the tracker's whole gain is continuity, and a cold start gives it up. Longer bursts do not buy it back (30 windows, 66 s of LED: {h['b30']}). Compute is {h['share_lo']}–{h['share_hi']} % of the energy in every configuration.
{END}"""


def edge_doc() -> str:
    n = _n(); h = headline(n)
    src = "\n".join(f"- **{k}:** {v}." for k, v in SOURCES.items())
    return f"""# On a wrist: the inference budget

Week 7, on the frozen mask+tracker(hist=15) pipeline (D-046), LOSO, every fold at the configuration it selected on its training subjects (`results/edge_fold_configs.csv`, reconciled to the published 14.704 bpm before anything else ran). Decision-log entries D-057 to D-059. Every number here is generated from `results/edge_*.csv` by `src/edge/write_docs.py`.

## What was asked, and why there is no ONNX model

The plan asked for an ONNX export, int8 quantisation, latency, model size and accuracy lost per activity, and a one-page budget for a five-minute update interval on a two-year battery. This pipeline has no learned weights: a 4th-order Butterworth band-pass, a 4,096-point spectrum of each 8-s PPG window, four 2,048-point accelerometer spectra, Gaussian notches at the motion lines and a peak tracker, with a handful of hyperparameters chosen in each fold. ONNX export and post-training int8 quantisation are tools for networks. The equivalent questions for signal processing were answered instead: what fixed-point arithmetic costs, how long a window takes, how much memory it needs, and where the energy goes.

## 1. Arithmetic: what a device would change, and what it costs

Each variant is the frozen pipeline with one class of change (`src/edge/pipeline.py`; the fixed-point simulators in `src/edge/fixed_point.py` are tested against exact references). Paired differences are per subject against causal float64, the first variant a device could actually run; bootstrap over the 15 subjects, 2,000 resamples, seed 42.

{variant_table(n)}

**Read it this way.** The published pipeline filters the whole record forwards and backwards (zero-phase), which needs the future; a device filters causally, and that is {h['c64']} bpm against {h['ref']} — not worse, and the interval includes zero. Float32, the native arithmetic of a Cortex-M4F, changes nothing to two decimals. Fixed point with 16-bit data and 8-bit storage moves the pooled error by less than a bpm in either direction with intervals spanning zero: the tracker is path-dependent, so any small perturbation flips some lock-in episodes one way and some the other, and the 8-bit variant landing below Q15 is that noise, not an ordering. The one arithmetic choice that is clearly worse is a band-pass that rounds each biquad's output to 16 bits: at a 0.4 Hz cutoff and 64 Hz sampling the poles sit close to the unit circle and amplify the rounding (31 dB signal-to-error in `tests/test_edge_fixed_point.py`, against 127 dB with 32-bit state). The CMSIS-DSP library provides the 32-bit-state form for exactly this case.

Per activity, mean of per-subject MAE (bpm):

{activity_table()}

## 2. Latency and compute

The device path was written as a device runs it (`src/edge/stream.py`): one 2-s step at a time, the band-pass state carried, a 512-sample ring buffer, an incremental tracker. A test asserts it gives the batch pipeline's estimates exactly, and that its tracker equals the original step for step across every configuration in the grids. Timed on one thread on this machine over {h['lat_n']} windows of S1 (macOS cannot pin a process to a core; one thread is the stand-in):

{latency_table(n)}

What a Cortex-M4F would execute per window, from a published FFT benchmark and loop sizes read off the code; the cycle costs assumed for the non-FFT arithmetic are stated in `src/edge/latency.py`:

{compute_table(n)}

That is {h['ms']} ms per window at 64 MHz, {h['step_pct']} % of the 2-s step.

## 3. Memory

RAM the device path holds per window, and the flash for CMSIS-DSP 5's FFT tables (sizes from `arm_common_tables.h`):

{memory_table()}

All three fit an nRF52840 (256 KB RAM, 1 MB flash) with room to spare.

## 4. Five-minute updates: what a cold start costs

A device that updates every five minutes sleeps between updates; on waking it records a burst, filters it from steady state at its first sample, and runs the tracker from an empty history (`src/edge/burst.py`). Accuracy at every 15th window ({h['points']} update points, the same points for every burst length), paired against the continuous pipeline at those points:

{burst_table(n)}

Per activity (bpm):

{burst_activity_table()}

**Read it this way.** Every cold-start burst lands near 19 bpm — the published no-tracker baseline is 18.98 — whatever its length. The tracker's value is continuity: in quiet activities a continuously running tracker stays on the heart rate (lunch {h['lunch']} bpm continuous), and a cold start picks whichever peak is largest and tracks from there. On stairs the effect reverses, because the continuous tracker had been locked onto a motion line for minutes and a cold start escapes it. A test asserts that a burst started from the record's first window reproduces the continuous run exactly, so the penalty is the cold start, not the burst code. Carrying the tracker's last estimate across sleeps would be the obvious remedy; it is a new component and was not built here.

## 5. The budget, on one page

Average battery current = sleep + accelerometer + (PPG front end + LED while sampling + compute) / 300 s, from the figures below; two years is 17,532 h; a CR2032 is {CAPACITY_MAH:.0f} mAh. The LED is reported at the datasheet's own operating point (62 mA, 117.3 µs pulses, scaled to 64 samples/s) and at an assumed lower point (31 mA, 58.7 µs) — the LED current a wrist needs depends on skin and fit, and the datasheet says its LED current is trimmed by up to ±50 %.

- **Continuous tracking** ({h['cont']} bpm): {h['lowc']}–{h['hic']} µA; two years needs {h['lowccap']}–{h['hiccap']} mAh; a CR2032 lasts {h['lowcm']} months at best.
- **One window every five minutes** ({h['b1']} bpm): {h['low1']} µA at the lower LED point; two years needs {h['low1cap']} mAh, so a CR2032 lasts {h['low1m']} months — the only configuration that reaches two years on a coin cell. At the datasheet LED point it lasts {h['hi1m']} months.
- **Compute** is {h['share_lo']}–{h['share_hi']} % of the energy in every configuration. Faster or quantised arithmetic cannot move the battery life; LED on-time and the accuracy cost of turning it off are the whole trade.

Sources for the hardware figures:

{src}
- **ASSUMPTION:** converter efficiency {ETA:.0%} for the 1.8 V front-end and 5 V LED rails; the reduced LED point; coin-cell capacity at the datasheet's light load (LED pulses would need a reservoir capacitor, and a CR2032's capacity falls at high pulse currents).
- **FFT cycles:** FFT4CM4F benchmark README (github.com/Stenzel/FFT4CM4F), CMSIS-DSP 5 on an STM32F446 (Cortex-M4F): 331,857 cycles for a 4,096-point and 151,726 for a 2,048-point real forward transform.

## Limitations

- **Simulated, not measured on silicon.** Latency was measured on a Mac; the microcontroller figures are a published benchmark plus counted operations; energy is datasheet arithmetic. No board was run.
- **The fixed-point FFT is a model** of CMSIS-DSP's Q15 scaling (halve and round every stage), not a bit-exact copy of `arm_rfft_q15`.
- **The tracker stays in float64** in every variant; it handles one number per window.
- **The duty-cycle result is for this tracker.** A tracker that carries state across sleeps, or a front end that samples sparsely but continuously, would change it; neither was built.
- **E4 data.** BVP and accelerometer are the Empatica E4's own outputs; a different front end would have different noise.
"""


def write_all() -> None:
    (ROOT / "docs" / "edge-budget.md").write_text(edge_doc())
    readme = (ROOT / "README.md").read_text()
    sec = readme_section()
    if START in readme:
        i, j = readme.index(START), readme.index(END) + len(END)
        readme = readme[:i] + sec + readme[j:]
    else:
        readme = readme.replace("## Limitations\n", sec + "\n\n## Limitations\n", 1)
    if "docs/edge-budget.md) —" not in readme:
        readme = readme.replace("- [docs/decisions.md](docs/decisions.md)",
                                "- [docs/edge-budget.md](docs/edge-budget.md) — the pipeline on a wrist: fixed point, latency, memory, duty cycling and a battery budget (Week 7)\n- [docs/decisions.md](docs/decisions.md)", 1)
    (ROOT / "README.md").write_text(readme)
    print("wrote docs/edge-budget.md and the README section")


if __name__ == "__main__":
    write_all()
