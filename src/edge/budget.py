"""The inference budget for a five-minute update interval on a two-year battery, from the measured
accuracy (results/edge_burst.csv), the per-window compute (results/edge_compute.csv) and published
hardware figures. Writes results/edge_budget.csv and results/edge_memory.csv.

Every hardware constant carries its source. Two are assumptions and are labelled as such: the
power-converter efficiency and the reduced LED design point.
"""
from __future__ import annotations

import csv
import sys

from src.data.loader import REPO_ROOT
from src.edge.burst import BURSTS, led_on_s

UPDATE_S = 300.0                      # five-minute updates
TWO_YEARS_H = 2 * 365.25 * 24         # 17,532 h
V_BATT = 3.0
ETA = 0.85                            # ASSUMPTION: buck (1.8 V AFE) and boost (5 V LED) converter efficiency

SOURCES = {
    "sleep": "nRF52840 Product Brief v2.3: 3.16 uA, System ON, full RAM retention, RTC, DC/DC at 3 V",
    "cpu": "nRF52840 Product Specification, I_CPU0: 3.3 mA, CoreMark at 64 MHz from flash, HFXO, DC/DC",
    "afe_vdd": "MAX86140/MAX86141 datasheet: MAX86141 VDD supply current 23 uA at 50 sps, 46 uA at 100 sps (LP_BOOST = 1, LED off), 1.8 V",
    "afe_vled": "MAX86140/MAX86141 datasheet: VLED supply current 370 uA at 50 sps, 735 uA at 100 sps (62 mA, 117.3 us, single pulse), VLED = 5.0 V",
    "acc": "ADXL362 datasheet: 1.8 uA at 100 Hz output data rate, 2.0 V",
    "battery": "VARTA CR 2032 data sheet: 230 mAh typical, 5.6 kOhm load to 2.0 V, 20 C",
}
I_SLEEP = 3.16e-6
I_CPU = 3.3e-3
AFE_VDD_PER_SPS, V_AFE = 46e-6 / 100, 1.8         # linear in sample rate between the 50 and 100 sps rows
LED_PER_SPS_DATASHEET, V_LED = 735e-6 / 100, 5.0   # at 62 mA, 117.3 us
LED_LOW_FACTOR = (31 / 62) * (58.7 / 117.3)        # ASSUMPTION: a 31 mA, 58.7 us design point, scaled linearly
I_ACC, V_ACC = 1.8e-6, 2.0
CAPACITY_MAH = 230.0
SPS = 64


def batt(i_rail: float, v_rail: float) -> float:
    """Battery-side current for a rail supplied through a converter."""
    return i_rail * v_rail / (V_BATT * ETA)


def main() -> None:
    burst = {r["burst_windows"]: r for r in csv.DictReader(open(REPO_ROOT / "results" / "edge_burst.csv"))}
    comp = {r["item"]: r for r in csv.DictReader(open(REPO_ROOT / "results" / "edge_compute.csv"))}
    t_window = int(comp["total per window"]["cycles"]) / 64e6
    rows = []
    for led_name, led_factor in (("datasheet point, 62 mA / 117.3 us", 1.0), ("low-power point, 31 mA / 58.7 us (assumed)", LED_LOW_FACTOR)):
        i_led = batt(LED_PER_SPS_DATASHEET * SPS * led_factor, V_LED)
        i_afe = batt(AFE_VDD_PER_SPS * SPS, V_AFE)
        i_acc = batt(I_ACC, V_ACC)
        for b in BURSTS:
            on = led_on_s(b)
            q_sense = on * (i_led + i_afe)
            q_cpu = b * t_window * I_CPU
            i_avg = I_SLEEP + i_acc + (q_sense + q_cpu) / UPDATE_S
            rows.append(dict(led=led_name, burst_windows=b, led_on_s=on, mae_bpm=burst[str(b)]["pooled_mae"],
                             sensing_uC_per_update=round(q_sense * 1e6, 1), compute_uC_per_update=round(q_cpu * 1e6, 2),
                             compute_share_pct=round(100 * q_cpu / (q_sense + q_cpu), 2), avg_current_uA=round(i_avg * 1e6, 2),
                             capacity_for_2y_mAh=round(i_avg * 1e3 * TWO_YEARS_H, 0), cr2032_months=round(CAPACITY_MAH / (i_avg * 1e3) / (TWO_YEARS_H / 24), 1)))
        # continuous operation: sensing and one window of compute every 2 s, never sleeping the front end
        i_avg = I_SLEEP + i_acc + i_led + i_afe + t_window * I_CPU / 2.0
        rows.append(dict(led=led_name, burst_windows="continuous", led_on_s=UPDATE_S, mae_bpm=burst["continuous"]["pooled_mae"],
                         sensing_uC_per_update=round(UPDATE_S * (i_led + i_afe) * 1e6, 1), compute_uC_per_update=round(150 * t_window * I_CPU * 1e6, 2),
                         compute_share_pct=round(100 * (t_window * I_CPU / 2.0) / (i_led + i_afe + t_window * I_CPU / 2.0), 2),
                         avg_current_uA=round(i_avg * 1e6, 2), capacity_for_2y_mAh=round(i_avg * 1e3 * TWO_YEARS_H, 0),
                         cr2032_months=round(CAPACITY_MAH / (i_avg * 1e3) / (TWO_YEARS_H / 24), 1)))
    with open(REPO_ROOT / "results" / "edge_budget.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  {r['led'][:15]:<15} B={str(r['burst_windows']):>10}  MAE {r['mae_bpm']:>6}  avg {r['avg_current_uA']:>8} uA  2 y needs {r['capacity_for_2y_mAh']:>7} mAh  CR2032 {r['cr2032_months']:>6} months  compute {r['compute_share_pct']}%")
    # memory: buffers the code holds per window, and the CMSIS-DSP 5 FFT tables (arm_common_tables.h)
    f32, q15, u16 = 4, 2, 2
    mem = [
        dict(item="PPG window ring buffer, 512 samples", float32=512 * f32, q15=512 * q15, q8=512 * 1),
        dict(item="real-FFT input and output buffers, 4096 points (reused for the ACC FFTs)", float32=2 * 4096 * f32, q15=(4096 + 2 * 4096) * q15, q8=(4096 + 2 * 4096) * q15),
        dict(item="ACC window, 3 axes (64 Hz resampled; q8 keeps the native 32 Hz, 1/64 g)", float32=512 * 3 * f32, q15=512 * 3 * q15, q8=256 * 3 * 1),
        dict(item="motion power spectrum, 1025 bins", float32=1025 * f32, q15=1025 * q15, q8=1025 * 1),
        dict(item="in-band PPG spectrum and notch gains, 2 x 231 bins", float32=2 * 231 * f32, q15=2 * 231 * q15, q8=2 * 231 * 1),
        dict(item="band-pass state (4 biquads) and tracker history (15)", float32=4 * 2 * f32 + 15 * f32, q15=4 * 4 * 4 + 15 * f32, q8=4 * 4 * 4 + 15 * f32),
    ]
    mem.append(dict(item="RAM total", **{k: sum(m[k] for m in mem) for k in ("float32", "q15", "q8")}))
    flash_f32 = (4096 + 4096) * f32 + 3808 * u16 + (2048 + 2048) * f32 + 1800 * u16   # rfft_fast 4096 (cfft 2048) and 2048 (cfft 1024)
    flash_q15 = (3072 + 1536) * q15 + 2 * 8192 * q15 + (1984 + 992) * u16                 # cfft q15 2048 and 1024 twiddles, realCoefA/B, bit-reversal
    mem.append(dict(item="FFT tables in flash (CMSIS-DSP arm_common_tables.h sizes)", float32=flash_f32, q15=flash_q15, q8=flash_q15))
    with open(REPO_ROOT / "results" / "edge_memory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mem[0])); w.writeheader(); w.writerows(mem)
    for m in mem:
        print(f"  {m['item'][:60]:<60} f32 {m['float32']:>7,}  q15 {m['q15']:>7,}  q8 {m['q8']:>7,} bytes")
    print("wrote results/edge_budget.csv, results/edge_memory.csv")


if __name__ == "__main__":
    sys.exit(main())
