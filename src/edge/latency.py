"""Per-window latency of the streaming device path (float32), single-threaded on this machine,
split by stage; and the operation count a Cortex-M4F would execute, from a published benchmark
and the code's own loop sizes. Writes results/edge_latency.csv and results/edge_compute.csv.

Threads: every numerical library is pinned to one thread before numpy loads (macOS cannot pin a
process to one core; one thread is the stand-in, stated as such).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from src.data.loader import REPO_ROOT, load_subject  # noqa: E402
from src.data.windows import window_subject  # noqa: E402
from src.edge.pipeline import FS, acc_windows, fold_configs  # noqa: E402
from src.edge.stream import DEVICE, STEP, WINDOW, DeviceStream  # noqa: E402
from src.features.preprocess import CARDIAC_BAND_HZ  # noqa: E402
from src.models.baselines import ZERO_PAD_NFFT  # noqa: E402
from src.models.masking import ACC_NFFT  # noqa: E402

SUBJECT, N_WINDOWS = "S1", 1500

# Published cycle counts for CMSIS-DSP 5 real forward FFTs on a Cortex-M4F (STM32F446), from the
# FFT4CM4F benchmark README (github.com/Stenzel/FFT4CM4F): "4096 r+ CMSIS_5 331857", "2048 r+ CMSIS_5 151726".
CYCLES_RFFT = {4096: 331_857, 2048: 151_726}
CLOCK_HZ = 64e6                       # nRF52840 (Cortex-M4F, 64 MHz)
# Cycle costs assumed for the non-FFT arithmetic, stated so they can be challenged: a float
# multiply-accumulate on the M4F FPU is 3 cycles (VFMA.F32); a single-precision exp() from a
# math library ~100 cycles; a compare-and-branch ~2.
MAC, EXP, CMP = 3, 100, 2


def non_fft_cycles(n_band: int, n_lines: int) -> dict[str, int]:
    """Loop sizes read off the code (src/edge/stream.py, src/models/masking.py, tracker)."""
    acc_band = int(round((CARDIAC_BAND_HZ[1] * 2 - CARDIAC_BAND_HZ[0] * 0.5) / (FS / ACC_NFFT)))
    return {
        "band-pass, 128 new samples x 4 biquads x 5 MAC": 128 * 4 * 5 * MAC,
        "z-score, 512 samples (mean, variance, scale)": 512 * 3 * MAC,
        "PPG power spectrum, in-band bins": n_band * 3 * MAC,
        "ACC detrend, magnitude, taper: 512 x 4 signals": 512 * 4 * 4 * MAC,
        "ACC power, 4 x 1025 bins, sum and normalise": 4 * (ACC_NFFT // 2 + 1) * 3 * MAC,
        "ACC peak search over the motion band": acc_band * 10 * CMP,
        "notch gains: lines x in-band bins x (exp + 4 MAC)": n_lines * n_band * (EXP + 4 * MAC),
        "mask multiply and PPG peak search": n_band * (MAC + 10 * CMP),
        "tracker step": 200,
    }


def main() -> None:
    configs = fold_configs()
    mcfg, tcfg = configs[SUBJECT]
    rec = load_subject(SUBJECT)
    ws = window_subject(rec)
    n = min(N_WINDOWS, len(ws))
    accw = acc_windows(rec["wrist_acc"].data, rec["wrist_acc"].fs, len(ws), DEVICE)
    bvp = rec["wrist_bvp"].data
    dev = DeviceStream(mcfg, tcfg)
    dev.push(bvp[: WINDOW - STEP])
    totals, stages = [], {}
    for k in range(n):
        lo = WINDOW - STEP + k * STEP
        t0 = time.perf_counter()
        dev.push(bvp[lo: lo + STEP], accw[k], clock=time.perf_counter)
        totals.append(time.perf_counter() - t0)
        for s, v in dev.timings.items():
            stages.setdefault(s, []).append(v)
    totals = np.array(totals[10:]) * 1e6
    rows = [dict(stage="total per window", median_us=round(float(np.median(totals)), 1), p95_us=round(float(np.percentile(totals, 95)), 1),
                 max_us=round(float(totals.max()), 1), windows=len(totals), machine=f"{platform.machine()} {platform.processor() or ''}".strip(),
                 python=platform.python_version(), numpy=np.__version__, threads=1)]
    for s, v in stages.items():
        v = np.array(v[10:]) * 1e6
        rows.append(dict(stage=s, median_us=round(float(np.median(v)), 1), p95_us=round(float(np.percentile(v, 95)), 1), max_us=round(float(v.max()), 1),
                         windows=len(v), machine=rows[0]["machine"], python=rows[0]["python"], numpy=rows[0]["numpy"], threads=1))
    for r in rows:
        print(f"  {r['stage']:<24} median {r['median_us']:>8.1f} us   p95 {r['p95_us']:>8.1f} us")
    with open(REPO_ROOT / "results" / "edge_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # the MCU estimate
    freqs = np.fft.rfftfreq(ZERO_PAD_NFFT, 1 / FS)
    n_band = int(((freqs >= CARDIAC_BAND_HZ[0]) & (freqs <= CARDIAC_BAND_HZ[1])).sum())
    n_lines = mcfg.n_peaks * 3
    comp = [dict(item="PPG real FFT, 4096 points (CMSIS-DSP 5, Cortex-M4F, published)", cycles=CYCLES_RFFT[4096], source="FFT4CM4F benchmark"),
            dict(item="ACC real FFTs, 4 x 2048 points (published)", cycles=4 * CYCLES_RFFT[2048], source="FFT4CM4F benchmark")]
    comp += [dict(item=k, cycles=v, source="loop size from the code x assumed cycles per op") for k, v in non_fft_cycles(n_band, n_lines).items()]
    total = sum(c["cycles"] for c in comp)
    comp.append(dict(item="total per window", cycles=total, source=f"{total / CLOCK_HZ * 1e3:.2f} ms at 64 MHz"))
    with open(REPO_ROOT / "results" / "edge_compute.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0])); w.writeheader(); w.writerows(comp)
    print(f"  in-band bins {n_band}, notch lines {n_lines}; Cortex-M4F estimate {total:,} cycles = {total / CLOCK_HZ * 1e3:.2f} ms per window at 64 MHz")
    print("wrote results/edge_latency.csv, results/edge_compute.csv")


if __name__ == "__main__":
    sys.exit(main())
