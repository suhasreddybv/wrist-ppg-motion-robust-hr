# wrist-ppg-motion-robust-hr

Accelerometer-referenced motion-artefact cancellation and heart-rate estimation from wrist PPG, benchmarked against published TROIKA / IEEE Signal Processing Cup results.

**Result.** _Pending. Headline number goes here once subject-level cross-validated MAE is computed._

![Per-activity HR error vs ECG reference](figures/hero.png)

## Method

_To be written (21 Sep – 4 Oct 2026)._ Planned: band-pass and windowing; accelerometer-referenced adaptive cancellation; spectral peak tracking; MAE (bpm) by activity; leave-one-subject-out cross-validation, never pooled windows; ablation table. On-device section (26 Oct – 1 Nov): ONNX export, int8 quantisation, latency/size/accuracy loss, on-wrist inference budget for a 5-minute update interval on a two-year battery.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# commands added as the pipeline lands
```

Data: see [data/README.md](data/README.md). No data is included in this repository.

## Limitations and failure cases

_To be written. Must include per-activity failure cases, subjects where the method is worst, and any gap to published benchmarks._

## References

- Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: Large-scale heart rate estimation with convolutional neural networks. *Sensors*, 19(14), 3079.
- Zhang, Z., Pi, Z., & Liu, B. (2015). TROIKA: A general framework for heart rate monitoring using wrist-type photoplethysmographic signals during intensive physical exercise. *IEEE Transactions on Biomedical Engineering*, 62(2), 522–531.

## Licence

MIT
