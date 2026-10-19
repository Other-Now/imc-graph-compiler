Calibration: 1000 training images. Accuracy: 10,000 test images. pp = percentage points.

| model | configuration | test accuracy | change vs fp32 |
|---|---|---|---|
| mlp | fp32 (PyTorch eager) | 58.71% | +0.00 pp |
| mlp | fp32, 0 conv+BN folded | 58.71% | +0.00 pp |
| mlp | int8, per-tensor weights, min-max activations | 58.73% | +0.02 pp |
| mlp | int8, per-channel weights, min-max activations | 58.68% | -0.03 pp |
| mlp | int8, per-channel weights, 99.99th-pct activations | 58.65% | -0.06 pp |
| mlp | + 8-bit ADC (minmax range), 256-row crossbar | 58.51% | -0.20 pp |
| mlp | + 8-bit ADC (pct range), 256-row crossbar | 58.51% | -0.20 pp |
| mlp | + 6-bit ADC (minmax range), 256-row crossbar | 56.17% | -2.54 pp |
| mlp | + 6-bit ADC (pct range), 256-row crossbar | 57.13% | -1.58 pp |
| mlp | + 4-bit ADC (minmax range), 256-row crossbar | 33.65% | -25.06 pp |
| mlp | + 4-bit ADC (pct range), 256-row crossbar | 33.65% | -25.06 pp |
| cnn | fp32 (PyTorch eager) | 85.36% | +0.00 pp |
| cnn | fp32, 4 conv+BN folded | 85.36% | +0.00 pp |
| cnn | int8, per-tensor weights, min-max activations | 85.30% | -0.06 pp |
| cnn | int8, per-channel weights, min-max activations | 85.29% | -0.07 pp |
| cnn | int8, per-channel weights, 99.99th-pct activations | 85.38% | +0.02 pp |
| cnn | + 8-bit ADC (minmax range), 256-row crossbar | 85.06% | -0.30 pp |
| cnn | + 8-bit ADC (pct range), 256-row crossbar | 85.14% | -0.22 pp |
| cnn | + 6-bit ADC (minmax range), 256-row crossbar | 82.47% | -2.89 pp |
| cnn | + 6-bit ADC (pct range), 256-row crossbar | 84.17% | -1.19 pp |
| cnn | + 4-bit ADC (minmax range), 256-row crossbar | 15.28% | -70.08 pp |
| cnn | + 4-bit ADC (pct range), 256-row crossbar | 52.26% | -33.10 pp |
