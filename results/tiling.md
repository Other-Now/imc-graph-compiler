Analytical model (imc/crossbar.py), NOT hardware. Chip = 4M int8 cells. ADC range calibrated at the 99.99th percentile.
Accuracy columns are measured (emulator, 10k test images).

| model | crossbar | tiles | utilization | arrays on chip | fits? | ADC conv./img | energy/img (6-bit) | latency/img | acc 6-bit ADC | acc 4-bit ADC |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp | 64x64 | 202 | 99% | 1024 | yes | 12,820 | 10.9 nJ | 0.3 us | 58.08% | 47.69% |
| mlp | 128x128 | 51 | 98% | 256 | yes | 6,410 | 7.5 nJ | 0.3 us | 58.09% | 42.69% |
| mlp | 256x256 | 14 | 89% | 64 | yes | 3,210 | 5.8 nJ | 0.3 us | 57.13% | 33.65% |
| mlp | 512x512 | 8 | 39% | 16 | yes | 1,674 | 5.0 nJ | 0.3 us | 58.03% | 36.54% |
| mlp | 256x64 | 51 | 98% | 256 | yes | 3,210 | 5.8 nJ | 0.3 us | 57.13% | 33.65% |
| cnn | 64x64 | 150 | 96% | 1024 | yes | 434,196 | 352.2 nJ | 256.2 us | 84.68% | 66.12% |
| cnn | 128x128 | 45 | 80% | 256 | yes | 266,250 | 263.2 nJ | 256.2 us | 84.65% | 58.76% |
| cnn | 256x256 | 25 | 36% | 64 | yes | 182,282 | 218.7 nJ | 256.2 us | 84.17% | 52.26% |
| cnn | 512x512 | 14 | 16% | 16 | yes | 115,722 | 183.4 nJ | 256.2 us | 84.41% | 59.56% |
| cnn | 256x64 | 41 | 88% | 256 | yes | 182,282 | 218.7 nJ | 256.2 us | 84.17% | 52.26% |

CNN per layer:

| layer | K x N | vectors/img | tiles @256x256 | utilization | ADC conv./img |
|---|---|---|---|---|---|
| conv1 | 27 x 32 | 1024 | 1 | 1.3% | 32,768 |
| conv2 | 288 x 32 | 1024 | 2 | 7.0% | 65,536 |
| conv3 | 288 x 64 | 256 | 2 | 14.1% | 32,768 |
| conv4 | 576 x 64 | 256 | 3 | 18.8% | 49,152 |
| fc1 | 4096 x 128 | 1 | 16 | 50.0% | 2,048 |
| fc2 | 128 x 10 | 1 | 1 | 2.0% | 10 |
