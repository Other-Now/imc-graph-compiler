Analytical model only (imc/crossbar.py). Weight-stationary: if the network's tiles exceed the arrays on chip, all weights are rewritten once per batch.

| model | arrays on chip (256x256) | tiles needed | fits? | weight writes/img, batch 1 | weight writes/img, batch 16 | weight writes/img, batch 256 | energy/img batch 1 -> 256 (6-bit) |
|---|---|---|---|---|---|---|---|
| mlp | 4 | 14 | no | 820,480 | 51,280 | 3,205 | 17,236 -> 73.1 nJ |
| mlp | 16 | 14 | yes | 0 | 0 | 0 | 6 -> 5.8 nJ |
| mlp | 64 | 14 | yes | 0 | 0 | 0 | 6 -> 5.8 nJ |
| cnn | 4 | 25 | no | 590,944 | 36,934 | 2,308 | 12,628 -> 267.1 nJ |
| cnn | 16 | 25 | no | 590,944 | 36,934 | 2,308 | 12,628 -> 267.1 nJ |
| cnn | 64 | 25 | yes | 0 | 0 | 0 | 219 -> 218.7 nJ |

Same 1M-cell budget (16 x 256x256), CNN, batch 1, 6-bit ADC, by crossbar shape:

| crossbar | arrays | tiles needed | fits? | energy/img |
|---|---|---|---|---|
| 256x256 | 16 | 25 | no | 12,628 nJ |
| 128x128 | 64 | 45 | yes | 263 nJ |
| 64x64 | 256 | 150 | yes | 352 nJ |

The CNN has only 0.59M weights (9 arrays' worth), but its conv layers use 1-19% of a 256x256 array, so it
needs 25 arrays and does not fit.
