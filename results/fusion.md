Batch 1 (CNN, MLP) and 16 tokens (transformer). Kernels = linalg ops other than fill; conv padding counts as a kernel.

| model | kernels unfused | intermediate bytes unfused | kernels fused (mine) | intermediate bytes fused | upstream `-linalg-fuse-elementwise-ops` | Inductor fused C++ kernels + extern calls |
|---|---|---|---|---|---|---|
| mlp | 8 | 4,648 | 6 | 3,112 (33% less) | 6 | 2 + 3 |
| cnn | 19 | 1,516,632 | 14 | 1,122,904 (26% less) | 14 | 6 + 6 |
| transformer | 29 | 90,496 | 26 | 74,112 (18% less) | 26 | 4 + 8 |
