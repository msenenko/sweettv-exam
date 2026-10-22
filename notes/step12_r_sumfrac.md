# Step 12: R as log1p(sum of frac)

Suggested by reviewing assistant (Claude). R = log1p(sum of frac over qualifying training airings of the show) weights fully-watched episodes more.

## Small grid
| R def | wM | wG | wE | MAP@5 all |
|---|---:|---:|---:|---:|
| count | 0.5 | 2 | 4 | 0.590785 |
| count | 0.5 | 2 | 5 | 0.589163 |
| count | 0.5 | 2 | 3 | 0.588934 |
| count | 0 | 2 | 4 | 0.588929 |
| count | 0.5 | 1 | 3 | 0.588537 |
| count | 0 | 1 | 3 | 0.588507 |
| count | 0.5 | 3 | 5 | 0.588198 |
| count | 0 | 2 | 3 | 0.587443 |
| count | 0 | 2 | 5 | 0.587394 |
| count | 0 | 3 | 5 | 0.586909 |
| count | 0 | 1 | 4 | 0.586222 |
| count | 0.5 | 1 | 5 | 0.585983 |
| count | 0.5 | 1 | 4 | 0.585950 |
| count | 0 | 1 | 5 | 0.585436 |
| count | 0.5 | 3 | 4 | 0.584694 |

Best overall: count wM=0.5 wG=2 wE=4 MAP=0.590785
Best with count R: 0.590785; best with sum-of-frac R: 0.558058
Improvement: 0.000000

