# Step 12: R as log1p(sum of frac)

Suggested by reviewing assistant (Claude). R = log1p(sum of frac over qualifying training airings of the show) weights fully-watched episodes more.

Discriminating checks (validation, frac >= 0.3 airings of weeks 11-20):
- R-only ranking (popularity tie-break): count R MAP@5 = 0.537770, sum-of-frac R MAP@5 = 0.541396 (sum-of-frac better).
- User 8377619604347126107 top R values: 2400480 count 4.465908 vs sum-frac 4.284541; 240081 4.094345 vs 3.657078; 20088 3.970292 vs 3.419852 (sum-of-frac gives different ranking of weights).
- My earlier step12 script computed sum-of-frac by taking `train_qualifying.groupby(['user_id','tv_show_id'])['frac'].sum()` from a dataframe where I accidentally re-aggregated incorrectly in that pass (gave raw_history_frac all zeros for those shows), producing wrong low results (0.558). Correct aggregation matches the discriminating numbers.

## Small grid (corrected)
| R def | wM | wG | wE | MAP@5 all |
|---|---:|---:|---:|---:|
| count | 0.5 | 2 | 4 | 0.590785 |
| sumfrac | 0.5 | 2 | 3 | 0.592977 |
| count | 0.5 | 2 | 5 | 0.589163 |
| sumfrac | 0.5 | 2 | 4 | 0.588615 |
| sumfrac | 0.5 | 2 | 5 | 0.587966 |
| count | 0.5 | 2 | 3 | 0.588934 |
| sumfrac | 0 | 2 | 3 | 0.587019 |
| count | 0 | 2 | 4 | 0.588929 |
| sumfrac | 0.5 | 3 | 3 | 0.586738 |
| sumfrac | 0 | 2 | 4 | 0.586204 |
| sumfrac | 0.5 | 3 | 4 | 0.585921 |
| sumfrac | 0 | 3 | 3 | 0.585666 |
| count | 0.5 | 3 | 5 | 0.588198 |
| sumfrac | 0.5 | 3 | 5 | 0.585326 |
| count | 0 | 2 | 3 | 0.587443 |

Best with count R: 0.590785; best with sum-of-frac R: 0.592977 (wM=0.5, wG=2, wE=3). Improvement: +0.002192 >= 0.002.
