# Step 4: PPR and implicit MF

Same validation protocol as Step 3: fit on ISO weeks 11-20, validate on ISO weeks 21-30, truth is `frac >= 0.8`, all 1,260 submission users are evaluated, and the candidate pool is the 6,361 shows airing in weeks 21-30.

## Personalized PageRank

- Bipartite user-show graph from training interactions with `frac >= t`.
- Edge weight: `log1p(number of qualifying airings)`.
- Row-normalised transitions for both blocks; random walk with restart to the target user; `alpha` is edge-following probability.
- Score: stationary mass on show nodes, restricted to the pool.
- Zero-degree graph nodes are excluded; submission users with no qualifying training edges use the popularity fallback.
- L1 tolerance: `1e-10`; maximum iterations: 300.

### Graph checks

| t | train rows | graph users | graph shows | edges | inactive target users | pool shows absent from graph | user row-sum min/max | show row-sum min/max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 716,371 | 4,816 | 2,776 | 286,922 | 12 | 4,296 | 1.000000000000/1.000000000000 | 1.000000000000/1.000000000000 |
| 0.5 | 517,387 | 4,789 | 2,707 | 199,855 | 15 | 4,347 | 1.000000000000/1.000000000000 | 1.000000000000/1.000000000000 |

All graph nodes have positive degree by construction; zero-degree graph nodes: 0. Mass conservation is checked each iteration.

### PPR results

| alpha | t | iterations | final L1 | max mass error | MAP@5 all | MAP@5 truth | runtime, s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.3 | 26 | 6.345e-11 | 3.586e-14 | 0.553376 | 0.563210 | 8.02 |
| 0.50 | 0.3 | 45 | 7.094e-11 | 4.774e-14 | 0.566897 | 0.576972 | 13.57 |
| 0.70 | 0.3 | 87 | 8.334e-11 | 1.821e-14 | 0.571957 | 0.582121 | 25.97 |
| 0.85 | 0.3 | 190 | 9.700e-11 | 2.842e-14 | 0.545525 | 0.555219 | 56.69 |
| 0.30 | 0.5 | 26 | 6.329e-11 | 2.398e-14 | 0.549744 | 0.559513 | 6.33 |
| 0.50 | 0.5 | 45 | 7.076e-11 | 3.197e-14 | 0.563697 | 0.573714 | 10.78 |
| 0.70 | 0.5 | 87 | 8.313e-11 | 1.965e-14 | 0.568457 | 0.578559 | 20.48 |
| 0.85 | 0.5 | 190 | 9.680e-11 | 2.109e-14 | 0.549226 | 0.558986 | 49.70 |

Best PPR: alpha=0.70, t=0.3, MAP@5 all=0.571957, truth=0.582121.

Total PPR runtime: 193.23 s. Seeds: none (PPR is deterministic).
