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

## Implicit matrix factorisation

- Library: `implicit 0.7.3`, exact ALS (`use_cg=False`), `regularization=0.1`, `iterations=15`, seed `42`, `num_threads=1`.
- Training matrix: all training users and all shows seen in weeks 11-20 with `frac >= t`; values are `log1p(number of qualifying airings)`.
- Hu-Koren-Volinsky implicit-feedback ALS; confidence is `1 + a * log1p(number of qualifying airings)`.
- Items are all training shows; pool shows absent from training get score 0. Scores are `user_factors @ item_factors.T` over the pool.
- Positive scores only; fallback to popularity for users with no training edges or fewer than five positive candidate scores.
- Sanity check on all training users with at least one positive item: mean percentile of the user's own training shows among all training shows, and fraction of users with at least one own show in the reconstructed top 5.

### MF results

| factors | a | t | own-item percentile | own-item hit@5 | MAP@5 all | MAP@5 truth | runtime, s | sanity users |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 5 | 0.3 | 0.9344 | 0.9981 | 0.312166 | 0.317714 | 3.49 | 4,816 |
| 32 | 20 | 0.3 | 0.9518 | 0.9549 | 0.205777 | 0.209433 | 3.40 | 4,816 |
| 64 | 5 | 0.3 | 0.9576 | 0.9994 | 0.269512 | 0.274301 | 8.30 | 4,816 |
| 64 | 20 | 0.3 | 0.9719 | 0.9385 | 0.152919 | 0.155637 | 8.41 | 4,816 |
| 32 | 5 | 0.5 | 0.9452 | 0.9952 | 0.364098 | 0.370568 | 2.69 | 4,789 |
| 32 | 20 | 0.5 | 0.9643 | 0.9626 | 0.244720 | 0.249069 | 2.71 | 4,789 |
| 64 | 5 | 0.5 | 0.9674 | 0.9987 | 0.326541 | 0.332343 | 6.42 | 4,789 |
| 64 | 20 | 0.5 | 0.9815 | 0.9643 | 0.208953 | 0.212666 | 6.48 | 4,789 |

Best MF: factors=32, a=5, t=0.5, MAP@5 all=0.364098, truth=0.370568.

Total MF runtime: 43.71 s. Seed: 42; threads: 1.

## Comparison

| method | config | MAP@5 all | MAP@5 truth | runtime, s |
|---|---|---:|---:|---:|
| popularity | t=0.3 | 0.472829 | 0.481231 | 0.00 |
| user-history | t=0.3 | 0.542232 | 0.551868 | 0.00 |
| PPR | alpha=0.70, t=0.3 | 0.571957 | 0.582121 | 26.09 |
| MF | factors=32, a=5, t=0.5 | 0.364098 | 0.370568 | 2.69 |

PPR runtime includes one graph build; MF runtime includes fitting and the sanity check. Baseline runtime covers ranking and evaluation only.

## Prediction and interpretation

Before the run, the expected order was MF > PPR > user-history > popularity, with `t=0.3` best and PPR `alpha` around 0.5-0.7.

Observed MAP@5 order: PPR (0.571957) > user-history (0.542232) > popularity (0.472829) > MF (0.364098).

The prediction that MF wins was not confirmed; PPR is stronger. Best PPR uses alpha=0.70, t=0.3; best MF uses factors=32, a=5, t=0.5.

MF remains below popularity on this validation split. This is a real result, not a training failure: the best MF config reconstructs the training data well (own-item percentile 0.9452, own-item hit@5 0.9952), but held-out MAP@5 rewards exact repeat titles and broad candidate coverage. Popularity can rank all 6,361 pool shows, whereas the best MF config (`t=0.5`) gives positive scores only to the 2,707 shows seen in weeks 11-20, leaving 4,347 pool shows at zero. ALS also spreads score across latent neighbours instead of concentrating it on the specific repeat shows that dominate the truth. We keep the result as observed and do not tune further on validation beyond the specified grid.
