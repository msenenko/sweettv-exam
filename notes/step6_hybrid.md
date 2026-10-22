# Step 6: hybrid score fusion

Validation protocol is unchanged: fit on ISO weeks 11-20, validate on 21-30, truth is `frac >= 0.8`, all 1,260 submission users are evaluated, and the candidate pool is the 6,361 shows airing in weeks 21-30.

## Components and normalisation

- `R`: `log1p(user's qualifying training airings of the show)`; qualifying means `frac >= 0.3`.
- `P`: PPR stationary show mass with `alpha=0.7`, `t=0.3`.
- `M`: implicit MF score with factors=32, a=5, `t=0.5`, `regularization=0.1`, 15 iterations, seed 42.
- `G`: global popularity, defined as the number of unique training users with `frac >= 0.3` for the show.
- `A`: `log1p(number of schedule airings of the show in the target window)`; the schedule is used only for airing information.
- `C`: maximum, over channels the show airs on in the target window, of the user's share of their qualifying training airings on that channel.
- Every component is min-max normalised per user over all 6,361 pool shows. For `G` and `A`, which are user-independent, the same pool-wide min-max is applied. If a user's max equals min, that component is set to zero for that user.
- Fusion: `score = R + wP*P + wM*M + wG*G + wA*A + wC*C`; positive scores only, with popularity fill for short rankings.

## Pre-run prediction

Expected ordering of contribution: `C` and `R` largest, `P` useful, `G` and `A` small positive signals, and `M` near zero or negative because the best standalone MF was below popularity.

## Top 10 validation configs

| rank | wP | wM | wG | wA | wC | MAP@5 all | MAP@5 truth | delta to best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.0 | 0.5 | 0.5 | 1.5 | 1.0 | 0.575618 | 0.585847 | 0.000000 |
| 2 | 2.0 | 0.0 | 1.0 | 1.5 | 1.0 | 0.575471 | 0.585698 | 0.000147 |
| 3 | 2.0 | 0.5 | 1.0 | 1.0 | 1.0 | 0.575362 | 0.585587 | 0.000256 |
| 4 | 2.0 | 0.0 | 0.5 | 1.5 | 1.0 | 0.575102 | 0.585322 | 0.000516 |
| 5 | 2.0 | 0.5 | 1.0 | 1.5 | 1.0 | 0.574807 | 0.585022 | 0.000811 |
| 6 | 2.0 | 0.5 | 0.5 | 0.5 | 0.0 | 0.574279 | 0.584484 | 0.001339 |
| 7 | 2.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.574060 | 0.584261 | 0.001558 |
| 8 | 2.0 | 0.5 | 0.5 | 1.0 | 1.0 | 0.573716 | 0.583912 | 0.001902 |
| 9 | 2.0 | 0.0 | 0.5 | 1.0 | 1.0 | 0.573681 | 0.583876 | 0.001937 |
| 10 | 0.0 | 0.5 | 0.5 | 0.5 | 0.0 | 0.573500 | 0.583692 | 0.002118 |

Best: wP=2.0, wM=0.5, wG=0.5, wA=1.5, wC=1.0, MAP@5 all=0.575618, truth=0.585847.

Chosen by the simplicity rule (fewest non-zero optional weights, then smallest total weight, then lexicographic weights, among configs within 0.002 of the best): wP=2.0, wM=0.5, wG=0.5, wA=0.5, wC=0.0, MAP@5 all=0.574279, truth=0.584484.

## Ablation from the best config

Each row drops one component from the best config; larger positive `delta` means that component contributed more.

| dropped | weight | MAP@5 all | MAP@5 truth | delta vs best |
|---|---:|---:|---:|---:|
| R | 1.0 | 0.568588 | 0.578692 | 0.007030 |
| P | 2.0 | 0.554058 | 0.563904 | 0.021560 |
| M | 0.5 | 0.575102 | 0.585322 | 0.000516 |
| G | 0.5 | 0.574127 | 0.584330 | 0.001491 |
| A | 1.5 | 0.561132 | 0.571103 | 0.014486 |
| C | 1.0 | 0.570831 | 0.580975 | 0.004787 |

## Interpretation

Observed ablation deltas: P +0.021560, A +0.014486, R +0.007030, C +0.004787, G +0.001491, M +0.000516.

The validation grid was used only for weight selection. Test predictions use the chosen config wP=2.0, wM=0.5, wG=0.5, wA=0.5, wC=0.0 refit on weeks 11-30 with the target window 31-42.

Grid runtime: 45.90 s. Components and grid are deterministic except MF, which uses seed 42.

## Extended grid

The first grid hit its `wP` and `wA` maxima, so it was extended once: `wP in {2,3,4,6}`, `wA in {1,1.5,2,3}`, `wG in {0.5,1,2}`, `wC in {0,1,2}`, and `wM in {0,0.5}`. Default min-max normalisation is unchanged.

### Extended grid top 10

| rank | wP | wM | wG | wA | wC | MAP@5 all | MAP@5 truth | delta to best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.0 | 0.5 | 0.5 | 1.5 | 1.0 | 0.579086 | 0.589377 | 0.000000 |
| 2 | 4.0 | 0.5 | 0.5 | 1.0 | 1.0 | 0.578784 | 0.589069 | 0.000302 |
| 3 | 6.0 | 0.5 | 1.0 | 1.0 | 1.0 | 0.578525 | 0.588806 | 0.000561 |
| 4 | 6.0 | 0.5 | 0.5 | 1.0 | 1.0 | 0.578465 | 0.588745 | 0.000621 |
| 5 | 4.0 | 0.5 | 0.5 | 2.0 | 1.0 | 0.578377 | 0.588655 | 0.000709 |
| 6 | 6.0 | 0.0 | 0.5 | 1.5 | 1.0 | 0.578283 | 0.588560 | 0.000803 |
| 7 | 4.0 | 0.5 | 1.0 | 1.5 | 1.0 | 0.578252 | 0.588528 | 0.000834 |
| 8 | 3.0 | 0.5 | 0.5 | 1.5 | 1.0 | 0.578245 | 0.588521 | 0.000841 |
| 9 | 4.0 | 0.5 | 1.0 | 1.0 | 1.0 | 0.578214 | 0.588489 | 0.000872 |
| 10 | 6.0 | 0.0 | 0.5 | 1.0 | 1.0 | 0.578196 | 0.588471 | 0.000890 |

Extended-grid best: wP=4.0, wM=0.5, wG=0.5, wA=1.5, wC=1.0, MAP@5 all=0.579086, truth=0.589377.

Extended-grid chosen by the same simplicity rule: wP=3.0, wM=0.0, wG=0.5, wA=1.5, wC=1.0, MAP@5 all=0.578165, truth=0.588439.

## Alternative normalisation

This variant leaves `R` as the raw `log1p` count and `C` as the raw channel share, divides `P` by the user's maximum `P`, keeps `M` as per-user min-max, and keeps `G` and `A` as pool-wide min-max.

### Alternative-normalisation top 10

| rank | wP | wM | wG | wA | wC | MAP@5 all | MAP@5 truth | delta to best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.0 | 0.5 | 1.0 | 2.0 | 2.0 | 0.579722 | 0.590024 | 0.000000 |
| 2 | 2.0 | 0.5 | 1.0 | 1.0 | 2.0 | 0.579502 | 0.589801 | 0.000219 |
| 3 | 2.0 | 0.5 | 1.0 | 1.5 | 2.0 | 0.579409 | 0.589705 | 0.000313 |
| 4 | 3.0 | 0.5 | 1.0 | 2.0 | 2.0 | 0.579405 | 0.589702 | 0.000316 |
| 5 | 2.0 | 0.5 | 0.5 | 2.0 | 2.0 | 0.579308 | 0.589603 | 0.000413 |
| 6 | 3.0 | 0.5 | 1.0 | 1.5 | 2.0 | 0.579267 | 0.589561 | 0.000454 |
| 7 | 2.0 | 0.0 | 1.0 | 1.5 | 2.0 | 0.579237 | 0.589530 | 0.000485 |
| 8 | 4.0 | 0.5 | 0.5 | 3.0 | 2.0 | 0.579187 | 0.589480 | 0.000534 |
| 9 | 2.0 | 0.0 | 2.0 | 1.5 | 1.0 | 0.579105 | 0.589396 | 0.000616 |
| 10 | 3.0 | 0.5 | 1.0 | 3.0 | 2.0 | 0.579067 | 0.589357 | 0.000655 |

Alternative-normalisation best: wP=2.0, wM=0.5, wG=1.0, wA=2.0, wC=2.0, MAP@5 all=0.579722, truth=0.590024.

Alternative-normalisation chosen by the same simplicity rule: wP=2.0, wM=0.0, wG=1.0, wA=1.5, wC=1.0, MAP@5 all=0.577820, truth=0.588088.

## Overall selection

The simplicity rule is applied across both variants: fewest non-zero optional weights, then smallest total weight, then lexicographic weights; default min-max wins exact ties.

| role | normalisation | wP | wM | wG | wA | wC | MAP@5 all | MAP@5 truth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| best overall | alternative | 2.0 | 0.5 | 1.0 | 2.0 | 2.0 | 0.579722 | 0.590024 |
| chosen overall | alternative | 2.0 | 0.0 | 1.0 | 1.5 | 1.0 | 0.577820 | 0.588088 |

Final chosen config: wP=2.0, wM=0.0, wG=1.0, wA=1.5, wC=1.0 with `alternative` normalisation, MAP@5 all=0.577820, truth=0.588088.

Extended/alternative runtime: 113.73 s.

