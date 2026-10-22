# Step 10: analysis

Validation split: train weeks 11-20, val weeks 21-30. For "watched in training" we use all interactions in weeks 11-20 regardless of frac threshold (to measure familiarity). User-history/popularity use training_rows filtered to the validation candidate pool with frac>=0.3 as in step 3.

| method | MAP@5 | familiar share (slots) | new share | hit rate familiar | hit rate new | catalog coverage | mean train pop (distinct users) | scoring runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.472829 | 0.9030 | 0.0970 | 0.5548 | 0.1440 | 5 | 4108.00 | 0.00 |
| user-history | 0.542232 | 0.9860 | 0.0140 | 0.5959 | 0.2045 | 240 | 3032.28 | 0.00 |
| PPR | 0.572562 | 0.9848 | 0.0152 | 0.6214 | 0.2083 | 188 | 3427.39 | 17.37 |
| MF | 0.364193 | 0.9425 | 0.0575 | 0.4813 | 0.1657 | 138 | 3003.58 | 2.04 |
| CBF | 0.209451 | 0.5063 | 0.4937 | 0.4966 | 0.0177 | 300 | 1447.75 | 3.44 |
| hybrid_alt_best | 0.579722 | 0.9824 | 0.0176 | 0.6259 | 0.1712 | 170 | 3295.79 | 27.60 |

Runtimes above are scoring/inference only (graph build/fitting done separately in full runs); fit+score times reported in earlier steps (e.g. PPR ~26s per config including graph, MF ~3s).

## Category mix (top 10)
### popularity
- Инфо: 5040
- Познавательное: 1260

### user-history
- Инфо: 2632
- Развлечения: 1367
- Детям: 992
- Сериалы: 686
- Познавательное: 430
- Другое: 100
- Фильмы: 93

### PPR
- Инфо: 3652
- Развлечения: 965
- Детям: 788
- Сериалы: 391
- Познавательное: 327
- Другое: 103
- Фильмы: 74

### MF
- Инфо: 2216
- Развлечения: 1624
- Детям: 850
- Сериалы: 806
- Познавательное: 606
- Другое: 114
- Фильмы: 84

### CBF
- Фильмы: 3416
- Инфо: 1143
- Развлечения: 728
- Детям: 591
- Сериалы: 323
- Познавательное: 61
- Другое: 38

### hybrid_alt_best
- Инфо: 3246
- Детям: 1037
- Развлечения: 903
- Сериалы: 513
- Познавательное: 402
- Другое: 156
- Фильмы: 43

## Ablation of final hybrid (alt, wP2 wM0.5 wG1 wA2 wC2) dropping one component
- drop R: MAP=0.569801, delta=-0.009920
- drop P: MAP=0.579702, delta=-0.000019
- drop M: MAP=0.578876, delta=-0.000846
- drop G: MAP=0.576125, delta=-0.003597
- drop A: MAP=0.572898, delta=-0.006824
- drop C: MAP=0.575065, delta=-0.004656

Note: dropping P changes MAP by only -0.00002, so PPR adds almost nothing once R, G, A, C are present in this configuration.

## Test file analysis
Test file: submissions/hybrid_wp2_wm0.5_wg1_wa2_wc2_val0.5797.csv
Users with no qualifying interaction (frac>=0.3) in weeks 11-30 (would use popularity fallback in PPR test): 3
Distinct ids recommended: 150
Share of slots that are shows user watched in weeks 11-30: 0.9933

