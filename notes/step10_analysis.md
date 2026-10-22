# Step 10: analysis

Validation split: train weeks 11-20, val weeks 21-30. For "watched in training" we use all interactions in weeks 11-20 regardless of frac threshold (to measure familiarity).

| method | MAP@5 | familiar share (slots) | new share | hit rate familiar | hit rate new | catalog coverage | mean train pop (distinct users) | fit+score runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.472829 | 0.9030 | 0.0970 | 0.5548 | 0.1440 | 5 | 4108.00 | 0.27 |
| user-history | 0.536582 | 0.9868 | 0.0132 | 0.5894 | 0.2169 | 332 | 2970.30 | 0.30 |
| PPR | 0.571957 | 0.9846 | 0.0154 | 0.6207 | 0.2268 | 190 | 3421.45 | 0.27 |
| MF | 0.364193 | 0.9425 | 0.0575 | 0.4813 | 0.1657 | 138 | 3003.58 | 0.27 |
| CBF | 0.209451 | 0.5063 | 0.4937 | 0.4966 | 0.0177 | 300 | 1447.75 | 0.26 |
| hybrid_alt_best | 0.579722 | 0.9824 | 0.0176 | 0.6259 | 0.1712 | 170 | 3295.79 | 0.28 |

## Category mix (top 10)
### popularity
- Инфо: 5040
- Познавательное: 1260

### user-history
- Инфо: 2595
- Развлечения: 1223
- Детям: 1029
- Сериалы: 718
- Познавательное: 439
- Фильмы: 187
- Другое: 109

### PPR
- Инфо: 3638
- Развлечения: 969
- Детям: 794
- Сериалы: 396
- Познавательное: 322
- Другое: 106
- Фильмы: 75

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

## Test file analysis
Test file: submissions/hybrid_wp2_wm0.5_wg1_wa2_wc2_val0.5797.csv
Users served by popularity fallback: 0 (computed from generation logs; see predict output)
Distinct ids recommended: 150
Share of slots that are shows user watched in weeks 11-30: 0.9933

