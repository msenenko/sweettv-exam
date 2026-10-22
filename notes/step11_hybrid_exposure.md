# Step 11: channel-exposure component (suggested by reviewing assistant)

Suggested by reviewing assistant (Claude). Definition: E(u,i)=log1p(10 * sum_c share_u(c) * n_airings(i,c,target window)) where share_u(c)=user qualifying training airings on c / user total; schedule used only for airing info; E divided by user max E (alternative normalization). Score = R + wP*P + wM*M + wG*G + wA*A + wC*C + wE*E.

## Prediction
Quick check expectation ~0.587-0.588 for R + 1*P + 2*G + 4*E.

## Top 10 (validation)
| rank | wP | wM | wG | wA | wC | wE | MAP@5 all | delta to best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.5 | 2 | 0 | 0 | 4 | 0.590785 | 0.000000 |
| 2 | 1 | 0.5 | 2 | 0 | 0 | 4 | 0.590516 | 0.000269 |
| 3 | 2 | 0.5 | 2 | 0 | 0 | 6 | 0.590256 | 0.000529 |
| 4 | 0 | 0.5 | 2 | 0 | 1 | 4 | 0.589959 | 0.000826 |
| 5 | 1 | 0.5 | 2 | 1 | 1 | 4 | 0.589907 | 0.000879 |
| 6 | 1 | 0 | 2 | 0 | 0 | 4 | 0.589812 | 0.000974 |
| 7 | 2 | 0 | 1 | 0 | 0 | 4 | 0.589791 | 0.000994 |
| 8 | 2 | 0.5 | 1 | 0 | 0 | 4 | 0.589638 | 0.001148 |
| 9 | 2 | 0.5 | 2 | 1 | 0 | 4 | 0.589597 | 0.001188 |
| 10 | 0 | 0.5 | 2 | 0 | 1 | 3 | 0.589528 | 0.001258 |

Best: wP=0, wM=0.5, wG=2, wA=0, wC=0, wE=4 MAP=0.590785
Simplicity pick: wP=0, wM=0, wG=2, wA=0, wC=0, wE=4 MAP=0.588929

## Ablation of best
- drop R: MAP=0.552014, delta=-0.038772
- drop P: MAP=0.590785, delta=0.000000
- drop M: MAP=0.588929, delta=-0.001856
- drop G: MAP=0.586241, delta=-0.004545
- drop A: MAP=0.590785, delta=0.000000
- drop C: MAP=0.590785, delta=0.000000
- drop E: MAP=0.567146, delta=-0.023640
