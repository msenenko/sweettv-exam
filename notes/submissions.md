# Submissions

| date | competition | method | validation MAP@5 | file | git commit | public score | private score | Kaggle ref | status |
|---|---|---|---:|---|---|---:|---:|---:|---|
| 2026-10-22 | sweet.tv - TV program recommender | PPR, alpha=0.7, t=0.3, refit weeks 11-30 | 0.571957 | `submissions/ppr_a0.7_t0.3_val0.5720.csv` | `92584cc` | 0.41709 | 0.39225 | 57533675 | complete |
| 2026-10-22 | sweet.tv - TV program recommender | Hybrid (alt), wP=2.0, wM=0.0, wG=1.0, wA=1.5, wC=1.0, refit weeks 11-30, target 31-42 | 0.577820 | `submissions/hybrid_wp2_wm0_wg1_wa1.5_wc1_val0.5778.csv` | `d83242b` | 0.42382 | 0.40377 | 57534227 | complete |
| 2026-10-22 | sweet.tv - TV program recommender | Hybrid (alt, best overall), wP=2.0, wM=0.5, wG=1.0, wA=2.0, wC=2.0, refit weeks 11-30, target 31-42 | 0.579722 | `submissions/hybrid_wp2_wm0.5_wg1_wa2_wc2_val0.5797.csv` | `d83242b` | 0.43759 | 0.41308 | 57534231 | complete |
| 2026-10-22 | sweet.tv - TV program recommender | Hybrid + E (Claude sugg.), wP=0.0, wM=0.5, wG=2.0, wA=0.0, wC=0.0, wE=4.0, alt norm, refit 11-30, target 31-42 | 0.590785 | `submissions/hybrid_exp_wp0_wm0.5_wg2_wa0_wc0_we4_val0.5908.csv` | `7d58294` | 0.44162 | 0.42530 | 57539529 | complete |

- Kaggle username: `mariasenenko`.
- Messages: first hybrid `Сененко Мария & mariasenenko | hyb wp2 wm0 wg1 wa1.5 wc1 val0.5778`; second `Сененко Мария & mariasenenko | hyb wp2 wm0.5 wg1 wa2 wc2 val0.5797`.
- All files validated (1,260 users in template order, 5 distinct ids, ids in test pool). Normalisation for hybrid is the `alternative` variant specified.
- Validation gains transferred to test: PPR 0.572 -> hybrid chosen 0.578 (+0.006) gave public 0.417->0.424, private 0.392->0.404. Best-overall 0.580 gave public 0.438, private 0.413 (further improvement). Test remains below validation due to distribution shift.
