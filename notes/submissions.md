# Submissions

| date | competition | method | validation MAP@5 | file | git commit | public score | private score | Kaggle ref | status |
|---|---|---|---:|---|---|---:|---:|---:|---|
| 2026-10-22 | sweet.tv - TV program recommender | PPR, alpha=0.7, t=0.3, refit weeks 11-30 | 0.571957 | `submissions/ppr_a0.7_t0.3_val0.5720.csv` | `92584cc` | 0.41709 | 0.39225 | 57533675 | complete |

- Kaggle submission time: 2026-10-22 17:44:02 (Kaggle display).
- Kaggle username: `mariasenenko`.
- Submission message: `Сененко Мария & mariasenenko | PPR a0.7 t0.3 val0.5720`.
- Only one calibration submission was made.
- File validation passed: 1,260 rows, users exactly in `data/submission.csv` order, 5 distinct integer ids per row, and every id in the 6,632-show test pool from ISO weeks 31-42.
- Test pool: 6,632 shows; graph: 4,880 users, 4,146 shows, 507,371 edges; 1,257 target users active, 3 popularity fallbacks; PPR converged in 87 iterations with final L1 `8.393e-11`.
- Test scores are below validation MAP@5: public 0.41709, private 0.39225. This is expected distribution shift from the 11-20 -> 21-30 validation split to the 11-30 -> 31-42 test split.
