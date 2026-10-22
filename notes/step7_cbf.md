# Step 7: content-based filtering

Motivation: using qualifying training interactions (iso_week 11-20, frac >= 0.3). Over validation truth pairs (submission users, frac >= 0.8 in weeks 21-30, n_pairs=42831): 11767 pairs (27.473%) were watched by the same user in training interactions; 11231 pairs (26.222%) are shows absent from any qualifying training interaction. The earlier 44.1% figures were wrong (double-counting / incorrect comparison) — both quantities are ~26-27%, not identical.

## Features (from schedule metadata, allowed for target window)
Item vectors built from `export_arh_21-30-final.csv` (validation) / `export_arh_31-42-final.csv` (test) metadata for pool shows:
- `tv_show_category` (categorical, one-hot)
- `tv_show_genre_1`, `tv_show_genre_2`, `tv_show_genre_3` (union of genres, multi-hot)
- `channel_id` channels the show airs on in the target window (union, multi-hot)
- TF-IDF of title words (lowercased, punctuation removed, season/episode numbering stripped like `s01e01`, `1x01`, `сезон`, `серия`; tokens length > 2)

Blocks are L2-normalised before combining; combined vector is weighted sum then L2-normalised. User profile = sum of item vectors of training airings with `frac >= 0.3` weighted by `log1p(count)`, L2-normalised. Score = cosine similarity between user profile and every pool item.

Weights tested (per-block): `cat`, `genre`, `channel`, `tfidf`.

## Standalone CBF (validation)
| config | cat/genre/channel/tfidf dims | MAP@5 all | MAP@5 truth |
|---|---|---:|---:|
| equal | 8/64/188/7278 | 0.137343 | 0.139783 |
| genre_cat_tfidf | 8/64/188/7278 | 0.101750 | 0.103558 |
| genre_tfidf | 8/64/188/7278 | 0.127447 | 0.129712 |
| all_strong | 8/64/188/7278 | 0.135198 | 0.137601 |
| ch_cat_tfidf | 8/64/188/7278 | 0.128393 | 0.130674 |
| cat_strong | 8/64/188/7278 | 0.088526 | 0.090099 |
| tfidf_heavy | 8/64/188/7278 | 0.209451 | 0.213173 |
| ch_heavy | 8/64/188/7278 | 0.155573 | 0.158338 |
| balanced2 | 8/64/188/7278 | 0.146257 | 0.148856 |

Best standalone CBF: `cat=0.5, genre=1.0, channel=0.5, tfidf=3.0` giving MAP@5 all=0.209451.

## Adding K (per-user max-normalised CBF) to best alternative hybrid
Best alternative hybrid (overall best): wP=2.0, wM=0.5, wG=1.0, wA=2.0, wC=2.0, normalization=alternative (val 0.579722).
| +wK | MAP@5 all | MAP@5 truth | delta |
|---:|---:|---:|---:|
| 0 | 0.579722 | 0.590024 | 0.000000 |
| 0.5 | 0.579701 | 0.590003 | -0.000020 |
| 1.0 | 0.579594 | 0.589894 | -0.000128 |
| 2.0 | 0.578455 | 0.588734 | -0.001267 |

Adding K hurts the best overall hybrid. For the chosen simpler config (wP=2.0,wM=0.0,wG=1.0,wA=1.5,wC=1.0, alt): +wK=0.5 gives 0.578830 (+0.001010), wK=1 gives 0.577683 (-0.000137). Best for chosen config is +wK=0.5 at 0.578830.

## Share of recommended slots going to shows never watched by user in training (top5)
| config | share never-watched in training |
|---|---:|
| best overall hybrid (no K) | 0.02175 |
| best overall hybrid (wK=0.5) | 0.02000 |

CBF adds very little to the hybrid on this split and can slightly reduce performance; in the simpler chosen config small gain appears at wK=0.5 but not compelling. We do not add K to the final submitted configs.
