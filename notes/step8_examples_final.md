# Step 8 (updated for final config): example users

Selection rule (stated before looking at results): among submission users with non-empty validation truth, rank by number of qualifying training airings (weeks 11-20, frac >= 0.3); heavy viewer = user at the 90th percentile, light viewer = user at the 10th percentile (nearest rank, ties by smallest user_id). Index formula used: k = int(round(p * (len(vals)-1))) over values sorted ascending, then among candidates with that value pick smallest user_id.

Note on titles/categories: looked up from schedule CSVs (first occurrence per show_id); some airing rows have concatenated strings (e.g. 10002451 appears as "Великолепный векПсы под прикрытием" in some dumps) — title is taken from the stored schedule row used for lookup, not reconstructed.

Light viewer (10th): user_id=901367918461480780, qualifying training airings=49
Heavy viewer (90th): user_id=4103666576780286256, qualifying training airings=368

## User 901367918461480780 (training airings=49)

### Top 5 shows watched in training (weeks 11-20, frac>=0.3)
| rank | tv_show_id | title | category | count |
|---:|---:|---|---|---:|
| 1 | 2400480 | 'ТСН. Неделя'. | Инфо | 7 |
| 2 | 200432 | Дизель-шоу. 8 марта. | Развлечения | 2 |
| 3 | 400426 | 'Меняю жену'. | Развлечения | 2 |
| 4 | 10002451 | Т/с 'Великолепный векПсы под прикрытием'. | Сериалы | 2 |
| 5 | 6700482 | 'Секретные материалы'. | Другое | 2 |

Validation truth (frac>=0.8, weeks 21-30): 24 shows

### Final hybrid (R=sum frac, wM=0.5,wG=2,wE=3) Top-5 validation (AP@5=0.100000)
| rank | tv_show_id | title | category | in_truth | watched_in_train |
|---:|---:|---|---|---:|---:|
| 1 | 2400480 | 'ТСН. Неделя'. | Инфо | N | Y |
| 2 | 240081 | Факты. День. | Инфо | Y | Y |
| 3 | 6700482 | 'Секретные материалы'. | Другое | N | Y |
| 4 | 6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное | N | Y |
| 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | N | N |

### Final hybrid Top-5 in TEST submission (hybrid_exp2_wp0_wm0.5_wg2_wa0_wc0_we3_rfrac_val0.5930.csv)
| rank | tv_show_id | title | category |
|---:|---:|---|---|
| 1 | 2400480 | 'ТСН. Неделя'. | Инфо |
| 2 | 240081 | Факты. День. | Инфо |
| 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо |
| 4 | 12002856 | М/с 'Элвин и бурундуки'. | Детям |
| 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения |

## User 4103666576780286256 (training airings=368)

### Top 5 shows watched in training (weeks 11-20, frac>=0.3)
| rank | tv_show_id | title | category | count |
|---:|---:|---|---|---:|
| 1 | 1100411 | Премьера. 'Вещдок. Особый случай. Око за око'. | Познавательное | 36 |
| 2 | 240081 | Факты. День. | Инфо | 35 |
| 3 | 5200303 | 'Следствие вели...' с Леонидом Каневским. | Познавательное | 31 |
| 4 | 2400480 | 'ТСН. Неделя'. | Инфо | 29 |
| 5 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 29 |

Validation truth (frac>=0.8, weeks 21-30): 27 shows

### Final hybrid (R=sum frac, wM=0.5,wG=2,wE=3) Top-5 validation (AP@5=0.600000)
| rank | tv_show_id | title | category | in_truth | watched_in_train |
|---:|---:|---|---|---:|---:|
| 1 | 240081 | Факты. День. | Инфо | Y | Y |
| 2 | 2400480 | 'ТСН. Неделя'. | Инфо | Y | Y |
| 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | Y | Y |
| 4 | 1100411 | Премьера. 'Вещдок. Особый случай. Око за око'. | Познавательное | N | Y |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | N | Y |

### Final hybrid Top-5 in TEST submission (hybrid_exp2_wp0_wm0.5_wg2_wa0_wc0_we3_rfrac_val0.5930.csv)
| rank | tv_show_id | title | category |
|---:|---:|---|---|
| 1 | 2400480 | 'ТСН. Неделя'. | Инфо |
| 2 | 240081 | Факты. День. | Инфо |
| 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо |
| 4 | 1100411 | Премьера. 'Вещдок. Особый случай. Око за око'. | Познавательное |
| 5 | 500281 | Ток-шоу 'Говорит Украина'. | Развлечения |

