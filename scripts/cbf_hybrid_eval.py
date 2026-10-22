#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluate import *
import json
from cbf import compute_cbf, rank_cbf, evaluate_cbf_predictions
from hybrid import build_components, score_components, rank_scores, evaluate_rankings

inter = load_interactions('data/interactions.pkl')
users = load_submission_users('data/submission.csv')
truth = build_truth(inter, users, *VALIDATION_WEEKS)
train = inter[inter.iso_week.between(*TRAIN_WEEKS) & (inter.frac >= 0.3)]
train_user_show = set((int(u), int(s)) for u, s in train[['user_id', 'tv_show_id']].drop_duplicates().itertuples(index=False))
train_shows = set(train.tv_show_id)
pairs = []
for u, ts in truth.items():
    for s in ts:
        pairs.append((int(u), int(s)))
n_pairs = len(pairs)
watched_by_user = sum(1 for u, s in pairs if (u, s) in train_user_show)
show_absent_from_train = sum(1 for u, s in pairs if s not in train_shows)
print(f"n_pairs={n_pairs}, watched_by_user_pairs={watched_by_user} ({watched_by_user/n_pairs:.6f}), show_absent_from_train_pairs={show_absent_from_train} ({show_absent_from_train/n_pairs:.6f})")
print(f"training_set: interactions with iso_week in {TRAIN_WEEKS} and frac>=0.3 (qualifying training interactions)")

pool = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
train_qual_pool = train[train.tv_show_id.isin(pool)]
fallback_order = popularity_order(train_qual_pool, pool)
fallback = fill_ranking(fallback_order, fallback_order, K)

cbf_w = {"cat": 0.5, "genre": 1.0, "channel": 0.5, "tfidf": 3.0}
cbf_scores, dims, _ = compute_cbf(inter, users, pool, Path('data'), TRAIN_WEEKS, VALIDATION_WEEKS, cbf_w)
cbf_rankings = rank_cbf(cbf_scores, pool, fallback)
m_cbf = evaluate_cbf_predictions(users, truth, cbf_rankings)
print('CBF standalone:', m_cbf, dims)

components, metadata = build_components(inter, users, pool, Path('data'), TRAIN_WEEKS, VALIDATION_WEEKS, normalization='alternative', include_cbf=True)
res = json.loads(Path('data/hybrid_results.json').read_text())
base = res['best']
for wk in [0, 0.5, 1, 2]:
    w = {'P': base['P'], 'M': base['M'], 'G': base['G'], 'A': base['A'], 'C': base['C'], 'K': wk}
    scores = score_components(components, w)
    rankings = rank_scores(scores, pool, metadata['fallback'])
    m = evaluate_rankings(users, truth, rankings)
    print(f'base+K={wk}:', m)

chosen = res['chosen']
for wk in [0, 0.5, 1, 2]:
    w = {'P': chosen['P'], 'M': chosen['M'], 'G': chosen['G'], 'A': chosen['A'], 'C': chosen['C'], 'K': wk}
    scores = score_components(components, w)
    rankings = rank_scores(scores, pool, metadata['fallback'])
    m = evaluate_rankings(users, truth, rankings)
    print(f'chosen+K={wk}:', m)

hist = {u:set(g.tv_show_id) for u,g in train.groupby('user_id')}
for label, w in [('best_noK', {'P':base['P'],'M':base['M'],'G':base['G'],'A':base['A'],'C':base['C'],'K':0}),
                 ('best_K05', {'P':base['P'],'M':base['M'],'G':base['G'],'A':base['A'],'C':base['C'],'K':0.5})]:
    scores = score_components(components, w)
    rankings = rank_scores(scores, pool, metadata['fallback'])
    total = 0; never = 0
    for i,u in enumerate(users):
        hu = hist.get(int(u), set())
        for s in rankings[i]:
            total += 1
            if int(s) not in hu: never += 1
    print(label, never/total if total else 0)
