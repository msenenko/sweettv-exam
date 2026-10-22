#!/usr/bin/env python3
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from evaluate import *
from ppr import build_graph, transition_blocks, ppr, rank_pool
from implicit.als import AlternatingLeastSquares
from mf import build_matrix, rank_scores
from cbf import compute_cbf, rank_cbf, evaluate_cbf_predictions
from hybrid import build_components, score_components, rank_scores as rank_hybrid_scores

def get_meta():
    meta = {}
    for fn in ['export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']:
        try:
            df = pd.read_csv(f'data/{fn}', usecols=['tv_show_id','tv_show_title','tv_show_category'])
            for _,r in df.iterrows():
                sid = int(r.tv_show_id)
                if sid not in meta:
                    meta[sid] = (str(r.tv_show_title), str(r.tv_show_category))
        except Exception:
            pass
    return meta

def main():
    meta = get_meta()
    inter = load_interactions('data/interactions.pkl')
    users = load_submission_users('data/submission.csv')
    truth = build_truth(inter, users, *VALIDATION_WEEKS)
    train = inter[inter.iso_week.between(*TRAIN_WEEKS) & (inter.frac >= 0.3)]
    # select users
    cnts = train.groupby('user_id').size()
    has_truth = set([int(u) for u in users if len(truth.get(int(u),set()))>0])
    cnts_t = cnts[cnts.index.isin(has_truth)].copy()
    def pick_percentile(c, p):
        s = c.sort_values()
        vals = s.to_numpy()
        k = int(round(p * (len(vals)-1)))
        target_val = vals[k]
        candidates = sorted([int(x) for x in c[c == target_val].index])
        return candidates[0], int(target_val)
    u10,v10 = pick_percentile(cnts_t, 0.1)
    u90,v90 = pick_percentile(cnts_t, 0.9)
    print('SELECTED')
    print(u10, v10)
    print(u90, v90)
    
    pool_val = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    train_qual_pool = train[train.tv_show_id.isin(pool_val)]
    fallback_order = popularity_order(train_qual_pool, pool_val)
    fallback = fill_ranking(fallback_order, fallback_order, K)
    
    results = {}
    for user_id in [u10, u90]:
        out = {}
        ut = train[train.user_id == user_id].groupby('tv_show_id').size().sort_values(ascending=False)
        top_train = []
        for sid,c in ut.head(5).items():
            t,categ = meta.get(int(sid), ('?','?'))
            top_train.append((int(sid), t, categ, int(c)))
        out['top_train'] = top_train
        out['truth'] = sorted(list(truth.get(user_id, set())))
        results[user_id] = out
    
    # popularity
    for uid in [u10,u90]: results[uid]['pop'] = fallback[:]
    # user history
    hist = {}
    for uid,g in train.groupby('user_id'):
        hist[int(uid)] = g.tv_show_id.value_counts().index.tolist()
    for uid in [u10,u90]: results[uid]['hist'] = fill_ranking(hist.get(uid,[]), fallback, K)
    # PPR
    matrix, user_ids, show_ids = build_graph(train)
    user_to_show, show_to_user = transition_blocks(matrix)
    uidx = pd.Index(user_ids)
    sidx = pd.Index(show_ids)
    target_idx_all = uidx.get_indexer([u10,u90])
    active = target_idx_all >= 0
    tuser_idx = target_idx_all[active]
    tusers = np.array([u10,u90])[active]
    pool_to_graph = sidx.get_indexer(pool_val)
    show_mass, iters, l1, me = ppr(user_to_show, show_to_user, tuser_idx, 0.7)
    pranks = rank_pool(show_mass, pool_val, pool_to_graph, fallback)
    pred_ppr = {u10:fallback, u90:fallback}
    for u,r in zip(tusers, pranks): pred_ppr[int(u)]=r
    for uid in [u10,u90]: results[uid]['ppr']=pred_ppr[uid]
    # MF
    ttrain = inter[inter.iso_week.between(*TRAIN_WEEKS) & (inter.frac>=0.5)]
    matrix_mf, uids_mf, sids_mf = build_matrix(ttrain)
    model = AlternatingLeastSquares(factors=32, regularization=0.1, alpha=5.0, iterations=15, use_cg=False, random_state=42, num_threads=1)
    model.fit(matrix_mf, show_progress=False)
    tidx = pd.Index(uids_mf).get_indexer([u10,u90])
    act = tidx >= 0
    tusers_m = np.array([u10,u90])[act]
    p2i = pd.Index(sids_mf).get_indexer(pool_val)
    pred_mf = {u10:fallback, u90:fallback}
    # compute scores for active
    if np.any(act):
        user_scores = model.user_factors[tidx[act]] @ model.item_factors.T
        # map to pool
        pool_scores = np.zeros((len(tusers_m), len(pool_val)), dtype=np.float32)
        pres = p2i >= 0
        pool_scores[:, pres] = user_scores[:, p2i[pres]]
        order = np.argsort(-pool_scores, axis=1, kind='stable')
        for i,u in enumerate(tusers_m):
            primary = []
            for pos in order[i]:
                if pool_scores[i,pos] > 0:
                    primary.append(int(pool_val[pos]))
                    if len(primary)==K: break
            pred_mf[int(u)]=fill_ranking(primary,fallback,K)
    for uid in [u10,u90]: results[uid]['mf']=pred_mf[uid]
    # CBF
    cbf_w = {"cat": 0.5, "genre": 1.0, "channel": 0.5, "tfidf": 3.0}
    cbf_scores, dims, _ = compute_cbf(inter, [u10,u90], pool_val, Path('data'), TRAIN_WEEKS, VALIDATION_WEEKS, cbf_w)
    cbf_ranks = rank_cbf(cbf_scores, pool_val, fallback)
    for i,uid in enumerate([u10,u90]): results[uid]['cbf']=cbf_ranks[i]
    # hybrid
    res_h = json.loads(Path('data/hybrid_results.json').read_text())
    base_h = res_h['best']
    comps, meta_h = build_components(inter, users, pool_val, Path('data'), TRAIN_WEEKS, VALIDATION_WEEKS, normalization='alternative', include_cbf=True)
    w_h = {'P': base_h['P'], 'M': base_h['M'], 'G': base_h['G'], 'A': base_h['A'], 'C': base_h['C'], 'K': 0}
    scores_h = score_components(comps, w_h)
    hranks = rank_hybrid_scores(scores_h, pool_val, meta_h['fallback'])
    uid_to_idx = {int(u):i for i,u in enumerate(users)}
    for uid in [u10,u90]: results[uid]['hybrid']=hranks[uid_to_idx[uid]]
    # test
    test_hybrid_file = Path('submissions/hybrid_wp2_wm0.5_wg1_wa2_wc2_val0.5797.csv')
    test_sub = pd.read_csv(test_hybrid_file)
    test_map = {}
    for _,r in test_sub.iterrows():
        test_map[int(r.user_id)] = [int(x) for x in str(r.tv_show_id).split()]
    for uid in [u10,u90]: results[uid]['hybrid_test']=test_map[uid]
    
    def ap_at5(actual_set, pred_list):
        if not actual_set: return 0.0
        pred = pred_list[:5]
        score = 0.0
        num_hits = 0.0
        for i,item in enumerate(pred):
            if item in actual_set and item not in pred[:i]:
                num_hits += 1.0
                score += num_hits/(i+1.0)
        return score/min(len(actual_set),5)
    
    out_lines = []
    out_lines.append('# Step 8: example users')
    out_lines.append('')
    out_lines.append('Selection rule (stated before looking at results): among submission users with non-empty validation truth, rank by number of qualifying training airings (weeks 11-20, frac >= 0.3); heavy viewer = user at the 90th percentile, light viewer = user at the 10th percentile (nearest rank, ties by smallest user_id).')
    out_lines.append('')
    out_lines.append(f'Light viewer (10th): user_id={u10}, qualifying training airings={v10}')
    out_lines.append(f'Heavy viewer (90th): user_id={u90}, qualifying training airings={v90}')
    out_lines.append('')
    
    for uid in [u10,u90]:
        res = results[uid]
        out_lines.append(f'## User {uid} (training airings={v10 if uid==u10 else v90})')
        out_lines.append('')
        out_lines.append('### Top 5 shows watched in training (weeks 11-20, frac>=0.3)')
        out_lines.append('| rank | tv_show_id | title | category | count |')
        out_lines.append('|---:|---:|---|---|---:|')
        for i,(sid,t,c,cc) in enumerate(res['top_train'],1):
            out_lines.append(f'| {i} | {sid} | {t} | {c} | {cc} |')
        out_lines.append('')
        out_lines.append(f'Validation truth (frac>=0.8, weeks 21-30): {len(res["truth"])} shows - {res["truth"]}')
        out_lines.append('')
        for mname in ['pop','hist','ppr','mf','cbf','hybrid']:
            pred = res[mname]
            ap = ap_at5(set(res['truth']), pred)
            utset = set([x[0] for x in res['top_train']]) | set(train[train.user_id==uid].tv_show_id.unique())
            out_lines.append(f'### Top-5 {mname} (AP@5={ap:.6f})')
            out_lines.append('| rank | tv_show_id | title | category | in_truth | watched_in_train |')
            out_lines.append('|---:|---:|---|---|---:|---:|')
            for i,sid in enumerate(pred,1):
                t,c = meta.get(int(sid),('?','?'))
                out_lines.append(f'| {i} | {sid} | {t} | {c} | {"Y" if sid in set(res["truth"]) else "N"} | {"Y" if sid in utset else "N"} |')
            out_lines.append('')
        out_lines.append(f'### Final hybrid Top-5 in TEST submission (hybrid_wp2_wm0.5_wg1_wa2_wc2_val0.5797.csv)')
        out_lines.append('| rank | tv_show_id | title | category |')
        out_lines.append('|---:|---:|---|---|')
        for i,sid in enumerate(res['hybrid_test'],1):
            t,c = meta.get(int(sid),('?','?'))
            out_lines.append(f'| {i} | {sid} | {t} | {c} |')
        out_lines.append('')
    
    lu = u10
    lres = results[lu]
    actual = set(lres['truth'])
    pred_h = lres['hybrid'][:5]
    out_lines.append('## Raw material to recompute AP@5 by hand (light viewer)')
    out_lines.append(f'Light viewer user_id: {lu}')
    out_lines.append(f'Truth set size (weeks 21-30, frac>=0.8): {len(actual)}')
    out_lines.append(f'Truth: {sorted(list(actual))}')
    out_lines.append(f'Hybrid Top-5 prediction: {pred_h}')
    out_lines.append('Hits in order:')
    for i,sid in enumerate(pred_h):
        hit = sid in actual
        out_lines.append(f'  pos {i+1} (id={sid}): hit={hit}')
    num_hits=0; score=0
    for i,sid in enumerate(pred_h):
        if sid in actual and sid not in pred_h[:i]:
            num_hits+=1; score += num_hits/(i+1.0)
    out_lines.append(f'AP@5 = {score} / min({len(actual)},5) = {score/min(len(actual),5):.6f}')
    
    Path('notes/step8_examples.md').write_text('\n'.join(out_lines) + '\n')
    print('done')

if __name__=='__main__':
    main()
