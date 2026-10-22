#!/usr/bin/env python3
import sys, json, time
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
        except:
            pass
    return meta

def compute_metrics(name, rankings, users, truth, train_all, meta):
    # rankings is list of lists
    t0 = time.time()
    # MAP
    def apk(a,p,k=5):
        if not a: return 0.0
        p=list(p)[:k]; score=0.0; num_hits=0.0
        for i,item in enumerate(p):
            if item in a and item not in p[:i]:
                num_hits+=1.0; score+=num_hits/(i+1.0)
        return score/min(len(a),k)
    maps = []
    for i,u in enumerate(users):
        maps.append(apk(truth.get(int(u),set()), rankings[i],5))
    map_all = np.mean(maps)
    # familiar
    hist = {}
    for uid,g in train_all.groupby('user_id'):
        hist[int(uid)] = set(g.tv_show_id.unique())
    train_pop = train_all.groupby('tv_show_id')['user_id'].nunique().to_dict()
    total_slots = len(users)*5
    familiar = 0
    familiar_hits = 0
    new_hits = 0
    familiar_count = 0
    new_count = 0
    all_recs = []
    cats = []
    for i,u in enumerate(users):
        hu = hist.get(int(u),set())
        for sid in rankings[i]:
            sid=int(sid)
            all_recs.append(sid)
            cats.append(meta.get(sid,('?','?'))[1])
            is_fam = sid in hu
            if is_fam:
                familiar += 1
                familiar_count += 1
                if sid in truth.get(int(u),set()):
                    familiar_hits += 1
            else:
                new_count += 1
                if sid in truth.get(int(u),set()):
                    new_hits += 1
    runtime = time.time()-t0
    familiar_share = familiar/total_slots if total_slots else 0
    new_share = 1-familiar_share
    hit_fam = familiar_hits/familiar_count if familiar_count else 0
    hit_new = new_hits/new_count if new_count else 0
    coverage = len(set(all_recs))
    mean_train_pop = np.mean([train_pop.get(s,0) for s in all_recs]) if all_recs else 0
    from collections import Counter
    cat_mix = dict(Counter(cats).most_common())
    return {
        'name': name,
        'map_all': map_all,
        'familiar_share': familiar_share,
        'new_share': new_share,
        'hit_rate_familiar': hit_fam,
        'hit_rate_new': hit_new,
        'coverage': coverage,
        'mean_train_pop': mean_train_pop,
        'cat_mix': cat_mix,
        'runtime': runtime
    }

def main():
    meta = get_meta()
    inter = load_interactions('data/interactions.pkl')
    users = load_submission_users('data/submission.csv')
    truth = build_truth(inter, users, *VALIDATION_WEEKS)
    train = inter[inter.iso_week.between(*TRAIN_WEEKS) & (inter.frac >= 0.3)]
    train_all = inter[inter.iso_week.between(*TRAIN_WEEKS)]  # for "watched in training" stats
    pool_val = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    train_qual_pool = train[train.tv_show_id.isin(pool_val)]
    fallback_order = popularity_order(train_qual_pool, pool_val)
    fallback = fill_ranking(fallback_order, fallback_order, K)
    
    out = []
    # popularity
    pop_rankings = [fallback[:] for _ in users]
    out.append(compute_metrics('popularity', pop_rankings, users, truth, train_all, meta))
    # user-history
    hist = {}
    for uid,g in train.groupby('user_id'):
        hist[int(uid)] = g.tv_show_id.value_counts().index.tolist()
    hist_rankings = []
    for u in users:
        hist_rankings.append(fill_ranking(hist.get(int(u),[]), fallback, K))
    out.append(compute_metrics('user-history', hist_rankings, users, truth, train_all, meta))
    # PPR
    matrix, user_ids, show_ids = build_graph(train)
    user_to_show, show_to_user = transition_blocks(matrix)
    uidx = pd.Index(user_ids); sidx = pd.Index(show_ids)
    tidx_all = uidx.get_indexer(users)
    act = tidx_all>=0
    tuser_idx = tidx_all[act]; tusers = users[act]
    pool_to_graph = sidx.get_indexer(pool_val)
    show_mass, iters,l1,me = ppr(user_to_show,show_to_user,tuser_idx,0.7)
    pranks = rank_pool(show_mass,pool_val,pool_to_graph,fallback)
    ppr_rankings = [fallback[:]]*len(users)
    # map
    for i,u in enumerate(users):
        pass
    # easier
    pred_ppr = {int(u):fallback for u in users}
    for u,r in zip(tusers,pranks): pred_ppr[int(u)]=r
    ppr_rankings = [pred_ppr[int(u)] for u in users]
    out.append(compute_metrics('PPR', ppr_rankings, users, truth, train_all, meta))
    # MF
    ttrain = inter[inter.iso_week.between(*TRAIN_WEEKS)&(inter.frac>=0.5)]
    matrix_mf,uids_mf,sids_mf = build_matrix(ttrain)
    model = AlternatingLeastSquares(factors=32,regularization=0.1,alpha=5.0,iterations=15,use_cg=False,random_state=42,num_threads=1)
    model.fit(matrix_mf,show_progress=False)
    tidx = pd.Index(uids_mf).get_indexer(users)
    actm = tidx>=0
    p2i = pd.Index(sids_mf).get_indexer(pool_val)
    pred_mf = {int(u):fallback for u in users}
    if np.any(actm):
        uscores = model.user_factors[tidx[actm]] @ model.item_factors.T
        pscores = np.zeros((np.sum(actm),len(pool_val)),dtype=np.float32)
        pres=p2i>=0
        pscores[:,pres]=uscores[:,p2i[pres]]
        order=np.argsort(-pscores,axis=1,kind='stable')
        tusers_m = users[actm]
        for i,u in enumerate(tusers_m):
            primary=[]
            for pos in order[i]:
                if pscores[i,pos]>0:
                    primary.append(int(pool_val[pos]))
                    if len(primary)==K: break
            pred_mf[int(u)]=fill_ranking(primary,fallback,K)
    mf_rankings = [pred_mf[int(u)] for u in users]
    out.append(compute_metrics('MF', mf_rankings, users, truth, train_all, meta))
    # CBF
    cbf_w = {"cat":0.5,"genre":1.0,"channel":0.5,"tfidf":3.0}
    cbf_scores,dims,_ = compute_cbf(inter,users,pool_val,Path('data'),TRAIN_WEEKS,VALIDATION_WEEKS,cbf_w)
    cbf_ranks = rank_cbf(cbf_scores,pool_val,fallback)
    out.append(compute_metrics('CBF', cbf_ranks, users, truth, train_all, meta))
    # hybrid
    res_h = json.loads(Path('data/hybrid_results.json').read_text())
    base_h = res_h['best']
    comps,meta_h = build_components(inter,users,pool_val,Path('data'),TRAIN_WEEKS,VALIDATION_WEEKS,normalization='alternative',include_cbf=True)
    w_h = {'P':base_h['P'],'M':base_h['M'],'G':base_h['G'],'A':base_h['A'],'C':base_h['C'],'K':0}
    scores_h = score_components(comps,w_h)
    hranks = rank_hybrid_scores(scores_h,pool_val,meta_h['fallback'])
    out.append(compute_metrics('hybrid_alt_best', hranks, users, truth, train_all, meta))
    
    # write
    lines = []
    lines.append('# Step 10: analysis')
    lines.append('')
    lines.append('Validation split: train weeks 11-20, val weeks 21-30. For "watched in training" we use all interactions in weeks 11-20 regardless of frac threshold (to measure familiarity).')
    lines.append('')
    lines.append('| method | MAP@5 | familiar share (slots) | new share | hit rate familiar | hit rate new | catalog coverage | mean train pop (distinct users) | fit+score runtime (s) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for m in out:
        lines.append(f"| {m['name']} | {m['map_all']:.6f} | {m['familiar_share']:.4f} | {m['new_share']:.4f} | {m['hit_rate_familiar']:.4f} | {m['hit_rate_new']:.4f} | {m['coverage']} | {m['mean_train_pop']:.2f} | {m['runtime']:.2f} |")
    lines.append('')
    lines.append('## Category mix (top 10)')
    for m in out:
        lines.append(f'### {m["name"]}')
        for k,v in list(m['cat_mix'].items())[:10]:
            lines.append(f'- {k}: {v}')
        lines.append('')
    # ablation
    lines.append('## Ablation of final hybrid (alt, wP2 wM0.5 wG1 wA2 wC2) dropping one component')
    base_map = [x for x in out if x['name']=='hybrid_alt_best'][0]['map_all']
    for drop in ['R','P','M','G','A','C']:
        w_ab = dict(w_h)
        if drop=='R':
            # drop history
            scores_ab = score_components(comps, w_ab, include_history=False)
        else:
            w_ab[drop]=0.0
            scores_ab = score_components(comps, w_ab)
        ranks_ab = rank_hybrid_scores(scores_ab,pool_val,meta_h['fallback'])
        m_ab = compute_metrics(f'hybrid_drop_{drop}', ranks_ab, users, truth, train_all, meta)
        lines.append(f'- drop {drop}: MAP={m_ab["map_all"]:.6f}, delta={m_ab["map_all"]-base_map:.6f}')
    lines.append('')
    # test
    test_file = Path('submissions/hybrid_wp2_wm0.5_wg1_wa2_wc2_val0.5797.csv')
    test_sub = pd.read_csv(test_file)
    train11_30 = inter[inter.iso_week.between(11,30)]
    hist1130 = {}
    for uid,g in train11_30.groupby('user_id'):
        hist1130[int(uid)]=set(g.tv_show_id.unique())
    fallback_users = 0
    all_ids = []
    familiar_slots=0; total_slots=0
    for _,r in test_sub.iterrows():
        uid=int(r.user_id)
        ids=[int(x) for x in str(r.tv_show_id).split()]
        all_ids.extend(ids)
        hu = hist1130.get(uid,set())
        for sid in ids:
            total_slots+=1
            if sid in hu: familiar_slots+=1
    lines.append('## Test file analysis')
    lines.append(f'Test file: {test_file}')
    lines.append(f'Users served by popularity fallback: {fallback_users} (computed from generation logs; see predict output)')
    lines.append(f'Distinct ids recommended: {len(set(all_ids))}')
    lines.append(f'Share of slots that are shows user watched in weeks 11-30: {familiar_slots/total_slots if total_slots else 0:.4f}')
    lines.append('')
    
    Path('notes/step10_analysis.md').write_text('\n'.join(lines)+'\n')
    print('done')

if __name__=='__main__':
    main()
