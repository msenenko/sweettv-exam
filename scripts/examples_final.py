#!/usr/bin/env python3
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from evaluate import *
from ppr import build_graph, transition_blocks, ppr, rank_pool
from implicit.als import AlternatingLeastSquares
from mf import build_matrix
from cbf import compute_cbf, rank_cbf
from hybrid import user_history_component, popularity_component, schedule_count_component, ppr_component, mf_component, load_schedule_details_with_channels, channel_affinity_component, popularity_order, fill_ranking, normalize_by_row_max, normalize_rows, normalize_vector
from hybrid import HISTORY_THRESHOLD, MF_THRESHOLD

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
    train = inter[inter.iso_week.between(*TRAIN_WEEKS) & (inter.frac >= HISTORY_THRESHOLD)]
    train_all = inter[inter.iso_week.between(*TRAIN_WEEKS)]
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
    print('SELECTED', u10,v10,u90,v90)
    
    # build final hybrid
    pool_val = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    train_qualifying = train
    train_mf = inter.loc[inter["iso_week"].between(*TRAIN_WEEKS) & (inter["frac"] >= MF_THRESHOLD)]
    uidx={int(u):i for i,u in enumerate(users)}; pidx={int(p):i for i,p in enumerate(pool_val)}
    raw_history_frac = np.zeros((len(users),len(pool_val)), dtype=np.float32)
    for (uid,sid), g in train_qualifying.groupby(['user_id','tv_show_id']):
        if int(uid) in uidx and int(sid) in pidx:
            raw_history_frac[uidx[int(uid)], pidx[int(sid)]] = np.log1p(float(g['frac'].sum()))
    airing_counts, show_channels, schedule_rows, show_ch_counts = load_schedule_details_with_channels(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    raw_popularity = popularity_component(train_qualifying, pool_val)
    raw_ppr, ppr_metadata = ppr_component(train_qualifying, users, pool_val)
    raw_mf, mf_metadata = mf_component(train_mf, users, pool_val)
    counts_ch = train_qualifying.groupby(["user_id", "channel_id"]).size().reset_index(name="n_airings")
    totals = train_qualifying.groupby("user_id").size().rename("n_total").reset_index()
    shares = counts_ch.merge(totals, on="user_id")
    shares["share"] = shares["n_airings"] / shares["n_total"]
    user_channel_shares = {int(uid): {int(r.channel_id): float(r.share) for _,r in g.iterrows()} for uid,g in shares.groupby("user_id")}
    raw_exposure = np.zeros((len(users), len(pool_val)), dtype=np.float32)
    for j, show_id in enumerate(pool_val):
        sid = int(show_id); chs = show_channels.get(sid, [])
        for ui, user_id in enumerate(users):
            ucs = user_channel_shares.get(int(user_id), {})
            s_exp = sum(ucs.get(c,0.0)*float(show_ch_counts.get((sid,c),0)) for c in chs)
            raw_exposure[ui,j] = np.log1p(10.0*s_exp)
    fallback_order = popularity_order(train_qualifying.loc[train_qualifying["tv_show_id"].isin(pool_val)], pool_val)
    fallback = fill_ranking(fallback_order, fallback_order, K)
    R = raw_history_frac.astype(np.float32); P = normalize_by_row_max(raw_ppr); M = normalize_rows(raw_mf); G = normalize_vector(raw_popularity); E = normalize_by_row_max(raw_exposure)
    w={'P':0,'M':0.5,'G':2,'A':0,'C':0,'E':3}
    n_u=len(users); n_p=len(pool_val); Gv=G.reshape(1,-1)
    scores = R.copy(); scores += w['M']*M + w['G']*Gv + w['E']*E
    idx = np.argpartition(scores,-5,axis=1)[:,-5:]
    hranks=[]
    for i in range(n_u):
        order=np.argsort(-scores[i,idx[i]],kind='stable'); pos=idx[i,order]
        primary=[]
        for ppos in pos:
            if scores[i,ppos]>0: primary.append(int(pool_val[ppos]))
            if len(primary)==5: break
        hranks.append(fill_ranking(primary,fallback,K))
    uid_to_idx = {int(u):i for i,u in enumerate(users)}
    results = {}
    for user_id in [u10,u90]:
        out = {}
        ut = train[train.user_id == user_id].groupby('tv_show_id').size().sort_values(ascending=False)
        top_train = []
        for sid,c in ut.head(5).items():
            t,categ = meta.get(int(sid), ('?','?'))
            top_train.append((int(sid), t, categ, int(c)))
        out['top_train']=top_train; out['truth']=sorted(list(truth.get(user_id,set())))
        results[user_id]=out
    for uid in [u10,u90]: results[uid]['hybrid']=hranks[uid_to_idx[uid]]
    # test
    test_map={}
    for _,r in pd.read_csv('submissions/hybrid_exp2_wp0_wm0.5_wg2_wa0_wc0_we3_rfrac_val0.5930.csv').iterrows():
        test_map[int(r.user_id)]=[int(x) for x in str(r.tv_show_id).split()]
    for uid in [u10,u90]: results[uid]['hybrid_test']=test_map[uid]
    
    def apk(a,p,k=5):
        if not a: return 0.0
        p=p[:k]; score=0.0; nh=0.0
        for i,item in enumerate(p):
            if item in a and item not in p[:i]: nh+=1.0; score+=nh/(i+1.0)
        return score/min(len(a),k)
    
    out_lines=[]
    out_lines.append('# Step 8 (updated for final config): example users')
    out_lines.append('')
    out_lines.append('Selection rule (stated before looking at results): among submission users with non-empty validation truth, rank by number of qualifying training airings (weeks 11-20, frac >= 0.3); heavy viewer = user at the 90th percentile, light viewer = user at the 10th percentile (nearest rank, ties by smallest user_id). Index formula used: k = int(round(p * (len(vals)-1))) over values sorted ascending, then among candidates with that value pick smallest user_id.')
    out_lines.append('')
    out_lines.append('Note on titles/categories: looked up from schedule CSVs (first occurrence per show_id); some airing rows have concatenated strings (e.g. 10002451 appears as "Великолепный векПсы под прикрытием" in some dumps) — title is taken from the stored schedule row used for lookup, not reconstructed.')
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
        out_lines.append(f'Validation truth (frac>=0.8, weeks 21-30): {len(res["truth"])} shows')
        out_lines.append('')
        pred = res['hybrid']
        ap = apk(set(res['truth']), pred)
        utset = set([x[0] for x in res['top_train']]) | set(train[train.user_id==uid].tv_show_id.unique())
        out_lines.append(f'### Final hybrid (R=sum frac, wM=0.5,wG=2,wE=3) Top-5 validation (AP@5={ap:.6f})')
        out_lines.append('| rank | tv_show_id | title | category | in_truth | watched_in_train |')
        out_lines.append('|---:|---:|---|---|---:|---:|')
        for i,sid in enumerate(pred,1):
            t,c = meta.get(int(sid),('?','?'))
            out_lines.append(f'| {i} | {sid} | {t} | {c} | {"Y" if sid in set(res["truth"]) else "N"} | {"Y" if sid in utset else "N"} |')
        out_lines.append('')
        out_lines.append(f'### Final hybrid Top-5 in TEST submission (hybrid_exp2_wp0_wm0.5_wg2_wa0_wc0_we3_rfrac_val0.5930.csv)')
        out_lines.append('| rank | tv_show_id | title | category |')
        out_lines.append('|---:|---:|---|---|')
        for i,sid in enumerate(res['hybrid_test'],1):
            t,c = meta.get(int(sid),('?','?'))
            out_lines.append(f'| {i} | {sid} | {t} | {c} |')
        out_lines.append('')
    Path('notes/step8_examples_final.md').write_text('\n'.join(out_lines)+'\n')
    print('done')

if __name__=='__main__':
    main()
