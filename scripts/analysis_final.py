#!/usr/bin/env python3
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from evaluate import *
from hybrid import user_history_component, popularity_component, schedule_count_component, ppr_component, mf_component, load_schedule_details_with_channels, channel_affinity_component, popularity_order, fill_ranking, normalize_by_row_max, normalize_rows, normalize_vector
from hybrid import HISTORY_THRESHOLD, MF_THRESHOLD

def get_meta():
    meta = {}
    for fn in ['export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']:
        try:
            df = pd.read_csv(f'data/{fn}', usecols=['tv_show_id','tv_show_title','tv_show_category'])
            for _,r in df.iterrows():
                sid = int(r.tv_show_id)
                if sid not in meta: meta[sid] = (str(r.tv_show_title), str(r.tv_show_category))
        except: pass
    return meta

def main():
    meta = get_meta()
    inter = load_interactions('data/interactions.pkl')
    users = load_submission_users('data/submission.csv')
    truth = build_truth(inter, users, *VALIDATION_WEEKS)
    train = inter[inter.iso_week.between(*TRAIN_WEEKS) & (inter.frac >= HISTORY_THRESHOLD)]
    train_all = inter[inter.iso_week.between(*TRAIN_WEEKS)]
    pool_val = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    train_mf = inter.loc[inter["iso_week"].between(*TRAIN_WEEKS) & (inter["frac"] >= MF_THRESHOLD)]
    uidx={int(u):i for i,u in enumerate(users)}; pidx={int(p):i for i,p in enumerate(pool_val)}
    raw_history_frac = np.zeros((len(users),len(pool_val)), dtype=np.float32)
    for (uid,sid), g in train.groupby(['user_id','tv_show_id']):
        if int(uid) in uidx and int(sid) in pidx:
            raw_history_frac[uidx[int(uid)], pidx[int(sid)]] = np.log1p(float(g['frac'].sum()))
    airing_counts, show_channels, schedule_rows, show_ch_counts = load_schedule_details_with_channels(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    raw_popularity = popularity_component(train, pool_val)
    raw_ppr, ppr_metadata = ppr_component(train, users, pool_val)
    raw_mf, mf_metadata = mf_component(train_mf, users, pool_val)
    counts_ch = train.groupby(["user_id", "channel_id"]).size().reset_index(name="n_airings")
    totals = train.groupby("user_id").size().rename("n_total").reset_index()
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
    fallback_order = popularity_order(train.loc[train["tv_show_id"].isin(pool_val)], pool_val)
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
    
    def apk(a,p,k=5):
        if not a: return 0.0
        p=p[:k]; score=0.0; nh=0.0
        for i2,it in enumerate(p):
            if it in a and it not in p[:i2]: nh+=1.0; score+=nh/(i2+1.0)
        return score/min(len(a),k)
    hist={}
    for uid,g in train_all.groupby('user_id'): hist[int(uid)]=set(g.tv_show_id.unique())
    train_pop = train_all.groupby('tv_show_id')['user_id'].nunique().to_dict()
    total_slots=n_u*5; familiar=0; familiar_hits=0; new_hits=0; familiar_count=0; new_count=0; all_recs=[]; cats=[]
    for i,u in enumerate(users):
        hu=hist.get(int(u),set())
        for sid in hranks[i]:
            sid=int(sid); all_recs.append(sid); cats.append(meta.get(sid,('?','?'))[1])
            if sid in hu:
                familiar+=1; familiar_count+=1
                if sid in truth.get(int(u),set()): familiar_hits+=1
            else:
                new_count+=1
                if sid in truth.get(int(u),set()): new_hits+=1
    from collections import Counter
    cat_mix = dict(Counter(cats).most_common())
    map_all = np.mean([apk(truth.get(int(users[i]),set()),hranks[i],5) for i in range(n_u)])
    lines=[]
    lines.append('### Final config (R=sum frac, wM=0.5,wG=2,wE=3)')
    lines.append(f'- MAP@5: {map_all:.6f}')
    lines.append(f'- familiar share: {familiar/total_slots:.4f}, new share: {(total_slots-familiar)/total_slots:.4f}')
    lines.append(f'- hit rate familiar: {familiar_hits/familiar_count:.4f}, hit rate new: {new_hits/new_count:.4f}')
    lines.append(f'- distinct ids: {len(set(all_recs))}')
    lines.append(f'- mean train pop: {np.mean([train_pop.get(s,0) for s in all_recs]):.2f}')
    lines.append(f'- cat mix: {cat_mix}')
    lines.append('')
    lines.append('Ablation:')
    for drop in ['R','M','G','E']:
        if drop=='R':
            scores_ab = np.zeros((n_u,n_p),dtype=np.float32)
            scores_ab += w['M']*M + w['G']*Gv + w['E']*E
        elif drop=='M':
            scores_ab = R.copy(); scores_ab += w['G']*Gv + w['E']*E
        elif drop=='G':
            scores_ab = R.copy(); scores_ab += w['M']*M + w['E']*E
        elif drop=='E':
            scores_ab = R.copy(); scores_ab += w['M']*M + w['G']*Gv
        idx_ab = np.argpartition(scores_ab,-5,axis=1)[:,-5:]
        ranks_ab=[]
        for i in range(n_u):
            order=np.argsort(-scores_ab[i,idx_ab[i]],kind='stable'); pos=idx_ab[i,order]
            primary=[]
            for ppos in pos:
                if scores_ab[i,ppos]>0: primary.append(int(pool_val[ppos]))
                if len(primary)==5: break
            ranks_ab.append(fill_ranking(primary,fallback,K))
        m_ab = np.mean([apk(truth.get(int(users[i]),set()),ranks_ab[i],5) for i in range(n_u)])
        lines.append(f'- drop {drop}: MAP={m_ab:.6f}, delta={m_ab-map_all:.6f}')
    Path('notes/step10_analysis_final.md').write_text('\n'.join(lines)+'\n')
    print('done')

if __name__=='__main__':
    main()
