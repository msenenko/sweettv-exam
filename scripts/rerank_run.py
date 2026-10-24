#!/usr/bin/env python3
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from evaluate import *
from hybrid import user_history_component, popularity_component, schedule_count_component, ppr_component, mf_component, load_schedule_details_with_channels, channel_affinity_component, popularity_order, fill_ranking, normalize_by_row_max, normalize_rows, normalize_vector
from hybrid import HISTORY_THRESHOLD, MF_THRESHOLD
from cbf import compute_cbf

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
np.random.seed(SEED)

def get_meta():
    meta = {}
    for fn in ['export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']:
        try:
            df = pd.read_csv(f'data/{fn}', usecols=['tv_show_id','tv_show_title','tv_show_category'])
            for _,r in df.iterrows():
                sid = int(r.tv_show_id)
                if sid not in meta: meta[sid] = str(r.tv_show_category)
        except Exception:
            pass
    cats = sorted(set(meta.values()))
    return meta, {c:i for i,c in enumerate(cats)}

def apk(a,p,k=5):
    if not a: return 0.0
    p=p[:k]; score=0.0; nh=0.0
    for i2,it in enumerate(p):
        if it in a and it not in p[:i2]: nh+=1.0; score+=nh/(i2+1.0)
    return score/min(len(a),k)

def main():
    t0 = time.time()
    print('PREDICT_GAIN_TARGET: >= +0.0030 (to >=0.5960) from hybrid 0.592977')
    Path('logs').mkdir(exist_ok=True)
    cache = Path('data/rerank_val_features.pkl')
    meta_cat, catmap = get_meta()
    inter = load_interactions('data/interactions.pkl')
    users = load_submission_users('data/submission.csv')
    truth = build_truth(inter, users, *VALIDATION_WEEKS)
    train_qual = inter[inter.iso_week.between(*TRAIN_WEEKS) & (inter.frac >= HISTORY_THRESHOLD)]
    train_mf = inter.loc[inter["iso_week"].between(*TRAIN_WEEKS) & (inter["frac"] >= MF_THRESHOLD)]
    train_all = inter[inter.iso_week.between(*TRAIN_WEEKS)]
    pool_val = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    if cache.exists():
        print('loading cache')
        df = pd.read_pickle(cache)
    else:
        uidx={int(u):i for i,u in enumerate(users)}; pidx={int(p):i for i,p in enumerate(pool_val)}
        raw_history_cnt = user_history_component(train_qual, users, pool_val)
        raw_history_frac = np.zeros((len(users),len(pool_val)), dtype=np.float32)
        for (uid,sid), g in train_qual.groupby(['user_id','tv_show_id']):
            if int(uid) in uidx and int(sid) in pidx:
                raw_history_frac[uidx[int(uid)], pidx[int(sid)]] = np.log1p(float(g['frac'].sum()))
        airing_counts, show_channels, schedule_rows, show_ch_counts = load_schedule_details_with_channels(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
        raw_popularity = popularity_component(train_qual, pool_val)
        raw_schedule = schedule_count_component(airing_counts, pool_val)
        raw_ppr, _ = ppr_component(train_qual, users, pool_val)
        raw_mf, _ = mf_component(train_mf, users, pool_val)
        counts_ch = train_qual.groupby(["user_id", "channel_id"]).size().reset_index(name="n_airings")
        totals = train_qual.groupby("user_id").size().rename("n_total").reset_index()
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
        cbf_scores, cbf_dims, cbf_item_df = compute_cbf(inter, users, pool_val, Path('data'), TRAIN_WEEKS, VALIDATION_WEEKS, {"cat":1.0,"genre":1.0,"channel":1.0,"tfidf":2.0})
        R = raw_history_frac.astype(np.float32)
        P = normalize_by_row_max(raw_ppr); M = normalize_rows(raw_mf); G = normalize_vector(raw_popularity); A = normalize_vector(raw_schedule); E = normalize_by_row_max(raw_exposure); C = cbf_scores.astype(np.float32)
        w={'P':0,'M':0.5,'G':2,'A':0,'C':0,'E':3}
        Gv=G.reshape(1,-1)
        hybrid_scores = R.copy(); hybrid_scores += w['M']*M + w['G']*Gv + w['E']*E
        user_qual_cnt = train_qual.groupby('user_id').size().to_dict()
        user_dist_shows = train_qual.groupby('user_id')['tv_show_id'].nunique().to_dict()
        cand_rows=[]
        for i,u in enumerate(users):
            u = int(u)
            hs = hybrid_scores[i]
            top50 = np.argpartition(hs, -50)[-50:]
            top50 = top50[np.argsort(-hs[top50])]
            cand = set([int(pool_val[x]) for x in top50])
            uh = train_qual[train_qual.user_id==u]['tv_show_id'].unique()
            for sid in uh:
                sid=int(sid)
                if sid in pidx: cand.add(sid)
            truth_set = truth.get(u,set())
            Ru = raw_history_cnt[i,:]; Eu = raw_exposure[i,:]
            Rmax = float(Ru.max()) if Ru.max()>0 else 1.0
            Emax = float(Eu.max()) if Eu.max()>0 else 1.0
            for sid in cand:
                j = pidx[sid]
                hrows = train_all[train_all.user_id==u]
                hsub = hrows[hrows.tv_show_id==sid]
                if len(hsub)==0:
                    weeks_since = 999.0; weeks_cnt = 0.0
                else:
                    wks = sorted(hsub.iso_week.unique())
                    weeks_since = float(TRAIN_WEEKS[1] - max(wks))
                    weeks_cnt = float(len(wks))
                Rcntv = float(raw_history_cnt[i,j]); Rfv = float(R[i,j])
                Pv = float(P[i,j]); Mv = float(M[i,j]); Gv2 = float(G[j]); Av2 = float(A[j]); Cv = float(C[i,j]); Ev = float(E[i,j]); HFv = float(hybrid_scores[i,j])
                hrank = float((np.argsort(-hs).tolist().index(j))+1)
                ucs = user_channel_shares.get(u,{})
                topch = max(ucs.items(), key=lambda x:x[1])[0] if ucs else -1
                sh_air_target = float(show_ch_counts.get((sid, topch),0)) if topch>=0 else 0.0
                cand_rows.append({
                    'user_id':u,'sid':int(sid),
                    'Rcnt':Rcntv,'Rfrac':Rfv,'M':Mv,'P':Pv,'G':Gv2,'A':Av2,'C':Cv,'E':Ev,'HF':HFv,'Hrank':hrank,
                    'weeks_since':weeks_since,'weeks_cnt':weeks_cnt,
                    'u_qual_cnt':float(user_qual_cnt.get(u,0)),'u_dist':float(user_dist_shows.get(u,0)),
                    'cat':float(catmap.get(meta_cat.get(sid,''),-1)),
                    'target_air':float(airing_counts.get(sid,0)),
                    'share_topch_target':float(ucs.get(topch,0.0))*sh_air_target,
                    'Rcnt_maxn':Rcntv/Rmax,'E_maxn':Ev/Emax,
                    'label':1.0 if int(sid) in truth_set else 0.0
                })
        df = pd.DataFrame(cand_rows)
        df.to_pickle(cache)
        print('cached')
    feat_cols = ['Rcnt','Rfrac','M','P','G','A','C','E','HF','Hrank','weeks_since','weeks_cnt','u_qual_cnt','u_dist','cat','target_air','share_topch_target','Rcnt_maxn','E_maxn']
    print('shape', df.shape, 'positives', int(df.label.sum()), 'elapsed_prep', time.time()-t0)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(df))
    fold_maps=[]
    for fold,(tr,te) in enumerate(gkf.split(df[feat_cols], df['label'], groups=df['user_id'])):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=7, random_state=SEED+fold)
        clf.fit(df.iloc[tr][feat_cols], df.iloc[tr]['label'])
        oof_te = clf.predict_proba(df.iloc[te][feat_cols])[:,1]
        oof[te] = oof_te
        # map fold
        te_idx = te
        # group by user
        maps=[]
        # iterate by original groups
        for uid in set(df.iloc[te_idx]['user_id']):
            mask = df['user_id']==uid
            # get positions in df that are in te_idx
            g2 = df[mask].copy()
            g2['oofv'] = oof[mask.values]
            g2 = g2.sort_values('oofv',ascending=False)
            pred = list(g2.sid.values)[:5]
            a = truth.get(int(uid),set())
            if not a: maps.append(0.0)
            else: maps.append(apk(a,pred,5))
        fm = np.mean(maps)
        fold_maps.append(fm)
        print('fold', fold, fm)
    maps_all=[]
    for uid in set(df['user_id']):
        g2 = df[df['user_id']==uid].copy()
        g2['oofv']=oof[df['user_id']==uid]
        g2=g2.sort_values('oofv',ascending=False)
        pred=list(g2.sid.values)[:5]
        a=truth.get(int(uid),set())
        if not a: maps_all.append(0.0)
        else: maps_all.append(apk(a,pred,5))
    print('OOF_MAP_ALL', np.mean(maps_all), 'HYBRID_BASELINE', 0.592977, 'gain', np.mean(maps_all)-0.592977)
    clf_imp = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=7, random_state=SEED)
    clf_imp.fit(df[feat_cols], df['label'])
    imp = pd.Series(clf_imp.feature_importances_ if hasattr(clf_imp,'feature_importances_') else np.zeros(len(feat_cols)), index=feat_cols).sort_values(ascending=False)
    imp.to_csv('logs/rerank_feat_importance.csv')
    print(imp.head(10))
    print('elapsed_total', time.time()-t0)

if __name__=='__main__':
    main()
