#!/usr/bin/env python3
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from evaluate import *
from hybrid import user_history_component, popularity_component, schedule_count_component, ppr_component, mf_component, load_schedule_details_with_channels, channel_affinity_component, popularity_order, fill_ranking, normalize_by_row_max, normalize_rows, normalize_vector
from hybrid import HISTORY_THRESHOLD, MF_THRESHOLD

def main():
    start = time.time()
    inter = load_interactions('data/interactions.pkl')
    users = load_submission_users('data/submission.csv')
    truth = build_truth(inter, users, *VALIDATION_WEEKS)
    pool = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    train_qualifying = inter.loc[inter["iso_week"].between(*TRAIN_WEEKS) & (inter["frac"] >= HISTORY_THRESHOLD)]
    train_mf = inter.loc[inter["iso_week"].between(*TRAIN_WEEKS) & (inter["frac"] >= MF_THRESHOLD)]
    airing_counts, show_channels, schedule_rows, show_ch_counts = load_schedule_details_with_channels(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    # R definitions
    # count
    raw_history_cnt = user_history_component(train_qualifying, users, pool)
    # sum of frac
    pairs = train_qualifying.groupby(['user_id','tv_show_id'])['frac'].sum().reset_index()
    uidx = {int(u):i for i,u in enumerate(users)}
    pidx = {int(p):i for i,p in enumerate(pool)}
    raw_history_frac = np.zeros((len(users),len(pool)), dtype=np.float32)
    for _,r in pairs.iterrows():
        if int(r.user_id) in uidx and int(r.tv_show_id) in pidx:
            raw_history_frac[uidx[int(r.user_id)], pidx[int(r.tv_show_id)]] = np.log1p(float(r.frac))
    raw_popularity = popularity_component(train_qualifying, users, pool) if False else None
    # easier use same
    raw_popularity = popularity_component(train_qualifying, pool)
    raw_schedule = schedule_count_component(airing_counts, pool)
    raw_ppr, ppr_metadata = ppr_component(train_qualifying, users, pool)
    raw_mf, mf_metadata = mf_component(train_mf, users, pool)
    counts_ch = train_qualifying.groupby(["user_id", "channel_id"]).size().reset_index(name="n_airings")
    totals = train_qualifying.groupby("user_id").size().rename("n_total").reset_index()
    shares = counts_ch.merge(totals, on="user_id")
    shares["share"] = shares["n_airings"] / shares["n_total"]
    user_channel_shares = {int(uid): {int(r.channel_id): float(r.share) for _,r in g.iterrows()} for uid,g in shares.groupby("user_id")}
    raw_exposure = np.zeros((len(users), len(pool)), dtype=np.float32)
    for j, show_id in enumerate(pool):
        sid = int(show_id); chs = show_channels.get(sid, [])
        for ui, user_id in enumerate(users):
            ucs = user_channel_shares.get(int(user_id), {})
            s_exp = sum(ucs.get(c,0.0)*float(show_ch_counts.get((sid,c),0)) for c in chs)
            raw_exposure[ui,j] = np.log1p(10.0*s_exp)
    fallback_order = popularity_order(train_qualifying.loc[train_qualifying["tv_show_id"].isin(pool)], pool)
    fallback = fill_ranking(fallback_order, fallback_order, K)
    raw_channel_alt = channel_affinity_component(train_qualifying, users, pool, show_channels).astype(np.float32)
    P = normalize_by_row_max(raw_ppr); M = normalize_rows(raw_mf); G = normalize_vector(raw_popularity); A = normalize_vector(raw_schedule); C = raw_channel_alt; E = normalize_by_row_max(raw_exposure)
    R_cnt = raw_history_cnt.astype(np.float32); R_frac = raw_history_frac.astype(np.float32)
    n_u=len(users); n_p=len(pool); Gv=G.reshape(1,-1); Av=A.reshape(1,-1)
    
    def eval_set(Rm):
        configs=[]
        for wm in [0,0.5]:
            for wg in [1,2,3]:
                for we in [3,4,5]:
                    w={'P':0,'M':wm,'G':wg,'A':0,'C':0,'E':we}
                    scores = Rm.copy()
                    scores += w['M']*M + w['G']*Gv + w['E']*E
                    idx = np.argpartition(scores,-5,axis=1)[:,-5:]
                    maps=[]
                    for i in range(n_u):
                        order=np.argsort(-scores[i,idx[i]],kind='stable'); pos=idx[i,order]
                        primary=[]
                        for ppos in pos:
                            if scores[i,ppos]>0: primary.append(int(pool[ppos]))
                            if len(primary)==5: break
                        ranking=fill_ranking(primary,fallback,K)
                        actual = truth.get(int(users[i]),set())
                        if not actual: maps.append(0.0); continue
                        p=ranking[:5]; score=0.0; nh=0.0
                        for i2,it in enumerate(p):
                            if it in actual and it not in p[:i2]: nh+=1.0; score+=nh/(i2+1.0)
                        maps.append(score/min(len(actual),5))
                    configs.append({**w,'map_all':float(np.mean(maps)),'R':'count' if np.allclose(Rm,R_cnt) else 'sumfrac'})
        return configs
    
    c1 = eval_set(R_cnt)
    c2 = eval_set(R_frac)
    allc = c1+c2
    best = max(allc, key=lambda x:x['map_all'])
    print('best', best)
    res = {'count_best': max(c1,key=lambda x:x['map_all']), 'frac_best': max(c2,key=lambda x:x['map_all']), 'best_overall': best}
    Path('data/hybrid_step12.json').write_text(json.dumps(res))
    
    lines=[]
    lines.append('# Step 12: R as log1p(sum of frac)')
    lines.append('')
    lines.append('Suggested by reviewing assistant (Claude). R = log1p(sum of frac over qualifying training airings of the show) weights fully-watched episodes more.')
    lines.append('')
    lines.append('## Small grid')
    lines.append('| R def | wM | wG | wE | MAP@5 all |')
    lines.append('|---|---:|---:|---:|---:|')
    for c in sorted(allc, key=lambda x:(-x['map_all']))[:15]:
        lines.append(f"| {c['R']} | {c['M']} | {c['G']} | {c['E']} | {c['map_all']:.6f} |")
    lines.append('')
    lines.append(f"Best overall: {best['R']} wM={best['M']} wG={best['G']} wE={best['E']} MAP={best['map_all']:.6f}")
    lines.append(f"Best with count R: {res['count_best']['map_all']:.6f}; best with sum-of-frac R: {res['frac_best']['map_all']:.6f}")
    lines.append(f"Improvement: {best['map_all']-0.590785:.6f}")
    lines.append('')
    Path('notes/step12_r_sumfrac.md').write_text('\n'.join(lines)+'\n')
    print('done', time.time()-start)

if __name__=='__main__':
    main()
