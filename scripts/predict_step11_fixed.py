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
    inter = load_interactions('data/interactions.pkl')
    users = load_submission_users('data/submission.csv')
    pool = load_schedule_pool(Path('data'), TEST_WEEKS[0], TEST_WEEKS[1])
    train_weeks = (11,30)
    target_weeks = TEST_WEEKS
    train_qualifying = inter.loc[inter["iso_week"].between(*train_weeks) & (inter["frac"] >= HISTORY_THRESHOLD)]
    train_mf = inter.loc[inter["iso_week"].between(*train_weeks) & (inter["frac"] >= MF_THRESHOLD)]
    airing_counts, show_channels, schedule_rows, show_ch_counts = load_schedule_details_with_channels(Path('data'), target_weeks[0], target_weeks[1])
    raw_history = user_history_component(train_qualifying, users, pool)
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
    R = raw_history.astype(np.float32); P = normalize_by_row_max(raw_ppr); M = normalize_rows(raw_mf); G = normalize_vector(raw_popularity); A = normalize_vector(raw_schedule); C = raw_channel_alt; E = normalize_by_row_max(raw_exposure)
    res = json.loads(Path('data/hybrid_step11.json').read_text())
    best = res['best']
    w = {'P':best['P'],'M':best['M'],'G':best['G'],'A':best['A'],'C':best['C'],'E':best['E']}
    n_u=len(users); n_p=len(pool)
    Gv=G.reshape(1,-1); Av=A.reshape(1,-1)
    scores = R.copy()
    scores += w['P']*P + w['M']*M + w['C']*C + w['E']*E
    scores += w['G']*Gv + w['A']*Av
    idx = np.argpartition(scores,-5,axis=1)[:,-5:]
    rankings=[]
    for i in range(n_u):
        order=np.argsort(-scores[i,idx[i]],kind='stable'); pos=idx[i,order]
        primary=[]
        for ppos in pos:
            if scores[i,ppos]>0: primary.append(int(pool[ppos]))
            if len(primary)==5: break
        rankings.append(fill_ranking(primary,fallback,K))
    out_path = Path('submissions/hybrid_exp_wp0_wm0.5_wg2_wa0_wc0_we4_val0.5908.csv')
    lines_out=['user_id,tv_show_id']
    for i,u in enumerate(users):
        lines_out.append(f'{int(u)},{ " ".join(str(x) for x in rankings[i]) }')
    out_path.write_text('\n'.join(lines_out)+'\n')
    print('wrote', out_path)

if __name__=='__main__':
    main()
