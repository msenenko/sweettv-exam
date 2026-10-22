#!/usr/bin/env python3
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from evaluate import *

def main():
    start = time.time()
    inter = load_interactions('data/interactions.pkl')
    users = load_submission_users('data/submission.csv')
    truth = build_truth(inter, users, *VALIDATION_WEEKS)
    pool = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    
    comp_path = Path('data/components_step11_val.npz')
    if comp_path.exists():
        data = np.load(comp_path, allow_pickle=True)
        R = data['R']; P = data['P']; M = data['M']; G = data['G']; A = data['A']; C = data['C']; E = data['E']
        fallback = data['fallback'].tolist()
    else:
        import importlib, sys as _sys
        # run hybrid_step11 logic inline
        from hybrid import user_history_component, popularity_component, schedule_count_component, ppr_component, mf_component, load_schedule_details_with_channels, channel_affinity_component, popularity_order, fill_ranking, normalize_by_row_max, normalize_rows, normalize_vector
        from hybrid import HISTORY_THRESHOLD, PPR_ALPHA, MF_THRESHOLD
        train_qualifying = inter.loc[inter["iso_week"].between(*TRAIN_WEEKS) & (inter["frac"] >= HISTORY_THRESHOLD)]
        train_mf = inter.loc[inter["iso_week"].between(*TRAIN_WEEKS) & (inter["frac"] >= MF_THRESHOLD)]
        airing_counts, show_channels, schedule_rows, show_ch_counts = load_schedule_details_with_channels(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
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
        np.savez_compressed(comp_path, R=R,P=P,M=M,G=G,A=A,C=C,E=E,fallback=np.array(fallback))
    print('loaded', time.time()-start, flush=True)
    
    n_u = len(users); n_p = len(pool)
    Gv = G.reshape(1,-1); Av = A.reshape(1,-1)
    def eval_config(w):
        scores = R.copy()
        scores += w['P']*P + w['M']*M + w['C']*C + w['E']*E
        scores += w['G']*Gv + w['A']*Av
        idx = np.argpartition(scores, -5, axis=1)[:, -5:]
        maps = []
        for i in range(n_u):
            order = np.argsort(-scores[i,idx[i]], kind='stable')
            pos = idx[i,order]
            primary = []
            for ppos in pos:
                if scores[i,ppos]>0:
                    primary.append(int(pool[ppos]))
                    if len(primary)==5: break
            ranking = list(primary); seen=set(ranking)
            for x in fallback:
                if x in seen: continue
                ranking.append(int(x)); seen.add(x)
                if len(ranking)==5: break
            actual = truth.get(int(users[i]), set())
            if not actual: maps.append(0.0); continue
            p = ranking[:5]; score=0.0; nh=0.0
            for i2,it in enumerate(p):
                if it in actual and it not in p[:i2]:
                    nh+=1.0; score+=nh/(i2+1.0)
            maps.append(score/min(len(actual),5))
        return np.mean(maps)
    
    WE = (0,1,2,3,4,6)
    WP = (0,1,2)
    WG = (0.5,1,2)
    WA = (0,1,2)
    WC = (0,1,2)
    WM = (0,0.5)
    configs = []
    for we in WE:
        for wp in WP:
            for wm in WM:
                for wg in WG:
                    for wa in WA:
                        for wc in WC:
                            w = {'P':wp,'M':wm,'G':wg,'A':wa,'C':wc,'E':we}
                            configs.append({**w,'map_all':float(eval_config(w))})
    best = max(configs, key=lambda x:x['map_all'])
    def simp(c):
        return (sum(c[k]!=0 for k in ['P','M','G','A','C','E']), sum(c[k] for k in ['P','M','G','A','C','E']), c['P'],c['M'],c['G'],c['A'],c['C'],c['E'])
    near = [c for c in configs if c['map_all']>=best['map_all']-0.002]
    chosen = min(near, key=simp)
    ranked = sorted(configs, key=lambda x:(-x['map_all'], simp(x)))
    res = {'best':best,'chosen':chosen,'top10':ranked[:10]}
    Path('data/hybrid_step11.json').write_text(json.dumps(res))
    print('best', best, flush=True)
    print('chosen', chosen, flush=True)
    
    lines = []
    lines.append('# Step 11: channel-exposure component (suggested by reviewing assistant)')
    lines.append('')
    lines.append('Suggested by reviewing assistant (Claude). Definition: E(u,i)=log1p(10 * sum_c share_u(c) * n_airings(i,c,target window)) where share_u(c)=user qualifying training airings on c / user total; schedule used only for airing info; E divided by user max E (alternative normalization). Score = R + wP*P + wM*M + wG*G + wA*A + wC*C + wE*E.')
    lines.append('')
    lines.append('## Prediction')
    lines.append('Quick check expectation ~0.587-0.588 for R + 1*P + 2*G + 4*E.')
    lines.append('')
    lines.append('## Top 10 (validation)')
    lines.append('| rank | wP | wM | wG | wA | wC | wE | MAP@5 all | delta to best |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for i,c in enumerate(ranked[:10],1):
        lines.append(f"| {i} | {c['P']} | {c['M']} | {c['G']} | {c['A']} | {c['C']} | {c['E']} | {c['map_all']:.6f} | {best['map_all']-c['map_all']:.6f} |")
    lines.append('')
    lines.append(f"Best: wP={best['P']}, wM={best['M']}, wG={best['G']}, wA={best['A']}, wC={best['C']}, wE={best['E']} MAP={best['map_all']:.6f}")
    lines.append(f"Simplicity pick: wP={chosen['P']}, wM={chosen['M']}, wG={chosen['G']}, wA={chosen['A']}, wC={chosen['C']}, wE={chosen['E']} MAP={chosen['map_all']:.6f}")
    lines.append('')
    lines.append('## Ablation of best')
    def apk2(a,p,k=5):
        if not a: return 0.0
        p=p[:k]; score=0.0; nh=0.0
        for i2,it in enumerate(p):
            if it in a and it not in p[:i2]: nh+=1.0; score+=nh/(i2+1.0)
        return score/min(len(a),k)
    for drop in ['R','P','M','G','A','C','E']:
        if drop=='R':
            scores = np.zeros((n_u,n_p),dtype=np.float32)
            scores += best['P']*P + best['M']*M + best['C']*C + best['E']*E
            scores += best['G']*Gv + best['A']*Av
        else:
            w_ab = dict(best); w_ab[drop]=0.0
            scores = R.copy()
            scores += w_ab['P']*P + w_ab['M']*M + w_ab['C']*C + w_ab['E']*E
            scores += w_ab['G']*Gv + w_ab['A']*Av
        idx = np.argpartition(scores,-5,axis=1)[:,-5:]
        rankings=[]
        for i in range(n_u):
            order=np.argsort(-scores[i,idx[i]],kind='stable'); pos=idx[i,order]
            primary=[]
            for ppos in pos:
                if scores[i,ppos]>0: primary.append(int(pool[ppos])); 
                if len(primary)==5: break
            ranking=list(primary); seen=set(ranking)
            for x in fallback:
                if x in seen: continue
                ranking.append(int(x)); seen.add(x)
                if len(ranking)==5: break
            rankings.append(ranking)
        m_ab = np.mean([apk2(truth.get(int(users[i]),set()),rankings[i],5) for i in range(n_u)])
        lines.append(f'- drop {drop}: MAP={m_ab:.6f}, delta={m_ab-best["map_all"]:.6f}')
    Path('notes/step11_hybrid_exposure.md').write_text('\n'.join(lines)+'\n')
    print('done', time.time()-start, flush=True)

if __name__=='__main__':
    main()
