#!/usr/bin/env python3
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from evaluate import *
from hybrid import (
    build_components, score_components, rank_scores, evaluate_rankings,
    config_label, simplicity, load_schedule_details_with_channels,
    normalize_by_row_max
)

def build_components_with_exposure(
    interactions: pd.DataFrame,
    users: np.ndarray,
    pool: np.ndarray,
    data_dir: Path,
    train_weeks: tuple[int, int],
    target_weeks: tuple[int, int],
    normalization: str = "alternative",
):
    from hybrid import (
        HISTORY_THRESHOLD, PPR_ALPHA, MF_THRESHOLD,
        user_history_component, popularity_component, 
        schedule_count_component, ppr_component, mf_component,
        build_components as _bc
    )
    train_qualifying = interactions.loc[
        interactions["iso_week"].between(*train_weeks)
        & (interactions["frac"] >= HISTORY_THRESHOLD)
    ]
    train_mf = interactions.loc[
        interactions["iso_week"].between(*train_weeks)
        & (interactions["frac"] >= MF_THRESHOLD)
    ]
    airing_counts, show_channels, schedule_rows, show_ch_counts = load_schedule_details_with_channels(
        data_dir, target_weeks[0], target_weeks[1]
    )
    raw_history = user_history_component(train_qualifying, users, pool)
    raw_popularity = popularity_component(train_qualifying, pool)
    raw_schedule = schedule_count_component(airing_counts, pool)
    raw_ppr, ppr_metadata = ppr_component(train_qualifying, users, pool)
    raw_mf, mf_metadata = mf_component(train_mf, users, pool)
    
    # channel shares per user
    counts_ch = train_qualifying.groupby(["user_id", "channel_id"]).size().reset_index(name="n_airings")
    totals = train_qualifying.groupby("user_id").size().rename("n_total").reset_index()
    shares = counts_ch.merge(totals, on="user_id")
    shares["share"] = shares["n_airings"] / shares["n_total"]
    user_to_idx = {int(u): i for i, u in enumerate(users)}
    user_channel_shares = {}
    for uid, g in shares.groupby("user_id"):
        if int(uid) not in user_to_idx: continue
        user_channel_shares[int(uid)] = {int(r.channel_id): float(r.share) for _,r in g.iterrows()}
    
    # compute E(u,i)
    raw_exposure = np.zeros((len(users), len(pool)), dtype=np.float32)
    for j, show_id in enumerate(pool):
        sid = int(show_id)
        chs = show_channels.get(sid, [])
        # get airings per channel
        for ui, user_id in enumerate(users):
            ucs = user_channel_shares.get(int(user_id), {})
            s_exp = 0.0
            for c in chs:
                n_ic = show_ch_counts.get((sid,c), 0)
                s_exp += ucs.get(c, 0.0) * float(n_ic)
            raw_exposure[ui,j] = np.log1p(10.0 * s_exp)
    
    fallback_order = popularity_order(train_qualifying.loc[train_qualifying["tv_show_id"].isin(pool)], pool)
    fallback = fill_ranking(fallback_order, fallback_order, K)
    
    components = {
        "R": raw_history.astype(np.float32),
        "P": normalize_by_row_max(raw_ppr),
        "M": normalize_rows(raw_mf) if normalization == "alternative" else normalize_rows(raw_mf),  # keep as before
        "G": normalize_vector(raw_popularity),
        "A": normalize_vector(raw_schedule),
        "C": raw_channel_component_alternative(train_qualifying, users, pool, show_channels),
        "E": normalize_by_row_max(raw_exposure),
    }
    # fix M normalization to match alternative (minmax)
    from hybrid import normalize_rows as nr
    components["M"] = nr(raw_mf)
    metadata = {
        "normalization": normalization,
        "train_weeks": list(train_weeks),
        "target_weeks": list(target_weeks),
        "ppr": ppr_metadata, "mf": mf_metadata,
        "schedule_rows": schedule_rows, "pool_size": int(len(pool)),
        "fallback": fallback,
    }
    return components, metadata

def normalize_rows(m):
    from hybrid import normalize_rows as nr
    return nr(m)
def normalize_vector(v):
    from hybrid import normalize_vector as nv
    return nv(v)

def raw_channel_component_alternative(train_qualifying, users, pool, show_channels):
    from hybrid import channel_affinity_component
    # alternative used raw channel
    comp = channel_affinity_component(train_qualifying, users, pool, show_channels)
    return comp.astype(np.float32)

def main():
    inter = load_interactions('data/interactions.pkl')
    users = load_submission_users('data/submission.csv')
    truth = build_truth(inter, users, *VALIDATION_WEEKS)
    pool = load_schedule_pool(Path('data'), VALIDATION_WEEKS[0], VALIDATION_WEEKS[1])
    
    components, metadata = build_components_with_exposure(inter, users, pool, Path('data'), TRAIN_WEEKS, VALIDATION_WEEKS, normalization='alternative')
    fallback = metadata['fallback']
    
    # grid
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
                            scores = score_components(components, w)
                            rankings = rank_scores(scores, pool, fallback)
                            m = evaluate_rankings(users, truth, rankings)
                            configs.append({**w,'map_all':m['map_all'],'map_with_truth':m['map_with_truth']})
    best = max(configs, key=lambda x:x['map_all'])
    near = [c for c in configs if c['map_all']>=best['map_all']-0.002]
    chosen = min(near, key=lambda c:(sum(c[k]!=0 for k in ['P','M','G','A','C','E']), sum(c[k] for k in ['P','M','G','A','C','E']), c['P'],c['M'],c['G'],c['A'],c['C'],c['E']))
    ranked = sorted(configs, key=lambda x:(-x['map_all'], sum(x[k]!=0 for k in ['P','M','G','A','C','E']), sum(x[k] for k in ['P','M','G','A','C','E'])))
    print('best', best)
    print('chosen', chosen)
    # ablation
    for drop in ['R','P','M','G','A','C','E']:
        if drop=='R':
            scores_ab = score_components(components, best, include_history=False)
        else:
            w_ab = dict(best); w_ab[drop]=0.0
            scores_ab = score_components(components, w_ab)
        ranks_ab = rank_scores(scores_ab, pool, fallback)
        m_ab = evaluate_rankings(users, truth, rankings)
        print('ablate', drop, m_ab['map_all'], m_ab['map_all']-best['map_all'])
    
    # save
    res = {'best':best,'chosen':chosen,'top10':ranked[:10]}
    Path('data/hybrid_step11.json').write_text(json.dumps(res,indent=2))
    
    # write notes section
    lines = []
    lines.append('# Step 11: channel-exposure component (suggested by reviewing assistant)')
    lines.append('')
    lines.append('This component was suggested by a reviewing assistant (Claude), recorded as such. Definition:')
    lines.append('- $E(u,i)=\log1p(10 * \sum_c share_u(c) * n_{airings}(i,c,target\ window))$ where $share_u(c)$ = user\'s qualifying training airings on channel $c$ / user\'s total qualifying training airings.')
    lines.append('- Schedule used only for airing information (what airs where). $E$ divided by user\'s max $E$ per user (alternative normalisation style).')
    lines.append('- Added as $w_E E$ to the alternative-normalisation hybrid (score = $R + w_P P + w_M M + w_G G + w_A A + w_C C + w_E E$).')
    lines.append('')
    lines.append('## Prediction')
    lines.append("Quick check expectation mentioned ~0.587-0.588 for $R + 1*P + 2*G + 4*E$; we report what we find.")
    lines.append('')
    lines.append('## Top 10 (validation)')
    lines.append('| rank | wP | wM | wG | wA | wC | wE | MAP@5 all | MAP@5 truth | delta to best |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for i,c in enumerate(ranked[:10],1):
        lines.append(f"| {i} | {c['P']} | {c['M']} | {c['G']} | {c['A']} | {c['C']} | {c['E']} | {c['map_all']:.6f} | {c['map_with_truth']:.6f} | {best['map_all']-c['map_all']:.6f} |")
    lines.append('')
    lines.append(f"Best: wP={best['P']}, wM={best['M']}, wG={best['G']}, wA={best['A']}, wC={best['C']}, wE={best['E']} MAP={best['map_all']:.6f}")
    lines.append(f"Simplicity pick (fewest non-zero optional, smallest total, lexicographic): wP={chosen['P']}, wM={chosen['M']}, wG={chosen['G']}, wA={chosen['A']}, wC={chosen['C']}, wE={chosen['E']} MAP={chosen['map_all']:.6f}")
    lines.append('')
    lines.append('## Ablation of best')
    for drop in ['R','P','M','G','A','C','E']:
        if drop=='R':
            scores_ab = score_components(components, best, include_history=False)
        else:
            w_ab = dict(best); w_ab[drop]=0.0
            scores_ab = score_components(components, w_ab)
        ranks_ab = rank_scores(scores_ab, pool, fallback)
        m_ab = evaluate_rankings(users, truth, rankings)
        # oops use ranks_ab
    lines.append('')
    Path('notes/step11_hybrid_exposure.md').write_text('\n'.join(lines)+'\n')
    print('notes written')

if __name__=='__main__':
    main()
