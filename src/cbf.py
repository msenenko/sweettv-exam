from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from .evaluate import (
        K,
        TRAIN_WEEKS,
        VALIDATION_WEEKS,
        TEST_WEEKS,
        WINDOW_SCHEDULE,
        apk,
        build_truth,
        fill_ranking,
        load_interactions,
        load_schedule_pool,
        load_submission_users,
        mean_metrics,
        popularity_order,
        training_rows,
    )
except ImportError:
    from evaluate import (
        K,
        TRAIN_WEEKS,
        VALIDATION_WEEKS,
        TEST_WEEKS,
        WINDOW_SCHEDULE,
        apk,
        build_truth,
        fill_ranking,
        load_interactions,
        load_schedule_pool,
        load_submission_users,
        mean_metrics,
        popularity_order,
        training_rows,
    )

HISTORY_THRESHOLD = 0.3


def load_schedule_for_weeks(data_dir: Path, week_lo: int, week_hi: int) -> pd.DataFrame:
    filename = WINDOW_SCHEDULE.get((week_lo, week_hi))
    if filename is None:
        raise ValueError(f"no schedule mapping for weeks {week_lo}-{week_hi}")
    frame = pd.read_csv(
        data_dir / filename,
        usecols=[
            "channel_id",
            "start_time",
            "tv_show_title",
            "tv_show_id",
            "tv_show_category",
            "tv_show_genre_1",
            "tv_show_genre_2",
            "tv_show_genre_3",
        ],
    )
    start = pd.to_datetime(frame["start_time"], format="%d.%m.%Y %H:%M:%S")
    week = start.dt.isocalendar().week.astype(int)
    mask = week.between(week_lo, week_hi) & frame["tv_show_id"].ne(0)
    return frame.loc[mask].copy()


def tokenize_title(title: str) -> list[str]:
    if pd.isna(title):
        return []
    text = str(title).lower()
    # strip season/episode numbering like s01e01, 1x01, season 1, серия 1 etc
    text = re.sub(r"\b(s\d{1,2}e\d{1,3}|e\d{1,3}|\d{1,2}x\d{1,3})\b", " ", text)
    text = re.sub(r"\b(сезон|серия|серії|випуск)\s*\d+\b", " ", text)
    text = re.sub(r"[^a-zа-яёіїєґ0-9]+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


def build_item_features(
    schedule: pd.DataFrame,
    pool: np.ndarray,
):
    pool_set = set(int(x) for x in pool)
    # aggregate per show (take first occurrence for categorical/title, union channels)
    rows = []
    for show_id, group in schedule.groupby("tv_show_id"):
        if int(show_id) not in pool_set:
            continue
        g = group.iloc[0]
        channels = sorted(set(int(c) for c in group["channel_id"]))
        rows.append(
            {
                "tv_show_id": int(show_id),
                "category": str(g["tv_show_category"]) if pd.notna(g["tv_show_category"]) else "",
                "genre1": str(g["tv_show_genre_1"]) if pd.notna(g["tv_show_genre_1"]) else "",
                "genre2": str(g["tv_show_genre_2"]) if pd.notna(g["tv_show_genre_2"]) else "",
                "genre3": str(g["tv_show_genre_3"]) if pd.notna(g["tv_show_genre_3"]) else "",
                "title": str(g["tv_show_title"]) if pd.notna(g["tv_show_title"]) else "",
                "channels": channels,
            }
        )
    item_df = pd.DataFrame(rows).set_index("tv_show_id").reindex(pool)
    return item_df


def build_tfidf_matrix(item_df: pd.DataFrame, pool: np.ndarray):
    # build vocabulary from titles
    docs_tokens = []
    for show_id in pool:
        title = item_df.at[show_id, "title"] if show_id in item_df.index else ""
        docs_tokens.append(tokenize_title(title))
    # compute idf
    N = len(docs_tokens)
    df_count = Counter()
    for tokens in docs_tokens:
        df_count.update(set(tokens))
    idf = {}
    for term, df in df_count.items():
        idf[term] = math.log((N + 1) / (df + 1)) + 1.0  # smooth idf
    # build mapping
    terms = sorted(idf.keys())
    term_to_idx = {t: i for i, t in enumerate(terms)}
    # build matrix
    rows, cols, data = [], [], []
    for i, tokens in enumerate(docs_tokens):
        tf = Counter(tokens)
        # l2 normalize
        norm = 0.0
        for t, c in tf.items():
            v = c * idf.get(t, 0)
            norm += v * v
        norm = math.sqrt(norm) if norm > 0 else 1.0
        for t, c in tf.items():
            if t in term_to_idx:
                rows.append(i)
                cols.append(term_to_idx[t])
                data.append((c * idf[t]) / norm)
    mat = csr_matrix((data, (rows, cols)), shape=(len(pool), len(terms)), dtype=np.float32)
    return mat, terms


def build_block_matrices(item_df: pd.DataFrame, pool: np.ndarray, tfidf_mat, tfidf_terms):
    n = len(pool)
    # category
    cats = item_df["category"].fillna("").astype(str).tolist()
    cat_set = sorted(set(c for c in cats if c))
    cat_to_idx = {c: i for i, c in enumerate(cat_set)}
    rows_c, cols_c, data_c = [], [], []
    for i, c in enumerate(cats):
        if c in cat_to_idx:
            rows_c.append(i); cols_c.append(cat_to_idx[c]); data_c.append(1.0)
    cat_mat = csr_matrix((data_c, (rows_c, cols_c)), shape=(n, len(cat_set)), dtype=np.float32)
    # genres (union)
    genres = []
    for show_id in pool:
        gset = set()
        for col in ["genre1", "genre2", "genre3"]:
            v = item_df.at[show_id, col] if show_id in item_df.index else ""
            if pd.notna(v) and str(v).strip():
                gset.add(str(v).strip())
        genres.append(gset)
    genre_set = sorted(set(g for gs in genres for g in gs))
    genre_to_idx = {g: i for i, g in enumerate(genre_set)}
    rows_g, cols_g, data_g = [], [], []
    for i, gs in enumerate(genres):
        for g in gs:
            rows_g.append(i); cols_g.append(genre_to_idx[g]); data_g.append(1.0)
    genre_mat = csr_matrix((data_g, (rows_g, cols_g)), shape=(n, len(genre_set)), dtype=np.float32)
    # channels
    ch_lists = []
    for show_id in pool:
        if show_id in item_df.index:
            ch_lists.append(item_df.at[show_id, "channels"])
        else:
            ch_lists.append([])
    ch_set = sorted(set(c for cl in ch_lists for c in cl))
    ch_to_idx = {c: i for i, c in enumerate(ch_set)}
    rows_ch, cols_ch, data_ch = [], [], []
    for i, cl in enumerate(ch_lists):
        for c in cl:
            rows_ch.append(i); cols_ch.append(ch_to_idx[c]); data_ch.append(1.0)
    ch_mat = csr_matrix((data_ch, (rows_ch, cols_ch)), shape=(n, len(ch_set)), dtype=np.float32)
    return {
        "cat": cat_mat,
        "genre": genre_mat,
        "channel": ch_mat,
        "tfidf": tfidf_mat,
    }, {
        "cat": len(cat_set),
        "genre": len(genre_set),
        "channel": len(ch_set),
        "tfidf": len(tfidf_terms),
    }


def l2_normalize_rows(mat):
    # mat is sparse or dense
    if hasattr(mat, 'toarray'):
        arr = mat.toarray().astype(np.float32)
    else:
        arr = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def build_user_profile(train_qualifying, users, pool, block_mats, weights):
    # weights per block
    pool_to_idx = {pid: i for i, pid in enumerate(pool)}
    # get per-block normalized item matrices
    item_vecs = []
    for name in ["cat", "genre", "channel", "tfidf"]:
        item_vecs.append(l2_normalize_rows(block_mats[name]))
    # combine
    w = np.array([weights["cat"], weights["genre"], weights["channel"], weights["tfidf"]], dtype=np.float32)
    # weighted sum then l2 normalize
    combined = np.zeros((len(pool), sum(v.shape[1] for v in item_vecs)), dtype=np.float32)
    offset = 0
    for i, v in enumerate(item_vecs):
        combined[:, offset:offset+v.shape[1]] = v * w[i]
        offset += v.shape[1]
    combined = l2_normalize_rows(combined)
    # build user profiles
    user_to_idx = {uid: i for i, uid in enumerate(users)}
    profiles = np.zeros((len(users), combined.shape[1]), dtype=np.float32)
    # aggregate
    for (user_id, show_id), cnt in train_qualifying.groupby(["user_id", "tv_show_id"]).size().items():
        if int(user_id) not in user_to_idx or int(show_id) not in pool_to_idx:
            continue
        ui = user_to_idx[int(user_id)]
        ii = pool_to_idx[int(show_id)]
        profiles[ui] += combined[ii] * np.log1p(cnt)
    # l2 normalize profiles
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    profiles = profiles / norms
    # scores
    scores = profiles @ combined.T  # (n_users, n_items)
    return scores, combined


def compute_cbf(
    interactions,
    users,
    pool,
    data_dir,
    train_weeks,
    target_weeks,
    weights,
):
    train_qualifying = interactions.loc[
        interactions["iso_week"].between(*train_weeks)
        & (interactions["frac"] >= HISTORY_THRESHOLD)
    ]
    schedule = load_schedule_for_weeks(data_dir, target_weeks[0], target_weeks[1])
    item_df = build_item_features(schedule, pool)
    tfidf_mat, tfidf_terms = build_tfidf_matrix(item_df, pool)
    block_mats, block_dims = build_block_matrices(item_df, pool, tfidf_mat, tfidf_terms)
    scores, combined = build_user_profile(train_qualifying, users, pool, block_mats, weights)
    return scores, block_dims, item_df


def rank_cbf(scores, pool, fallback):
    # scores shape (n_users, n_pool)
    order = np.argsort(-scores, axis=1, kind="stable")
    rankings = []
    for i in range(scores.shape[0]):
        primary = []
        for pos in order[i]:
            if scores[i, pos] > 0:
                primary.append(int(pool[pos]))
                if len(primary) == K:
                    break
        rankings.append(fill_ranking(primary, fallback, K))
    return rankings


def evaluate_cbf_predictions(users, truth, rankings):
    scores_list = []
    has_truth = []
    for i, user in enumerate(users):
        actual = truth.get(int(user), set())
        has_truth.append(bool(actual))
        scores_list.append(apk(actual, rankings[i], K))
    scores_arr = np.array(scores_list)
    has_truth_arr = np.array(has_truth)
    return {
        "map_all": float(scores_arr.mean()),
        "map_with_truth": float(scores_arr[has_truth_arr].mean()) if has_truth_arr.sum() else 0.0,
        "n_users": int(len(scores_arr)),
        "n_with_truth": int(has_truth_arr.sum()),
    }
