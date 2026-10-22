from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

K = 5
TRUTH_FRAC = 0.8
TRAIN_WEEKS = (11, 20)
VALIDATION_WEEKS = (21, 30)
TEST_WEEKS = (31, 42)
WINDOW_SCHEDULE = {
    (11, 20): "export_arh_11-20-final.csv",
    (21, 30): "export_arh_21-30-final.csv",
    (31, 42): "export_arh_31-42-final.csv",
}


def apk(actual, predicted, k: int = K) -> float:
    if not actual:
        return 0.0
    predicted = list(predicted)[:k]
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(predicted):
        if item in actual and item not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def load_interactions(path: Path) -> pd.DataFrame:
    interactions = pd.read_pickle(path)
    required = {
        "user_id",
        "channel_id",
        "tv_show_id",
        "airing_start",
        "duration",
        "overlap",
        "frac",
        "iso_week",
    }
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")
    return interactions


def load_submission_users(path: Path) -> np.ndarray:
    submission = pd.read_csv(path, usecols=["user_id"])
    if submission["user_id"].duplicated().any():
        raise ValueError("duplicate users in submission")
    return submission["user_id"].unique()


def load_schedule_pool(
    data_dir: Path, week_lo: int, week_hi: int
) -> np.ndarray:
    filename = WINDOW_SCHEDULE.get((week_lo, week_hi))
    if filename is None:
        raise ValueError(f"no schedule mapping for weeks {week_lo}-{week_hi}")
    frame = pd.read_csv(
        data_dir / filename, usecols=["start_time", "tv_show_id"]
    )
    start = pd.to_datetime(frame["start_time"], format="%d.%m.%Y %H:%M:%S")
    week = start.dt.isocalendar().week.astype(int)
    mask = week.between(week_lo, week_hi) & frame["tv_show_id"].ne(0)
    return np.sort(frame.loc[mask, "tv_show_id"].unique()).astype(np.int64)


def build_truth(
    interactions: pd.DataFrame,
    users: np.ndarray,
    week_lo: int,
    week_hi: int,
    frac_threshold: float = TRUTH_FRAC,
) -> dict[int, set[int]]:
    mask = (
        interactions["iso_week"].between(week_lo, week_hi)
        & (interactions["frac"] >= frac_threshold)
        & interactions["user_id"].isin(users)
    )
    truth_rows = interactions.loc[mask, ["user_id", "tv_show_id"]].drop_duplicates()
    truth = {int(user): set() for user in users}
    for user, frame in truth_rows.groupby("user_id", sort=False):
        truth[int(user)] = set(frame["tv_show_id"].tolist())
    return truth


def fill_ranking(primary, fallback, k: int = K) -> list[int]:
    ranking: list[int] = []
    seen: set[int] = set()
    for source in (primary, fallback):
        for item in source:
            show_id = int(item)
            if show_id in seen:
                continue
            seen.add(show_id)
            ranking.append(show_id)
            if len(ranking) == k:
                return ranking
    return ranking


def evaluate_predictions(
    users: np.ndarray,
    truth: dict[int, set[int]],
    predict,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(len(users), dtype=float)
    has_truth = np.zeros(len(users), dtype=bool)
    for i, user in enumerate(users):
        user_id = int(user)
        actual = truth.get(user_id, set())
        has_truth[i] = bool(actual)
        scores[i] = apk(actual, predict(user_id), k)
    return scores, has_truth


def training_rows(
    interactions: pd.DataFrame,
    week_lo: int,
    week_hi: int,
    frac_threshold: float,
    pool: np.ndarray,
) -> pd.DataFrame:
    mask = (
        interactions["iso_week"].between(week_lo, week_hi)
        & (interactions["frac"] >= frac_threshold)
        & interactions["tv_show_id"].isin(pool)
    )
    return interactions.loc[mask]


def popularity_order(
    train: pd.DataFrame, pool: np.ndarray
) -> np.ndarray:
    counts = (
        train.groupby("tv_show_id")["user_id"]
        .nunique()
        .rename("n_users")
        .reset_index()
    )
    ordered = pd.DataFrame({"tv_show_id": pool}).merge(
        counts, on="tv_show_id", how="left"
    )
    ordered["n_users"] = ordered["n_users"].fillna(0).astype(np.int64)
    ordered = ordered.sort_values(
        ["n_users", "tv_show_id"],
        ascending=[False, True],
        ignore_index=True,
    )
    return ordered["tv_show_id"].to_numpy()


def user_history_orders(
    train: pd.DataFrame, users: np.ndarray
) -> dict[int, np.ndarray]:
    subset = train.loc[train["user_id"].isin(users)]
    history = (
        subset.groupby(["user_id", "tv_show_id"], as_index=False)
        .size()
        .rename(columns={"size": "n_airings"})
    )
    history = history.sort_values(
        ["user_id", "n_airings", "tv_show_id"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    orders: dict[int, np.ndarray] = {}
    for user, frame in history.groupby("user_id", sort=False):
        orders[int(user)] = frame["tv_show_id"].to_numpy()
    return orders


def mean_metrics(scores: np.ndarray, has_truth: np.ndarray) -> dict[str, float]:
    return {
        "map_all": float(scores.mean()),
        "map_with_truth": float(scores[has_truth].mean()),
        "n_users": int(len(scores)),
        "n_with_truth": int(has_truth.sum()),
    }
