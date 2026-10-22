from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

try:
    from .evaluate import (
        K,
        TRAIN_WEEKS,
        VALIDATION_WEEKS,
        WINDOW_SCHEDULE,
        apk,
        build_truth,
        fill_ranking,
        load_interactions,
        load_schedule_pool,
        load_submission_users,
        popularity_order,
    )
    from .mf import build_matrix
    from .ppr import (
        build_graph,
        ppr,
        transition_blocks,
    )
except ImportError:
    from evaluate import (
        K,
        TRAIN_WEEKS,
        VALIDATION_WEEKS,
        WINDOW_SCHEDULE,
        apk,
        build_truth,
        fill_ranking,
        load_interactions,
        load_schedule_pool,
        load_submission_users,
        popularity_order,
    )
    from mf import build_matrix
    from ppr import (
        build_graph,
        ppr,
        transition_blocks,
    )

HISTORY_THRESHOLD = 0.3
PPR_ALPHA = 0.7
PPR_THRESHOLD = 0.3
MF_FACTORS = 32
MF_CONFIDENCE_ALPHA = 5.0
MF_THRESHOLD = 0.5
MF_REGULARIZATION = 0.1
MF_ITERATIONS = 15
MF_SEED = 42
MF_NUM_THREADS = 1

WP_VALUES = (2.0, 3.0, 4.0, 6.0)
WM_VALUES = (0.0, 0.5)
WG_VALUES = (0.5, 1.0, 2.0)
WA_VALUES = (1.0, 1.5, 2.0, 3.0)
WC_VALUES = (0.0, 1.0, 2.0)
COMPONENT_WEIGHTS = ("P", "M", "G", "A", "C")
NORMALIZATIONS = ("minmax", "alternative")


def load_schedule_details(
    data_dir: Path, week_lo: int, week_hi: int
):
    filename = WINDOW_SCHEDULE.get((week_lo, week_hi))
    if filename is None:
        raise ValueError(f"no schedule mapping for weeks {week_lo}-{week_hi}")
    frame = pd.read_csv(
        data_dir / filename,
        usecols=["start_time", "channel_id", "tv_show_id"],
    )
    start = pd.to_datetime(frame["start_time"], format="%d.%m.%Y %H:%M:%S")
    week = start.dt.isocalendar().week.astype(int)
    mask = week.between(week_lo, week_hi) & frame["tv_show_id"].ne(0)
    schedule = frame.loc[mask]
    counts = schedule.groupby("tv_show_id").size()
    channels = schedule.groupby("tv_show_id")["channel_id"].agg(
        lambda values: sorted({int(value) for value in values})
    )
    return counts, channels.to_dict(), int(len(schedule))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    span = maximum - minimum
    output = np.zeros_like(values)
    np.divide(
        values - minimum,
        span,
        out=output,
        where=span > 0,
    )
    return output


def normalize_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    minimum = float(vector.min())
    maximum = float(vector.max())
    if maximum <= minimum:
        return np.zeros_like(vector)
    return ((vector - minimum) / (maximum - minimum)).astype(np.float32)


def normalize_by_row_max(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    maximum = values.max(axis=1, keepdims=True)
    output = np.zeros_like(values)
    np.divide(values, maximum, out=output, where=maximum > 0)
    return output


def user_history_component(
    train: pd.DataFrame, users: np.ndarray, pool: np.ndarray
) -> np.ndarray:
    pairs = (
        train.groupby(["user_id", "tv_show_id"], sort=False)
        .size()
        .reset_index(name="n_airings")
    )
    rows = pd.Index(users).get_indexer(pairs["user_id"])
    cols = pd.Index(pool).get_indexer(pairs["tv_show_id"])
    keep = (rows >= 0) & (cols >= 0)
    component = np.zeros((len(users), len(pool)), dtype=np.float32)
    component[rows[keep], cols[keep]] = np.log1p(
        pairs.loc[keep, "n_airings"].to_numpy(dtype=np.float32)
    )
    return component


def popularity_component(
    train: pd.DataFrame, pool: np.ndarray
) -> np.ndarray:
    counts = (
        train.groupby("tv_show_id")["user_id"]
        .nunique()
        .reindex(pool)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    return counts


def schedule_count_component(
    airing_counts: pd.Series, pool: np.ndarray
) -> np.ndarray:
    counts = (
        airing_counts.reindex(pool)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    return np.log1p(counts)


def channel_affinity_component(
    train: pd.DataFrame,
    users: np.ndarray,
    pool: np.ndarray,
    show_channels: dict[int, list[int]],
) -> np.ndarray:
    counts = (
        train.groupby(["user_id", "channel_id"], sort=False)
        .size()
        .reset_index(name="n_airings")
    )
    totals = (
        train.groupby("user_id", sort=False)
        .size()
        .rename("n_total")
        .reset_index()
    )
    shares = counts.merge(totals, on="user_id", how="left")
    shares["share"] = shares["n_airings"] / shares["n_total"]
    user_positions = pd.Index(users).get_indexer(shares["user_id"])
    keep = user_positions >= 0
    shares = shares.loc[keep]
    user_positions = user_positions[keep]
    channel_ids = np.sort(shares["channel_id"].unique())
    channel_positions = {
        int(channel): position
        for position, channel in enumerate(channel_ids)
    }
    share_matrix = np.zeros(
        (len(users), len(channel_ids)), dtype=np.float32
    )
    columns = shares["channel_id"].map(channel_positions).to_numpy()
    share_matrix[user_positions, columns] = shares["share"].to_numpy(
        dtype=np.float32
    )
    component = np.zeros((len(users), len(pool)), dtype=np.float32)
    for pool_position, show_id in enumerate(pool):
        channels = show_channels.get(int(show_id), [])
        positions = [
            channel_positions[channel]
            for channel in channels
            if channel in channel_positions
        ]
        if positions:
            component[:, pool_position] = share_matrix[:, positions].max(
                axis=1
            )
    return component


def ppr_component(
    train: pd.DataFrame, users: np.ndarray, pool: np.ndarray
):
    matrix, user_ids, show_ids = build_graph(train)
    user_to_show, show_to_user = transition_blocks(matrix)
    user_index = pd.Index(user_ids)
    show_index = pd.Index(show_ids)
    target_index_all = user_index.get_indexer(users)
    active_targets = target_index_all >= 0
    target_user_index = target_index_all[active_targets]
    show_mass, iterations, final_l1, max_mass_error = ppr(
        user_to_show,
        show_to_user,
        target_user_index,
        PPR_ALPHA,
    )
    pool_to_graph = show_index.get_indexer(pool)
    present = pool_to_graph >= 0
    active_component = np.zeros(
        (int(active_targets.sum()), len(pool)), dtype=np.float32
    )
    active_component[:, present] = show_mass[:, pool_to_graph[present]]
    component = np.zeros((len(users), len(pool)), dtype=np.float32)
    component[active_targets] = active_component
    metadata = {
        "train_rows": int(len(train)),
        "graph_users": int(len(user_ids)),
        "graph_shows": int(len(show_ids)),
        "edges": int(matrix.nnz),
        "active_target_users": int(active_targets.sum()),
        "inactive_target_users": int((~active_targets).sum()),
        "iterations": int(iterations),
        "final_l1": float(final_l1),
        "max_mass_error": float(max_mass_error),
    }
    return component, metadata


def mf_component(
    train: pd.DataFrame, users: np.ndarray, pool: np.ndarray
):
    matrix, user_ids, show_ids = build_matrix(train)
    model = AlternatingLeastSquares(
        factors=MF_FACTORS,
        regularization=MF_REGULARIZATION,
        alpha=MF_CONFIDENCE_ALPHA,
        iterations=MF_ITERATIONS,
        use_cg=False,
        random_state=MF_SEED,
        num_threads=MF_NUM_THREADS,
    )
    model.fit(matrix, show_progress=False)
    target_index_all = pd.Index(user_ids).get_indexer(users)
    active_targets = target_index_all >= 0
    pool_to_item = pd.Index(show_ids).get_indexer(pool)
    present = pool_to_item >= 0
    active_component = np.zeros(
        (int(active_targets.sum()), len(pool)), dtype=np.float32
    )
    active_component[:, present] = (
        model.user_factors[target_index_all[active_targets]]
        @ model.item_factors.T
    )[:, pool_to_item[present]]
    component = np.zeros((len(users), len(pool)), dtype=np.float32)
    component[active_targets] = active_component
    metadata = {
        "train_rows": int(len(train)),
        "train_users": int(len(user_ids)),
        "train_shows": int(len(show_ids)),
        "active_target_users": int(active_targets.sum()),
        "inactive_target_users": int((~active_targets).sum()),
        "factors": MF_FACTORS,
        "confidence_alpha": MF_CONFIDENCE_ALPHA,
        "threshold": MF_THRESHOLD,
        "regularization": MF_REGULARIZATION,
        "iterations": MF_ITERATIONS,
        "seed": MF_SEED,
        "num_threads": MF_NUM_THREADS,
    }
    return component, metadata


def build_components(
    interactions: pd.DataFrame,
    users: np.ndarray,
    pool: np.ndarray,
    data_dir: Path,
    train_weeks: tuple[int, int],
    target_weeks: tuple[int, int],
    normalization: str = "minmax",
):
    train_qualifying = interactions.loc[
        interactions["iso_week"].between(*train_weeks)
        & (interactions["frac"] >= HISTORY_THRESHOLD)
    ]
    train_mf = interactions.loc[
        interactions["iso_week"].between(*train_weeks)
        & (interactions["frac"] >= MF_THRESHOLD)
    ]
    airing_counts, show_channels, schedule_rows = load_schedule_details(
        data_dir, target_weeks[0], target_weeks[1]
    )
    raw_history = user_history_component(
        train_qualifying, users, pool
    )
    raw_popularity = popularity_component(train_qualifying, pool)
    raw_schedule = schedule_count_component(airing_counts, pool)
    raw_channel = channel_affinity_component(
        train_qualifying, users, pool, show_channels
    )
    raw_ppr, ppr_metadata = ppr_component(
        train_qualifying, users, pool
    )
    raw_mf, mf_metadata = mf_component(train_mf, users, pool)
    fallback_order = popularity_order(
        train_qualifying.loc[
            train_qualifying["tv_show_id"].isin(pool)
        ],
        pool,
    )
    fallback = fill_ranking(fallback_order, fallback_order, K)
    if normalization == "minmax":
        components = {
            "R": normalize_rows(raw_history),
            "P": normalize_rows(raw_ppr),
            "M": normalize_rows(raw_mf),
            "G": normalize_vector(raw_popularity),
            "A": normalize_vector(raw_schedule),
            "C": normalize_rows(raw_channel),
        }
    elif normalization == "alternative":
        components = {
            "R": raw_history.astype(np.float32),
            "P": normalize_by_row_max(raw_ppr),
            "M": normalize_rows(raw_mf),
            "G": normalize_vector(raw_popularity),
            "A": normalize_vector(raw_schedule),
            "C": raw_channel.astype(np.float32),
        }
    else:
        raise ValueError(f"unknown normalization: {normalization}")
    metadata = {
        "normalization": normalization,
        "train_weeks": list(train_weeks),
        "target_weeks": list(target_weeks),
        "history_threshold": HISTORY_THRESHOLD,
        "ppr_alpha": PPR_ALPHA,
        "ppr_threshold": PPR_THRESHOLD,
        "ppr": ppr_metadata,
        "mf": mf_metadata,
        "schedule_rows": schedule_rows,
        "pool_size": int(len(pool)),
        "fallback": fallback,
        "component_ranges": {
            name: {
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in components.items()
        },
    }
    return components, metadata


def score_components(
    components: dict[str, np.ndarray],
    weights: dict[str, float],
    include_history: bool = True,
) -> np.ndarray:
    scores = (
        components["R"].copy()
        if include_history
        else np.zeros_like(components["R"])
    )
    for name in COMPONENT_WEIGHTS:
        weight = weights.get(name, 0.0)
        if weight == 0.0:
            continue
        if name in ("G", "A"):
            scores += weight * components[name][None, :]
        else:
            scores += weight * components[name]
    return scores


def rank_scores(
    scores: np.ndarray,
    pool: np.ndarray,
    fallback: list[int],
):
    top_positions = np.argpartition(scores, -K, axis=1)[:, -K:]
    rankings: list[list[int]] = []
    for row, candidate_positions in enumerate(top_positions):
        order = np.argsort(
            -scores[row, candidate_positions], kind="stable"
        )
        positions = candidate_positions[order]
        primary = [
            int(pool[position])
            for position in positions
            if scores[row, position] > 0.0
        ]
        rankings.append(fill_ranking(primary, fallback, K))
    return rankings


def evaluate_rankings(
    users: np.ndarray,
    truth: dict[int, set[int]],
    rankings: list[list[int]],
) -> dict[str, float]:
    scores = np.zeros(len(users), dtype=float)
    has_truth = np.zeros(len(users), dtype=bool)
    for row, (user, ranking) in enumerate(zip(users, rankings)):
        actual = truth.get(int(user), set())
        has_truth[row] = bool(actual)
        scores[row] = apk(actual, ranking, K)
    return {
        "map_all": float(scores.mean()),
        "map_with_truth": float(scores[has_truth].mean()),
        "n_users": int(len(scores)),
        "n_with_truth": int(has_truth.sum()),
    }


def evaluate_config(
    components: dict[str, np.ndarray],
    users: np.ndarray,
    truth: dict[int, set[int]],
    pool: np.ndarray,
    fallback: list[int],
    weights: dict[str, float],
    include_history: bool = True,
):
    scores = score_components(
        components, weights, include_history=include_history
    )
    rankings = rank_scores(scores, pool, fallback)
    metrics = evaluate_rankings(users, truth, rankings)
    return metrics


def config_label(config: dict[str, float]) -> str:
    return (
        f"wP={config['P']:.1f}, wM={config['M']:.1f}, "
        f"wG={config['G']:.1f}, wA={config['A']:.1f}, "
        f"wC={config['C']:.1f}"
    )


def simplicity(config: dict[str, float]):
    active = sum(config[name] != 0.0 for name in COMPONENT_WEIGHTS)
    total = sum(config[name] for name in COMPONENT_WEIGHTS)
    return (
        active,
        total,
        config["P"],
        config["M"],
        config["G"],
        config["A"],
        config["C"],
    )


def grid_search(
    components: dict[str, np.ndarray],
    users: np.ndarray,
    truth: dict[int, set[int]],
    pool: np.ndarray,
    fallback: list[int],
):
    configs = []
    for wp in WP_VALUES:
        for wm in WM_VALUES:
            for wg in WG_VALUES:
                for wa in WA_VALUES:
                    for wc in WC_VALUES:
                        weights = {
                            "P": wp,
                            "M": wm,
                            "G": wg,
                            "A": wa,
                            "C": wc,
                        }
                        metrics = evaluate_config(
                            components,
                            users,
                            truth,
                            pool,
                            fallback,
                            weights,
                        )
                        configs.append(
                            {
                                **weights,
                                "map_all": metrics["map_all"],
                                "map_with_truth": metrics[
                                    "map_with_truth"
                                ],
                            }
                        )
    best = max(configs, key=lambda config: config["map_all"])
    near_best = [
        config
        for config in configs
        if config["map_all"] >= best["map_all"] - 0.002
    ]
    chosen = min(near_best, key=simplicity)
    ranked = sorted(
        configs,
        key=lambda config: (
            -config["map_all"],
            simplicity(config),
        ),
    )
    return configs, best, chosen, ranked


def ablation_results(
    components: dict[str, np.ndarray],
    users: np.ndarray,
    truth: dict[int, set[int]],
    pool: np.ndarray,
    fallback: list[int],
    best: dict[str, float],
):
    rows = []
    for name in ("R", "P", "M", "G", "A", "C"):
        weights = {
            key: best[key]
            for key in COMPONENT_WEIGHTS
        }
        include_history = True
        if name == "R":
            include_history = False
        else:
            weights[name] = 0.0
        metrics = evaluate_config(
            components,
            users,
            truth,
            pool,
            fallback,
            weights,
            include_history=include_history,
        )
        rows.append(
            {
                "component": name,
                "weight": 1.0 if name == "R" else best[name],
                "map_all": metrics["map_all"],
                "map_with_truth": metrics["map_with_truth"],
                "delta": best["map_all"] - metrics["map_all"],
            }
        )
    return rows


def notes_section(
    best: dict[str, float],
    chosen: dict[str, float],
    ranked: list[dict[str, float]],
    ablations: list[dict[str, float]],
    metadata: dict[str, object],
    total_runtime: float,
) -> list[str]:
    lines = [
        "# Step 6: hybrid score fusion",
        "",
        "Validation protocol is unchanged: fit on ISO weeks 11-20, validate on "
        "21-30, truth is `frac >= 0.8`, all 1,260 submission users are evaluated, "
        "and the candidate pool is the 6,361 shows airing in weeks 21-30.",
        "",
        "## Components and normalisation",
        "",
        "- `R`: `log1p(user's qualifying training airings of the show)`; "
        "qualifying means `frac >= 0.3`.",
        "- `P`: PPR stationary show mass with `alpha=0.7`, `t=0.3`.",
        f"- `M`: implicit MF score with factors={MF_FACTORS}, "
        f"a={MF_CONFIDENCE_ALPHA:.0f}, `t={MF_THRESHOLD}`, "
        f"`regularization={MF_REGULARIZATION}`, {MF_ITERATIONS} iterations, "
        f"seed {MF_SEED}.",
        "- `G`: global popularity, defined as the number of unique training users "
        "with `frac >= 0.3` for the show.",
        "- `A`: `log1p(number of schedule airings of the show in the target "
        "window)`; the schedule is used only for airing information.",
        "- `C`: maximum, over channels the show airs on in the target window, of "
        "the user's share of their qualifying training airings on that channel.",
        "- Every component is min-max normalised per user over all 6,361 pool "
        "shows. For `G` and `A`, which are user-independent, the same pool-wide "
        "min-max is applied. If a user's max equals min, that component is set "
        "to zero for that user.",
        "- Fusion: `score = R + wP*P + wM*M + wG*G + wA*A + wC*C`; positive "
        "scores only, with popularity fill for short rankings.",
        "",
        "## Pre-run prediction",
        "",
        "Expected ordering of contribution: `C` and `R` largest, `P` useful, "
        "`G` and `A` small positive signals, and `M` near zero or negative "
        "because the best standalone MF was below popularity.",
        "",
        "## Top 10 validation configs",
        "",
        "| rank | wP | wM | wG | wA | wC | MAP@5 all | MAP@5 truth | delta to best |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, config in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {rank} | {config['P']:.1f} | {config['M']:.1f} | "
            f"{config['G']:.1f} | {config['A']:.1f} | "
            f"{config['C']:.1f} | {config['map_all']:.6f} | "
            f"{config['map_with_truth']:.6f} | "
            f"{best['map_all'] - config['map_all']:.6f} |"
        )
    lines += [
        "",
        f"Best: {config_label(best)}, MAP@5 all={best['map_all']:.6f}, "
        f"truth={best['map_with_truth']:.6f}.",
        "",
        f"Chosen by the simplicity rule (fewest non-zero optional weights, then "
        f"smallest total weight, then lexicographic weights, among configs "
        f"within 0.002 of the best): {config_label(chosen)}, "
        f"MAP@5 all={chosen['map_all']:.6f}, "
        f"truth={chosen['map_with_truth']:.6f}.",
        "",
        "## Ablation from the best config",
        "",
        "Each row drops one component from the best config; larger positive "
        "`delta` means that component contributed more.",
        "",
        "| dropped | weight | MAP@5 all | MAP@5 truth | delta vs best |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ablations:
        lines.append(
            f"| {row['component']} | {row['weight']:.1f} | "
            f"{row['map_all']:.6f} | {row['map_with_truth']:.6f} | "
            f"{row['delta']:.6f} |"
        )
    ranked_ablations = sorted(
        ablations, key=lambda row: row["delta"], reverse=True
    )
    observed = ", ".join(
        f"{row['component']} {row['delta']:+.6f}"
        for row in ranked_ablations
    )
    lines += [
        "",
        "## Interpretation",
        "",
        f"Observed ablation deltas: {observed}.",
        "",
        f"The validation grid was used only for weight selection. Test predictions "
        f"use the chosen config {config_label(chosen)} refit on weeks 11-30 with "
        f"the target window 31-42.",
        "",
        f"Grid runtime: {total_runtime:.2f} s. Components and grid are "
        f"deterministic except MF, which uses seed {MF_SEED}.",
        "",
    ]
    return lines


def top10_table_lines(ranked: list[dict[str, float]], best: dict[str, float]) -> list[str]:
    lines = [
        "| rank | wP | wM | wG | wA | wC | MAP@5 all | MAP@5 truth | delta to best |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, config in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {rank} | {config['P']:.1f} | {config['M']:.1f} | "
            f"{config['G']:.1f} | {config['A']:.1f} | "
            f"{config['C']:.1f} | {config['map_all']:.6f} | "
            f"{config['map_with_truth']:.6f} | "
            f"{best['map_all'] - config['map_all']:.6f} |"
        )
    return lines


def extended_notes_lines(
    extended: dict[str, object],
    alternative: dict[str, object],
    overall_best: dict[str, float],
    overall_chosen: dict[str, float],
    total_runtime: float,
) -> list[str]:
    lines = [
        "## Extended grid",
        "",
        "The first grid hit its `wP` and `wA` maxima, so it was extended once: "
        "`wP in {2,3,4,6}`, `wA in {1,1.5,2,3}`, `wG in {0.5,1,2}`, "
        "`wC in {0,1,2}`, and `wM in {0,0.5}`. Default min-max normalisation "
        "is unchanged.",
        "",
        "### Extended grid top 10",
        "",
    ]
    extended_best = extended["best"]
    lines += top10_table_lines(extended["ranked"], extended_best)
    lines += [
        "",
        f"Extended-grid best: {config_label(extended_best)}, "
        f"MAP@5 all={extended_best['map_all']:.6f}, "
        f"truth={extended_best['map_with_truth']:.6f}.",
        "",
        f"Extended-grid chosen by the same simplicity rule: "
        f"{config_label(extended['chosen'])}, "
        f"MAP@5 all={extended['chosen']['map_all']:.6f}, "
        f"truth={extended['chosen']['map_with_truth']:.6f}.",
        "",
        "## Alternative normalisation",
        "",
        "This variant leaves `R` as the raw `log1p` count and `C` as the raw "
        "channel share, divides `P` by the user's maximum `P`, keeps `M` as "
        "per-user min-max, and keeps `G` and `A` as pool-wide min-max.",
        "",
        "### Alternative-normalisation top 10",
        "",
    ]
    alternative_best = alternative["best"]
    lines += top10_table_lines(alternative["ranked"], alternative_best)
    lines += [
        "",
        f"Alternative-normalisation best: {config_label(alternative_best)}, "
        f"MAP@5 all={alternative_best['map_all']:.6f}, "
        f"truth={alternative_best['map_with_truth']:.6f}.",
        "",
        f"Alternative-normalisation chosen by the same simplicity rule: "
        f"{config_label(alternative['chosen'])}, "
        f"MAP@5 all={alternative['chosen']['map_all']:.6f}, "
        f"truth={alternative['chosen']['map_with_truth']:.6f}.",
        "",
        "## Overall selection",
        "",
        "The simplicity rule is applied across both variants: fewest non-zero "
        "optional weights, then smallest total weight, then lexicographic "
        "weights; default min-max wins exact ties.",
        "",
        "| role | normalisation | wP | wM | wG | wA | wC | MAP@5 all | MAP@5 truth |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| best overall | {overall_best['normalization']} | "
        f"{overall_best['P']:.1f} | {overall_best['M']:.1f} | "
        f"{overall_best['G']:.1f} | {overall_best['A']:.1f} | "
        f"{overall_best['C']:.1f} | {overall_best['map_all']:.6f} | "
        f"{overall_best['map_with_truth']:.6f} |",
        f"| chosen overall | {overall_chosen['normalization']} | "
        f"{overall_chosen['P']:.1f} | {overall_chosen['M']:.1f} | "
        f"{overall_chosen['G']:.1f} | {overall_chosen['A']:.1f} | "
        f"{overall_chosen['C']:.1f} | {overall_chosen['map_all']:.6f} | "
        f"{overall_chosen['map_with_truth']:.6f} |",
        "",
        f"Final chosen config: {config_label(overall_chosen)} with "
        f"`{overall_chosen['normalization']}` normalisation, "
        f"MAP@5 all={overall_chosen['map_all']:.6f}, "
        f"truth={overall_chosen['map_with_truth']:.6f}.",
        "",
        f"Extended/alternative runtime: {total_runtime:.2f} s.",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactions", type=Path, default=Path("data/interactions.pkl")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--notes", type=Path, default=Path("notes/step6_hybrid.md")
    )
    parser.add_argument(
        "--results", type=Path, default=Path("data/hybrid_results.json")
    )
    args = parser.parse_args()

    started = time.perf_counter()
    interactions = load_interactions(args.interactions)
    users = load_submission_users(args.data_dir / "submission.csv")
    pool = load_schedule_pool(
        args.data_dir, VALIDATION_WEEKS[0], VALIDATION_WEEKS[1]
    )
    truth = build_truth(
        interactions,
        users,
        VALIDATION_WEEKS[0],
        VALIDATION_WEEKS[1],
    )
    variants: dict[str, dict[str, object]] = {}
    for normalization in NORMALIZATIONS:
        variant_started = time.perf_counter()
        components, metadata = build_components(
            interactions,
            users,
            pool,
            args.data_dir,
            TRAIN_WEEKS,
            VALIDATION_WEEKS,
            normalization=normalization,
        )
        fallback = metadata["fallback"]
        configs, best, chosen, ranked = grid_search(
            components, users, truth, pool, fallback
        )
        for config in configs:
            config["normalization"] = normalization
        variants[normalization] = {
            "normalization": normalization,
            "metadata": metadata,
            "configs": configs,
            "best": best,
            "chosen": chosen,
            "ranked": ranked,
            "top10": ranked[:10],
            "runtime": time.perf_counter() - variant_started,
        }
        print(
            f"{normalization}: best {config_label(best)} "
            f"MAP all={best['map_all']:.6f}; "
            f"chosen {config_label(chosen)} "
            f"MAP all={chosen['map_all']:.6f}"
        )

    all_configs = (
        variants["minmax"]["configs"]
        + variants["alternative"]["configs"]
    )
    overall_best = max(
        all_configs, key=lambda config: config["map_all"]
    )
    near_best = [
        config
        for config in all_configs
        if config["map_all"] >= overall_best["map_all"] - 0.002
    ]
    overall_chosen = min(
        near_best,
        key=lambda config: (
            *simplicity(config),
            0 if config["normalization"] == "minmax" else 1,
        ),
    )
    total_runtime = time.perf_counter() - started
    results = {
        "method": "hybrid_score_fusion",
        "components": {
            "R": "log1p qualifying training airings",
            "P": {
                "alpha": PPR_ALPHA,
                "threshold": PPR_THRESHOLD,
            },
            "M": {
                "factors": MF_FACTORS,
                "confidence_alpha": MF_CONFIDENCE_ALPHA,
                "threshold": MF_THRESHOLD,
                "regularization": MF_REGULARIZATION,
                "iterations": MF_ITERATIONS,
                "seed": MF_SEED,
                "num_threads": MF_NUM_THREADS,
            },
            "G": "unique qualifying training users",
            "A": "log1p target-window schedule airings",
            "C": "max target-window channel share",
        },
        "grid": {
            "P": list(WP_VALUES),
            "M": list(WM_VALUES),
            "G": list(WG_VALUES),
            "A": list(WA_VALUES),
            "C": list(WC_VALUES),
        },
        "variants": {
            "minmax": {
                "best": variants["minmax"]["best"],
                "chosen": variants["minmax"]["chosen"],
                "top10": variants["minmax"]["top10"],
                "runtime": variants["minmax"]["runtime"],
            },
            "alternative": {
                "best": variants["alternative"]["best"],
                "chosen": variants["alternative"]["chosen"],
                "top10": variants["alternative"]["top10"],
                "runtime": variants["alternative"]["runtime"],
            },
        },
        "best": overall_best,
        "chosen": overall_chosen,
        "total_runtime": total_runtime,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    existing = args.notes.read_text(encoding="utf-8")
    appended = extended_notes_lines(
        variants["minmax"],
        variants["alternative"],
        overall_best,
        overall_chosen,
        total_runtime,
    )
    args.notes.write_text(
        existing.rstrip() + "\n\n" + "\n".join(appended) + "\n",
        encoding="utf-8",
    )
    print(
        f"overall best: {config_label(overall_best)} "
        f"normalization={overall_best['normalization']} "
        f"MAP all={overall_best['map_all']:.6f} "
        f"truth={overall_best['map_with_truth']:.6f}"
    )
    print(
        f"overall chosen: {config_label(overall_chosen)} "
        f"normalization={overall_chosen['normalization']} "
        f"MAP all={overall_chosen['map_all']:.6f} "
        f"truth={overall_chosen['map_with_truth']:.6f}"
    )
    print(f"runtime: {total_runtime:.2f} s")
    print(f"wrote {args.results}")
    print(f"wrote {args.notes}")


if __name__ == "__main__":
    main()
