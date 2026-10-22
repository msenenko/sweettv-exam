from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from .evaluate import (
        K,
        TRAIN_WEEKS,
        VALIDATION_WEEKS,
        build_truth,
        evaluate_predictions,
        fill_ranking,
        load_interactions,
        load_schedule_pool,
        load_submission_users,
        mean_metrics,
        popularity_order,
    )
except ImportError:
    from evaluate import (
        K,
        TRAIN_WEEKS,
        VALIDATION_WEEKS,
        build_truth,
        evaluate_predictions,
        fill_ranking,
        load_interactions,
        load_schedule_pool,
        load_submission_users,
        mean_metrics,
        popularity_order,
    )

ALPHAS = (0.3, 0.5, 0.7, 0.85)
THRESHOLDS = (0.3, 0.5)
TOLERANCE = 1e-10
MAX_ITERATIONS = 300


def build_graph(train: pd.DataFrame):
    pairs = (
        train.groupby(["user_id", "tv_show_id"], sort=False)
        .size()
        .reset_index(name="n_airings")
    )
    user_ids, user_index = np.unique(
        pairs["user_id"].to_numpy(), return_inverse=True
    )
    show_ids, show_index = np.unique(
        pairs["tv_show_id"].to_numpy(), return_inverse=True
    )
    weights = np.log1p(pairs["n_airings"].to_numpy(dtype=np.float64))
    matrix = sp.csr_matrix(
        (weights, (user_index, show_index)),
        shape=(len(user_ids), len(show_ids)),
    )
    return matrix, user_ids, show_ids


def transition_blocks(matrix: sp.csr_matrix):
    user_degree = np.asarray(matrix.sum(axis=1)).ravel()
    show_degree = np.asarray(matrix.sum(axis=0)).ravel()
    if (user_degree <= 0).any() or (show_degree <= 0).any():
        raise ValueError("zero-degree graph nodes")
    user_to_show = (sp.diags(1.0 / user_degree) @ matrix).tocsr()
    show_to_user = (sp.diags(1.0 / show_degree) @ matrix.T).tocsr()
    return user_to_show, show_to_user


def ppr(
    user_to_show: sp.csr_matrix,
    show_to_user: sp.csr_matrix,
    target_user_index: np.ndarray,
    alpha: float,
):
    n_targets = len(target_user_index)
    user_mass = np.zeros(
        (n_targets, user_to_show.shape[0]), dtype=np.float64
    )
    user_mass[np.arange(n_targets), target_user_index] = 1.0
    show_mass = np.zeros(
        (n_targets, show_to_user.shape[0]), dtype=np.float64
    )
    max_mass_error = 0.0
    difference = np.inf
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        new_show_mass = alpha * (user_mass @ user_to_show)
        new_user_mass = alpha * (show_mass @ show_to_user)
        new_user_mass[np.arange(n_targets), target_user_index] += 1.0 - alpha
        mass_error = float(
            np.abs(
                new_user_mass.sum(axis=1)
                + new_show_mass.sum(axis=1)
                - 1.0
            ).max()
        )
        max_mass_error = max(max_mass_error, mass_error)
        difference = float(
            np.abs(new_user_mass - user_mass).sum()
            + np.abs(new_show_mass - show_mass).sum()
        )
        user_mass = new_user_mass
        show_mass = new_show_mass
        if difference < TOLERANCE:
            break
    return show_mass, iterations, difference, max_mass_error


def rank_pool(
    scores: np.ndarray,
    pool: np.ndarray,
    pool_to_graph: np.ndarray,
    fallback: list[int],
):
    pool_scores = np.zeros((len(scores), len(pool)), dtype=np.float64)
    present = pool_to_graph >= 0
    pool_scores[:, present] = scores[:, pool_to_graph[present]]
    order = np.argsort(-pool_scores, axis=1, kind="stable")[:, :K]
    rankings: list[list[int]] = []
    for row, candidate_positions in enumerate(order):
        primary = [
            int(pool[position])
            for position in candidate_positions
            if pool_scores[row, position] > 0.0
        ]
        rankings.append(fill_ranking(primary, fallback, K))
    return rankings


def notes_section(results: dict[str, object]) -> list[str]:
    lines = [
        "# Step 4: PPR and implicit MF",
        "",
        "Same validation protocol as Step 3: fit on ISO weeks 11-20, validate on "
        "ISO weeks 21-30, truth is `frac >= 0.8`, all 1,260 submission users are "
        "evaluated, and the candidate pool is the 6,361 shows airing in weeks 21-30.",
        "",
        "## Personalized PageRank",
        "",
        "- Bipartite user-show graph from training interactions with `frac >= t`.",
        "- Edge weight: `log1p(number of qualifying airings)`.",
        "- Row-normalised transitions for both blocks; random walk with restart to "
        "the target user; `alpha` is edge-following probability.",
        "- Score: stationary mass on show nodes, restricted to the pool.",
        "- Zero-degree graph nodes are excluded; submission users with no qualifying "
        "training edges use the popularity fallback.",
        f"- L1 tolerance: `{TOLERANCE:.0e}`; maximum iterations: {MAX_ITERATIONS}.",
        "",
        "### Graph checks",
        "",
        "| t | train rows | graph users | graph shows | edges | inactive target users | pool shows absent from graph | user row-sum min/max | show row-sum min/max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for check in results["graph_checks"]:
        lines.append(
            f"| {check['t']:.1f} | {check['train_rows']:,} | "
            f"{check['graph_users']:,} | {check['graph_shows']:,} | "
            f"{check['edges']:,} | {check['inactive_target_users']:,} | "
            f"{check['pool_shows_absent_from_graph']:,} | "
            f"{check['user_row_min']:.12f}/{check['user_row_max']:.12f} | "
            f"{check['show_row_min']:.12f}/{check['show_row_max']:.12f} |"
        )
    lines += [
        "",
        "All graph nodes have positive degree by construction; zero-degree graph "
        "nodes: 0. Mass conservation is checked each iteration.",
        "",
        "### PPR results",
        "",
        "| alpha | t | iterations | final L1 | max mass error | MAP@5 all | MAP@5 truth | runtime, s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in results["configs"]:
        lines.append(
            f"| {config['alpha']:.2f} | {config['t']:.1f} | "
            f"{config['iterations']} | {config['final_l1']:.3e} | "
            f"{config['max_mass_error']:.3e} | {config['map_all']:.6f} | "
            f"{config['map_with_truth']:.6f} | {config['runtime']:.2f} |"
        )
    best = results["best"]
    lines += [
        "",
        f"Best PPR: alpha={best['alpha']:.2f}, t={best['t']:.1f}, "
        f"MAP@5 all={best['map_all']:.6f}, truth={best['map_with_truth']:.6f}.",
        "",
        f"Total PPR runtime: {results['total_runtime']:.2f} s. Seeds: none "
        "(PPR is deterministic).",
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
        "--notes", type=Path, default=Path("notes/step4_methods.md")
    )
    parser.add_argument(
        "--results", type=Path, default=Path("data/ppr_results.json")
    )
    args = parser.parse_args()

    started = time.perf_counter()
    interactions = load_interactions(args.interactions)
    users = load_submission_users(args.data_dir / "submission.csv")
    pool = load_schedule_pool(
        args.data_dir, VALIDATION_WEEKS[0], VALIDATION_WEEKS[1]
    )
    truth = build_truth(
        interactions, users, VALIDATION_WEEKS[0], VALIDATION_WEEKS[1]
    )
    graph_checks: list[dict[str, object]] = []
    configs: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        graph_started = time.perf_counter()
        train = interactions.loc[
            interactions["iso_week"].between(*TRAIN_WEEKS)
            & (interactions["frac"] >= threshold)
        ]
        matrix, user_ids, show_ids = build_graph(train)
        user_to_show, show_to_user = transition_blocks(matrix)
        graph_runtime = time.perf_counter() - graph_started
        user_index = pd.Index(user_ids)
        show_index = pd.Index(show_ids)
        target_index_all = user_index.get_indexer(users)
        active_targets = target_index_all >= 0
        target_user_index = target_index_all[active_targets]
        target_users = users[active_targets]
        pool_to_graph = show_index.get_indexer(pool)
        fallback_order = popularity_order(
            train.loc[train["tv_show_id"].isin(pool)], pool
        )
        fallback = fill_ranking(fallback_order, fallback_order, K)
        user_rows = np.asarray(user_to_show.sum(axis=1)).ravel()
        show_rows = np.asarray(show_to_user.sum(axis=1)).ravel()
        graph_checks.append(
            {
                "t": threshold,
                "train_rows": int(len(train)),
                "train_users": int(train["user_id"].nunique()),
                "train_shows": int(train["tv_show_id"].nunique()),
                "graph_users": int(len(user_ids)),
                "graph_shows": int(len(show_ids)),
                "edges": int(matrix.nnz),
                "inactive_target_users": int((~active_targets).sum()),
                "pool_shows_absent_from_graph": int((pool_to_graph < 0).sum()),
                "zero_degree_graph_nodes": 0,
                "user_row_min": float(user_rows.min()),
                "user_row_max": float(user_rows.max()),
                "show_row_min": float(show_rows.min()),
                "show_row_max": float(show_rows.max()),
                "graph_runtime": float(graph_runtime),
            }
        )
        for alpha in ALPHAS:
            config_started = time.perf_counter()
            show_mass, iterations, final_l1, max_mass_error = ppr(
                user_to_show, show_to_user, target_user_index, alpha
            )
            rankings = rank_pool(
                show_mass, pool, pool_to_graph, fallback
            )
            predictions = {int(user): fallback for user in users}
            for user, ranking in zip(target_users, rankings):
                predictions[int(user)] = ranking
            scores, has_truth = evaluate_predictions(
                users, truth, lambda user_id: predictions[user_id], K
            )
            metrics = mean_metrics(scores, has_truth)
            runtime = time.perf_counter() - config_started
            configs.append(
                {
                    "alpha": alpha,
                    "t": threshold,
                    "iterations": int(iterations),
                    "final_l1": float(final_l1),
                    "max_mass_error": float(max_mass_error),
                    "map_all": metrics["map_all"],
                    "map_with_truth": metrics["map_with_truth"],
                    "runtime": float(runtime),
                    "graph_runtime": float(graph_runtime),
                }
            )
            print(
                f"PPR alpha={alpha:.2f} t={threshold:.1f}: "
                f"iters={iterations} L1={final_l1:.3e} "
                f"mass_err={max_mass_error:.3e} "
                f"MAP all={metrics['map_all']:.6f} "
                f"truth={metrics['map_with_truth']:.6f} "
                f"runtime={runtime:.2f}s"
            )

    best = max(configs, key=lambda config: config["map_all"])
    total_runtime = time.perf_counter() - started
    results = {
        "method": "ppr",
        "tolerance": TOLERANCE,
        "max_iterations": MAX_ITERATIONS,
        "graph_checks": graph_checks,
        "configs": configs,
        "best": best,
        "total_runtime": total_runtime,
        "pool_size": int(len(pool)),
        "n_users": int(len(users)),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    args.notes.write_text(
        "\n".join(notes_section(results)), encoding="utf-8"
    )
    print(
        f"best PPR: alpha={best['alpha']:.2f} t={best['t']:.1f} "
        f"MAP all={best['map_all']:.6f} truth={best['map_with_truth']:.6f}"
    )
    print(f"total runtime: {total_runtime:.2f} s")
    print(f"wrote {args.results}")
    print(f"wrote {args.notes}")


if __name__ == "__main__":
    main()
