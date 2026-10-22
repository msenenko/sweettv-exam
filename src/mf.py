from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

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
        training_rows,
        user_history_orders,
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
        training_rows,
        user_history_orders,
    )

try:
    from .ppr import notes_section as ppr_notes_section
except ImportError:
    from ppr import notes_section as ppr_notes_section

FACTORS = (32, 64)
CONFIDENCE_ALPHAS = (5.0, 20.0)
THRESHOLDS = (0.3, 0.5)
REGULARIZATION = 0.1
ITERATIONS = 15
SEED = 42
NUM_THREADS = 1


def build_matrix(train: pd.DataFrame):
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
    values = np.log1p(
        pairs["n_airings"].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    matrix = sp.csr_matrix(
        (values, (user_index, show_index)),
        shape=(len(user_ids), len(show_ids)),
        dtype=np.float32,
    )
    return matrix, user_ids, show_ids


def rank_scores(
    scores: np.ndarray,
    pool: np.ndarray,
    pool_to_item: np.ndarray,
    fallback: list[int],
):
    pool_scores = np.zeros((len(scores), len(pool)), dtype=np.float32)
    present = pool_to_item >= 0
    pool_scores[:, present] = scores[:, pool_to_item[present]]
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


def sanity_check(
    model: AlternatingLeastSquares, matrix: sp.csr_matrix
) -> tuple[float, float, int]:
    scores = model.user_factors @ model.item_factors.T
    n_items = scores.shape[1]
    percentiles: list[float] = []
    n_hit5 = 0
    for user_index in range(scores.shape[0]):
        start = matrix.indptr[user_index]
        end = matrix.indptr[user_index + 1]
        positive_items = matrix.indices[start:end]
        if len(positive_items) == 0:
            continue
        order = np.argsort(-scores[user_index], kind="stable")
        ranks = np.empty(n_items, dtype=np.int32)
        ranks[order] = np.arange(n_items, dtype=np.int32)
        positive_ranks = ranks[positive_items]
        percentiles.append(
            float((1.0 - positive_ranks / n_items).mean())
        )
        if (positive_ranks < K).any():
            n_hit5 += 1
    n_users = len(percentiles)
    return (
        float(np.mean(percentiles)),
        float(n_hit5 / n_users),
        int(n_users),
    )


def baseline_results(
    interactions: pd.DataFrame,
    users: np.ndarray,
    pool: np.ndarray,
    truth: dict[int, set[int]],
) -> list[dict[str, object]]:
    train = training_rows(
        interactions, TRAIN_WEEKS[0], TRAIN_WEEKS[1], 0.3, pool
    )
    fallback_order = popularity_order(train, pool)
    fallback = fill_ranking(fallback_order, fallback_order, K)
    history = user_history_orders(train, users)
    results: list[dict[str, object]] = []

    started = time.perf_counter()
    popularity_predictions = {
        int(user): fallback for user in users
    }
    scores, has_truth = evaluate_predictions(
        users, truth, lambda user_id: popularity_predictions[user_id], K
    )
    metrics = mean_metrics(scores, has_truth)
    results.append(
        {
            "method": "popularity",
            "config": "t=0.3",
            "map_all": metrics["map_all"],
            "map_with_truth": metrics["map_with_truth"],
            "runtime": float(time.perf_counter() - started),
        }
    )

    started = time.perf_counter()
    history_predictions = {
        int(user): fill_ranking(
            history.get(int(user), np.array([], dtype=np.int64)),
            fallback,
            K,
        )
        for user in users
    }
    scores, has_truth = evaluate_predictions(
        users, truth, lambda user_id: history_predictions[user_id], K
    )
    metrics = mean_metrics(scores, has_truth)
    results.append(
        {
            "method": "user-history",
            "config": "t=0.3",
            "map_all": metrics["map_all"],
            "map_with_truth": metrics["map_with_truth"],
            "runtime": float(time.perf_counter() - started),
        }
    )
    return results


def notes_section(
    ppr_results: dict[str, object],
    results: dict[str, object],
    baselines: list[dict[str, object]],
) -> list[str]:
    lines = ppr_notes_section(ppr_results)
    lines += [
        "## Implicit matrix factorisation",
        "",
        "- Library: `implicit 0.7.3`, exact ALS (`use_cg=False`), "
        f"`regularization={REGULARIZATION}`, `iterations={ITERATIONS}`, "
        f"seed `{SEED}`, `num_threads={NUM_THREADS}`.",
        "- Training matrix: all training users and all shows seen in weeks 11-20 "
        "with `frac >= t`; values are `log1p(number of qualifying airings)`.",
        "- Hu-Koren-Volinsky implicit-feedback ALS; confidence is "
        "`1 + a * log1p(number of qualifying airings)`.",
        "- Items are all training shows; pool shows absent from training get "
        "score 0. Scores are `user_factors @ item_factors.T` over the pool.",
        "- Positive scores only; fallback to popularity for users with no training "
        "edges or fewer than five positive candidate scores.",
        "- Sanity check on all training users with at least one positive item: "
        "mean percentile of the user's own training shows among all training shows, "
        "and fraction of users with at least one own show in the reconstructed top 5.",
        "",
        "### MF results",
        "",
        "| factors | a | t | own-item percentile | own-item hit@5 | MAP@5 all | MAP@5 truth | runtime, s | sanity users |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in results["configs"]:
        lines.append(
            f"| {config['factors']} | {config['alpha']:.0f} | "
            f"{config['t']:.1f} | {config['sanity_percentile']:.4f} | "
            f"{config['sanity_hit5']:.4f} | {config['map_all']:.6f} | "
            f"{config['map_with_truth']:.6f} | {config['runtime']:.2f} | "
            f"{config['sanity_users']:,} |"
        )
    best = results["best"]
    lines += [
        "",
        f"Best MF: factors={best['factors']}, a={best['alpha']:.0f}, "
        f"t={best['t']:.1f}, MAP@5 all={best['map_all']:.6f}, "
        f"truth={best['map_with_truth']:.6f}.",
        "",
        f"Total MF runtime: {results['total_runtime']:.2f} s. "
        f"Seed: {SEED}; threads: {NUM_THREADS}.",
        "",
        "## Comparison",
        "",
        "| method | config | MAP@5 all | MAP@5 truth | runtime, s |",
        "|---|---|---:|---:|---:|",
    ]
    for baseline in baselines:
        lines.append(
            f"| {baseline['method']} | {baseline['config']} | "
            f"{baseline['map_all']:.6f} | {baseline['map_with_truth']:.6f} | "
            f"{baseline['runtime']:.2f} |"
        )
    ppr_best = ppr_results["best"]
    lines.append(
        f"| PPR | alpha={ppr_best['alpha']:.2f}, t={ppr_best['t']:.1f} | "
        f"{ppr_best['map_all']:.6f} | {ppr_best['map_with_truth']:.6f} | "
        f"{ppr_best['runtime'] + ppr_best['graph_runtime']:.2f} |"
    )
    lines.append(
        f"| MF | factors={best['factors']}, a={best['alpha']:.0f}, "
        f"t={best['t']:.1f} | {best['map_all']:.6f} | "
        f"{best['map_with_truth']:.6f} | {best['runtime']:.2f} |"
    )
    rows = [
        (baseline["method"], baseline["map_all"])
        for baseline in baselines
    ] + [
        ("PPR", ppr_best["map_all"]),
        ("MF", best["map_all"]),
    ]
    order = " > ".join(
        f"{method} ({score:.6f})"
        for method, score in sorted(rows, key=lambda row: row[1], reverse=True)
    )
    mf_wins = best["map_all"] > ppr_best["map_all"]
    verdict = (
        "The prediction that MF narrowly wins was confirmed."
        if mf_wins
        else "The prediction that MF wins was not confirmed; PPR is stronger."
    )
    lines += [
        "",
        "PPR runtime includes one graph build; MF runtime includes fitting and the "
        "sanity check. Baseline runtime covers ranking and evaluation only.",
        "",
        "## Prediction and interpretation",
        "",
        "Before the run, the expected order was MF > PPR > user-history > popularity, "
        "with `t=0.3` best and PPR `alpha` around 0.5-0.7.",
        "",
        f"Observed MAP@5 order: {order}.",
        "",
        f"{verdict} Best PPR uses alpha={ppr_best['alpha']:.2f}, "
        f"t={ppr_best['t']:.1f}; best MF uses factors={best['factors']}, "
        f"a={best['alpha']:.0f}, t={best['t']:.1f}.",
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
        "--ppr-results",
        type=Path,
        default=Path("data/ppr_results.json"),
    )
    parser.add_argument(
        "--notes", type=Path, default=Path("notes/step4_methods.md")
    )
    parser.add_argument(
        "--results", type=Path, default=Path("data/mf_results.json")
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
    ppr_results = json.loads(args.ppr_results.read_text(encoding="utf-8"))
    baselines = baseline_results(interactions, users, pool, truth)
    configs: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        train = interactions.loc[
            interactions["iso_week"].between(*TRAIN_WEEKS)
            & (interactions["frac"] >= threshold)
        ]
        matrix, user_ids, show_ids = build_matrix(train)
        user_index = pd.Index(user_ids)
        show_index = pd.Index(show_ids)
        target_index_all = user_index.get_indexer(users)
        active_targets = target_index_all >= 0
        target_user_index = target_index_all[active_targets]
        target_users = users[active_targets]
        pool_to_item = show_index.get_indexer(pool)
        fallback_order = popularity_order(
            train.loc[train["tv_show_id"].isin(pool)], pool
        )
        fallback = fill_ranking(fallback_order, fallback_order, K)

        for factors in FACTORS:
            for confidence_alpha in CONFIDENCE_ALPHAS:
                config_started = time.perf_counter()
                model = AlternatingLeastSquares(
                    factors=factors,
                    regularization=REGULARIZATION,
                    alpha=confidence_alpha,
                    iterations=ITERATIONS,
                    use_cg=False,
                    random_state=SEED,
                    num_threads=NUM_THREADS,
                )
                model.fit(matrix, show_progress=False)
                user_scores = (
                    model.user_factors[target_user_index]
                    @ model.item_factors.T
                )
                rankings = rank_scores(
                    user_scores, pool, pool_to_item, fallback
                )
                predictions = {int(user): fallback for user in users}
                for user, ranking in zip(target_users, rankings):
                    predictions[int(user)] = ranking
                scores, has_truth = evaluate_predictions(
                    users,
                    truth,
                    lambda user_id: predictions[user_id],
                    K,
                )
                metrics = mean_metrics(scores, has_truth)
                (
                    sanity_percentile,
                    sanity_hit5,
                    sanity_users,
                ) = sanity_check(model, matrix)
                runtime = time.perf_counter() - config_started
                configs.append(
                    {
                        "factors": factors,
                        "alpha": confidence_alpha,
                        "t": threshold,
                        "sanity_percentile": sanity_percentile,
                        "sanity_hit5": sanity_hit5,
                        "sanity_users": sanity_users,
                        "map_all": metrics["map_all"],
                        "map_with_truth": metrics["map_with_truth"],
                        "runtime": float(runtime),
                    }
                )
                print(
                    f"MF factors={factors} a={confidence_alpha:.0f} "
                    f"t={threshold:.1f}: "
                    f"own_pct={sanity_percentile:.4f} "
                    f"own_hit5={sanity_hit5:.4f} "
                    f"MAP all={metrics['map_all']:.6f} "
                    f"truth={metrics['map_with_truth']:.6f} "
                    f"runtime={runtime:.2f}s"
                )

    best = max(configs, key=lambda config: config["map_all"])
    total_runtime = time.perf_counter() - started
    results = {
        "method": "implicit_mf",
        "library": "implicit 0.7.3",
        "regularization": REGULARIZATION,
        "iterations": ITERATIONS,
        "seed": SEED,
        "num_threads": NUM_THREADS,
        "use_cg": False,
        "configs": configs,
        "best": best,
        "baselines": baselines,
        "total_runtime": total_runtime,
        "pool_size": int(len(pool)),
        "n_users": int(len(users)),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    args.notes.write_text(
        "\n".join(notes_section(ppr_results, results, baselines)),
        encoding="utf-8",
    )
    print(
        f"best MF: factors={best['factors']} a={best['alpha']:.0f} "
        f"t={best['t']:.1f} MAP all={best['map_all']:.6f} "
        f"truth={best['map_with_truth']:.6f}"
    )
    print(f"total runtime: {total_runtime:.2f} s")
    print(f"wrote {args.results}")
    print(f"wrote {args.notes}")


if __name__ == "__main__":
    main()
