from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .evaluate import (
        K,
        TEST_WEEKS,
        fill_ranking,
        load_interactions,
        load_schedule_pool,
        load_submission_users,
        popularity_order,
    )
    from .hybrid import (
        build_components as build_hybrid_components,
        rank_scores as rank_hybrid_scores,
        score_components,
    )
    from .ppr import (
        build_graph,
        ppr,
        rank_pool,
        transition_blocks,
    )
except ImportError:
    from evaluate import (
        K,
        TEST_WEEKS,
        fill_ranking,
        load_interactions,
        load_schedule_pool,
        load_submission_users,
        popularity_order,
    )
    from hybrid import (
        build_components as build_hybrid_components,
        rank_scores as rank_hybrid_scores,
        score_components,
    )
    from ppr import (
        build_graph,
        ppr,
        rank_pool,
        transition_blocks,
    )

REFIT_WEEKS = (11, 30)
ALPHA = 0.7
THRESHOLD = 0.3
VAL_MAP = 0.571957
DEFAULT_NAME = "ppr_a0.7_t0.3_val0.5720"


def validate_submission(
    output_path: Path,
    template_path: Path,
    pool: np.ndarray,
) -> None:
    template = pd.read_csv(template_path, usecols=["user_id"])
    output = pd.read_csv(
        output_path,
        dtype={"user_id": "uint64", "tv_show_id": str},
    )
    if list(output.columns) != ["user_id", "tv_show_id"]:
        raise ValueError(f"unexpected columns: {list(output.columns)}")
    if len(output) != len(template):
        raise ValueError(
            f"row count mismatch: {len(output)} != {len(template)}"
        )
    if output["user_id"].duplicated().any():
        raise ValueError("duplicate users in submission")
    if output["user_id"].tolist() != template["user_id"].tolist():
        raise ValueError("users are not in template order")
    if output["tv_show_id"].isna().any():
        raise ValueError("missing tv_show_id values")
    pool_set = {int(show) for show in pool}
    for row_number, value in enumerate(output["tv_show_id"], start=1):
        parts = str(value).split()
        if len(parts) != K:
            raise ValueError(
                f"row {row_number} does not have {K} ids: {value!r}"
            )
        ids = [int(part) for part in parts]
        if len(set(ids)) != K:
            raise ValueError(
                f"row {row_number} has duplicate ids: {value!r}"
            )
        if any(show not in pool_set for show in ids):
            raise ValueError(
                f"row {row_number} has ids outside test pool: {value!r}"
            )


def write_submission(
    output_path: Path,
    users: np.ndarray,
    predictions: dict[int, list[int]],
) -> None:
    lines = ["user_id,tv_show_id"]
    for user in users:
        ranking = predictions[int(user)]
        lines.append(
            f"{int(user)},{' '.join(str(int(show)) for show in ranking)}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def weight_tag(value: float) -> str:
    return f"{value:g}"


def run_ppr(
    interactions,
    users: np.ndarray,
    pool: np.ndarray,
    args,
    started: float,
) -> None:
    train = interactions.loc[
        interactions["iso_week"].between(*REFIT_WEEKS)
        & (interactions["frac"] >= args.threshold)
    ]
    matrix, user_ids, show_ids = build_graph(train)
    user_to_show, show_to_user = transition_blocks(matrix)
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
    show_mass, iterations, final_l1, max_mass_error = ppr(
        user_to_show, show_to_user, target_user_index, args.alpha
    )
    rankings = rank_pool(show_mass, pool, pool_to_graph, fallback)
    predictions = {int(user): fallback for user in users}
    for user, ranking in zip(target_users, rankings):
        predictions[int(user)] = ranking
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.name}.csv"
    write_submission(output_path, users, predictions)
    validate_submission(output_path, args.submission_template, pool)
    runtime = time.perf_counter() - started
    print(
        f"wrote {output_path} "
        f"({len(users):,} users, pool={len(pool):,}, "
        f"graph_users={len(user_ids):,}, graph_shows={len(show_ids):,}, "
        f"edges={matrix.nnz:,})"
    )
    print(
        f"PPR alpha={args.alpha:.2f} t={args.threshold:.1f}: "
        f"iterations={iterations} final_L1={final_l1:.3e} "
        f"max_mass_error={max_mass_error:.3e} "
        f"active_users={int(active_targets.sum()):,} "
        f"fallback_users={int((~active_targets).sum()):,} "
        f"validation_MAP={VAL_MAP:.6f} runtime={runtime:.2f}s"
    )
    print("submission validation passed")


def run_hybrid(
    interactions,
    users: np.ndarray,
    pool: np.ndarray,
    args,
    started: float,
) -> None:
    validation = json.loads(
        args.hybrid_results.read_text(encoding="utf-8")
    )
    if getattr(args, "weights", None):
        weights = json.loads(args.weights)
        validation_map = float(getattr(args, "validation_map", 0.0))
    else:
        chosen = validation["chosen"]
        weights = {
            name: float(chosen[name])
            for name in ("P", "M", "G", "A", "C")
        }
        validation_map = float(chosen["map_all"])
    normalization = validation.get("best", {}).get("normalization", "alternative")
    if getattr(args, "weights", None):
        best_overall = json.loads(args.weights)
        normalization = best_overall.get("normalization", normalization)
    elif weights == {
        name: float(validation["best"][name])
        for name in ("P", "M", "G", "A", "C")
    }:
        normalization = validation["best"].get("normalization", normalization)
    elif weights == {
        name: float(validation["chosen"][name])
        for name in ("P", "M", "G", "A", "C")
    }:
        normalization = validation["chosen"].get("normalization", normalization)
    components, metadata = build_hybrid_components(
        interactions,
        users,
        pool,
        args.data_dir,
        REFIT_WEEKS,
        TEST_WEEKS,
        normalization=normalization,
    )
    scores = score_components(components, weights)
    rankings = rank_hybrid_scores(
        scores, pool, metadata["fallback"]
    )
    predictions = {
        int(user): ranking
        for user, ranking in zip(users, rankings)
    }
    if args.name is None:
        name = (
            f"hybrid_wp{weight_tag(weights['P'])}"
            f"_wm{weight_tag(weights['M'])}"
            f"_wg{weight_tag(weights['G'])}"
            f"_wa{weight_tag(weights['A'])}"
            f"_wc{weight_tag(weights['C'])}"
            f"_val{validation_map:.4f}"
        )
    else:
        name = args.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{name}.csv"
    write_submission(output_path, users, predictions)
    validate_submission(output_path, args.submission_template, pool)
    runtime = time.perf_counter() - started
    ppr_metadata = metadata["ppr"]
    mf_metadata = metadata["mf"]
    print(
        f"wrote {output_path} "
        f"({len(users):,} users, pool={len(pool):,}, "
        f"graph_users={ppr_metadata['graph_users']:,}, "
        f"graph_shows={ppr_metadata['graph_shows']:,}, "
        f"edges={ppr_metadata['edges']:,}, "
        f"schedule_rows={metadata['schedule_rows']:,})"
    )
    print(
        f"hybrid wP={weights['P']:.1f} wM={weights['M']:.1f} "
        f"wG={weights['G']:.1f} wA={weights['A']:.1f} "
        f"wC={weights['C']:.1f}: "
        f"PPR_iters={ppr_metadata['iterations']} "
        f"PPR_L1={ppr_metadata['final_l1']:.3e} "
        f"MF_shows={mf_metadata['train_shows']:,} "
        f"active_users={ppr_metadata['active_target_users']:,} "
        f"fallback_users={ppr_metadata['inactive_target_users']:,} "
        f"validation_MAP={validation_map:.6f} runtime={runtime:.2f}s"
    )
    print("submission validation passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("ppr", "hybrid"), default="ppr"
    )
    parser.add_argument(
        "--interactions", type=Path, default=Path("data/interactions.pkl")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--submission-template",
        type=Path,
        default=Path("data/submission.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("submissions")
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--hybrid-results",
        type=Path,
        default=Path("data/hybrid_results.json"),
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--validation-map", type=float, default=None)
    args = parser.parse_args()
    if args.method == "ppr" and args.name is None:
        args.name = DEFAULT_NAME

    started = time.perf_counter()
    interactions = load_interactions(args.interactions)
    users = load_submission_users(args.submission_template)
    if len(users) != 1260:
        raise ValueError(f"expected 1,260 submission users, got {len(users)}")
    pool = load_schedule_pool(
        args.data_dir, TEST_WEEKS[0], TEST_WEEKS[1]
    )
    if args.method == "ppr":
        run_ppr(interactions, users, pool, args, started)
    else:
        run_hybrid(interactions, users, pool, args, started)


if __name__ == "__main__":
    main()
