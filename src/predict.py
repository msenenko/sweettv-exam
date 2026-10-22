from __future__ import annotations

import argparse
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
    from .ppr import (
        MAX_ITERATIONS,
        TOLERANCE,
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
    from ppr import (
        MAX_ITERATIONS,
        TOLERANCE,
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


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    started = time.perf_counter()
    interactions = load_interactions(args.interactions)
    users = load_submission_users(args.submission_template)
    if len(users) != 1260:
        raise ValueError(f"expected 1,260 submission users, got {len(users)}")
    pool = load_schedule_pool(
        args.data_dir, TEST_WEEKS[0], TEST_WEEKS[1]
    )
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
    lines = ["user_id,tv_show_id"]
    for user in users:
        ranking = predictions[int(user)]
        lines.append(
            f"{int(user)},{' '.join(str(int(show)) for show in ranking)}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


if __name__ == "__main__":
    main()
