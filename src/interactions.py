from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCHEDULE_SPECS = (
    ("export_arh_11-20-final.csv", 11, 20),
    ("export_arh_21-30-final.csv", 21, 30),
    ("export_arh_31-42-final.csv", 31, 42),
)
VIEW_FILE = "dataset11-30.csv"
SUBMISSION_FILE = "submission.csv"
PERCENTILES = (10, 25, 50, 75, 80, 90)
FRAC_CUTS = (0.3, 0.5, 0.8)
MAX_CANDIDATES = 4_000_000


def peak_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024.0 ** 2)
    return value / 1024.0


def read_views(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    frame = pd.read_csv(data_dir / VIEW_FILE)
    raw_rows = len(frame)
    frame = frame.drop_duplicates()
    duplicate_rows = raw_rows - len(frame)
    start = pd.to_datetime(frame["start_time"], format="%Y-%m-%d %H:%M:%S")
    stop = pd.to_datetime(frame["stop_time"], format="%Y-%m-%d %H:%M:%S")
    if start.isna().any() or stop.isna().any():
        raise ValueError("unparsed view timestamps")
    if (stop <= start).any():
        raise ValueError("non-positive view intervals")
    return (
        frame["user_id"].to_numpy(),
        frame["vsetv_id"].to_numpy(),
        start.to_numpy(),
        stop.to_numpy(),
        raw_rows,
        duplicate_rows,
    )


def read_schedules(
    data_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    parts: list[pd.DataFrame] = []
    stats: list[dict[str, int | str]] = []
    for filename, week_lo, week_hi in SCHEDULE_SPECS:
        frame = pd.read_csv(
            data_dir / filename,
            usecols=["channel_id", "start_time", "duration", "tv_show_id"],
        )
        loaded_rows = len(frame)
        frame["airing_start"] = pd.to_datetime(
            frame["start_time"], format="%d.%m.%Y %H:%M:%S"
        )
        if frame["airing_start"].isna().any():
            raise ValueError(f"unparsed schedule timestamps in {filename}")
        week = frame["airing_start"].dt.isocalendar().week.astype(np.int16)
        week_mask = week.between(week_lo, week_hi)
        week_rows = int(week_mask.sum())
        frame = frame.loc[week_mask].copy()
        frame["iso_week"] = week.loc[week_mask].to_numpy()
        nonzero_mask = frame["tv_show_id"].ne(0)
        item_rows = int(nonzero_mask.sum())
        zero_rows = week_rows - item_rows
        frame = frame.loc[
            nonzero_mask,
            ["channel_id", "tv_show_id", "airing_start", "duration", "iso_week"],
        ].copy()
        parts.append(frame)
        stats.append(
            {
                "file": filename,
                "week_lo": week_lo,
                "week_hi": week_hi,
                "loaded_rows": loaded_rows,
                "outside_week_rows": loaded_rows - week_rows,
                "week_rows": week_rows,
                "zero_show_rows": zero_rows,
                "item_rows": item_rows,
            }
        )
    schedule = pd.concat(parts, ignore_index=True)
    if schedule.duplicated(["channel_id", "airing_start"]).any():
        raise ValueError("duplicate airing key (channel_id, airing_start)")
    schedule = schedule.sort_values(
        ["channel_id", "airing_start"], kind="mergesort", ignore_index=True
    )
    schedule["airing_end"] = schedule["airing_start"] + pd.to_timedelta(
        schedule["duration"], unit="s"
    )
    return schedule, stats


def build_interactions(
    user: np.ndarray,
    channel: np.ndarray,
    view_start: np.ndarray,
    view_stop: np.ndarray,
    schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    touched = np.zeros(len(user), dtype=bool)
    schedule_by_channel: dict[int, tuple[np.ndarray, ...]] = {}
    for channel_id, frame in schedule.groupby("channel_id", sort=False):
        schedule_by_channel[int(channel_id)] = (
            frame["airing_start"].to_numpy(),
            frame["airing_end"].to_numpy(),
            frame["duration"].to_numpy(),
            frame["tv_show_id"].to_numpy(),
            frame["iso_week"].to_numpy(),
        )

    order = np.argsort(channel, kind="stable")
    sorted_channels = channel[order]
    boundaries = np.flatnonzero(sorted_channels[1:] != sorted_channels[:-1]) + 1
    group_starts = np.r_[0, boundaries]
    group_ends = np.r_[boundaries, len(sorted_channels)]
    channel_results: list[pd.DataFrame] = []

    for group_start, group_end in zip(group_starts, group_ends):
        channel_id = int(sorted_channels[group_start])
        schedule_arrays = schedule_by_channel.get(channel_id)
        if schedule_arrays is None:
            continue
        s_start, s_end, s_duration, s_show, s_week = schedule_arrays
        max_duration = int(s_duration.max())
        view_indices = order[group_start:group_end]
        vs = view_start[view_indices]
        ve = view_stop[view_indices]
        lo = np.searchsorted(
            s_start, vs - np.timedelta64(max_duration, "s"), side="left"
        )
        hi = np.searchsorted(s_start, ve, side="left")
        counts = hi - lo
        if int(counts.sum()) == 0:
            continue
        cumulative = np.cumsum(counts)
        pieces: list[pd.DataFrame] = []
        p0 = 0
        while p0 < len(view_indices):
            base = int(cumulative[p0 - 1]) if p0 else 0
            p1 = int(
                np.searchsorted(
                    cumulative, base + MAX_CANDIDATES, side="right"
                )
            )
            if p1 <= p0:
                p1 = p0 + 1
            c = counts[p0:p1]
            candidate_total = int(c.sum())
            if candidate_total:
                vpos = np.repeat(np.arange(p0, p1, dtype=np.int64), c)
                starts = np.repeat(lo[p0:p1], c)
                cumulative_piece = np.cumsum(c)
                offsets = np.arange(candidate_total, dtype=np.int64) - np.repeat(
                    cumulative_piece - c, c
                )
                pidx = starts + offsets
                overlap = np.minimum(ve[vpos], s_end[pidx]) - np.maximum(
                    vs[vpos], s_start[pidx]
                )
                overlap_seconds = overlap.astype("timedelta64[s]").astype(np.int64)
                keep = overlap_seconds > 0
                if keep.any():
                    v_keep = vpos[keep]
                    p_keep = pidx[keep]
                    touched[view_indices[v_keep]] = True
                    pieces.append(
                        pd.DataFrame(
                            {
                                "user_id": user[view_indices[v_keep]],
                                "channel_id": np.full(
                                    v_keep.shape, channel_id, dtype=np.int64
                                ),
                                "tv_show_id": s_show[p_keep],
                                "airing_start": s_start[p_keep],
                                "duration": s_duration[p_keep],
                                "overlap": overlap_seconds[keep],
                                "iso_week": s_week[p_keep],
                            }
                        )
                    )
            p0 = p1
        if pieces:
            channel_frame = pd.concat(pieces, ignore_index=True)
            channel_frame = channel_frame.groupby(
                ["user_id", "airing_start"], as_index=False, sort=False
            ).agg(
                tv_show_id=("tv_show_id", "first"),
                channel_id=("channel_id", "first"),
                duration=("duration", "first"),
                iso_week=("iso_week", "first"),
                overlap=("overlap", "sum"),
            )
            channel_results.append(channel_frame)

    if not channel_results:
        raise ValueError("no interactions built")
    interactions = pd.concat(channel_results, ignore_index=True)
    interactions["frac"] = np.minimum(
        1.0,
        interactions["overlap"].to_numpy()
        / interactions["duration"].to_numpy(),
    )
    interactions = interactions[
        [
            "user_id",
            "channel_id",
            "tv_show_id",
            "airing_start",
            "duration",
            "overlap",
            "frac",
            "iso_week",
        ]
    ]
    interactions = interactions.sort_values(
        ["user_id", "airing_start", "channel_id"],
        ignore_index=True,
    )
    return interactions, int(touched.sum())


def percentile_dict(values: pd.Series | np.ndarray) -> dict[int, float]:
    array = np.asarray(values)
    return {p: float(np.percentile(array, p)) for p in PERCENTILES}


def write_notes(
    path: Path,
    *,
    data_dir: Path,
    interactions_path: Path,
    view_raw_rows: int,
    duplicate_rows: int,
    view_rows: int,
    schedule_stats: list[dict[str, int | str]],
    touched_views: int,
    runtime_seconds: float,
    peak_memory_mib: float,
) -> dict[str, object]:
    interactions = pd.read_pickle(interactions_path)
    n_rows = len(interactions)
    n_users = int(interactions["user_id"].nunique())
    n_shows = int(interactions["tv_show_id"].nunique())
    frac_percentiles = percentile_dict(interactions["frac"])
    duration_percentiles = percentile_dict(interactions["duration"])
    threshold_counts = {
        cutoff: int((interactions["frac"] >= cutoff).sum())
        for cutoff in FRAC_CUTS
    }
    overlap_300 = int((interactions["overlap"] >= 300).sum())
    touched_share = touched_views / view_rows
    submission = pd.read_csv(data_dir / SUBMISSION_FILE, usecols=["user_id"])
    submission_users = submission["user_id"].unique()
    submission_mask = interactions["user_id"].isin(submission_users)
    submission_with_any = int(
        interactions.loc[submission_mask, "user_id"].nunique()
    )
    high_frac_counts: dict[tuple[int, int], int] = {}
    for week_lo, week_hi in ((11, 20), (21, 30)):
        mask = (
            submission_mask
            & (interactions["frac"] >= 0.8)
            & interactions["iso_week"].between(week_lo, week_hi)
        )
        high_frac_counts[(week_lo, week_hi)] = int(
            interactions.loc[mask, "user_id"].nunique()
        )
    max_week = int(interactions["iso_week"].max())
    min_week = int(interactions["iso_week"].min())

    lines = [
        "# Step 2: user-programme interactions",
        "",
        "Generated by `python3 src/interactions.py`.",
        "",
        "## Protocol",
        "",
        f"- View log: `{VIEW_FILE}`. Dropped {duplicate_rows:,} exact duplicate view rows "
        f"({view_raw_rows:,} -> {view_rows:,}).",
        "- Each schedule file was restricted to its own ISO weeks before joining.",
        "- `tv_show_id = 0` airtime is not an item and was dropped from the join.",
        "- Join keys: `vsetv_id = channel_id`; overlap = "
        "`min(view_stop, programme_end) - max(view_start, programme_start) > 0`.",
        "- Programme end = schedule `start_time + duration` seconds.",
        "- Overlap was summed over all view rows of the same user on the same airing "
        "(airing = channel + programme start); "
        "`frac = min(1, summed_overlap / duration)`.",
        "- No model was fit.",
        "",
        "## Schedule filtering",
        "",
        "| file | ISO weeks | loaded | outside own weeks | in own weeks | tv_show_id=0 dropped | item rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in schedule_stats:
        lines.append(
            f"| `{item['file']}` | {item['week_lo']}-{item['week_hi']} | "
            f"{item['loaded_rows']:,} | {item['outside_week_rows']:,} | "
            f"{item['week_rows']:,} | {item['zero_show_rows']:,} | "
            f"{item['item_rows']:,} |"
        )
    lines += [
        "",
        "The `31-42` schedule contributes no interactions because the view log ends "
        "before that schedule starts.",
        "",
        "## Saved table",
        "",
        f"`{interactions_path}`",
        "",
        f"- User-airing rows: **{n_rows:,}**",
        f"- Distinct users: **{n_users:,}**",
        f"- Distinct shows: **{n_shows:,}**",
        f"- ISO week range: **{min_week}-{max_week}**",
        "",
        "## Fractions and durations",
        "",
        "| percentile | frac | duration, s |",
        "|---:|---:|---:|",
    ]
    for p in PERCENTILES:
        lines.append(
            f"| {p} | {frac_percentiles[p]:.6f} | "
            f"{duration_percentiles[p]:.1f} |"
        )
    lines += [
        "",
        "## Threshold counts",
        "",
        "| condition | rows | share of user-airing rows |",
        "|---|---:|---:|",
    ]
    for cutoff in FRAC_CUTS:
        count = threshold_counts[cutoff]
        lines.append(
            f"| `frac >= {cutoff}` | {count:,} | {count / n_rows:.4%} |"
        )
    lines.append(
        f"| `overlap >= 300 s` | {overlap_300:,} | {overlap_300 / n_rows:.4%} |"
    )
    lines += [
        "",
        "## View coverage",
        "",
        f"- View rows touching at least one labelled airing: **{touched_views:,} / "
        f"{view_rows:,} = {touched_share:.4%}**.",
        f"- View rows touching none: **{view_rows - touched_views:,}** "
        f"({1 - touched_share:.4%}).",
        "",
        "## Submission users",
        "",
        f"- Submission users with any interaction: **{submission_with_any:,} / "
        f"{len(submission_users):,}**.",
    ]
    for (week_lo, week_hi), count in high_frac_counts.items():
        lines.append(
            f"- Submission users with at least one interaction at "
            f"`frac >= 0.8` in weeks {week_lo}-{week_hi}: **{count:,} / "
            f"{len(submission_users):,} = {count / len(submission_users):.4%}**."
        )
    lines += [
        "",
        "## Runtime",
        "",
        f"- Runtime: **{runtime_seconds:.2f} s**",
        f"- Peak RSS: **{peak_memory_mib:.1f} MiB**",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "n_rows": n_rows,
        "n_users": n_users,
        "n_shows": n_shows,
        "frac_percentiles": frac_percentiles,
        "duration_percentiles": duration_percentiles,
        "threshold_counts": threshold_counts,
        "overlap_300": overlap_300,
        "touched_views": touched_views,
        "touched_share": touched_share,
        "submission_with_any": submission_with_any,
        "high_frac_counts": high_frac_counts,
        "min_week": min_week,
        "max_week": max_week,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/interactions.pkl")
    )
    parser.add_argument(
        "--notes", type=Path, default=Path("notes/step2_interactions.md")
    )
    args = parser.parse_args()

    started = time.perf_counter()
    user, channel, view_start, view_stop, view_raw_rows, duplicate_rows = read_views(
        args.data_dir
    )
    view_rows = len(user)
    schedule, schedule_stats = read_schedules(args.data_dir)
    interactions, touched_views = build_interactions(
        user, channel, view_start, view_stop, schedule
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_pickle(args.out)
    runtime_seconds = time.perf_counter() - started
    peak_memory_mib = peak_mib()
    summary = write_notes(
        args.notes,
        data_dir=args.data_dir,
        interactions_path=args.out,
        view_raw_rows=view_raw_rows,
        duplicate_rows=duplicate_rows,
        view_rows=view_rows,
        schedule_stats=schedule_stats,
        touched_views=touched_views,
        runtime_seconds=runtime_seconds,
        peak_memory_mib=peak_memory_mib,
    )

    print(f"view rows: {view_raw_rows:,} -> {view_rows:,} "
          f"(dropped {duplicate_rows:,} exact duplicates)")
    for item in schedule_stats:
        print(
            f"{item['file']}: loaded {item['loaded_rows']:,}, "
            f"outside weeks {item['outside_week_rows']:,}, "
            f"in weeks {item['week_rows']:,}, "
            f"tv_show_id=0 dropped {item['zero_show_rows']:,}, "
            f"items {item['item_rows']:,}"
        )
    print("tv_show_id=0 airtime is not an item and is dropped from the join")
    print(f"user-airing rows: {summary['n_rows']:,}")
    print(f"distinct users: {summary['n_users']:,}")
    print(f"distinct shows: {summary['n_shows']:,}")
    print(
        "frac percentiles: "
        + ", ".join(
            f"p{p}={summary['frac_percentiles'][p]:.6f}"
            for p in PERCENTILES
        )
    )
    print(
        "duration percentiles (s): "
        + ", ".join(
            f"p{p}={summary['duration_percentiles'][p]:.1f}"
            for p in PERCENTILES
        )
    )
    for cutoff, count in summary["threshold_counts"].items():
        print(f"frac >= {cutoff}: {count:,} rows")
    print(f"overlap >= 300 s: {summary['overlap_300']:,} rows")
    print(
        f"views touching at least one labelled airing: "
        f"{summary['touched_views']:,} / {view_rows:,} = "
        f"{summary['touched_share']:.4%}"
    )
    print(
        f"submission users with any interaction: "
        f"{summary['submission_with_any']:,}"
    )
    for (week_lo, week_hi), count in summary["high_frac_counts"].items():
        print(
            f"submission users with frac >= 0.8 in weeks "
            f"{week_lo}-{week_hi}: {count:,}"
        )
    print(f"runtime: {runtime_seconds:.2f} s")
    print(f"peak RSS: {peak_memory_mib:.1f} MiB")
    print(f"wrote {args.out}")
    print(f"wrote {args.notes}")


if __name__ == "__main__":
    main()
