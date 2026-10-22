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
    from .hybrid import (
        build_components,
        score_components,
        rank_scores,
        evaluate_rankings,
        config_label,
        simplicity,
        load_schedule_details,
        channel_affinity_component,
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
    from hybrid import (
        build_components,
        score_components,
        rank_scores,
        evaluate_rankings,
        config_label,
        simplicity,
        load_schedule_details,
        channel_affinity_component,
    )
