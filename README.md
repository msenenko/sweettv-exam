# sweet.tv exam

Reproduction of the HSE LLM4Rec / sweet.tv TV program recommender steps (Kaggle: `sweettv-tv-program-recommender`, metric MAP@5).

## Environment (as used)
- OS: macOS arm64, Python 3.13.5
- Packages: pandas 3.1.0, numpy 2.5.3, scipy 1.18.1, implicit 0.7.3, Kaggle CLI 2.2.4

## Data (not included)
Download competition data with Kaggle CLI into `data/`:
```bash
mkdir -p data
kaggle competitions download -c sweettv-tv-program-recommender -p data
unzip -o data/sweettv-tv-program-recommender.zip -d data
```
Expected files: `submission.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv`, `dataset11-30.csv`. Tokens/credentials must not be committed.

## Reproduction order
1. `python3 src/interactions.py`  (builds `data/interactions.pkl`) - Step 2
2. `python3 src/baselines.py`     (validation baselines) - Step 3
3. `python3 src/ppr.py`          (PPR grid) - Step 4
4. `python3 src/mf.py`           (implicit ALS grid) - Step 4
5. `python3 src/predict.py`      (PPR test submission generator) - Step 5
6. `python3 src/hybrid.py`       (hybrid grid search) - Step 6
7. `python3 src/predict.py --method hybrid` (hybrid test generator) - Step 6
8. `python3 src/cbf.py` style tests via `scripts/cbf_hybrid_eval.py` - Step 7
9. `python3 scripts/examples.py` - Step 8

Seeds: MF uses `random_state=42`, `num_threads=1`. PPR is deterministic.

## Notes
All results are in `notes/`. Scripts in `src/` and `scripts/`. Submissions in `submissions/`.

## Designated final submission
Final config: R = log1p(sum of frac), wM=0.5, wG=2, wE=3, others 0, alternative normalization. Test file: `submissions/hybrid_exp2_wp0_wm0.5_wg2_wa0_wc0_we3_rfrac_val0.5930.csv`. Reproduce: `python3 scripts/predict_step12.py` (refits on weeks 11-30, target window 31-42).
