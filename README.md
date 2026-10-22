# sweet.tv TV program recommender

## Environment
- Python 3.13.5
- pandas 3.1.0, numpy 2.5.3, scipy 1.18.1, implicit 0.7.3, scikit-learn 1.7.2, tqdm, kaggle 2.2.4 (macOS arm64)

## Data
Place original competition data in `data/`. Download from Kaggle:
```bash
kaggle competitions download -c sweettv-tv-program-recommender -p data
unzip -o data/sweettv-tv-program-recommender.zip -d data
```
`data/` is gitignored.

## Pipeline
- `src/interactions.py` - build user-item interactions from logs and schedules
- `src/baselines.py` - popularity and user-history baselines
- `src/ppr.py` - Personalized PageRank
- `src/mf.py` - implicit ALS MF
- `src/cbf.py` - content-based filtering
- `src/predict.py`, `src/hybrid.py` - prediction and hybrid fusion
- `src/evaluate.py` - validation protocol, MAP@5, ranking helpers

## Validation
Fit on ISO weeks 11-20, validate on 21-30. Truth = user-show with `frac >= 0.8`. Pool restricted to shows airing in validation weeks (6,361). MAP@5 per competition metric; fill ranking to 5 by popularity order when needed.

## Designated final submission
**Final file:** `submissions/hybrid_exp2_wp0_wm0.5_wg2_wa0_wc0_we3_rfrac_val0.5930.csv`  
**Validation MAP@5:** 0.592977  
**Kaggle:** public 0.44283, private 0.42824, ref 57539850, team `Сененко Мария & mariasenenko`

### Exact command sequence to regenerate from raw data
From repo root:
1. `PYTHONPATH=src python3 src/interactions.py` (build `data/interactions.pkl`)
2. `PYTHONPATH=src python3 scripts/predict_step12.py` (builds components from weeks 11-30: R = log1p(sum of frac, frac>=0.3), E via channel exposure, hybrid wM=0.5,wG=2,wE=3; edit output path or rename to produce `submissions/hybrid_exp2_wp0_wm0.5_wg2_wa0_wc0_we3_rfrac_val0.5930.csv`)

## Steps and notes
- Step 1-2: `notes/step2_interactions.md`
- Step 3: `notes/step3_baselines.md`
- Step 4: `notes/step4_methods.md`
- Step 6: `notes/step6_hybrid.md`
- Step 7: `notes/step7_cbf.md`
- Step 8: `notes/step8_examples.md`, `notes/step8_examples_final.md`
- Step 10: `notes/step10_analysis.md`, `notes/step10_analysis_final.md`
- Step 11: `notes/step11_hybrid_exposure.md`
- Step 12: `notes/step12_r_sumfrac.md`
- Submissions: `notes/submissions.md`
