# Project Procedure and Troubleshooting Log

Date: 2025-11-07

## Overview
This document summarizes how the project was started from scratch, key issues encountered, and the steps taken to address them. It also records design decisions around modeling, training, scoring, and evaluation to help future contributors reproduce and extend the workflow.

## Initial Setup
- Created Python virtual environment and installed core dependencies:
  - pandas, numpy, scikit-learn, matplotlib, seaborn
  - tensorflow, scapy, scikeras (later avoided for AE tuning)
- Organized repository structure under `src/`, `Dataset/`, `model/`, `report/`.
- Implemented data pipeline in `utils/data_processor.py` to:
  - Load CSV, drop unused columns, label encode categoricals
  - Scale numeric features (Standard/Robust/MinMax via env `SCALER_TYPE`)
  - Train/test split and persisted scaler as `model/scaler.pkl`

## Early Issues and Fixes
1. Import errors (e.g., `utils.preprocessing` not found)
   - Root cause: Module name mismatch. The correct file is `utils/data_processor.py`.
   - Fix: Update imports to `from utils.data_processor import ...`.

2. Missing functions (`preprocess_packet` in live detector)
   - Root cause: Function not implemented.
   - Fix: Removed/guarded usage until a clear packet schema and preprocessing logic are defined.

3. Hyperparameter tuning with `scikeras`/`GridSearchCV` failures
   - Symptoms: Errors related to sklearn tags (`__sklearn_tags__`) and incompatible wrappers.
   - Fix: Replaced with a manual hyperparameter search loop for Keras autoencoder (robust and simple).

4. Slow runtime
   - Fixes: Reduce parameter grid, enable EarlyStopping, and use a manageable validation size.

## Modeling Evolution
1. Baseline models
   - Implemented `OneClassSVM` and `IsolationForest` baselines in `models/baseline.py`.
   - These provide solid benchmarks; OCSVM performed best in initial runs.

2. Autoencoder (AE)
   - Started with a sparse AE (Dense 64→32→16 bottleneck→32→64→output).
   - Manual hyperparameter search over optimizer/loss/epochs.
   - Added EarlyStopping (patience=10, restore best weights).

3. Scoring and Thresholding
   - Reconstruction MSE as initial score.
   - Added weighted reconstruction error (per-feature inverse variance) to emphasize informative features.
   - Extracted latent features (encoder bottleneck) and built an ensemble score:
     - Components: normalized MSE, KMeans center distance (latent), IF and OCSVM scores (latent).
     - Ensemble weights initially: 0.5 (MSE), 0.25 (KMeans dist), 0.15 (IF), 0.10 (OCSVM).
   - Validation-based threshold search optimizing F1 and Youden’s J.
   - Added plots: ROC, Precision-Recall (AP), and error histogram.

4. Results progression
   - Initial AE: low recall and ROC-AUC ≈ 0.5.
   - After improvements: higher Precision/Recall/F1 and AP; however, OCSVM remained stronger overall.

## Chronological Timeline (Nov 6–7, 2025)
| Date/Phase | Action | Issue / Observation | Resolution | Metrics Snapshot |
|------------|--------|---------------------|------------|------------------|
| Nov 6 (Start) | Set up repo, data loader | Import error: `utils.preprocessing` | Corrected to `utils.data_processor` | N/A |
| Nov 6 | Live detector editing | Missing `preprocess_packet` function | Deferred until packet schema defined | N/A |
| Nov 6 | Tried scikeras GridSearch | `__sklearn_tags__` attribute errors | Replaced with manual loop tuning | N/A |
| Nov 6 | First AE training | Very low recall (<1%), ROC-AUC ≈0.50 | Identified threshold/feature mismatch | Precision ~0.25, Recall ~0.009 |
| Nov 7 (AM) | Added calibration threshold | Still poor ranking | Planned validation-based search | Slight F1 improvement |
| Nov 7 | Added validation threshold search | AE F1 improved but ROC-AUC ≈0.50 | Accepted need for more features | F1 ~0.62 |
| Nov 7 | Added latent features + ensemble | Ensemble logic working | OCSVM baseline superior | OCSVM F1 ~0.83 |
| Nov 7 | Weighted reconstruction error | Increased F1, AP good; ROC-AUC dipped | Added potential score flip guard plan | AE F1 ~0.71, AP ~0.85 |
| Nov 7 (Later) | Score flip & target recall mode | Potential inverted ranking | Implemented flip if AUC<0.5 & recall targeting | Prepped for higher recall tuning |

> Note: Numerical snapshots are approximate from console logs; final authoritative metrics are persisted in `Dataset/results/anomaly_scores.csv` and plots under `report/figures/`.

## Target Recall Mode
If `TARGET_RECALL` is set (e.g., `TARGET_RECALL=0.85`), the system sweeps thresholds to achieve at least that recall, then maximizes precision (tie-breaking with F1). If no threshold satisfies the target recall, the previously selected threshold is retained.

## Automatic Score Flip Guard
For each candidate score (MSE, weighted MSE, ensemble):
- Compute ROC-AUC
- If AUC < 0.5, multiply the score by -1 to restore intuitive ranking (higher = more anomalous)
- Print diagnostic lines indicating flips and AUC values

## Environment Variables Summary
| Variable | Purpose | Example |
|----------|---------|---------|
| `SCALER_TYPE` | Choose feature scaler | `standard`, `robust`, `minmax` |
| `USE_ENSEMBLE` | Force ensemble score selection | `1` |
| `TARGET_RECALL` | Enable recall-targeted thresholding | `0.85` |
| `USE_F1_CALIBRATION` | Legacy calibration mode (deprecated by validation search) | `1` |

## Current Status (Latest Run)
- AE (selected score type) shows improved F1 and AP; ROC-AUC still a weak general rank indicator.
- One-Class SVM remains best overall balanced metric performer (higher recall + F1).
- Isolation Forest useful for high precision filtering but lower recall.

## Recommendations Going Forward
1. Consider integrating Mahalanobis distance in latent space for refined anomaly scoring.
2. Evaluate EVT tail fitting on normal score distribution for dynamic thresholding.
3. Prototype VAE (already scaffolded) and compare latent likelihood vs reconstruction error.
4. Add a lightweight logistic regression stack over component scores (mse_n, kdist_n, if_n, oc_n) if labeled validation is sufficient.
5. Track metric drift over time if live data shifts (store per-run summaries).

## Key Learnings
- For imbalanced anomaly detection, PR-AUC is more informative than ROC-AUC.
- Threshold selection heavily influences operational metrics; validation search beats fixed percentiles.
- Latent-space models (OCSVM/IF) often outperform raw reconstruction error.
- `scikeras` wrappers can introduce compatibility issues; manual loops are simpler for Keras models.

## Current Best Practices
- Use OCSVM as a reliable baseline.
- Use ensemble score with validation threshold search for the AE pipeline.
- Persist chosen threshold and score type in `model/threshold.json`.
- Inspect PR curves and error histograms to diagnose separability.

## Next Steps (Optional)
- Add Mahalanobis distance in latent space.
- EVT tail modeling for thresholding.
- Variational Autoencoder (VAE) likelihood-based scoring.
- Logistic stacking of component scores.

## Repro Tips
- Ensure TensorFlow is installed in the active environment.
- Verify `FEATURE_NAMES` alignment with scaler columns at inference.
- Use `USE_ENSEMBLE=1` env var to force ensemble scoring.
- Use `TARGET_RECALL=0.85` (for example) to target a recall level in threshold selection.
