# Program Flow — Network Anomaly Detector (Network DNA Fingerprinting)

This document describes the high-level program flow, I/O contract, and internal steps for the code in `src/` (as of Oct 2025). Use this as a quick reference for developers and operators.

## Overview
The pipeline processes raw flow CSVs into numeric "pseudo-DNA" vectors, trains an autoencoder (optionally LSTM), tunes an anomaly threshold using a calibration set, evaluates on a test set, saves model/scaler/artifacts, and supports a live detector that ingests packets, preprocesses them, and scores them in real time.

## High-level steps

1. Preprocessing
   - Entry point: `src/main.py` calls `utils.data_processor.load_and_preprocess_data(RAW_DATA_PATH)`
   - Input: CSV path (set via `RAW_DATA_PATH` in `main.py`). Example: `D:\Network-Anomaly-DetectorV3\Dataset\raw\Training and Testing Sets\UNSW_NB15_training-set.csv`.
   - Output: (X_train, X_test, y_train, y_test, scaler)
     - `X_train` / `X_test`: numpy arrays (or pandas-convertible arrays) of preprocessed features (scaled, encoded).
     - `y_train` / `y_test`: ground-truth labels used only for evaluation.
     - `scaler`: fitted scaler object used to encode live data similarly.
   - Additional helper: `utils.data_processor.get_normal_training_data(X_train, y_train)` returns only the normal samples from the training set; used for unsupervised training.

2. Model construction and training
   - Build model: `src/models/anomaly_detector.build_autoencoder(input_dim)` (dense sparse autoencoder) or `build_lstm_autoencoder(seq_len, n_features)` for sequences.
   - Train: `train_model(model, X_train_normal, X_test)` trains the autoencoder with training-on-normal and validation on `X_test`.
   - Save model: `save_model(autoencoder, MODEL_PATH)` writes model to `model/autoencoder_model.keras`.
   - Save scaler: Pickle-dump `scaler` to `model/scaler.pkl`.

3. Anomaly scoring
   - After training the autoencoder, obtain reconstructions: `reconstructed = autoencoder.predict(X_test)`
   - Compute anomaly score per sample: MSE between input and reconstruction, i.e. `mse = mean((X_test - reconstructed)**2, axis=1)`.

4. Threshold tuning (calibration)
   - A calibration set is taken from normal training samples: first N normal samples (N = min(1000, available)).
   - Default behavior (safer): set the threshold to a calibration percentile (95th percentile by default) of the calibration MSE distribution. This is robust when the calibration set contains only normal examples.
   - Optional F1-based search: for research/experimentation you can enable a search that scans percentiles (90–99) and selects the threshold that maximizes an F1-style objective by setting the environment variable `USE_F1_CALIBRATION=1`. This is not the default because calibration sets without positives produce degenerate F1 values.
   - Final `threshold` is used to convert continuous anomaly scores into binary predictions: `predicted_label = (mse > threshold).astype(int)`.

5. Evaluation and reporting
   - Compute metrics using `y_test` and `predicted_label`: precision, recall, F1-score, ROC-AUC.
   - Save results to CSV: `Dataset/results/anomaly_scores.csv` with columns: `anomaly_score`, `true_label`, `predicted_label`.
   - Plot ROC curve and save: `report/figures/performance_metrics.png`.

6. Baseline comparisons
   - The `src/models/baseline.py` (imported in `main.py`) provides wrapper methods for One-Class SVM and Isolation Forest.
   - The pipeline fits baselines on `X_train_normal`, predicts on `X_test`, and prints/compares metrics.

7. Live detection (runtime)
   - Entry point: `src/live_detector.py`.
   - It loads the pre-trained model and scaler (paths in `live_detector.py`: `MODEL_PATH`, `SCALER_PATH` — currently pointing to `D:/Network-Anomaly-DetectorV2/...`).
   - For each sniffed packet (via `scapy.sniff`), `process_packet(packet)` builds a feature dict matching the training `FEATURE_NAMES`, constructs a DataFrame row, enforces column order, applies `scaler.transform`, runs `autoencoder.predict`, computes MSE, and compares to `ANOMALY_THRESHOLD`. It prints alerts when a packet's anomaly score exceeds the threshold.

## I/O contracts and important shapes
- load_and_preprocess_data(raw_csv_path) -> (X_train, X_test, y_train, y_test, scaler)
  - X_* shapes: (n_samples, n_features)
  - scaler: fitted scikit-learn scaler object with `feature_names_in_` attribute used to align live features

- get_normal_training_data(X_train, y_train) -> X_train_normal (only rows where label==0)

- build_autoencoder(input_dim) -> compiled Keras Model accepting arrays with shape (n_samples, input_dim)

- train_model(model, X_train_normal, X_test, epochs=50, batch_size=32) -> History object

- save_model(model, path) -> writes Keras model to disk

- live_detector: expects `scaler.feature_names_in_` to hold trained feature column names. The live DataFrame must be ordered to match that.

## Design notes and potential pitfalls
- File paths: `main.py` and `live_detector.py` currently reference absolute paths (e.g., `D:\Network-Anomaly-DetectorV2\...` and `D:/Network-Anomaly-DetectorV2/...`). For portability, change them to relative paths or accept CLI args / environment variables.
- Scaler serialization: `main.py` pickles the scaler to `model/scaler.pkl`, while `live_detector.py` loads via `joblib.load(SCALER_PATH)` and expects a scikit-learn StandardScaler. Ensure the serializer/loader match (pickle vs joblib both work but keep consistent file path and type).
-- Thresholds: `main.py` computes and prints an optimal threshold. The pipeline now persists the chosen threshold to `model/threshold.json` so `live_detector.py` can load the same calibrated value instead of relying on a hard-coded `ANOMALY_THRESHOLD`.
- LSTM vs dense: If using `build_lstm_autoencoder`, preprocessing must return sequences of fixed `seq_len` and shapes (n_samples, seq_len, n_features). The live detector will need batching logic to build sequences from consecutive packets or flows.

## Recommended improvements (next steps)
1. Make file paths configurable (CLI flags or environment variables). Example: add argparse to `src/main.py` and `src/live_detector.py`.
2. Persist the chosen `threshold` after calibration to `model/threshold.json` and have `live_detector.py` read it. (Implemented.)
   - New behavior: `src/main.py` writes rich metadata to `model/threshold.json` with fields: `threshold`, `method` (e.g., `calibration_percentile` or `f1_calibration`), `percentile` (if used), `calibration_size`, `calibration_mse` statistics (mean/std/min/max), and `timestamp`.
   - `src/live_detector.py` reads `model/threshold.json` and uses the persisted `threshold` by default; it falls back to a configured default only if the file is missing or malformed.
3. Use a consistent serialization method for scaler (prefer `joblib.dump/load`) and update `main.py` to use joblib instead of pickle for clarity.
4. Add a small `scripts/run.bat` to create venv, install dependencies, and run training + evaluation.
5. Add unit tests for `utils.data_processor` and for evaluation metrics.

---

If you'd like, I can (pick one):
- implement the threshold persistence and update `live_detector.py` to read it, or
- switch `main.py` to use `joblib.dump` for the scaler and update code accordingly, or
- create `PROGRAM_FLOW.md` improvements into `docs/` and a `scripts/run.bat` to automate runs.
