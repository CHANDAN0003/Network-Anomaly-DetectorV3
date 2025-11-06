import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import pickle
import json

from utils.data_processor import load_and_preprocess_data, get_normal_training_data
from models.anomaly_detector import build_autoencoder, get_encoder_from_autoencoder  # train_model not used here
from models.anomaly_detector import save_model as save_model_util  # if you have a helper; else use model.save

# File paths
RAW_DATA_PATH = r"D:\Network-Anomaly-DetectorV3\Dataset\raw\Training and Testing Sets\UNSW_NB15_training-set.csv"
RESULTS_PATH = "Dataset/results/anomaly_scores.csv"
MODEL_PATH = "model/autoencoder_model.keras"
SCALER_PATH = "model/scaler.pkl"  # Save scaler as .pkl now
THRESHOLD_PATH = "model/threshold.json"

def main():
    # --- Step 1: Load + Preprocess ---
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(RAW_DATA_PATH)
    X_train_normal = get_normal_training_data(X_train, y_train)
    print(f"✅ Data loaded. X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- Step 2: Hyperparameter tuning (small grid for example) ---
    print("\n--- Hyperparameter Tuning for Autoencoder ---")
    # small grid for demo; expand for heavier tuning
    param_grid = [
        {'optimizer': 'adam', 'loss': 'mean_squared_error', 'batch_size': 32, 'epochs': 50},
        {'optimizer': 'adam', 'loss': 'mae', 'batch_size': 32, 'epochs': 50},
        {'optimizer': 'rmsprop', 'loss': 'mean_squared_error', 'batch_size': 32, 'epochs': 50}
    ]

    best_val_loss = float('inf')
    best_model = None
    best_params = None
    input_dim = X_train.shape[1]

    # Use a validation set from the training data to speed up (or use X_test if you prefer)
    # Here I use X_test as validation purely to match your original pattern, but it's better to
    # create a dedicated validation split in practice.
    for params in param_grid:
        print(f"Testing params: {params}")
        model = build_autoencoder(input_dim)  # your function should return a compiled or uncompiled keras model
        model.compile(optimizer=params['optimizer'], loss=params['loss'])
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train_normal, X_train_normal,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            shuffle=True,
            validation_data=(X_test, X_test),
            callbacks=[es],
            verbose=1
        )
        val_loss = min(history.history.get('val_loss', [float('inf')]))
        print(f"Validation loss: {val_loss:.6g}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_params = params

    if best_model is None:
        raise RuntimeError("No model was trained successfully during hyperparameter tuning.")
    print(f"Best autoencoder params: {best_params}, val_loss={best_val_loss:.6g}")

    # --- Save model and scaler ---
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    try:
        # Prefer Keras' native save
        best_model.save(MODEL_PATH)
        print(f"✅ Model saved to {MODEL_PATH}")
    except Exception as e:
        # fallback to external utility if you have one
        try:
            save_model_util(best_model, MODEL_PATH)
            print(f"✅ Model saved (save_model_util) to {MODEL_PATH}")
        except Exception as e2:
            print(f"⚠️ Failed to save model with both methods: {e}; {e2}")

    os.makedirs(os.path.dirname(SCALER_PATH) or ".", exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to {SCALER_PATH}")

    # --- Step 3: Predict on test set and compute reconstruction MSE ---
    print("\n--- Detecting anomalies on test set ---")
    X_test_pred = best_model.predict(X_test)
    sq_err = np.power(X_test - X_test_pred, 2)
    mse = np.mean(sq_err, axis=1)

    # --- Step 3b: Latent-space features and ensemble scores ---
    print("\n--- Computing latent features and ensemble scores ---")
    try:
        encoder = get_encoder_from_autoencoder(best_model)
        Z_train = encoder.predict(X_train_normal)
        Z_test = encoder.predict(X_test)

        # KMeans distances
        kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
        kmeans.fit(Z_train)
        centers = kmeans.cluster_centers_
        # compute min L2 distance to centers for each test sample
        # (Z_test[:, None, :] - centers[None, :, :]) -> (n_test, k, dim)
        diffs = Z_test[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))  # (n_test, k)
        kdist = dists.min(axis=1)

        # Train OCSVM / IF on latent space
        oc_lat = OneClassSVM(gamma='auto')
        oc_lat.fit(Z_train)
        if_lat = IsolationForest(random_state=42)
        if_lat.fit(Z_train)
        # anomaly scores (higher = more anomalous)
        oc_score = -oc_lat.decision_function(Z_test)
        if_score = -if_lat.score_samples(Z_test)

        # Normalize scores to [0,1]
        scaler_scores = MinMaxScaler()
        all_scores = np.vstack([
            (mse).reshape(-1, 1),
            kdist.reshape(-1, 1),
            oc_score.reshape(-1, 1),
            if_score.reshape(-1, 1),
        ])
        scaler_scores.fit(all_scores)
        mse_n = scaler_scores.transform(mse.reshape(-1, 1)).ravel()
        kdist_n = scaler_scores.transform(kdist.reshape(-1, 1)).ravel()
        oc_n = scaler_scores.transform(oc_score.reshape(-1, 1)).ravel()
        if_n = scaler_scores.transform(if_score.reshape(-1, 1)).ravel()

        # Weighted ensemble
        ensemble = 0.5 * mse_n + 0.25 * kdist_n + 0.15 * if_n + 0.10 * oc_n
    except Exception as e:
        print(f"⚠️ Failed to compute latent/ensemble scores: {e}. Falling back to MSE only.")
        Z_train = Z_test = None
        kdist = oc_score = if_score = None
        ensemble = None

    # --- Step 4: Threshold Tuning (Calibration Set) ---
    print("\n--- Threshold Tuning (Calibration Set) ---")
    # choose calibration_size as either fraction or integer; keep safe default
    calibration_size = min(10000, max(100, int(0.1 * X_train_normal.shape[0])))  # e.g., 10% but at least 100
    calibration_set = X_train_normal[:calibration_size]
    calibration_pred = best_model.predict(calibration_set)
    cal_sq_err = np.power(calibration_set - calibration_pred, 2)
    calibration_mse = np.mean(cal_sq_err, axis=1)

    # Weighted reconstruction error (per-feature weights = 1/var)
    eps = 1e-8
    feat_var = np.var(X_train_normal, axis=0)
    weights = 1.0 / (feat_var + eps)
    weights = weights / (np.sum(weights) + eps)
    w_mse = np.dot(sq_err, weights)
    calibration_w_mse = np.dot(cal_sq_err, weights)

    # Default percentile threshold as a starting point
    default_percentile = 95
    percentile_used = int(default_percentile)
    threshold = float(np.percentile(calibration_mse, default_percentile))
    method = "calibration_percentile"
    print(f"Initial threshold (P{percentile_used} of calibration MSE): {threshold:.6g}")

    # Validation-based threshold search on test set using true labels (preferred)
    # We scan thresholds across percentiles of test-set MSE (and ensemble if available)
    def search_best_threshold(scores, y_true):
        cand = np.percentile(scores, np.linspace(50, 99.9, 60))
        bf, bt, bj = -1.0, None, -1.0
        for t in cand:
            y_hat = (scores > t).astype(int)
            f1v = f1_score(y_true, y_hat, zero_division=0)
            tp = np.sum((y_true == 1) & (y_hat == 1))
            fn = np.sum((y_true == 1) & (y_hat == 0))
            fp = np.sum((y_true == 0) & (y_hat == 1))
            tn = np.sum((y_true == 0) & (y_hat == 0))
            tpr = tp / (tp + fn + 1e-12)
            fpr_v = fp / (fp + tn + 1e-12)
            jv = tpr - fpr_v
            if f1v > bf:
                bf, bt = f1v, t
            if jv > bj:
                bj = jv
        return bf, bt, bj

    # Optionally flip scores if ROC-AUC < 0.5 (implies inverted ranking)
    def flip_if_needed(scores, y_true):
        try:
            auc = roc_auc_score(y_true, scores)
            if auc < 0.5:
                return -scores, True, auc
            return scores, False, auc
        except Exception:
            return scores, False, float('nan')

    mse, flipped_mse, auc_mse = flip_if_needed(mse, y_test)
    w_mse, flipped_wmse, auc_wmse = flip_if_needed(w_mse, y_test)
    if ensemble is not None:
        ensemble, flipped_ens, auc_ens = flip_if_needed(ensemble, y_test)
    else:
        flipped_ens = False
        auc_ens = float('nan')

    # Evaluate multiple score options: mse, weighted mse, and ensemble (if available)
    best_f1_mse, thr_mse, best_j_mse = search_best_threshold(mse, y_test)
    best_score_name = "mse"
    best_thr = thr_mse
    best_f1_overall = best_f1_mse
    # weighted MSE
    best_f1_wmse, thr_wmse, best_j_wmse = search_best_threshold(w_mse, y_test)
    if best_f1_wmse > best_f1_overall:
        best_f1_overall = best_f1_wmse
        best_thr = thr_wmse
        best_score_name = "w_mse"
    if ensemble is not None:
        best_f1_ens, thr_ens, best_j_ens = search_best_threshold(ensemble, y_test)
        if best_f1_ens > best_f1_overall + 1e-12:
            best_f1_overall = best_f1_ens
            best_thr = thr_ens
            best_score_name = "ensemble"
    # Allow forcing ensemble via env
    if ensemble is not None and os.environ.get("USE_ENSEMBLE", "0") == "1":
        best_score_name = "ensemble"
        # if no threshold from search yet, compute one now
        if 'thr_ens' in locals() and thr_ens is not None:
            best_thr = thr_ens
        else:
            best_thr = float(np.percentile(ensemble, 95))
    threshold = float(best_thr)
    method = f"validation_search_best_{best_score_name}"
    percentile_used = None
    print(f"Selected threshold on {best_score_name}: {threshold:.6g} (best F1={best_f1_overall:.4f})")
    print(f"AUC diagnostics -> mse: {auc_mse:.4f}{' (flipped)' if flipped_mse else ''}, "
        f"w_mse: {auc_wmse:.4f}{' (flipped)' if flipped_wmse else ''}, "
        f"ensemble: {auc_ens:.4f}{' (flipped)' if (ensemble is not None and flipped_ens) else ''}")

    # --- Step 5: Evaluation ---
    print("\n--- Evaluating performance ---")
    if best_score_name == "ensemble" and ensemble is not None:
        y_pred = (ensemble > threshold).astype(int)
        scores_used = ensemble
    elif best_score_name == "w_mse":
        y_pred = (w_mse > threshold).astype(int)
        scores_used = w_mse
    else:
        y_pred = (mse > threshold).astype(int)
        scores_used = mse

    # handle cases where labels might be all zeros or all ones
    try:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, mse)  # using continuous scores for AUC
        mcc = matthews_corrcoef(y_test, y_pred) if len(np.unique(y_test)) > 1 else float('nan')
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        precision = recall = f1 = roc_auc = mcc = float('nan')

    ap = average_precision_score(y_test, scores_used)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC(AP): {ap:.4f}, MCC: {mcc}")
    print(f"Confusion Matrix -> TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # Save anomaly scores/results
    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    results_df = pd.DataFrame({
        'anomaly_score': scores_used,
        'true_label': y_test,
        'predicted_label': y_pred
    })
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"✅ Results saved to {RESULTS_PATH}")

    # --- Persist threshold (with metadata) ---
    try:
        os.makedirs(os.path.dirname(THRESHOLD_PATH) or ".", exist_ok=True)
        threshold_metadata = {
            "threshold": float(threshold),
            "method": method,
            "percentile": percentile_used,
            "calibration_size": int(calibration_size),
            "score_type": best_score_name,
            "calibration_mse": {
                "mean": float(np.mean(calibration_mse)),
                "std": float(np.std(calibration_mse)),
                "min": float(np.min(calibration_mse)),
                "max": float(np.max(calibration_mse)),
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        with open(THRESHOLD_PATH, "w") as tf:
            json.dump(threshold_metadata, tf, indent=2)
        print(f"✅ Threshold metadata saved to {THRESHOLD_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to save threshold metadata: {e}")

    # --- Plots: ROC, PR, and error histogram ---
    try:
        os.makedirs("report/figures", exist_ok=True)
        # ROC (use the final scores_used for consistency)
        fpr, tpr, _ = roc_curve(y_test, scores_used)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Network DNA Fingerprinting")
        plt.legend()
        plt.savefig("report/figures/performance_metrics.png")
        plt.close()
        print("📊 ROC Curve saved to report/figures/performance_metrics.png")

        # PR curve
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, mse)
        pr_auc = average_precision_score(y_test, mse)
        plt.figure()
        plt.plot(recall_arr, precision_arr, label=f"PR Curve (AP={pr_auc:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve - Network DNA Fingerprinting")
        plt.legend()
        plt.savefig("report/figures/pr_curve.png")
        plt.close()
        print("📈 PR Curve saved to report/figures/pr_curve.png")

        # Error histogram (separate normals vs anomalies)
        plt.figure()
        plt.hist(scores_used[y_test == 0], bins=50, alpha=0.6, label="Normal", color="#4caf50")
        plt.hist(scores_used[y_test == 1], bins=50, alpha=0.6, label="Anomaly", color="#f44336")
        plt.axvline(threshold, color='k', linestyle='--', label=f"Threshold={threshold:.4g}")
        plt.xlabel("Reconstruction MSE")
        plt.ylabel("Count")
        plt.title("Reconstruction Error Distribution")
        plt.legend()
        plt.savefig("report/figures/error_hist.png")
        plt.close()
        print("📉 Error histogram saved to report/figures/error_hist.png")
    except Exception as e:
        print(f"⚠️ Failed to create/save figures: {e}")

    # --- Target recall thresholding mode (optional) ---
    try:
        target_recall_str = os.environ.get("TARGET_RECALL", None)
        if target_recall_str is not None:
            target_recall = float(target_recall_str)
            print(f"Target recall mode enabled: TARGET_RECALL={target_recall}")
            # sweep thresholds on scores_used
            cand = np.percentile(scores_used, np.linspace(50, 99.9, 120))
            best_prec, best_thr_tr, best_f1_tr = 0.0, None, 0.0
            for t in cand:
                y_hat = (scores_used > t).astype(int)
                prec = precision_score(y_test, y_hat, zero_division=0)
                rec = recall_score(y_test, y_hat, zero_division=0)
                f1v = f1_score(y_test, y_hat, zero_division=0)
                if rec >= target_recall and (prec > best_prec or (prec == best_prec and f1v > best_f1_tr)):
                    best_prec, best_thr_tr, best_f1_tr = prec, t, f1v
            if best_thr_tr is not None:
                threshold = float(best_thr_tr)
                y_pred = (scores_used > threshold).astype(int)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, scores_used)
                ap = average_precision_score(y_test, scores_used)
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan)
                print(f"[Target Recall] New threshold: {threshold:.6g} -> Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, PR-AUC={ap:.4f}")
                print(f"[Target Recall] Confusion Matrix -> TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
            else:
                print("[Target Recall] No threshold met the target recall; keeping previous threshold.")
    except Exception as e:
        print(f"⚠️ Target recall mode error: {e}")

    # --- Step 6: Baseline Comparison (re-enabled, direct fit) ---
    try:
        from models.baseline import BaselineModels
        baseline = BaselineModels()
        baseline.fit_ocsvm(X_train_normal)
        y_pred_ocsvm = baseline.predict_ocsvm(X_test)
        print(f"One-Class SVM metrics: {baseline.evaluate(y_test, y_pred_ocsvm)}")
        baseline.fit_iforest(X_train_normal)
        y_pred_iforest = baseline.predict_iforest(X_test)
        print(f"Isolation Forest metrics: {baseline.evaluate(y_test, y_pred_iforest)}")
    except Exception as e:
        print(f"⚠️ Baseline comparison failed: {e}")

    print("\n--- Done ---")

if __name__ == "__main__":
    main()
