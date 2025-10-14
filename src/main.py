import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
import pickle  # Use pickle instead of joblib

from utils.data_processor import load_and_preprocess_data, get_normal_training_data
from models.anomaly_detector import build_autoencoder, train_model, save_model

# File paths
RAW_DATA_PATH = r"D:\Network-Anomaly-DetectorV2\Dataset\raw\Training and Testing Sets\UNSW_NB15_training-set.csv"
RESULTS_PATH = "Dataset/results/anomaly_scores.csv"
MODEL_PATH = "model/autoencoder_model.keras"
SCALER_PATH = "model/scaler.pkl"  # Save scaler as .pkl now

def main():
    # --- Step 1: Load + Preprocess ---
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(RAW_DATA_PATH)
    X_train_normal = get_normal_training_data(X_train, y_train)
    print(f"✅ Pseudo-DNA sequences generated: shape {X_train.shape}")


    # --- Step 2: Model ---
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)
    train_model(autoencoder, X_train_normal, X_test)

    # Save the trained model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model(autoencoder, MODEL_PATH)

    # Save scaler using pickle
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to {SCALER_PATH}")

    # --- Step 3: Anomaly Detection ---
    print("\n--- Detecting anomalies ---")
    X_test_predictions = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - X_test_predictions, 2), axis=1)

    # --- Step 4: Threshold Tuning (Calibration Set) ---
    print("\n--- Threshold Tuning (Calibration Set) ---")
    # Use a subset of normal traffic as calibration set
    calibration_size = min(1000, X_train_normal.shape[0])
    calibration_set = X_train_normal[:calibration_size]
    calibration_pred = autoencoder.predict(calibration_set)
    calibration_mse = np.mean(np.power(calibration_set - calibration_pred, 2), axis=1)
    # Find threshold that maximizes F1 or achieves target FPR
    best_f1, best_thresh = 0, None
    for t in np.percentile(calibration_mse, np.linspace(90, 99, 20)):
        y_cal_pred = (calibration_mse > t).astype(int)
        f1 = f1_score(np.zeros_like(y_cal_pred), y_cal_pred)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    threshold = best_thresh if best_thresh is not None else np.percentile(mse, 95)
    print(f"Optimal threshold (calibration): {threshold:.4f} (F1={best_f1:.4f})")

    # --- Step 5: Evaluation ---
    print("\n--- Evaluating performance ---")
    y_pred = (mse > threshold).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, mse)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Save anomaly scores
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df = pd.DataFrame({
        'anomaly_score': mse,
        'true_label': y_test,
        'predicted_label': y_pred
    })
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"✅ Results saved to {RESULTS_PATH}")

    # Optional: Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, mse)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Network DNA Fingerprinting")
    plt.legend()
    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/performance_metrics.png")
    print("📊 ROC Curve saved to report/figures/performance_metrics.png")

    # --- Step 6: Baseline Comparison ---
    print("\n--- Baseline Model Comparison ---")
    from models.baseline import BaselineModels
    baseline = BaselineModels()
    baseline.fit_ocsvm(X_train_normal)
    y_pred_ocsvm = baseline.predict_ocsvm(X_test)
    ocsvm_metrics = baseline.evaluate(y_test, y_pred_ocsvm)
    print(f"One-Class SVM: {ocsvm_metrics}")
    baseline.fit_iforest(X_train_normal)
    y_pred_iforest = baseline.predict_iforest(X_test)
    iforest_metrics = baseline.evaluate(y_test, y_pred_iforest)
    print(f"Isolation Forest: {iforest_metrics}")

if __name__ == "__main__":
    main()
