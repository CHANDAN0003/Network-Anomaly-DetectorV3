import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle  # Using pickle instead of joblib
# scapy functions are imported only when needed to avoid side-effects during module import


def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses a raw dataset into pseudo-DNA sequences.

    Args:
        file_path (str): Path to raw dataset CSV.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        exit()

    # --- Drop unnecessary columns ---
    df = df.drop(columns=['id'], errors='ignore')

    # --- Separate labels ---
    labels = df[['attack_cat', 'label']] if {'attack_cat', 'label'}.issubset(df.columns) else None
    features = df.drop(columns=['attack_cat', 'label'], errors='ignore')

    # --- Encode categorical features ---
    categorical_features = features.select_dtypes(include='object').columns.tolist()
    if categorical_features:
        for col in categorical_features:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col])
        print(f"🔤 Encoded categorical features: {categorical_features}")

    # --- Scale numeric features ---
    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    # Choose scaler: StandardScaler (default), RobustScaler, MinMaxScaler
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    scaler_type = os.environ.get('SCALER_TYPE', 'standard').lower()
    if scaler_type == 'robust':
        scaler = RobustScaler()
        print("📏 Using RobustScaler for feature scaling.")
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
        print("📏 Using MinMaxScaler for feature scaling.")
    else:
        scaler = StandardScaler()
        print("📏 Using StandardScaler for feature scaling.")
    features_scaled = scaler.fit_transform(features[numeric_features])
    print(f"📏 Scaled numeric features: {len(numeric_features)} columns")

    # --- Binary labels: 1 = anomaly, 0 = normal ---
    if labels is not None:
        y = (labels['label'] != 0).astype(int).values
    else:
        y = np.zeros(len(df))

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, y, test_size=0.3, random_state=42
    )
    print(f"📊 Data split: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # --- Save processed data ---
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(file_path), '..', 'processed'))
    os.makedirs(processed_dir, exist_ok=True)
    processed_file = os.path.join(processed_dir, 'processed_data.csv')
    pd.DataFrame(features_scaled).to_csv(processed_file, index=False)
    print(f"💾 Encoded data saved to {processed_file}")

    # --- Save scaler as .pkl ---
    scaler_dir = "D:/Network-Anomaly-DetectorV2/model"
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_file = os.path.join(scaler_dir, "scaler.pkl")
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to {scaler_file}")

    return X_train, X_test, y_train, y_test, scaler


def get_normal_training_data(X_train, y_train):
    """Filter only normal traffic (label = 0)."""
    return X_train[y_train == 0]


def save_processed_data(X_processed, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(X_processed).to_csv(output_path, index=False)
    print(f"💾 Encoded data saved to {output_path}")





if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        "Dataset/raw/live_capture.csv"
    )
    save_processed_data(X_train, "Dataset/processed/processed_data_train.csv")
    save_processed_data(X_test, "Dataset/processed/processed_data_test.csv")

    # If run directly, optionally list network interfaces and do a short sniff for debugging.
    try:
        from scapy.all import get_if_list, sniff, conf
        import os

        NPCAP_IFACE = os.environ.get("NPCAP_IFACE", None)
        if NPCAP_IFACE is None:
            print("Available interfaces:", get_if_list())
            NPCAP_IFACE = conf.iface
        else:
            print("Using NPCAP_IFACE:", NPCAP_IFACE)

        # Quick sniff (count=5) for debug if environment variable RUN_SNIFF is set
        if os.environ.get("RUN_SNIFF", "0") == "1":
            print("Starting quick sniff for 5 packets on:", NPCAP_IFACE)
            sniff(count=5, prn=lambda p: print(p.summary()), iface=NPCAP_IFACE)
    except Exception:
        # If scapy or permissions not available, silently skip interactive sniffing
        pass
