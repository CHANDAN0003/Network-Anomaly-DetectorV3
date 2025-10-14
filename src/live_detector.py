import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP
import joblib
from tensorflow.keras.models import load_model

# --- Paths --- 
MODEL_PATH = "D:/Network-Anomaly-DetectorV2/model/autoencoder_model.keras"
SCALER_PATH = "D:/Network-Anomaly-DetectorV2/model/scaler.pkl"  # Use your .pkl scaler
ANOMALY_THRESHOLD = 0.7  # Adjust according to training results

# --- Load pre-trained model and scaler ---
try:
    autoencoder = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Ensure scaler is actually a StandardScaler
    from sklearn.preprocessing import StandardScaler
    if not isinstance(scaler, StandardScaler):
        raise ValueError("Loaded scaler is not a StandardScaler. Re-save the scaler correctly.")
    
    print("✅ Pre-trained model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    exit()

# --- Features used during training ---
FEATURE_NAMES = list(scaler.feature_names_in_)  # ensures exact match

def process_packet(packet):
    if IP in packet:
        try:
            proto = 6 if TCP in packet else 17 if UDP in packet else 0

            # --- Create a feature dict with all required columns ---
            flow_features = {feat: 0 for feat in FEATURE_NAMES}

            # Fill actual packet values
            flow_features['proto'] = proto
            flow_features['sbytes'] = len(packet)
            flow_features['sttl'] = packet[IP].ttl

            # Handle missing columns like 'service' or 'state'
            for col in ['service', 'state']:
                if col not in flow_features:
                    flow_features[col] = 0

            # Convert to DataFrame and enforce correct column order
            df_live = pd.DataFrame([flow_features])
            df_live = df_live[FEATURE_NAMES]  # same order as during scaler.fit()

            # Scale features
            live_encoded = scaler.transform(df_live)

            # Run anomaly detection
            reconstructed = autoencoder.predict(live_encoded, verbose=0)
            mse = np.mean(np.power(live_encoded - reconstructed, 2), axis=1)
            anomaly_score = mse[0]

            src_ip, dst_ip = packet[IP].src, packet[IP].dst
            if anomaly_score > ANOMALY_THRESHOLD:
                print(f"🚨 ANOMALY! Score: {anomaly_score:.4f} - {src_ip} -> {dst_ip}")
            else:
                print(f"🟢 Normal. Score: {anomaly_score:.4f} - {src_ip} -> {dst_ip}")

        except Exception as e:
            print(f"⚠️ Error processing packet: {e}")

def start_sniffing():
    print("🚀 Starting live traffic analysis. Press Ctrl+C to stop.")
    sniff(prn=process_packet, store=False)

if __name__ == "__main__":
    start_sniffing()
