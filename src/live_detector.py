import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, get_if_list, conf
import joblib
from tensorflow.keras.models import load_model
import json
import os

# --- Paths --- 
MODEL_PATH = "D:/Network-Anomaly-DetectorV2/model/autoencoder_model.keras"
SCALER_PATH = "D:/Network-Anomaly-DetectorV2/model/scaler.pkl"  # Use your .pkl scaler
ANOMALY_THRESHOLD = 0.7  # Default fallback; training pipeline will persist calibrated threshold to model/threshold.json

# Try to read calibrated threshold persisted by training pipeline
THRESHOLD_PATH = "model/threshold.json"
if os.path.exists(THRESHOLD_PATH):
    try:
        with open(THRESHOLD_PATH, "r") as fh:
            data = json.load(fh)
            if "threshold" in data:
                ANOMALY_THRESHOLD = float(data["threshold"])
                print(f"✅ Loaded calibrated threshold from {THRESHOLD_PATH}: {ANOMALY_THRESHOLD}")
            else:
                print(f"⚠️ {THRESHOLD_PATH} does not contain 'threshold' key; using default {ANOMALY_THRESHOLD}")
    except Exception as e:
        print(f"⚠️ Failed to read {THRESHOLD_PATH}: {e}; using default threshold {ANOMALY_THRESHOLD}")
else:
    print(f"ℹ️ Threshold file {THRESHOLD_PATH} not found; using default threshold {ANOMALY_THRESHOLD}")

# --- Network interface detection (Npcap/NPF) ---
# Choose interface by env var or auto-detect NPF-like interfaces
NPCAP_IFACE = os.environ.get("NPCAP_IFACE")
try:
    if NPCAP_IFACE:
        print(f"Using NPCAP_IFACE from env: {NPCAP_IFACE}")
    else:
        if_list = get_if_list()
        print("Detected interfaces (sample):", if_list[:10])
        npf_ifaces = [i for i in if_list if i.startswith("\\Device\\NPF") or i.startswith("NPF_") or ('Npcap' in i)]
        print("NPF/Npcap interfaces found:", npf_ifaces)
        if npf_ifaces:
            NPCAP_IFACE = npf_ifaces[0]
            print(f"Selecting NPF interface: {NPCAP_IFACE}")
        else:
            NPCAP_IFACE = conf.iface
            print(f"No NPF interface found; using scapy default: {NPCAP_IFACE}")
except Exception as e:
    print(f"⚠️ Failed to detect interfaces: {e}")
    NPCAP_IFACE = None

# Optional quick sniff for verification (set RUN_SNIFF=1 to enable)
if os.environ.get("RUN_SNIFF", "0") == "1":
    try:
        test_iface = NPCAP_IFACE if NPCAP_IFACE else None
        print(f"🔍 RUN_SNIFF=1: performing short sniff on: {test_iface}")
        # Use timeout to ensure sniff returns even if no packets arrive, enable promisc mode
        packets = sniff(count=5, timeout=10, prn=lambda p: print(p.summary()), iface=test_iface, promisc=True)
        try:
            pkt_count = len(packets)
        except Exception:
            pkt_count = 0
        print(f"🔍 Short sniff complete. Packets captured: {pkt_count}")
        if pkt_count == 0:
            # Helpful guidance for debugging
            print("⚠️ No packets captured. Possible causes: wrong interface, no traffic on that interface, or insufficient privileges.")
            print(" - Try running as Administrator, or set NPCAP_IFACE to a different interface from the detected list.")
            print(" - Generate traffic (e.g., open a webpage or run `ping 8.8.8.8 -n 5`) while sniffing.")
            # If we selected a NPF iface, try the scapy default as a fallback
            if test_iface is not None:
                try:
                    fallback_iface = conf.iface
                    if fallback_iface != test_iface:
                        print(f"ℹ️ Trying fallback interface: {fallback_iface} for 5 seconds...")
                        fallback_pkts = sniff(count=5, timeout=5, prn=lambda p: print(p.summary()), iface=fallback_iface, promisc=True)
                        print(f"ℹ️ Fallback sniff captured: {len(fallback_pkts)} packets")
                except Exception as e:
                    print(f"⚠️ Fallback sniff failed: {e}")
    except Exception as e:
        print(f"⚠️ Short sniff failed: {e}")

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
