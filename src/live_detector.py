import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, get_if_list, conf
import joblib
from tensorflow.keras.models import load_model
import json
import os
import time
from collections import defaultdict
import threading

# --- Paths --- 
MODEL_PATH = "model/autoencoder_model.keras"
SCALER_PATH = "model/scaler.pkl"
THRESHOLD_PATH = "model/threshold.json"

# --- 1. SET YOUR PHONE'S IP ADDRESS HERE ---
# Find this on your phone's Wi-Fi settings (e.g., "192.168.1.12")
TARGET_IP = "Your_phone _ip_address"
# ------------------------------------------

# --- Load Artifacts ---
try:
    autoencoder = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        data = json.load(f)
        ANOMALY_THRESHOLD = float(data["threshold"])
    
    # Ensure scaler is a valid scikit-learn scaler
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    if not isinstance(scaler, (StandardScaler, RobustScaler, MinMaxScaler)):
        raise ValueError("Loaded scaler is not a recognized scaler. Re-save the scaler correctly.")

    FEATURE_NAMES = list(scaler.feature_names_in_)
    print(f"✅ Loaded calibrated threshold: {ANOMALY_THRESHOLD}")
    print("✅ Pre-trained model and scaler loaded successfully.")
    if TARGET_IP == "YOUR_PHONE_IP_HERE":
        print("⚠️ WARNING: Please edit TARGET_IP in this script to your phone's IP address.")
    else:
        print(f"🎧 Listening only for packets from: {TARGET_IP}")
except Exception as e:
    print(f"❌ Error loading model/scaler/threshold: {e}")
    exit()

# --- Network interface detection (Npcap/NPF) ---
NPCAP_IFACE = os.environ.get("NPCAP_IFACE")
try:
    if NPCAP_IFACE:
        print(f"Using NPCAP_IFACE from env: {NPCAP_IFACE}")
    else:
        if_list = get_if_list()
        npf_ifaces = [i for i in if_list if i.startswith("\\Device\\NPF") or i.startswith("NPF_") or ('Npcap' in i)]
        if npf_ifaces:
            NPCAP_IFACE = npf_ifaces[0]
            print(f"Selecting NPF interface: {NPCAP_IFACE}")
        else:
            NPCAP_IFACE = conf.iface
            print(f"No NPF interface found; using scapy default: {NPCAP_IFACE}")
except Exception as e:
    print(f"⚠️ Failed to detect interfaces: {e}")
    NPCAP_IFACE = None # Will let scapy try to auto-detect

# --- Stateful Flow Aggregator ---
active_flows = defaultdict(lambda: {
    'start_time': time.time(),
    'spkts': 0,
    'sbytes': 0,
    'sttl': 0,
    'proto_num': 0,
    'last_seen': time.time()
})
# --- FIX: Reduced timeout for faster response ---
FLOW_TIMEOUT = 1  # Seconds of inactivity to consider a flow "done"

def get_flow_key(packet):
    """Creates a 5-tuple key for the flow"""
    if IP in packet:
        proto = packet[IP].proto
        src = packet[IP].src
        dst = packet[IP].dst
        sport = 0
        dport = 0
        if TCP in packet:
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        elif UDP in packet:
            sport = packet[UDP].sport
            dport = packet[UDP].dport
        return (src, dst, sport, dport, proto)
    return None

def process_packet(packet):
    """Adds packet information to our stateful flow tracker"""
    key = get_flow_key(packet)
    if not key:
        return

    # --- IP FILTER ---
    if key[0] != TARGET_IP:
        return
    # ------------------

    flow = active_flows[key]

    # Update flow stats
    flow['last_seen'] = time.time()
    flow['spkts'] += 1
    flow['sbytes'] += len(packet)
    
    if flow['spkts'] == 1:
         flow['sttl'] = packet[IP].ttl
         flow['proto_num'] = packet[IP].proto

def score_flow(flow_data, key):
    """Calculates flow features, scales, and predicts when a flow times out"""
    try:
        # 1. Calculate derived features
        dur = flow_data['last_seen'] - flow_data['start_time']
        
        # --- FIX: Handle single-packet flows (like pings) ---
        if flow_data['spkts'] <= 1 or dur == 0:
            rate = 0.0
            dur = 0.0 # Match training data, where dur=0 often means rate=0
        else:
            rate = flow_data['spkts'] / dur
        # --- END FIX ---

        # 2. Create the feature vector
        flow_features = {feat: 0 for feat in FEATURE_NAMES}
        flow_features['dur'] = dur
        flow_features['spkts'] = flow_data['spkts']
        flow_features['sbytes'] = flow_data['sbytes']
        flow_features['rate'] = rate
        flow_features['sttl'] = flow_data['sttl']
        flow_features['proto'] = flow_data['proto_num'] 

        # 3. Scale and Predict
        df_live = pd.DataFrame([flow_features])
        df_live = df_live[FEATURE_NAMES]
        live_scaled = scaler.transform(df_live)
        
        reconstructed = autoencoder.predict(live_scaled, verbose=0)
        mse = np.mean(np.power(live_scaled - reconstructed, 2), axis=1)
        anomaly_score = mse[0]

        src_ip, dst_ip = key[0], key[1]
        if anomaly_score > ANOMALY_THRESHOLD:
            print(f"🚨 [FLOW] ANOMALY! Score: {anomaly_score:.4f} - {src_ip} -> {dst_ip} (Pkts: {flow_data['spkts']}, Rate: {rate:.0f}/s)")
        else:
            print(f"🟢 [FLOW] Normal. Score: {anomaly_score:.4f} - {src_ip} -> {dst_ip} (Pkts: {flow_data['spkts']}, Rate: {rate:.0f}/s)")
    
    except Exception as e:
        print(f"⚠️ Error scoring flow: {e}")

def check_timed_out_flows():
    """Periodically checks for flows that have not seen a packet in FLOW_TIMEOUT seconds"""
    current_time = time.time()
    timed_out_keys = []
    
    try:
        for key, flow in list(active_flows.items()):
            if current_time - flow['last_seen'] > FLOW_TIMEOUT:
                timed_out_keys.append(key)

        for key in timed_out_keys:
            if key[0] == TARGET_IP:
                flow_data = active_flows.pop(key, None)
                if flow_data:
                    score_flow(flow_data, key)
            else:
                active_flows.pop(key, None)
    except Exception as e:
        print(f"Error in timeout checker: {e}")


def start_sniffing():
    print(f"🚀 Starting stateful traffic analysis (Flow timeout: {FLOW_TIMEOUT}s). Press Ctrl+C to stop.")
    
    def timeout_checker_loop():
        while True:
            time.sleep(FLOW_TIMEOUT)
            check_timed_out_flows()
    
    checker_thread = threading.Thread(target=timeout_checker_loop, daemon=True)
    checker_thread.start()

    print(f"Sniffing on interface: {NPCAP_IFACE}")
    sniff(prn=process_packet, store=False, iface=NPCAP_IFACE, promisc=True)

if __name__ == "__main__":
    start_sniffing()
