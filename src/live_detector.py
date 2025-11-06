import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, get_if_list, conf
import joblib
from tensorflow.keras.models import load_model as keras_load_model
import json
import os
import time
from collections import defaultdict
import threading
import socket
from models.anomaly_detector import load_model
from utils.data_processor import load_and_preprocess_data, get_normal_training_data, save_processed_data
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# ================================
# CONFIGURATION
# ================================

MODEL_PATH = "model/autoencoder_model.keras"
SCALER_PATH = "model/scaler.pkl"
THRESHOLD_PATH = "model/threshold.json"

# --- MOBILE IP ---
TARGET_IP = "172.16.22.68"  # ✅ Your mobile device IP address
# ================================

# --- LOAD MODEL, SCALER, THRESHOLD ---
try:
    autoencoder = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        data = json.load(f)
        ANOMALY_THRESHOLD = float(data["threshold"])

    if not isinstance(scaler, (StandardScaler, RobustScaler, MinMaxScaler)):
        raise ValueError("Loaded scaler is not recognized. Please re-save the scaler.")

    FEATURE_NAMES = list(scaler.feature_names_in_)
    print(f"✅ Loaded model, scaler, and threshold ({ANOMALY_THRESHOLD}) successfully.")
    print(f"🎧 Listening for packets from: {TARGET_IP}")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    exit()

# ================================
# NETWORK INTERFACE SETUP
# ================================

NPCAP_IFACE = os.environ.get("NPCAP_IFACE")
try:
    if NPCAP_IFACE:
        print(f"Using NPCAP_IFACE from environment: {NPCAP_IFACE}")
    else:
        interfaces = get_if_list()
        npf_ifaces = [i for i in interfaces if "NPF" in i or "Npcap" in i]
        if npf_ifaces:
            NPCAP_IFACE = npf_ifaces[0]
            print(f"Using Npcap interface: {NPCAP_IFACE}")
        else:
            NPCAP_IFACE = conf.iface
            print(f"No specific Npcap interface found. Using default: {NPCAP_IFACE}")
except Exception as e:
    print(f"⚠️ Interface detection error: {e}")
    NPCAP_IFACE = None

# ================================
# STATEFUL FLOW TRACKING
# ================================

active_flows = defaultdict(lambda: {
    "start_time": time.time(),
    "spkts": 0,
    "sbytes": 0,
    "sttl": 0,
    "proto_num": 0,
    "last_seen": time.time()
})

FLOW_TIMEOUT = 1  # seconds

def get_flow_key(packet):
    if IP in packet:
        proto = packet[IP].proto
        src = packet[IP].src
        dst = packet[IP].dst
        sport = packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else 0)
        dport = packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0)
        return (src, dst, sport, dport, proto)
    return None

def process_packet(packet):
    key = get_flow_key(packet)
    if not key:
        return
    if key[0] != TARGET_IP:
        return

    flow = active_flows[key]
    flow["last_seen"] = time.time()
    flow["spkts"] += 1
    flow["sbytes"] += len(packet)

    if flow["spkts"] == 1:
        flow["sttl"] = packet[IP].ttl
        flow["proto_num"] = packet[IP].proto

def score_flow(flow_data, key):
    try:
        dur = flow_data["last_seen"] - flow_data["start_time"]
        rate = flow_data["spkts"] / dur if dur > 0 else 0.0

        flow_features = {feat: 0 for feat in FEATURE_NAMES}
        flow_features.update({
            "dur": dur,
            "spkts": flow_data["spkts"],
            "sbytes": flow_data["sbytes"],
            "rate": rate,
            "sttl": flow_data["sttl"],
            "proto": flow_data["proto_num"]
        })

        df = pd.DataFrame([flow_features])[FEATURE_NAMES]
        scaled = scaler.transform(df)

        recon = autoencoder.predict(scaled, verbose=0)
        mse = np.mean(np.power(scaled - recon, 2), axis=1)[0]

        if mse > ANOMALY_THRESHOLD:
            print(f"🚨 ANOMALY [{key[0]} → {key[1]}] Score={mse:.5f}")
        else:
            print(f"🟢 NORMAL [{key[0]} → {key[1]}] Score={mse:.5f}")

    except Exception as e:
        print(f"⚠️ Error scoring flow: {e}")

def check_timed_out_flows():
    while True:
        time.sleep(FLOW_TIMEOUT)
        now = time.time()
        for key in list(active_flows.keys()):
            if now - active_flows[key]["last_seen"] > FLOW_TIMEOUT:
                flow = active_flows.pop(key)
                score_flow(flow, key)

def start_sniffing():
    print(f"🚀 Starting network sniffing (Flow timeout: {FLOW_TIMEOUT}s). Press Ctrl+C to stop.")
    threading.Thread(target=check_timed_out_flows, daemon=True).start()
    sniff(prn=process_packet, store=False, iface=NPCAP_IFACE, promisc=True)

# ================================
# UDP SERVER (For Mobile App)
# ================================

def preprocess_packet(packet_data):
    try:
        if isinstance(packet_data, bytes):
            packet_str = packet_data.decode("utf-8", errors="ignore").strip()
        else:
            packet_str = str(packet_data).strip()

        try:
            data = json.loads(packet_str)
        except Exception:
            values = list(map(float, packet_str.split(",")))
            data = {name: val for name, val in zip(FEATURE_NAMES, values)}

        row = {feat: float(data.get(feat, 0.0)) for feat in FEATURE_NAMES}
        df = pd.DataFrame([row])[FEATURE_NAMES]
        return scaler.transform(df)
    except Exception as e:
        raise ValueError(f"Failed to preprocess packet: {e}")

def handle_packet(packet_data):
    try:
        x = preprocess_packet(packet_data)
        recon = autoencoder.predict(x, verbose=0)
        mse = float(np.mean(np.power(x - recon, 2)))

        if mse > ANOMALY_THRESHOLD:
            print(f"🚨 [UDP] Anomaly detected! Score={mse:.6f}")
        else:
            print(f"🟢 [UDP] Normal packet. Score={mse:.6f}")
    except Exception as e:
        print(f"⚠️ Error processing packet: {e}")

def start_server(ip, port):
    def listen():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.bind((ip, port))
            print(f"📡 UDP server listening on {ip}:{port}")
            while True:
                data, addr = server.recvfrom(1024)
                print(f"📩 Packet received from {addr}")
                threading.Thread(target=handle_packet, args=(data,), daemon=True).start()

    threading.Thread(target=listen, daemon=True).start()

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    # UDP setup
    SERVER_IP = "0.0.0.0"   # Listen on all network interfaces
    SERVER_PORT = 12345     # Must match mobile app port

    start_server(SERVER_IP, SERVER_PORT)
    print("✅ Real-time UDP anomaly detection server running...\n")

    try:
        start_sniffing()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
