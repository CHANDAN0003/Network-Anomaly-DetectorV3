from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
import time
import os

# Ensure the output directory exists
os.makedirs("Dataset/raw", exist_ok=True)

OUTPUT_FILE = "Dataset/raw/live_capture.csv"

captured_data = []

def process_packet(packet):
    """Extract minimal flow features from live packets."""
    if IP in packet:
        proto = "tcp" if TCP in packet else "udp" if UDP in packet else "other"
        length = len(packet)
        timestamp = time.time()

        captured_data.append({
            "timestamp": timestamp,
            "src": packet[IP].src,
            "dst": packet[IP].dst,
            "proto": proto,
            "length": length
        })

        # Save in small batches
        if len(captured_data) % 50 == 0:
            df = pd.DataFrame(captured_data)
            df.to_csv(OUTPUT_FILE, mode="a", header=not pd.io.common.file_exists(OUTPUT_FILE), index=False)
            print(f"Saved {len(captured_data)} packets to {OUTPUT_FILE}")
            captured_data.clear()

def start_sniffing():
    print("🚀 Starting live packet capture... Press Ctrl+C to stop.")
    sniff(prn=process_packet, store=False)

if __name__ == "__main__":
    start_sniffing()
