import os
import argparse
from scapy.all import PcapReader, IP
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

# Configuration
SEQUENCE_LENGTH = 300
NUM_FEATURES = 3  # [Size, IAT, Direction]


def is_private_ip(ip_str):
    """Check whether an IP address belongs to a private (RFC 1918) subnet."""
    if ip_str is None:
        return False
    if ip_str.startswith("192.168."):
        return True
    if ip_str.startswith("10."):
        return True
    if ip_str.startswith("172."):
        try:
            second_octet = int(ip_str.split(".")[1])
            if 16 <= second_octet <= 31:
                return True
        except Exception:
            pass
    return False


def extract_seq_from_pcap(pcap_path, seq_len=SEQUENCE_LENGTH):
    pkt_info = []  # Store dict with (size, time, src_ip, dst_ip)

    try:
        with PcapReader(pcap_path) as rdr:
            for i, pkt in enumerate(rdr):
                if i >= seq_len:
                    break

                ts = float(getattr(pkt, 'time', 0))
                try:
                    size = len(pkt)
                except Exception:
                    size = 0

                src_ip = None
                dst_ip = None
                if IP in pkt:
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst

                pkt_info.append({
                    'size': size,
                    'time': ts,
                    'src': src_ip,
                    'dst': dst_ip,
                })
    except Exception:
        return None

    if len(pkt_info) < 5:
        return None

    # Infer the monitored device's private IP by frequency
    ip_candidates = []
    for p in pkt_info:
        if is_private_ip(p['src']):
            ip_candidates.append(p['src'])
        if is_private_ip(p['dst']):
            ip_candidates.append(p['dst'])

    inferred_device_ip = None
    if ip_candidates:
        inferred_device_ip = Counter(ip_candidates).most_common(1)[0][0]

    # Compute Inter-Arrival Time (IAT)
    start_ts = pkt_info[0]['time']
    rel_times = [p['time'] - start_ts for p in pkt_info]
    iat = [0.0]
    for k in range(1, len(rel_times)):
        diff = rel_times[k] - rel_times[k - 1]
        iat.append(diff if diff > 0 else 0.0)

    # Build feature matrix
    seq_data = np.zeros((seq_len, NUM_FEATURES), dtype=np.float32)
    curr_len = min(len(pkt_info), seq_len)

    for i in range(curr_len):
        # Feature 0: Packet Size (bytes)
        seq_data[i, 0] = pkt_info[i]['size']

        # Feature 1: Inter-Arrival Time (seconds)
        seq_data[i, 1] = iat[i]

        # Feature 2: Direction (+1: Uplink (Outgoing), -1: Downlink (Incoming), 0: Unknown)
        direction = 0
        if inferred_device_ip:
            if pkt_info[i]['src'] == inferred_device_ip:
                direction = 1  # Uplink (Outgoing)
            elif pkt_info[i]['dst'] == inferred_device_ip:
                direction = -1  # Downlink (Incoming)
        seq_data[i, 2] = direction

    # Extract class label from filename prefix
    base = os.path.basename(pcap_path)
    label_hint = base.split("_")[0] if "_" in base else "unknown"

    return seq_data, label_hint


def main():
    parser = argparse.ArgumentParser(
        description="Extract fixed-length packet sequence feature arrays (X.npy, y.npy) from PCAP files."
    )
    parser.add_argument("--pcap_dir", required=True, help="Directory containing chopped .pcap/.pcapng files.")
    parser.add_argument("--out_dir", required=True, help="Output directory for X.npy and y.npy.")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (default: CPU count).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pcap_files = [
        os.path.join(args.pcap_dir, f)
        for f in os.listdir(args.pcap_dir)
        if f.endswith(('.pcap', '.pcapng'))
    ]

    if not pcap_files:
        print("[!] No PCAP files found.")
        return

    X_list = []
    y_list = []

    max_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    print(f"[*] Processing {len(pcap_files)} files using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_seq_from_pcap, p): p for p in pcap_files}

        for future in tqdm(as_completed(futures), total=len(pcap_files)):
            res = future.result()
            if res is not None:
                seq, label = res
                X_list.append(seq)
                y_list.append(label)

    if len(X_list) == 0:
        print("[!] No feature data extracted.")
        return

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"[*] Result: X shape={X.shape}, y shape={y.shape}")

    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    print("[*] Saved X.npy and y.npy")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
