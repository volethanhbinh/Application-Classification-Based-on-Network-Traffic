import argparse
import os
import joblib
import numpy as np
from collections import Counter
from scapy.all import PcapReader, IP

import tensorflow as tf
from tensorflow.keras.models import load_model

SEQUENCE_LENGTH = 100
NUM_FEATURES = 3 

def is_private_ip(ip_str):
    if ip_str is None: return False
    if ip_str.startswith("192.168."): return True
    if ip_str.startswith("10."): return True
    if ip_str.startswith("172."):
        try:
            second_octet = int(ip_str.split(".")[1])
            if 16 <= second_octet <= 31: return True
        except: pass
    return False

def extract_features_from_pcap(pcap_path):
    pkt_info = []
    
    try:
        with PcapReader(pcap_path) as rdr:
            for i, pkt in enumerate(rdr):
                if i >= SEQUENCE_LENGTH: break
                
                ts = float(getattr(pkt, 'time', 0))
                try: size = len(pkt)
                except: size = 0
                
                src_ip = None
                dst_ip = None
                if IP in pkt:
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                
                pkt_info.append({
                    'size': size,
                    'time': ts,
                    'src': src_ip,
                    'dst': dst_ip
                })
    except Exception as e:
        print(f"[!] Lỗi đọc file pcap: {e}")
        return None

    if len(pkt_info) < 3:
        print("[!] File pcap quá ngắn (ít hơn 3 gói tin).")
        return None

    ip_candidates = []
    for p in pkt_info:
        if is_private_ip(p['src']): ip_candidates.append(p['src'])
        if is_private_ip(p['dst']): ip_candidates.append(p['dst'])
    
    inferred_device_ip = None
    if ip_candidates:
        inferred_device_ip = Counter(ip_candidates).most_common(1)[0][0]

    start_ts = pkt_info[0]['time']
    rel_times = [p['time'] - start_ts for p in pkt_info]
    iat = [0.0]
    for k in range(1, len(rel_times)):
        diff = rel_times[k] - rel_times[k-1]
        iat.append(diff if diff > 0 else 0.0)

    seq_data = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
    curr_len = min(len(pkt_info), SEQUENCE_LENGTH)
    
    for i in range(curr_len):
        seq_data[i, 0] = pkt_info[i]['size']
        seq_data[i, 1] = iat[i]
        direction = 0
        if inferred_device_ip:
            if pkt_info[i]['src'] == inferred_device_ip:
                direction = 1 
            elif pkt_info[i]['dst'] == inferred_device_ip:
                direction = -1 
        seq_data[i, 2] = direction
        
    return seq_data

def main():
    parser = argparse.ArgumentParser(description="Dự đoán loại ứng dụng từ file PCAP (Web, Video, VoIP, Game)")
    parser.add_argument("pcap_file", help="Đường dẫn đến file .pcap hoặc .pcapng")
    parser.add_argument("--model", default="traffic_model_dl.keras", help="Đường dẫn file model (default: traffic_model_dl.keras)")
    parser.add_argument("--scaler", default="scaler_dl.joblib", help="Đường dẫn file scaler (default: scaler_dl.joblib)")
    parser.add_argument("--encoder", default="encoder_dl.joblib", help="Đường dẫn file encoder (default: encoder_dl.joblib)")
    
    args = parser.parse_args()

    if not os.path.exists(args.pcap_file):
        print(f"[ERROR] Không tìm thấy file pcap: {args.pcap_file}")
        return
    if not os.path.exists(args.model) or not os.path.exists(args.scaler) or not os.path.exists(args.encoder):
        print("[ERROR] Thiếu file model, scaler hoặc encoder. Vui lòng kiểm tra lại đường dẫn.")
        return

    print("[-] Đang tải mô hình")
    try:
        model = load_model(args.model)
        scaler = joblib.load(args.scaler)
        encoder = joblib.load(args.encoder)
        classes = sorted(encoder.classes_) #
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải model/scaler: {e}")
        return

    print(f"[-] Đang xử lý file: {args.pcap_file}")
    raw_features = extract_features_from_pcap(args.pcap_file)
    
    if raw_features is None:
        print("[!] Không thể trích xuất đặc trưng.")
        return
    try:
        seq_flat = raw_features.reshape(-1, NUM_FEATURES)
        seq_scaled_flat = scaler.transform(seq_flat)
        X_input = seq_scaled_flat.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
    except Exception as e:
        print(f"[ERROR] Lỗi khi chuẩn hóa dữ liệu (Scaler): {e}")
        return

    print("[-] Đang dự đoán...")
    probs = model.predict(X_input, verbose=0)[0]
    idx = np.argmax(probs)
    label = classes[idx]
    confidence = probs[idx] * 100

    print("\n" + "="*30)
    print(f"KẾT QUẢ DỰ ĐOÁN")
    print("="*30)
    print(f"File:      {os.path.basename(args.pcap_file)}")
    print(f"Dự đoán:   {label.upper()}")
    print(f"Độ tin cậy: {confidence:.2f}%")
    print("-" * 30)
    print("Phân phối xác suất:")
    for i, class_name in enumerate(classes):
        print(f"  - {class_name}: {probs[i]*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()