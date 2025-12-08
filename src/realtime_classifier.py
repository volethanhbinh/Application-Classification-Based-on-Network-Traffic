import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import joblib
import numpy as np
from collections import deque
from scapy.all import sniff, IP, get_if_addr

import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
plt.style.use('ggplot')

#CẤU HÌNH
SEQUENCE_LENGTH = 100     
NUM_FEATURES = 3 # [Size, IAT, Direction]

#Cấu hình giao diện & Cảnh báo
HISTORY_LENGTH = 10      
HIGH_TRAFFIC_MB = 0.12      
STRANGE_IP_COUNT = 15      
LOW_CONFIDENCE_THRESH = 0.50 

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

class RealtimeDLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG GIÁM SÁT netRUNNER (DL + Direction)")
        self.root.geometry("1280x720")
        self.root.config(bg="#f0f0f0") 
        
        self.model = None
        self.scaler = None
        self.encoder = None
        self.running = True
        self.classes = []
        
        # Data storage
        self.history = {} 
        self.class_labels_ui = {} 
        
        # UI Elements
        self.setup_ui()
        
        # Start
        self.root.after(100, self.initialize_model)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        title_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        title_frame.pack(fill="x")
        self.title_label = tk.Label(title_frame, text="HỆ THỐNG GIÁM SÁT VÀ PHÂN LOẠI LƯU LƯỢNG MẠNG", 
                                    font=("Arial", 18, "bold"), bg="#f0f0f0")
        self.title_label.pack()

        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        chart_frame = tk.Frame(main_frame, bg="white", relief="sunken", borderwidth=1)
        chart_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        dashboard_frame = tk.Frame(main_frame, bg="#f0f0f0", width=380)
        dashboard_frame.pack(side="right", fill="y")
        dashboard_frame.pack_propagate(False) 

        status_group = tk.LabelFrame(dashboard_frame, text="Trạng thái", font=("Arial", 12, "bold"), bg="#f0f0f0", padx=10, pady=10)
        status_group.pack(fill="x", pady=(0, 10))
        self.lbl_status = tk.Label(status_group, text="Khởi tạo...", font=("Arial", 11), fg="blue", bg="#f0f0f0")
        self.lbl_status.pack()

        prediction_group = tk.LabelFrame(dashboard_frame, text="Dự đoán Hiện tại", font=("Arial", 12, "bold"), bg="#f0f0f0", padx=10, pady=10)
        prediction_group.pack(fill="x", pady=10)
        self.lbl_result = tk.Label(prediction_group, text="--", font=("Arial", 26, "bold"), fg="red", bg="#f0f0f0", pady=10)
        self.lbl_result.pack()

        self.details_group = tk.LabelFrame(dashboard_frame, text="Chi tiết các lớp", font=("Arial", 12, "bold"), bg="#f0f0f0", padx=10, pady=10)
        self.details_group.pack(fill="x", pady=10)

        warning_group = tk.LabelFrame(dashboard_frame, text="Cảnh báo", font=("Arial", 12, "bold"), bg="#f0f0f0", padx=10, pady=10)
        warning_group.pack(fill="x", pady=10)
        self.lbl_warning = tk.Label(warning_group, text="Bình thường", font=("Arial", 11, "bold"), fg="green", bg="#f0f0f0", justify=tk.LEFT, wraplength=320)
        self.lbl_warning.pack(fill="x")

        self.fig = plt.Figure(figsize=(7, 5), dpi=100, facecolor="#ffffff")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#ffffff")
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def initialize_model(self):
        try:
            self.lbl_status.config(text="Đang tải mô hình...")
            self.model = load_model("traffic_model_dl.keras")
            self.scaler = joblib.load("scaler_dl.joblib")
            self.encoder = joblib.load("encoder_dl.joblib")
            
            self.classes = sorted(self.encoder.classes_)
            
            for c in self.classes:
                self.history[c] = deque([0]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
                lbl = tk.Label(self.details_group, text=f"{c}: 0.0%", font=("Arial", 12), bg="#f0f0f0", anchor="w")
                lbl.pack(fill="x", padx=5, pady=2)
                self.class_labels_ui[c] = lbl

            self.update_chart() 
            self.lbl_status.config(text="Sẵn sàng! Đang bắt gói tin...", fg="green")
            
            t = threading.Thread(target=self.sniff_loop, daemon=True)
            t.start()
            
        except Exception as e:
            self.lbl_status.config(text=f"Lỗi load model: {e}", fg="red")

    def process_packets_to_sequence(self, packets):
        # 1. Thu thập thông tin gói tin
        pkt_info = []
        for pkt in packets:
            if len(pkt_info) >= SEQUENCE_LENGTH: break
            try:
                size = len(pkt)
                ts = float(getattr(pkt, 'time', 0))
                src = pkt[IP].src if IP in pkt else None
                dst = pkt[IP].dst if IP in pkt else None
                pkt_info.append({'size': size, 'time': ts, 'src': src, 'dst': dst})
            except: pass
            
        if len(pkt_info) < 15: return None 

        # 2. Lấy IP thiết bị trực tiếp
        device_ip = get_if_addr("Wi-Fi") 
        
        # 3. Tính toán Features
        start_ts = pkt_info[0]['time']
        rel_times = [p['time'] - start_ts for p in pkt_info]
        iat = [0.0]
        for i in range(1, len(rel_times)):
            val = rel_times[i] - rel_times[i-1]
            iat.append(val if val > 0 else 0)
            
        seq_raw = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES)) 
        curr_len = len(pkt_info)
        
        for i in range(curr_len):
            # Feature 0: Size
            seq_raw[i, 0] = pkt_info[i]['size']
            # Feature 1: IAT
            seq_raw[i, 1] = iat[i]
            
            # Feature 2: Direction
            direction = 0
            if device_ip:
                if pkt_info[i]['src'] == device_ip:
                    direction = 1  # Uplink (Từ máy mình gửi đi)
                elif pkt_info[i]['dst'] == device_ip:
                    direction = -1 # Downlink (Từ ngoài gửi về máy mình)
            
            seq_raw[i, 2] = direction
            
        # 4. Scale dữ liệu
        seq_flat = seq_raw.reshape(-1, NUM_FEATURES)
        try:
            seq_scaled_flat = self.scaler.transform(seq_flat)
        except ValueError as e:
            print(f"[!] Lỗi Scaler: {e}")
            return None

        seq_input = seq_scaled_flat.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        
        return seq_input

    def calculate_stats_for_warning(self, packets):
        total_bytes = sum(len(pkt) for pkt in packets)
        unique_ips = set()
        for pkt in packets:
            if IP in pkt:
                unique_ips.add(pkt[IP].dst)
        return total_bytes, len(unique_ips)

    def sniff_loop(self):
        while self.running:
            self.root.after(0, lambda: self.lbl_status.config(
                text=f"Đang thu thập gói tin...", fg="blue"
            ))
            pkts = sniff(count=SEQUENCE_LENGTH, iface="Wi-Fi")
            
            if not pkts:
                self.root.after(0, lambda: self.update_display(None, 0, 0))
                continue
            
            X_input = self.process_packets_to_sequence(pkts)
            total_bytes, num_ips = self.calculate_stats_for_warning(pkts)

            if X_input is not None:
                probs = self.model.predict(X_input, verbose=0)[0] 
                self.root.after(0, lambda p=probs, b=total_bytes, i=num_ips: self.update_display(p, b, i))
            else:
                 self.root.after(0, lambda: self.lbl_status.config(text="Dữ liệu không hợp lệ", fg="gray"))

    def update_display(self, probs, total_bytes, num_ips):
        if probs is None:
            return

        idx = np.argmax(probs)
        label = self.classes[idx]
        conf = probs[idx]
        
        color = "#006400" if conf > 0.7 else ("#FF8C00" if conf > 0.5 else "red") 
        self.lbl_result.config(text=f"{label.upper()}\n({conf*100:.1f}%)", fg=color)
        
        for i, c in enumerate(self.classes):
            p_val = probs[i] * 100
            self.history[c].append(p_val)
            
            if c in self.class_labels_ui:
                font_w = "bold" if i == idx else "normal"
                self.class_labels_ui[c].config(text=f"{c}: {p_val:.1f}%", font=("Arial", 12, font_w))
        
        warnings = []
        if conf < LOW_CONFIDENCE_THRESH:
            warnings.append(f"• Độ tin cậy thấp (<{LOW_CONFIDENCE_THRESH*100:.0f}%)")
        
        total_mb = total_bytes / (1024 * 1024)
        if total_mb > HIGH_TRAFFIC_MB:
            warnings.append(f"• Lưu lượng lớn: {total_mb:.1f} MB")
            
        if num_ips > STRANGE_IP_COUNT:
            warnings.append(f"• Kết nối nhiều IP: {num_ips} IP")

        if warnings:
            self.lbl_warning.config(text="\n".join(warnings), fg="red")
        else:
            self.lbl_warning.config(text="Hệ thống bình thường", fg="green")

        self.update_chart()

    def update_chart(self):
        self.ax.clear()
        for c in self.classes:
            if c in self.history and len(self.history[c]) > 0:
                self.ax.plot(self.history[c], label=c, marker='o', markersize=3, linewidth=2)
        
        self.ax.set_ylim(-5, 105)
        self.ax.legend(loc='upper left', fontsize='small')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_xticks([])
        
        self.fig.tight_layout()
        self.canvas.draw()

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealtimeDLApp(root)
    root.mainloop()