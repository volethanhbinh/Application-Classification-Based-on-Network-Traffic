import os
import argparse
from scapy.all import PcapReader, PcapWriter
from collections import deque
from tqdm import tqdm


def chop_pcap_streaming(pcap_path, out_dir, chunk_duration, step_size, min_packets):
    chunks_created = 0
    base_name = os.path.basename(pcap_path)
    name_without_ext, ext = os.path.splitext(base_name)

    try:
        reader = PcapReader(pcap_path)
    except Exception as e:
        print(f"[!] Error opening {pcap_path}: {e}")
        return 0

    buffer = deque()
    last_pkt_time = None
    try:
        pkt = reader.read_packet()
    except Exception as e:
        reader.close()
        print(f"[!] Error reading first packet: {e}")
        return 0

    if pkt is None:
        reader.close()
        return 0
    start_ts = float(pkt.time)
    last_pkt_time = float(pkt.time)
    buffer.append(pkt)

    current_window_start = float(start_ts)
    current_window_end = current_window_start + chunk_duration

    def write_chunk(window_start, window_end, index_seconds):
        nonlocal chunks_created
        out_name = f"{name_without_ext}_slide{int(index_seconds)}s{ext}"
        out_path = os.path.join(out_dir, out_name)
        writer = PcapWriter(out_path, append=False, sync=True)
        count = 0
        for p in buffer:
            t = float(p.time)
            if t >= window_start and t < window_end:
                writer.write(p)
                count += 1
        writer.close()
        if count >= min_packets:
            chunks_created += 1
        else:
            try:
                os.remove(out_path)
            except Exception:
                pass

    for pkt in reader:
        if pkt is None:
            break
        pkt_time = float(pkt.time)
        last_pkt_time = pkt_time
        buffer.append(pkt)

        while pkt_time >= current_window_end:
            index_seconds = int(current_window_start - start_ts)
            write_chunk(current_window_start, current_window_end, index_seconds)

            current_window_start += step_size
            current_window_end = current_window_start + chunk_duration

            while buffer and float(buffer[0].time) < current_window_start:
                buffer.popleft()

    while current_window_start <= last_pkt_time:
        index_seconds = int(current_window_start - start_ts)
        write_chunk(current_window_start, current_window_end, index_seconds)

        current_window_start += step_size
        current_window_end = current_window_start + chunk_duration

        while buffer and float(buffer[0].time) < current_window_start:
            buffer.popleft()
    reader.close()
    return chunks_created


def main():
    parser = argparse.ArgumentParser(
        description="Split large PCAP/PCAPNG files into fixed-duration chunks using a sliding window."
    )
    parser.add_argument("--pcap_dir", required=True, help="Directory containing input .pcap/.pcapng files.")
    parser.add_argument("--out_dir", required=True, help="Output directory for chopped PCAP chunks.")
    parser.add_argument("--duration", type=float, default=4.0, help="Window duration in seconds (default: 4.0).")
    parser.add_argument("--step_size", type=float, default=2.0, help="Sliding step size in seconds (default: 2.0).")
    parser.add_argument(
        "--min_packets",
        type=int,
        default=20,
        help="Minimum number of packets required per chunk (default: 20).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    extensions = ('.pcap', '.pcapng')
    pcap_files = [os.path.join(args.pcap_dir, f) for f in os.listdir(args.pcap_dir) if f.endswith(extensions)]
    if not pcap_files:
        print(f"[!] No .pcap/.pcapng files found in {args.pcap_dir}")
        return

    total_chunks = 0
    for p in tqdm(pcap_files, desc="Files"):
        c = chop_pcap_streaming(
            p,
            args.out_dir,
            chunk_duration=args.duration,
            step_size=args.step_size,
            min_packets=args.min_packets,
        )
        total_chunks += c

    print(f"Done. Total chunks created: {total_chunks}")


if __name__ == "__main__":
    main()
