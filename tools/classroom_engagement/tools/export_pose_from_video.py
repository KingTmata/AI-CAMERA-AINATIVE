"""
Công cụ xuất toàn bộ Pose, Bbox và Keypoints từ video thành file pose_data.json.
Sau khi xuất xong, hệ thống backend sẽ tự động chuyển sang CHẾ ĐỘ 1:
- Hoàn toàn KHÔNG chạy YOLO nữa.
- Chỉ đọc file pose_data.json để xử lý nhãn và đếm số lượng.
- Chạy siêu nhẹ và mượt mà trên mọi máy tính.

Cách dùng:
  python tools/export_pose_from_video.py [duong_dan_video] [duong_dan_output_json]
"""

import sys
import os
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
import config as C

def export_video_to_json(video_path=None, output_json="pose_data.json"):
    from ultralytics import YOLO

    if not video_path:
        video_path = C.SOURCE

    if not os.path.exists(str(video_path)):
        print(f"[Lỗi] Không tìm thấy file video tại: {video_path}")
        return

    print(f"[Export] Đang đọc video từ: {video_path}")
    print(f"[Export] Đang nạp YOLO model: {C.MODEL_PATH}")
    model = YOLO(C.MODEL_PATH)

    all_frames = {}
    frame_idx = 0

    print("[Export] Bắt đầu trích xuất Pose & Bbox...")
    for r in model.track(source=video_path, persist=True, tracker="bytetrack.yaml", stream=True, verbose=False):
        frame_tracks = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            kps = r.keypoints.xy.cpu().numpy()
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else [None] * len(boxes)
            tids = r.boxes.id.int().tolist()

            for b, k, c, tid in zip(boxes, kps, confs, tids):
                # Chuẩn hóa format
                kp_list = k.tolist()
                conf_list = c.tolist() if c is not None else [1.0] * 17
                frame_tracks.append({
                    "id": tid,
                    "box": [round(float(v), 1) for v in b],
                    "keypoints": [[round(float(x), 1), round(float(y), 1)] for x, y in kp_list],
                    "conf": [round(float(sc), 2) for sc in conf_list]
                })

        all_frames[str(frame_idx)] = frame_tracks
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  -> Đã xử lý {frame_idx} frames...")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_frames, f, indent=2)

    print("\n" + "=" * 60)
    print(f"XUẤT DỮ LIỆU THÀNH CÔNG! Đã lưu {frame_idx} frames vào file:")
    print(f"-> {os.path.abspath(output_json)}")
    print("Từ bây giờ, hệ thống backend sẽ tự động chạy CHẾ ĐỘ 1 (đọc trực tiếp file này)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    v_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else "pose_data.json"
    export_video_to_json(v_path, out_path)
