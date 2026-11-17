"""
Ví dụ minh họa CHẾ ĐỘ 2:
Gửi trực tiếp dữ liệu Pose & Bbox từ một tiến trình Python bên ngoài vào backend qua API /api/feed.
Không cần backend phải chạy YOLO hay đọc video.
"""

import time
import requests
import json
import numpy as np

API_URL = "http://localhost:8000/api/feed"

def run_stream_demo():
    print(f"[Client Stream] Bắt đầu gửi dữ liệu pose demo tới {API_URL}...")
    frame_idx = 0

    while True:
        # Giả lập 3 học sinh
        # Học sinh 1: Giơ tay ở các frame 10 -> 40
        is_raising = (10 <= (frame_idx % 60) <= 40)
        wrist_y = 180 if is_raising else 350

        # Học sinh 2: Lên bảng ở các frame 30 -> 60
        is_at_board = (30 <= (frame_idx % 60) <= 60)
        hip_x = 1000 if is_at_board else 500
        hip_y = 300 if is_at_board else 450

        detections = [
            {
                "id": 1,
                "box": [300, 200, 450, 550],
                "keypoints": [
                    [375, 220], [370, 215], [380, 215], [360, 220], [390, 220],
                    [340, 270], [410, 270], [330, 320], [420, 250],
                    [330, 370], [425, wrist_y], # Cổ tay phải
                    [350, 420], [400, 420], [350, 480], [400, 480], [350, 540], [400, 540]
                ],
                "conf": [0.9] * 17
            },
            {
                "id": 2,
                "box": [hip_x - 50, hip_y - 150, hip_x + 50, hip_y + 150],
                "keypoints": [
                    [hip_x, hip_y - 130], [hip_x - 5, hip_y - 135], [hip_x + 5, hip_y - 135],
                    [hip_x - 15, hip_y - 130], [hip_x + 15, hip_y - 130],
                    [hip_x - 30, hip_y - 80], [hip_x + 30, hip_y - 80],
                    [hip_x - 40, hip_y - 30], [hip_x + 40, hip_y - 30],
                    [hip_x - 40, hip_y + 20], [hip_x + 40, hip_y + 20],
                    [hip_x - 25, hip_y], [hip_x + 25, hip_y],
                    [hip_x - 25, hip_y + 60], [hip_x + 25, hip_y + 60],
                    [hip_x - 25, hip_y + 120], [hip_x + 25, hip_y + 120]
                ],
                "conf": [0.95] * 17
            }
        ]

        payload = {
            "image_b64": None,  # Hoặc truyền chuỗi ảnh base64 nếu muốn
            "detections": detections
        }

        try:
            res = requests.post(API_URL, json=payload, timeout=1.0)
            if res.status_code == 200:
                stats = res.json().get("stats", {})
                print(f"Frame #{frame_idx}: Đang giơ tay={stats.get('currently_raising')}, Ở bảng={stats.get('currently_at_board')}, Điểm={stats.get('score')}")
        except Exception as e:
            print(f"[Client Lỗi]: {e}")
            time.sleep(2)

        frame_idx += 1
        time.sleep(0.04)  # 25 FPS

if __name__ == "__main__":
    run_stream_demo()
