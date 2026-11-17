"""
Tạo file dữ liệu mẫu pose_data.json để kiểm thử CHẾ ĐỘ 1 & BỘ LỌC NGOẠI LỆ:
- Học sinh #1: Giơ 1 tay phát biểu chuẩn (frame 15 -> 45) -> Nhãn "GIO TAY" (+1 điểm)
- Học sinh #2: Lên bảng làm bài (frame 30 -> 75) -> Nhãn "O BANG" (+3 điểm)
- Học sinh #3: VƯƠN VAI / HÍT THỞ giơ cả 2 tay lên cao (frame 50 -> 95)
  -> Bộ lọc ngoại lệ kích hoạt -> Nhãn "VUON VAI (LỌC)" (0 điểm, không tính giơ tay!)
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pose_data.json"))

def generate():
    total_frames = 150
    data = {}

    for f in range(total_frames):
        # Học sinh 1: Giơ 1 tay phát biểu ở frame 15 -> 45
        is_raising_1 = (15 <= f <= 45)
        wrist1_y = 170.0 if is_raising_1 else 340.0

        # Học sinh 2: Lên bảng ở frame 30 -> 75 (ở vùng x: 1000..1150, y: 250..400)
        is_at_board_2 = (30 <= f <= 75)
        pos2_x = 1050.0 if is_at_board_2 else 550.0
        pos2_y = 300.0 if is_at_board_2 else 450.0

        # Học sinh 3: VƯƠN VAI (cả 2 tay cùng giơ cao qua vai) ở frame 50 -> 95
        is_stretching_3 = (50 <= f <= 95)
        wrist3_left_y = 160.0 if is_stretching_3 else 380.0
        wrist3_right_y = 160.0 if is_stretching_3 else 380.0

        tracks = [
            {
                "id": 1,
                "box": [280.0, 190.0, 420.0, 520.0],
                "keypoints": [
                    [350.0, 210.0], [345.0, 205.0], [355.0, 205.0], [335.0, 210.0], [365.0, 210.0],
                    [320.0, 260.0], [380.0, 260.0], # Vai trái, vai phải (y=260)
                    [310.0, 310.0], [390.0, 240.0], # Khuỷu tay
                    [310.0, 360.0], [395.0, wrist1_y], # Cổ tay trái, cổ tay phải (y=170 < 260 khi giơ tay)
                    [330.0, 410.0], [370.0, 410.0], # Hông
                    [330.0, 460.0], [370.0, 460.0],
                    [330.0, 510.0], [370.0, 510.0]
                ],
                "conf": [0.9] * 17
            },
            {
                "id": 2,
                "box": [pos2_x - 50.0, pos2_y - 120.0, pos2_x + 50.0, pos2_y + 120.0],
                "keypoints": [
                    [pos2_x, pos2_y - 110.0], [pos2_x - 5.0, pos2_y - 115.0], [pos2_x + 5.0, pos2_y - 115.0],
                    [pos2_x - 15.0, pos2_y - 110.0], [pos2_x + 15.0, pos2_y - 110.0],
                    [pos2_x - 30.0, pos2_y - 70.0], [pos2_x + 30.0, pos2_y - 70.0],
                    [pos2_x - 35.0, pos2_y - 20.0], [pos2_x + 35.0, pos2_y - 20.0],
                    [pos2_x - 35.0, pos2_y + 30.0], [pos2_x + 35.0, pos2_y + 30.0],
                    [pos2_x - 20.0, pos2_y], [pos2_x + 20.0, pos2_y], # Hông (pos2 nằm trong BOARD_POLY)
                    [pos2_x - 20.0, pos2_y + 50.0], [pos2_x + 20.0, pos2_y + 50.0],
                    [pos2_x - 20.0, pos2_y + 100.0], [pos2_x + 20.0, pos2_y + 100.0]
                ],
                "conf": [0.92] * 17
            },
            {
                "id": 3,
                "box": [680.0, 220.0, 810.0, 530.0],
                "keypoints": [
                    [745.0, 240.0], [740.0, 235.0], [750.0, 235.0], [730.0, 240.0], [760.0, 240.0],
                    [710.0, 280.0], [780.0, 280.0], # Vai (y=280)
                    [690.0, 220.0 if is_stretching_3 else 330.0], [800.0, 220.0 if is_stretching_3 else 330.0], # Khuỷu tay
                    [690.0, wrist3_left_y], [800.0, wrist3_right_y], # Cả 2 cổ tay cùng giơ cao qua vai khi vươn vai!
                    [720.0, 430.0], [770.0, 430.0],
                    [720.0, 480.0], [770.0, 480.0],
                    [720.0, 520.0], [770.0, 520.0]
                ],
                "conf": [0.88] * 17
            }
        ]
        data[str(f)] = tracks

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[Done] Đã cập nhật file mẫu '{OUTPUT_PATH}' với 150 frames (bao gồm kịch bản vươn vai).")

if __name__ == "__main__":
    generate()
