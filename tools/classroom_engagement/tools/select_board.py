"""
Công cụ hỗ trợ chấm chọn 4 điểm vùng bảng (BOARD_POLY) bằng chuột.
Cách dùng:
  python tools/select_board.py
Nhấp chuột trái 4 lần trên video để chọn 4 góc của bảng.
Nhấn 'r' để reset, nhấn 'q' hoặc 'ESC' để thoát và in ra tọa độ.
"""

import cv2
import os
import sys

# Thêm thư mục backend vào path để đọc config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
import config as C

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Đã chọn điểm {len(points)}: ({x}, {y})")
        if len(points) == 4:
            print("\n" + "=" * 50)
            print("ĐÃ CHỌN ĐỦ 4 ĐIỂM VÙNG BẢNG:")
            print(f"BOARD_POLY = {points}")
            print("Hãy sao chép dòng trên và dán vào file backend/config.py!")
            print("=" * 50 + "\n")

def main():
    global points
    source = C.SOURCE
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Lỗi] Không thể mở nguồn video/camera: {source}")
        return

    window_name = "Chon 4 diem Vung Bang (Click chuot trai, 'r' de reset, 'q' de thoat)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- HƯỚNG DẪN CHỌN VÙNG BẢNG ---")
    print("1. Nhấp chuột trái vào 4 góc của bảng theo thứ tự vòng cung.")
    print("2. Nhấn phím 'r' trên bàn phím nếu muốn chọn lại từ đầu.")
    print("3. Nhấn phím 'q' hoặc 'ESC' để hoàn tất.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Nếu hết video thì quay lại từ đầu
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # Vẽ các điểm đã chọn
        for i, pt in enumerate(points):
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)
            cv2.putText(frame, f"P{i+1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Vẽ đường nối đa giác
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], (0, 255, 255), 2)
        if len(points) == 4:
            cv2.line(frame, points[3], points[0], (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            points = []
            print("Đã reset các điểm. Hãy nhấp lại 4 điểm.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
