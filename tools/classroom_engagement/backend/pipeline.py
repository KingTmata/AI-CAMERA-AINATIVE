import time
import os
import sys
import cv2
import threading
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as C
import db
from detectors import PersonClassifier
from engagement import Engagement
from data_loader import PoseDataLoader

# Các cặp nối khung xương chuẩn COCO 17 điểm
SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),       # Hai cánh tay & vai
    (5, 11), (6, 12), (11, 12),                    # Thân mình
    (11, 13), (13, 15), (12, 14), (14, 16),        # Hai chân
    (0, 1), (0, 2), (1, 3), (2, 4)                 # Khuôn mặt
]

class Pipeline:
    """
    Pipeline chính hỗ trợ 3 chế độ hoạt động:
    - Chế độ 1: Đọc từ file dữ liệu pose/bbox đã có sẵn (pose_data.json) + Video -> Xử lý nhãn & đếm.
    - Chế độ 2: Nhận dữ liệu stream trực tiếp (API hoặc Python generator).
    - Chế độ 3: Tự động chạy YOLO Pose nếu chưa có file dữ liệu sẵn.
    """
    def __init__(self):
        self.eng = Engagement()
        self.frame_jpg = None
        self.stats = {
            "currently_raising": 0, "currently_at_board": 0, "currently_stretching": 0, "people": 0,
            "raises": 0, "boards": 0, "combos": 0,
            "score": 0, "active": 0, "rate": 0.0, "level": "yên lặng"
        }
        self.lock = threading.Lock()
        self.running = False
        self.mode = "unknown"
        self.data_loader = None
        self.yolo_model = None
        self.stream_classifier = None

    def _determine_mode(self):
        """Xác định chế độ hoạt động tối ưu dựa trên config và dữ liệu sẵn có."""
        if C.DATA_MODE == "file" or (C.DATA_MODE == "auto" and C.DATA_PATH and os.path.exists(C.DATA_PATH)):
            self.mode = "file"
            print(f"[Pipeline] Khởi động CHẾ ĐỘ 1: Đọc nhãn & tọa độ có sẵn từ '{C.DATA_PATH}' (Không dùng GPU/YOLO)")
            self.data_loader = PoseDataLoader(C.DATA_PATH)
        elif C.DATA_MODE == "stream":
            self.mode = "stream"
            print("[Pipeline] Khởi động CHẾ ĐỘ 2: Chờ nhận dữ liệu từ Stream/API ngoài.")
        else:
            self.mode = "yolo"
            print(f"[Pipeline] Khởi động CHẾ ĐỘ 3: Tự động chạy YOLO Pose từ '{C.MODEL_PATH}'.")
            from ultralytics import YOLO
            self.yolo_model = YOLO(C.MODEL_PATH)

    def process_detections(self, img, detections, classifier, now):
        """
        Lõi xử lý nhãn và đếm số lượng cho 1 frame:
        detections: list of dict [{"id": 1, "box": [...], "keypoints": [...], "conf": [...]}]
        """
        poly_pts = np.array(C.BOARD_POLY, dtype=np.int32)
        if len(poly_pts) >= 3:
            cv2.polylines(img, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(
                img, "Vung Bang (Board)",
                (poly_pts[0][0], max(25, poly_pts[0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

        currently_raising = 0
        currently_at_board = 0
        currently_stretching = 0
        n_people = len(detections)

        for det in detections:
            tid = det["id"]
            box = det["box"]
            kp = det["keypoints"]
            conf = det.get("conf", None)

            # 1. Xử lý nhãn hành vi
            res = classifier.classify(tid, kp, conf, box, now)

            if res["is_raising"]:
                currently_raising += 1
            if res["is_at_board"]:
                currently_at_board += 1
            if res.get("is_stretching"):
                currently_stretching += 1

            # 2. Đếm sự kiện tích lũy & lưu DB
            if res["is_new_raise"]:
                print(f"[Sự kiện] Học sinh #{tid} BẮT ĐẦU GIƠ TAY!")
                self.eng.add("raise", tid, now)
                db.save(now, "raise", tid)

            if res["is_new_board"]:
                print(f"[Sự kiện] Học sinh #{tid} BƯỚC LÊN BẢNG!")
                is_combo = self.eng.add("board", tid, now)
                db.save(now, "board", tid)
                if is_combo:
                    print(f"[COMBO] Học sinh #{tid} ĐẠT COMBO XUNG PHONG (+2 điểm)!")
                    res["label"] = "COMBO!"
                    res["color"] = (255, 0, 255)

            # 3. Vẽ nhãn & khung xương lên frame ảnh
            if C.DRAW_LABELS and len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                color = res["color"]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                tag_text = f"#{tid}: {res['label']}"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(
                    img, tag_text, (x1 + 3, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
                )

            if C.DRAW_SKELETON and kp is not None and len(kp) == 17:
                self._draw_skeleton(img, res.get("smoothed_kp", kp), conf)

        # 4. Tính toán toàn bộ chỉ số trong cửa sổ 5 phút
        current_stats = self.eng.snapshot(
            n_people=n_people,
            now=now,
            currently_raising=currently_raising,
            currently_at_board=currently_at_board,
            currently_stretching=currently_stretching
        )

        # 5. Vẽ bảng điều khiển HUD lên góc video
        self._draw_hud(img, current_stats, n_people)

        # 6. Mã hóa JPG đưa vào luồng MJPEG
        success, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            with self.lock:
                self.stats = current_stats
                self.frame_jpg = enc.tobytes()

        return current_stats

    def _draw_skeleton(self, img, kp, conf):
        """Vẽ các khớp và đường nối khung xương người."""
        for p1, p2 in SKELETON_PAIRS:
            c1 = conf[p1] if conf is not None and len(conf) > p1 else 1.0
            c2 = conf[p2] if conf is not None and len(conf) > p2 else 1.0
            if c1 > 0.4 and c2 > 0.4:
                pt1 = (int(kp[p1][0]), int(kp[p1][1]))
                pt2 = (int(kp[p2][0]), int(kp[p2][1]))
                if pt1 != (0, 0) and pt2 != (0, 0):
                    cv2.line(img, pt1, pt2, (0, 255, 255), 2)
        for i, pt in enumerate(kp):
            c = conf[i] if conf is not None and len(conf) > i else 1.0
            if c > 0.4 and (int(pt[0]), int(pt[1])) != (0, 0):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 165, 255), -1)

    def _draw_hud(self, img, stats, n_people):
        """Vẽ thanh thông số realtime trên video."""
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (330, 105), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        color_lv = (0, 220, 100) if stats["level"] == "sôi nổi" else (
            (0, 190, 255) if stats["level"] == "bình thường" else (160, 160, 160)
        )
        cv2.putText(
            img, f"TRANG THAI: {stats['level'].upper()}",
            (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_lv, 2
        )
        cv2.putText(
            img, f"Dang gio tay: {stats['currently_raising']} | Vuon vai (loc): {stats.get('currently_stretching', 0)} | O bang: {stats['currently_at_board']}",
            (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1
        )
        cv2.putText(
            img, f"Diem 5m: {stats['score']} | Ti le tham gia: {int(stats['rate']*100)}% ({stats['active']}/{n_people})",
            (18, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1
        )

    def feed_frame(self, img, detections):
        """Dành cho CHẾ ĐỘ 2: Nhận frame & danh sách pose trực tiếp từ bên ngoài."""
        now = time.time()
        if self.stream_classifier is None:
            self.stream_classifier = PersonClassifier(fps=25)
        return self.process_detections(img, detections, self.stream_classifier, now)

    def run(self, fps=25):
        """Vòng lặp chạy chính."""
        self.running = True
        self._determine_mode()

        source = C.SOURCE
        if isinstance(source, str) and not source.isdigit() and not source.startswith(("rtsp://", "http://", "https://")):
            if not os.path.exists(source):
                alt = os.path.join(C.BASE_DIR, source)
                if os.path.exists(alt):
                    source = alt
                else:
                    source = 0

        # Nếu là file video, đọc FPS gốc của video
        cap = None
        if self.mode in ["file", "yolo"]:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                v_fps = cap.get(cv2.CAP_PROP_FPS)
                if v_fps and v_fps > 5:
                    fps = v_fps
            else:
                print(f"[Pipeline] Không thể mở nguồn video: {source}")

        classifier = PersonClassifier(fps=fps)
        delay = 1.0 / fps

        # -------------------------------------------------------------
        # CHẾ ĐỘ 1: ĐỌC DỮ LIỆU ĐÃ CÓ SẴN (Không cần chạy YOLO)
        # -------------------------------------------------------------
        if self.mode == "file":
            frame_idx = 0
            while self.running and cap and cap.isOpened():
                t_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    # Lặp lại video và reset bộ lọc làm mượt tránh kéo lệch giữa cuối và đầu video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    classifier = PersonClassifier(fps=fps)
                    continue

                detections = self.data_loader.get_frame(frame_idx)
                self.process_detections(frame, detections, classifier, time.time())
                frame_idx += 1

                elapsed = time.time() - t_start
                if elapsed < delay:
                    time.sleep(delay - elapsed)

        # -------------------------------------------------------------
        # CHẾ ĐỘ 3: TỰ ĐỘNG CHẠY YOLO POSE INFERENCE
        # -------------------------------------------------------------
        elif self.mode == "yolo":
            while self.running:
                try:
                    for r in self.yolo_model.track(
                        source=source,
                        persist=True,
                        tracker="bytetrack.yaml",
                        stream=True,
                        verbose=False
                    ):
                        if not self.running:
                            break

                        img = r.orig_img.copy() if hasattr(r, "orig_img") and r.orig_img is not None else r.plot()
                        detections = []
                        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            kps = r.keypoints.xy.cpu().numpy()
                            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else [None] * len(boxes)
                            tids = r.boxes.id.int().tolist()
                            for b, k, c, tid in zip(boxes, kps, confs, tids):
                                detections.append({
                                    "id": tid,
                                    "box": b,
                                    "keypoints": k,
                                    "conf": c
                                })

                        self.process_detections(img, detections, classifier, time.time())
                except Exception as e:
                    print(f"[Pipeline YOLO Error]: {e}")
                    time.sleep(1)

        # -------------------------------------------------------------
        # CHẾ ĐỘ 2: ĐỢI DỮ LIỆU TỪ STREAM / API
        # -------------------------------------------------------------
        elif self.mode == "stream":
            print("[Pipeline] Sẵn sàng nhận frame tại endpoint /api/feed...")
            while self.running:
                time.sleep(0.5)

        if cap:
            cap.release()
