import json
import os
import sys
import csv
import pickle
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class PoseDataLoader:
    """
    Bộ nạp dữ liệu pose, bbox, keypoint đã được trích xuất từ trước.
    Hỗ trợ nạp từ JSON, CSV, PKL hoặc danh sách Python.
    Chuẩn hóa cấu trúc trả về theo từng frame:
    [
        {
            "id": int (track_id),
            "box": [x1, y1, x2, y2],
            "keypoints": np.ndarray (17, 2),
            "conf": np.ndarray (17,) hoặc None
        },
        ...
    ]
    """
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.frames_data = {}  # frame_idx (int) -> list of detection dicts
        self.total_frames = 0
        if data_path and os.path.exists(data_path):
            self.load(data_path)

    def load(self, path):
        self.data_path = path
        ext = os.path.splitext(path)[1].lower()

        if ext == ".json":
            self._load_json(path)
        elif ext == ".csv":
            self._load_csv(path)
        elif ext in [".pkl", ".pickle"]:
            self._load_pickle(path)
        else:
            raise ValueError(f"Định dạng file không hỗ trợ: {ext}. Hãy dùng .json, .csv, hoặc .pkl")

        self.total_frames = max(self.frames_data.keys()) + 1 if self.frames_data else 0
        print(f"[PoseDataLoader] Đã nạp thành công {len(self.frames_data)} frames từ: {path}")

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            # Dạng: {"0": [...], "1": [...]} hoặc {"frame_0": [...]}
            for k, items in raw.items():
                # Trích xuất số frame từ key
                idx = int(''.join(filter(str.isdigit, str(k))) or 0)
                self.frames_data[idx] = [self._normalize_item(it) for it in items]
        elif isinstance(raw, list):
            # Dạng danh sách: [{"frame_idx": 0, "tracks": [...]}, ...] hoặc [[...], [...]]
            for i, item in enumerate(raw):
                if isinstance(item, dict) and ("tracks" in item or "detections" in item or "objects" in item):
                    idx = item.get("frame_idx", item.get("frame", i))
                    tracks = item.get("tracks", item.get("detections", item.get("objects", [])))
                    self.frames_data[int(idx)] = [self._normalize_item(t) for t in tracks]
                elif isinstance(item, list):
                    self.frames_data[i] = [self._normalize_item(t) for t in item]
                elif isinstance(item, dict) and "id" in item:
                    # Toàn bộ list là các dòng riêng lẻ chứa frame_idx
                    idx = int(item.get("frame_idx", 0))
                    self.frames_data.setdefault(idx, []).append(self._normalize_item(item))

    def _load_csv(self, path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row.get("frame_idx", row.get("frame", 0)))
                tid = int(row.get("id", row.get("track_id", 0)))
                box = [
                    float(row.get("x1", 0)), float(row.get("y1", 0)),
                    float(row.get("x2", 0)), float(row.get("y2", 0))
                ]
                # Đọc 17 keypoints
                kps = []
                confs = []
                for k in range(17):
                    kx = float(row.get(f"kp{k}_x", row.get(f"x_{k}", 0)))
                    ky = float(row.get(f"kp{k}_y", row.get(f"y_{k}", 0)))
                    kc = float(row.get(f"kp{k}_c", row.get(f"c_{k}", 1.0)))
                    kps.append([kx, ky])
                    confs.append(kc)

                self.frames_data.setdefault(frame_idx, []).append({
                    "id": tid,
                    "box": box,
                    "keypoints": np.array(kps, dtype=np.float32),
                    "conf": np.array(confs, dtype=np.float32)
                })

    def _load_pickle(self, path):
        with open(path, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            for k, items in raw.items():
                idx = int(''.join(filter(str.isdigit, str(k))) or 0)
                self.frames_data[idx] = [self._normalize_item(it) for it in items]
        elif isinstance(raw, list):
            for i, items in enumerate(raw):
                self.frames_data[i] = [self._normalize_item(it) for it in items]

    def _normalize_item(self, item):
        tid = item.get("id", item.get("track_id", 0))
        box = item.get("box", item.get("bbox", [0, 0, 0, 0]))

        raw_kp = item.get("keypoints", item.get("kpts", []))
        if isinstance(raw_kp, list):
            raw_kp = np.array(raw_kp, dtype=np.float32)

        # Xử lý trường hợp keypoints là (17, 3) bao gồm cả confidence [x, y, conf]
        if raw_kp is not None and len(raw_kp) == 17:
            if raw_kp.ndim == 2 and raw_kp.shape[1] == 3:
                kp = raw_kp[:, :2]
                conf = raw_kp[:, 2]
            else:
                kp = raw_kp
                raw_conf = item.get("conf", item.get("scores", None))
                conf = np.array(raw_conf, dtype=np.float32) if raw_conf is not None else np.ones(17, dtype=np.float32)
        else:
            kp = np.zeros((17, 2), dtype=np.float32)
            conf = np.zeros(17, dtype=np.float32)

        return {
            "id": int(tid),
            "box": box,
            "keypoints": kp,
            "conf": conf
        }

    def get_frame(self, frame_idx):
        """Trả về danh sách các học sinh đã phát hiện trong frame_idx."""
        return self.frames_data.get(frame_idx, [])
