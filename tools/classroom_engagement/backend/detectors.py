from collections import deque
from matplotlib.path import Path
import numpy as np
import config as C

# COCO Pose keypoint indices (17 keypoints):
# 0: mũi (NOSE), 1: mắt trái, 2: mắt phải, 3: tai trái, 4: tai phải
# 5: vai trái (L_SH), 6: vai phải (R_SH)
# 7: khuỷu tay trái (L_ELB), 8: khuỷu tay phải (R_ELB)
# 9: cổ tay trái (L_WR), 10: cổ tay phải (R_WR)
# 11: hông trái (L_HIP), 12: hông phải (R_HIP)
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SH, R_SH = 5, 6
L_ELB, R_ELB = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12


class KeypointSmoother:
    """
    Làm mượt tọa độ 17 keypoints theo thời gian để triệt tiêu rung lắc (jitter):
    - Mặc định sử dụng One-Euro Filter (lọc mạnh khi đứng yên, giảm trễ khi chuyển động nhanh).
    - Có thể chuyển về EMA cố định thông qua config USE_ONE_EURO_FILTER = False.
    - Bỏ qua các điểm khuyết (0, 0) hoặc độ tin cậy quá thấp để không kéo lệch khung xương.
    """
    def __init__(self, fps=25):
        self.dt = 1.0 / max(float(fps), 1.0)
        self.d_cutoff = 1.0
        self.state = {}  # tid -> {"x": np.ndarray(17, 2), "dx": np.ndarray(17, 2)}

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)

    def smooth(self, tid, kp, conf=None):
        kp_arr = np.asarray(kp, dtype=np.float32)
        if kp_arr.shape != (17, 2) or not C.USE_ANTI_FLICKER:
            return kp_arr

        if tid not in self.state:
            self.state[tid] = {
                "x": kp_arr.copy(),
                "dx": np.zeros_like(kp_arr)
            }
            return kp_arr.copy()

        st = self.state[tid]
        prev_x = st["x"]
        prev_dx = st["dx"]

        # Chỉ cập nhật những khớp có tọa độ hợp lệ
        valid = (kp_arr[:, 0] != 0) | (kp_arr[:, 1] != 0)
        if conf is not None and len(conf) == 17:
            valid = valid & (np.asarray(conf) > C.CONF_MIN * 0.5)

        cur_x = np.where(valid[:, None], kp_arr, prev_x)

        if not C.USE_ONE_EURO_FILTER:
            alpha = float(np.clip(C.KP_SMOOTH_ALPHA, 0.05, 1.0))
            x_hat = alpha * cur_x + (1.0 - alpha) * prev_x
            st["x"] = x_hat
            return x_hat.copy()

        # One-Euro Filter vector hóa cho toàn bộ 17 keypoints
        dx = (cur_x - prev_x) / self.dt
        a_d = self._alpha(self.d_cutoff, self.dt)
        dx_hat = a_d * dx + (1.0 - a_d) * prev_dx

        cutoff = C.KP_ONE_EURO_MIN_CUTOFF + C.KP_ONE_EURO_BETA * np.abs(dx_hat)
        a = self._alpha(cutoff, self.dt)
        x_hat = a * cur_x + (1.0 - a) * prev_x

        st["x"] = x_hat
        st["dx"] = dx_hat
        return x_hat.copy()


def analyze_arm_pose(kp, conf, prev_state="NORMAL"):
    """
    Phân tích tư thế cánh tay và thân người với ngưỡng kép (Hysteresis) & các bộ lọc ngoại lệ:
    - prev_state: trạng thái ổn định hiện tại ("RAISE", "STRETCH", "NORMAL") để tạo vùng trễ chống nháy.
    Trả về một trong các nhãn:
    - "RAISE"   : Giơ 1 tay phát biểu hợp lệ
    - "STRETCH" : Ngoại lệ vươn vai / hít thở / dang 2 tay
    - "NORMAL"  : Bình thường / Chống cằm / Gãi đầu
    """
    shoulder_width = max(abs(kp[L_SH][0] - kp[R_SH][0]), 1.0)
    torso = max(shoulder_width * 1.5, 1.0)

    # Ngưỡng kép có vùng trễ (Hysteresis) cho độ cao tay
    if C.USE_ANTI_FLICKER and prev_state == "RAISE":
        raise_thr = C.RAISE_THR_LOW
    else:
        raise_thr = C.RAISE_THR_HIGH if C.USE_ANTI_FLICKER else C.RAISE_THR

    # Ngưỡng kép có vùng trễ cho độ dang rộng cánh tay
    if C.USE_ANTI_FLICKER and prev_state == "STRETCH":
        spread_ratio = C.SPREAD_RATIO_EXIT
    else:
        spread_ratio = C.SPREAD_RATIO_ENTER if C.USE_ANTI_FLICKER else C.ARM_SPREAD_MAX_RATIO

    # 1. Kiểm tra điều kiện cơ bản: cổ tay có ở cao hơn vai không
    def check_arm_up(sh, wr):
        if conf is not None and len(conf) > max(sh, wr):
            if min(conf[sh], conf[wr]) <= C.CONF_MIN:
                return False
        # Lưu ý: y=0 ở mép trên cùng ảnh, y_wrist < y_shoulder nghĩa là giơ cao hơn vai
        return kp[wr][1] < kp[sh][1] - raise_thr * torso

    left_up = check_arm_up(L_SH, L_WR)
    right_up = check_arm_up(R_SH, R_WR)

    # Nếu không tay nào giơ cao -> Bình thường
    if not left_up and not right_up:
        return "NORMAL"

    # --------------------------------------------------------------------------
    # BỘ LỌC NGOẠI LỆ 1: VƯƠN VAI / HÍT THỞ BẰNG CẢ 2 TAY (Two-hand raise)
    # --------------------------------------------------------------------------
    if C.FILTER_TWO_HAND_STRETCH and left_up and right_up:
        return "STRETCH"

    # --------------------------------------------------------------------------
    # BỘ LỌC NGOẠI LỆ 2: VƯƠN VAI DANG TAY SANG HAI BÊN (Wide arm spread)
    # --------------------------------------------------------------------------
    if left_up:
        spread_x = abs(kp[L_WR][0] - kp[L_SH][0])
        if spread_x > spread_ratio * shoulder_width:
            return "STRETCH"

    if right_up:
        spread_x = abs(kp[R_WR][0] - kp[R_SH][0])
        if spread_x > spread_ratio * shoulder_width:
            return "STRETCH"

    # --------------------------------------------------------------------------
    # BỘ LỌC NGOẠI LỆ 3: NGỬA ĐẦU RA SAU KHI VƯƠN VAI / NGÁP (Head tilt back)
    # --------------------------------------------------------------------------
    if C.HEAD_TILT_BACK_CHECK and conf is not None and len(conf) > max(NOSE, L_EAR, R_EAR):
        if conf[NOSE] > C.CONF_MIN and min(conf[L_EAR], conf[R_EAR]) > C.CONF_MIN:
            avg_ear_y = (kp[L_EAR][1] + kp[R_EAR][1]) / 2
            if kp[NOSE][1] <= avg_ear_y - 0.08 * torso:
                return "STRETCH"

    # --------------------------------------------------------------------------
    # BỘ LỌC NGOẠI LỆ 4: CHỐNG CẰM / GÃI MẶT (Elbow on desk, wrist near face)
    # --------------------------------------------------------------------------
    if left_up and conf is not None and len(conf) > max(NOSE, L_ELB):
        dist_face = np.linalg.norm(kp[L_WR] - kp[NOSE])
        if dist_face < C.CHIN_REST_MAX_RATIO * torso and kp[L_ELB][1] > kp[L_SH][1] - 0.1 * torso:
            return "NORMAL"

    if right_up and conf is not None and len(conf) > max(NOSE, R_ELB):
        dist_face = np.linalg.norm(kp[R_WR] - kp[NOSE])
        if dist_face < C.CHIN_REST_MAX_RATIO * torso and kp[R_ELB][1] > kp[R_SH][1] - 0.1 * torso:
            return "NORMAL"

    return "RAISE"


def is_hand_above_shoulder(kp, conf):
    """Hàm tương thích ngược: trả về True nếu tư thế là giơ tay hợp lệ."""
    return analyze_arm_pose(kp, conf) == "RAISE"


# Alias tương thích
is_raised = is_hand_above_shoulder


class RaiseHandDetector:
    """
    Máy trạng thái phát hiện hành vi giơ tay, lọc ngoại lệ & chống nháy nhãn:
    - Làm mượt keypoints bằng One-Euro Filter (chống rung).
    - Ngưỡng kép có vùng trễ Hysteresis (chống nhảy ở vùng biên).
    - Biểu quyết đa số theo cửa sổ trượt (bỏ qua nhiễu thoáng qua 1-3 frames).
    - Debounce đối xứng cho cả RAISE và STRETCH + khóa giữ nhãn tối thiểu (Label Hold).
    - Lọc vận tốc cổ tay khi vung tay / cử chỉ động liên tục.
    """
    def __init__(self, fps=25):
        self.fps = max(float(fps), 1.0)
        self.dt = 1.0 / self.fps
        self.up_need = max(1, int(self.fps * C.RAISE_MIN_SEC))
        self.down_need = max(1, int(self.fps * C.LOWER_SEC))
        self.stretch_need = max(1, int(self.fps * C.STRETCH_MIN_SEC)) if C.USE_ANTI_FLICKER else 1
        self.hold_need = max(0, int(self.fps * C.LABEL_HOLD_SEC)) if C.USE_ANTI_FLICKER else 0
        self.vote_len = max(3, min(int(self.fps * C.VOTE_WINDOW_SEC), self.up_need))

        self.smoother = KeypointSmoother(fps=self.fps)
        self.prev_raw_wrists = {}
        self.fast_counts = {}
        self.s = {}

    def _is_moving_too_fast(self, tid, raw_kp, conf):
        if not C.USE_ANTI_FLICKER:
            return False
        kp_arr = np.asarray(raw_kp, dtype=np.float32)
        if kp_arr.shape != (17, 2):
            return False

        shoulder_width = max(abs(kp_arr[L_SH][0] - kp_arr[R_SH][0]), 1.0)
        torso = max(shoulder_width * 1.5, 1.0)
        max_step = C.WRIST_MAX_SPEED_RATIO * torso * self.dt

        prev = self.prev_raw_wrists.get(tid)
        cur_wrists = kp_arr[[L_WR, R_WR]].copy()
        self.prev_raw_wrists[tid] = cur_wrists

        if prev is None:
            return False

        fast = False
        for idx, wr_idx in enumerate([L_WR, R_WR]):
            conf_ok = True
            if conf is not None and len(conf) > wr_idx:
                conf_ok = conf[wr_idx] > C.CONF_MIN
            if conf_ok and np.any(cur_wrists[idx] != 0) and np.any(prev[idx] != 0):
                if float(np.linalg.norm(cur_wrists[idx] - prev[idx])) > max_step:
                    fast = True
                    break

        cnt = self.fast_counts.get(tid, 0) + 1 if fast else 0
        self.fast_counts[tid] = cnt
        return cnt >= 2

    def update(self, tid, kp, conf, smoothed_kp=None):
        """
        Trả về tuple: (is_new_event, is_currently_raising, is_stretching)
        """
        st = self.s.setdefault(tid, {
            "up": 0,
            "down": 0,
            "stretch": 0,
            "is_raising": False,
            "is_stretching": False,
            "hold": self.hold_need,
            "history": deque(maxlen=self.vote_len)
        })

        kp_used = smoothed_kp if smoothed_kp is not None else self.smoother.smooth(tid, kp, conf)
        cur_label_state = "RAISE" if st["is_raising"] else ("STRETCH" if st["is_stretching"] else "NORMAL")

        if self._is_moving_too_fast(tid, kp, conf):
            # Nếu cổ tay đang vung nhanh liên tục -> giữ nguyên trạng thái hiện tại, không đếm mới
            raw_pose = cur_label_state
        else:
            raw_pose = analyze_arm_pose(kp_used, conf, prev_state=cur_label_state)

        st["history"].append(raw_pose)
        st["hold"] += 1

        # Tính tỉ lệ đồng thuận trong cửa sổ biểu quyết
        n_hist = len(st["history"])
        raise_ratio = sum(1 for x in st["history"] if x == "RAISE") / n_hist
        stretch_ratio = sum(1 for x in st["history"] if x == "STRETCH") / n_hist
        normal_ratio = sum(1 for x in st["history"] if x == "NORMAL") / n_hist

        if not C.USE_ANTI_FLICKER:
            # Chế độ không chống nháy (fallback thuần túy)
            return self._update_legacy(st, raw_pose)

        # Cập nhật các bộ đếm có khả năng kháng nhiễu (không reset cứng khi gặp 1 frame nhiễu)
        if raw_pose == "RAISE":
            st["up"] += 1
            st["down"] = 0
            st["stretch"] = max(0, st["stretch"] - 1)
        elif raw_pose == "STRETCH":
            st["stretch"] += 1
            st["down"] = 0
            st["up"] = max(0, st["up"] - 1)
        else:  # NORMAL
            st["down"] += 1
            if st["down"] >= self.down_need:
                st["up"] = 0
                st["stretch"] = 0

        vote_ok = C.VOTE_MIN_RATIO
        can_switch = st["hold"] >= self.hold_need

        # 1. Xử lý chuyển sang trạng thái STRETCH (cần đủ stretch_need + đồng thuận)
        if st["stretch"] >= self.stretch_need and stretch_ratio >= vote_ok:
            if not st["is_stretching"] and (can_switch or not st["is_raising"]):
                st["is_raising"] = False
                st["is_stretching"] = True
                st["up"] = 0
                st["hold"] = 0
            return False, st["is_raising"], st["is_stretching"]

        # 2. Xử lý chuyển sang trạng thái RAISE (cần đủ up_need + đồng thuận)
        if st["up"] >= self.up_need and raise_ratio >= vote_ok:
            if not st["is_raising"] and (can_switch or not st["is_stretching"]):
                st["is_raising"] = True
                st["is_stretching"] = False
                st["stretch"] = 0
                st["hold"] = 0
                return True, True, False  # Rising edge: sự kiện giơ tay mới!
            return False, st["is_raising"], st["is_stretching"]

        # 3. Xử lý thoát trạng thái về NORMAL khi hạ tay rõ ràng
        if st["is_stretching"]:
            stretch_exit_need = max(2, self.stretch_need)
            if st["down"] >= stretch_exit_need and normal_ratio >= vote_ok and can_switch:
                st["is_stretching"] = False
                st["stretch"] = 0
                st["hold"] = 0
        elif st["is_raising"]:
            if st["down"] >= self.down_need and normal_ratio >= vote_ok and can_switch:
                st["is_raising"] = False
                st["up"] = 0
                st["hold"] = 0

        return False, st["is_raising"], st["is_stretching"]

    def _update_legacy(self, st, pose_type):
        if pose_type == "RAISE":
            st["up"] += 1
            st["down"] = 0
            st["is_stretching"] = False
            if st["up"] >= self.up_need and not st["is_raising"]:
                st["is_raising"] = True
                return True, True, False
            return False, st["is_raising"], False
        elif pose_type == "STRETCH":
            st["up"] = 0
            st["is_raising"] = False
            st["is_stretching"] = True
            return False, False, True
        else:
            st["down"] += 1
            st["is_stretching"] = False
            if st["down"] >= self.down_need:
                st["up"] = 0
                st["is_raising"] = False
            return False, st["is_raising"], False


class BoardDetector:
    """
    Máy trạng thái phát hiện hành vi lên bảng:
    - Kiểm tra tọa độ hông (hoặc đáy bounding box) nằm trong đa giác BOARD_POLY.
    - Có độ trễ thoát vùng bảng (BOARD_EXIT_SEC) chống nháy khi đứng ở mép đa giác.
    """
    def __init__(self, fps=25):
        self.poly = Path(C.BOARD_POLY) if len(C.BOARD_POLY) >= 3 else None
        self.need = max(1, int(fps * C.BOARD_MIN_SEC))
        self.exit_need = max(1, int(fps * C.BOARD_EXIT_SEC)) if C.USE_ANTI_FLICKER else 1
        self.inside = {}    # tid -> frame_count trong vùng bảng
        self.outside = {}   # tid -> frame_count ngoài vùng bảng
        self.active = {}    # tid -> bool trạng thái đang ở bảng
        self.last = {}      # tid -> timestamp lần ghi nhận trước

    def update(self, tid, kp, conf, box, now):
        """
        Trả về tuple: (is_new_event, is_currently_at_board)
        """
        if not self.poly:
            return False, False

        conf_ok = False
        if conf is not None and len(conf) > max(L_HIP, R_HIP):
            conf_ok = min(conf[L_HIP], conf[R_HIP]) > C.CONF_MIN

        if conf_ok:
            pos = ((kp[L_HIP][0] + kp[R_HIP][0]) / 2, (kp[L_HIP][1] + kp[R_HIP][1]) / 2)
        else:
            x1, y1, x2, y2 = box
            pos = ((x1 + x2) / 2, y2)  # Đáy bounding box

        if self.poly.contains_point(pos):
            self.inside[tid] = self.inside.get(tid, 0) + 1
            self.outside[tid] = 0
            if self.inside[tid] >= self.need:
                self.active[tid] = True
                if now - self.last.get(tid, -1e9) > C.BOARD_COOLDOWN:
                    self.last[tid] = now
                    return True, True
            return False, self.active.get(tid, False)
        else:
            self.outside[tid] = self.outside.get(tid, 0) + 1
            if self.outside[tid] >= self.exit_need:
                self.inside[tid] = 0
                self.active[tid] = False
            return False, self.active.get(tid, False)


class PersonClassifier:
    """
    Xử lý nhãn tổng hợp cho từng cá nhân trong frame (đã tích hợp làm mượt & chống nháy):
    Gán nhãn:
    - "GIO TAY"        : Xanh lá cây (Hợp lệ)
    - "VUON VAI (LỌC)" : Vàng cam (Ngoại lệ - Đã lọc bỏ)
    - "O BANG"         : Xanh dương / Cyan
    - "BINH THUONG"    : Xám trắng
    """
    def __init__(self, fps=25):
        self.smoother = KeypointSmoother(fps)
        self.raise_detector = RaiseHandDetector(fps)
        self.board_detector = BoardDetector(fps)

    def classify(self, tid, kp, conf, box, now):
        smoothed_kp = self.smoother.smooth(tid, kp, conf)
        is_new_raise, is_raising, is_stretching = self.raise_detector.update(
            tid, kp, conf, smoothed_kp=smoothed_kp
        )
        is_new_board, is_at_board = self.board_detector.update(
            tid, smoothed_kp, conf, box, now
        )

        if is_raising:
            label = "GIO TAY"
            color = (0, 255, 0)       # Xanh lá cây
        elif is_stretching:
            label = "VUON VAI (LỌC)"
            color = (0, 165, 255)     # Vàng cam - cảnh báo ngoại lệ!
        elif is_at_board:
            label = "O BANG"
            color = (255, 191, 0)     # Xanh dương / Cyan
        else:
            label = "BINH THUONG"
            color = (200, 200, 200)   # Xám trắng

        return {
            "tid": tid,
            "label": label,
            "color": color,
            "smoothed_kp": smoothed_kp,
            "is_raising": is_raising,
            "is_new_raise": is_new_raise,
            "is_stretching": is_stretching,
            "is_at_board": is_at_board,
            "is_new_board": is_new_board
        }
