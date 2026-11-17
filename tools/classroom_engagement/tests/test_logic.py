import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
import config as C
from detectors import (
    RaiseHandDetector,
    BoardDetector,
    KeypointSmoother,
    PersonClassifier,
    is_raised,
    analyze_arm_pose,
)
from engagement import Engagement
import db


def make_base_kp():
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[0] = [450, 220]   # mũi (NOSE)
    kp[3] = [430, 225]   # tai trái
    kp[4] = [470, 225]   # tai phải
    kp[5] = [400, 300]   # vai trái: x=400, y=300
    kp[6] = [500, 300]   # vai phải: x=500, y=300 (rộng vai = 100, torso = 150)
    kp[7] = [380, 350]   # khuỷu tay trái
    kp[8] = [520, 350]   # khuỷu tay phải
    kp[9] = [400, 400]   # cổ tay trái hạ thấp
    kp[10] = [500, 400]  # cổ tay phải hạ thấp
    kp[11] = [420, 450]  # hông trái
    kp[12] = [480, 450]  # hông phải
    return kp


def test_detectors_and_engagement():
    print("Testing detector logic, exception handling & anti-flicker filters...")
    fps = 25
    raise_detector = RaiseHandDetector(fps=fps)
    board_detector = BoardDetector(fps=fps)
    eng = Engagement()

    kp = make_base_kp()
    conf = np.ones(17, dtype=np.float32)

    # 1A. Chưa giơ tay
    assert analyze_arm_pose(kp, conf) == "NORMAL"
    assert not is_raised(kp, conf)
    is_new, is_curr, is_stretch = raise_detector.update(1, kp, conf)
    assert not is_new and not is_curr and not is_stretch

    # 1B. Giơ 1 tay phải hợp lệ: y = 200
    kp[10] = [500, 200]
    assert analyze_arm_pose(kp, conf) == "RAISE", "Single vertical arm should be RAISE"
    assert is_raised(kp, conf)

    up_need = raise_detector.up_need
    triggered = False
    for _ in range(up_need):
        is_new, is_curr, is_stretch = raise_detector.update(1, kp, conf)
        if is_new:
            triggered = True
    assert triggered, f"Raise detector should trigger after {up_need} frames"
    assert is_curr and not is_stretch

    # 1C. NGOẠI LỆ: Vươn vai bằng cả 2 tay (cần duy trì đủ stretch_need mới kích hoạt)
    kp_stretch = make_base_kp()
    kp_stretch[9] = [400, 180]
    kp_stretch[10] = [500, 200]
    assert analyze_arm_pose(kp_stretch, conf) == "STRETCH", "Both arms up should be STRETCH"

    # 1 frame thoáng qua không được kích hoạt STRETCH ngay (chống nháy)
    is_new, is_curr, is_stretch = raise_detector.update(2, kp_stretch, conf)
    assert not is_stretch, "Single frame stretch should NOT flicker ON immediately"

    # Duy trì đủ thời gian vươn vai -> kích hoạt nhãn STRETCH ổn định, không cộng điểm giơ tay
    for _ in range(max(raise_detector.stretch_need, raise_detector.vote_len) + 2):
        is_new, is_curr, is_stretch = raise_detector.update(2, kp_stretch, conf)
        assert not is_new and not is_curr, "Stretch should NEVER trigger raise event"
    assert is_stretch, "Sustained two-hand stretch should activate is_stretch"

    # 1D. NGOẠI LỆ: Dang tay quá rộng
    kp_wide = make_base_kp()
    kp_wide[10] = [700, 200]
    assert analyze_arm_pose(kp_wide, conf) == "STRETCH", "Wide arm spread should be STRETCH"

    # 1E. NGOẠI LỆ: Chống cằm
    kp_chin = make_base_kp()
    kp_chin[10] = [455, 225]
    kp_chin[8] = [510, 360]
    assert analyze_arm_pose(kp_chin, conf) == "NORMAL", "Resting chin should be NORMAL"

    # ------------------------------------------------------------------
    # 2. KIỂM THỬ BỘ LỌC CHỐNG NHÁY NHÃN (ANTI-FLICKER TESTS)
    # ------------------------------------------------------------------

    # 2A. One-Euro Keypoint Smoother: giảm rung lắc jitter
    smoother = KeypointSmoother(fps=fps)
    raw_ys = []
    smooth_ys = []
    for i in range(20):
        kp_jit = make_base_kp()
        jitter = 8.0 if i % 2 == 0 else -8.0
        kp_jit[10] = [500, 255.0 + jitter]
        raw_ys.append(kp_jit[10][1])
        sm = smoother.smooth(99, kp_jit, conf)
        smooth_ys.append(float(sm[10][1]))
    assert np.std(smooth_ys[4:]) < np.std(raw_ys[4:]), "Smoother must reduce coordinate jitter"

    # 2B. Hysteresis (Ngưỡng kép vùng trễ):
    # Người tid=10 giơ tay chuẩn vào trạng thái RAISE
    kp_raise = make_base_kp()
    kp_raise[10] = [500, 200]
    for _ in range(up_need + 2):
        raise_detector.update(10, kp_raise, conf)

    # Hạ nhẹ vào vùng dead-zone: 255 < y=270 < 282 (thấp hơn ngưỡng vào 255, cao hơn ngưỡng thoát 282)
    kp_deadzone = make_base_kp()
    kp_deadzone[10] = [500, 270]
    for _ in range(15):
        is_new, is_curr, is_stretch = raise_detector.update(10, kp_deadzone, conf)
        assert not is_new and is_curr and not is_stretch, "Dead-zone must keep RAISE stable without flicker"

    # 2C. Kháng nhiễu thoáng qua (Majority Voting + Stretch Debounce):
    # Đang giơ tay tid=10, chèn 2 frame nhiễu STRETCH và 2 frame nhiễu NORMAL xen kẽ
    for noisy_kp in [kp_stretch, kp_stretch, make_base_kp(), make_base_kp()]:
        is_new, is_curr, is_stretch = raise_detector.update(10, noisy_kp, conf)
        assert is_curr and not is_stretch and not is_new, "Transient noise must NOT flicker label or reset RAISE"

    # 2D. Lọc vận tốc cổ tay khi vung tay nhanh liên tục (Velocity Damping):
    rd_vel = RaiseHandDetector(fps=fps)
    kp_wave = make_base_kp()
    kp_wave[10] = [500, 240]
    rd_vel.update(20, kp_wave, conf)
    # Vung tay qua lại cực nhanh trong vùng giơ tay suốt 15 frame (mỗi frame dịch 80px ~ 2000px/s)
    for i in range(15):
        kp_wave[10] = [440 if i % 2 == 0 else 560, 210]
        is_new, is_curr, _ = rd_vel.update(20, kp_wave, conf)
        assert not is_new and not is_curr, "Rapid continuous hand waving must be damped"

    # 2E. Chống nháy ở mép vùng bảng (BoardDetector Exit Grace):
    in_board_box = [1000, 200, 1100, 450]
    kp_in = make_base_kp()
    kp_in[11] = [1030, 300]
    kp_in[12] = [1070, 300]
    for f in range(board_detector.need + 2):
        board_detector.update(30, kp_in, conf, in_board_box, now=10.0 + f * 0.04)
    # Chèn 2 frame hông bị rung ra ngoài đa giác -> vẫn giữ trạng thái ở bảng
    kp_out = make_base_kp()
    kp_out[11] = [800, 300]
    kp_out[12] = [850, 300]
    _, is_at_board = board_detector.update(30, kp_out, conf, [800, 200, 850, 450], now=12.0)
    assert is_at_board, "1 noisy frame outside board must NOT flicker O BANG off"

    # 3. Test PersonClassifier trả về smoothed_kp & Engagement
    clf = PersonClassifier(fps=fps)
    res = clf.classify(40, kp_raise, conf, [350, 180, 520, 500], now=100.0)
    assert "smoothed_kp" in res and res["smoothed_kp"].shape == (17, 2)

    t0 = 1000.0
    eng.add("raise", tid=1, ts=t0)
    db.save(t0, "raise", 1)
    snap = eng.snapshot(n_people=5, now=t0 + 5)
    assert snap["raises"] == 1 and snap["score"] == 1 and snap["level"] == "yên lặng"

    t1 = t0 + 10.0
    eng.add("board", tid=1, ts=t1)
    db.save(t1, "board", 1)
    snap = eng.snapshot(n_people=5, now=t1 + 1)
    assert snap["boards"] == 1 and snap["combos"] == 1 and snap["score"] == 6 and snap["level"] == "bình thường"

    eng.add("raise", tid=2, ts=t1 + 5)
    snap2 = eng.snapshot(n_people=5, now=t1 + 6, currently_stretching=1)
    assert snap2["active"] == 2 and snap2["rate"] == 0.4 and snap2["level"] == "sôi nổi"
    assert snap2["currently_stretching"] == 1

    print("ALL TESTS PASSED! Exception handling & Anti-flicker filters work accurately!")


if __name__ == "__main__":
    test_detectors_and_engagement()
