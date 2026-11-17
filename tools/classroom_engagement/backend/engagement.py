from collections import deque
import config as C

class Engagement:
    """
    Bộ đếm số lượng & Tính toán điểm sôi nổi:
    - Đếm số lượng tức thời (Frame hiện tại): đang giơ tay, đang ở bảng, tổng số người.
    - Đếm số lượng tích lũy (Cửa sổ trượt WINDOW_SEC): lượt giơ tay, lượt lên bảng, combo xung phong.
    - Tính tỉ lệ tham gia = số bạn có sự kiện / số người đang thấy.
    - Phân mức độ sôi nổi.
    """
    def __init__(self):
        self.events = deque()      # Lưu trữ (ts, kind, tid, points)
        self.last_raise = {}       # tid -> ts (lần giơ tay gần nhất để tính combo)

    def add(self, kind, tid, ts):
        """Thêm một sự kiện mới phát sinh."""
        pts = C.W_RAISE if kind == "raise" else C.W_BOARD
        self.events.append((ts, kind, tid, pts))

        if kind == "raise":
            self.last_raise[tid] = ts
        elif kind == "board" and (ts - self.last_raise.get(tid, -1e9) <= C.COMBO_SEC):
            # Cộng thêm điểm combo: giơ tay rồi lên bảng trong vòng 60s
            self.events.append((ts, "combo", tid, C.W_COMBO))
            return True  # Báo hiệu có combo
        return False

    def snapshot(self, n_people, now, currently_raising=0, currently_at_board=0, currently_stretching=0):
        """Tính toán các chỉ số trong cửa sổ trượt."""
        # Loại bỏ sự kiện ngoài cửa sổ 5 phút (300 giây)
        while self.events and now - self.events[0][0] > C.WINDOW_SEC:
            self.events.popleft()

        raises = sum(1 for e in self.events if e[1] == "raise")
        boards = sum(1 for e in self.events if e[1] == "board")
        combos = sum(1 for e in self.events if e[1] == "combo")
        score = sum(e[3] for e in self.events)

        # Số học sinh khác nhau có đóng góp sự kiện
        active = len({e[2] for e in self.events if e[1] != "combo"})
        rate = (active / n_people) if n_people > 0 else 0.0

        # Phân loại mức độ
        if score >= 8 or (rate >= 0.3 and active > 0):
            level = "sôi nổi"
        elif score >= 3:
            level = "bình thường"
        else:
            level = "yên lặng"

        return dict(
            # Số liệu tức thời (Live)
            currently_raising=currently_raising,
            currently_at_board=currently_at_board,
            currently_stretching=currently_stretching,
            people=n_people,

            # Số liệu tích lũy trong 5 phút
            raises=raises,
            boards=boards,
            combos=combos,
            score=score,
            active=active,
            rate=round(rate, 2),
            level=level
        )
