import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# ==============================================================================
# 1. CẤU HÌNH NGUỒN VÀO & CHẾ ĐỘ HOẠT ĐỘNG
# ==============================================================================
# Chế độ xử lý:
# - "auto"   : Tự động - Nếu có file dữ liệu pose (JSON/CSV) thì đọc file (Mode 1); 
#              ngược lại nếu có video thì inference (Mode 3); hỗ trợ nhận API stream (Mode 2).
# - "file"   : Mode 1 - Đọc trực tiếp từ file dữ liệu pose & bbox đã có sẵn (KHÔNG chạy YOLO).
# - "stream" : Mode 2 - Chờ nhận dữ liệu frame + pose từ code Python ngoài hoặc API /api/feed.
# - "yolo"   : Mode 3 - Tự chạy YOLO Pose inference nếu chỉ có video thô.
DATA_MODE = "auto"

# File dữ liệu pose/bbox đã có sẵn (JSON / CSV / PKL):
DATA_PATH = None
for candidate in ["pose_data.json", "labels.json", "tracks.json", "annotations.json"]:
    candidate_path = os.path.join(PROJECT_DIR, candidate)
    if os.path.exists(candidate_path):
        DATA_PATH = candidate_path
        break
    candidate_sub = os.path.join(BASE_DIR, candidate)
    if os.path.exists(candidate_sub):
        DATA_PATH = candidate_sub
        break

# Nguồn video (file video MP4, webcam 0, hoặc RTSP):
if os.path.exists(os.path.join(PROJECT_DIR, "video.mp4")):
    SOURCE = os.path.join(PROJECT_DIR, "video.mp4")
elif os.path.exists(os.path.join(BASE_DIR, "video.mp4")):
    SOURCE = os.path.join(BASE_DIR, "video.mp4")
elif os.path.exists("video.mp4"):
    SOURCE = "video.mp4"
else:
    SOURCE = 0

# Đường dẫn mô hình YOLO Pose (chỉ dùng cho Mode 3 hoặc công cụ xuất dữ liệu)
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "pose.pt")
if not os.path.exists(MODEL_PATH):
    if os.path.exists(os.path.join(PROJECT_DIR, "yolov8n-pose.pt")):
        MODEL_PATH = os.path.join(PROJECT_DIR, "yolov8n-pose.pt")
    else:
        MODEL_PATH = "yolov8n-pose.pt"

# ==============================================================================
# 2. CẤU HÌNH QUY TẮC XỬ LÝ NHÃN (LABELING RULES)
# ==============================================================================
CONF_MIN = 0.5
RAISE_THR = 0.3               # Ngưỡng cổ tay cao hơn vai theo tỉ lệ thân
RAISE_MIN_SEC = 0.5           # Thời gian giữ tay tối thiểu (giây) để xác nhận giơ tay
LOWER_SEC = 0.5               # Thời gian hạ tay để cho phép đếm lượt mới

# ==============================================================================
# 3. BỘ LỌC NGOẠI LỆ (VƯƠN VAI, HÍT THỞ, CHỐNG CẰM, GÃI ĐẦU)
# ==============================================================================
FILTER_TWO_HAND_STRETCH = True   # Giơ cả 2 tay -> Coi là vươn vai / hít thở (không tính giơ tay phát biểu)
ARM_SPREAD_MAX_RATIO = 1.15      # Dang tay ngang quá 1.15 lần độ rộng vai -> Vươn vai dang tay
CHIN_REST_MAX_RATIO = 0.28       # Cổ tay gần mặt/mũi trong khi khuỷu tay thấp -> Chống cằm
HEAD_TILT_BACK_CHECK = True      # Kiểm tra ngửa đầu ra sau khi vươn vai / ngáp

# Vùng bảng lớp học trên khung hình: [(x1, y1), (x2, y2), ...]
BOARD_POLY = [(900, 100), (1250, 100), (1250, 500), (900, 500)]
BOARD_MIN_SEC = 1.0           # Đứng trong vùng bảng tối thiểu 1.0 giây mới tính là lên bảng
BOARD_COOLDOWN = 30           # Cooldown giữa 2 lần tính lên bảng của cùng 1 người (giây)

# ==============================================================================
# 4. BỘ LỌC CHỐNG NHÁY NHÃN (ANTI-FLICKER & SMOOTHING)
# ==============================================================================
USE_ANTI_FLICKER = True          # Bật/tắt toàn bộ bộ lọc chống nháy & làm mượt
USE_ONE_EURO_FILTER = True       # Dùng One-Euro Filter (tự thích ứng tốc độ); False = EMA cố định
KP_SMOOTH_ALPHA = 0.45           # Hệ số EMA khi USE_ONE_EURO_FILTER = False
KP_ONE_EURO_MIN_CUTOFF = 1.0     # Tần số cắt khi đứng yên (Hz): nhỏ = mượt rung lắc mạnh hơn
KP_ONE_EURO_BETA = 0.01          # Hệ số theo tốc độ di chuyển: lớn = giảm độ trễ khi giơ tay nhanh

# Ngưỡng kép có vùng trễ (Hysteresis / Dead-zone) chống nháy ở ranh giới
RAISE_THR_HIGH = RAISE_THR       # Ngưỡng VÀO giơ tay (0.30)
RAISE_THR_LOW = 0.12             # Ngưỡng THOÁT giơ tay (0.12) - phải hạ tay thấp hơn rõ mới tắt
SPREAD_RATIO_ENTER = ARM_SPREAD_MAX_RATIO   # Vào STRETCH khi dang tay > 1.15 * độ rộng vai
SPREAD_RATIO_EXIT = 0.95                    # Thoát STRETCH khi thu tay về < 0.95 * độ rộng vai

# Biểu quyết đa số theo cửa sổ trượt & Thời gian giữ nhãn ổn định
VOTE_WINDOW_SEC = 0.4            # Độ dài cửa sổ biểu quyết (giây), ~10 frames @25fps
VOTE_MIN_RATIO = 0.65            # Nhãn cần chiếm >= 65% số frame trong cửa sổ mới được chuyển
STRETCH_MIN_SEC = 0.25           # Phải vươn vai liên tục >= 0.25s mới hiện nhãn VUON VAI
LABEL_HOLD_SEC = 0.5             # Giữ nhãn ổn định tối thiểu 0.5s trước khi cho phép đổi nhãn khác
BOARD_EXIT_SEC = 0.3             # Độ trễ thoát vùng bảng (tránh nháy khi đứng ở mép bảng)

# Lọc vận tốc cổ tay (Velocity Damping) chống vung tay / cử chỉ nhanh
WRIST_MAX_SPEED_RATIO = 4.0      # Tốc độ tối đa cho phép (tính theo lần chiều dài thân / giây)

# ==============================================================================
# 5. CỬA SỔ TRƯỢT & TRỌNG SỐ ĐIỂM SÔI NỔI
# ==============================================================================
WINDOW_SEC = 300              # Cửa sổ trượt tính điểm (5 phút = 300 giây)
COMBO_SEC = 60                # Combo nếu lên bảng trong vòng 60 giây sau khi giơ tay
W_RAISE = 1                   # Điểm mỗi lần giơ tay
W_BOARD = 3                   # Điểm mỗi lần lên bảng
W_COMBO = 2                   # Điểm cộng combo xung phong rồi lên bảng

# Cấu hình hiển thị
DRAW_LABELS = True            # Vẽ nhãn (GIƠ TAY, VƯƠN VAI, LÊN BẢNG, BÌNH THƯỜNG)
DRAW_SKELETON = True          # Vẽ khung xương keypoints nếu có dữ liệu
