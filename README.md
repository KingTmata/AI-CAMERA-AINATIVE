# HƯỚNG DẪN CÀI ĐẶT VÀ KHỞI CHẠY DỰ ÁN

> **Đề tài:** Nghiên cứu xây dựng hệ thống nhận dạng và đánh giá tính tương tác của sinh viên trong quá trình học tập trên lớp với hệ sinh thái AI-Native  
> **Mục tiêu chính thức:** Xem tại [tasks/plan.md](tasks/plan.md#mục-tiêu-và-yêu-cầu-chính-thức--15102026)

---

## ⚡ Bảng lệnh khởi chạy nhanh (Quick Start)

| Thành phần | Lệnh thực thi / File khởi chạy | Địa chỉ truy cập / Cổng | Ghi chú |
| :--- | :--- | :--- | :--- |
| **1. Cài đặt môi trường** | `.\setup_demo.cmd` | N/A | Tự động tạo `.venv`, cài đặt PyTorch, Ultralytics và tải weights ban đầu |
| **2. Giao diện Web chính (Khuyên dùng)** | `.\run_demo.cmd` | **[http://127.0.0.1:8090](http://127.0.0.1:8090)** | Giao diện AI nhận diện học sinh, vẽ khung xương Pose, HUD và bảng đếm tương tác |
| **3. Backend Đánh giá Sôi nổi** | `.\run_engagement.cmd` | **[http://127.0.0.1:8000](http://127.0.0.1:8000)** | API & WebSocket phân tích tương tác, lọc ngoại lệ vươn vai, tính điểm trượt |
| **4. Trang Thuyết minh Đề tài** | `.\run_project_site.cmd` | **[http://127.0.0.1:8765](http://127.0.0.1:8765)** | Thuyết minh tổng quan đề tài, sơ đồ kiến trúc hệ thống và luồng dữ liệu |
| **5. Phân tích Video qua Terminal (TUI)** | `.\run_video.cmd --tui` | Terminal Console | Xử lý video trực tiếp qua giao diện dòng lệnh không cần trình duyệt |

> [!TIP]
> **Khuyến nghị vận hành chuẩn:** Để có trải nghiệm đầy đủ nhất, hãy bật đồng thời cả **`run_demo.cmd`** (cổng 8090) và **`run_engagement.cmd`** (cổng 8000). Giao diện web 8090 sẽ tự động kết nối qua CORS tới backend 8000 để cập nhật đồng bộ các chỉ số sôi nổi theo thời gian thực.

---

## 1. Yêu cầu hệ thống

* **Hệ điều hành:** Windows 10 / Windows 11 (64-bit) (Hỗ trợ cả Linux/macOS qua lệnh Python tương đương)
* **Phiên bản Python:** **Python 3.11** (khuyến nghị tải từ [python.org](https://www.python.org/downloads/release/python-3119/) và tích chọn *Add Python to PATH* khi cài đặt)
* **Phần cứng:**
  * CPU: Intel Core i5 / AMD Ryzen 5 trở lên
  * RAM: Tối thiểu 8 GB
  * GPU (không bắt buộc): Tự động phát hiện và hỗ trợ NVIDIA CUDA hoặc Intel iGPU (OpenVINO); nếu không có GPU hệ thống vẫn chạy ổn định trên CPU.

---

## 2. Cài đặt môi trường tự động (Lần đầu tiên)

Nếu dự án chưa có môi trường ảo `.venv` hoặc vừa chuyển sang máy mới:

1. Mở thư mục gốc của dự án.
2. Nhấp đúp chuột vào file:
   ```text
   setup_demo.cmd
   ```
   *(Hoặc mở PowerShell / Command Prompt tại thư mục dự án và gõ: `.\setup_demo.cmd`)*.
3. Chờ script tự động:
   * Tạo môi trường ảo `.venv` với Python 3.11.
   * Cài đặt PyTorch CPU/GPU, torchvision và các thư viện cần thiết (`ultralytics`, `fastapi`, `uvicorn`, `opencv-python`, `lap`, `av`).
   * Tải các file trọng số AI ban đầu vào thư mục `weights/` (`yolo26n.pt`, `yolo26n-pose.pt`).
4. Khi hoàn tất, cửa sổ sẽ hiển thị thông báo `Setup complete. Open run_demo.cmd or run_video.cmd.`

---

## 3. Các cách khởi chạy hệ thống

### Cách 1: Chạy Web Demo chính (Khuyên dùng)
Giao diện Web tập trung đầy đủ tính năng: phát hiện đối tượng, vẽ khung xương Pose, làm mượt chuyển động và thanh HUD đếm tương tác học tập.

1. Nhấp đúp vào file:
   ```text
   run_demo.cmd
   ```
2. Giữ nguyên cửa sổ console đang chạy máy chủ.
3. Mở trình duyệt web và truy cập địa chỉ:
   👉 **[http://127.0.0.1:8090](http://127.0.0.1:8090)**

---

### Cách 2: Chạy Backend Đánh giá mức độ sôi nổi (Engagement Server)
Phân hệ chuyên sâu phân tích hành vi tương tác, lọc ngoại lệ vươn vai và đồng bộ chỉ số sôi nổi với Web Demo qua CORS:

1. Nhấp đúp vào file:
   ```text
   run_engagement.cmd
   ```
2. Giữ nguyên cửa sổ console đang chạy Uvicorn.
3. Địa chỉ dịch vụ:
   👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)** (API kiểm tra: `http://127.0.0.1:8000/api/status`)

> [!TIP]
> **Khuyến nghị:** Bạn nên bật đồng thời cả **`run_demo.cmd`** (cổng 8090) và **`run_engagement.cmd`** (cổng 8000) để giao diện web chính vừa phân tích hình ảnh AI vừa nhận dữ liệu đánh giá tương tác thời gian thực.

---

### Cách 3: Chạy phân tích Video qua dòng lệnh (CLI / TUI)
Dành cho trường hợp muốn xử lý video hàng loạt hoặc kiểm tra pipeline mà không cần mở trình duyệt:

* **Chạy giao diện dòng lệnh trực quan (TUI):**
  ```powershell
  .\run_video.cmd --tui
  ```
* **Chạy lệnh xử lý một video cụ thể:**
  ```powershell
  .\run_video.cmd --source "data/videos/Đô.MOV" --mode full --weights weights/yolo26n-pose.pt --device auto
  ```

---

### Cách 4: Mở trang tổng quan đề tài & báo cáo kiến trúc
Trang web tĩnh giới thiệu chi tiết về đề tài, kiến trúc pipeline và các mốc thực nghiệm:

1. Nhấp đúp vào file:
   ```text
   run_project_site.cmd
   ```
2. Mở trình duyệt tại:
   👉 **[http://127.0.0.1:8765](http://127.0.0.1:8765)**

---

## 4. Hướng dẫn sử dụng Giao diện Web (`http://127.0.0.1:8090`)

1. **Chọn Video cần phân tích:**
   * Tại mục danh sách video có sẵn, chọn các video lớp học trong `data/videos/` (như `Đô.MOV`, `Thành (1).MOV`, `Khánh.MOV`, `Phong.MOV`, `Sơn.MOV`) hoặc tải video mới lên.
2. **Chọn Chế độ xử lý:**
   * **Full (Xử lý toàn bộ):** Nhận diện người, trích xuất 13 điểm pose, theo dõi ByteTrack ID và lưu toàn bộ kết quả để tua xem lại bất kỳ lúc nào.
   * **Replay (Phát lại lấy mẫu):** Phân tích và phát video theo thời gian thực (tốc độ 1x).
   * **Chuẩn bị gán nhãn:** Chuẩn bị video MP4 và mốc thời gian PTS để gán nhãn chuỗi hành vi.
3. **Theo dõi thanh HUD trên màn hình video:**
   * Thanh HUD nổi hiển thị tự động trên góc video stage:
     * ✋ **Giơ tay**: Đếm số người đang giơ tay phát biểu.
     * 📋 **Lên bảng**: Đếm số người di chuyển hoặc đang đứng tại khu vực bảng.
     * 🙆 **Vươn vai**: Nhận diện động tác vươn vai (có bộ lọc tránh nhầm với giơ tay).
     * 👥 **Có mặt**: Tổng số học sinh quan sát được.
     * ⚡ **Mức độ sôi nổi**: Trạng thái lớp (*Trầm lắng / Bình thường / Sôi nổi*).
4. **Bảng điều khiển ★ Đánh giá mức độ sôi nổi:**
   * Nằm ngay dưới khung video, hiển thị điểm tích lũy, tỷ lệ phần trăm tham gia và nhật ký chi tiết các sự kiện tương tác theo từng mốc giây.
5. **Xuất Video kết quả (MP4 with Annotations):**
   * Sau khi một job Full hoàn thành, bấm nút **Lưu MP4 có nhãn** để tạo file video MP4 đã vẽ sẵn bounding box, bộ xương pose và nhãn hành vi.

---

## 5. Cấu trúc thư mục chính

```text
AI-CAMERA-AINATIVE/
├── configs/                  # Cấu hình định nghĩa hành vi (action_rubric.yaml)
├── data/
│   └── videos/               # Video thực tế lớp học (Đô, Thành, Khánh, Sơn, Phong)
│       └── samples/          # Các video mẫu ngắn
├── docs/
│   ├── de_cuong/             # Đề cương đề tài và đề cương NCKH AI+Camera
│   └── guides/               # Hướng dẫn kỹ thuật
├── reports/                  # Báo cáo và sơ đồ kiến trúc hệ thống
├── runs/                     # Kết quả suy luận, dữ liệu gán nhãn pose và video export
├── src/
│   └── classroom_ai/         # Mã nguồn lõi: Video Pipeline, Web Demo, YOLO Detect & Pose
├── tasks/                    # Kế hoạch phát triển và ghi nhớ dự án (plan.md, project_memory.md)
├── tests/                    # Bộ kiểm thử tự động
├── tools/
│   ├── classroom_engagement/ # Module phân tích chuyên sâu mức độ sôi nổi lớp học
│   └── label_studio/         # Bộ công cụ gán nhãn dữ liệu chuẩn
├── weights/                  # Trọng số mô hình AI (YOLO Pose, Detect)
├── setup_demo.cmd            # File cài đặt môi trường ban đầu
├── run_demo.cmd              # File chạy Web Demo chính (cổng 8090)
├── run_engagement.cmd        # File chạy Engagement Backend (cổng 8000)
├── run_video.cmd             # File chạy phân tích CLI/TUI
├── run_project_site.cmd      # File mở trang tài liệu (cổng 8765)
└── README.md   # File hướng dẫn này
```

---

## 6. Xử lý các sự cố thường gặp (Troubleshooting)

* **Lỗi `No Python at ...`:**
  * Nguyên nhân: Do máy tính cài đặt Python ở đường dẫn khác với file cấu hình `.venv`.
  * Khắc phục: Chạy lại `setup_demo.cmd` để script cập nhật lại đường dẫn Python chuẩn.
* **Lỗi cổng 8090 hoặc 8000 bị chiếm:**
  * Mở PowerShell và kiểm tra tiến trình đang chiếm cổng:
    ```powershell
    Get-NetTCPConnection -LocalPort 8090, 8000 -State Listen
    ```
  * Tắt tiến trình cũ hoặc khởi động lại terminal trước khi chạy lại script `.cmd`.
* **Lỗi trình duyệt không kết nối được backend 8000:**
  * Đảm bảo cửa sổ `run_engagement.cmd` đang mở và hiển thị `Uvicorn running on http://0.0.0.0:8000`.
