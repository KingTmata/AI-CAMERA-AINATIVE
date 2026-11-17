# HƯỚNG DẪN CHI TIẾT HỆ THỐNG XỬ LÝ NHÃN & ĐẾM ĐIỂM SÔI NỔI LỚP HỌC

Hệ thống được thiết kế linh hoạt với mục tiêu chính: **Nhận dữ liệu Pose, Bbox & Keypoints để Xử lý Nhãn Hành Vi và Đếm Số Lượng Tích Lũy**.

---

## 1. Ba Chế Độ Hoạt Động (Multi-Mode)

Hệ thống tự động phát hiện hoặc bạn có thể chỉ định trong `backend/config.py` qua biến `DATA_MODE`:

| Chế độ | Tên gọi | Cách thức hoạt động | Ưu điểm |
| :---: | :--- | :--- | :--- |
| **Chế độ 1** *(Khuyên dùng khi đã có dữ liệu)* | **Đọc từ File dữ liệu (`file`)** | Đọc video để lấy hình ảnh + đọc file `pose_data.json` chứa tọa độ keypoints & bbox đã có sẵn. | **Không cần GPU**, không tốn tài nguyên chạy YOLO. Chạy siêu nhẹ và mượt mà. |
| **Chế độ 2** | **Nhận Stream API / Generator (`stream`)** | Nhận trực tiếp frame và detections từ một module AI ngoài hoặc tool gán nhãn qua API `POST /api/feed`. | Tích hợp linh hoạt với bất kỳ pipeline Python nào. |
| **Chế độ 3** | **Tự động suy luận YOLO (`yolo`)** | Tự động dùng YOLOv8 Pose trích xuất tọa độ nếu bạn chỉ có video thô mà chưa có file dữ liệu. | Tự động hóa hoàn toàn từ video chưa gán nhãn. |

---

## 2. Cách Chạy Nhanh

### Bước 1: Khởi động Server
Tại thư mục `classroom-engagement`:
```bash
python backend/main.py
```
Hoặc:
```bash
cd backend
uvicorn main:app --port 8000
```

### Bước 2: Xem kết quả trên giao diện web
Mở trình duyệt truy cập:
👉 **http://localhost:8000**

---

## 3. Chi Tiết Từng Chế Độ

### 👉 Chế độ 1: Khi bạn ĐÃ CÓ dữ liệu Pose, Bbox, Keypoint
1. Đặt file dữ liệu của bạn vào thư mục dự án với tên **`pose_data.json`** (hoặc `labels.json`, `tracks.json`).
   - Cấu trúc mẫu chuẩn của file `pose_data.json`:
   ```json
   {
     "0": [
       {
         "id": 1,
         "box": [280.0, 190.0, 420.0, 520.0],
         "keypoints": [[350.0, 210.0], ..., [395.0, 170.0]],
         "conf": [0.9, 0.95, ...]
       }
     ],
     "1": [ ... ]
   }
   ```
2. Nếu muốn tạo file dữ liệu mẫu để chạy thử ngay:
   ```bash
   python tools/generate_sample_data.py
   ```
3. Khởi động server (`python backend/main.py`). Hệ thống sẽ tự nhận diện file `pose_data.json`, hiển thị tag:
   `Chế độ 1: Đọc nhãn & Pose từ File (No GPU)` và vẽ nhãn, đếm số lượng ngay lập tức.

---

### 👉 Chế độ 2: Gửi dữ liệu trực tiếp từ Python Code ngoài (API Feed)
Nếu bạn có một script Python đang chạy model hoặc tool gán nhãn, chỉ cần gửi POST request tới `http://localhost:8000/api/feed`:

```python
import requests

payload = {
    "image_b64": None,  # Hoặc chuỗi base64 của frame ảnh
    "detections": [
        {
            "id": 1,
            "box": [100, 150, 250, 450],
            "keypoints": [[x0, y0], ..., [x16, y16]],
            "conf": [0.9] * 17
        }
    ]
}

res = requests.post("http://localhost:8000/api/feed", json=payload)
print(res.json()["stats"])
```
*(Có thể chạy thử script demo: `python tools/client_stream_demo.py`)*

---

### 👉 Chế độ 3: Xuất dữ liệu từ Video nếu chưa có sẵn file JSON
Nếu bạn có một video bài giảng và muốn xuất toàn bộ Pose & Bbox ra JSON một lần duy nhất để dùng mãi mãi về sau:
```bash
python tools/export_pose_from_video.py video.mp4 pose_data.json
```
Sau lệnh này, file `pose_data.json` sẽ được tạo và hệ thống tự động chạy theo **Chế độ 1**.

---

## 4. Cách Hệ Thống Xử Lý Nhãn & Đếm Số Lượng

### A. Xử lý nhãn hành vi & Bộ lọc ngoại lệ (`detectors.py`):
1. **Nhãn "GIƠ TAY" (Hợp lệ)**:
   - Giơ duy nhất **1 tay** (tay trái hoặc tay phải).
   - Cổ tay vượt lên trên vai: $y_{wrist} < y_{shoulder} - 0.3 \times \text{chiều dài thân}$.
   - Cẳng tay tương đối thẳng đứng (không dang rộng sang ngang).
   - Duy trì tối thiểu `0.5s` $\rightarrow$ Xác nhận nhãn `GIO TAY` (màu xanh lá, +1 điểm).

2. **Bộ lọc ngoại lệ: Nhãn "VUON VAI (LỌC)" (Vươn vai / Hít thở / Dang tay)**:
   - **Giơ cả 2 tay**: Trong lớp học, giơ tay phát biểu là giơ 1 tay. Nếu cả 2 tay cùng giơ cao $\rightarrow$ Phân loại là vươn vai hít thở.
   - **Dang tay quá rộng**: Khoảng cách ngang giữa cổ tay và vai $> 1.15$ lần độ rộng vai (vươn vai dang ngang hình chữ V/W).
   - **Ngửa đầu ra sau**: Khi vươn vai/ngáp, đầu ngửa ra sau (mũi dâng cao ngang hoặc trên tai).
   - $\rightarrow$ **Hệ thống tự động gán nhãn màu vàng cam `VUON VAI (LỌC)` và TUYỆT ĐỐI KHÔNG TÍNH ĐIỂM!**

3. **Bộ lọc ngoại lệ: Chống cằm / Gãi đầu**:
   - Cổ tay đặt sát mũi/mặt trong khi khuỷu tay tì thấp trên bàn $\rightarrow$ Giữ nhãn `BINH THUONG`, không đếm nhầm.

4. **Nhãn "LÊN BẢNG" / "Ở BẢNG"**:
   - Kiểm tra điểm hông nằm trong đa giác `BOARD_POLY` tối thiểu `1.0s` $\rightarrow$ Xác nhận nhãn `O BANG` (màu xanh dương, +3 điểm).

5. **Nhãn "BÌNH THƯỜNG"**:
   - Khi học sinh ngồi nghe, không giơ tay và không đứng ở bảng.

### B. Đếm số lượng & Tính điểm (`engagement.py`):
1. **Đếm tức thời (Live trong frame)**:
   - `Số bạn đang giơ tay`: Đếm số học sinh đang giơ tay cùng lúc.
   - `Số bạn đang ở bảng`: Đếm số học sinh đang đứng ở khu vực bảng.
   - `Tổng số học sinh`: Số lượng học sinh xuất hiện.
2. **Đếm tích lũy trong cửa sổ 5 phút**:
   - `Tổng lượt giơ tay`: Đếm mỗi lần học sinh mới giơ tay (+1đ).
   - `Tổng lượt lên bảng`: Đếm mỗi lần học sinh lên bảng (+3đ).
   - `Combo xung phong`: Giơ tay rồi lên bảng trong 60s (+2đ).
   - `Tỉ lệ tham gia` = $\frac{\text{Số bạn có tương tác}}{\text{Tổng số học sinh}}$.
   - **Xếp hạng mức độ**: `SÔI NỔI` ($\ge 8$ điểm hoặc tham gia $\ge 30\%$), `BÌNH THƯỜNG` ($3 \le \text{điểm} < 8$), `YÊN LẶNG` ($< 3$ điểm).

### C. Bộ lọc chống nháy nhãn & làm mượt Pose (`detectors.py`, `config.py`):
- **One-Euro Filter (`KeypointSmoother`)**: Làm mượt tọa độ 17 keypoints theo thời gian thực (giảm rung lắc khi đứng yên, giữ phản hồi nhanh khi giơ tay).
- **Ngưỡng kép vùng trễ (Hysteresis)**: Vào `GIO TAY` ở ngưỡng `RAISE_THR_HIGH = 0.30`, thoát ở `RAISE_THR_LOW = 0.12` để tránh nhấp nháy ở ranh giới.
- **Biểu quyết đa số & Giữ nhãn (`VOTE_WINDOW_SEC`, `LABEL_HOLD_SEC`)**: Yêu cầu đồng thuận $\ge 65\%$ trong cửa sổ trượt, debounce cho cả `VUON VAI`, và khóa nhãn tối thiểu `0.5s` trước khi đổi nhãn khác.
- **Lọc vận tốc (`WRIST_MAX_SPEED_RATIO`)**: Bỏ qua các frame cổ tay đang vung nhanh liên tục.

---

## 5. Công Cụ Hỗ Trợ

- **Vẽ vùng bảng bằng chuột**: `python tools/select_board.py`
- **Tạo dữ liệu mẫu kiểm thử**: `python tools/generate_sample_data.py`
- **Xuất pose từ video thành JSON**: `python tools/export_pose_from_video.py`
- **Demo gửi stream API**: `python tools/client_stream_demo.py`
