# Classroom Engagement AI (Hệ Thống Chấm Điểm Sôi Nổi Lớp Học)

Hệ thống AI nhận diện và chấm điểm mức độ sôi nổi của lớp học theo thời gian thực (Realtime) sử dụng YOLOv8 Pose, ByteTrack, FastAPI và WebSocket.

---

## 1. Cấu trúc thư mục

```
classroom-engagement/
├── backend/
│   ├── main.py          # FastAPI server, WebSocket, luồng MJPEG, worker
│   ├── config.py        # Cấu hình nguồn video, ngưỡng, vùng bảng, trọng số điểm
│   ├── pipeline.py      # Đọc video, YOLO pose tracking (ByteTrack), vẽ overlay
│   ├── detectors.py     # Bộ phát hiện giơ tay (RaiseHand) & lên bảng (Board)
│   ├── engagement.py    # Tính điểm cửa sổ trượt (5 phút), tỉ lệ tham gia, combo
│   ├── db.py            # Lưu trữ sự kiện vào SQLite
│   └── requirements.txt # Danh sách thư viện Python
├── frontend/
│   ├── index.html       # Giao diện web hiển thị video stream & số liệu
│   ├── app.js           # Xử lý WebSocket & cập nhật realtime
│   └── style.css        # Giao diện Dark mode hiện đại
├── models/
│   └── pose.pt          # File weights YOLOv8 Pose
├── tools/
│   └── select_board.py  # Tool nhấp chuột để vẽ và lấy tọa độ vùng bảng (BOARD_POLY)
└── README.md
```

---

## 2. Cách hoạt động & Cách tính điểm

### Trọng số sự kiện (Cửa sổ trượt 5 phút):
- ✋ **Giơ tay phát biểu**: `+1 điểm` (cổ tay giơ cao hơn vai vượt ngưỡng `RAISE_THR * chiều dài thân`, giữ tối thiểu 0.5s).
- 🎓 **Lên bảng làm bài**: `+3 điểm` (vào vùng bảng đã định nghĩa tối thiểu 1.0s).
- ⚡ **Combo xung phong**: `+2 điểm` (giơ tay rồi lên bảng trong vòng 60 giây).

### Phân mức độ sôi nổi:
- **Tỉ lệ tham gia** = `Số học sinh khác nhau có tương tác / Tổng số học sinh hiện diện`.
- **Sôi nổi**: Tổng điểm $\ge 8$ hoặc Tỉ lệ tham gia $\ge 30\%$.
- **Bình thường**: $3 \le$ Tổng điểm $< 8$.
- **Yên lặng**: Tổng điểm $< 3$.

---

## 3. Cài đặt & Khởi chạy

### Bước 1: Cài đặt thư viện
```bash
pip install -r backend/requirements.txt
```

### Bước 2: Cấu hình nguồn vào (`backend/config.py`)
Mặc định hệ thống sẽ tự động dùng file `video.mp4` nếu có, hoặc chuyển sang webcam `0`. Bạn có thể tùy chỉnh:
- Dùng webcam: `SOURCE = 0`
- Dùng video có sẵn: `SOURCE = "video.mp4"`
- Dùng camera RTSP lớp học: `SOURCE = "rtsp://admin:password@192.168.1.100:554/stream"`

### Bước 3: Xác định vùng bảng bằng chuột (Tùy chọn)
Chạy script chọn 4 góc của bảng trên khung hình camera:
```bash
python tools/select_board.py
```
Nhấp 4 điểm quanh bảng, sau đó copy tọa độ `BOARD_POLY` được in ra vào `backend/config.py`.

### Bước 4: Khởi chạy server
```bash
cd backend
uvicorn main:app --port 8000
```
Hoặc từ thư mục gốc:
```bash
python backend/main.py
```

### Bước 5: Xem giao diện
Mở trình duyệt web và truy cập:
👉 **http://localhost:8000**
