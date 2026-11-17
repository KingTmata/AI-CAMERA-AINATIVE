import asyncio
import os
import threading
import time
import base64
import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pipeline import Pipeline
import db
import config as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../frontend"))

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hệ thống chấm điểm sôi nổi lớp học - Multi-Mode")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pipe = Pipeline()

@app.on_event("startup")
def start():
    print("[Main] Đang khởi động AI Pipeline trong background...")
    thread = threading.Thread(target=pipe.run, daemon=True)
    thread.start()

def mjpeg_generator():
    """Tạo luồng dữ liệu MJPEG cho thẻ <img> trên frontend."""
    while True:
        with pipe.lock:
            f = pipe.frame_jpg
        if f is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.04)  # ~25 FPS

@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with pipe.lock:
                current_stats = dict(pipe.stats)
            current_stats["mode"] = pipe.mode
            await websocket.send_json(current_stats)
            await asyncio.sleep(1)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:
        print(f"[WebSocket] Đóng kết nối: {e}")

@app.get("/api/events")
def events():
    return db.recent(limit=50)

@app.get("/api/status")
def status():
    with pipe.lock:
        st = dict(pipe.stats)
    st["mode"] = pipe.mode
    st["source"] = str(C.SOURCE)
    st["data_path"] = str(C.DATA_PATH)
    return st

# Dành cho CHẾ ĐỘ 2 (Stream / Generator ngoài gửi frame và detections vào):
class DetectionItem(BaseModel):
    id: int
    box: List[float]
    keypoints: List[List[float]]
    conf: Optional[List[float]] = None

class FeedPayload(BaseModel):
    image_b64: Optional[str] = None
    detections: List[DetectionItem]

@app.post("/api/feed")
def feed_data(payload: FeedPayload):
    """
    Endpoint nhận dữ liệu trực tiếp từ bên ngoài (Chế độ 2).
    Có thể gửi kèm ảnh base64 hoặc chỉ gửi danh sách phát hiện.
    """
    try:
        if payload.image_b64:
            img_bytes = base64.b64decode(payload.image_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            with pipe.lock:
                f = pipe.frame_jpg
            if f:
                img = cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)

        dets = []
        for d in payload.detections:
            dets.append({
                "id": d.id,
                "box": d.box,
                "keypoints": np.array(d.keypoints, dtype=np.float32),
                "conf": np.array(d.conf, dtype=np.float32) if d.conf else None
            })

        stats = pipe.feed_frame(img, dets)
        return {"status": "ok", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount thư mục frontend để phục vụ giao diện web
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="fe")
else:
    print(f"[CẢNH BÁO] Không tìm thấy thư mục giao diện tại: {FRONTEND_DIR}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
