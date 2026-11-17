// Quản lý WebSocket & Giao diện Dashboard Realtime
let ws = null;
let lastEventCount = -1;

function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${location.host}/ws`;
  const connBadge = document.getElementById("conn-badge");
  const connText = document.getElementById("conn-text");
  const modeBadge = document.getElementById("mode-badge");

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[WS] Đã kết nối với backend!");
    if (connBadge && connText) {
      connBadge.className = "badge-online";
      connText.textContent = "Trực tuyến";
    }
    fetchRecentEvents();
  };

  ws.onmessage = (e) => {
    try {
      const s = JSON.parse(e.data);
      if (!s) return;

      // Cập nhật thẻ chế độ hoạt động
      if (modeBadge && s.mode) {
        if (s.mode === "file") {
          modeBadge.textContent = "Chế độ 1: Đọc nhãn & Pose từ File (No GPU)";
          modeBadge.className = "mode-tag mode-file";
        } else if (s.mode === "stream") {
          modeBadge.textContent = "Chế độ 2: Nhận Stream / API ngoài";
          modeBadge.className = "mode-tag mode-stream";
        } else {
          modeBadge.textContent = "Chế độ 3: Tự động chạy YOLO Pose";
          modeBadge.className = "mode-tag mode-yolo";
        }
      }

      // Cập nhật tất cả các trường số liệu
      const keys = [
        "currently_raising", "currently_at_board", "currently_stretching", "people",
        "raises", "boards", "combos", "score", "active"
      ];
      for (const k of keys) {
        const el = document.getElementById(k);
        if (el && s[k] !== undefined) {
          el.textContent = s[k];
        }
      }

      // Cập nhật tỉ lệ tham gia
      const rateVal = Math.round((s.rate || 0) * 100);
      const rateEl = document.getElementById("rate");
      if (rateEl) rateEl.textContent = rateVal;

      const rateBar = document.getElementById("rate-bar");
      if (rateBar) rateBar.style.width = `${Math.min(100, rateVal)}%`;

      // Cập nhật mức độ sôi nổi
      const lv = document.getElementById("level");
      if (lv && s.level) {
        lv.textContent = s.level.toUpperCase();
        lv.className = "level " + (s.level === "sôi nổi" ? "hot" : s.level === "bình thường" ? "mid" : "low");
      }

      // Nếu có sự kiện mới thì làm mới danh sách nhật ký
      const currentCount = (s.raises || 0) + (s.boards || 0) + (s.combos || 0);
      if (currentCount !== lastEventCount) {
        lastEventCount = currentCount;
        fetchRecentEvents();
      }
    } catch (err) {
      console.error("[WS] Lỗi parse dữ liệu:", err);
    }
  };

  ws.onclose = () => {
    console.warn("[WS] Mất kết nối, đang thử kết nối lại sau 2 giây...");
    if (connBadge && connText) {
      connBadge.className = "badge-online badge-offline";
      connText.textContent = "Mất kết nối...";
    }
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("[WS Error]:", err);
    ws.close();
  };
}

// Lấy lịch sử sự kiện từ API /api/events
async function fetchRecentEvents() {
  try {
    const res = await fetch("/api/events");
    if (!res.ok) return;
    const events = await res.json();
    const listEl = document.getElementById("event-list");
    const countEl = document.getElementById("event-count");
    if (!listEl) return;

    if (!events || events.length === 0) {
      listEl.innerHTML = `<div class="log-empty">Chưa có sự kiện nào ghi nhận...</div>`;
      if (countEl) countEl.textContent = "0 sự kiện";
      return;
    }

    if (countEl) countEl.textContent = `${events.length} sự kiện gần nhất`;

    listEl.innerHTML = events.slice(0, 20).map(ev => {
      const ts = ev.ts !== undefined ? ev.ts : ev[0];
      const kind = ev.kind !== undefined ? ev.kind : ev[1];
      const tid = ev.tid !== undefined ? ev.tid : ev[2];

      const timeStr = new Date(ts * 1000).toLocaleTimeString("vi-VN");
      let icon = "✋";
      let text = `Học sinh #${tid} giơ tay`;
      let badgeClass = "event-badge-raise";

      if (kind === "board") {
        icon = "🎓";
        text = `Học sinh #${tid} lên bảng`;
        badgeClass = "event-badge-board";
      } else if (kind === "combo") {
        icon = "⚡";
        text = `Học sinh #${tid} COMBO XUNG PHONG`;
        badgeClass = "event-badge-combo";
      }

      return `
        <div class="event-item">
          <div>
            <span>${icon}</span>
            <span class="${badgeClass}">${text}</span>
          </div>
          <span class="event-time">${timeStr}</span>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("[API Events Error]:", err);
  }
}

connectWebSocket();
setInterval(fetchRecentEvents, 4000);
