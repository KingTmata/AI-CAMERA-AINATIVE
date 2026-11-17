import sqlite3
import threading
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "events.db")

_lock = threading.Lock()
conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)

with _lock:
    conn.execute("CREATE TABLE IF NOT EXISTS events(ts REAL, kind TEXT, tid INT)")
    conn.commit()

def save(ts, kind, tid):
    with _lock:
        conn.execute("INSERT INTO events VALUES(?,?,?)", (ts, kind, tid))
        conn.commit()

def recent(limit=100):
    with _lock:
        rows = conn.execute(
            "SELECT ts, kind, tid FROM events ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [{"ts": r[0], "kind": r[1], "tid": r[2]} for r in rows]
