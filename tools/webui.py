"""Live view + keyboard for the interactive tools on a box with NO monitor.

The teach/focus/wrist tools were written around cv2.imshow + cv2.waitKey.
The GPU box is headless (reached over SSH / Tailscale), so this module gives
them the same two calls backed by a tiny HTTP server instead of a window:

    ui = make_display("teach")      # web UI when there is no $DISPLAY (or --web)
    ui.show(frame_bgr)              # latest frame -> MJPEG stream
    key = ui.waitkey(30)            # -1 or a key code, exactly like cv2.waitKey
    say("grid point recorded")      # prints AND shows in the browser log

Open  http://<box-ip>:8765  in any browser (Mac over Tailscale: http://100.68.73.7:8765),
click the page once, and type keys as before. Nothing else in the tools changes.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8765   # 8000 belongs to LeLab

_ACTIVE = None        # the display that say() mirrors into


def say(msg: str) -> None:
    """print() that also lands in the browser log when a WebDisplay is active."""
    print(msg, flush=True)
    if _ACTIVE is not None:
        _ACTIVE.log(msg)


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{background:#111;color:#eee;font-family:monospace;margin:0}}
#hud{{padding:8px 10px;white-space:pre-wrap;font-size:15px;background:#222}}
img{{display:block;max-width:100vw;max-height:70vh}}
#log{{padding:6px 10px;height:22vh;overflow:auto;font-size:13px;color:#9f9;white-space:pre-wrap}}
#hint{{color:#888;font-size:12px;padding:2px 10px}}</style></head>
<body><div id="hud">connecting...</div><img src="/stream"><div id="hint">click this page once, then type keys (they go straight to the tool)</div>
<div id="log"></div>
<script>
document.addEventListener('keydown',e=>{{ if(e.key.length==1||e.key=='Escape'){{fetch('/key?k='+encodeURIComponent(e.key));e.preventDefault();}} }});
setInterval(()=>fetch('/status').then(r=>r.json()).then(j=>{{
  document.getElementById('hud').textContent=j.status;
  const l=document.getElementById('log');
  if(l.dataset.n!=j.log.length){{ l.textContent=j.log.join('\\n'); l.dataset.n=j.log.length; l.scrollTop=l.scrollHeight; }}
}}).catch(()=>{{}}),400);
</script></body></html>"""


class WebDisplay:
    def __init__(self, title: str, port: int = DEFAULT_PORT):
        self.title = title
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._keys: queue.Queue = queue.Queue()
        self._status = f"{title}: waiting for the first frame"
        self._log: list[str] = []
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):   # keep the tool's terminal clean
                pass

            def do_GET(self):
                if self.path.startswith("/key"):
                    from urllib.parse import parse_qs, urlparse
                    k = parse_qs(urlparse(self.path).query).get("k", [""])[0]
                    if k:
                        ui._keys.put(k)
                    self.send_response(204); self.end_headers()
                elif self.path.startswith("/status"):
                    body = json.dumps({"status": ui._status, "log": ui._log[-200:]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers(); self.wfile.write(body)
                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            with ui._lock:
                                j = ui._jpeg
                            if j is not None and j is not last:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                                 + f"Content-Length: {len(j)}\r\n\r\n".encode()
                                                 + j + b"\r\n")
                                last = j
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    body = _PAGE.format(title=ui.title).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers(); self.wfile.write(body)

        self._srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        global _ACTIVE
        _ACTIVE = self
        print(f"[webui] live view at http://0.0.0.0:{port}/  (from the Mac: http://100.68.73.7:{port}/)",
              flush=True)

    # --- the cv2-shaped API --------------------------------------------
    def show(self, frame_bgr) -> None:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def waitkey(self, ms: int) -> int:
        """-1 when no key arrived within ms, else the key code (cv2 convention)."""
        try:
            k = self._keys.get(timeout=ms / 1000.0)
        except queue.Empty:
            return -1
        if k == "Escape":
            return 27
        return ord(k[0])

    def status(self, text: str) -> None:
        self._status = text

    def log(self, msg: str) -> None:
        self._log.append(msg)

    def close(self) -> None:
        global _ACTIVE
        _ACTIVE = None
        self._srv.shutdown()


class CvDisplay:
    """Original behaviour: an OpenCV window (needs a display)."""

    def __init__(self, title: str):
        self.title = title

    def show(self, frame_bgr) -> None:
        import cv2
        cv2.imshow(self.title, frame_bgr)

    def waitkey(self, ms: int) -> int:
        import cv2
        return cv2.waitKey(ms)

    def status(self, text: str) -> None:
        pass   # the tools draw status onto the frame already

    def log(self, msg: str) -> None:
        pass

    def close(self) -> None:
        import cv2
        cv2.destroyAllWindows()


def make_display(title: str, web: bool | None = None, port: int = DEFAULT_PORT):
    """Web UI when asked for (or when there is no display), else a cv2 window."""
    if web is None:
        web = "--web" in os.sys.argv or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return WebDisplay(title, port) if web else CvDisplay(title)
