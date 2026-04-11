from __future__ import annotations

import queue
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pixie_solver.utils.serialization import JsonValue, canonical_json


class ViewerEventHub:
    def __init__(self, *, history_limit: int = 2000) -> None:
        self._history_limit = history_limit
        self._history: list[dict[str, JsonValue]] = []
        self._clients: list[queue.Queue[dict[str, JsonValue] | None]] = []
        self._lock = threading.Lock()
        self._closed = False

    def publish(self, frame: dict[str, JsonValue]) -> None:
        with self._lock:
            if self._closed:
                return
            self._history.append(frame)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]
            clients = list(self._clients)
        for client in clients:
            client.put(frame)

    def history(self) -> list[dict[str, JsonValue]]:
        with self._lock:
            return list(self._history)

    def subscribe(self) -> queue.Queue[dict[str, JsonValue] | None]:
        client: queue.Queue[dict[str, JsonValue] | None] = queue.Queue()
        with self._lock:
            for frame in self._history:
                client.put(frame)
            if self._closed:
                client.put(None)
            else:
                self._clients.append(client)
        return client

    def unsubscribe(self, client: queue.Queue[dict[str, JsonValue] | None]) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            client.put(None)


@dataclass(slots=True)
class ViewerServer:
    host: str = "127.0.0.1"
    port: int = 0
    replay_payload: dict[str, JsonValue] | None = None
    open_browser: bool = False
    history_limit: int = 2000
    _hub: ViewerEventHub = field(init=False)
    _httpd: ThreadingHTTPServer | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _url: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._hub = ViewerEventHub(history_limit=self.history_limit)

    @property
    def url(self) -> str:
        if self._url is None:
            raise RuntimeError("Viewer server has not been started")
        return self._url

    def start(self) -> str:
        if self._httpd is not None:
            return self.url

        static_dir = Path(__file__).resolve().parent / "static"
        hub = self._hub
        replay_payload = self.replay_payload

        class Handler(BaseHTTPRequestHandler):
            server_version = "PixieViewer/0.1"

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._serve_file(static_dir / "index.html", "text/html; charset=utf-8")
                    return
                if parsed.path == "/app.js":
                    self._serve_file(static_dir / "app.js", "text/javascript; charset=utf-8")
                    return
                if parsed.path == "/styles.css":
                    self._serve_file(static_dir / "styles.css", "text/css; charset=utf-8")
                    return
                if parsed.path == "/api/replay":
                    payload = replay_payload or {
                        "mode": "live",
                        "source": None,
                        "frames": [],
                        "game_count": None,
                    }
                    self._serve_json(payload)
                    return
                if parsed.path == "/events":
                    self._serve_events(hub)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _serve_file(self, path: Path, content_type: str) -> None:
                try:
                    payload = path.read_bytes()
                except OSError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _serve_json(self, payload: object) -> None:
                body = canonical_json(payload, indent=None).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_events(self, event_hub: ViewerEventHub) -> None:
                client = event_hub.subscribe()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        try:
                            frame = client.get(timeout=15.0)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        if frame is None:
                            break
                        body = canonical_json(frame, indent=None)
                        self.wfile.write(f"data: {body}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    event_hub.unsubscribe(client)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        actual_host, actual_port = self._httpd.server_address[:2]
        self._url = f"http://{actual_host}:{actual_port}/"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="pixie-viewer-server",
            daemon=True,
        )
        self._thread.start()
        if self.open_browser:
            webbrowser.open(self._url)
        return self._url

    def publish(self, frame: dict[str, JsonValue]) -> None:
        self._hub.publish(frame)

    def publish_many(self, frames: list[dict[str, JsonValue]]) -> None:
        for frame in frames:
            self.publish(frame)

    def stop(self) -> None:
        self._hub.close()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def wait_for_viewer_shutdown(url: str) -> None:
    print(f"Pixie viewer is still running at {url}. Press Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return
