"""Local HTTP transport for the control API: JSON commands in, server-sent events out.

Bound to 127.0.0.1 on an ephemeral port and guarded by a random token, so only the UI this
process launched can drive the engine. Standard library only.

    GET  /api/state                 the state document
    GET  /api/events                text/event-stream of state / meters / perf / toast
    POST /api/command               {"cmd": name, "args": {...}} -> {"ok": true, "result": ...}
    POST /api/upload?name=x.pth     raw file body, stored for a following import command
    GET  /<path>                    static files of the UI (``static_dir``), no token needed

The token travels in the ``X-Vox-Token`` header, or as ``?token=`` where a header cannot be
set (EventSource). The server knows nothing about the UI beyond the directory it serves.
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vox.api.controller import ApiError, EngineController
from vox.api.events import STATE

KEEPALIVE_S = 10.0
UPLOAD_CHUNK = 1 << 20
MAX_COMMAND_BYTES = 1 << 20
# Explicit: on Windows mimetypes reads the registry, which may call .js text/plain.
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml",
                 ".woff2": "font/woff2", ".json": "application/json; charset=utf-8",
                 ".txt": "text/plain; charset=utf-8", ".png": "image/png"}


class ApiServer:
    def __init__(self, controller: EngineController, static_dir: Path | None = None,
                 port: int = 0, token: str | None = None) -> None:
        self.controller = controller
        self.static_dir = Path(static_dir).resolve() if static_dir else None
        self.token = token or secrets.token_urlsafe(24)
        self._closing = threading.Event()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="vox-http",
                                        daemon=True)

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._closing.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    # ------------------------------------------------------------------ the handler

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # the engine log is not an access log
                pass

            # helpers ---------------------------------------------------------------

            def _json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _fail(self, status: HTTPStatus, message: str) -> None:
                self._json({"ok": False, "error": message}, status)

            def _authorised(self, query: dict) -> bool:
                given = self.headers.get("X-Vox-Token") or (query.get("token") or [""])[0]
                return secrets.compare_digest(given, server.token)

            def _length(self) -> int:
                try:
                    return int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return 0

            # routes ----------------------------------------------------------------

            def do_GET(self) -> None:
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if not url.path.startswith("/api/"):
                    return self._static(url.path)
                if not self._authorised(query):
                    return self._fail(HTTPStatus.FORBIDDEN, "bad token")
                if url.path == "/api/state":
                    return self._json(server.controller.state())
                if url.path == "/api/events":
                    return self._events()
                self._fail(HTTPStatus.NOT_FOUND, "no such endpoint")

            def do_POST(self) -> None:
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if not self._authorised(query):
                    self.close_connection = True  # the unread body would poison keep-alive
                    return self._fail(HTTPStatus.FORBIDDEN, "bad token")
                if url.path == "/api/command":
                    return self._command()
                if url.path == "/api/upload":
                    return self._upload((query.get("name") or [""])[0])
                self.close_connection = True
                self._fail(HTTPStatus.NOT_FOUND, "no such endpoint")

            def _command(self) -> None:
                length = self._length()
                if length > MAX_COMMAND_BYTES:
                    self.close_connection = True
                    return self._fail(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "command too large")
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    name, args = body["cmd"], body.get("args") or {}
                    if not isinstance(name, str) or not isinstance(args, dict):
                        raise TypeError("cmd must be a string and args an object")
                except (ValueError, KeyError, TypeError) as err:
                    return self._fail(HTTPStatus.BAD_REQUEST, f"bad command: {err}")
                try:
                    result = server.controller.command(name, args)
                except ApiError as err:
                    return self._json({"ok": False, "error": str(err)})
                except Exception as err:  # noqa: BLE001 - report, keep serving
                    return self._json({"ok": False, "error": f"{type(err).__name__}: {err}"})
                self._json({"ok": True, "result": result})

            def _upload(self, name: str) -> None:
                try:
                    target = server.controller.upload_path(name)
                except ApiError as err:
                    self.close_connection = True
                    return self._fail(HTTPStatus.BAD_REQUEST, str(err))
                remaining = self._length()
                with open(target, "wb") as fh:
                    while remaining > 0:
                        chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
                if remaining:
                    target.unlink(missing_ok=True)
                    self.close_connection = True
                    return self._fail(HTTPStatus.BAD_REQUEST, "upload was cut short")
                self._json({"ok": True, "result": {"name": target.name}})

            def _events(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                bus = server.controller.bus
                events = bus.subscribe()
                try:
                    self._send(STATE, server.controller.state())
                    while not server._closing.is_set():
                        try:
                            kind, data = events.get(timeout=KEEPALIVE_S)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        self._send(kind, data)
                except (BrokenPipeError, ConnectionError, OSError):
                    pass  # the UI went away
                finally:
                    bus.unsubscribe(events)

            def _send(self, kind: str, data) -> None:
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()

            def _static(self, path: str) -> None:
                root = server.static_dir
                if root is None:
                    return self._fail(HTTPStatus.NOT_FOUND, "no UI is being served")
                target = (root / (path.lstrip("/") or "index.html")).resolve()
                if root not in target.parents or not target.is_file():
                    return self._fail(HTTPStatus.NOT_FOUND, "not found")
                body = target.read_bytes()
                kind = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

        return Handler
