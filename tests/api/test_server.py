"""The local HTTP transport: token, commands, uploads, static files and the event stream."""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from vox.api.controller import ApiError
from vox.api.events import EventBus
from vox.api.server import ApiServer


class FakeController:
    def __init__(self, uploads: Path) -> None:
        self.bus = EventBus()
        self.uploads = uploads
        self.commands: list[tuple] = []

    def state(self):
        return {"engine": {"status": "stopped"}}

    def command(self, name, args):
        self.commands.append((name, args))
        if name == "boom":
            raise ApiError("not possible")
        if name == "crash":
            raise RuntimeError("bug")
        return {"echo": args}

    def upload_path(self, filename):
        if not filename.endswith(".pth"):
            raise ApiError("only .pth and .index files can be imported")
        return self.uploads / Path(filename).name


@pytest.fixture
def served(tmp_path):
    static = tmp_path / "web"
    (static / "fonts").mkdir(parents=True)
    (static / "index.html").write_text("<h1>DEXAIV</h1>", encoding="utf-8")
    (static / "app.js").write_text("let x = 1;", encoding="utf-8")
    (static / "fonts" / "a.woff2").write_bytes(b"wOF2")
    (tmp_path / "secret.txt").write_text("outside the static root")
    controller = FakeController(tmp_path)
    server = ApiServer(controller, static)
    server.start()
    yield server, controller
    server.stop()


def call(server, path, data=None, token=True, method=None):
    headers = {"X-Vox-Token": server.token} if token else {}
    request = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=data,
                                     headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers.get("Content-Type"), response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type"), err.read()


def post(server, name, args=None, **kwargs):
    body = json.dumps({"cmd": name, "args": args or {}}).encode()
    status, _, raw = call(server, "/api/command", body, **kwargs)
    return status, json.loads(raw)


def test_url_carries_the_token_and_binds_to_loopback(served):
    server, _ = served
    assert server.url == f"http://127.0.0.1:{server.port}/?token={server.token}"
    assert server.httpd.server_address[0] == "127.0.0.1" and len(server.token) >= 24


def test_api_needs_the_token(served):
    server, controller = served
    assert call(server, "/api/state", token=False)[0] == 403
    assert call(server, "/api/state?token=wrong", token=False)[0] == 403
    assert post(server, "start", token=False)[0] == 403 and controller.commands == []
    assert call(server, f"/api/state?token={server.token}", token=False)[0] == 200
    status, kind, raw = call(server, "/api/state")
    assert status == 200 and kind.startswith("application/json")
    assert json.loads(raw) == {"engine": {"status": "stopped"}}


def test_commands(served):
    server, controller = served
    assert post(server, "set_param", {"name": "pitch", "value": 3}) == \
        (200, {"ok": True, "result": {"echo": {"name": "pitch", "value": 3}}})
    assert controller.commands == [("set_param", {"name": "pitch", "value": 3})]
    assert post(server, "boom") == (200, {"ok": False, "error": "not possible"})
    assert post(server, "crash") == (200, {"ok": False, "error": "RuntimeError: bug"})
    assert call(server, "/api/command", b"not json")[0] == 400
    assert call(server, "/api/command", b'{"args": {}}')[0] == 400
    assert call(server, "/api/command", b'{"cmd": "x", "args": [1]}')[0] == 400
    assert call(server, "/api/nothing", b"{}")[0] == 404
    assert post(server, "after_errors")[1]["ok"] is True  # the server is still fine


def test_upload_streams_to_the_controllers_path(served, tmp_path):
    server, _ = served
    payload = bytes(range(256)) * 5000  # 1.28 MB: more than one chunk
    status, _, raw = call(server, "/api/upload?name=My%20Voice.pth", payload)
    assert status == 200 and json.loads(raw)["result"] == {"name": "My Voice.pth"}
    assert (tmp_path / "My Voice.pth").read_bytes() == payload
    assert call(server, "/api/upload?name=tool.exe", b"MZ")[0] == 400
    assert not (tmp_path / "tool.exe").exists()


def test_static_files(served):
    server, _ = served
    status, kind, raw = call(server, "/?token=whatever", token=False)
    assert (status, kind, raw) == (200, "text/html; charset=utf-8", b"<h1>DEXAIV</h1>")
    assert call(server, "/app.js", token=False)[1] == "text/javascript; charset=utf-8"
    assert call(server, "/fonts/a.woff2", token=False)[1:] == ("font/woff2", b"wOF2")
    assert call(server, "/missing.css", token=False)[0] == 404
    for escape in ("/../secret.txt", "/..%2Fsecret.txt", "/fonts/../../secret.txt"):
        assert call(server, escape, token=False)[0] == 404


def test_event_stream(served):
    server, controller = served
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}/api/events?token={server.token}")
    with urllib.request.urlopen(request, timeout=10) as stream:
        assert stream.headers.get("Content-Type") == "text/event-stream"

        def read_event():
            kind = stream.readline().decode().strip()
            data = stream.readline().decode().strip()
            assert stream.readline() == b"\n"
            return kind, json.loads(data.removeprefix("data: "))

        # the state document arrives first, without waiting for a change
        assert read_event() == ("event: state", {"engine": {"status": "stopped"}})
        assert controller.bus.subscribers == 1
        controller.bus.publish("meters", {"in_db": -20.5})
        controller.bus.publish("toast", {"message": "hi"})
        assert read_event() == ("event: meters", {"in_db": -20.5})
        assert read_event() == ("event: toast", {"message": "hi"})
    assert call(server, "/api/events", token=False)[0] == 403


def test_event_bus_drops_the_oldest_for_a_slow_subscriber():
    bus = EventBus(maxsize=3)
    q = bus.subscribe()
    for n in range(10):
        bus.publish("meters", n)
    assert [q.get_nowait()[1] for _ in range(3)] == [7, 8, 9]
    bus.unsubscribe(q)
    bus.publish("meters", 11)
    assert q.empty() and bus.subscribers == 0
