import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RAGSEARCH_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live test: set RAGSEARCH_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal, valid PDF with one line of Helvetica text per page."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
            ).encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def make_pdf():
    return write_pdf


# --- a tiny real HTTP server for testing network clients ---------------------


class StubServer:
    """Serve canned responses by path and record every request it receives.

    ``routes[path]`` is ``(status, headers, body)`` or a callable taking the
    recorded request dict and returning that tuple. A dict/list body is sent as JSON.
    """

    def __init__(self) -> None:
        self.routes: dict = {}
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _make_handler(stub: StubServer):
    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            request = {
                "method": self.command,
                "path": self.path.split("?", 1)[0],
                "headers": dict(self.headers),
                "json": json.loads(raw) if raw else None,
            }
            stub.requests.append(request)
            route = stub.routes.get(request["path"], (404, {}, "not found"))
            status, headers, body = route(request) if callable(route) else route
            if isinstance(body, (dict, list)):
                body = json.dumps(body)
                headers = {"Content-Type": "application/json", **headers}
            payload = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = _handle

        def log_message(self, *args) -> None:
            pass

    return Handler


@pytest.fixture
def stub_server():
    server = StubServer()
    server.start()
    yield server
    server.stop()
