"""A tiny threaded HTTP server for fetcher tests and benchmarks (no network needed)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

Route = Callable[[BaseHTTPRequestHandler], None]


def send(
    handler: BaseHTTPRequestHandler,
    body: bytes | str,
    status: int = 200,
    ctype: str = "text/html; charset=utf-8",
    headers: dict[str, str] | None = None,
) -> None:
    data = body.encode("utf-8") if isinstance(body, str) else body
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    for k, v in (headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    with suppress(BrokenPipeError, ConnectionResetError):
        handler.wfile.write(data)


def page(title: str, body: str) -> str:
    return (
        f"<html><head><title>{title}</title></head><body><main><p>{body}</p></main></body></html>"
    )


def default_routes() -> dict[str, Route]:
    def slow(h: BaseHTTPRequestHandler) -> None:
        time.sleep(1.5)
        send(h, page("Slow", "finally"))

    def huge(h: BaseHTTPRequestHandler) -> None:
        body = "<html><body>" + "<p>lorem ipsum dolor sit amet</p>" * 200_000 + "</body></html>"
        send(h, body)

    def stream(h: BaseHTTPRequestHandler) -> None:  # no Content-Length: size only known by reading
        h.send_response(200)
        h.send_header("Content-Type", "text/html")
        h.end_headers()
        try:
            h.wfile.write(b"<html><body>")
            for _ in range(2000):
                h.wfile.write(b"<p>lorem ipsum dolor sit amet</p>" * 20)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def redirect(h: BaseHTTPRequestHandler) -> None:
        h.send_response(302)
        h.send_header("Location", "/ok")
        h.send_header("Content-Length", "0")
        h.end_headers()

    return {
        "/ok": lambda h: send(h, page("OK page", "Velmora has 13,050 people.")),
        "/forbidden": lambda h: send(h, "nope", status=403),
        "/missing": lambda h: send(h, "gone", status=404),
        "/error": lambda h: send(h, "boom", status=500),
        "/pdf": lambda h: send(h, b"%PDF-1.4 ...", ctype="application/pdf"),
        "/text": lambda h: send(h, "plain   text\n\n\nfile", ctype="text/plain"),
        "/notype": lambda h: send(h, page("No type", "sniffed html"), ctype=""),
        "/cp1251": lambda h: send(
            h,
            page("Город", "Население 13 050 человек").encode("cp1251"),
            ctype="text/html; charset=windows-1251",
        ),
        "/metacharset": lambda h: send(
            h,
            (
                '<html><head><meta charset="windows-1251"><title>x</title></head>'
                "<body><p>Привет мир</p></body></html>"
            ).encode("cp1251"),
            ctype="text/html",
        ),
        "/empty": lambda h: send(h, "<html><body><script>x()</script></body></html>"),
        "/slow": slow,
        "/huge": huge,
        "/stream": stream,
        "/redirect": redirect,
    }


@contextmanager
def serve(routes: dict[str, Route] | None = None, delay: float = 0.0) -> Iterator[str]:
    """Run a server in a thread; yields its base URL. ``delay`` is added to every request."""
    table = routes or default_routes()
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            if delay:
                time.sleep(delay)
            route = table.get(self.path.split("?")[0])
            if route is None:
                send(self, "not found", status=404)
            else:
                route(self)

        def log_message(self, *args: object) -> None:
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 128  # default backlog (5) drops concurrent SYNs -> 1 s retries

    server = Server(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
