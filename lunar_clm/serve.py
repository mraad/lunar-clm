"""Interactive flights: pick a start, tilt and pad in the browser, watch the pilot fly it live."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from .cli import new_recording, replay_page, run_episode
from .game import PADS, START_CEILING, START_MARGIN, custom_start
from .pilot import MODES, Pilot


class Server(HTTPServer):
    def __init__(self, pilot, page, port):
        self.pilot, self.page = pilot, page
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    server: Server  # pyright: ignore[reportIncompatibleVariableOverride]

    def reply(self, status, body, content_type="application/json"):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/":
            return self.reply(404, json.dumps({"error": "not found"}))
        self.reply(200, self.server.page, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/fly":
            return self.reply(404, json.dumps({"error": "not found"}))
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 1024:
                raise ValueError("request body must be 1 to 1024 bytes")
            request = json.loads(self.rfile.read(length))
            target = request["target"]
            if type(target) is not int or target not in range(len(PADS)):
                raise ValueError("target must be 0, 1 or 2")
            start = custom_start(request["x"], request["y"], request["angle"])
        except (ValueError, KeyError, TypeError) as exc:
            return self.reply(400, json.dumps({"error": str(exc)}))
        # HTTP/1.0 response without a length: the stream ends when the connection closes.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def send(message):
            self.wfile.write((json.dumps(message, allow_nan=False) + "\n").encode())

        try:
            episode = run_episode(self.server.pilot, 0, target, 900, start=start,
                                  on_frame=lambda frame: send({"frame": frame}))
            send({"summary": episode["summary"]})
        except (BrokenPipeError, ConnectionResetError):
            pass  # The viewer left; stop flying.
        except (RuntimeError, ValueError) as exc:
            send({"error": str(exc)})


def make_server(pilot, port=8000):
    record = new_recording(pilot)
    record["interactive"] = True
    record["start"] = {"margin": START_MARGIN, "ceiling": START_CEILING}
    # ponytail: one flight at a time on one thread; the pilot and model are not shared safely.
    return Server(pilot, replay_page(record), port)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", choices=MODES, default="small")
    parser.add_argument("--model", help="CLM API model name or a small-head .npz path")
    parser.add_argument("--base-url", help="CLM server URL; defaults to CLM_BASE_URL or localhost:8700")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        pilot = Pilot(args.pilot, args.model, args.base_url, args.timeout)
        server = make_server(pilot, args.port)
    except (RuntimeError, OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"lunar-clm-serve: {exc}\n")
    print(f"Lunar CLM ({args.pilot} pilot): http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        pilot.close()


if __name__ == "__main__":
    main()
