"""A separately owned arbitrary-200 listener for process-ownership tests."""

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'{"service":"other","profile":"legacy","ready":true}'
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    with HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler) as server:
        print(server.server_port, flush=True)
        server.serve_forever()
