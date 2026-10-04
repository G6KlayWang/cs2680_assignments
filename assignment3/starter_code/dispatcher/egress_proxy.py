#!/usr/bin/env python3
"""egress_proxy.py — the agent container's only way to the internet.

Dispatcher-owned (CS2680 A3). evaluation_scripts/run_all.py runs this in a
container that sits on the normal bridge network AND on the private "egress"
network shared with the agent container; the agent itself is on internal
networks only (no gateway, no DNS to the outside). The agent's OpenAI client
is pointed here through HTTPS_PROXY.

    python3 egress_proxy.py --port 3128 --allow api.cs2680.com:443

Behaviour:
- Only the HTTP CONNECT method is accepted (an HTTPS tunnel). Plain GET/POST
  through the proxy is refused (405), so there is no unencrypted path.
- The CONNECT target must match an --allow entry literally (host:port,
  case-insensitive host); an IP address, another port or another name gets a
  403 before any outbound connection is made. The proxy resolves the allowed
  name itself.
- After "200 Connection Established" bytes are copied both ways until one
  side closes; the TLS session is end to end (the proxy never sees the API
  key, prompts or replies — only the destination name, which it logs).
Stdlib only, one thread per tunnel.
"""

import argparse
import select
import socket
import socketserver
import sys
import threading
import time

ALLOW = set()          # {("host", port)}
CONNECT_TIMEOUT_S = 30
IDLE_TIMEOUT_S = 600   # a tunnel with no traffic for this long is closed


def _log(msg: str):
    print(f"[egress] {time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


class Tunnel(socketserver.BaseRequestHandler):
    def _reply(self, code: int, text: str):
        body = (text + "\n").encode()
        self.request.sendall(
            f"HTTP/1.1 {code} {text}\r\nContent-Type: text/plain\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)

    def handle(self):
        peer = self.client_address[0]
        self.request.settimeout(CONNECT_TIMEOUT_S)
        try:
            head = b""
            while b"\r\n\r\n" not in head and len(head) < 65536:
                chunk = self.request.recv(4096)
                if not chunk:
                    return
                head += chunk
        except (socket.timeout, OSError):
            return
        line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = line.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            _log(f"{peer} REFUSED non-CONNECT: {line[:120]!r}")
            return self._reply(405, "Method Not Allowed: only CONNECT to an allowed host:port")
        host, _, port = parts[1].rpartition(":")
        host = host.strip("[]").lower()
        try:
            port = int(port)
        except ValueError:
            return self._reply(400, "Bad Request")
        if (host, port) not in ALLOW:
            _log(f"{peer} REFUSED CONNECT {host}:{port}")
            return self._reply(403, f"Forbidden: {host}:{port} is not an allowed egress destination")
        try:
            upstream = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S)
        except OSError as e:
            _log(f"{peer} CONNECT {host}:{port} failed: {e}")
            return self._reply(502, f"Bad Gateway: {e}")
        _log(f"{peer} CONNECT {host}:{port} ok")
        self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.request.settimeout(None)
        upstream.settimeout(None)
        socks = [self.request, upstream]
        try:
            while True:
                readable, _, _ = select.select(socks, [], [], IDLE_TIMEOUT_S)
                if not readable:
                    break                      # idle
                for s in readable:
                    data = s.recv(65536)
                    if not data:
                        return
                    (upstream if s is self.request else self.request).sendall(data)
        except OSError:
            pass
        finally:
            upstream.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    ap = argparse.ArgumentParser(description="CS2680 A3 egress proxy (CONNECT allowlist)")
    ap.add_argument("--port", type=int, default=3128)
    ap.add_argument("--allow", action="append", required=True,
                    help="host:port the agent may tunnel to (repeatable)")
    args = ap.parse_args()
    for entry in args.allow:
        host, _, port = entry.rpartition(":")
        ALLOW.add((host.strip().lower(), int(port)))
    srv = Server(("0.0.0.0", args.port), Tunnel)
    _log(f"listening on {args.port}; allowed: {sorted(ALLOW)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
