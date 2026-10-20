# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from responses_api_agents.codex_agent.sandbox_transport import rate_limit_proxy


@pytest.mark.parametrize("statuses", [[429, 429, 200], [429, 429, 429], [400], [500]])
def test_sandbox_forwarder_retries_only_429_and_retains_every_attempt(tmp_path, statuses):
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(
                (self.path, self.headers["x-session-id"], self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(statuses[len(requests) - 1])
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b"data: first\n\ndata: last\n\n")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    attempts_path = tmp_path / "attempts.jsonl"
    try:
        with rate_limit_proxy(
            f"http://127.0.0.1:{upstream.server_port}/v1", "invocation", attempts_path, timeout=10
        ) as proxy:
            try:
                response = urlopen(
                    Request(f"{proxy}/responses", data=b'{"input":"task"}', headers={"x-session-id": "native-session"})
                )
            except HTTPError as error:
                response = error
            with response:
                assert response.status == statuses[-1]
                assert response.headers["Content-Type"] == "text/event-stream"
                assert response.read() == b"data: first\n\ndata: last\n\n"
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()
    assert requests == [("/v1/responses", "invocation", b'{"input":"task"}')] * len(statuses)
    assert [json.loads(line)["status"] for line in attempts_path.read_text().splitlines()] == statuses
