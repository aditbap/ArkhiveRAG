import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from arkhiverag import ArkhiveRAGClient, ArkhiveRAGError


class _StubHandler(BaseHTTPRequestHandler):
    recorded: list[tuple[str, str, dict[str, str], bytes]] = []

    def log_message(self, format, *args):  # noqa: A002
        return

    def _record(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        hdrs = {k.lower(): v for k, v in self.headers.items()}
        self.__class__.recorded.append((self.command, self.path, hdrs, body))
        return body

    def do_GET(self):
        self._record()
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/api/v1/documents":
            self._send_json(401, {"error": {"code": "UNAUTHORIZED", "message": "Invalid key", "requestId": "req-py-1"}})
            return
        self._send_json(200, {"data": []})

    def do_POST(self):
        body = self._record()
        if self.path == "/api/v1/agent/query":
            parsed = json.loads(body.decode("utf-8"))
            self._send_json(200, {"data": {"decision": "execute", "answer": f"Echo: {parsed['query']}", "refused": False}})
            return
        if self.path == "/api/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"type":"chunk","content":"Hi "}\n\n')
            self.wfile.write(b'data: {"type":"chunk","content":"Python"}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if self.path == "/api/v1/documents/upload":
            self._send_json(201, {"data": {"uploaded": True, "size": len(body)}})
            return
        self._send_json(404, {"error": {"code": "NOT_FOUND", "message": "Not found"}})

    def _send_json(self, status: int, payload: dict):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class TestArkhiveRAGClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _StubHandler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _StubHandler.recorded.clear()

    def test_agent_query_headers_and_payload(self):
        client = ArkhiveRAGClient(self.base_url, api_key="ak_live_test_key")
        res = client.agent_query("Check inventory")
        self.assertEqual(res["data"]["answer"], "Echo: Check inventory")
        method, path, hdrs, _ = _StubHandler.recorded[-1]
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/api/v1/agent/query")
        self.assertEqual(hdrs.get("x-api-key"), "ak_live_test_key")

    def test_chat_stream_parses_sse(self):
        client = ArkhiveRAGClient(self.base_url, token="jwt-123")
        chunks = [ev["content"] for ev in client.chat_stream([{"role": "user", "content": "Hi"}]) if ev.get("type") == "chunk"]
        self.assertEqual("".join(chunks), "Hi Python")

    def test_upload_document_multipart(self):
        client = ArkhiveRAGClient(self.base_url, api_key="ak_live_test_key")
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            tf.write(b"hello arkhiverag")
            tmp_path = tf.name
        try:
            res = client.upload_document(tmp_path, notes="test upload")
            self.assertTrue(res["data"]["uploaded"])
            _, _, hdrs, body = _StubHandler.recorded[-1]
            self.assertIn("multipart/form-data; boundary=", hdrs.get("content-type", ""))
            self.assertIn(b"hello arkhiverag", body)
        finally:
            os.unlink(tmp_path)

    def test_structured_http_error(self):
        client = ArkhiveRAGClient(self.base_url, api_key="ak_live_bad")
        with self.assertRaises(ArkhiveRAGError) as ctx:
            client.list_documents()
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.code, "UNAUTHORIZED")
        self.assertEqual(ctx.exception.request_id, "req-py-1")


if __name__ == "__main__":
    unittest.main()
