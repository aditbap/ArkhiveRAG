from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Iterator


class ArkhiveRAGError(Exception):
    """Structured error raised on non-2xx API responses."""

    def __init__(self, status: int, code: str, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id


class ArkhiveRAGClient:
    """Zero-dependency HTTP/SSE client for ArkhiveRAG."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        token: str | None = None,
        org_id: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token = token
        self.org_id = org_id
        self.timeout = timeout

    def _headers(self, extra: dict[str, str] | None = None, json_body: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        elif self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.org_id:
            headers["X-Org-Id"] = self.org_id
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        data = raw_body
        hdrs = self._headers(headers, json_body=(body is not None and raw_body is None))
        if body is not None and raw_body is None:
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 204:
                    return None
                raw = resp.read()
                return json.loads(raw.decode("utf-8")) if raw else None
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8", errors="replace")
            try:
                err_obj = json.loads(raw_err).get("error", {})
            except Exception:
                err_obj = {}
            raise ArkhiveRAGError(
                exc.code,
                str(err_obj.get("code", "HTTP_ERROR")),
                str(err_obj.get("message", raw_err or f"HTTP {exc.code}")),
                err_obj.get("requestId"),
            ) from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def ready(self) -> dict[str, Any]:
        return self._request("GET", "/ready")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        mode: str = "auto",
        agentic: bool | None = None,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
        include_drafts: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"messages": messages, "mode": mode, "stream": False}
        if agentic is not None:
            payload["agentic"] = agentic
        if conversation_id is not None:
            payload["conversationId"] = conversation_id
        if document_ids is not None:
            payload["documentIds"] = document_ids
        if include_drafts is not None:
            payload["includeDrafts"] = include_drafts
        return self._request("POST", "/api/v1/chat/completions", body=payload)

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        mode: str = "auto",
        agentic: bool | None = None,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {"messages": messages, "mode": mode, "stream": True}
        if agentic is not None:
            payload["agentic"] = agentic
        if conversation_id is not None:
            payload["conversationId"] = conversation_id
        if document_ids is not None:
            payload["documentIds"] = document_ids

        req = urllib.request.Request(
            f"{self.base_url}/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(json_body=True),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8", errors="replace")
            try:
                err_obj = json.loads(raw_err).get("error", {})
            except Exception:
                err_obj = {}
            raise ArkhiveRAGError(
                exc.code,
                str(err_obj.get("code", "HTTP_ERROR")),
                str(err_obj.get("message", raw_err or f"HTTP {exc.code}")),
                err_obj.get("requestId"),
            ) from exc

    def agent_tools(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/agent/tools")

    def agent_query(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        synthesize: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "synthesize": synthesize}
        if document_ids is not None:
            payload["documentIds"] = document_ids
        return self._request("POST", "/api/v1/agent/query", body=payload)

    def list_documents(
        self, *, limit: int | None = None, offset: int | None = None, folder_id: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        if folder_id:
            params["folderId"] = folder_id
        qs = f"?{urllib.parse.urlencode(params)}" if params else ""
        return self._request("GET", f"/api/v1/documents{qs}")

    def upload_document(
        self, file_path: str, *, folder_id: str | None = None, notes: str | None = None
    ) -> dict[str, Any]:
        boundary = f"----ArkhiveRAGBoundary{uuid.uuid4().hex}"
        filename = os.path.basename(file_path)
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(file_path, "rb") as fh:
            file_bytes = fh.read()

        chunks: list[bytes] = []
        if folder_id:
            chunks.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="folderId"\r\n\r\n{folder_id}\r\n'.encode("utf-8")
            )
        if notes:
            chunks.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="notes"\r\n\r\n{notes}\r\n'.encode("utf-8")
            )
        chunks.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n"
            ).encode("utf-8")
            + file_bytes
            + f"\r\n--{boundary}--\r\n".encode("utf-8")
        )
        body_bytes = b"".join(chunks)
        return self._request(
            "POST",
            "/api/v1/documents/upload",
            raw_body=body_bytes,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def update_governance(self, document_id: str, governance_status: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"governanceStatus": governance_status, **kwargs}
        return self._request("PATCH", f"/api/v1/documents/{urllib.parse.quote(document_id)}/governance", body=payload)

    def list_libraries(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/libraries")

    def list_api_keys(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/api-keys")

    def create_api_key(
        self, name: str, *, role: str = "member", expires_in_days: int | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name, "role": role}
        if expires_in_days is not None:
            payload["expiresInDays"] = expires_in_days
        return self._request("POST", "/api/v1/api-keys", body=payload)

    def revoke_api_key(self, key_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/api-keys/{urllib.parse.quote(key_id)}")
