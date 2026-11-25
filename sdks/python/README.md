# `arkhiverag` (Python SDK)

Zero-dependency Python 3.10+ client for **ArkhiveRAG** built on the Python standard library.

## Quick Start

```python
from arkhiverag import ArkhiveRAGClient

client = ArkhiveRAGClient(
    base_url="http://localhost:3002",
    api_key="ak_live_...",  # or pass token="<keycloak-jwt>"
)

# 1. Adaptive RAG chat
reply = client.chat(
    messages=[{"role": "user", "content": "Summarize our expense policy."}],
    mode="auto",
    agentic=True,
)
print(reply["choices"][0]["message"]["content"])

# 2. Streaming SSE chat
for event in client.chat_stream(
    messages=[{"role": "user", "content": "Key highlights?"}],
    mode="mix",
):
    if event.get("type") == "chunk":
        print(event.get("content", ""), end="", flush=True)

# 3. RBAC Company Agent
agent = client.agent_query("How many active documents are indexed?")
print(agent["data"]["answer"])
```
