# Python SDK

## Model discovery and OCR

`node.models` lists every workload advertised in the joined mesh, including
local and remote OCR, speech, embedding and decision providers. The existing
`node.inference.list_models()` list contains only compatible LLM offerings.

```python
from pathlib import Path
from meshllm import DocumentOcrOptions

page = await node.models.list(workload_class="document_ocr", available=True)
model = await node.models.get(page.data[0].id)
response = await node.inference.document_ocr(
    Path("scan.pdf").read_bytes(),
    DocumentOcrOptions(model=model.id, filename="scan.pdf",
                       content_type="application/pdf", response_format="markdown"),
)
print(response.body)
```

Lists return a typed page with `data` and `next_cursor`. Keep filters unchanged
when continuing; expired snapshot cursors require restarting the list. Provider
runtime limits and observations remain separate. Missing metadata is unknown.
OCR preserves JSON/text/Markdown provider responses; non-success statuses raise
by default. Use `raise_for_status=False` for an explicit raw response.

Membership controls, live transcription, native runtime management and the
full native `Node.from_config` constructor have the same capabilities as the
other four SDKs. See the [package guide](../sdk/README.md) for complete examples,
cleanup and matching native bridge requirements.

## Create an embedded node

The Python package exposes one `Node` with `client`, `serve`, and `combined`
roles. `serve` is serve-only; inference methods reject that role.

```python
from meshllm import Node

async with Node.create(mode="client", auto_join=True) as node:
    models = await node.inference.list_models()
    response = await node.inference.chat_completions({
        "model": models[0].id,
        "messages": [{"role": "user", "content": "Hello"}],
    })
```

For a private mesh, use `join_tokens=(token,)` instead of `auto_join=True`.
To serve, use `mode="serve"` or `mode="combined"` and pass
`models=("publisher/model:quant",)`. Serving requires an installed matching
native runtime. Set `owner_key_path` to a Mesh LLM owner keystore path when
owner identity is required. The old in-memory keypair hex constructor and
`Client` class have been removed.

Streaming is available through `node.inference.stream_chat_completions(...)`
and `stream_responses(...)`; each event carries the original SSE data and raw
frame. See [native runtime guidance](https://github.com/Mesh-LLM/mesh-llm/blob/20444d0fe80edac46584ef5a2f065c1044ea4aaf/mesh/docs/SDK.md#native-runtime-artifacts).
