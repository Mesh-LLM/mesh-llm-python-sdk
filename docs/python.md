# Python SDK

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

## Discovering thinking effort levels

`list_models()` keeps the reasoning effort levels each provider advertises for
a serving ID, so an application can select a level the model supports instead
of sending one blindly:

```python
models = await node.inference.list_models()
model = next(
    model for model in models
    if model.thinking_efforts and "high" in model.thinking_efforts
)

response = await node.inference.chat_completions({
    "model": model.id,
    "messages": [{"role": "user", "content": "Explain lock ordering."}],
    "reasoning_effort": "high",
})
```

`thinking_efforts` is `None` when the provider advertises no effort metadata —
an older provider or a model the host has not probed yet — and an empty tuple
when the provider reports that the model's template exposes no effort levels.
The list comes from the node's OpenAI-compatible `/v1/models` view, which
merges the mesh providers serving that ID, so it describes the level set the
mesh offers for the model rather than one peer's private view.
