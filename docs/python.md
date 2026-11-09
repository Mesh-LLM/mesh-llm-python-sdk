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

`provider.metadata` preserves advertised output formats, independently of image
or audio input capabilities:

```python
model = await node.models.get("org/model@revision#profile")
for provider in model.providers:
    outputs = (provider.metadata.get("output_modalities")
               if provider.metadata is not None else None)
    print(provider.provider_id, outputs)
```

Missing or empty `output_modalities` means unknown, never text-only. Future names
are preserved. Read individual providers: model-level metadata is present only
when providers agree. This declaration does not choose what a prompt generates
or change routing. Use the matching SDK and native bridge together.

Membership controls, live transcription, native runtime management and the
full native `Node.from_config` constructor have the same capabilities as the
other four SDKs. See the [package guide](../sdk/README.md) for complete examples,
cleanup and matching native bridge requirements.

## Live transcription

```python
from meshllm import LiveTranscriptionOptions

session = await node.start_live_transcription(LiveTranscriptionOptions(
    model="speech-model", sample_rate=16000, turn_detection=None,
    step_ms=400, tentative=True,
))
async with session:
    await session.send_audio(pcm16_samples)
    await session.commit()
    event = await session.next_event()
    remaining = await session.stop()
```

`step_ms` accepts 200 to 5000 ms; `None` keeps the node's 1000 ms default.
Shorter steps run more transcription passes. `tentative` defaults to `False`.
Each tentative event replaces the previous unconfirmed text for its `item_id`;
empty text clears it. Append `delta` text to the confirmed transcript and let
`completed` replace the whole turn, including tentative text. Read events
concurrently with microphone input. `stop()` finishes the turn and returns
unread events; context exit closes the session, including on errors.

Confirmed deltas are append-only. Recognition errors in confirmed text cannot
be corrected by later passes or the completed event. Shorter transcription
intervals do not guarantee better recognition accuracy.

Native transcription failures end the session after an error event. Start a
new session to retry. Request-validation errors may leave the session open.

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

`node.inference.list_models()` also preserves advertised `thinking_efforts` as a
tuple of supported levels. `None` means unknown; an empty tuple means the model
advertises no effort levels. Select an advertised level before setting
`reasoning_effort` in a request.
