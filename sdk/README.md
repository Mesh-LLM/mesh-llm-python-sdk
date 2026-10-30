# MeshLLM Python SDK

The package exposes one real embedded `Node` with client, serve-only, and
combined roles. The legacy thin `Client` is removed. See the
[python usage guide](../docs/python.md) for current constructors,
streaming, role selection, and native runtime requirements.

The `serve` role serves local models; `combined` also permits mesh inference.
Client mode starts no native serving runtime. Supply a Mesh LLM owner keystore
path through `ownerKeyPath`/`owner_key_path` if owner identity is required.

## Component tests

From the repository root, use an existing Python 3.10 or newer interpreter:

```bash
python3 -I sdk/tests/test_client.py
```

This existing unittest runner adds only this component's source directory to
its import path. It tests the actual Python node lifecycle, cancellation and
stream behavior with local native-handle fixtures; no external Python test
packages, installed wheel, native build, model or endpoint are required for
these cases. Packaging separately uses the Hatchling build dependency declared
in `sdk/pyproject.toml`; building and publishing a wheel still requires
the matching native bridge.

This is an optional language-component command, separate from generic
`just ci-validate`. It does not replace or qualify the four retained external
SDK clients, their required smoke/release or explicit embedding-workload
cadence, or real platform/model compatibility.

## Native bridge bindings

Generate bindings from the exact prebuilt Mesh FFI library using UniFFI 0.32.0.
The source checkout supplies the matching UDL and crate metadata. The library
supplies the exported Rust metadata and API checksums; UDL-only generation
can produce incompatible checksums for Mesh's combined UDL/proc-macro surface.

```bash
sdk/scripts/generate-python-bindings.sh /absolute/mesh-source /absolute/uniffi-bindgen /absolute/libmeshllm_ffi.dylib
sdk/scripts/build-native.sh /absolute/libmeshllm_ffi.dylib
python3 -I -B sdk/tests/test_binding.py
```

Use the matching `.so` or `.dll` on other platforms. The SDK-owned lazy loader
checks every generated API checksum before caching the native module. Generated
UniFFI output is preserved apart from trailing whitespace normalization.
The component tests above use mock handles; the binding tests require the real
sibling bridge and test checksum refusal and native version readback without
constructing a node or starting services. Neither test suite qualifies model
serving or a published wheel.
## Mesh model discovery

`node.models` queries the embedded node's observed mesh advertisements directly,
including local and remote LLM, speech, embedding, decision and document OCR
providers. Existing `node.inference.list_models()` remains the LLM-only
compatibility list.

```python
page = await node.models.list()
ocr = await node.models.list(workload_class="document_ocr", available=True)
model = await node.models.get("org/model@revision#profile")
if page.next_cursor is not None:
    following = await node.models.list(cursor=page.next_cursor)
```

Results are `MeshModelPage`, `MeshModel`, and `MeshModelProvider` dataclasses.
Provider runtime limits, capabilities and optional thinking/payment observations
remain separate. Missing metadata is unknown. Workload strings preserve future
values. Filters combine with AND and use the same native validation as the Mesh
HTTP API. Discovery reflects observed advertisements, not a reservation or a
download catalog. Use a native bridge generated from the matching Mesh release;
older bridges do not expose these methods.

## OCR inference

Discover `document_ocr` models, then submit PDF or image bytes directly:

```python
from pathlib import Path
from meshllm import DocumentOcrOptions

page = await node.models.list(workload_class="document_ocr", available=True)
response = await node.inference.document_ocr(
    Path("scan.pdf").read_bytes(),
    DocumentOcrOptions(model=page.data[0].id, filename="scan.pdf",
                       content_type="application/pdf", response_format="markdown"),
)
print(response.body)
```

The response preserves provider status, content type and body. JSON responses
also expose `.json()`. Non-success statuses raise `OpenAIRequestError` by default;
pass `raise_for_status=False` to inspect a raw error response. OCR uses multipart
`/mesh/v1/documents/ocr`, never chat completions. Empty files, files over 64 MiB and
invalid options are rejected by the native SDK.

## Additional mesh memberships

```python
from meshllm import MeshMembershipConfig

team = await node.add_membership(MeshMembershipConfig(
    name="team", api_port=9338, join_tokens=(invite,), trusted_owners=(owner,),
))
memberships = await node.memberships()
await node.remove_membership("team")
```

Memberships isolate peer visibility and routing. `node.models` observes the
primary mesh and does not merge other memberships' inventories.

## Live transcription

```python
from meshllm import LiveTranscriptionOptions

live = await node.start_live_transcription(LiveTranscriptionOptions(
    model="whisper-model", sample_rate=48000, turn_detection=None,
))
async with live:
    await live.send_audio(pcm16_samples)
    await live.commit()
    event = await live.next_event()
    remaining = await live.stop()
```

Use signed 16-bit mono samples or `send_audio_f32` with float samples. Read events
concurrently with microphone input through `next_event()` or `events()`. Deltas
are confirmed text; completed events contain a full turn. `stop()` finishes the
turn and returns unread events; `close()` cancels. Context exit closes the
session. Transcription uses the existing WebSocket inference transport; only
the model catalog queries promise no HTTP dependency.

## Native runtime management

```python
from meshllm import (
    NativeRuntimeInstallOptions, install_native_runtime, installed_native_runtimes,
    resolve_native_runtime, validate_native_runtime,
)

outcome = await install_native_runtime(NativeRuntimeInstallOptions(selection="recommended"))
cached = await installed_native_runtimes()
runtime = await resolve_native_runtime()  # offline unless allow_download=True
verified = await validate_native_runtime("/absolute/runtime-bundle")
```

`current_mesh_version`, `current_skippy_abi_version`, `remove_native_runtime`
and `prune_native_runtimes` expose the corresponding native controls. Installation
requires checksums by default and accepts an optional progress callback. These
are explicit application operations; node startup does not install missing
runtimes automatically.

For advanced network, storage and admission settings use the same full native
configuration schema in every SDK:

```python
node = Node.from_config({"mode": "client", "network": {"mesh_name": "private"}})
```

This constructor preserves native defaults and rejects unknown fields. Native
duration values use `{"secs": 30, "nanos": 0}`. `Node.create` remains the shorter
constructor for common settings; full configuration does not introduce an
alternate trust policy or bypass native startup validation.
