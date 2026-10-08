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
