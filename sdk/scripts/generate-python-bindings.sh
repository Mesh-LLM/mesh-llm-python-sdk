#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && "$1" == /* && -d "$1" && "$2" == /* && -x "$2" && "$3" == /* && -f "$3" && ! -L "$3" ]] || { echo 'usage: generate-python-bindings.sh ABS_MESH_SOURCE ABS_UNIFFI_0_32_BINDGEN ABS_PREBUILT_MESH_FFI_LIBRARY' >&2; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "$1/mesh/crates/mesh-llm-ffi/src/mesh_ffi.udl" && ! -L "$1/mesh/crates/mesh-llm-ffi/src/mesh_ffi.udl" ]] || exit 1
[[ "$("$2" --version)" == 'uniffi-bindgen 0.32.0' ]] || { echo 'UniFFI 0.32.0 required' >&2; exit 1; }
# Resolve the matching Mesh crate/UDL while deriving checksums from the actual library metadata.
cd "$1"
"$2" generate "$3" --language python --out-dir "$ROOT/sdk/src/meshllm/_generated" --no-format --crate meshllm_ffi --config "$ROOT/sdk/uniffi.toml" --metadata-no-deps
perl -pi -e 's/[ \t]+$//' "$ROOT/sdk/src/meshllm/_generated/mesh_ffi.py"
