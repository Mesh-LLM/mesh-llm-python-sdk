#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* && -d "$1" && "$2" == /* && -x "$2" ]] || { echo 'usage: generate-python-bindings.sh ABS_MESH_SOURCE ABS_UNIFFI_0_32_BINDGEN' >&2; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UDL="$1/mesh/crates/mesh-llm-ffi/src/mesh_ffi.udl"
[[ -f "$UDL" && ! -L "$UDL" ]] || exit 1
[[ "$("$2" --version)" == *'0.32.0'* ]] || { echo 'UniFFI 0.32.0 required' >&2; exit 1; }
"$2" generate "$UDL" --language python --out-dir "$ROOT/sdk/src/meshllm/_generated" --no-format
perl -pi -e 's/[ \t]+$//' "$ROOT/sdk/src/meshllm/_generated/mesh_ffi.py"
