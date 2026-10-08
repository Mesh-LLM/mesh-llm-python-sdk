# MeshLLM Python SDK

The Python SDK and genuine compatibility clients are versioned here separately from MeshLLM.

See [SDK guide](docs/python.md), [package](sdk/README.md), and the exact dependency locks under ci/. Mesh consumers admit an immutable source checkpoint before executing these clients. Native bridge production stays in MeshLLM; consumers never compile a bridge as a fallback.

The immutable extraction source is published at [Mesh-LLM/mesh-llm-python-sdk](https://github.com/Mesh-LLM/mesh-llm-python-sdk). Source publication does not qualify a native bridge, packaged wheel, or model-serving environment.
