from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from meshllm import (
    DocumentOcrOptions,
    LiveTranscriptionOptions,
    MeshMembershipConfig,
    NativeRuntimeInstallOptions,
    install_native_runtime,
    installed_native_runtimes,
    prune_native_runtimes,
    remove_native_runtime,
    resolve_native_runtime,
    validate_native_runtime,
    Node,
    OpenAIRequestError,
    OpenAIStreamChunk,
    OpenAIStreamStarted,
)


class FakeOpenAIStreamEvent:
    def __init__(self, kind: str, **values: object) -> None:
        self.kind = kind
        self.request_id = "stream-1"
        self.status_code = values.get("status_code")
        self.content_type = values.get("content_type")
        self.event_type = values.get("event_type")
        self.data = values.get("data")
        self.raw = values.get("raw")
        self.error = values.get("error")
        self.body = values.get("body")

    def is_started(self) -> bool:
        return self.kind == "started"

    def is_sse(self) -> bool:
        return self.kind == "sse"

    def is_completed(self) -> bool:
        return self.kind == "completed"

    def is_failed(self) -> bool:
        return self.kind == "failed"


class FakeHandle:
    def __init__(self) -> None:
        self.started = False
        self.cancelled: list[str] = []
        self.last_openai: tuple[str, dict[str, object]] | None = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def status(self) -> object:
        return SimpleNamespace(
            running=self.started,
            mode="client",
            api_base_url="http://127.0.0.1:9337/v1" if self.started else "",
            console_url="http://127.0.0.1:3131" if self.started else "",
            payload_json='{"peers":[]}' if self.started else "null",
        )

    def inference_list_models(self) -> list[object]:
        return [SimpleNamespace(id="model-a", name="Model A", context_length=131_072)]

    def openai_request(self, path: str, body_json: str) -> object:
        body = json.loads(body_json)
        self.last_openai = (path, body)
        if body.get("model") == "missing":
            return SimpleNamespace(status_code=404, content_type="application/json", body='{"error":"missing"}')
        response = {
            "choices": [{
                "message": {"tool_calls": body.get("tools", [])},
                "finish_reason": "tool_calls",
            }],
            "usage": {"total_tokens": 12},
        }
        return SimpleNamespace(status_code=200, content_type="application/json", body=json.dumps(response))

    def openai_stream(self, path: str, body_json: str, listener: object) -> str:
        body = json.loads(body_json)
        self.last_openai = (path, body)
        if body.get("model") == "missing":
            listener.on_event(
                FakeOpenAIStreamEvent(
                    "failed",
                    status_code=404,
                    error="model not found",
                    body='{"error":"missing"}',
                )
            )
            return "stream-1"
        listener.on_event(FakeOpenAIStreamEvent(
            "started", status_code=200, content_type="text/event-stream"
        ))
        tool_delta = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"city":"Syd'},
                    }],
                },
            }],
        }
        data = json.dumps(tool_delta)
        listener.on_event(FakeOpenAIStreamEvent(
            "sse",
            event_type=(
                "response.function_call_arguments.delta"
                if path == "/v1/responses"
                else None
            ),
            data=data,
            raw=f"data: {data}\n\n",
        ))
        listener.on_event(FakeOpenAIStreamEvent("sse", data="[DONE]", raw="data: [DONE]\n\n"))
        listener.on_event(FakeOpenAIStreamEvent("completed"))
        return "stream-1"

    def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


class NodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_native_configuration_is_forwarded_without_losing_settings(self) -> None:
        calls = []
        handle = FakeHandle()
        config = {"mode": "client", "network": {"mesh_name": "private", "disable_iroh_relays": True},
                  "storage": {"isolated_config": True}}
        binding = SimpleNamespace(create_node_config_json=lambda data: (calls.append(json.loads(data)), handle)[1])
        with patch("meshllm.client.native", return_value=binding):
            node = Node.from_config(config)
        self.assertIs(node._handle, handle)
        self.assertEqual(calls, [config])

    async def test_native_runtime_facade_preserves_policy_and_offline_resolution(self) -> None:
        calls = []
        installed = SimpleNamespace(
            mesh_version="1.0", native_runtime_id="cpu", flavor="cpu", path="/cache/cpu",
            skippy_abi_version="abi",
        )

        def install(options, progress):
            calls.append(options)
            if progress is not None:
                progress.on_progress(SimpleNamespace(
                    native_runtime_id="cpu", url="https://example.invalid/cpu",
                    downloaded_bytes=10, total_bytes=10, finished=True,
                ))
            return SimpleNamespace(
                status="cached", runtime=installed, selected_native_runtime_id="cpu",
                selected_source="bundle",
            )

        binding = SimpleNamespace(
            NativeRuntimeVerificationPolicyNative=SimpleNamespace(
                REQUIRE_CHECKSUM="checksum", REQUIRE_CHECKSUM_AND_SIGNATURE="signature",
            ),
            NativeRuntimePruneModeNative=SimpleNamespace(
                KEEP_ACTIVE_AND_PREVIOUS="keep", ACTIVE_ONLY="active",
            ),
            NativeRuntimeInstallOptionsNative=lambda **kw: SimpleNamespace(**kw),
            install_native_runtime=install,
            installed_native_runtimes=lambda cache: [installed],
            remove_native_runtime=lambda cache, version, runtime: (cache, version, runtime) == ("/cache", "1.0", "cpu"),
            prune_native_runtimes=lambda cache, version, mode: SimpleNamespace(removed_dirs=["/old"]),
        )
        progress = []
        with patch("meshllm.native_runtime.native", return_value=binding):
            outcome = await install_native_runtime(
                NativeRuntimeInstallOptions(verification_policy="require_checksum_and_signature"),
                on_progress=progress.append,
            )
            self.assertEqual(calls[-1].verification_policy, "signature")
            self.assertEqual(outcome.runtime.native_runtime_id, "cpu")
            self.assertTrue(progress[0].finished)
            self.assertEqual((await installed_native_runtimes())[0], outcome.runtime)
            await resolve_native_runtime()
            self.assertFalse(calls[-1].allow_download)
            await resolve_native_runtime(allow_download=True)
            self.assertTrue(calls[-1].allow_download)
            await validate_native_runtime("/explicit")
            self.assertEqual(calls[-1].bundle_dirs, ["/explicit"])
            self.assertFalse(calls[-1].allow_download)
            self.assertTrue(await remove_native_runtime("1.0", "cpu", cache_dir="/cache"))
            self.assertEqual((await prune_native_runtimes()).removed_dirs, ("/old",))
            with self.assertRaises(ValueError):
                await install_native_runtime(NativeRuntimeInstallOptions(verification_policy="none"))
            with self.assertRaises(ValueError):
                await prune_native_runtimes(mode="everything")

    async def test_ocr_binary_payload_and_raw_provider_response(self) -> None:
        calls = []

        def ocr(file, options_json):
            calls.append((file, json.loads(options_json)))
            return SimpleNamespace(status_code=200, content_type="text/markdown", body="# page\n")

        file = b"%PDF-1.7\x00\xff\r\n"
        node = Node(SimpleNamespace(document_ocr=ocr))
        response = await node.inference.document_ocr(file, DocumentOcrOptions(
            model="org/ocr@rev#profile", filename="scan.pdf", content_type="application/pdf",
            response_format="markdown", max_pages=2,
        ))
        self.assertEqual(calls, [(file, {
            "model": "org/ocr@rev#profile", "filename": "scan.pdf",
            "content_type": "application/pdf", "response_format": "markdown", "max_pages": 2,
        })])
        self.assertEqual(response.body, "# page\n")
        self.assertEqual(response.content_type, "text/markdown")

        def failed(*_):
            return SimpleNamespace(status_code=422, content_type="application/json", body='{"error":"pdf"}')

        node = Node(SimpleNamespace(document_ocr=failed))
        with self.assertRaises(OpenAIRequestError) as raised:
            await node.inference.document_ocr(file, DocumentOcrOptions(model="ocr"))
        self.assertEqual(raised.exception.status_code, 422)
        raw = await node.inference.document_ocr(
            file, DocumentOcrOptions(model="ocr"), raise_for_status=False,
        )
        self.assertEqual(raw.json(), {"error": "pdf"})

    async def test_membership_native_config_and_status_round_trip(self) -> None:
        calls = []
        status = {
            "name": "team", "api_base_url": "http://127.0.0.1:9338/v1",
            "invite_token": "invite", "endpoint_id": "endpoint", "peers": 2,
            "join_error": "peer pending",
        }

        def add(config_json):
            calls.append(json.loads(config_json))
            return json.dumps(status)

        def remove(name):
            calls.append(name)
            return True

        node = Node(SimpleNamespace(
            add_membership_json=add, remove_membership=remove,
            memberships_json=lambda: json.dumps([status]),
        ))
        added = await node.add_membership(MeshMembershipConfig(
            name="team", api_port=9338, join_tokens=("token",), trusted_owners=("owner",),
        ))
        self.assertEqual(calls[0], {
            "name": "team", "api_port": 9338, "join_tokens": ["token"],
            "trusted_owners": ["owner"], "node_key_path": None, "enumerate_host": False,
        })
        self.assertEqual((await node.memberships())[0], added)
        self.assertEqual(added.join_error, "peer pending")
        self.assertTrue(await node.remove_membership("team"))
        self.assertEqual(calls[-1], "team")

    async def test_live_transcription_audio_turns_and_cleanup(self) -> None:
        calls = []

        class Session:
            def send_audio(self, values): calls.append(("pcm", values))
            def send_audio_f32(self, values): calls.append(("float", values))
            def commit(self): calls.append("commit")
            def next_event_json(self): return json.dumps({"type": "delta", "item_id": "turn", "text": "Hi"})
            def stop_json(self): return json.dumps([{"type": "completed", "item_id": "turn", "transcript": "Hi!"}])
            def close_audio(self): calls.append("close")

        def start(options_json):
            calls.append(json.loads(options_json))
            return Session()

        node = Node(SimpleNamespace(start_live_transcription_json=start))
        live = await node.start_live_transcription(LiveTranscriptionOptions(
            model="whisper", sample_rate=48000, turn_detection=None,
        ))
        self.assertEqual(calls[0]["sample_rate"], 48000)
        self.assertIsNone(calls[0]["turn_detection"])
        async with live:
            await live.send_audio((-32768, 0, 32767))
            await live.send_audio_f32((-1.0, 0.0, 1.0))
            await live.commit()
            event = await live.next_event()
            self.assertEqual((event.type, event.item_id, event.text), ("delta", "turn", "Hi"))
            remaining = await live.stop()
            self.assertEqual(remaining[0].transcript, "Hi!")
        self.assertIn(("pcm", [-32768, 0, 32767]), calls)
        self.assertIn(("float", [-1.0, 0.0, 1.0]), calls)
        self.assertEqual(calls[-1], "close")

    async def test_mesh_discovery_preserves_ocr_and_provider_observations(self) -> None:
        model = {
            "id": "org/ocr@revision#profile",
            "display_name": "OCR",
            "kind": "model",
            "workload_class": "document_ocr",
            "availability": "available",
            "operations": ["document_ocr"],
            "providers": [
                {
                    "provider_id": "local", "local": True, "available": True,
                    "operations": ["document_ocr"], "workload_class": "document_ocr",
                    "runtime": {"context_length": 4096},
                },
                {
                    "provider_id": "remote", "local": False, "available": True,
                    "operations": ["document_ocr"], "workload_class": "document_ocr",
                },
            ],
        }
        calls = []

        def list_models(query_json):
            calls.append(json.loads(query_json))
            return json.dumps({"data": [model], "next_cursor": "opaque-cursor"})

        def lookup(model_id):
            calls.append(model_id)
            return json.dumps(model)

        handle = SimpleNamespace(mesh_models_json=list_models, mesh_model_json=lookup)
        node = Node(handle)  # No OpenAI/inference discovery method exists on this handle.
        page = await node.models.list(
            workload_class="document_ocr", kind="model", available=False,
            limit=1, cursor="prior-cursor",
        )
        self.assertEqual(calls[0], {
            "workload_class": "document_ocr", "kind": "model", "available": False,
            "limit": 1, "cursor": "prior-cursor",
        })
        self.assertEqual(page.next_cursor, "opaque-cursor")
        self.assertEqual(page.data[0].workload_class, "document_ocr")
        self.assertEqual(page.data[0].providers[0].runtime["context_length"], 4096)
        self.assertIsNone(page.data[0].providers[1].runtime)
        result = await node.models.get(model["id"])
        self.assertEqual(calls[1], model["id"])
        self.assertEqual(result, page.data[0])

    async def test_mesh_discovery_defaults_empty_page_and_unknown_values(self) -> None:
        queries = []
        payload = {"data": [], "next_cursor": None}

        def list_models(query_json):
            queries.append(json.loads(query_json))
            return json.dumps(payload)

        node = Node(SimpleNamespace(mesh_models_json=list_models))
        page = await node.models.list()
        self.assertEqual(queries, [{}])
        self.assertEqual(page.data, ())
        self.assertIsNone(page.next_cursor)
        payload["data"] = [{
            "id": "future", "display_name": "Future", "kind": "future_kind",
            "workload_class": "future_workload", "availability": "future_state",
            "operations": ["future_operation"], "providers": [], "future_field": True,
        }]
        model = (await node.models.list()).data[0]
        self.assertEqual(model.workload_class, "future_workload")
        self.assertEqual(model.kind, "future_kind")
        self.assertEqual(model.availability, "future_state")
        self.assertIsNone(model.metadata)

    async def test_mesh_discovery_propagates_native_errors(self) -> None:
        def fail(_):
            raise RuntimeError("catalog unavailable")

        node = Node(SimpleNamespace(mesh_models_json=fail, mesh_model_json=fail))
        with self.assertRaisesRegex(RuntimeError, "catalog unavailable"):
            await node.models.list()
        with self.assertRaisesRegex(RuntimeError, "catalog unavailable"):
            await node.models.get("missing")

    async def test_create_embedded_roles(self) -> None:
        calls: list[tuple[object, ...]] = []
        handle = FakeHandle()
        binding = SimpleNamespace(create_node=lambda *args: (calls.append(args), handle)[1])
        with patch("meshllm.client.native", return_value=binding):
            for mode in ("client", "serve", "combined"):
                node = Node.create(mode=mode, models=("model-a",))
                self.assertIs(node._handle, handle)
        self.assertEqual([call[0] for call in calls], ["client", "serve", "combined"])
        self.assertTrue(all(call[2] == ["model-a"] for call in calls))

    async def test_lifecycle_and_models(self) -> None:
        handle = FakeHandle()
        client = Node(handle)

        async with client:
            self.assertTrue((await client.status()).running)
            model = (await client.inference.list_models())[0]
            self.assertEqual(model.id, "model-a")
            self.assertEqual(model.context_length, 131_072)

        self.assertFalse(handle.started)

    async def test_agent_request_preserves_tools_and_full_response(self) -> None:
        handle = FakeHandle()
        client = Node(handle)
        tool = {"type": "function", "function": {"name": "search"}}

        result = await client.inference.chat_completions({
            "model": "model-a",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "find it"}]}],
            "tools": [tool],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}},
            "stream": True,
        })

        self.assertEqual(handle.last_openai[0], "/v1/chat/completions")
        self.assertEqual(handle.last_openai[1]["tools"], [tool])
        self.assertFalse(handle.last_openai[1]["stream"])
        self.assertEqual(result["choices"][0]["message"]["tool_calls"], [tool])
        self.assertEqual(result["usage"]["total_tokens"], 12)

    async def test_non_success_response_raises_typed_error(self) -> None:
        client = Node(FakeHandle())

        with self.assertRaises(OpenAIRequestError) as raised:
            await client.inference.chat_completions({"model": "missing", "messages": []})

        self.assertEqual(raised.exception.status_code, 404)

    async def test_agent_stream_preserves_tool_call_deltas_and_raw_sse(self) -> None:
        handle = FakeHandle()
        events = [
            event
            async for event in Node(handle).inference.stream_chat_completions({
                "model": "model-a",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{"type": "function", "function": {"name": "weather"}}],
            })
        ]

        self.assertIsInstance(events[0], OpenAIStreamStarted)
        self.assertIsInstance(events[1], OpenAIStreamChunk)
        self.assertEqual(
            events[1].json()["choices"][0]["delta"]["tool_calls"][0]["index"],
            0,
        )
        self.assertIn("data:", events[1].raw)
        self.assertTrue(events[2].done)
        self.assertTrue(handle.last_openai[1]["stream"])

    async def test_responses_stream_preserves_named_events(self) -> None:
        handle = FakeHandle()
        events = [
            event
            async for event in Node(handle).inference.stream_responses({
                "model": "model-a",
                "input": "weather?",
                "tools": [{"type": "function", "name": "weather"}],
            })
        ]

        self.assertEqual(handle.last_openai[0], "/v1/responses")
        self.assertEqual(events[1].event, "response.function_call_arguments.delta")

    async def test_stream_failure_raises_typed_error_with_http_context(self) -> None:
        stream = Node(FakeHandle()).inference.stream_chat_completions({
            "model": "missing",
            "messages": [],
        })

        with self.assertRaises(OpenAIRequestError) as raised:
            await anext(stream)

        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.body, '{"error":"missing"}')

    async def test_closing_agent_stream_cancels_native_request(self) -> None:
        handle = FakeHandle()
        stream = Node(handle).inference.stream_chat_completions({
            "model": "model-a",
            "messages": [{"role": "user", "content": "weather?"}],
        })

        first = await anext(stream)
        self.assertIsInstance(first, OpenAIStreamStarted)
        await stream.aclose()

        self.assertEqual(handle.cancelled, ["stream-1"])

    async def test_cancelling_during_stream_startup_cancels_native_request(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class SlowHandle(FakeHandle):
            def openai_stream(self, path: str, body_json: str, listener: object) -> str:
                started.set()
                release.wait(timeout=2)
                return "slow-stream"

        handle = SlowHandle()
        stream = Node(handle).inference.stream_chat_completions({"model": "model-a"})
        task = asyncio.create_task(anext(stream))
        await asyncio.to_thread(started.wait, 2)

        task.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(handle.cancelled, ["slow-stream"])

    async def test_stream_queue_overflow_fails_and_cancels_native_request(self) -> None:
        class FastHandle(FakeHandle):
            def openai_stream(self, path: str, body_json: str, listener: object) -> str:
                for _ in range(300):
                    listener.on_event(FakeOpenAIStreamEvent("sse", data="{}", raw="data: {}\n\n"))
                return "fast-stream"

        handle = FastHandle()
        stream = Node(handle).inference.stream_chat_completions({"model": "model-a"})

        with self.assertRaisesRegex(RuntimeError, "consumer fell behind"):
            await anext(stream)

        self.assertEqual(handle.cancelled, ["fast-stream"])


if __name__ == "__main__":
    unittest.main()
