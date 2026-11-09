"""Real embedded-node smoke; requires the matching staged native bridge."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import socket
import sys
import unittest
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from meshllm import Node


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def http_json(url: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


class NativeDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_catalog_preserves_nonempty_provider_output_modalities(self) -> None:
        join_file = os.environ.get("MESH_PYTHON_DISCOVERY_JOIN_FILE")
        model_id = os.environ.get("MESH_PYTHON_DISCOVERY_MODEL")
        if join_file is None and model_id is None:
            self.skipTest("set private fixture join-file and expected model for output qualification")
        self.assertTrue(join_file and model_id, "both discovery fixture inputs are required")
        expected = json.loads(os.environ.get(
            "MESH_PYTHON_DISCOVERY_OUTPUT_MODALITIES", '["text","image","future"]',
        ))
        self.assertIsInstance(expected, list)
        self.assertTrue(all(isinstance(value, str) and value for value in expected))
        self.assertTrue({"text", "image"}.issubset(expected), "fixture must advertise text and image output")
        path = pathlib.Path(join_file)
        self.assertTrue(path.is_absolute() and path.is_file(), "join-file must be an existing absolute path")
        token = path.read_text().strip()
        self.assertTrue(token, "private fixture invite is empty")
        api_port, console_port = free_port(), free_port()
        while console_port == api_port:
            console_port = free_port()
        node = Node.create(mode="client", join_tokens=(token,),
                           api_port=api_port, console_port=console_port)
        try:
            await asyncio.wait_for(node.start(), timeout=45)
            deadline = asyncio.get_running_loop().time() + 30
            found = None
            while asyncio.get_running_loop().time() < deadline:
                page = await asyncio.wait_for(node.models.list(), timeout=10)
                found = next((model for model in page.data if model.id == model_id), None)
                if found is not None:
                    break
                await asyncio.sleep(0.25)
            self.assertIsNotNone(found, "expected controlled provider was not discovered")
            self.assertTrue(any(
                not provider.local and provider.metadata is not None
                and isinstance(provider.metadata.get("output_modalities"), list)
                and set(provider.metadata["output_modalities"]) == set(expected)
                for provider in found.providers
            ), "expected remote provider output modalities were not preserved")
            looked_up = await asyncio.wait_for(node.models.get(model_id), timeout=10)
            self.assertEqual(looked_up, found)
            payload = await asyncio.to_thread(http_json, f"http://127.0.0.1:{api_port}/mesh/v1/models")
            raw = next(item for item in payload["data"] if item["id"] == model_id)
            providers = {item["provider_id"]: item for item in raw["providers"]}
            for provider in found.providers:
                self.assertEqual(provider.metadata, providers[provider.provider_id].get("metadata"))
        finally:
            await asyncio.wait_for(node.stop(), timeout=20)

    async def test_native_catalog_matches_http_and_compatibility_shapes(self) -> None:
        api_port, console_port = free_port(), free_port()
        while console_port == api_port:
            console_port = free_port()
        node = Node.create(mode="client", api_port=api_port, console_port=console_port)
        try:
            await node.start()
            native_page = await node.models.list()
            mesh_page = await asyncio.to_thread(http_json, f"http://127.0.0.1:{api_port}/mesh/v1/models")
            self.assertEqual([model.id for model in native_page.data], [row["id"] for row in mesh_page["data"]])
            self.assertEqual(native_page.next_cursor, mesh_page["next_cursor"])
            url = f"http://127.0.0.1:{api_port}/v1/models"
            openai = await asyncio.to_thread(http_json, url)
            self.assertEqual(openai["object"], "list")
            anthropic = await asyncio.to_thread(http_json, url, {"anthropic-version": "2023-06-01"})
            self.assertIn("has_more", anthropic)
            self.assertNotIn("object", anthropic)
            with self.assertRaises(Exception) as caught:
                await node.models.get("missing-native-smoke-model")
            self.assertIn("model not found", str(caught.exception))
        finally:
            await node.stop()


if __name__ == "__main__":
    unittest.main()
