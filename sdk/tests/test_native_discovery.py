"""Real embedded-node smoke; requires the matching staged native bridge."""

from __future__ import annotations

import asyncio
import json
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
