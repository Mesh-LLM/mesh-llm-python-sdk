from __future__ import annotations

import pathlib
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from meshllm import _binding


class BindingAdmissionTests(unittest.TestCase):
    def tearDown(self) -> None:
        _binding._set_native_for_testing(None)

    def test_actual_generated_guard_and_version(self) -> None:
        _binding._set_native_for_testing(None)
        module = _binding.native()
        module._uniffi_check_api_checksums(module._UniffiLib)
        version = module.current_mesh_version()
        self.assertTrue(version)
        print("native_mesh_version=" + version)
        self.assertIs(module, _binding.native())

    def test_actual_checksum_mismatch_is_not_cached(self) -> None:
        _binding._set_native_for_testing(None)
        from meshllm._generated import mesh_ffi as generated

        class WrongChecksumLibrary:
            def uniffi_meshllm_ffi_checksum_func_create_node(self) -> int:
                return generated._UniffiLib.uniffi_meshllm_ffi_checksum_func_create_node() ^ 1

            def __getattr__(self, name: str) -> object:
                return getattr(generated._UniffiLib, name)

        bad = SimpleNamespace(
            _UniffiLib=WrongChecksumLibrary(),
            _uniffi_check_api_checksums=generated._uniffi_check_api_checksums,
        )
        with patch.object(_binding, "import_module", side_effect=[bad, generated]) as load:
            with self.assertRaisesRegex(generated.InternalError, "API checksum mismatch"):
                _binding.native()
            self.assertIsNone(_binding._native_module)
            self.assertIs(_binding.native(), generated)
            self.assertEqual(load.call_count, 2)

    def test_actual_invalid_mode_has_typed_error_payload(self) -> None:
        _binding._set_native_for_testing(None)
        module = _binding.native()
        mode = "invalid-sdk-binding-fixture"
        with self.assertRaises(module.FfiError.BuildFailed) as caught:
            module.create_node(mode, [], [], False, None, 0, 0)
        self.assertEqual(caught.exception[0], "unknown node mode: " + mode)

    def test_missing_guard_is_not_cached(self) -> None:
        _binding._set_native_for_testing(None)
        with patch.object(_binding, "import_module", return_value=SimpleNamespace()):
            with self.assertRaises(AttributeError):
                _binding.native()
        self.assertIsNone(_binding._native_module)

    def test_new_native_capabilities_reject_stopped_node_without_http(self) -> None:
        module = _binding.native()
        handle = module.create_node("client", [], [], False, None, 0, 0)
        calls = [
            lambda: handle.mesh_models_json("{}"),
            lambda: handle.mesh_model_json("org/model#profile"),
            lambda: handle.document_ocr(b"%PDF-1.7", '{"model":"ocr"}'),
            lambda: handle.start_live_transcription_json("{}"),
            lambda: handle.memberships_json(),
            lambda: handle.remove_membership("team"),
            lambda: handle.add_membership_json(json.dumps({
                "name": "team", "join_tokens": [], "trusted_owners": [],
                "api_port": 9338, "node_key_path": None, "enumerate_host": False,
            })),
        ]
        for call in calls:
            with self.subTest(method=call):
                with self.assertRaises(module.FfiError.HostUnavailable):
                    call()

    def test_full_configuration_factory_matches_native_validation(self) -> None:
        module = _binding.native()
        node = module.create_node_config_json('{"mode":"client","network":{"auto_join":false}}')
        self.assertFalse(node.status().running)
        with self.assertRaises(module.FfiError.BuildFailed):
            module.create_node_config_json('{"network":{"autoJoin":true}}')
        with self.assertRaises(module.FfiError.BuildFailed):
            module.create_node_config_json('{"http":{"api_port":70000}}')


if __name__ == "__main__":
    unittest.main()
