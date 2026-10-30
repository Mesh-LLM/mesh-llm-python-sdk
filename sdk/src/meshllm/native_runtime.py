"""Explicit native-runtime installation and cache management."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Callable

from ._binding import native


@dataclass(frozen=True, slots=True)
class NativeRuntimeInstallOptions:
    mesh_version: str | None = None
    skippy_abi_version: str | None = None
    selection: str = "recommended"
    manifest_path: str | None = None
    manifest_url: str | None = None
    bundle_dirs: tuple[str, ...] = ()
    cache_dir: str | None = None
    verification_policy: str = "require_checksum"
    allow_download: bool = True


@dataclass(frozen=True, slots=True)
class InstalledNativeRuntime:
    mesh_version: str
    native_runtime_id: str
    flavor: str
    path: str
    skippy_abi_version: str | None

    @classmethod
    def _decode(cls, value: object) -> InstalledNativeRuntime:
        return cls(value.mesh_version, value.native_runtime_id, value.flavor,
                   value.path, value.skippy_abi_version)


@dataclass(frozen=True, slots=True)
class NativeRuntimeInstallOutcome:
    status: str
    runtime: InstalledNativeRuntime
    selected_native_runtime_id: str
    selected_source: str


@dataclass(frozen=True, slots=True)
class NativeRuntimeDownloadProgress:
    native_runtime_id: str
    url: str
    downloaded_bytes: int
    total_bytes: int | None
    finished: bool


@dataclass(frozen=True, slots=True)
class NativeRuntimePruneResult:
    removed_dirs: tuple[str, ...]


def current_mesh_version() -> str:
    return native().current_mesh_version()


def current_skippy_abi_version() -> str:
    return native().current_skippy_abi_version()


async def install_native_runtime(
    options: NativeRuntimeInstallOptions | None = None,
    *, on_progress: Callable[[NativeRuntimeDownloadProgress], None] | None = None,
) -> NativeRuntimeInstallOutcome:
    options = options if options is not None else NativeRuntimeInstallOptions()
    binding = native()
    policies = {
        "require_checksum": binding.NativeRuntimeVerificationPolicyNative.REQUIRE_CHECKSUM,
        "require_checksum_and_signature": binding.NativeRuntimeVerificationPolicyNative.REQUIRE_CHECKSUM_AND_SIGNATURE,
    }
    if options.verification_policy not in policies:
        raise ValueError("unknown native runtime verification policy")
    args = binding.NativeRuntimeInstallOptionsNative(
        mesh_version=options.mesh_version, skippy_abi_version=options.skippy_abi_version,
        selection=options.selection, manifest_path=options.manifest_path,
        manifest_url=options.manifest_url, bundle_dirs=list(options.bundle_dirs),
        cache_dir=options.cache_dir, verification_policy=policies[options.verification_policy],
        allow_download=options.allow_download,
    )
    loop = asyncio.get_running_loop()

    class Progress:
        def on_progress(self, event: object) -> None:
            update = NativeRuntimeDownloadProgress(
                event.native_runtime_id, event.url, event.downloaded_bytes,
                event.total_bytes, event.finished,
            )
            if on_progress is not None:
                loop.call_soon_threadsafe(on_progress, update)

    result = await asyncio.to_thread(
        binding.install_native_runtime, args, Progress() if on_progress is not None else None,
    )
    return NativeRuntimeInstallOutcome(
        result.status, InstalledNativeRuntime._decode(result.runtime),
        result.selected_native_runtime_id, result.selected_source,
    )


async def installed_native_runtimes(cache_dir: str | None = None) -> list[InstalledNativeRuntime]:
    result = await asyncio.to_thread(native().installed_native_runtimes, cache_dir)
    return [InstalledNativeRuntime._decode(item) for item in result]


async def resolve_native_runtime(
    options: NativeRuntimeInstallOptions | None = None,
    *, allow_download: bool = False,
) -> InstalledNativeRuntime:
    """Resolve from explicit inputs; downloads require explicit opt-in."""
    options = options if options is not None else NativeRuntimeInstallOptions()
    return (await install_native_runtime(replace(options, allow_download=allow_download))).runtime


async def validate_native_runtime(
    artifact_dir: str, options: NativeRuntimeInstallOptions | None = None,
) -> InstalledNativeRuntime:
    """Validate an explicit bundle without permitting downloads."""
    options = options if options is not None else NativeRuntimeInstallOptions()
    return (await install_native_runtime(replace(
        options, bundle_dirs=(artifact_dir,), allow_download=False,
    ))).runtime


async def remove_native_runtime(
    mesh_version: str, native_runtime_id: str, *, cache_dir: str | None = None,
) -> bool:
    return await asyncio.to_thread(
        native().remove_native_runtime, cache_dir, mesh_version, native_runtime_id,
    )


async def prune_native_runtimes(
    *, cache_dir: str | None = None, active_mesh_version: str | None = None,
    mode: str = "keep_active_and_previous",
) -> NativeRuntimePruneResult:
    binding = native()
    modes = {
        "keep_active_and_previous": binding.NativeRuntimePruneModeNative.KEEP_ACTIVE_AND_PREVIOUS,
        "active_only": binding.NativeRuntimePruneModeNative.ACTIVE_ONLY,
    }
    if mode not in modes:
        raise ValueError("unknown native runtime prune mode")
    result = await asyncio.to_thread(
        binding.prune_native_runtimes, cache_dir, active_mesh_version, modes[mode],
    )
    return NativeRuntimePruneResult(tuple(result.removed_dirs))
