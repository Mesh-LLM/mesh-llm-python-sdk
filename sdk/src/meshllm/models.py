"""Typed discovery of models advertised in the embedded node's mesh."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MeshModelProvider:
    """Observations from one provider; optional facts remain unknown when absent."""

    provider_id: str
    local: bool
    available: bool
    operations: tuple[str, ...]
    endpoint_id: str | None = None
    plugin: str | None = None
    workload_class: str | None = None
    capabilities: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    runtime: Mapping[str, Any] | None = None
    thinking: Any = None
    payment: Any = None

    @classmethod
    def _decode(cls, value: Mapping[str, Any]) -> MeshModelProvider:
        return cls(
            provider_id=value["provider_id"],
            local=value["local"],
            available=value["available"],
            operations=tuple(value["operations"]),
            endpoint_id=value.get("endpoint_id"),
            plugin=value.get("plugin"),
            workload_class=value.get("workload_class"),
            capabilities=value.get("capabilities"),
            metadata=value.get("metadata"),
            runtime=value.get("runtime"),
            thinking=value.get("thinking"),
            payment=value.get("payment"),
        )


@dataclass(frozen=True, slots=True)
class MeshModel:
    """One public serving ID, including distinct observations for its providers.

    Workload, kind and availability strings preserve future values. Operations
    describe what at least one provider advertises, not every replica.
    """

    id: str
    display_name: str
    kind: str
    availability: str
    operations: tuple[str, ...]
    providers: tuple[MeshModelProvider, ...]
    workload_class: str | None = None
    metadata: Mapping[str, Any] | None = None
    virtual_model: Mapping[str, Any] | None = None

    @classmethod
    def _decode(cls, value: Mapping[str, Any]) -> MeshModel:
        return cls(
            id=value["id"],
            display_name=value["display_name"],
            kind=value["kind"],
            availability=value["availability"],
            operations=tuple(value["operations"]),
            providers=tuple(MeshModelProvider._decode(item) for item in value["providers"]),
            workload_class=value.get("workload_class"),
            metadata=value.get("metadata"),
            virtual_model=value.get("virtual_model"),
        )


@dataclass(frozen=True, slots=True)
class MeshModelPage:
    """A bounded catalog page; pass the opaque cursor back to continue."""

    data: tuple[MeshModel, ...]
    next_cursor: str | None


class Models:
    """Discover every advertised workload through the native embedded node.

    The catalog is the node's observed mesh inventory. It includes local and
    remote providers and is independent of the LLM-only compatibility list.
    """

    def __init__(self, handle: object) -> None:
        self._handle = handle

    async def list(
        self,
        *,
        workload_class: str | None = None,
        kind: str | None = None,
        available: bool | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> MeshModelPage:
        """List advertisements; supplied filters combine with AND.

        Native validation and pagination match the Mesh HTTP API. Native errors
        propagate to the caller. No HTTP listener is used for this query.
        """
        query = {
            key: value
            for key, value in {
                "workload_class": workload_class,
                "kind": kind,
                "available": available,
                "limit": limit,
                "cursor": cursor,
            }.items()
            if value is not None
        }
        payload = await asyncio.to_thread(
            self._handle.mesh_models_json,
            json.dumps(query, separators=(",", ":")),
        )
        value = json.loads(payload)
        return MeshModelPage(
            data=tuple(MeshModel._decode(item) for item in value["data"]),
            next_cursor=value["next_cursor"],
        )

    async def get(self, model_id: str) -> MeshModel:
        """Look up an exact serving ID; missing IDs raise a native error."""
        payload = await asyncio.to_thread(self._handle.mesh_model_json, model_id)
        return MeshModel._decode(json.loads(payload))
