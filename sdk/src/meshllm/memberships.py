"""Native controls for serving into additional, isolated meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MeshMembershipConfig:
    name: str
    api_port: int
    join_tokens: tuple[str, ...] = ()
    trusted_owners: tuple[str, ...] = ()
    node_key_path: str | None = None
    enumerate_host: bool = False

    def _wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "api_port": self.api_port,
            "join_tokens": list(self.join_tokens),
            "trusted_owners": list(self.trusted_owners),
            "node_key_path": self.node_key_path,
            "enumerate_host": self.enumerate_host,
        }


@dataclass(frozen=True, slots=True)
class MeshMembershipStatus:
    name: str
    api_base_url: str
    invite_token: str
    endpoint_id: str
    peers: int
    join_error: str | None = None

    @classmethod
    def _decode(cls, value: Mapping[str, Any]) -> MeshMembershipStatus:
        return cls(
            name=value["name"], api_base_url=value["api_base_url"],
            invite_token=value["invite_token"], endpoint_id=value["endpoint_id"],
            peers=value["peers"], join_error=value.get("join_error"),
        )
