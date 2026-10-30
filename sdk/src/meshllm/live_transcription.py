"""Live audio sessions with the same turn/event contract as the Rust SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LiveTurnDetection:
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


@dataclass(frozen=True, slots=True)
class LiveTranscriptionOptions:
    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    sample_rate: int = 16000
    turn_detection: LiveTurnDetection | None = field(default_factory=LiveTurnDetection)

    def _wire(self) -> dict[str, Any]:
        # Explicit null disables automatic turn detection (push to talk).
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LiveTranscriptionEvent:
    type: str
    item_id: str | None = None
    text: str | None = None
    transcript: str | None = None
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    message: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def _decode(cls, value: Mapping[str, Any]) -> LiveTranscriptionEvent:
        return cls(
            type=value["type"], item_id=value.get("item_id"), text=value.get("text"),
            transcript=value.get("transcript"), audio_start_ms=value.get("audio_start_ms"),
            audio_end_ms=value.get("audio_end_ms"), message=value.get("message"), raw=value,
        )


class LiveTranscription:
    """Send mono PCM samples and read confirmed transcript events concurrently.

    Call ``stop`` to finish the current turn and collect remaining events, or
    ``close`` to cancel. An async context manager closes the session on exit.
    """

    def __init__(self, handle: object) -> None:
        self._handle = handle

    async def send_audio(self, samples: Sequence[int]) -> None:
        await asyncio.to_thread(self._handle.send_audio, list(samples))

    async def send_audio_f32(self, samples: Sequence[float]) -> None:
        await asyncio.to_thread(self._handle.send_audio_f32, list(samples))

    async def commit(self) -> None:
        await asyncio.to_thread(self._handle.commit)

    async def next_event(self) -> LiveTranscriptionEvent | None:
        try:
            payload = await asyncio.to_thread(self._handle.next_event_json)
        except asyncio.CancelledError:
            await self.close()
            raise
        return None if payload is None else LiveTranscriptionEvent._decode(json.loads(payload))

    async def events(self) -> AsyncIterator[LiveTranscriptionEvent]:
        try:
            while (event := await self.next_event()) is not None:
                yield event
        finally:
            await self.close()

    async def stop(self) -> tuple[LiveTranscriptionEvent, ...]:
        payload = await asyncio.to_thread(self._handle.stop_json)
        return tuple(LiveTranscriptionEvent._decode(item) for item in json.loads(payload))

    async def close(self) -> None:
        await asyncio.to_thread(self._handle.close_audio)

    async def __aenter__(self) -> LiveTranscription:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
