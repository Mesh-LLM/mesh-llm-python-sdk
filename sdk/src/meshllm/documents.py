"""Options for lossless document OCR inference through the mesh."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentOcrOptions:
    model: str
    filename: str = "document"
    content_type: str = "application/octet-stream"
    response_format: str | None = None
    max_pages: int | None = None

    def _wire(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}
