from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    image: Any | None
    resize_to: int
    max_new_tokens: int
    seed: int
    think: bool


@dataclass(frozen=True)
class GenerationResult:
    full_output: str


class CaptionBackend(ABC):
    """Format-independent lifecycle used by the ComfyUI node layer."""

    @abstractmethod
    def load(self, model_path: Path) -> Any:
        """Load or retrieve a backend-specific model handle."""

    @abstractmethod
    def generate(self, loaded_model: Any, request: GenerationRequest) -> GenerationResult:
        """Generate text using an already loaded backend-specific model handle."""

    @abstractmethod
    def unload(self, loaded_model: Any) -> None:
        """Remove a loaded model handle from this backend's cache."""

