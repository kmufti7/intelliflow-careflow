"""
Chaos Mode — Deterministic failure injection for CareFlow resilience testing.

Simulates infrastructure failures (FAISS unavailable, Pinecone unavailable)
with controlled blast radius. Failures are toggled, not random — making demos
reproducible and tests deterministic.

In production, you'd use percentage-based injection with kill switches.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ChaosFailureType(str, Enum):
    """Types of failures that can be injected."""
    FAISS_UNAVAILABLE = "faiss_unavailable"
    PINECONE_UNAVAILABLE = "pinecone_unavailable"


@dataclass
class ChaosConfig:
    """Configuration for chaos mode."""
    enabled: bool = False
    failure_type: ChaosFailureType = ChaosFailureType.FAISS_UNAVAILABLE

    def is_faiss_failure(self) -> bool:
        return self.enabled and self.failure_type == ChaosFailureType.FAISS_UNAVAILABLE

    def is_pinecone_failure(self) -> bool:
        return self.enabled and self.failure_type == ChaosFailureType.PINECONE_UNAVAILABLE


class ChaosError(Exception):
    """Raised when chaos mode injects a simulated failure."""

    def __init__(self, failure_type: ChaosFailureType, message: str):
        self.failure_type = failure_type
        super().__init__(message)


# Module-level config (singleton)
_chaos_config = ChaosConfig()


def get_chaos_config() -> ChaosConfig:
    """Get the global chaos configuration."""
    return _chaos_config


def set_chaos_config(enabled: bool, failure_type: ChaosFailureType = ChaosFailureType.FAISS_UNAVAILABLE) -> ChaosConfig:
    """Update the global chaos configuration."""
    global _chaos_config
    _chaos_config = ChaosConfig(enabled=enabled, failure_type=failure_type)
    return _chaos_config


def check_faiss_chaos() -> None:
    """Check if FAISS chaos failure should be triggered. Raises ChaosError if so."""
    config = get_chaos_config()
    if config.is_faiss_failure():
        raise ChaosError(
            ChaosFailureType.FAISS_UNAVAILABLE,
            "FAISS index unavailable (simulated chaos failure)"
        )


def check_pinecone_chaos() -> None:
    """Check if Pinecone chaos failure should be triggered. Raises ChaosError if so."""
    config = get_chaos_config()
    if config.is_pinecone_failure():
        raise ChaosError(
            ChaosFailureType.PINECONE_UNAVAILABLE,
            "Pinecone service unavailable (simulated chaos failure)"
        )


FALLBACK_RESPONSE = (
    "The system is currently experiencing a service disruption and cannot complete "
    "your request. Our clinical data retrieval service is temporarily unavailable. "
    "Please try again shortly. If the issue persists, contact your system administrator. "
    "No clinical decisions should be made based on this message."
)
