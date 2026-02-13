"""TwoTierConfig with environment variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Final


_DEFAULT_QUALITY_WEIGHT: Final = 0.7
_DEFAULT_RRF_K: Final = 60.0
_DEFAULT_TIMEOUT_MS: Final = 500.0
_DEFAULT_MULTIPLIER: Final = 3


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, *, default: bool) -> bool:
    raw = os.environ.get(key, "").lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_model_dir() -> Path:
    return Path.home() / ".cache" / "brr" / "models"


@dataclass(slots=True)
class TwoTierConfig:
    """Configuration for the two-tier search pipeline."""

    quality_weight: float = field(
        default_factory=lambda: _env_float(
            "BRR_QUALITY_WEIGHT",
            _DEFAULT_QUALITY_WEIGHT,
        ),
    )
    rrf_k: float = field(
        default_factory=lambda: _env_float("BRR_RRF_K", _DEFAULT_RRF_K),
    )
    candidate_multiplier: int = field(
        default_factory=lambda: _env_int(
            "BRR_CANDIDATE_MULTIPLIER",
            _DEFAULT_MULTIPLIER,
        ),
    )
    quality_timeout_ms: float = field(
        default_factory=lambda: _env_float(
            "BRR_QUALITY_TIMEOUT_MS",
            _DEFAULT_TIMEOUT_MS,
        ),
    )
    fast_only: bool = field(
        default_factory=lambda: _env_bool("BRR_FAST_ONLY", default=False),
    )
    model_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("BRR_MODEL_DIR", str(_default_model_dir())),
        ),
    )
    fast_model: str = field(
        default_factory=lambda: os.environ.get(
            "BRR_FAST_MODEL",
            "potion-multilingual-128M",
        ),
    )
    quality_model: str = field(
        default_factory=lambda: os.environ.get(
            "BRR_QUALITY_MODEL",
            "all-MiniLM-L6-v2",
        ),
    )
