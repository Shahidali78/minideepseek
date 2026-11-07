"""Freeze hashes only after selecting training settings using validation."""

import json
from pathlib import Path

from .data import digest


def freeze(config, data, adapters):
    return {
        "config_sha256": digest(config),
        "data": {
            s: digest(Path(data) / f"{s}.jsonl") for s in ["train", "val", "test"]
        },
        "adapters": {
            str(Path(a).resolve()): digest(Path(a) / "adapter_model.safetensors")
            for a in adapters
        },
    }


def verify_frozen(path, config, data, adapter):
    frozen = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    current = freeze(config, data, [adapter] if adapter else [])
    if (
        current["config_sha256"] != frozen["config_sha256"]
        or current["data"] != frozen["data"]
    ):
        raise ValueError("protocol changed since freezing")
    if adapter and current["adapters"].items() - frozen["adapters"].items():
        raise ValueError("adapter not frozen")
