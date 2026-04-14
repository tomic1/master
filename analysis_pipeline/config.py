from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_analysis_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("PyYAML is required for YAML config files") from exc
        with path.open("r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
    elif suffix == ".json":
        import json

        with path.open("r", encoding="utf-8") as f:
            parsed = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")

    if not isinstance(parsed, dict):
        raise ValueError("Config root must be a mapping")

    return parsed


def merge_overrides(config: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    if not overrides:
        from copy import deepcopy

        return deepcopy(config)

    from copy import deepcopy

    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    return _deep_merge(config, overrides)
