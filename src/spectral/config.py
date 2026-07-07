"""Config via dataclasses + YAML.

Why not Hydra/OmegaConf? The Working Agreement wants minimal deps and code Tanmay can
read and defend. A dataclass gives one obvious place where every parameter lives, with
type hints and defaults; the YAML file holds the actual values so nothing is hardcoded.

`YamlConfig` is a small mixin: subclass it, declare fields (nested dataclasses allowed),
and call `.from_yaml(path)`. The loader constructs nested dataclasses recursively and
errors on unknown keys so typos in a config file don't silently do nothing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T", bound="YamlConfig")


class YamlConfig:
    """Mixin adding YAML (de)serialization to a dataclass."""

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Build the dataclass from a plain dict, recursing into nested dataclasses."""
        # get_type_hints resolves string annotations (from `from __future__ import
        # annotations`) back into real types, so we can detect nested dataclass fields.
        hints = get_type_hints(cls)
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"{cls.__name__}: unknown config key(s) {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}."
            )

        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            ftype = hints[name]
            if is_dataclass(ftype) and isinstance(value, dict):
                kwargs[name] = ftype.from_dict(value)  # type: ignore[attr-defined]
            else:
                kwargs[name] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        """Load a YAML file and construct the config."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (nested dataclasses included), handy for logging."""
        return dataclasses.asdict(self)  # type: ignore[call-overload]


@dataclass
class RunConfig(YamlConfig):
    """Top-level run settings shared across every entry point.

    Phase 1 will add a `DataConfig` field here; for now it just carries the seed so that
    the seeding/reproducibility path is exercised end to end.
    """

    seed: int = 0
    deterministic: bool = True
