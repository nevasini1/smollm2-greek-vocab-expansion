"""Tiny plugin registry.

Every swappable piece of the pipeline (data sources, tokenizer extenders, embedding
initializers, eval sets, downstream tasks) lives in a ``Registry``. Adding a method
is: write a function/class, decorate it with ``@REGISTRY.register("name")``, and
reference ``"name"`` from a YAML config. Modules listed under ``plugins:`` in a
config are imported before the run starts, so new methods can live outside this
package entirely.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Iterable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            key = name or getattr(obj, "__name__", None)
            if not key:
                raise ValueError(f"{self.kind}: cannot infer a name for {obj!r}")
            if key in self._items and self._items[key] is not obj:
                raise KeyError(f"{self.kind} '{key}' is already registered")
            self._items[key] = obj
            return obj

        return deco

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry({self.kind}: {self.names()})"


def import_plugins(modules: Iterable[str]) -> None:
    """Import user plugin modules (dotted paths or ``path/to/file.py``)."""
    import importlib.util
    import os
    import sys

    import hashlib

    for mod in modules:
        if mod.endswith(".py") or os.path.sep in mod:
            path = os.path.abspath(mod)
            # stable name per file: importing the same plugin again (every run of a plan) is a no-op
            name = "vocabx_plugin_" + hashlib.sha1(path.encode()).hexdigest()[:10]
            if name in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load plugin file {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(mod)
