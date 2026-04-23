from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class AdapterCacheStats:
    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "max_size": self.max_size,
        }


class AdapterBundleCache(Generic[ValueT]):
    def __init__(self, *, max_size: int = 128) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        self._values: OrderedDict[tuple[str, str | None], ValueT] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: tuple[str, str | None]) -> ValueT | None:
        if key not in self._values:
            self._misses += 1
            return None
        self._hits += 1
        value = self._values.pop(key)
        self._values[key] = value
        return value

    def put(self, key: tuple[str, str | None], value: ValueT) -> None:
        if key in self._values:
            self._values.pop(key)
        elif len(self._values) >= self.max_size:
            self._values.popitem(last=False)
        self._values[key] = value

    def clear(self) -> None:
        self._values.clear()

    def stats(self) -> AdapterCacheStats:
        return AdapterCacheStats(
            hits=self._hits,
            misses=self._misses,
            size=len(self._values),
            max_size=self.max_size,
        )
