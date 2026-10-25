"""Bounded LRU (Least-Recently-Used) cache backed by OrderedDict."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Fixed-capacity cache that evicts the least-recently-used entry first."""

    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("LRUCache capacity must be positive")
        self.capacity = capacity
        self._store: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        """Return the value and mark it most-recently-used, or None on miss."""
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def __contains__(self, key: K) -> bool:
        return key in self._store

    def set(self, key: K, value: V) -> None:
        """Insert/overwrite a value, then evict oldest entries past capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
