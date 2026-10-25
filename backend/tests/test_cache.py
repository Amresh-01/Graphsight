"""Unit tests for LRUCache."""

from cache import LRUCache


def test_lru_cache_eviction():
    cache: LRUCache[str, int] = LRUCache(capacity=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2

    # Access "a" so "b" becomes the least recently used
    assert cache.get("a") == 1

    # Insert "c", should evict "b"
    cache.set("c", 3)
    assert len(cache) == 2
    assert "b" not in cache
    assert cache.get("a") == 1
    assert cache.get("c") == 3
