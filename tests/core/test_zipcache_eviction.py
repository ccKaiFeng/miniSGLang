"""CPU regressions for mixed radix eviction and compressed-pool exhaustion.

Only the CUDA key-comparison primitive is replaced by a CPU comparison. Real
radix nodes, quantization, allocators, and manager entry ownership are exercised.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import minisgl.core as core
import minisgl.distributed.info as distributed_info
import pytest
import torch
from minisgl.kvcache.radix_cache import RadixCacheHandle, RadixPrefixCache, RadixTreeNode
from minisgl.zipcache.manager import ZipCacheV3Manager

# Load the CPU cache module without importing scheduler/__init__.py, which
# eagerly loads the complete CUDA engine. The module's implementation is unchanged.
SPEC = importlib.util.spec_from_file_location(
    "cache_under_test",
    Path(__file__).resolve().parents[2] / "python/minisgl/scheduler/cache.py",
)
cache_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache_module)


@pytest.fixture(autouse=True)
def context(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))
    monkeypatch.setattr(distributed_info, "_TP_INFO", distributed_info.DistributedInfo(0, 1))

    def compare(node, ids):
        n = min(len(ids), node.length)
        different = (node._key[:n] != ids[:n]).nonzero()
        return int(different[0]) if different.numel() else n

    monkeypatch.setattr(RadixTreeNode, "get_match_len", compare)


def node(cache, parent, keys, indices):
    result = RadixTreeNode(cache.key_fn)
    result.set_key_value(torch.tensor(keys), torch.tensor(indices, dtype=torch.int32))
    result.set_parent(parent)
    cache.evictable_size += len(keys)
    return result


def test_normal_parent_behind_compressed_child():
    cache = RadixPrefixCache(torch.device("cpu"))
    released = []
    cache.free_compressed_entry = released.append
    parent = node(cache, cache.root_node, [1, 2], [10, 11])
    child = node(cache, parent, [3, 4], [12, 13])
    cache.mark_node_compressed(child, 99)
    assert cache.evictable_size == 2
    assert cache.evict(1).tolist() == [10, 11]
    assert released == [99]
    assert not cache.root_node.children
    assert cache.evictable_size == 0


@pytest.mark.parametrize("page_size", [1, 4])
def test_page_allocation_reclaims_parent_without_reusing_compressed_indices(page_size):
    core._GLOBAL_CTX = core.Context(page_size=page_size)
    released = []
    cm = cache_module.CacheManager(
        4,
        page_size,
        torch.empty(1),
        "radix",
        zipcache_manager=SimpleNamespace(_free_entry=released.append),
    )
    indices = cm._page_to_token(cm._allocate(4))
    parent = node(
        cm.prefix_cache,
        cm.prefix_cache.root_node,
        list(range(2 * page_size)),
        indices[: 2 * page_size].tolist(),
    )
    child = node(
        cm.prefix_cache,
        parent,
        list(range(2 * page_size, 4 * page_size)),
        indices[2 * page_size :].tolist(),
    )
    cm.prefix_cache.mark_node_compressed(child, 99)
    cm._free(indices[2 * page_size :])
    active = cm._allocate(2)
    allocated = cm._allocate(1)
    assert released == [99]
    all_pages = torch.cat([active, allocated, cm.free_slots])
    assert sorted(all_pages.tolist()) == list(range(0, 4 * page_size, page_size))
    assert cm.prefix_cache.evictable_size == 0


def test_locked_compressed_leaf_survives_pressure():
    cache = RadixPrefixCache(torch.device("cpu"))
    released = []
    cache.free_compressed_entry = released.append
    locked = node(cache, cache.root_node, [1], [0])
    cache.mark_node_compressed(locked, 11)
    handle = RadixCacheHandle(1, locked)
    cache.lock_handle(handle)
    other = node(cache, cache.root_node, [2], [1])
    cache.mark_node_compressed(other, 12)
    assert cache.evict_compressed_leaf(set())
    assert released == [12]
    assert not cache.evict_compressed_leaf(set())
    cache.lock_handle(handle, unlock=True)
    assert not cache.evict_compressed_leaf({locked.uuid})
    assert cache.evict_compressed_leaf(set())
    assert released == [12, 11]


def test_partial_compressed_insert_keeps_old_subtree():
    cache = RadixPrefixCache(torch.device("cpu"))
    parent = node(cache, cache.root_node, [1], [0])
    child = node(cache, parent, [2, 3], [1, 2])
    cache.mark_node_compressed(child, 99)
    tail = node(cache, child, [4], [3])
    inserted = cache.insert_prefix(torch.tensor([1, 2, 9]), torch.tensor([4, 5, 6]))
    assert inserted.cached_len == inserted.handle.cached_len == 1
    assert parent.children[2] is child
    assert child.children[4] is tail
    assert cache.evictable_size == 2


class Pool:
    device = torch.device("cpu")
    dtype = torch.float16
    num_layers = 2

    def __init__(self):
        self.k = torch.randn(2, 512, 1, 2, 32, dtype=self.dtype)
        self.v = torch.randn_like(self.k)

    def k_cache(self, layer):
        return self.k[layer]

    def v_cache(self, layer):
        return self.v[layer]


def manager_and_cache():
    config = SimpleNamespace(
        enable_zipcache_v3=True,
        zipcache_v3_demote_on_finish=True,
        zipcache_v3_compressed_pool_mb=1,
        zipcache_unimportant_ratio=0.4,
        zipcache_protect_recent_tokens=1,
        zipcache_k_important_bit=4,
        zipcache_k_unimportant_bit=2,
        zipcache_v_important_bit=4,
        zipcache_v_unimportant_bit=2,
        # Small Q4 partition forces pressure with just a few nodes.
        zipcache_v3_q4_pool_ratio=0.02,
        zipcache_v3_q2_pool_ratio=0.3,
        zipcache_v3_scale_pool_ratio=0.4,
        zipcache_v3_ids_pool_ratio=0.28,
    )
    manager = ZipCacheV3Manager(config, Pool(), torch.empty(1))
    cache = RadixPrefixCache(torch.device("cpu"))
    cache.free_compressed_entry = manager._free_entry
    return manager, cache


def test_repeated_pool_pressure_reclaims_entries_and_restores():
    manager, cache = manager_and_cache()
    for iteration in range(12):
        current = node(
            cache,
            cache.root_node,
            list(range(iteration * 1000, iteration * 1000 + 128)),
            list(range(128)),
        )
        original = manager.kv_pool.k.clone()
        indices = manager.demote_node(current, prefix_cache=cache, excluded={current.uuid})
        assert indices is not None
        cache.mark_node_compressed(current, manager.entry_by_node_uuid[current.uuid])
        entry = manager.entries[current.compressed_id]
        manager._restore_entry_to_indices(entry, torch.arange(128, 256))
        assert torch.isfinite(manager.kv_pool.k[:, 128:256]).all()
        torch.testing.assert_close(
            manager.kv_pool.k[:, 128:256], original[:, :128], atol=1.5, rtol=0
        )
        assert cache.evictable_size == 0
        assert manager.pool.stats()["compressed_pool_used_bytes"] == sum(
            e.storage_bytes for e in manager.entries.values()
        )
    assert manager.stats()["num_compressed_freed"] > 0
    while cache.evict_compressed_leaf(set()):
        pass
    assert not manager.entries and not manager.entry_by_node_uuid
    assert manager.pool.stats()["compressed_pool_used_bytes"] == 0


def test_oversized_demote_rolls_back_all_slices():
    manager, cache = manager_and_cache()
    current = node(cache, cache.root_node, list(range(512)), list(range(512)))
    assert manager.demote_node(current, prefix_cache=cache) is None
    assert not current.is_compressed
    assert cache.evictable_size == 512
    assert manager.pool.stats()["compressed_pool_used_bytes"] == 0
    assert not manager.entries


def test_finished_requests_and_temporary_restore_do_not_leak_pages():
    manager, _ = manager_and_cache()
    table = torch.empty((1, 512), dtype=torch.int32)
    cm = cache_module.CacheManager(512, 1, table, "radix", zipcache_manager=manager)
    for iteration in range(12):
        ids = torch.arange(iteration * 1000, iteration * 1000 + 128)
        table[0, :128] = cm.allocate_token_indices(128)
        req = SimpleNamespace(
            input_ids=ids,
            cached_len=128,
            table_idx=0,
            cache_handle=RadixCacheHandle(0, cm.prefix_cache.root_node),
        )
        cm.cache_req(req, finished=True)
        cm.check_integrity()
        assert len(cm.free_slots) == 512
        pending = SimpleNamespace(input_ids=torch.cat([ids, torch.tensor([-1])]), input_len=129)
        handle = cm.match_req(pending).cuda_handle
        assert handle.cached_len == 128
        assert len(cm.free_slots) == 384
        cm.release_handle_resources(handle)
        cm.release_handle_resources(handle)
        cm.check_integrity()
        assert cm.free_slots.unique().numel() == 512
    assert manager.stats()["num_compressed_freed"] > 0
