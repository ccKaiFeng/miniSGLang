from __future__ import annotations

# 这个文件实现 radix prefix cache。
#
# radix cache 的目的：让多个请求共享相同 prompt 前缀的 KV cache。
# 例如很多请求都有同一个 system prompt，那么这个前缀只需要 prefill 一次，
# 后续请求可以直接复用已经写好的 KV cache。
#
# 数据结构上，它用一棵 radix tree：
# - tree 的 key 是 token id 前缀；
# - tree 的 value 是这些 token 对应的 KV cache 物理位置 indices；
# - 节点可以被 split，表示两个请求前缀只有一部分相同；
# - ref_count > 0 的节点正在被请求使用，不能被驱逐；
# - ref_count == 0 的叶子节点可以按 LRU timestamp 驱逐。

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    """radix tree 的一个节点。

    每个节点保存一段连续 token key 和对应的 KV cache value。
    从 root 走到某个节点，沿途所有 key 拼起来就是一个完整前缀。
    """

    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        """创建一个节点。

        key_fn 用来从 token 序列中提取 children 字典的索引 key。
        tic 是时间戳，用于 LRU 驱逐；不传时使用当前单调时钟。
        """

        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int
        self.value_kind: str = "fp16"
        self.compressed_id: int | None = None

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """设置当前节点保存的 token key 和 KV cache indices。"""

        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)
        self.value_kind = "fp16"
        self.compressed_id = None

    def mark_compressed(self, compressed_id: int) -> None:
        """标记当前节点的真实 KV 已经从 fp16 page demote 到 compressed pool。"""

        self.value_kind = "compressed"
        self.compressed_id = compressed_id

    def mark_restored(self, value: torch.Tensor) -> None:
        """把 compressed 节点重新物化成 normal fp16 page。"""

        assert len(value) == self.length
        self._value = value
        self.value_kind = "fp16"
        self.compressed_id = None

    @property
    def is_compressed(self) -> bool:
        return self.value_kind == "compressed"

    def set_parent(self, parent: RadixTreeNode) -> None:
        """把当前节点挂到 parent 下面。"""

        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        """当前节点是否是根节点。根节点不保存真实 key/value。"""

        return self._parent is None

    def is_leaf(self) -> bool:
        """当前节点是否没有子节点。只有未被引用的叶子节点可以直接驱逐。"""

        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """计算当前节点 key 与输入 token 序列的共同前缀长度。"""

        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """在 pos 位置拆分节点。

        原节点 key = A + B。拆分后：
        - 新节点保存 A，挂在原 parent 下；
        - 原节点保存 B，挂在新节点下；
        - 返回新节点。

        这用于处理“新请求和已有节点只匹配一部分”的情况。
        """

        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        """让 heapq 可以按 timestamp 比较节点，实现 LRU 驱逐。"""

        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    """radix cache 的命中句柄。

    cached_len 表示命中的 token 数；node 表示命中路径最后落在哪个树节点。
    """

    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        """从命中节点一路回溯到 root，拼出完整命中前缀的 KV indices。"""

        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    """基于 radix tree 的 prefix cache 实现。"""

    def __init__(self, device: torch.device):
        """初始化 radix tree 和容量统计。"""

        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """锁定或解锁一条命中路径。

        lock 时沿 handle.node 到 root 的路径增加 ref_count，并把这些节点从
        evictable_size 转到 protected_size。

        unlock 时反向减少 ref_count；当某节点 ref_count 变成 0，就重新允许驱逐。
        """

        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0 and not node.is_compressed:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0 and not node.is_compressed:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """查询 input_ids 在 radix tree 中能命中多长前缀。"""

        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """把 input_ids/indices 插入 radix tree。

        只插入 page_size 对齐的长度，因为 KV cache 以 page 为单位管理。
        如果前缀已经存在，只返回已有前缀长度；如果有新增部分，就创建新节点。
        """

        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        """驱逐至少 size 个可驱逐 token 的 KV indices。

        驱逐策略：
        - 只能驱逐 ref_count == 0 的叶子节点；
        - 使用 timestamp 小的节点优先，近似 LRU；
        - 驱逐一个叶子后，如果父节点也变成可驱逐叶子，就继续加入候选堆。
        """

        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            assert not node.is_compressed
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_compressed:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        """当前 radix cache 暂未实现 reset。"""

        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        """返回可驱逐和受保护的 token 数。"""

        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        """当前未实现完整一致性检查。"""

        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """收集所有可驱逐叶子节点，供 evict() 建堆使用。"""

        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0 and not node.is_compressed:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """从 root 开始沿 radix tree 匹配 input_ids。

        返回：
        - node：最终匹配到的节点；
        - prefix_len：已经命中的 token 数。

        如果发现只匹配到某个节点的一部分，会调用 split_at() 拆分节点。
        """

        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                if node.is_compressed:
                    # v2 初版不切分 compressed entry。否则需要同步切分 compressed
                    # pool handle，复杂且容易让 radix node 指向错误 KV。
                    return node.parent, prefix_len - match_len
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len

    def path_nodes(self, handle: RadixCacheHandle) -> List[RadixTreeNode]:
        """返回 handle 从 root 到命中节点的所有非 root 节点。"""

        node = handle.node
        nodes: List[RadixTreeNode] = []
        while not node.is_root():
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def mark_node_compressed(self, node: RadixTreeNode, compressed_id: int) -> None:
        """把一个 normal radix node 标记为 compressed，并更新 normal page 统计。"""

        if node.is_compressed:
            return
        assert node.ref_count == 0, "Only unlocked radix nodes can be compressed"
        node.mark_compressed(compressed_id)
        self.evictable_size -= node.length
        assert self.evictable_size >= 0

    def mark_node_restored(self, node: RadixTreeNode, indices: torch.Tensor) -> None:
        """把 compressed node 恢复为 normal node，并更新 normal page 统计。"""

        if not node.is_compressed:
            node.mark_restored(indices)
            return
        node.mark_restored(indices)
        if node.ref_count == 0:
            self.evictable_size += node.length
        else:
            self.protected_size += node.length


def _get_key_fn(page_size: int) -> KEY_FN:
    """生成 children 字典使用的 key 函数。

    page_size=1 时用单个 token 作为 key；page_size>1 时用一个 page 的 token
    tuple 作为 key，保证 radix tree 以 page 粒度组织。
    """

    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
