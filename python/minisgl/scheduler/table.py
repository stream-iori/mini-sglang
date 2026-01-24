import torch


class TableManager:
    """
    TableManager 负责管理正在运行的请求（running requests）的槽位（slots）分配以及相关的数据表。

    它主要维护了以下信息：
    1. 空闲槽位列表 (`_free_slots`)：用于快速分配和回收请求 ID (req_id) 对应的槽位索引。
    2. 页表 (`page_table`)：引用外部传入的张量，通常用于存储请求到物理页面的映射关系 (KV Cache)。
    3. 令牌池 (`token_pool`)：用于存储每个槽位对应的 token IDs，结构与 page_table 类似。

    Attributes:
        _max_running_reqs (int): 系统允许的最大并发运行请求数。
        _free_slots (list[int]): 当前可用的槽位索引列表。
        page_table (torch.Tensor): 外部传入的页表张量，用于存储物理页索引。
        token_pool (torch.Tensor): 内部创建的张量，用于存储请求的 token IDs。
    """

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        """
        初始化 TableManager。

        Args:
            max_running_reqs (int): 最大并发请求数。
            page_table (torch.Tensor): 用于存储页映射的张量 (通常是全局的或者由更高层级管理的)。
        """
        self._max_running_reqs = max_running_reqs

        # 初始化所有槽位为空闲状态，范围是 [0, max_running_reqs - 1]
        self._free_slots = list(range(max_running_reqs))

        # 引用外部的页表
        self.page_table = page_table

        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # 初始化 token_pool，结构与 page_table 相同，用于存储 token IDs。
        # 这里初始化为 0 是为了确保即使是 dummy request 也能访问到合法的 token id。
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        """
        获取当前可用的槽位数量。

        Returns:
            int: 剩余空闲槽位的个数。
        """
        return len(self._free_slots)

    def allocate(self) -> int:
        """
        分配一个空闲槽位。

        从空闲列表中弹出一个槽位索引供新请求使用。
        调用此方法前应确保 available_size > 0，否则会抛出 IndexError。

        Returns:
            int: 分配到的槽位索引。
        """
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """
        释放一个槽位。

        将使用完毕的槽位索引放回空闲列表。

        Args:
            slot (int): 要释放的槽位索引。
        """
        self._free_slots.append(slot)
