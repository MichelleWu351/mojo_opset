from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import sals_sfa_impl
from mojo_opset.core import MojoSALSSFA


class TTXSALSSFA(MojoSALSSFA):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scales: Optional[torch.Tensor],
        v_scales: Optional[torch.Tensor],
        block_tables: torch.Tensor,
        indices_flat: torch.Tensor,
        seq_len_flat: torch.Tensor,
        group_qid: torch.Tensor,
        group_q_start: torch.Tensor,
        group_q_len: torch.Tensor,
        cumsum_q_len: torch.Tensor,
        base_kv_len: torch.Tensor,
        group_use_dense: Optional[torch.Tensor],
        softmax_scale: float,
        num_kv_heads: int,
        num_query_heads: int,
        head_dim: int,
        sparse_block_size: int,
    ) -> torch.Tensor:
        # FIX (Bug q=64k indexer DDR MTE): 强制 block_tables 为 int32，与单算子测试统一，
        # 避免端到端 int64 block_tables 触发 BiSheng codegen 在长 q + cbs=512 时越界。
        if block_tables.dtype != torch.int32:
            block_tables = block_tables.to(torch.int32)
        return sals_sfa_impl(
            q, k_cache, v_cache, k_scales, v_scales,
            block_tables, indices_flat, seq_len_flat,
            group_qid, group_q_start, group_q_len,
            cumsum_q_len, base_kv_len, group_use_dense,
            softmax_scale,
            num_kv_heads, num_query_heads, head_dim, sparse_block_size,
        )
