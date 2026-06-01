from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ...core.operator import MojoOperator
from ...core.operators.kv_cache import build_paged_kv_chunk_metadata

__all__ = ["MojoFusedNormRoPEQuantStore"]


class MojoFusedNormRoPEQuantStore(MojoOperator):
    """Fused QK-Norm + RoPE + KV-StaticQuant + PagedKVStore.

    Combines GroupRMSNorm (on Q and K), RoPE (on Q and K), StaticQuant (on K
    and V), and paged KV cache store into a single operator to eliminate
    intermediate tensor materializations.

    When ``update_kv=False``, only Q norm+rope is computed — the K/V quant and
    store paths are skipped entirely (useful for YOCO reuse layers).
    """

    def __init__(
        self,
        num_heads_swa_q: int,
        num_heads_swa_k: int,
        num_heads_full_q: int,
        num_heads_full_k: int,
        head_dim: int,
        norm_eps: float = 1e-5,
        use_query_norm: bool = True,
        use_key_norm: bool = True,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads_swa_q = num_heads_swa_q
        self.num_heads_swa_k = num_heads_swa_k
        self.num_heads_full_q = num_heads_full_q
        self.num_heads_full_k = num_heads_full_k
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.use_query_norm = use_query_norm
        self.use_key_norm = use_key_norm
        self.quant_dtype = quant_dtype

        if quant_dtype == torch.int8:
            self.q_min = -128
            self.q_max = 127
        else:
            raise ValueError(f"Unsupported quant_dtype: {quant_dtype}")

        num_norm_groups = (2 if use_query_norm else 0) + (2 if use_key_norm else 0)
        if num_norm_groups > 0:
            self.qk_norm_weight = torch.nn.Parameter(
                torch.ones(num_norm_groups, head_dim, **self.tensor_factory_kwargs)
            )
        else:
            self.register_parameter("qk_norm_weight", None)

        self.full_k_scale = torch.nn.Parameter(
            torch.ones(num_heads_full_k, head_dim, **self.tensor_factory_kwargs)
        )
        self.full_v_scale = torch.nn.Parameter(
            torch.ones(num_heads_full_k, head_dim, **self.tensor_factory_kwargs)
        )
        self.swa_k_scale = torch.nn.Parameter(
            torch.ones(num_heads_swa_k, head_dim, **self.tensor_factory_kwargs)
        )
        self.swa_v_scale = torch.nn.Parameter(
            torch.ones(num_heads_swa_k, head_dim, **self.tensor_factory_kwargs)
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope_single(self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rope_dim = cos.shape[-1]
        nope_dim = t.shape[-1] - rope_dim
        if nope_dim > 0:
            t_nope, t_rope = torch.split(t, [nope_dim, rope_dim], dim=-1)
            t_rot = (t_rope * cos + self._rotate_half(t_rope) * sin).to(t.dtype)
            return torch.cat([t_nope, t_rot], dim=-1)
        return (t * cos + self._rotate_half(t) * sin).to(t.dtype)

    def _static_quant(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.round(x.float() / scale.float()), self.q_min, self.q_max
        ).to(self.quant_dtype)

    def _store_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_q_lens: Optional[torch.Tensor],
        context_kv_lens: torch.Tensor,
    ) -> None:
        chunk_metadata = build_paged_kv_chunk_metadata(
            block_table, cu_q_lens, context_kv_lens, key_cache.shape[2]
        )
        if chunk_metadata.shape[0] == 0:
            return
        for src_token_start, dst_block_id, dst_block_offset, chunk_len in chunk_metadata.tolist():
            src_end = src_token_start + chunk_len
            dst_end = dst_block_offset + chunk_len
            key_cache[dst_block_id, :, dst_block_offset:dst_end, :] = (
                key_states[src_token_start:src_end].permute(1, 0, 2)
            )
            value_cache[dst_block_id, :, dst_block_offset:dst_end, :] = (
                value_states[src_token_start:src_end].permute(1, 0, 2)
            )

    def forward(
        self,
        swa_query: torch.Tensor,
        swa_key: torch.Tensor,
        swa_value: torch.Tensor,
        full_query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        full_key_cache: torch.Tensor,
        full_value_cache: torch.Tensor,
        swa_key_cache: torch.Tensor,
        swa_value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        cu_q_lens: Optional[torch.Tensor],
        context_kv_lens: torch.Tensor,
        block_tables_sparse: torch.Tensor,
        cu_q_lens_sparse: Optional[torch.Tensor],
        context_kv_lens_sparse: torch.Tensor,
        update_kv: bool = True,
    ) -> Tuple[
        torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
    ]:
        """
        Args:
            swa_query: [T, swa_nh_q, head_dim] pre-norm SWA query
            swa_key: [T, swa_nkv, head_dim] pre-norm SWA key
            swa_value: [T, swa_nkv, head_dim] SWA value (no norm/rope)
            full_query: [T, full_nh_q, head_dim] pre-norm full query
            full_key: [T, full_nkv, head_dim] pre-norm full key
            full_value: [T, full_nkv, head_dim] full value (no norm/rope)
            cos: [T, rope_dim] RoPE cosine
            sin: [T, rope_dim] RoPE sine
            full_key_cache: paged KV cache for full attention keys
            full_value_cache: paged KV cache for full attention values
            swa_key_cache: paged KV cache for SWA keys
            swa_value_cache: paged KV cache for SWA values
            block_tables: [B, max_blocks] block table for full attn
            cu_q_lens: [B+1] cumulative query lengths (None for decode)
            context_kv_lens: [B] existing KV lengths before store
            block_tables_sparse: [B, max_blocks] block table for SWA
            cu_q_lens_sparse: [B+1] cumulative query lengths for SWA
            context_kv_lens_sparse: [B] existing KV lengths for SWA
            update_kv: if True, run full pipeline (norm+rope+quant+store);
                       if False, only compute Q norm+rope (skip K/V entirely)

        Returns:
            Tuple of:
              - swa_query_out: [T, swa_nh_q, head_dim] post-norm+rope
              - full_query_out: [T, full_nh_q, head_dim] post-norm+rope
              - full_key_out: [T, full_nkv, head_dim] int8 (None if update_kv=False)
              - full_k_scale: [full_nkv, head_dim] (None if update_kv=False)
              - swa_key_out: [T, swa_nkv, head_dim] int8 (None if update_kv=False)
              - swa_k_scale: [swa_nkv, head_dim] (None if update_kv=False)
              - full_value_out: [T, full_nkv, head_dim] int8 (None if update_kv=False)
              - full_v_scale: [full_nkv, head_dim] (None if update_kv=False)
              - swa_value_out: [T, swa_nkv, head_dim] int8 (None if update_kv=False)
              - swa_v_scale: [swa_nkv, head_dim] (None if update_kv=False)
        """
        hd = self.head_dim

        # --- 1. GroupRMSNorm on Q and/or K ---
        norm_idx = 0
        if self.use_query_norm:
            swa_query = F.rms_norm(swa_query, (hd,), self.qk_norm_weight[norm_idx], self.norm_eps)
            norm_idx += 1
            if self.use_key_norm:
                swa_key = F.rms_norm(swa_key, (hd,), self.qk_norm_weight[norm_idx], self.norm_eps)
                norm_idx += 1
            full_query = F.rms_norm(full_query, (hd,), self.qk_norm_weight[norm_idx], self.norm_eps)
            norm_idx += 1
            if self.use_key_norm:
                full_key = F.rms_norm(full_key, (hd,), self.qk_norm_weight[norm_idx], self.norm_eps)
                norm_idx += 1
        elif self.use_key_norm:
            swa_key = F.rms_norm(swa_key, (hd,), self.qk_norm_weight[norm_idx], self.norm_eps)
            norm_idx += 1
            full_key = F.rms_norm(full_key, (hd,), self.qk_norm_weight[norm_idx], self.norm_eps)
            norm_idx += 1

        # --- 2. RoPE on Q and K (head_first=False → unsqueeze dim=-2) ---
        cos_exp = cos.unsqueeze(-2)
        sin_exp = sin.unsqueeze(-2)
        swa_query = self._apply_rope_single(swa_query, cos_exp, sin_exp)
        full_query = self._apply_rope_single(full_query, cos_exp, sin_exp)

        if not update_kv:
            return (
                swa_query, full_query,
                None, None, None, None, None, None, None, None,
            )

        swa_key = self._apply_rope_single(swa_key, cos_exp, sin_exp)
        full_key = self._apply_rope_single(full_key, cos_exp, sin_exp)

        # --- 3. StaticQuant on K and V ---
        full_key_q = self._static_quant(full_key, self.full_k_scale)
        full_val_q = self._static_quant(full_value, self.full_v_scale)
        swa_key_q = self._static_quant(swa_key, self.swa_k_scale)
        swa_val_q = self._static_quant(swa_value, self.swa_v_scale)

        # --- 4. Store to paged KV cache ---
        self._store_kv(
            full_key_q, full_val_q,
            full_key_cache, full_value_cache,
            block_tables, cu_q_lens, context_kv_lens,
        )
        self._store_kv(
            swa_key_q, swa_val_q,
            swa_key_cache, swa_value_cache,
            block_tables_sparse, cu_q_lens_sparse, context_kv_lens_sparse,
        )

        return (
            swa_query, full_query,
            full_key_q, self.full_k_scale,
            swa_key_q, self.swa_k_scale,
            full_val_q, self.full_v_scale,
            swa_val_q, self.swa_v_scale,
        )

    def extra_repr(self) -> str:
        return (
            f"num_heads_swa_q={self.num_heads_swa_q}, "
            f"num_heads_swa_k={self.num_heads_swa_k}, "
            f"num_heads_full_q={self.num_heads_full_q}, "
            f"num_heads_full_k={self.num_heads_full_k}, "
            f"head_dim={self.head_dim}, "
            f"norm_eps={self.norm_eps}, "
            f"use_query_norm={self.use_query_norm}, "
            f"use_key_norm={self.use_key_norm}, "
            f"quant_dtype={self.quant_dtype}"
        )
