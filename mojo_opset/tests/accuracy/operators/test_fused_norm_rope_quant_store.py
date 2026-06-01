import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mojo_opset.experimental import MojoFusedNormRoPEQuantStore
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata
from mojo_opset.tests.utils import bypass_not_implemented

torch.manual_seed(42)

CONFIGS = [
    # (num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim)
    (8, 2, 32, 4, 128, 128),
    (8, 2, 32, 4, 128, 64),
    (16, 4, 48, 8, 96, 96),
]

SEQ_CONFIGS = [
    # (batch_size, q_lens_list, context_kv_lens_list)
    (1, [1], [0]),
    (1, [1], [15]),
    (2, [1, 1], [0, 7]),
    (1, [32], [0]),
    (2, [16, 8], [5, 10]),
    (1, [128], [0]),
]

BLOCK_SIZE = 16


def _build_kv_case(batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val):
    context_kv_lens = torch.tensor(context_kv_lens_val, dtype=torch.int32)
    q_lens = torch.tensor(q_lens_val, dtype=torch.int32)

    is_decode = all(q == 1 for q in q_lens_val)
    cu_q_lens = (
        torch.cat([
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(q_lens, dim=0, dtype=torch.int32),
        ])
        if not is_decode
        else None
    )

    total_tokens = int(q_lens.sum().item()) if not is_decode else batch_size

    max_kv_len = int(torch.clamp(context_kv_lens + q_lens, min=0).max().item())
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks_needed = sum(
        max(0, ckv + ql + block_size - 1) // block_size
        for ckv, ql in zip(context_kv_lens_val, q_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache = torch.zeros(cache_shape, dtype=torch.int8)
    v_cache = torch.zeros(cache_shape, dtype=torch.int8)

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
    next_block = 0
    for b in range(batch_size):
        needed = max(0, context_kv_lens_val[b] + q_lens_val[b] + block_size - 1) // block_size
        if needed > 0:
            block_table[b, :needed] = torch.arange(next_block, next_block + needed, dtype=torch.int32)
        next_block += needed

    return {
        "total_tokens": total_tokens,
        "cu_q_lens": cu_q_lens,
        "context_kv_lens": context_kv_lens,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
    }


def _ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_apply_rope(t, cos, sin):
    rope_dim = cos.shape[-1]
    nope_dim = t.shape[-1] - rope_dim
    if nope_dim > 0:
        t_nope, t_rope = torch.split(t, [nope_dim, rope_dim], dim=-1)
        t_rot = (t_rope * cos + _ref_rotate_half(t_rope) * sin).to(t.dtype)
        return torch.cat([t_nope, t_rot], dim=-1)
    return (t * cos + _ref_rotate_half(t) * sin).to(t.dtype)


def _ref_static_quant(x, scale):
    return torch.clamp(torch.round(x.float() / scale.float()), -128, 127).to(torch.int8)


def _ref_store_kv(key_states, value_states, k_cache, v_cache, block_table, cu_q_lens, context_kv_lens):
    block_size = k_cache.shape[2]
    chunk_metadata = build_paged_kv_chunk_metadata(block_table, cu_q_lens, context_kv_lens, block_size)
    if chunk_metadata.shape[0] == 0:
        return
    for src_token_start, dst_block_id, dst_block_offset, chunk_len in chunk_metadata.tolist():
        src_end = src_token_start + chunk_len
        dst_end = dst_block_offset + chunk_len
        k_cache[dst_block_id, :, dst_block_offset:dst_end, :] = key_states[src_token_start:src_end].permute(1, 0, 2)
        v_cache[dst_block_id, :, dst_block_offset:dst_end, :] = value_states[src_token_start:src_end].permute(1, 0, 2)


@pytest.mark.parametrize("num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim", CONFIGS)
@pytest.mark.parametrize("batch_size, q_lens_val, context_kv_lens_val", SEQ_CONFIGS)
@pytest.mark.parametrize("update_kv", [True, False])
@bypass_not_implemented
def test_fused_norm_rope_quant_store(
    num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim,
    batch_size, q_lens_val, context_kv_lens_val,
    update_kv,
):
    torch.manual_seed(42)

    op = MojoFusedNormRoPEQuantStore(
        num_heads_swa_q=num_heads_swa_q,
        num_heads_swa_k=num_heads_swa_k,
        num_heads_full_q=num_heads_full_q,
        num_heads_full_k=num_heads_full_k,
        head_dim=head_dim,
        norm_eps=1e-5,
        use_query_norm=True,
        use_key_norm=True,
        quant_dtype=torch.int8,
    )

    for p in op.parameters():
        nn.init.normal_(p, mean=1.0, std=0.1)

    full_kv_case = _build_kv_case(batch_size, num_heads_full_k, head_dim, BLOCK_SIZE, context_kv_lens_val, q_lens_val)
    swa_kv_case = _build_kv_case(batch_size, num_heads_swa_k, head_dim, BLOCK_SIZE, context_kv_lens_val, q_lens_val)

    T = full_kv_case["total_tokens"]
    swa_query = torch.randn(T, num_heads_swa_q, head_dim, dtype=torch.bfloat16)
    swa_key = torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16)
    swa_value = torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16)
    full_query = torch.randn(T, num_heads_full_q, head_dim, dtype=torch.bfloat16)
    full_key = torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16)
    full_value = torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16)

    cos = torch.randn(T, rope_dim, dtype=torch.bfloat16)
    sin = torch.randn(T, rope_dim, dtype=torch.bfloat16)

    # --- Run fused op ---
    full_k_cache_fused = full_kv_case["k_cache"].clone()
    full_v_cache_fused = full_kv_case["v_cache"].clone()
    swa_k_cache_fused = swa_kv_case["k_cache"].clone()
    swa_v_cache_fused = swa_kv_case["v_cache"].clone()

    result = op(
        swa_query.clone(), swa_key.clone(), swa_value.clone(),
        full_query.clone(), full_key.clone(), full_value.clone(),
        cos, sin,
        full_k_cache_fused, full_v_cache_fused,
        swa_k_cache_fused, swa_v_cache_fused,
        full_kv_case["block_table"], full_kv_case["cu_q_lens"], full_kv_case["context_kv_lens"],
        swa_kv_case["block_table"], swa_kv_case["cu_q_lens"], swa_kv_case["context_kv_lens"],
        update_kv=update_kv,
    )

    # --- Compute reference ---
    qk_norm_weight = op.qk_norm_weight.data.clone()
    norm_eps = op.norm_eps

    # Norm
    norm_idx = 0
    swa_q_ref = F.rms_norm(swa_query.clone(), (head_dim,), qk_norm_weight[norm_idx], norm_eps)
    norm_idx += 1
    swa_k_ref = F.rms_norm(swa_key.clone(), (head_dim,), qk_norm_weight[norm_idx], norm_eps)
    norm_idx += 1
    full_q_ref = F.rms_norm(full_query.clone(), (head_dim,), qk_norm_weight[norm_idx], norm_eps)
    norm_idx += 1
    full_k_ref = F.rms_norm(full_key.clone(), (head_dim,), qk_norm_weight[norm_idx], norm_eps)

    # RoPE
    cos_exp = cos.unsqueeze(-2)
    sin_exp = sin.unsqueeze(-2)
    swa_q_ref = _ref_apply_rope(swa_q_ref, cos_exp, sin_exp)
    full_q_ref = _ref_apply_rope(full_q_ref, cos_exp, sin_exp)

    if update_kv:
        swa_k_ref = _ref_apply_rope(swa_k_ref, cos_exp, sin_exp)
        full_k_ref = _ref_apply_rope(full_k_ref, cos_exp, sin_exp)

        # Quant
        full_k_q_ref = _ref_static_quant(full_k_ref, op.full_k_scale.data)
        full_v_q_ref = _ref_static_quant(full_value.clone(), op.full_v_scale.data)
        swa_k_q_ref = _ref_static_quant(swa_k_ref, op.swa_k_scale.data)
        swa_v_q_ref = _ref_static_quant(swa_value.clone(), op.swa_v_scale.data)

        # Store
        full_k_cache_ref = full_kv_case["k_cache"].clone()
        full_v_cache_ref = full_kv_case["v_cache"].clone()
        swa_k_cache_ref = swa_kv_case["k_cache"].clone()
        swa_v_cache_ref = swa_kv_case["v_cache"].clone()
        _ref_store_kv(
            full_k_q_ref, full_v_q_ref, full_k_cache_ref, full_v_cache_ref,
            full_kv_case["block_table"], full_kv_case["cu_q_lens"], full_kv_case["context_kv_lens"],
        )
        _ref_store_kv(
            swa_k_q_ref, swa_v_q_ref, swa_k_cache_ref, swa_v_cache_ref,
            swa_kv_case["block_table"], swa_kv_case["cu_q_lens"], swa_kv_case["context_kv_lens"],
        )

    # --- Verify ---
    (swa_query_out, full_query_out,
     full_key_out, full_k_scale_out, swa_key_out, swa_k_scale_out,
     full_value_out, full_v_scale_out, swa_value_out, swa_v_scale_out) = result

    torch.testing.assert_close(swa_query_out.float(), swa_q_ref.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(full_query_out.float(), full_q_ref.float(), atol=1e-3, rtol=1e-3)

    if update_kv:
        assert full_key_out is not None
        assert full_key_out.dtype == torch.int8
        torch.testing.assert_close(full_key_out.float(), full_k_q_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(swa_key_out.float(), swa_k_q_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(full_value_out.float(), full_v_q_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(swa_value_out.float(), swa_v_q_ref.float(), atol=0, rtol=0)

        torch.testing.assert_close(full_k_scale_out.float(), op.full_k_scale.data.float(), atol=0, rtol=0)
        torch.testing.assert_close(swa_k_scale_out.float(), op.swa_k_scale.data.float(), atol=0, rtol=0)
        torch.testing.assert_close(full_v_scale_out.float(), op.full_v_scale.data.float(), atol=0, rtol=0)
        torch.testing.assert_close(swa_v_scale_out.float(), op.swa_v_scale.data.float(), atol=0, rtol=0)

        # Verify KV cache writes
        torch.testing.assert_close(full_k_cache_fused.float(), full_k_cache_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(full_v_cache_fused.float(), full_v_cache_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(swa_k_cache_fused.float(), swa_k_cache_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(swa_v_cache_fused.float(), swa_v_cache_ref.float(), atol=0, rtol=0)
    else:
        assert full_key_out is None
        assert full_k_scale_out is None
        assert swa_key_out is None
        assert swa_k_scale_out is None
        assert full_value_out is None
        assert full_v_scale_out is None
        assert swa_value_out is None
        assert swa_v_scale_out is None
