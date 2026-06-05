from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


@triton.jit
def sals_score_kernel(
    query_ptr,
    key_ptr,
    block_table_ptr,
    actual_seq_lengths_key_ptr,
    act_n_counts_ptr,
    scores_ptr,
    meta_ptr,
    stride_qg, stride_qn, stride_qd,
    stride_kblk, stride_kbs, stride_kn, stride_kd,
    stride_btg, stride_btt,
    stride_sg, stride_sn, stride_sb,
    N: tl.constexpr,
    D: tl.constexpr,
    SBS: tl.constexpr,
    CBS: tl.constexpr,
    max_sort_n: tl.constexpr,
    sparse_count: tl.constexpr,
    sparse_ratio_f32,
    fixed_tail_count: tl.constexpr,
    scale,
    BLOCKS_PER_PROG: tl.constexpr,
    G_total: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_sbs = tl.arange(0, SBS).to(tl.int32)
    offs_d = tl.arange(0, D)

    for idx in range(0, BLOCKS_PER_PROG):
        flat = pid * BLOCKS_PER_PROG + idx
        if flat < G_total * N * max_sort_n:
            g = flat // (N * max_sort_n)
            remainder = flat % (N * max_sort_n)
            n = remainder // max_sort_n
            b = remainder % max_sort_n

            act_s2 = tl.load(actual_seq_lengths_key_ptr + g).to(tl.int32)
            act_n_count = tl.load(act_n_counts_ptr + g).to(tl.int32)

            # Write meta on first block
            if b == 0:
                if act_n_count > 0 and act_n_count < fixed_tail_count + 4:
                    keep_n = act_n_count
                    if keep_n > sparse_count:
                        keep_n = sparse_count
                    tl.store(meta_ptr + g * 3 + 0, 0)
                    tl.store(meta_ptr + g * 3 + 1, 0)
                    tl.store(meta_ptr + g * 3 + 2, keep_n)
                elif act_n_count > 0:
                    sort_n_count = act_n_count - fixed_tail_count
                    tmp = (sort_n_count * sparse_ratio_f32 + 0.5).to(tl.int32)
                    topk_n_count = sort_n_count
                    if tmp < 1:
                        tmp = 1
                    if tmp < sort_n_count:
                        topk_n_count = tmp
                    max_sparse_topk = sparse_count - fixed_tail_count
                    if max_sparse_topk < 1:
                        max_sparse_topk = 1
                    if topk_n_count > max_sparse_topk:
                        topk_n_count = max_sparse_topk
                    tl.store(meta_ptr + g * 3 + 0, sort_n_count)
                    tl.store(meta_ptr + g * 3 + 1, topk_n_count)
                    tl.store(meta_ptr + g * 3 + 2, topk_n_count + fixed_tail_count)
                else:
                    tl.store(meta_ptr + g * 3 + 0, 0)
                    tl.store(meta_ptr + g * 3 + 1, 0)
                    tl.store(meta_ptr + g * 3 + 2, 0)

            # Compute score for block b
            if act_n_count > 0 and act_n_count >= fixed_tail_count + 4:
                sort_n_count = act_n_count - fixed_tail_count
                if b < sort_n_count:
                    q_vec = tl.load(
                        query_ptr + g * stride_qg + n * stride_qn + offs_d * stride_qd,
                    ).to(tl.float32)

                    block_start = b * SBS
                    mask_s = (block_start + offs_sbs) < act_s2
                    token_pos = block_start + offs_sbs
                    page_ids = token_pos // CBS
                    page_offsets = token_pos - page_ids * CBS
                    phys_id = tl.load(
                        block_table_ptr + g * stride_btg + page_ids * stride_btt
                    ).to(tl.int32)
                    valid_page = phys_id >= 0

                    k_block = tl.load(
                        key_ptr + phys_id[:, None] * stride_kblk
                        + page_offsets[:, None] * stride_kbs
                        + n * stride_kn
                        + offs_d[None, :] * stride_kd,
                        mask=mask_s[:, None] & valid_page[:, None],
                        other=0.0,
                    ).to(tl.float32)

                    dot_scores = tl.sum(k_block * q_vec[None, :], axis=1) * scale
                    masked_scores = tl.where(mask_s & valid_page, dot_scores, -1e30)
                    m = tl.max(masked_scores)
                    e = tl.exp(masked_scores - m)
                    lse = m + tl.log(tl.sum(e))

                    tl.store(
                        scores_ptr + g * stride_sg + n * stride_sn + b * stride_sb,
                        lse,
                    )
                else:
                    tl.store(
                        scores_ptr + g * stride_sg + n * stride_sn + b * stride_sb,
                        -1e30,
                    )
            else:
                tl.store(
                    scores_ptr + g * stride_sg + n * stride_sn + b * stride_sb,
                    -1e30,
                )


@triton.jit
def sals_topk_kernel(
    scores_ptr,
    meta_ptr,
    sparse_indices_ptr,
    sparse_seq_lengths_key_ptr,
    stride_sg, stride_sn, stride_sb,
    stride_si_g, stride_si_n, stride_si_k,
    max_sort_n: tl.constexpr,
    sparse_count: tl.constexpr,
    fixed_tail_count: tl.constexpr,
    t1_per_prog,
    G_total,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_sort = tl.arange(0, max_sort_n).to(tl.int32)

    for idx_sub in range(0, t1_per_prog):
        flat = pid * t1_per_prog + idx_sub

        if flat < G_total * N:
            g = flat // N
            n = flat % N

            sort_n_count = tl.load(meta_ptr + g * 3 + 0)
            topk_n_count = tl.load(meta_ptr + g * 3 + 1)
            keep_n = tl.load(meta_ptr + g * 3 + 2)

            if sort_n_count == 0:
                if n == 0:
                    tl.store(sparse_seq_lengths_key_ptr + g, keep_n)
                for ki in range(0, sparse_count):
                    if ki < keep_n:
                        tl.store(
                            sparse_indices_ptr + g * stride_si_g + n * stride_si_n + ki * stride_si_k,
                            ki,
                        )
            else:
                if n == 0:
                    tl.store(sparse_seq_lengths_key_ptr + g, keep_n)

                scores_vec = tl.load(
                    scores_ptr + g * stride_sg + n * stride_sn + offs_sort * stride_sb,
                )

                sort_mask = offs_sort < sort_n_count
                working_scores = tl.where(sort_mask, scores_vec, -1e30)
                sparse_indices_g_n = sparse_indices_ptr + g * stride_si_g + n * stride_si_n
                for ki in range(0, max_sort_n):
                    if ki < topk_n_count:
                        best_score = tl.max(working_scores)
                        is_best = (working_scores == best_score) & sort_mask
                        idx_candidates = tl.where(
                            is_best,
                            offs_sort.to(tl.float32),
                            float(max_sort_n),
                        )
                        best_idx = tl.min(idx_candidates).to(tl.int32)

                        working_scores = tl.where(
                            offs_sort == best_idx,
                            -1e30,
                            working_scores,
                        )

                        tl.store(
                            sparse_indices_g_n + ki * stride_si_k,
                            best_idx,
                        )

                for t in range(0, fixed_tail_count):
                    if topk_n_count + t < sparse_count:
                        tl.store(
                            sparse_indices_g_n + (topk_n_count + t) * stride_si_k,
                            sort_n_count + t,
                        )


def sals_indexer_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    act_n_counts: torch.Tensor,
    sparse_block_size: int,
    sparse_ratio: float,
    fixed_tail_count: int,
    sparse_count: int,
    score_mode: str,
    max_seqlen_key: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    G, N_dim, D_dim = query.shape
    T1 = G * N_dim

    max_sort_n = (max_seqlen_key + sparse_block_size - 1) // sparse_block_size
    scale = 1.0 / (D_dim ** 0.5)

    sparse_indices = torch.full((G, N_dim, sparse_count), -1, dtype=torch.int32, device=query.device)
    sparse_seq_lengths_key = torch.zeros((G,), dtype=torch.int32, device=query.device)

    scores_buf = torch.full((G, N_dim, max_sort_n), -1e30, dtype=torch.float32, device=query.device)
    meta_buf = torch.zeros((G, 3), dtype=torch.int32, device=query.device)

    q_s = query.stride()
    k_s = key.stride()
    bt_s = block_table.stride()
    sc_s = scores_buf.stride()

    # Score kernel: parallelize over (G, N, max_sort_n)
    total_score_tasks = G * N_dim * max_sort_n
    core_num = get_num_cores("vector")
    score_prog_num = min(total_score_tasks, core_num)
    blocks_per_prog = triton.cdiv(total_score_tasks, score_prog_num)

    grid = (score_prog_num,)
    sals_score_kernel[grid](
        query, key, block_table, actual_seq_lengths_key, act_n_counts,
        scores_buf, meta_buf,
        q_s[0], q_s[1], q_s[2],
        k_s[0], k_s[1], k_s[2], k_s[3],
        bt_s[0], bt_s[1] if len(bt_s) > 1 else 1,
        sc_s[0], sc_s[1], sc_s[2],
        N_dim, D_dim, sparse_block_size, key.shape[1], max_sort_n, sparse_count,
        float(sparse_ratio), fixed_tail_count,
        scale,
        BLOCKS_PER_PROG=blocks_per_prog,
        G_total=G,
    )

    # TopK kernel: parallelize over (G, N)
    if T1 <= core_num:
        topk_prog_num = T1
        t1_per_prog = 1
    else:
        topk_prog_num = core_num
        t1_per_prog = triton.cdiv(T1, core_num)

    si_s = sparse_indices.stride()
    topk_grid = (topk_prog_num,)
    sals_topk_kernel[topk_grid](
        scores_buf, meta_buf,
        sparse_indices, sparse_seq_lengths_key,
        sc_s[0], sc_s[1], sc_s[2],
        si_s[0], si_s[1], si_s[2],
        max_sort_n, sparse_count, fixed_tail_count,
        t1_per_prog=t1_per_prog,
        G_total=G,
        N=N_dim,
    )

    return sparse_indices, sparse_seq_lengths_key