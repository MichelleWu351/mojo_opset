"""UC backend for ``MojoStoreLowrank`` (experimental).

The high-level mojo op (``mojo_opset/experimental/operators/store_lowrank.py``)
is the indirect store

    label_cache[block_idxs, :, token_idxs, :] = key_lr[:token_num]

with shapes

    label_cache : (num_blocks, num_kv_heads, block_size, head_dim)  bf16  (BNSD)
    key_lr      : (token_num,  num_kv_heads, head_dim)              bf16  (SND)
    block_idxs  : (token_num,)                                      int32
    token_idxs  : (token_num,)                                      int32
    token_num   : int (number of *valid* leading rows of ``key_lr`` to consume)

Performance design history
--------------------------
v1 (pre-P2-28): per-head loop with full ``cache_h`` materialisation +
write-back — ~768 MB of overhead DRAM traffic per call on H=8 shapes.

v2 (P2-28, 2026-06-05): all-heads-fused single-launch SCATTER with flat
``(num_blocks * H * BS, D)`` slot view; eliminated per-head launches and
per-head ``.contiguous()`` cache materialisation. See ``_uc_store_lowrank_bf16``
docstring below.

v3 (P1-G6, 2026-06-11): wrapper host plan micro-optimisation while keeping
the v2 kernel ABI intact. Three independent host-side wins:

  * **Cache ``h_off = arange(H, int32, device) * BS``** at module level —
    these two NPU ops never depend on per-call data, so they should run
    once per (H, BS, device) tuple, not every call. Saves ~70 µs/call on
    NPU (full plan rebuild: 104 µs → 32 µs on the H=8 hot path).
  * **H=1 broadcast collapse** — when ``num_kv_heads == 1`` the broadcast
    + reshape chain reduces to identity (``slot_idx = base``), saving the
    final ~31 µs of NPU ops for all H=1 shapes (~7/14 perf test rows).
  * **Drop redundant ``.contiguous()``** after ``.reshape(-1)`` — verified
    is_contiguous()=True after broadcast+reshape, the trailing ``.contiguous()``
    is a wasted NPU op.
  * **Drop redundant ``.to(torch.int32)``** — public ``forward`` already
    asserts both ``block_idxs`` / ``token_idxs`` are int32, the inner
    helper does not need defensive casts on the hot path.

Kernel ABI is unchanged from v2 (still ``mojo_store_lowrank_bf16(key_lr,
slot_idx, label_cache, M, D, S)``), reuses the wheel _kernels.so without
rebuild.

Fallback (raise NotImplementedError) routes to other backends whenever the
fast path's preconditions fail: dtype != bf16, kernel API missing,
key_lr/label_cache rank wrong, token_num <= 0, or label_cache non-contiguous
(rare — a fresh ``torch.zeros(...)`` BNSD is always contiguous).
"""

import torch

from mojo_opset.experimental import MojoStoreLowrank

from ._utils import _uc_kernels


_API = "mojo_store_lowrank_bf16"

# Must match ROW_TILE in ``uc-kernel/kernels/mojo_store_lowrank_bf16.py``.
# The kernel processes M source rows in ROW_TILE-row blocks (one batched
# GM->UB gather per block, then ROW_TILE per-row SCATTERs). The wrapper must
# ensure the value of M passed into the kernel is a multiple of ROW_TILE;
# tail rows are scattered on the host (cheap — at most ROW_TILE-1 rows, and
# torch advanced indexing on a contiguous flat view is fast on NPU).
_KERNEL_ROW_TILE = 128

# Cache for the (H,) int32 head-offset vector ``arange(H) * BS``.
#
# This vector only depends on (num_kv_heads, block_size, device). All three
# are effectively static across inference (per-model config + single device
# per wrapper instance), so building it on every call wastes 25-30 µs of
# NPU launch overhead (one ``torch.arange`` + one element-wise mul on top of
# the scalar scaffold). Micro-bench on 910B NPU=4, kv=13312 H=8:
#   full host plan rebuild: 104 µs  (uncached, every call)
#   full host plan rebuild:  32 µs  (h_off cached)
# Savings scale with call count — for hot inference loops this is pure win.
# Pattern follows lessons §I.5 (host launch floor) / §D.2 (per-instance
# cache by hashable spec) introduced by UCRelativeEmbedding.
#
# Cache invariants:
#   * (H, BS): purely structural integers, never mutate after init
#   * device: pinned per cache entry; multi-device callers get one entry
#     per device (extra 25 µs once per device)
# We never invalidate; entries live for the process lifetime. Total size is
# O(distinct_H × distinct_BS × distinct_device), bounded by a handful in
# practice (kv_heads ∈ {1, 8, ...}, block_size ∈ {512, ...}).
_H_OFF_CACHE: dict = {}


def _get_h_off(H: int, BS: int, device: torch.device) -> torch.Tensor:
    """Cached ``arange(H, int32, device) * BS`` to skip 2 NPU op launches
    per call. See module-level ``_H_OFF_CACHE`` docstring and lessons §I.5.
    """
    key = (H, BS, device)
    cached = _H_OFF_CACHE.get(key)
    if cached is None:
        cached = torch.arange(H, dtype=torch.int32, device=device) * BS
        _H_OFF_CACHE[key] = cached
    return cached


def _uc_store_lowrank_bf16(
    label_cache: torch.Tensor,
    key_lr: torch.Tensor,
    block_idxs: torch.Tensor,
    token_idxs: torch.Tensor,
    token_num: int,
) -> torch.Tensor:
    """All-heads-fused single-launch SCATTER.

    Mutates ``label_cache`` in place via a flat view and also returns it
    (mirrors the mojo contract in ``MojoStoreLowrank.forward``).
    """
    num_blocks, num_kv_heads, block_size, head_dim = label_cache.shape
    assert key_lr.shape[1] == num_kv_heads and key_lr.shape[2] == head_dim, (
        "key_lr must be (token_num, num_kv_heads, head_dim) matching label_cache "
        "(num_blocks, num_kv_heads, block_size, head_dim)"
    )

    device = label_cache.device

    # Build the fused (M*H,) int32 slot vector with one broadcast on device.
    # slot_idx[m*H + h] = block_idxs[m] * (H*BS) + h * BS + token_idxs[m]
    # Layout matches `key_lr[:M].view(M*H, D)` row-major.
    #
    # Skip ``.to(torch.int32)`` casts on inputs: ``UCStoreLowrank.forward``
    # below already asserts both ``block_idxs`` / ``token_idxs`` are int32,
    # so the conversion is purely defensive and short-circuits to a no-op
    # in PyTorch — but each call still pays Python-side attribute lookups.
    # Drop them on the fast path; the public ``forward`` is the contract
    # boundary.
    H = num_kv_heads
    BS = block_size
    base = block_idxs[:token_num] * (H * BS) + token_idxs[:token_num]   # (M,)
    if H == 1:
        # H=1 — the per-head broadcast collapses to identity (only h=0).
        # Skip the ``unsqueeze + broadcast + reshape`` chain entirely:
        # micro-bench on 910B saves ~31 µs/call (3 NPU ops fused away).
        slot_idx = base
    else:
        # H>1 — fuse per-head broadcast with cached ``h_off``.
        h_off = _get_h_off(H, BS, device)                                  # (H,)
        # ``.reshape(-1)`` after a broadcasted add always materialises a
        # contiguous buffer (verified is_contiguous()=True, stride=(1,));
        # the historical ``.contiguous()`` tail is a redundant NPU op —
        # drop it.
        slot_idx = (base.unsqueeze(1) + h_off.unsqueeze(0)).reshape(-1)    # (M*H,)

    # Source: (token_num, H, D) -> (M*H, D). When caller passes a contiguous
    # tensor (the common case), .contiguous() is a free no-op.
    key_lr_flat = key_lr[:token_num].contiguous().view(token_num * H, head_dim)

    # Destination: flat view over the same storage, no allocation, no copy.
    label_cache_flat = label_cache.view(num_blocks * H * BS, head_dim)

    # ROW_TILE alignment: the kernel processes ROW_TILE source rows per
    # iteration in one batched gather. Split off the tail rows (< ROW_TILE)
    # and scatter them on the host — the tail is at most ``ROW_TILE - 1``
    # rows so cost is negligible vs the kernel-handled bulk.
    total_rows = token_num * H
    main_rows = (total_rows // _KERNEL_ROW_TILE) * _KERNEL_ROW_TILE
    tail_rows = total_rows - main_rows

    # Small-shape fast path: when ``main_rows`` is so small that the kernel
    # launch + only-a-few-programs-active overhead dominates, just do the
    # whole scatter on the host (torch advanced indexing) — measured faster
    # for ``total_rows < 256`` on 910B (e.g. kv=24, H=8 case: 200us kernel
    # vs ~190us pure-host, see worker-reports/P2-28-store-lowrank.md).
    _MIN_KERNEL_ROWS = 256
    if main_rows < _MIN_KERNEL_ROWS:
        label_cache_flat[slot_idx.to(torch.int64)] = key_lr_flat
        return label_cache

    kernels = _uc_kernels()
    api = kernels[_API]
    api(
        key_lr_flat[:main_rows],
        slot_idx[:main_rows],
        label_cache_flat,
        main_rows,                # M' (rows handled by the kernel)
        head_dim,                 # D
        num_blocks * H * BS,      # S' (total slots)
    )

    if tail_rows > 0:
        # Cheap host scatter for the leftover ROW_TILE-1 rows.
        tail_slots = slot_idx[main_rows:main_rows + tail_rows].to(torch.int64)
        label_cache_flat[tail_slots] = key_lr_flat[main_rows:main_rows + tail_rows]

    return label_cache


class UCStoreLowrank(MojoStoreLowrank):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        label_cache: torch.Tensor,
        key_lr: torch.Tensor,
        block_idxs: torch.Tensor,
        token_idxs: torch.Tensor,
        token_num: int,
    ) -> torch.Tensor:
        # Mirror the mojo contract assertions before any device dispatch so
        # the fallback path also enforces them.
        assert block_idxs.dtype == torch.int32
        assert token_idxs.dtype == torch.int32
        assert label_cache.dim() == 4, "Expected label_cache is BNSD"
        assert key_lr.dim() == 3, "Expected key_lr is SND"

        # dtype fence: wheel kernel is bf16-only.
        if label_cache.dtype != torch.bfloat16 or key_lr.dtype != torch.bfloat16:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # No-op path -> let torch handle it (zero-row scatter is well-defined
        # for advanced indexing but ill-defined for our M=0 launch grid).
        if token_num <= 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # The all-heads-fused fast path requires label_cache to be a normal
        # contiguous BNSD tensor so the flat .view is legal. Tests + real
        # callsites always produce one; non-contiguous → fall back safely.
        if not label_cache.is_contiguous():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # API availability fence: if the wheel doesn't carry the kernel yet,
        # safely fall back to the torch reference rather than KeyError.
        # NB: ``KernelRegistry`` lacks ``__contains__``; use ``.keys()``.
        kernels = _uc_kernels()
        if _API not in kernels.keys():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        return _uc_store_lowrank_bf16(label_cache, key_lr, block_idxs, token_idxs, token_num)
