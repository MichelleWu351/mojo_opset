"""Accuracy tests for MojoDeepEPDispatch / MojoDeepEPCombine."""

import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mojo_opset import MojoDeepEPCombine
from mojo_opset import MojoDeepEPDispatch
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _make_global_inputs(world_size, num_tokens_sp, hidden, num_experts, top_k, dtype, device):
    global_tokens = num_tokens_sp * world_size
    if dtype == torch.int8:
        hidden_states = torch.randint(-128, 127, (global_tokens, hidden), dtype=torch.int8, device=device)
    else:
        hidden_states = torch.randn(global_tokens, hidden, dtype=dtype, device=device)
    gating = torch.rand(global_tokens, num_experts, dtype=torch.float32, device=device)
    top_k_logits, top_k_indices = torch.topk(gating, top_k)
    top_k_gates = torch.nn.functional.softmax(top_k_logits, dim=-1)
    return hidden_states, top_k_gates, top_k_indices.to(torch.int32)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _xops_skip_if_unsupported(num_experts, world_size):
    if world_size < 1:
        pytest.skip("MOJO_XOPS_TEST_WORLD_SIZE must be >= 1")
    if num_experts % world_size != 0:
        pytest.skip(f"num_experts={num_experts} must be divisible by world_size={world_size}")
    if torch.npu.device_count() < world_size:
        pytest.skip(f"Need {world_size} NPU devices, got {torch.npu.device_count()}")
    local_experts = num_experts // world_size
    from mojo_opset_ext.backends.xpu_ops.operators.moe import is_deep_ep_local_experts_supported

    if world_size > 1 and not is_deep_ep_local_experts_supported(local_experts):
        pytest.skip(
            f"DeepEPMoe kernels require local_experts==1 or local_experts%8==0, got {local_experts}"
        )


def _run_distributed(case_args, world_size, worker):
    ctx = mp.get_context("forkserver")
    port = _find_free_port()
    result_queue = ctx.Queue()
    processes = []
    for rank in range(world_size):
        process = ctx.Process(
            target=worker,
            args=(rank, world_size, port, result_queue, case_args),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    errors = [(rank, error) for rank, error in results if error is not None]
    if errors:
        message = "\n".join(f"[Rank {rank}]\n{error}" for rank, error in errors)
        pytest.fail(f"Distributed DeepEP test failed:\n{message}")

    for rank, process in enumerate(processes):
        if process.exitcode != 0:
            pytest.fail(f"[Rank {rank}] exited with code {process.exitcode}")


# (num_experts, top_k, hidden, num_tokens_sp) — kept moderate to keep CI cost bounded.
deep_ep_cases = [
    (8, 2, 256, 16),
    (16, 4, 512, 32),
    (64, 8, 1024, 64),
    (384, 8, 3072, 64),
]


# ---------------------------------------------------------------------------
# Dispatch — single unified test driving forward_diff_with for ops vs torch.
# ---------------------------------------------------------------------------


def _dispatch_compare(rank, world_size, port, queue, case_args):
    """Ops-vs-torch dispatch comparison. Sets up hccl + symmetric memory when
    world_size>1; ``queue`` is the multiprocess result queue (``None`` for an
    in-process single-rank call)."""
    shmem_manager = None
    try:
        if world_size > 1:
            import torch_npu
            from mojo_opset.runtime import MojoSymmetricMemoryManager

            torch_npu.npu.set_device(rank)
            dist.init_process_group(
                backend="hccl",
                rank=rank,
                world_size=world_size,
                init_method=f"tcp://127.0.0.1:{port}",
            )
            shmem_manager = MojoSymmetricMemoryManager.get_or_create(
                backend="xops", shmem_heap_size_mb=2048
            )
            shmem_manager.get_backend_manager()

        num_tokens_sp, hidden, top_k, num_experts, dtype, use_smooth_scale = case_args
        device = get_torch_device()
        torch.manual_seed(0)
        global_hidden, global_gates, global_indices = _make_global_inputs(
            world_size, num_tokens_sp, hidden, num_experts, top_k, dtype, device
        )
        smooth_scale = (
            torch.rand(num_experts, hidden, dtype=torch.float32, device=device) + 0.5
            if use_smooth_scale
            else None
        )
        s = rank * num_tokens_sp
        e = s + num_tokens_sp
        local_hidden = global_hidden[s:e].contiguous()
        local_gates = global_gates[s:e].contiguous()
        local_indices = global_indices[s:e].contiguous()

        op = MojoDeepEPDispatch(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)
        op_ref = MojoDeepEPDispatch._registry.get("torch")(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)

        # Return tuple = (expand_hidden_states, expert_token_cnt_per_rank,
        # expert_token_cnt_cumsum, expand_scale, scatter_index, expert_token_count).
        # Index 3 (expand_scale) is a meaningless placeholder when smooth_scale is None;
        # widen its tolerance so the rest of the tuple still gates the comparison.
        scale_tol = 1e-5 if use_smooth_scale else float("inf")
        op.forward_diff_with(
            op_ref,
            local_hidden,
            local_gates,
            local_indices,
            smooth_scale=smooth_scale,
            atol=(0, 0, 0, scale_tol, 0, 0),
            rtol=(0, 0, 0, scale_tol, 0, 0),
        )

        if queue is not None:
            queue.put((rank, None))
    except Exception:
        if queue is not None:
            queue.put((rank, traceback.format_exc()))
        else:
            raise
    finally:
        if shmem_manager is not None:
            shmem_manager.close()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("num_experts, top_k, hidden, num_tokens_sp", deep_ep_cases)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
@pytest.mark.parametrize("use_smooth_scale", [False, True])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@auto_switch_platform()
@bypass_not_implemented
def test_deep_ep_dispatch(world_size, num_experts, top_k, hidden, num_tokens_sp, dtype, use_smooth_scale):
    """Compare active backend's dispatch with torch backend via forward_diff_with."""
    if dtype == torch.int8 and use_smooth_scale:
        pytest.skip("int8 input + per_token quant is not supported by the kernel.")
    if os.environ.get("MOJO_BACKEND", "").strip().lower() != "xops":
        pytest.skip("ops-vs-torch comparison requires MOJO_BACKEND=xops")

    _xops_skip_if_unsupported(num_experts, world_size)
    case_args = (num_tokens_sp, hidden, top_k, num_experts, dtype, use_smooth_scale)
    _run_distributed(case_args, world_size, _dispatch_compare)


# ---------------------------------------------------------------------------
# Combine — single unified test driving forward_diff_with for ops vs torch.
# ---------------------------------------------------------------------------


def _combine_compare(rank, world_size, port, queue, case_args):
    """Ops-vs-torch combine comparison. Sets up hccl + symmetric memory when
    world_size>1; ``queue`` is the multiprocess result queue (``None`` for an
    in-process single-rank call)."""
    shmem_manager = None
    try:
        if world_size > 1:
            import torch_npu
            from mojo_opset.runtime import MojoSymmetricMemoryManager

            torch_npu.npu.set_device(rank)
            dist.init_process_group(
                backend="hccl",
                rank=rank,
                world_size=world_size,
                init_method=f"tcp://127.0.0.1:{port}",
            )
            shmem_manager = MojoSymmetricMemoryManager.get_or_create(
                backend="xops", shmem_heap_size_mb=2048
            )
            shmem_manager.get_backend_manager()

        num_tokens_sp, hidden, top_k, num_experts, dtype = case_args
        device = get_torch_device()
        torch.manual_seed(0)
        global_hidden, global_gates, global_indices = _make_global_inputs(
            world_size, num_tokens_sp, hidden, num_experts, top_k, dtype, device
        )
        s = rank * num_tokens_sp
        e = s + num_tokens_sp
        local_hidden = global_hidden[s:e].contiguous()
        local_gates = global_gates[s:e].contiguous()
        local_indices = global_indices[s:e].contiguous()

        # Build deterministic combine inputs by running torch dispatch.
        dispatch_op = MojoDeepEPDispatch._registry.get("torch")(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)
        expand, _, _, _, scatter_index, expert_token_count = dispatch_op(
            local_hidden, local_gates, local_indices
        )

        op = MojoDeepEPCombine(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)
        op_ref = MojoDeepEPCombine._registry.get("torch")(
            num_experts=num_experts, top_k=top_k, group_size=world_size, rank=rank,
        ).to(device)

        op.forward_diff_with(
            op_ref,
            expand, local_gates, scatter_index, expert_token_count, num_tokens_sp,
            atol=2**-6, rtol=2**-6, mixed_tol=True,
        )

        if queue is not None:
            queue.put((rank, None))
    except Exception:
        if queue is not None:
            queue.put((rank, traceback.format_exc()))
        else:
            raise
    finally:
        if shmem_manager is not None:
            shmem_manager.close()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("num_experts, top_k, hidden, num_tokens_sp", deep_ep_cases)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@auto_switch_platform()
@bypass_not_implemented
def test_deep_ep_combine(world_size, num_experts, top_k, hidden, num_tokens_sp, dtype):
    """Compare active backend's combine with torch backend via forward_diff_with."""
    if os.environ.get("MOJO_BACKEND", "").strip().lower() != "xops":
        pytest.skip("ops-vs-torch comparison requires MOJO_BACKEND=xops")

    _xops_skip_if_unsupported(num_experts, world_size)
    case_args = (num_tokens_sp, hidden, top_k, num_experts, dtype)
    _run_distributed(case_args, world_size, _combine_compare)
