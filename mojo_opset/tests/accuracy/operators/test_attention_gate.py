import pytest
import torch

from mojo_opset import MojoAttnOutputGate
from mojo_opset.tests.utils import auto_switch_platform, bypass_not_implemented

torch.manual_seed(42)

DTYPES = [torch.bfloat16, torch.float16]

SHAPES = [
    (1, 4096, 32, 128),
    (7, 4096, 32, 128),
    (13, 4096, 32, 128),
    (128, 4096, 32, 128),
    (255, 4096, 32, 128),
    (1000, 4096, 32, 128),
    (4096, 4096, 32, 128),
    (8192, 4096, 32, 128),
    (1, 2048, 16, 128),
    (37, 2048, 16, 128),
    (4096, 2048, 16, 128),
    (1, 6144, 48, 128),
    (63, 6144, 48, 128),
    (1, 3072, 32, 96),
    (11, 3072, 32, 96),
    (128, 3072, 32, 96),
    (999, 3072, 32, 96),
    (4096, 3072, 32, 96),
    (1, 1536, 16, 96),
    (53, 1536, 16, 96),
    (8191, 1536, 16, 96),
]


def _reference_forward(hidden_states, attn_output, weight, bias, num_heads, head_dim, method):
    """Pure-torch reference matching original M13 modeling logic."""
    gate = torch.matmul(hidden_states.float(), weight.t().float())
    if bias is not None:
        gate = gate + bias.float()
    gate = torch.sigmoid(gate)

    if method == "head":
        gate = gate.view(-1, num_heads, 1)
        out = attn_output.float().view(-1, num_heads, head_dim) * gate
        return out.view(-1, num_heads * head_dim).to(hidden_states.dtype)
    else:
        return (attn_output.float() * gate).to(hidden_states.dtype)


@pytest.mark.parametrize("seq_len, hidden_size, num_heads, head_dim", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("method", ["head", "element"])
@pytest.mark.parametrize("bias", [False, True])
@bypass_not_implemented
def test_attn_output_gate(seq_len, hidden_size, num_heads, head_dim, dtype, method, bias):
    op = MojoAttnOutputGate(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        method=method,
        bias=bias,
        dtype=dtype,
    )
    op_ref = MojoAttnOutputGate._registry.get("torch")(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        method=method,
        bias=bias,
        dtype=dtype,
    )
    op_ref.load_state_dict(op.state_dict())

    hidden_states = torch.randn(seq_len, hidden_size, dtype=dtype)
    attn_output = torch.randn(seq_len, num_heads * head_dim, dtype=dtype)

    op.forward_diff_with(op_ref, hidden_states, attn_output, mixed_tol=True)


@pytest.mark.parametrize("seq_len, hidden_size, num_heads, head_dim", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("method", ["head", "element"])
@pytest.mark.parametrize("bias", [False, True])
def test_attn_output_gate_vs_manual_reference(seq_len, hidden_size, num_heads, head_dim, dtype, method, bias):
    """Verify op output matches the exact M13 modeling computation."""
    op = MojoAttnOutputGate(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        method=method,
        bias=bias,
        dtype=dtype,
    )
    torch.nn.init.normal_(op.weight, std=0.02)
    if op.bias is not None:
        torch.nn.init.zeros_(op.bias)

    hidden_states = torch.randn(seq_len, hidden_size, dtype=dtype)
    attn_output = torch.randn(seq_len, num_heads * head_dim, dtype=dtype)

    actual = op(hidden_states, attn_output)
    expected = _reference_forward(
        hidden_states, attn_output,
        op.weight, op.bias,
        num_heads, head_dim, method,
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_attn_output_gate_invalid_method():
    with pytest.raises(ValueError, match="method must be"):
        MojoAttnOutputGate(hidden_size=128, num_heads=4, head_dim=32, method="invalid")
