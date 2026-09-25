import pytest
import torch
from src.model.attention import CausalSelfAttention


def test_causal_attention_shapes():
    batch_size = 2
    seq_len = 16
    d_model = 64
    num_heads = 4

    layer = CausalSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=64)
    layer.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    out = layer(x)

    assert out.shape == (batch_size, seq_len, d_model), (
        f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"
    )
    assert out.dtype == torch.float32


def test_causal_mask_prevents_future_leakage():
    """
    CRITICAL VIVA VERIFICATION:
    In a causal decoder, token at position t MUST NOT be affected by any token at position > t.
    We verify this by:
    1. Running forward pass on sequence X1.
    2. Modifying ONLY the last token to create X2.
    3. Verifying that the outputs for positions 0 to S-2 remain EXACTLY identical.
    """
    batch_size = 1
    seq_len = 8
    d_model = 32
    num_heads = 4

    layer = CausalSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=32, dropout=0.0)
    layer.eval()

    # Create base sequence
    x1 = torch.randn(batch_size, seq_len, d_model)

    # Create second sequence where only the last token is different
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(batch_size, d_model)  # Perturb only the last token

    with torch.no_grad():
        out1 = layer(x1)
        out2 = layer(x2)

    # Outputs for tokens 0 through seq_len - 2 MUST be identical
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6), (
        "Causal masking failed: Earlier tokens were influenced by changes to a future token!"
    )
    # The last token output SHOULD be different
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :]), (
        "Expected output at modified token position to differ."
    )


def test_attention_weights_properties():
    batch_size = 1
    seq_len = 6
    d_model = 32
    num_heads = 2

    layer = CausalSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=16, dropout=0.0)
    layer.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        _, attn_weights = layer(x, return_attn_weights=True)

    # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # Softmax probabilities along the key dimension (last dimension) must sum to 1.0
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "Attention weights do not sum to 1.0 across the key dimension."
    )

    # Future positions (upper triangle above diagonal) must have 0 attention weight
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert torch.all(attn_weights[:, :, i, j] == 0.0), (
                f"Attention weight at row {i}, col {j} should be 0 (future token)."
            )


def test_divisibility_requirement():
    # d_model=35 is not divisible by num_heads=4
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        CausalSelfAttention(d_model=35, num_heads=4)


def test_causal_attention_gradients():
    d_model = 32
    num_heads = 2
    layer = CausalSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=16, dropout=0.0)

    x = torch.randn(2, 4, d_model, requires_grad=True)
    out = layer(x)

    loss = out.sum()
    loss.backward()

    # Verify gradients reach all projection weights and input tensor
    assert x.grad is not None and torch.any(x.grad != 0)
    assert layer.q_proj.weight.grad is not None and torch.any(layer.q_proj.weight.grad != 0)
    assert layer.k_proj.weight.grad is not None and torch.any(layer.k_proj.weight.grad != 0)
    assert layer.v_proj.weight.grad is not None and torch.any(layer.v_proj.weight.grad != 0)
    assert layer.out_proj.weight.grad is not None and torch.any(layer.out_proj.weight.grad != 0)


def test_max_seq_len_exceeded():
    layer = CausalSelfAttention(d_model=16, num_heads=2, max_seq_len=8)
    x = torch.randn(1, 10, 16)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        layer(x)


if __name__ == "__main__":
    test_causal_attention_shapes()
    test_causal_mask_prevents_future_leakage()
    test_attention_weights_properties()
    test_divisibility_requirement()
    test_causal_attention_gradients()
    test_max_seq_len_exceeded()
    print("All CausalSelfAttention tests passed successfully!")
