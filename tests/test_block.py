import torch
from src.model.block import TransformerBlock


def test_transformer_block_shapes():
    batch_size = 2
    seq_len = 12
    d_model = 64
    num_heads = 4

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=32,
        dropout=0.1,
    )
    block.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    out = block(x)

    assert out.shape == (batch_size, seq_len, d_model), (
        f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"
    )
    assert out.dtype == torch.float32

    # Test returning attention weights
    out2, attn_weights = block(x, return_attn_weights=True)
    assert out2.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_transformer_block_causality():
    """
    CRITICAL VIVA VERIFICATION:
    The full Transformer Block must strictly maintain the causal invariant.
    Perturbing token t = S-1 must NOT alter outputs for tokens 0 through S-2.
    """
    batch_size = 1
    seq_len = 8
    d_model = 32
    num_heads = 4

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=16,
        dropout=0.0,
    )
    block.eval()

    x1 = torch.randn(batch_size, seq_len, d_model)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(batch_size, d_model)

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5), (
        "Causality broken in TransformerBlock: Earlier tokens changed when future token changed!"
    )
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :]), (
        "Expected output at perturbed token to differ."
    )


def test_transformer_block_gradients():
    d_model = 32
    num_heads = 2
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=16,
        dropout=0.0,
    )

    x = torch.randn(2, 6, d_model, requires_grad=True)
    out = block(x)

    loss = out.sum()
    loss.backward()

    # Residual path + Attention + FFN gradients must propagate to input and all submodules
    assert x.grad is not None and torch.any(x.grad != 0)
    assert block.attn.q_proj.weight.grad is not None
    assert block.attn.out_proj.weight.grad is not None
    assert block.ffn.fc1.weight.grad is not None
    assert block.ffn.fc2.weight.grad is not None
    assert block.ln1.weight.grad is not None
    assert block.ln2.weight.grad is not None


if __name__ == "__main__":
    test_transformer_block_shapes()
    test_transformer_block_causality()
    test_transformer_block_gradients()
    print("All TransformerBlock tests passed successfully!")
