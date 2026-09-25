import pytest
import torch
from src.model.feedforward import FeedForward


def test_feedforward_default_shapes():
    batch_size = 2
    seq_len = 16
    d_model = 64

    ffn = FeedForward(d_model=d_model)
    assert ffn.d_ff == 4 * d_model

    x = torch.randn(batch_size, seq_len, d_model)
    out = ffn(x)

    assert out.shape == (batch_size, seq_len, d_model), (
        f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"
    )
    assert out.dtype == torch.float32


def test_feedforward_custom_d_ff():
    d_model = 32
    d_ff = 128
    ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    assert ffn.d_ff == d_ff
    x = torch.randn(3, 8, d_model)
    out = ffn(x)
    assert out.shape == (3, 8, d_model)


def test_feedforward_activations():
    d_model = 16
    # GELU
    ffn_gelu = FeedForward(d_model=d_model, activation="gelu")
    assert isinstance(ffn_gelu.act, torch.nn.GELU)

    # ReLU
    ffn_relu = FeedForward(d_model=d_model, activation="relu")
    assert isinstance(ffn_relu.act, torch.nn.ReLU)

    # Invalid activation
    with pytest.raises(ValueError, match="Unsupported activation"):
        FeedForward(d_model=d_model, activation="sigmoid")


def test_feedforward_gradients():
    d_model = 32
    ffn = FeedForward(d_model=d_model, dropout=0.0)

    x = torch.randn(2, 4, d_model, requires_grad=True)
    out = ffn(x)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.any(x.grad != 0)
    assert ffn.fc1.weight.grad is not None and torch.any(ffn.fc1.weight.grad != 0)
    assert ffn.fc2.weight.grad is not None and torch.any(ffn.fc2.weight.grad != 0)


def test_position_wise_independence():
    """
    CRITICAL VIVA VERIFICATION:
    The FFN is 'position-wise', meaning it acts independently on each token.
    Evaluating token t in a sequence of length N must yield the EXACT same
    output vector as evaluating token t in isolation as a sequence of length 1.
    """
    d_model = 32
    ffn = FeedForward(d_model=d_model, dropout=0.0)
    ffn.eval()

    full_seq = torch.randn(1, 10, d_model)
    with torch.no_grad():
        out_full = ffn(full_seq)

        # Pick token at index 5 and pass it in isolation
        token_5 = full_seq[:, 5:6, :]
        out_single = ffn(token_5)

    # The vectors must be identical
    assert torch.allclose(out_full[:, 5:6, :], out_single, atol=1e-6), (
        "Position-wise property violated: Token output depended on surrounding sequence!"
    )


if __name__ == "__main__":
    test_feedforward_default_shapes()
    test_feedforward_custom_d_ff()
    test_feedforward_activations()
    test_feedforward_gradients()
    test_position_wise_independence()
    print("All FeedForward tests passed successfully!")
