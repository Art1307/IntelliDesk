import pytest
import torch
from src.model.embeddings import (
    TokenEmbedding,
    PositionalEmbedding,
    TransformerEmbedding,
)


def test_token_embedding_shapes():
    vocab_size = 1000
    d_model = 128
    batch_size = 2
    seq_len = 16

    embedding_layer = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)

    # Input: tensor of integer token IDs in range [0, vocab_size - 1]
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Forward pass
    output = embedding_layer(input_ids)

    # Assertions on tensor shapes and dtypes
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )
    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"


def test_token_embedding_gradients():
    vocab_size = 50
    d_model = 32
    embedding_layer = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)

    input_ids = torch.tensor([[1, 5, 10], [2, 7, 12]], dtype=torch.long)
    output = embedding_layer(input_ids)

    loss = output.sum()
    loss.backward()

    # Verify gradients exist and are populated
    assert embedding_layer.embedding.weight.grad is not None
    # Specified tokens should have non-zero gradients
    assert torch.any(embedding_layer.embedding.weight.grad[1] != 0)


def test_positional_embedding_shapes():
    max_seq_len = 64
    d_model = 128
    batch_size = 4
    seq_len = 16

    pos_embedding = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)
    input_ids = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)

    pos_out = pos_embedding(input_ids)

    # Positional embedding produces (1, seq_len, d_model) for broadcasting
    assert pos_out.shape == (1, seq_len, d_model), (
        f"Expected shape (1, {seq_len}, {d_model}), got {pos_out.shape}"
    )
    assert pos_out.dtype == torch.float32


def test_positional_embedding_gradients():
    max_seq_len = 32
    d_model = 16
    pos_embedding = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)

    input_ids = torch.randint(0, 10, (2, 8), dtype=torch.long)
    pos_out = pos_embedding(input_ids)

    loss = pos_out.sum()
    loss.backward()

    assert pos_embedding.embedding.weight.grad is not None
    # Positions 0 through 7 must have non-zero gradients
    assert torch.all(pos_embedding.embedding.weight.grad[:8] != 0)
    # Positions above 7 must not have received gradients
    assert torch.all(pos_embedding.embedding.weight.grad[8:] == 0)


def test_positional_embedding_exceeds_max_len():
    max_seq_len = 16
    d_model = 32
    pos_embedding = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)

    # Attempt sequence length longer than maximum supported length
    input_ids = torch.randint(0, 50, (2, 20), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds maximum supported sequence length"):
        pos_embedding(input_ids)


def test_transformer_embedding_shapes_and_forward():
    vocab_size = 500
    d_model = 64
    max_seq_len = 128
    batch_size = 3
    seq_len = 24

    combined_layer = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    combined_layer.eval()  # Eval mode to disable stochastic dropout for shape validation
    out = combined_layer(input_ids)

    assert out.shape == (batch_size, seq_len, d_model), (
        f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"
    )
    assert out.dtype == torch.float32


def test_transformer_embedding_gradients():
    vocab_size = 100
    d_model = 32
    max_seq_len = 32
    batch_size = 2
    seq_len = 6

    combined_layer = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    out = combined_layer(input_ids)

    loss = out.sum()
    loss.backward()

    # Both token embedding weights and positional embedding weights should receive gradients
    assert combined_layer.token_embedding.embedding.weight.grad is not None
    assert combined_layer.positional_embedding.embedding.weight.grad is not None
    assert torch.any(combined_layer.token_embedding.embedding.weight.grad != 0)
    assert torch.any(combined_layer.positional_embedding.embedding.weight.grad != 0)


if __name__ == "__main__":
    test_token_embedding_shapes()
    test_token_embedding_gradients()
    test_positional_embedding_shapes()
    test_positional_embedding_gradients()
    test_positional_embedding_exceeds_max_len()
    test_transformer_embedding_shapes_and_forward()
    test_transformer_embedding_gradients()
    print("All Token, Positional, and TransformerEmbedding tests passed successfully!")
