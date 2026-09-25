import torch
from src.model.transformer import TransformerLM


def test_transformer_forward_shapes():
    vocab_size = 100
    batch_size = 2
    seq_len = 10
    d_model = 32
    num_heads = 2
    num_layers = 2

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=32,
    )
    model.eval()

    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, loss = model(idx)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is None


def test_transformer_loss_computation():
    vocab_size = 50
    batch_size = 2
    seq_len = 8
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
    )

    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets=targets)
    assert loss is not None
    assert loss.dim() == 0  # Scalar tensor
    assert loss.item() > 0.0


def test_weight_tying():
    """
    CRITICAL VIVA VERIFICATION:
    Verifies that the embedding weight and LM head weight share the EXACT
    same memory storage (pointer equality).
    """
    vocab_size = 80
    d_model = 32
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        num_layers=1,
        tie_weights=True,
    )

    emb_weight = model.embeddings.token_embedding.embedding.weight
    lm_weight = model.lm_head.weight

    # Memory pointer equality proves weight tying
    assert emb_weight.data_ptr() == lm_weight.data_ptr(), (
        "Weight tying failed: Embedding and LM head weights do not share memory!"
    )


def test_transformer_generation():
    vocab_size = 60
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=64,
    )
    model.eval()

    prompt = torch.tensor([[1, 5, 10, 15]], dtype=torch.long)
    initial_len = prompt.size(1)
    max_new_tokens = 6

    generated = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8, top_k=10)

    assert generated.shape == (1, initial_len + max_new_tokens)
    # The prompt prefix must remain unchanged
    assert torch.equal(generated[:, :initial_len], prompt)
    # Generated tokens must be valid vocabulary indices
    assert torch.all((generated >= 0) & (generated < vocab_size))


def test_transformer_gradients():
    vocab_size = 40
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        dropout=0.0,
    )

    idx = torch.randint(0, vocab_size, (2, 5))
    targets = torch.randint(0, vocab_size, (2, 5))

    _, loss = model(idx, targets=targets)
    loss.backward()

    # Gradients must propagate all the way to embeddings and block parameters
    assert model.embeddings.token_embedding.embedding.weight.grad is not None
    assert model.embeddings.positional_embedding.embedding.weight.grad is not None
    assert model.blocks[0].attn.q_proj.weight.grad is not None
    assert model.blocks[-1].ffn.fc2.weight.grad is not None


if __name__ == "__main__":
    test_transformer_forward_shapes()
    test_transformer_loss_computation()
    test_weight_tying()
    test_transformer_generation()
    test_transformer_gradients()
    print("All TransformerLM tests passed successfully!")
