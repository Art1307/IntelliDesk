from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.model.transformer import TransformerLM
from src.training.train import Trainer, configure_optimizers


def test_configure_optimizers_parameter_groups():
    """
    CRITICAL VIVA VERIFICATION:
    Verifies that 2D weight matrices (Linear, Embedding) receive weight decay,
    while 1D vectors (biases, LayerNorm scales/biases) have weight decay set to 0.0.
    """
    model = TransformerLM(vocab_size=30, d_model=16, num_heads=2, num_layers=1)
    optimizer = configure_optimizers(model, weight_decay=0.05, lr=1e-3)

    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    nodecay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.05
    assert nodecay_group["weight_decay"] == 0.0

    # Ensure all decayed parameters have dim >= 2
    for p in decay_group["params"]:
        assert p.dim() >= 2

    # Ensure all non-decayed parameters have dim < 2
    for p in nodecay_group["params"]:
        assert p.dim() < 2


def test_trainer_overfit_single_batch():
    """Verify that the model can learn and strictly reduce loss on a small batch."""
    vocab_size = 25
    seq_len = 8
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )

    optimizer = configure_optimizers(model, lr=5e-3)
    trainer = Trainer(model=model, optimizer=optimizer, grad_clip=1.0)

    # Fixed single batch
    x = torch.randint(0, vocab_size, (2, seq_len))
    y = torch.randint(0, vocab_size, (2, seq_len))
    dataloader = DataLoader(TensorDataset(x, y), batch_size=2)

    initial_loss = trainer.train_epoch(dataloader)

    # Train for 25 epochs
    losses = trainer.train(dataloader, epochs=25, verbose=False)

    final_loss = losses[-1]
    assert final_loss < initial_loss, (
        f"Model failed to optimize: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_checkpoint_save_and_load(tmp_path: Path):
    vocab_size = 20
    model1 = TransformerLM(vocab_size=vocab_size, d_model=16, num_heads=2, num_layers=1)
    optimizer1 = configure_optimizers(model1, lr=1e-3)
    trainer1 = Trainer(model=model1, optimizer=optimizer1)

    checkpoint_file = tmp_path / "model.pt"
    trainer1.save_checkpoint(checkpoint_file, epoch=5, loss=0.42)

    # Instantiate fresh model
    model2 = TransformerLM(vocab_size=vocab_size, d_model=16, num_heads=2, num_layers=1)
    optimizer2 = configure_optimizers(model2, lr=1e-3)

    # Parameters before loading should differ
    p1 = list(model1.parameters())[0]
    p2 = list(model2.parameters())[0]
    assert not torch.equal(p1, p2)

    # Load checkpoint
    meta = Trainer.load_checkpoint(checkpoint_file, model=model2, optimizer=optimizer2)
    assert meta["epoch"] == 5
    assert meta["loss"] == 0.42

    # Parameters after loading must match exactly
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(param1, param2)


if __name__ == "__main__":
    test_configure_optimizers_parameter_groups()
    test_trainer_overfit_single_batch()
    print("All Training tests passed successfully!")
