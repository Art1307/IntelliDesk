import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.model.transformer import TransformerLM
from src.data.tokenizer import Tokenizer
from src.data.dataset import TextDataset, create_dataloader
from src.data.document_loader import load_documents


def configure_optimizers(
    model: TransformerLM,
    weight_decay: float = 0.01,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """
    Configure AdamW optimizer with decoupled weight decay parameter grouping.

    Academic / Viva Reference:
    - Weight Decay Parameter Grouping (Karpathy / GPT-2 style):
      1. Decay applied to weights (dim >= 2: linear layers, embedding matrices).
      2. No decay applied to 1D vectors (biases, LayerNorm scale/shift gamma/beta).
    - Why AdamW: Decouples weight decay from gradient updates, preserving adaptive learning rates
      for individual features while effectively regularizing weights.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D parameters (LayerNorm weight and bias) should not be decayed
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)


class Trainer:
    """
    Training manager for the Transformer Language Model.

    Academic / Viva Reference:
    - What it does: Coordinates forward pass, loss calculation, backward propagation,
      gradient clipping, parameter updates, and checkpoint management.
    - Gradient Clipping: Scales the gradient vector if its Euclidean norm exceeds max_norm,
      preventing catastrophic divergence (exploding gradients) during training.
    """

    def __init__(
        self,
        model: TransformerLM,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            _, loss = self.model(batch_x, targets=batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        checkpoint_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> list[float]:
        """
        Train the model over a specified number of epochs.

        Args:
            dataloader: DataLoader with training minibatches.
            epochs: Total epochs to train.
            checkpoint_dir: Directory to save checkpoints.
            verbose: If True, prints epoch loss.

        Returns:
            List of average loss values per epoch.
        """
        loss_history = []
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            epoch_loss = self.train_epoch(dataloader)
            loss_history.append(epoch_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:3d}/{epochs:3d} | Train Loss: {epoch_loss:.4f}")

        if checkpoint_dir:
            self.save_checkpoint(
                Path(checkpoint_dir) / "model.pt",
                epoch=epochs,
                loss=loss_history[-1],
            )

        return loss_history

    def save_checkpoint(
        self, filepath: str | Path, epoch: int, loss: float
    ) -> None:
        """Persist model parameters and training state."""
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "num_heads": self.model.num_heads,
                "num_layers": self.model.num_layers,
                "max_seq_len": self.model.max_seq_len,
                "tie_weights": self.model.tie_weights,
            },
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str | Path,
        model: TransformerLM,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cpu",
    ) -> dict:
        """Restore model parameters from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint


def load_training_text(data_path: str | Path) -> str:
    """Read a TXT corpus or extract text from a PDF/directory of PDF and TXT files."""
    source = Path(data_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Training input not found: {source}")

    if source.is_dir() or source.suffix.lower() == ".pdf":
        chunks = load_documents([source])
        text = "\n".join(chunk.text for chunk in chunks)
    elif source.suffix.lower() == ".txt":
        text = source.read_text(encoding="utf-8-sig", errors="replace")
    else:
        raise ValueError(
            f"Unsupported training input '{source}'. Provide a .txt file, .pdf file, or directory of PDF/TXT files."
        )

    if not text.strip():
        raise ValueError(f"No training text could be extracted from {source}")
    return text


def run_training(
    data_path: str = "data/training.txt",
    output_dir: str = "checkpoints",
    epochs: int = 50,
    seq_len: int = 48,
    batch_size: int = 8,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    lr: float = 1e-3,
) -> tuple[TransformerLM, Tokenizer]:
    """Train language model end-to-end on data corpus and save artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Extract local text from the supplied TXT/PDF source(s), then build the tokenizer.
    training_text = load_training_text(data_path)
    (output_path / "training_corpus.txt").write_text(training_text, encoding="utf-8")
    tokenizer = Tokenizer.from_text(training_text)
    tokenizer.save(output_path / "tokenizer.json")

    # 2. Build Dataset & DataLoader from the extracted in-memory corpus.
    dataset = TextDataset.from_text(
        training_text, tokenizer=tokenizer, seq_len=seq_len, stride=2
    )
    dataloader = create_dataloader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # 3. Instantiate TransformerLM
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len + 16,
        dropout=0.1,
        tie_weights=True,
    )

    # 4. Optimizer and Trainer
    optimizer = configure_optimizers(model, lr=lr)
    trainer = Trainer(model=model, optimizer=optimizer)

    # 5. Train
    print(f"Beginning training on {data_path} ({len(dataset)} samples)...")
    trainer.train(
        dataloader=dataloader,
        epochs=epochs,
        checkpoint_dir=output_path,
        verbose=True,
    )

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IntelliDesk's custom Transformer on local TXT/PDF data.")
    parser.add_argument("--data", default="data/training.txt", help="TXT/PDF file or directory containing PDF/TXT files")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for model, tokenizer, and extracted corpus")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    run_training(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.learning_rate,
    )
