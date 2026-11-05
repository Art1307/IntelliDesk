
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.transformer import TransformerLM
from src.data.tokenizer import Tokenizer
from src.data.dataset import TextDataset, create_dataloader
from src.data.document_loader import load_documents


# ============================================================
# 1. OPTIMIZER
# ============================================================

def configure_optimizers(
    model: TransformerLM,
    weight_decay: float = 0.01,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Configure AdamW with appropriate weight-decay groups."""

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=betas,
    )


# ============================================================
# 2. TRAINER
# ============================================================

class Trainer:
    """Training manager for the custom TransformerLM."""

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
        """Train for one complete epoch."""

        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in dataloader:

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            _, loss = self.model(
                batch_x,
                targets=batch_y,
            )

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> float:
        """Calculate average loss without updating model weights."""

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():

            for batch_x, batch_y in dataloader:

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                _, loss = self.model(
                    batch_x,
                    targets=batch_y,
                )

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

        loss_history = []

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):

            epoch_loss = self.train_epoch(dataloader)
            loss_history.append(epoch_loss)

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs:3d} "
                    f"| Train Loss: {epoch_loss:.4f}"
                )

        if checkpoint_dir and loss_history:

            self.save_checkpoint(
                Path(checkpoint_dir) / "model.pt",
                epoch=epochs,
                loss=loss_history[-1],
            )

        return loss_history

    def save_checkpoint(
        self,
        filepath: str | Path,
        epoch: int,
        loss: float,
    ) -> None:
        """Save model weights and optimizer state."""

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
        """Restore model and optionally optimizer state."""

        checkpoint = torch.load(
            filepath,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return checkpoint


# ============================================================
# 3. LOAD TRAINING TEXT
# ============================================================

def load_training_text(data_path: str | Path) -> str:
    """Load text from TXT, PDF, or a directory."""

    source = Path(data_path).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"Training input not found: {source}"
        )

    if source.is_dir() or source.suffix.lower() == ".pdf":

        chunks = load_documents([source])

        text = "\n".join(
            chunk.text for chunk in chunks
        )

    elif source.suffix.lower() == ".txt":

        text = source.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

    else:
        raise ValueError(
            "Provide a TXT file, PDF file, or directory "
            "containing PDF/TXT files."
        )

    if not text.strip():
        raise ValueError(
            f"No training text could be extracted from {source}"
        )

    return text


# ============================================================
# 4. ORIGINAL TRAINING FROM SCRATCH
# ============================================================

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

    """Train a new TransformerLM from scratch."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_text = load_training_text(data_path)

    tokenizer = Tokenizer.from_text(training_text)

    tokenizer.save(
        output_path / "tokenizer.json"
    )

    (output_path / "training_corpus.txt").write_text(
        training_text,
        encoding="utf-8",
    )

    dataset = TextDataset.from_text(
        training_text,
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=2,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len + 16,
        dropout=0.1,
        tie_weights=True,
    )

    optimizer = configure_optimizers(
        model,
        lr=lr,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
    )

    print(
        f"Beginning training on {data_path} "
        f"({len(dataset)} samples)"
    )

    trainer.train(
        dataloader=dataloader,
        epochs=epochs,
        checkpoint_dir=output_path,
        verbose=True,
    )

    return model, tokenizer


# ============================================================
# 5. CONTEXT256 FINE-TUNING
# ============================================================

def fine_tune_context256(
    data_path: str = "data/training_v2.txt",
    checkpoint_path: str = "checkpoints/context256/model.pt",
    tokenizer_path: str = "checkpoints/context256/tokenizer.json",
    output_dir: str = "checkpoints/context256_v2",
    epochs: int = 5,
    seq_len: int = 240,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
):
    """Continue training Context256 with validation."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # A. Load existing tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer.load(tokenizer_path)

    training_text = load_training_text(data_path)

    unsupported = (
        set(training_text) -
        set(tokenizer.char_to_id)
    )

    if unsupported:
        raise ValueError(
            f"Unsupported tokenizer characters: {sorted(unsupported)}"
        )

    # --------------------------------------------------------
    # B. Sequential training/validation split
    # --------------------------------------------------------

    split_index = int(len(training_text) * 0.90)

    train_text = training_text[:split_index]

    val_text = training_text[split_index:]

    train_dataset = TextDataset.from_text(
        train_text,
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=8,
    )

    val_dataset = TextDataset.from_text(
        val_text,
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=8,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            "Training or validation dataset is too small."
        )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # C. Load existing Context256 checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["model_config"]

    if seq_len + 16 != config["max_seq_len"]:
        raise ValueError(
            "Sequence length does not match checkpoint context length."
        )

    if tokenizer.vocab_size != config["vocab_size"]:
        raise ValueError(
            "Tokenizer vocabulary does not match checkpoint."
        )

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        tie_weights=config["tie_weights"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # D. Create fresh optimizer
    # --------------------------------------------------------

    optimizer = configure_optimizers(
        model,
        lr=learning_rate,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
    )

    # --------------------------------------------------------
    # E. Save separate tokenizer and corpus
    # --------------------------------------------------------

    tokenizer.save(
        output_path / "tokenizer.json"
    )

    (output_path / "training_corpus.txt").write_text(
        training_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # F. Fine-tuning with validation
    # --------------------------------------------------------

    best_val_loss = float("inf")

    print("\n========== CONTEXT256 FINE-TUNING ==========")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Starting checkpoint epoch: {checkpoint['epoch']}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: CPU")
    print("============================================\n")

    for epoch in range(1, epochs + 1):

        train_loss = trainer.train_epoch(
            train_loader
        )

        val_loss = trainer.evaluate(
            val_loader
        )

        print(
            f"Epoch {epoch}/{epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Validation Loss: {val_loss:.4f}"
        )

        # Save only the best validation checkpoint
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            trainer.save_checkpoint(
                output_path / "model.pt",
                epoch=epoch,
                loss=val_loss,
            )

            print(
                "Best validation checkpoint saved."
            )

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved model: {output_path / 'model.pt'}")
    print("========================================\n")

    return trainer.model, tokenizer


# ============================================================
# 6. COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train IntelliDesk TransformerLM offline."
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # Original training command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a new model from scratch.",
    )

    train_parser.add_argument(
        "--data",
        default="data/training.txt",
    )

    train_parser.add_argument(
        "--output-dir",
        default="checkpoints",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    train_parser.add_argument(
        "--seq-len",
        type=int,
        default=48,
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    train_parser.add_argument(
        "--d-model",
        type=int,
        default=64,
    )

    train_parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    train_parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    # Context256 fine-tuning command
    fine_parser = subparsers.add_parser(
        "fine-tune",
        help="Fine-tune the existing Context256 checkpoint.",
    )

    fine_parser.add_argument(
        "--data",
        default="data/training_v2.txt",
    )

    fine_parser.add_argument(
        "--checkpoint",
        default="checkpoints/context256/model.pt",
    )

    fine_parser.add_argument(
        "--tokenizer",
        default="checkpoints/context256/tokenizer.json",
    )

    fine_parser.add_argument(
        "--output-dir",
        default="checkpoints/context256_v2",
    )

    fine_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    fine_parser.add_argument(
        "--seq-len",
        type=int,
        default=240,
    )

    fine_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    fine_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    if args.command == "train":

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

    elif args.command == "fine-tune":

        fine_tune_context256(
            data_path=args.data,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            epochs=args.epochs,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    else:

        parser.print_help()
