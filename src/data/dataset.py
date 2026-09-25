from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.tokenizer import Tokenizer


class TextDataset(Dataset):
    """
    Sliding-Window Autoregressive Next-Token Dataset for Language Modeling.

    Academic / Viva Reference:
    - What it does: Converts a continuous stream of token IDs into paired input-target sequences
      (x, y) where y is identical to x shifted by 1 position forward in time.
    - Why it is needed:
      1. Self-Supervised Learning: Eliminates the need for manually labeled data by using the
         natural order of human text as the supervision signal.
      2. Causal Alignment:
         Input  x: [t_0,   t_1,   ..., t_{S-1}]
         Target y: [t_1,   t_2,   ..., t_S    ]
         At every index i, the model attempts to predict y[i] given context x[0:i+1].
      3. Sliding Window: Using an overlapping stride extracts maximal training examples
         from compact domain corpora.
    - Output shapes:
      - x: (seq_len,) with dtype torch.long
      - y: (seq_len,) with dtype torch.long
    """

    def __init__(
        self,
        tokens: list[int],
        seq_len: int = 64,
        stride: int | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride if stride is not None else 1
        self.tokens = tokens

        if len(self.tokens) <= self.seq_len:
            raise ValueError(
                f"Token stream length ({len(self.tokens)}) must be greater than seq_len ({self.seq_len})."
            )

        # Total number of sliding window segments
        self.num_samples = (len(self.tokens) - self.seq_len) // self.stride

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        tokenizer: Tokenizer,
        seq_len: int = 64,
        stride: int | None = None,
    ) -> "TextDataset":
        """Load text from file, tokenize, and create TextDataset."""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        return cls(tokens=tokens, seq_len=seq_len, stride=stride)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len

        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1 : end + 1], dtype=torch.long)
        return x, y


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Convenience helper to create a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
