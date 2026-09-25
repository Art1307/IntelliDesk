import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network (FFN) for the Transformer Block.

    Academic / Viva Reference:
    - What it does: Applies two linear transformations with a non-linear activation in between,
      applied identically and independently to each token position.
    - Why it is needed:
      1. Attention mechanisms perform dynamic linear aggregation across positions; the FFN
         supplies the deep non-linear expressive capacity.
      2. Attention mixes representations across tokens, whereas the FFN processes intra-token
         feature interactions and functions as an associative memory store.
      3. Expands the model dimensionality into a higher-dimensional space (typically 4 * d_model),
         allowing rich feature transformations before projecting back down.
    - Input shape:  (batch_size, seq_len, d_model)
    - Output shape: (batch_size, seq_len, d_model)
    - Mathematical formulation:
      FFN(x) = Dropout(fc2(Activation(fc1(x))))
      where:
        fc1: R^(d_model) -> R^(d_ff)
        fc2: R^(d_ff) -> R^(d_model)
        d_ff typically equals 4 * d_model
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff, bias=True)

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. Must be 'gelu' or 'relu'."
            )

        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # x: (B, S, D) -> fc1 -> (B, S, d_ff) -> act -> fc2 -> (B, S, D) -> dropout
        hidden = self.act(self.fc1(x))
        out = self.fc2(hidden)
        return self.dropout(out)
