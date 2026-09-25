import torch
import torch.nn as nn
from src.model.attention import CausalSelfAttention
from src.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block with Pre-Layer Normalization and Residual Connections.

    Academic / Viva Reference:
    - What it does: Chains Causal Multi-Head Self-Attention and Position-Wise Feed-Forward
      sublayers with Layer Normalization and Residual Connections.
    - Why it is needed:
      1. Layer Stacking: Multiple identical blocks allow the model to build hierarchical semantic
         and syntactic representations of sequential context.
      2. Residual Connections (Skip Connections): x_next = x + Sublayer(x). Provides an unbroken
         gradient highway directly through the network, preventing vanishing gradients in deep models.
      3. Pre-LN vs Post-LN:
         - Post-LN: x = LayerNorm(x + Sublayer(x)) [Attention Is All You Need, 2017]
         - Pre-LN:  x = x + Sublayer(LayerNorm(x)) [Modern LLMs: GPT-2, GPT-3, LLaMA]
         Pre-LN significantly stabilizes training from step 0, avoiding sensitive learning rate warmups.
    - Input shape:  (batch_size, seq_len, d_model)
    - Output shape: (batch_size, seq_len, d_model)
    - Mathematical Formulation:
      h_1 = x + CausalSelfAttention(LayerNorm_1(x))
      h_2 = h_1 + FeedForward(LayerNorm_2(h_1))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Sublayer 1: Pre-LN and Causal Multi-Head Self-Attention
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Sublayer 2: Pre-LN and Position-Wise Feed-Forward Network
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            return_attn_weights: Whether to return self-attention weights.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model),
            or tuple of (output, attn_weights) if return_attn_weights is True.
        """
        # 1. Attention sublayer with Pre-LN & Residual connection
        norm_x1 = self.ln1(x)
        if return_attn_weights:
            attn_out, attn_weights = self.attn(norm_x1, return_attn_weights=True)
        else:
            attn_out = self.attn(norm_x1, return_attn_weights=False)

        x = x + attn_out

        # 2. Feedforward sublayer with Pre-LN & Residual connection
        norm_x2 = self.ln2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out

        if return_attn_weights:
            return x, attn_weights
        return x
