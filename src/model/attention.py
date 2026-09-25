import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention Layer for the Decoder-Only Transformer.

    Academic / Viva Reference:
    - What it does: Computes scaled dot-product attention across multiple representation
      subspaces simultaneously, enforcing autoregressive causality.
    - Why it is needed:
      1. Self-Attention: Allows tokens to dynamically weigh the importance of other tokens in the context.
      2. Multi-Head: Enables the model to jointly attend to information from different representation
         subspaces at different positions (e.g., syntactic vs. semantic relationships).
      3. Causal Masking: Prevents current tokens from attending to subsequent tokens (future information),
         preserving the autoregressive property required for generative language modeling.
    - Input shape:  (batch_size, seq_len, d_model)
    - Output shape: (batch_size, seq_len, d_model)
    - Mathematical Formulation:
      Q = X * W_q,  K = X * W_k,  V = X * W_v
      Scores = (Q * K^T) / sqrt(d_k)
      Masked_Scores = Scores + M, where M_ij = 0 if j <= i else -inf
      Attention_Weights = softmax(Masked_Scores, dim=-1)
      Head_Out = Attention_Weights * V
      MultiHead_Out = Concat(head_1, ..., head_h) * W_o
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Regularization
        self.attn_dropout = nn.Dropout(p=dropout)
        self.out_dropout = nn.Dropout(p=dropout)

        # Causal mask buffer: lower triangular matrix of ones
        # Shape: (1, 1, max_seq_len, max_seq_len) for broadcastable head and batch ops
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
            1, 1, max_seq_len, max_seq_len
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            return_attn_weights: Whether to return the attention weight matrix.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model),
            or (output, attn_weights) if return_attn_weights is True.
        """
        batch_size, seq_len, d_model = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )

        # 1. Linear projections: (B, S, D) -> (B, S, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into multi-head format: (B, S, H, d_k) -> transpose to (B, H, S, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention scores: (B, H, S, d_k) @ (B, H, d_k, S) -> (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. Apply causal mask (set upper triangle to -infinity)
        sub_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(sub_mask == 0, float("-inf"))

        # 5. Softmax along key dimension to get attention probabilities
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # 6. Context matrix: (B, H, S, S) @ (B, H, S, d_k) -> (B, H, S, d_k)
        context = torch.matmul(attn_weights_dropped, v)

        # 7. Concatenate heads: (B, H, S, d_k) -> (B, S, H, d_k) -> (B, S, D)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 8. Output projection and dropout
        out = self.out_dropout(self.out_proj(context))

        if return_attn_weights:
            return out, attn_weights
        return out
