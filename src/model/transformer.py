import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.embeddings import TransformerEmbedding
from src.model.block import TransformerBlock


class TransformerLM(nn.Module):
    """
    Full Decoder-Only Autoregressive Transformer Language Model.

    Academic / Viva Reference:
    - What it does: Stacks N Transformer Decoder Blocks on top of token and positional embeddings,
      followed by final layer normalization and an affine language model head that projects hidden
      states to vocabulary logits.
    - Why it is needed:
      1. Generative Modeling: Models the joint probability of a sequence of tokens by decomposing
         it autoregressively: P(w_1, ..., w_T) = prod_{t=1}^T P(w_t | w_1, ..., w_{t-1}).
      2. Weight Tying: Ties the weights of the input token embedding and the output LM head
         (Press & Wolf, 2017). This cuts parameters nearly in half and improves generalization.
      3. Deep Hierarchical Feature Extraction: Multiple stacked self-attention blocks allow lower
         layers to capture local syntax while higher layers capture global semantics.
    - Input shape:
      - idx: (batch_size, seq_len) with integer token IDs in [0, vocab_size - 1]
      - targets (optional): (batch_size, seq_len) with target token IDs for loss computation
    - Output shape:
      - logits: (batch_size, seq_len, vocab_size)
      - loss: scalar cross-entropy loss if targets are provided, else None
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int | None = None,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        # 1. Embedding layer (Token + Learned Positional)
        self.embeddings = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # 2. Stack of N Decoder Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Final Pre-head Layer Normalization
        self.ln_f = nn.LayerNorm(d_model)

        # 4. Language Model Head (Projects hidden state onto vocabulary space)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying (Embedding matrix shared with LM Head)
        if tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with standard normal distribution (std=0.02)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the language model.

        Args:
            idx: Tensor of shape (batch_size, seq_len) with token indices.
            targets: Optional tensor of shape (batch_size, seq_len) with target indices.

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size).
            loss: Scalar cross-entropy loss tensor, or None if targets not provided.
        """
        batch_size, seq_len = idx.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )

        # 1. Embed tokens & positions
        x = self.embeddings(idx)

        # 2. Pass sequentially through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final layer normalization
        x = self.ln_f(x)

        # 4. Project to vocabulary logits: (B, S, D) -> (B, S, V)
        logits = self.lm_head(x)

        # 5. Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens given a conditioning prefix prompt.

        Args:
            idx: Conditioning input prompt tokens of shape (batch_size, seq_len).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (1.0 = normal, <1.0 = confident/greedy, >1.0 = creative).
            top_k: Optional integer to constrain sampling to top-k highest probability tokens.

        Returns:
            idx: Tensor of shape (batch_size, seq_len + max_new_tokens) containing prompt + generated tokens.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if sequence length exceeds max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]

            # Forward pass to get logits for the last token position
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Optional Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")

            # Softmax to obtain probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample next token from categorical distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append sampled token to sequence
            idx = torch.cat((idx, next_token), dim=1)

        return idx
