import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer for the Transformer.
    
    Academic / Viva Reference:
    - What it does: Maps discrete integer token IDs into continuous dense vectors.
    - Why it is needed: Neural networks operate on continuous vector spaces, not discrete symbols.
    - Input shape:  (batch_size, seq_len) with dtype torch.long
    - Output shape: (batch_size, seq_len, d_model) with dtype torch.float32
    - Mathematical operation: E(X) = X * W_E, where W_E in R^(vocab_size x d_model)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of token indices, shape (batch_size, seq_len)
        Returns:
            Tensor of token embeddings, shape (batch_size, seq_len, d_model)
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding Layer for the Transformer.

    Academic / Viva Reference:
    - What it does: Encodes sequential order of tokens into continuous dense vectors.
    - Why it is needed: Self-attention is permutation-invariant; without positional embeddings,
      the network treats sequences as unordered bags of tokens.
    - Input: Tensor of shape (batch_size, seq_len) or integer sequence length.
    - Output shape: (1, seq_len, d_model), broadcastable across batches.
    - Mathematical operation: P = [0, 1, ..., seq_len - 1], E_pos = Embedding(P, W_pos)
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing token indices.
        Returns:
            Positional embedding tensor of shape (1, seq_len, d_model).
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported sequence length {self.max_seq_len}."
            )
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        return self.embedding(positions).unsqueeze(0)


class TransformerEmbedding(nn.Module):
    """
    Combined Token and Positional Embedding Layer for the Transformer.

    Academic / Viva Reference:
    - What it does: Sums token embeddings and positional embeddings elementwise, then applies dropout.
    - Why it is needed: Merges identity information (which token) with structural information (where it appears)
      into a single representation vector per token position before self-attention.
    - Input shape:  (batch_size, seq_len) with dtype torch.long
    - Output shape: (batch_size, seq_len, d_model) with dtype torch.float32
    - Mathematical operation: E_final = Dropout(TokenEmbedding(X) + PositionalEmbedding(P))
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of token indices, shape (batch_size, seq_len)
        Returns:
            Combined embeddings tensor, shape (batch_size, seq_len, d_model)
        """
        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(x)
        return self.dropout(tok_emb + pos_emb)

