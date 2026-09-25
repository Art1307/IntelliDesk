import json
from pathlib import Path
import torch


class Tokenizer:
    """
    Character-level Tokenizer with Special Token support for the Transformer.

    Academic / Viva Reference:
    - What it does: Converts raw human text into discrete integer token sequences and back.
    - Tokenization Trade-offs (Classic Viva Question):
      1. Word-level: Large vocabulary (50k-100k+), struggles with unseen words (OOV) and morphology.
      2. Subword (BPE/WordPiece): Balanced vocabulary (30k-50k), handles rare words via word pieces,
         used in modern large models (GPT, BERT).
      3. Character-level: Tiny vocabulary (~60-100), zero OOV issues, robust to typos, perfectly
         suited for lightweight, self-contained educational & offline systems.
    - Special Tokens:
      - <pad> (ID 0): Padding token for batch alignment.
      - <unk> (ID 1): Unknown token for unseen characters.
      - <bos> (ID 2): Beginning-of-sequence delimiter.
      - <eos> (ID 3): End-of-sequence delimiter.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, chars: list[str] | None = None):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

        # Initialize special tokens
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token

        if chars:
            self._build_vocab(chars)

    def _build_vocab(self, unique_chars: list[str]) -> None:
        """Assign IDs to unique characters following the special tokens."""
        sorted_chars = sorted(set(unique_chars))
        current_id = len(self.SPECIAL_TOKENS)
        for ch in sorted_chars:
            if ch not in self.char_to_id:
                self.char_to_id[ch] = current_id
                self.id_to_char[current_id] = ch
                current_id += 1

    @classmethod
    def from_text(cls, text: str) -> "Tokenizer":
        """Instantiate and train tokenizer directly on text corpus."""
        unique_chars = list(set(text))
        return cls(chars=unique_chars)

    @classmethod
    def from_file(cls, filepath: str | Path) -> "Tokenizer":
        """Load text from file and train tokenizer."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_text(content)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.char_to_id)

    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> list[int]:
        """
        Encode raw string into a list of integer token IDs.

        Args:
            text: Input string.
            add_special_tokens: If True, prepends <bos> and appends <eos>.

        Returns:
            List of integer token IDs.
        """
        tokens = [self.char_to_id.get(ch, self.UNK_ID) for ch in text]
        if add_special_tokens:
            tokens = [self.BOS_ID] + tokens + [self.EOS_ID]
        return tokens

    def decode(
        self, tokens: list[int] | torch.Tensor, skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back into human-readable text.

        Args:
            tokens: List of token IDs or 1D Tensor of token IDs.
            skip_special_tokens: If True, strips special control tokens.

        Returns:
            Decoded string.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        result = []
        for tid in tokens:
            token = self.id_to_char.get(tid, self.UNK_TOKEN)
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            result.append(token)
        return "".join(result)

    def save(self, filepath: str | Path) -> None:
        """Persist tokenizer vocabulary to JSON file."""
        data = {
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | Path) -> "Tokenizer":
        """Load tokenizer vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls()
        instance.char_to_id = data["char_to_id"]
        instance.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        return instance
