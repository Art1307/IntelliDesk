from pathlib import Path
import pytest
import torch
from src.data.tokenizer import Tokenizer
from src.data.dataset import TextDataset, create_dataloader


def test_tokenizer_encode_decode_roundtrip():
    text = "Artificial intelligence enables offline document understanding."
    tokenizer = Tokenizer.from_text(text)

    # Encode string to token IDs
    token_ids = tokenizer.encode(text)
    assert len(token_ids) == len(text)

    # Decode token IDs back to string
    reconstructed = tokenizer.decode(token_ids)
    assert reconstructed == text, f"Expected '{text}', got '{reconstructed}'"


def test_tokenizer_special_tokens():
    text = "hello world"
    tokenizer = Tokenizer.from_text(text)

    # Encode with special tokens
    tokens_with_special = tokenizer.encode(text, add_special_tokens=True)
    assert tokens_with_special[0] == Tokenizer.BOS_ID
    assert tokens_with_special[-1] == Tokenizer.EOS_ID

    # Decoding with skip_special_tokens=True should return original text
    decoded = tokenizer.decode(tokens_with_special, skip_special_tokens=True)
    assert decoded == text

    # Unknown character mapping
    unseen_char = "Z"
    assert unseen_char not in tokenizer.char_to_id
    unk_encoded = tokenizer.encode(unseen_char)
    assert unk_encoded == [Tokenizer.UNK_ID]


def test_tokenizer_save_and_load(tmp_path: Path):
    text = "Testing persistence of tokenizer vocabulary."
    tokenizer = Tokenizer.from_text(text)

    save_path = tmp_path / "vocab.json"
    tokenizer.save(save_path)

    loaded_tokenizer = Tokenizer.load(save_path)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.char_to_id == tokenizer.char_to_id

    # Verify identical encode/decode
    assert loaded_tokenizer.encode(text) == tokenizer.encode(text)


def test_dataset_next_token_alignment():
    """
    CRITICAL VIVA VERIFICATION:
    Verifies that the target sequence y is precisely the input sequence x shifted
    by 1 position forward in time: y[t] == x[t + 1] for all t in [0, seq_len - 2].
    """
    tokens = list(range(20))
    seq_len = 8
    dataset = TextDataset(tokens=tokens, seq_len=seq_len, stride=1)

    assert len(dataset) == (20 - 8) // 1

    x, y = dataset[0]

    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # Mathematically verify the shift alignment
    assert torch.equal(x[1:], y[:-1]), (
        "Causal shift violated: Target sequence is not shifted by 1 token!"
    )
    assert x[0].item() == 0
    assert y[0].item() == 1
    assert y[-1].item() == 8


def test_dataset_from_training_file():
    training_file = Path("data/training.txt")
    assert training_file.exists()

    tokenizer = Tokenizer.from_file(training_file)
    assert tokenizer.vocab_size > 30

    seq_len = 32
    dataset = TextDataset.from_file(training_file, tokenizer=tokenizer, seq_len=seq_len, stride=4)
    assert len(dataset) > 50

    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)


def test_dataloader_batch_shapes():
    tokens = list(range(100))
    seq_len = 16
    batch_size = 4
    dataset = TextDataset(tokens=tokens, seq_len=seq_len, stride=2)
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (batch_size, seq_len)
    assert batch_y.shape == (batch_size, seq_len)
    assert batch_x.dtype == torch.long
    assert batch_y.dtype == torch.long


if __name__ == "__main__":
    test_tokenizer_encode_decode_roundtrip()
    test_tokenizer_special_tokens()
    test_dataset_next_token_alignment()
    test_dataset_from_training_file()
    test_dataloader_batch_shapes()
    print("All Tokenizer and Dataset tests passed successfully!")
