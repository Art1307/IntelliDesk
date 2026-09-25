# Intellidesk: Offline AI Document Assistant

An offline, privacy-first Retrieval-Augmented Generation (RAG) assistant powered by a custom-built, decoder-only Transformer Language Model implemented from scratch in PyTorch.

---

## Architecture Overview

```
                      +-----------------------------+
                      |       Raw Text Prompt       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      TF-IDF Retriever       | <--- data/training.txt
                      |  (Cosine Similarity Search) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Augmented Context Prompt   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Character-Level Tokenizer |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Transformer Embedding    |
                      |  (Token + Positional + Drop)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  N x Transformer Blocks     |
                      |  - Pre-LayerNorm (Pre-LN)   |
                      |  - Causal Multi-Head Attn   |
                      |  - Residual Connections     |
                      |  - Position-Wise FFN (GELU) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Final LayerNorm         |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Language Model Head     |
                      |  (Weight-Tied with Embeds)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Autoregressive Generation  |
                      +-----------------------------+
```

---

## Project Structure

```
Intellidesk/
├── checkpoints/              # Saved model weights and tokenizer vocab
│   ├── model.pt
│   └── tokenizer.json
├── data/                     # Training and domain reference corpus
│   └── training.txt
├── src/
│   ├── model/               # Neural network architecture modules
│   │   ├── embeddings.py    # Token, Positional, and Combined Embeddings
│   │   ├── attention.py     # Causal Multi-Head Self-Attention
│   │   ├── feedforward.py   # Position-Wise Feed-Forward Network (GELU)
│   │   ├── block.py         # Pre-LN Transformer Decoder Block
│   │   └── transformer.py   # Full TransformerLM with LM Head & Weight Tying
│   ├── data/                # Data processing and tokenization
│   │   ├── tokenizer.py     # Character-level tokenizer with special tokens
│   │   └── dataset.py       # Sliding-window next-token dataset & loader
│   ├── training/            # Training pipeline
│   │   └── train.py         # Trainer with AdamW parameter grouping & grad clip
│   └── inference/           # Retrieval and generation
│       └── retriever.py     # TF-IDF & Cosine Similarity vector retriever
├── tests/                   # Pytest unit testing suite (39 passing tests)
│   ├── test_embeddings.py
│   ├── test_attention.py
│   ├── test_feedforward.py
│   ├── test_block.py
│   ├── test_transformer.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_retriever.py
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -v
```

### 3. Train Model
```bash
python -m src.training.train
```

---

## Key Theoretical Concepts Implemented
- **Causal Masking**: Enforces strict autoregressive generation, preventing attention leakage from future tokens.
- **Pre-Layer Normalization (Pre-LN)**: Stabilizes training from step 0 by maintaining an unimpeded residual identity highway.
- **Weight Tying (Press & Wolf, 2017)**: Reuses the token embedding matrix for the final linear projection layer, halving parameters and improving generalization.
- **AdamW Decoupled Weight Decay**: Applies weight decay strictly to 2D weight matrices while keeping 1D biases and normalization scales unpenalized.
- **Offline RAG with TF-IDF**: Eliminates external API dependencies and guarantees privacy by grounding responses in local domain documents using vector-space similarity.
