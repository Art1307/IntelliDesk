import torch

from src.data.tokenizer import Tokenizer
from src.inference.rag_pipeline import RAGPipeline
from src.inference.retriever import DocumentRetriever
from src.model.transformer import TransformerLM


def test_rag_pipeline_generates_and_returns_expected_fields():
    corpus = [
        "Causal masking prevents tokens from attending to future positions.",
        "Offline document assistants operate locally and protect data privacy.",
    ]
    tokenizer = Tokenizer.from_text(" ".join(corpus) + " What is causal masking? Context: Query: Answer:")
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=64,
        dropout=0.0,
    )
    pipeline = RAGPipeline(model, tokenizer, DocumentRetriever(corpus))

    result = pipeline.answer_query("What is causal masking?", top_k=1, max_new_tokens=2)

    assert result["query"] == "What is causal masking?"
    assert result["retrieved_passages"]
    assert "Causal masking" in result["retrieved_passages"][0][0]
    assert isinstance(result["generated_text"], str)
    assert isinstance(result["full_output"], str)


def test_rag_prompt_without_retrieved_passages():
    tokenizer = Tokenizer.from_text("Question Answer:")
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=64,
        dropout=0.0,
    )
    pipeline = RAGPipeline(model, tokenizer, DocumentRetriever())

    assert pipeline.format_prompt("Question", []) == "Query: Question\nAnswer:"


def test_rag_prompt_fits_context_window_and_keeps_query():
    tokenizer = Tokenizer.from_text("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :\n")
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=64,
        dropout=0.0,
    )
    pipeline = RAGPipeline(model, tokenizer, DocumentRetriever())
    passages = [("important context " * 30, 0.9)]

    prompt = pipeline._fit_prompt_to_context("What is the important topic?", passages)

    assert len(tokenizer.encode(prompt)) <= model.max_seq_len
    assert "What is the important topic?" in prompt
    assert prompt.endswith("Answer:")
