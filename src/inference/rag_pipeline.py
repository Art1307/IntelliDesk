from pathlib import Path
from typing import Any
import torch
from src.model.transformer import TransformerLM
from src.data.tokenizer import Tokenizer
from src.inference.retriever import DocumentRetriever


class RAGPipeline:
    """
    End-to-End Retrieval-Augmented Generation (RAG) Inference Pipeline.

    Academic / Viva Reference:
    - What it does: Combines non-parametric document retrieval (TF-IDF + Cosine Similarity)
      with parametric text generation (Decoder-Only Transformer LM).
    - RAG Process Flow:
      1. Query Processing: User submits natural language question.
      2. Context Retrieval: TF-IDF retriever finds the top-k most semantically relevant
         passages from the offline document corpus.
      3. Prompt Augmentation: Combines retrieved factual context with the user query into
         a structured prompt.
      4. Autoregressive Generation: TransformerLM processes prompt embeddings and predicts
         the response token by token with causal masking.
    """

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: Tokenizer,
        retriever: DocumentRetriever,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.model.eval()

    @classmethod
    def from_checkpoints(
        cls,
        model_path: str | Path = "checkpoints/model.pt",
        tokenizer_path: str | Path = "checkpoints/tokenizer.json",
        data_path: str | Path | None = "data/training.txt",
        device: str = "cpu",
    ) -> "RAGPipeline":
        """
        Instantiate pipeline directly from saved model, tokenizer, and document files.
        """
        tokenizer = Tokenizer.load(tokenizer_path)

        ckpt = torch.load(model_path, map_location=device)
        model_config = ckpt["model_config"]
        model = TransformerLM(**model_config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        if data_path and Path(data_path).exists():
            source_path = Path(data_path)
            if source_path.is_dir() or source_path.suffix.lower() in {".pdf", ".txt"}:
                retriever = DocumentRetriever.from_paths([source_path])
            else:
                raise ValueError(
                    f"Unsupported retrieval source: {source_path}. Provide a PDF, TXT file, or directory."
                )
        else:
            retriever = DocumentRetriever()

        return cls(model=model, tokenizer=tokenizer, retriever=retriever)

    def retrieve_context(self, query: str, top_k: int = 2) -> list[tuple[str, float]]:
        """Retrieve top-k relevant document passages for a query."""
        return self.retriever.retrieve(query, top_k=top_k)

    def format_prompt(
        self, query: str, retrieved_passages: list[tuple[str, float]]
    ) -> str:
        """
        Construct augmented context prompt combining retrieved passages and user query.
        """
        if not retrieved_passages:
            return f"Query: {query}\nAnswer:"

        context_str = " ".join([passage for passage, _ in retrieved_passages])
        return f"Context: {context_str}\nQuery: {query}\nAnswer:"

    def _fit_prompt_to_context(
        self, query: str, retrieved_passages: list[tuple[str, float]]
    ) -> str:
        """Keep the question and answer marker inside the model's context window."""
        prompt = self.format_prompt(query, retrieved_passages)
        max_tokens = self.model.max_seq_len
        if len(self.tokenizer.encode(prompt, add_special_tokens=False)) <= max_tokens:
            return prompt

        # Keep the query at the end because the model conditions on the final context
        # window. Allocate only the remaining characters to retrieved context.
        suffix = f"\nQuery: {query}\nAnswer:"
        prefix = "Context: "
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)

        if len(suffix_ids) >= max_tokens:
            suffix = suffix[-max_tokens:]
            return suffix

        context_budget = max_tokens - len(prefix_ids) - len(suffix_ids)
        if context_budget <= 0:
            return suffix[-max_tokens:]

        context_text = " ".join(passage for passage, _ in retrieved_passages)
        context_text = context_text[:context_budget].rstrip()
        fitted_prompt = f"{prefix}{context_text}{suffix}"
        # Character-level tokenization makes token length equal to string length here;
        # retain a defensive crop in case a tokenizer implementation changes later.
        fitted_ids = self.tokenizer.encode(fitted_prompt, add_special_tokens=False)
        if len(fitted_ids) > max_tokens:
            fitted_prompt = self.tokenizer.decode(fitted_ids[-max_tokens:])
        return fitted_prompt

    @torch.no_grad()
    def answer_query(
        self,
        query: str,
        top_k: int = 2,
        max_new_tokens: int = 60,
        temperature: float = 0.7,
        top_k_sampling: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute full end-to-end RAG pipeline: retrieve context, build prompt, and generate answer.

        Args:
            query: User input query string.
            top_k: Number of document passages to retrieve.
            max_new_tokens: Maximum tokens to generate autoregressively.
            temperature: Sampling temperature for text generation.
            top_k_sampling: Optional top-k candidate truncation for sampling.

        Returns:
            Dict with 'query', 'retrieved_passages', 'prompt', and 'generated_text'.
        """
        retrieved_sources = self.retriever.retrieve_with_metadata(query, top_k=top_k)
        passages = [
            (str(item["text"]), float(item["score"]))
            for item in retrieved_sources
        ]
        prompt = self._fit_prompt_to_context(query, passages)

        # Encode input prompt string to token IDs tensor (shape: 1, seq_len)
        encoded_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([encoded_tokens], device=next(self.model.parameters()).device)

        # Autoregressively generate next token IDs
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sampling,
        )

        # Extract only the newly generated token IDs (after prompt)
        prompt_len = input_ids.shape[1]
        new_token_ids = generated_ids[0, prompt_len:].tolist()

        # Decode generated token IDs back to string
        generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        return {
            "query": query,
            "retrieved_passages": passages,
            "retrieved_sources": retrieved_sources,
            "prompt": prompt,
            "generated_text": generated_text,
            "full_output": self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IntelliDesk RAG Pipeline CLI")
    parser.add_argument("--query", type=str, default="What is offline document assistant?", help="User query")
    parser.add_argument("--model", type=str, default="checkpoints/model.pt", help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Tokenizer path")
    parser.add_argument("--data", type=str, default="data/training.txt", help="Document corpus path")
    parser.add_argument("--top_k", type=int, default=2, help="Number of retrieved passages")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Tokens to generate")
    args = parser.parse_args()

    pipeline = RAGPipeline.from_checkpoints(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        data_path=args.data,
    )

    result = pipeline.answer_query(args.query, top_k=args.top_k, max_new_tokens=args.max_new_tokens)
    print("\n--- RAG PIPELINE RESULT ---")
    print(f"Query: {result['query']}")
    print(f"Retrieved Passages ({len(result['retrieved_passages'])}):")
    for i, (pas, score) in enumerate(result['retrieved_passages'], 1):
        print(f"  [{i}] (Score: {score:.4f}) {pas}")
    print(f"Formatted Prompt:\n{result['prompt']}")
    print(f"Generated Answer:\n{result['generated_text']}")
