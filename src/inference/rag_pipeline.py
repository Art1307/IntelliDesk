
import re
from pathlib import Path
from typing import Any

import torch

from src.model.transformer import TransformerLM
from src.data.tokenizer import Tokenizer
from src.inference.retriever import DocumentRetriever


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for IntelliDesk.

    Combines offline TF-IDF retrieval with a decoder-only Transformer.

    Pipeline:
    1. Retrieve relevant document passages.
    2. Fit complete sentences into the model's context window.
    3. Append the user query and answer marker.
    4. Generate text using the Transformer.
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
        """Load a model, matching tokenizer, and document retriever."""
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}"
            )

        tokenizer = Tokenizer.load(tokenizer_path)

        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint["model_config"]

        model = TransformerLM(**model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        retriever = DocumentRetriever()

        if data_path:
            source_path = Path(data_path)

            if source_path.exists():
                if (
                    source_path.is_dir()
                    or source_path.suffix.lower() in {".pdf", ".txt"}
                ):
                    retriever = DocumentRetriever.from_paths(
                        [source_path]
                    )
                else:
                    raise ValueError(
                        "Unsupported retrieval source: "
                        f"{source_path}. Provide a PDF, TXT file, "
                        "or directory."
                    )

        return cls(
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
        )

    def retrieve_context(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[tuple[str, float]]:
        """Retrieve the top-k passages for a query."""
        return self.retriever.retrieve(query, top_k=top_k)

    def format_prompt(
        self,
        query: str,
        retrieved_passages: list[tuple[str, float]],
    ) -> str:
        """Format retrieved passages, query, and answer marker."""
        if not retrieved_passages:
            return f"Query: {query}\nAnswer:"

        context = " ".join(
            passage.strip()
            for passage, _score in retrieved_passages
            if passage.strip()
        )

        return f"Context: {context}\nQuery: {query}\nAnswer:"

    
    def _fit_prompt_to_context(
        self,
        query: str,
        retrieved_passages: list[tuple[str, float]],
    ) -> str:
        """Fit whole retrieved passages and the query into the context window."""
        max_tokens = int(self.model.max_seq_len)
        prefix = "Context: "
        suffix = f"\nQuery: {query}\nAnswer:"

        def token_count(text: str) -> int:
            return len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )

        # Preserve the query and answer marker even if no context fits.
        if token_count(suffix) >= max_tokens:
            return self.tokenizer.decode(
                self.tokenizer.encode(
                    suffix, add_special_tokens=False
                )[-max_tokens:]
            )

        selected = []

        # Passages are already ranked by relevance. Keep them in that order.
        for passage, _score in retrieved_passages:
            passage = passage.strip()
            if not passage:
                continue

            candidate = selected + [passage]
            context = " ".join(candidate)
            prompt = f"{prefix}{context}{suffix}"

            if token_count(prompt) <= max_tokens:
                selected.append(passage)

        if not selected:
            return suffix

        prompt = f"{prefix}{' '.join(selected)}{suffix}"

        if token_count(prompt) > max_tokens:
            raise RuntimeError("Prompt exceeds model context length.")

        return prompt

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
        Retrieve context and generate an answer.

        Returns the query, retrieved passages, formatted prompt,
        generated answer, and full decoded output.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        retrieved_sources = self.retriever.retrieve_with_metadata(
            query,
            top_k=top_k,
        )

        passages = [
            (str(item["text"]), float(item["score"]))
            for item in retrieved_sources
        ]

        prompt = self._fit_prompt_to_context(query, passages)

        encoded_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        if len(encoded_tokens) > self.model.max_seq_len:
            raise RuntimeError(
                "Prompt exceeds the model's context window: "
                f"{len(encoded_tokens)} > {self.model.max_seq_len}"
            )

        device = next(self.model.parameters()).device

        input_ids = torch.tensor(
            [encoded_tokens],
            dtype=torch.long,
            device=device,
        )

        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sampling,
        )

        prompt_len = input_ids.shape[1]
        new_token_ids = generated_ids[0, prompt_len:].tolist()

        generated_text = self.tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
        )

        return {
            "query": query,
            "retrieved_passages": passages,
            "retrieved_sources": retrieved_sources,
            "prompt": prompt,
            "prompt_token_count": len(encoded_tokens),
            "max_context_tokens": self.model.max_seq_len,
            "generated_text": generated_text,
            "full_output": self.tokenizer.decode(
                generated_ids[0].tolist(),
                skip_special_tokens=True,
            ),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="IntelliDesk RAG Pipeline CLI"
    )

    parser.add_argument(
        "--query",
        type=str,
        default="What is an offline document assistant?",
        help="User query",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Tokenizer path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training.txt",
        help="Document corpus path",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of retrieved passages",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=60,
        help="Maximum number of generated tokens",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline.from_checkpoints(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        data_path=args.data,
    )

    result = pipeline.answer_query(
        args.query,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n--- INTELLIDESK RAG RESULT ---")
    print(f"Query: {result['query']}")
    print(
        f"Context tokens: {result['prompt_token_count']}/"
        f"{result['max_context_tokens']}"
    )

    print(
        f"\nRetrieved Passages "
        f"({len(result['retrieved_passages'])}):"
    )

    for index, (passage, score) in enumerate(
        result["retrieved_passages"],
        start=1,
    ):
        print(f"\n[{index}] Score: {score:.4f}")
        print(passage)

    print("\nFormatted Prompt:")
    print(result["prompt"])

    print("\nGenerated Answer:")
    print(result["generated_text"])
