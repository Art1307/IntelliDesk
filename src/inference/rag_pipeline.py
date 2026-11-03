
import re
from pathlib import Path
from typing import Any

import torch

from src.model.transformer import TransformerLM
from src.data.tokenizer import Tokenizer
from src.inference.retriever import DocumentRetriever


class RAGPipeline:
    """Offline RAG pipeline for IntelliDesk."""

    STOP_WORDS = {
        "what", "is", "are", "was", "were", "who",
        "when", "where", "why", "how", "the", "a",
        "an", "in", "on", "of", "to", "and", "does",
        "do", "did", "for", "with", "which", "that",
        "tell", "me", "about", "explain", "define",
        "can", "could", "would", "should", "by",
        "using", "according", "based", "example"
    }

    NO_ANSWER = (
        "I couldn't find a relevant answer in the provided documents."
    )

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

        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )

        model = TransformerLM(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        retriever = DocumentRetriever()

        if data_path:
            source_path = Path(data_path)

            if not source_path.exists():
                raise FileNotFoundError(
                    f"Document source not found: {source_path}"
                )

            if (
                source_path.is_dir()
                or source_path.suffix.lower() in {".pdf", ".txt"}
            ):
                retriever = DocumentRetriever.from_paths(
                    [source_path]
                )
            else:
                raise ValueError(
                    f"Unsupported retrieval source: {source_path}"
                )

        return cls(model, tokenizer, retriever)

    def retrieve_context(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[tuple[str, float]]:

        return self.retriever.retrieve(query, top_k=top_k)

    def _get_query_words(self, query: str) -> set[str]:

        return {
            word.lower()
            for word in re.findall(r"[a-zA-Z0-9]+", query)
            if word.lower() not in self.STOP_WORDS
        }

    def _is_relevant_passage(
        self,
        query: str,
        passage: str,
        score: float,
    ) -> bool:

        if not passage.strip() or score <= 0:
            return False

        query_words = self._get_query_words(query)

        passage_words = {
            word.lower()
            for word in re.findall(r"[a-zA-Z0-9]+", passage)
        }

        return bool(query_words & passage_words)

    def _split_sentences(self, passage: str) -> list[str]:

        parts = re.split(
            r"(?<=[.!?])\s+|[•▪◊‡]\s*|\n+",
            passage
        )

        return [
            part.strip(" \t\r\n-–—")
            for part in parts
            if part.strip(" \t\r\n-–—")
        ]

    def _is_question_sentence(self, sentence: str) -> bool:
        """Reject questions and question-like fragments."""

        cleaned = sentence.strip()

        if "?" in cleaned:
            return True

        question_start = re.match(
            r"^(problem\s*[-:]\s*)?"
            r"(what|which|who|when|where|why|how|"
            r"does|do|did|is|are|was|were|can|could)\b",
            cleaned,
            re.IGNORECASE,
        )

        return question_start is not None

    def _sentence_score(
        self,
        query_words: set[str],
        sentence: str,
    ) -> float:

        if self._is_question_sentence(sentence):
            return 0.0

        sentence_words = {
            word.lower()
            for word in re.findall(r"[a-zA-Z0-9]+", sentence)
        }

        if not sentence_words:
            return 0.0

        matched_words = query_words & sentence_words

        if not matched_words:
            return 0.0

        coverage = len(matched_words) / len(query_words)

        length_penalty = min(len(sentence_words), 40) / 40

        return (
            len(matched_words)
            + coverage
            - (0.15 * length_penalty)
        )

    def extract_answer(
        self,
        query: str,
        passage: str,
    ) -> str:

        if not passage.strip():
            return self.NO_ANSWER

        query_words = self._get_query_words(query)

        if not query_words:
            return self.NO_ANSWER

        best_sentence = ""
        best_score = 0.0

        for sentence in self._split_sentences(passage):

            score = self._sentence_score(
                query_words,
                sentence,
            )

            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence if best_sentence else self.NO_ANSWER

    def _select_best_answer(
        self,
        query: str,
        relevant_sources: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any] | None]:

        query_words = self._get_query_words(query)

        best_answer = self.NO_ANSWER
        best_metadata = None
        best_score = 0.0

        for item in relevant_sources:

            passage = str(item.get("text", ""))
            metadata = item.get("metadata", {})

            for sentence in self._split_sentences(passage):

                score = self._sentence_score(
                    query_words,
                    sentence,
                )

                if score > best_score:
                    best_score = score
                    best_answer = sentence
                    best_metadata = metadata

        return best_answer, best_metadata

    def format_prompt(
        self,
        query: str,
        retrieved_passages: list[tuple[str, float]],
    ) -> str:

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

        max_tokens = int(self.model.max_seq_len)
        prefix = "Context: "
        suffix = f"\nQuery: {query}\nAnswer:"

        def token_count(text: str) -> int:
            return len(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )

        if token_count(suffix) >= max_tokens:
            return self.tokenizer.decode(
                self.tokenizer.encode(
                    suffix,
                    add_special_tokens=False,
                )[-max_tokens:]
            )

        selected = []

        for passage, _score in retrieved_passages:

            passage = passage.strip()

            if not passage:
                continue

            candidate = selected + [passage]
            prompt = f"{prefix}{' '.join(candidate)}{suffix}"

            if token_count(prompt) <= max_tokens:
                selected.append(passage)

        if not selected:
            return suffix

        prompt = f"{prefix}{' '.join(selected)}{suffix}"

        if token_count(prompt) > max_tokens:
            raise RuntimeError(
                "Prompt exceeds model context length."
            )

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

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        # Step 1: Retrieve passages.
        retrieved_sources = self.retriever.retrieve_with_metadata(
            query,
            top_k=top_k,
        )

        # Step 2: Filter relevant passages.
        relevant_sources = [
            item
            for item in retrieved_sources
            if self._is_relevant_passage(
                query,
                str(item["text"]),
                float(item["score"]),
            )
        ]

        passages = [
            (str(item["text"]), float(item["score"]))
            for item in relevant_sources
        ]

        # Step 3: Handle missing answers.
        if not relevant_sources:
            return {
                "query": query,
                "answer": self.NO_ANSWER,
                "answer_source": None,
                "retrieved_passages": [],
                "retrieved_sources": [],
                "prompt": f"Query: {query}\nAnswer:",
                "prompt_token_count": 0,
                "max_context_tokens": self.model.max_seq_len,
                "generated_text": (
                    "Not generated because no relevant context was found."
                ),
                "full_output": "",
            }

        # Step 4: Select the best answer from all passages.
        answer, answer_source = self._select_best_answer(
            query,
            relevant_sources,
        )

        # Step 5: Prepare the Transformer prompt.
        prompt = self._fit_prompt_to_context(
            query,
            passages,
        )

        encoded_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        if len(encoded_tokens) > self.model.max_seq_len:
            raise RuntimeError(
                "Prompt exceeds model context window."
            )

        device = next(self.model.parameters()).device

        input_ids = torch.tensor(
            [encoded_tokens],
            dtype=torch.long,
            device=device,
        )

        # Step 6: Experimental text generation.
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

        # Step 7: Return results.
        return {
            "query": query,
            "answer": answer,
            "answer_source": answer_source,
            "retrieved_passages": passages,
            "retrieved_sources": relevant_sources,
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
    )

    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model.pt",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/tokenizer.json",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/training.txt",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=60,
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

    print("\nExtractive Answer:")
    print(result["answer"])

    print("\nAnswer Source:")
    print(result["answer_source"])

    print("\nGenerated Answer (Experimental):")
    print(result["generated_text"])

    print("\nRetrieved Passages:")

    for index, (passage, score) in enumerate(
        result["retrieved_passages"],
        start=1,
    ):
        print(f"\n[{index}] Score: {score:.4f}")
        print(passage)
