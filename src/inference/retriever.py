
import math
import re
from pathlib import Path

from src.data.document_loader import load_documents


class DocumentRetriever:
    """Offline TF-IDF document retriever with source metadata."""

    def __init__(self, documents: list[str] | None = None):
        self.documents: list[str] = []
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []
        self.doc_norms: list[float] = []
        self.source_metadata: list[dict[str, object]] = []

        if documents:
            self.index_documents(documents)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())

    def index_documents(
        self,
        documents: list[str],
        metadata: list[dict[str, object]] | None = None,
    ) -> None:
        """Create TF-IDF vectors for the supplied documents."""

        indexed_pairs = [
            (
                doc.strip(),
                metadata[index]
                if metadata is not None and index < len(metadata)
                else {},
            )
            for index, doc in enumerate(documents)
            if doc.strip()
        ]

        self.documents = [doc for doc, _ in indexed_pairs]
        self.source_metadata = [source for _, source in indexed_pairs]

        num_docs = len(self.documents)

        if num_docs == 0:
            self.vocabulary = {}
            self.idf = {}
            self.doc_vectors = []
            self.doc_norms = []
            return

        doc_tokens = [
            self._tokenize(doc)
            for doc in self.documents
        ]

        doc_freqs: dict[str, int] = {}
        vocab_set = set()

        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            vocab_set.update(unique_tokens)

            for token in unique_tokens:
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        self.vocabulary = {
            term: index
            for index, term in enumerate(sorted(vocab_set))
        }

        self.idf = {
            term: math.log(
                (1 + num_docs) / (1 + doc_freqs[term])
            ) + 1.0
            for term in self.vocabulary
        }

        self.doc_vectors = []
        self.doc_norms = []

        for tokens in doc_tokens:
            if not tokens:
                self.doc_vectors.append({})
                self.doc_norms.append(0.0)
                continue

            counts: dict[str, int] = {}

            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

            total_tokens = len(tokens)
            vector: dict[str, float] = {}
            squared_sum = 0.0

            for token, count in counts.items():
                weight = (
                    count / total_tokens
                ) * self.idf[token]

                vector[token] = weight
                squared_sum += weight * weight

            self.doc_vectors.append(vector)
            self.doc_norms.append(math.sqrt(squared_sum))

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
    ) -> "DocumentRetriever":
        """Build a retriever from a text file."""

        path = Path(filepath)

        with open(path, "r", encoding="utf-8") as file:
            lines = [
                (line.strip(), number)
                for number, line in enumerate(file, start=1)
                if line.strip()
            ]

        instance = cls()

        instance.index_documents(
            [line for line, _ in lines],
            metadata=[
                {
                    "source": str(path.resolve()),
                    "line_number": number,
                }
                for _, number in lines
            ],
        )

        return instance

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path],
        chunk_size: int = 700,
        overlap: int = 100,
    ) -> "DocumentRetriever":
        """Build a retriever from PDF and TXT files or directories."""

        chunks = load_documents(
            paths,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        instance = cls()

        instance.index_documents(
            [chunk.text for chunk in chunks],
            metadata=[
                {
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "line_number": chunk.line_number,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

        return instance

    def _rank_indices(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[tuple[int, float]]:
        """Return document indices and their cosine similarity scores."""

        if top_k <= 0 or not self.documents:
            return []

        tokens = self._tokenize(query)

        if not tokens:
            return []

        counts: dict[str, int] = {}

        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        total_tokens = len(tokens)
        query_vector: dict[str, float] = {}
        squared_sum = 0.0

        for token, count in counts.items():
            if token in self.idf:
                weight = (
                    count / total_tokens
                ) * self.idf[token]

                query_vector[token] = weight
                squared_sum += weight * weight

        query_norm = math.sqrt(squared_sum)

        # If the query has no matching words,
        # return the first documents with zero scores.
        if query_norm == 0:
            return [
                (index, 0.0)
                for index in range(
                    min(top_k, len(self.documents))
                )
            ]

        scores: list[tuple[int, float]] = []

        for index, (vector, norm) in enumerate(
            zip(self.doc_vectors, self.doc_norms)
        ):
            if norm == 0:
                scores.append((index, 0.0))
                continue

            dot_product = sum(
                weight * vector.get(token, 0.0)
                for token, weight in query_vector.items()
            )

            score = dot_product / (query_norm * norm)
            scores.append((index, score))

        # Highest score first; original order breaks ties.
        scores.sort(
            key=lambda item: (-item[1], item[0])
        )

        return scores[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[tuple[str, float]]:
        """Retrieve the most relevant document passages."""

        ranked_indices = self._rank_indices(
            query,
            top_k,
        )

        return [
            (self.documents[index], score)
            for index, score in ranked_indices
        ]

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[dict[str, object]]:
        """Retrieve passages together with their original metadata."""

        ranked_indices = self._rank_indices(
            query,
            top_k,
        )

        results = []

        for index, score in ranked_indices:
            results.append(
                {
                    "text": self.documents[index],
                    "score": score,
                    "metadata": self.source_metadata[index],
                }
            )

        return results
