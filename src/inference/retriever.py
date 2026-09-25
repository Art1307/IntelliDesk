import math
import re
from pathlib import Path


class DocumentRetriever:
    """
    TF-IDF Document Retriever with Cosine Similarity for Offline RAG.

    Academic / Viva Reference:
    - What it does: Indexes unstructured document passages into sparse TF-IDF vectors
      and retrieves the most semantically relevant context chunks for an input query.
    - Why it is needed:
      1. Non-Parametric Memory: Language models have fixed parametric weights; RAG dynamically
         supplies relevant up-to-date document knowledge without retraining.
      2. Grounding & Anti-Hallucination: Conditioning generation on retrieved factual passages
         enforces factual accuracy.
      3. Fully Offline & Fast: Uses pure vector-space mathematics (TF-IDF + Cosine Similarity)
         requiring zero external vector databases, cloud APIs, or GPUs.
    - Mathematical Formulation:
      - Term Frequency: TF(t, d) = count(t, d) / len(d)
      - Inverse Document Frequency: IDF(t, D) = ln((1 + |D|) / (1 + DF(t))) + 1
      - TF-IDF Weight: w(t, d) = TF(t, d) * IDF(t)
      - Cosine Similarity: sim(q, d) = (q . d) / (||q|| * ||d||)
    """

    def __init__(self, documents: list[str] | None = None):
        self.documents: list[str] = []
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []
        self.doc_norms: list[float] = []

        if documents:
            self.index_documents(documents)

    def _tokenize(self, text: str) -> list[str]:
        """Convert text into normalized lowercase alphanumeric tokens."""
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())

    def index_documents(self, documents: list[str]) -> None:
        """
        Build TF-IDF inverted index from document passages.

        Args:
            documents: List of text passages or document chunks.
        """
        self.documents = [doc.strip() for doc in documents if doc.strip()]
        num_docs = len(self.documents)
        if num_docs == 0:
            return

        # 1. Tokenize all documents and track document frequencies (DF)
        doc_tokens = [self._tokenize(doc) for doc in self.documents]
        doc_freqs: dict[str, int] = {}
        vocab_set = set()

        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            vocab_set.update(unique_tokens)
            for token in unique_tokens:
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(vocab_set))}

        # 2. Compute smooth IDF for each term: ln((1 + N) / (1 + DF)) + 1
        self.idf = {
            term: math.log((1 + num_docs) / (1 + doc_freqs[term])) + 1.0
            for term in self.vocabulary
        }

        # 3. Compute TF-IDF vector and L2 norm for each document
        self.doc_vectors = []
        self.doc_norms = []

        for tokens in doc_tokens:
            if not tokens:
                self.doc_vectors.append({})
                self.doc_norms.append(0.0)
                continue

            # Term frequency
            tf_counts: dict[str, int] = {}
            for token in tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1

            total_tokens = len(tokens)
            vec: dict[str, float] = {}
            sq_sum = 0.0

            for token, count in tf_counts.items():
                if token in self.idf:
                    weight = (count / total_tokens) * self.idf[token]
                    vec[token] = weight
                    sq_sum += weight * weight

            norm = math.sqrt(sq_sum)
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm)

    @classmethod
    def from_file(cls, filepath: str | Path) -> "DocumentRetriever":
        """Load text file, split into non-empty line chunks, and index."""
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return cls(documents=lines)

    def retrieve(
        self, query: str, top_k: int = 2
    ) -> list[tuple[str, float]]:
        """
        Search document collection for passages with highest Cosine Similarity to query.

        Args:
            query: User search query string.
            top_k: Number of highest-ranking passages to return.

        Returns:
            List of tuples: (document_passage, cosine_similarity_score).
        """
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(self.documents[i], 0.0) for i in range(min(top_k, len(self.documents)))]

        # 1. Compute query TF-IDF vector and norm
        q_counts: dict[str, int] = {}
        for token in query_tokens:
            q_counts[token] = q_counts.get(token, 0) + 1

        total_q = len(query_tokens)
        q_vec: dict[str, float] = {}
        q_sq_sum = 0.0

        for token, count in q_counts.items():
            if token in self.idf:
                weight = (count / total_q) * self.idf[token]
                q_vec[token] = weight
                q_sq_sum += weight * weight

        q_norm = math.sqrt(q_sq_sum)
        if q_norm == 0.0:
            # Query contains no indexed terms
            return [(self.documents[i], 0.0) for i in range(min(top_k, len(self.documents)))]

        # 2. Compute Cosine Similarity with all documents
        scores: list[tuple[int, float]] = []

        for idx, (doc_vec, doc_norm) in enumerate(zip(self.doc_vectors, self.doc_norms)):
            if doc_norm == 0.0:
                scores.append((idx, 0.0))
                continue

            # Dot product between sparse vectors
            dot_product = sum(
                weight * doc_vec.get(token, 0.0)
                for token, weight in q_vec.items()
            )
            cosine_sim = dot_product / (q_norm * doc_norm)
            scores.append((idx, cosine_sim))

        # 3. Sort by similarity score descending
        scores.sort(key=lambda item: item[1], reverse=True)

        # 4. Return top-k documents
        results = [
            (self.documents[doc_idx], score)
            for doc_idx, score in scores[:top_k]
        ]
        return results
