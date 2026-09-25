from pathlib import Path
from src.inference.retriever import DocumentRetriever


def test_retriever_ranking():
    docs = [
        "Convolutional neural networks are commonly used for computer vision.",
        "The Transformer architecture relies on self-attention mechanisms.",
        "Recurrent neural networks process sequential data with hidden states.",
        "Retrieval-Augmented Generation combines indexing with language models.",
    ]

    retriever = DocumentRetriever(documents=docs)

    # Query clearly targeting document 1
    results = retriever.retrieve("What is self-attention in Transformer?", top_k=2)
    assert len(results) == 2

    top_doc, top_score = results[0]
    assert "Transformer architecture relies on self-attention" in top_doc
    assert top_score > 0.0

    # Query targeting document 3
    rag_results = retriever.retrieve("Tell me about Retrieval-Augmented Generation", top_k=1)
    assert "Retrieval-Augmented Generation" in rag_results[0][0]


def test_cosine_similarity_bounds():
    docs = [
        "Machine learning models recognize patterns in high dimensional data.",
        "Natural language processing helps computers understand human speech.",
    ]
    retriever = DocumentRetriever(documents=docs)

    results = retriever.retrieve("high dimensional patterns", top_k=2)
    for doc, score in results:
        # Cosine similarity on non-negative TF-IDF vectors must be in [0.0, 1.0]
        assert 0.0 <= score <= 1.0001


def test_retriever_from_training_file():
    training_file = Path("data/training.txt")
    assert training_file.exists()

    retriever = DocumentRetriever.from_file(training_file)
    assert len(retriever.documents) == 10

    # Query for causal masking
    results = retriever.retrieve("causal masking decoder", top_k=1)
    assert len(results) == 1
    assert "causal masking" in results[0][0]

    # Query for offline document assistant
    results_offline = retriever.retrieve("offline data privacy local", top_k=1)
    assert "Offline document assistants" in results_offline[0][0]


def test_unseen_query_tokens():
    docs = ["Python is a high-level programming language."]
    retriever = DocumentRetriever(documents=docs)

    # Query with words not in the index
    results = retriever.retrieve("quantum astrophysics cosmos", top_k=1)
    assert len(results) == 1
    assert results[0][1] == 0.0  # Zero similarity


if __name__ == "__main__":
    test_retriever_ranking()
    test_cosine_similarity_bounds()
    test_retriever_from_training_file()
    test_unseen_query_tokens()
    print("All DocumentRetriever tests passed successfully!")
