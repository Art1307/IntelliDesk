"""Repeatable offline evaluation for IntelliDesk retrieval and generation.

Example (from the repository root):
    python -m scripts.evaluate_rag --skip-generation
    python -m scripts.evaluate_rag --output evaluation/results.json

The retrieval metrics evaluate whether expected passages are found. The generated
keyword coverage is only a shallow diagnostic; it is not a factuality metric.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from src.inference.rag_pipeline import RAGPipeline
from src.inference.retriever import DocumentRetriever


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array of evaluation examples and validate required fields."""
    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Expected a non-empty JSON list in {path}")

    for index, item in enumerate(questions):
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError(f"Evaluation item {index} must contain a string 'question'")
        if not item["question"].strip():
            raise ValueError(f"Evaluation item {index} has an empty question")
        if not isinstance(item.get("expected_passage_contains"), str):
            raise ValueError(
                f"Evaluation item {index} must contain 'expected_passage_contains'"
            )
        keywords = item.get("answer_keywords", [])
        if not isinstance(keywords, list) or not all(isinstance(x, str) for x in keywords):
            raise ValueError(f"Evaluation item {index}: 'answer_keywords' must be a list of strings")
    return questions


def evaluate_retrieval(
    questions: list[dict[str, Any]], retriever: DocumentRetriever, top_k: int
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Calculate hit@k and reciprocal rank for manually identified passages."""
    rows: list[dict[str, Any]] = []
    hits = 0
    reciprocal_ranks = []

    for item in questions:
        query = item["question"]
        expected = item["expected_passage_contains"].casefold()
        results = retriever.retrieve(query, top_k=top_k)
        matched_rank = next(
            (
                rank
                for rank, (passage, _score) in enumerate(results, start=1)
                if expected in passage.casefold()
            ),
            None,
        )
        hit = matched_rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / matched_rank if matched_rank else 0.0)
        rows.append(
            {
                "id": item.get("id", f"question_{len(rows) + 1}"),
                "question": query,
                "expected_passage_contains": item["expected_passage_contains"],
                "retrieval_hit": hit,
                "matched_rank": matched_rank,
                "retrieved_passages": [
                    {"text": passage, "score": round(float(score), 6)}
                    for passage, score in results
                ],
            }
        )

    count = len(questions)
    metrics = {
        f"hit_at_{top_k}": hits / count,
        "mrr": sum(reciprocal_ranks) / count,
        "question_count": float(count),
    }
    return rows, metrics


def add_generation_results(
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    pipeline: RAGPipeline,
    max_new_tokens: int,
) -> dict[str, float]:
    """Run generation and report keyword coverage as a diagnostic only."""
    coverage_values = []
    for row, item in zip(rows, questions):
        result = pipeline.answer_query(
            item["question"], top_k=len(row["retrieved_passages"]),
            max_new_tokens=max_new_tokens, temperature=0.7, top_k_sampling=5,
        )
        answer = result["generated_text"]
        expected_keywords = item.get("answer_keywords", [])
        lowered_answer = answer.casefold()
        matched = [word for word in expected_keywords if word.casefold() in lowered_answer]
        coverage = len(matched) / len(expected_keywords) if expected_keywords else 0.0
        coverage_values.append(coverage)
        row["generated_answer"] = answer
        row["answer_keywords"] = expected_keywords
        row["matched_answer_keywords"] = matched
        row["answer_keyword_coverage"] = round(coverage, 4)
        row["prompt"] = result["prompt"]

    return {
        "mean_answer_keyword_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.json"))
    parser.add_argument("--data", type=Path, default=Path("data/training.txt"))
    parser.add_argument("--model", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-generation", action="store_true", help="Evaluate retrieval only")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not args.data.is_file():
        parser.error(f"Corpus file not found: {args.data}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    questions = load_questions(args.questions)
    retriever = DocumentRetriever.from_paths([args.data])
    rows, metrics = evaluate_retrieval(questions, retriever, args.top_k)

    if not args.skip_generation:
        if not args.model.is_file() or not args.tokenizer.is_file():
            parser.error("Checkpoint/tokenizer missing. Use --skip-generation for retrieval-only evaluation.")
        pipeline = RAGPipeline.from_checkpoints(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            data_path=args.data,
            device="cpu",
        )
        metrics.update(add_generation_results(rows, questions, pipeline, args.max_new_tokens))

    report = {
        "evaluation_version": 1,
        "seed": args.seed,
        "questions_file": str(args.questions),
        "corpus_file": str(args.data),
        "retrieval_metrics": {
            key: value for key, value in metrics.items()
            if key.startswith("hit_at_") or key in {"mrr", "question_count"}
        },
        "generation_metrics": {
            key: value for key, value in metrics.items()
            if key.startswith("mean_answer_")
        },
        "interpretation_note": (
            "Answer keyword coverage is a shallow diagnostic, not a factuality or semantic-quality metric. "
            "The supplied questions are a starter smoke-test set, not a held-out benchmark."
        ),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("IntelliDesk evaluation")
    print(f"Questions: {len(questions)}")
    for key, value in report["retrieval_metrics"].items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    for key, value in report["generation_metrics"].items():
        print(f"{key}: {value:.4f}")
    print(f"Detailed report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
