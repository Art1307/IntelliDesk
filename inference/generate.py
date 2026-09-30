import argparse
from pathlib import Path
import torch
from src.data.tokenizer import Tokenizer
from src.model.transformer import TransformerLM
from src.inference.retriever import DocumentRetriever


def load_model_and_tokenizer(
    checkpoint_path: str = "checkpoints/model.pt",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    device: str = "cpu",
) -> tuple[TransformerLM, Tokenizer]:
    """Load pretrained model weights and character tokenizer."""
    tok_file = Path(tokenizer_path)
    ckpt_file = Path(checkpoint_path)

    if not tok_file.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found at {tokenizer_path}. Run training first."
        )
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Run training first."
        )

    tokenizer = Tokenizer.load(tok_file)
    checkpoint = torch.load(ckpt_file, map_location=device)

    # Support either 'model_config' or 'config' key in checkpoint
    config = checkpoint.get("model_config") or checkpoint.get("config")
    if not config:
        raise KeyError("Checkpoint missing 'model_config' or 'config' dictionary.")

    model = TransformerLM(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 5,
    device: str = "cpu",
) -> str:
    """Generate completion for a given text prompt."""
    if not prompt:
        prompt = " "

    token_ids = tokenizer.encode(prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    generated_ids = out[0].tolist()
    return tokenizer.decode(generated_ids)


def run_interactive():
    parser = argparse.ArgumentParser(
        description="Intellidesk Offline AI Document Assistant - Inference & Generation"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Input prompt to complete"
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable TF-IDF RAG retrieval from data corpus before generation",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-K sampling constraint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training.txt",
        help="Path to corpus for RAG retrieval",
    )
    args = parser.parse_args()

    print("\n--- Intellidesk Offline AI Assistant ---")
    print("Loading model and tokenizer...")
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by running: python -m src.training.train")
        return

    retriever = None
    if Path(args.data).exists():
        retriever = DocumentRetriever.from_file(args.data)

    def process_prompt(prompt_text: str):
        if args.rag and retriever:
            retrieved = retriever.retrieve(prompt_text, top_k=1)
            if retrieved and retrieved[0][1] > 0.0:
                context, score = retrieved[0]
                print(f"\n[RAG Context (Similarity: {score:.3f})]: {context}")
                # Augment prompt with retrieved context
                prompt_text = f"{context} {prompt_text}"

        completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"\n[Output]:\n{completion}\n")

    if args.prompt:
        process_prompt(args.prompt)
    else:
        print("\nInteractive mode (type 'exit' or 'quit' to stop):")
        while True:
            try:
                user_input = input("\nEnter prompt > ").strip()
                if not user_input or user_input.lower() in ["exit", "quit"]:
                    break
                process_prompt(user_input)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting.")


if __name__ == "__main__":
    run_interactive()
