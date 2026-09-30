import ast
import argparse
import sys
from pathlib import Path


class PythonFileParser:
    """Extract structural metadata and generate human-readable summaries of Python files."""

    @staticmethod
    def parse(filepath: str | Path) -> dict:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            return {
                "filename": path.name,
                "filepath": str(path),
                "error": f"SyntaxError: {e}",
                "summary": f"Python module {path.name} contains syntax errors and could not be parsed.",
                "pairs": [],
            }

        # 1. Module Docstring
        module_doc = ast.get_docstring(tree) or ""

        # 2. Imports
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        # 3. Classes, methods, and docstrings
        classes = []
        code_summary_pairs = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_doc = ast.get_docstring(node) or ""
                methods = [
                    m.name
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                classes.append(
                    {"name": node.name, "docstring": cls_doc, "methods": methods}
                )
                if cls_doc:
                    clean_doc = " ".join(cls_doc.split())
                    code_summary_pairs.append(
                        f"Code: class {node.name} | Summary: {clean_doc}"
                    )

        # 4. Top-level Functions & Docstrings
        functions = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_doc = ast.get_docstring(node) or ""
                args = [arg.arg for arg in node.args.args if arg.arg != "self"]
                functions.append(
                    {"name": node.name, "docstring": fn_doc, "args": args}
                )
                if fn_doc:
                    clean_doc = " ".join(fn_doc.split())
                    arg_str = ", ".join(args[:4])
                    code_summary_pairs.append(
                        f"Code: def {node.name}({arg_str}) | Summary: {clean_doc}"
                    )

        # 5. Build clean, natural-language module summary
        summary_sentences = [
            f"The Python module {path.name} provides functionality for the system."
        ]

        if module_doc:
            clean_doc = " ".join(module_doc.split())
            summary_sentences.append(f"Description: {clean_doc}")

        if classes:
            class_desc = []
            for c in classes:
                m_list = (
                    f" with methods {', '.join(c['methods'][:4])}"
                    if c["methods"]
                    else ""
                )
                class_desc.append(f"{c['name']}{m_list}")
            summary_sentences.append(f"It defines classes: {'; '.join(class_desc)}.")

        if functions:
            fn_names = [f["name"] for f in functions[:8]]
            summary_sentences.append(
                f"It contains functions: {', '.join(fn_names)}."
            )

        if imports:
            clean_imports = list(dict.fromkeys(imports))[:5]
            summary_sentences.append(f"It imports: {', '.join(clean_imports)}.")

        full_summary = " ".join(summary_sentences)

        return {
            "filename": path.name,
            "filepath": str(path),
            "docstring": module_doc,
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "summary": full_summary,
            "pairs": code_summary_pairs,
        }


class DocumentManager:
    """Manages document ingestion, summarization, and training set synchronization."""

    def __init__(self, training_path: str = "data/training.txt"):
        self.training_path = Path(training_path)

    def summarize_python_file(self, filepath: str | Path) -> dict:
        """Parse and summarize a Python file."""
        return PythonFileParser.parse(filepath)

    def add_to_training_set(self, text_or_lines: str | list[str]) -> int:
        """
        Append summary text or lines to data/training.txt.
        Returns the number of new unique lines added.
        """
        self.training_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(text_or_lines, str):
            lines = [text_or_lines]
        else:
            lines = text_or_lines

        existing = set()
        if self.training_path.exists():
            with open(self.training_path, "r", encoding="utf-8") as f:
                existing = {line.strip() for line in f if line.strip()}

        added_count = 0
        with open(self.training_path, "a", encoding="utf-8") as f:
            for line in lines:
                cleaned = " ".join(line.split()).strip()
                if cleaned and cleaned not in existing:
                    f.write(f"{cleaned}\n")
                    existing.add(cleaned)
                    added_count += 1

        return added_count

    def harvest_directory(self, dir_path: str | Path) -> int:
        """
        Scan a local directory for .py files, extract docstrings and code-summary pairs,
        and append them to the training dataset without needing internet/HuggingFace.
        """
        dir_p = Path(dir_path)
        if not dir_p.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        py_files = list(dir_p.rglob("*.py"))
        total_added = 0

        print(f"Scanning {len(py_files)} Python files in {dir_path}...")
        for pf in py_files:
            # Skip virtual environments and hidden caches
            if any(part.startswith(".") or part in ["__pycache__", "venv", ".venv", "build", "dist"] for part in pf.parts):
                continue
            try:
                result = self.summarize_python_file(pf)
                lines_to_add = [result["summary"]] + result.get("pairs", [])
                added = self.add_to_training_set(lines_to_add)
                total_added += added
            except Exception:
                continue

        return total_added


def open_file_dialog() -> Path | None:
    """Prompt user using native Windows file dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Intellidesk - Select Python File to Upload & Summarize",
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")],
        )
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass
    return None


def interactive_upload_flow(training_path: str = "data/training.txt"):
    """Guided interactive CLI for file upload, summarization, and training set integration."""
    doc_mgr = DocumentManager(training_path=training_path)

    print("\n" + "=" * 55)
    print("   Intellidesk Offline Python File Assistant")
    print("=" * 55)
    print("How would you like to provide the Python file?")
    print("  [1] Open Windows file picker dialog (browse folder)")
    print("  [2] Enter or drag-and-drop file path here")
    print("  [3] Harvest dataset from a local folder of .py files (Offline)")
    print("  [q] Quit")

    choice = input("\nSelect an option [1/2/3/q] (default 1): ").strip().lower()

    target_file = None

    if choice in ["", "1"]:
        print("Opening file selection dialog...")
        target_file = open_file_dialog()
        if not target_file:
            print("No file selected via dialog.")
            fallback = input("Please type or drag-and-drop the file path: ").strip().strip('"').strip("'")
            if fallback:
                target_file = Path(fallback)
    elif choice == "2":
        path_input = input("Enter or drag-and-drop .py file path: ").strip().strip('"').strip("'")
        if path_input:
            target_file = Path(path_input)
    elif choice == "3":
        folder_input = input("Enter folder path to harvest (e.g. 'src' or '.'): ").strip().strip('"').strip("'") or "."
        added = doc_mgr.harvest_directory(folder_input)
        print(f"\n[Done] Extracted and added {added} new training samples to {training_path}!")
        retrain = input("\nWould you like to retrain the Transformer model now? (y/N): ").strip().lower()
        if retrain == "y":
            from src.training.train import run_training
            run_training(data_path=training_path)
            print("Model training complete!")
        return
    elif choice in ["q", "quit", "exit"]:
        print("Exiting.")
        return
    else:
        # User might have directly entered a file path
        cleaned = choice.strip('"').strip("'")
        if Path(cleaned).exists():
            target_file = Path(cleaned)
        else:
            print("Invalid selection.")
            return

    if not target_file or not target_file.exists():
        print(f"\n[Error] File could not be found: {target_file}")
        return

    print(f"\nAnalyzing and summarizing {target_file.name}...")
    result = doc_mgr.summarize_python_file(target_file)

    print("\n" + "-" * 50)
    print(f"File Name : {result['filename']}")
    if result["classes"]:
        print(f"Classes   : {', '.join(c['name'] for c in result['classes'])}")
    if result["functions"]:
        print(f"Functions : {', '.join(f['name'] for f in result['functions'])}")
    if result["imports"]:
        print(f"Imports   : {', '.join(result['imports'][:6])}")
    print("\nExtracted Summary Passage:")
    print(f"> {result['summary']}")
    print("-" * 50)

    # Ask user to add to training set
    add_confirm = input(f"\nAdd this summary to training set ({training_path})? [Y/n]: ").strip().lower()
    if add_confirm in ["", "y", "yes"]:
        lines = [result["summary"]] + result.get("pairs", [])
        num_added = doc_mgr.add_to_training_set(lines)
        if num_added > 0:
            print(f"[Success] Added {num_added} knowledge lines to {training_path}.")
        else:
            print("[Info] This summary already exists in the training set.")

        # Ask to retrain
        retrain_confirm = input("Retrain Transformer model now with the new data? [y/N]: ").strip().lower()
        if retrain_confirm in ["y", "yes"]:
            print("\nTriggering training on updated dataset...")
            from src.training.train import run_training
            run_training(data_path=training_path)
            print("Model retrained successfully with updated weights and vocabulary!")
    else:
        print("Skipped adding to training set.")


def main():
    parser = argparse.ArgumentParser(
        description="Intellidesk Document Manager - Upload & Summarize Python files"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Direct path to Python file (optional)"
    )
    parser.add_argument(
        "--add", action="store_true", help="Automatically add summary to training set"
    )
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain model after adding"
    )
    parser.add_argument(
        "--harvest", type=str, default=None, help="Harvest an entire directory of .py files"
    )
    parser.add_argument(
        "--training-path", type=str, default="data/training.txt", help="Path to training set"
    )
    args = parser.parse_args()

    # If no arguments provided, launch interactive prompt/file picker
    if len(sys.argv) == 1:
        interactive_upload_flow(training_path=args.training_path)
        return

    doc_mgr = DocumentManager(training_path=args.training_path)

    if args.harvest:
        added = doc_mgr.harvest_directory(args.harvest)
        print(f"Added {added} code-summary pairs from {args.harvest} to {args.training_path}")
        if args.retrain:
            from src.training.train import run_training
            run_training(data_path=args.training_path)
        return

    if args.file:
        result = doc_mgr.summarize_python_file(args.file)
        print("\nSummary:")
        print(result["summary"])
        if args.add:
            lines = [result["summary"]] + result.get("pairs", [])
            doc_mgr.add_to_training_set(lines)
            if args.retrain:
                from src.training.train import run_training
                run_training(data_path=args.training_path)


if __name__ == "__main__":
    main()
