from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import create_repo, upload_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a trained checkpoint or adapter to the Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="Example: QuangVoAI/multimodal-empathy-gemma26b-task1")
    parser.add_argument("--folder_path", type=str, required=True, help="Folder to upload, e.g. outputs/sft/run_01/final")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit_message", type=str, default="Upload trained checkpoint")
    parser.add_argument("--repo_type", type=str, default="model", choices=["model", "dataset", "space"])
    args = parser.parse_args()

    folder = Path(args.folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(folder),
        commit_message=args.commit_message,
    )
    print(f"Uploaded {folder} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
