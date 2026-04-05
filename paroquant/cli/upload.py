from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Local model directory to upload.")
    parser.add_argument("--repo-id", type=str, required=True, help="Target Hugging Face repo id, e.g. org/model-name.")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face token.")
    parser.add_argument("--revision", type=str, default="main", help="Target branch or revision to upload to.")
    parser.add_argument("--path-in-repo", type=str, default="", help="Optional subdirectory inside the repo.")
    parser.add_argument("--commit-message", type=str, default=None, help="Optional commit message.")
    parser.add_argument("--commit-description", type=str, default=None, help="Optional commit description.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    parser.add_argument("--create-pr", action="store_true", help="Upload via a pull request instead of directly.")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for upload. Install it directly or via the transformers extra."
        ) from exc

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path must be a directory: {model_path}")

    commit_message = args.commit_message or f"Upload {model_path.name}"

    api = HfApi(token=args.token)
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )
    print(f"Repo: {repo_url}")

    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=model_path,
        path_in_repo=args.path_in_repo or None,
        commit_message=commit_message,
        commit_description=args.commit_description,
        token=args.token,
        repo_type="model",
        revision=args.revision,
        create_pr=args.create_pr or None,
    )
    print(f"Commit: {commit_info.oid}")
    print(f"URL: {commit_info.commit_url}")
    if getattr(commit_info, "pr_url", None):
        print(f"PR: {commit_info.pr_url}")


if __name__ == "__main__":
    main()
