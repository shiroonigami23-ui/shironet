import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Upload local model/checkpoint artifacts to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help="Hugging Face repo id, e.g. username/shironet-checkpoints (optional if HF_USERNAME is set)",
    )
    parser.add_argument(
        "--repo-name",
        default="shironet-edge",
        help="Repo name used with HF_USERNAME when --repo-id is omitted",
    )
    parser.add_argument(
        "--local-dir",
        default="models",
        help="Local directory containing artifacts (default: models)",
    )
    parser.add_argument(
        "--path-in-repo",
        default="checkpoints",
        help="Target path within HF repo (default: checkpoints)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private if it does not exist",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload ShiroNet model artifacts",
        help="Commit message for upload",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export HF_TOKEN before running this script.")
    username = os.getenv("HF_USERNAME", "ShiroOnigami23")
    repo_id = args.repo_id or f"{username}/{args.repo_name}"

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise SystemExit(f"Local directory does not exist: {local_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        path_in_repo=args.path_in_repo,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print(f"Uploaded artifacts from {local_dir} to https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
