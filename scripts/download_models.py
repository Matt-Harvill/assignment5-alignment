#!/usr/bin/env python3
"""
Script to download Hugging Face models using HF_TOKEN environment variable.
Usage: python scripts/download_models.py <model_name> [--cache_dir <path>]
"""

import os
import sys
import argparse
from typing import Any
from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv


def load_environment() -> str:
    """Load environment variables from .env file."""
    load_dotenv()

    # Check if HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found in environment variables.")
        print("Please set HF_TOKEN in your .env file or environment.")
        sys.exit(1)

    return hf_token


def download_model(model_name: str, cache_dir: str | None = None, token: str | None = None) -> Any:
    """Download a Hugging Face model."""
    try:
        print(f"Downloading model: {model_name}")

        # Login with token
        if token:
            login(token=token)

        # Download the model
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=token,
            local_dir_use_symlinks=False,  # Use actual files instead of symlinks
        )

        print(f"✅ Successfully downloaded {model_name}")
        print(f"📁 Model saved to: {local_dir}")
        return local_dir

    except Exception as e:
        print(f"❌ Error downloading {model_name}: {str(e)}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Hugging Face models")
    parser.add_argument("model_name", help="Hugging Face model name (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--cache_dir", help="Directory to cache models (default: ~/.cache/huggingface)")
    parser.add_argument("--token", help="Hugging Face token (overrides HF_TOKEN env var)")

    args = parser.parse_args()

    # Load environment and get token
    if args.token:
        token = args.token
    else:
        token = load_environment()

    # Download the model
    download_model(args.model_name, args.cache_dir, token)


if __name__ == "__main__":
    main()
