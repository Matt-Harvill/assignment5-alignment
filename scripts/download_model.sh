#!/bin/bash
# Shell script wrapper for downloading Hugging Face models

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [cache_dir]"
    echo "Example: $0 meta-llama/Llama-2-7b-hf"
    echo "Example: $0 meta-llama/Llama-2-7b-hf /path/to/cache"
    exit 1
fi

MODEL_NAME="$1"
CACHE_DIR="${2:-}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the Python script
if [ -n "$CACHE_DIR" ]; then
    python scripts/download_models.py "$MODEL_NAME" --cache_dir "$CACHE_DIR"
else
    python scripts/download_models.py "$MODEL_NAME"
fi 