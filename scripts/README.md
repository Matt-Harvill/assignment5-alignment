# Model Download Scripts

This directory contains scripts for downloading Hugging Face models.

## Setup

1. **Get your Hugging Face token:**
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with read permissions
   - Copy the token

2. **Configure your environment:**
   - Edit the `.env` file in the project root
   - Replace `your_huggingface_token_here` with your actual token:
   ```
   HF_TOKEN=hf_your_actual_token_here
   ```

## Usage

### Using the Python script directly:

```bash
# Download a model to default cache location
python scripts/download_models.py meta-llama/Llama-2-7b-hf

# Download to a custom cache directory
python scripts/download_models.py meta-llama/Llama-2-7b-hf --cache_dir /path/to/cache

# Use a token directly (overrides .env file)
python scripts/download_models.py meta-llama/Llama-2-7b-hf --token hf_your_token
```

### Using the shell script wrapper:

```bash
# Download a model
./scripts/download_model.sh meta-llama/Llama-2-7b-hf

# Download to a custom cache directory
./scripts/download_model.sh meta-llama/Llama-2-7b-hf /path/to/cache
```

## Examples

```bash
# Download Llama 2 7B
./scripts/download_model.sh meta-llama/Llama-2-7b-hf

# Download Mistral 7B
./scripts/download_model.sh mistralai/Mistral-7B-v0.1

# Download a smaller model for testing
./scripts/download_model.sh microsoft/DialoGPT-medium
```

## Notes

- Models are downloaded to `~/.cache/huggingface/hub` by default
- The script automatically handles authentication using your HF_TOKEN
- Large models may take significant time and disk space to download
- Make sure you have sufficient disk space for the models you want to download
