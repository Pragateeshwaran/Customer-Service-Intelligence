import huggingface_hub
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set timeout for Hugging Face downloads
os.environ["HF_HUB_REQUEST_TIMEOUT"] = "60"

try:
    huggingface_hub.hf_hub_download(repo_id="speechbrain/emotion-diarization-wavlm-large", filename="model.safetensors")
    huggingface_hub.hf_hub_download(repo_id="speechbrain/emotion-diarization-wavlm-large", filename="wav2vec2.ckpt")
    logging.info("Model files downloaded successfully.")
except Exception as e:
    logging.error(f"Error downloading model files: {e}")